#!/usr/bin/env python3
"""Time synchronisation of the Jetson bridge against the ESP32-S3 controller.

Answers the question "when was this packet sent?" in the Jetson's clock.
Two building blocks, see section 5 in JETSON_BRIDGE.md:

  * ``EspClock``   - holds the clock offset and the rate drift between the
                     ESP clock (esp_timer, us since boot) and the Jetson's
                     CLOCK_MONOTONIC. Converts ESP timestamps to Jetson time.
  * ``FrameParser`` - splits the RX stream into packets and understands both
                     frame types, 0xA5 (without) and 0xA6 (with send timestamp).

And on top of that ``TimeSync``, which runs the ping-pong 0xB0/0xB1.

The module only needs pyserial for real port operation; parser, maths
and self-test run without hardware:

    python3 timesync_jetson.py --selftest
    python3 timesync_jetson.py --port /dev/ttyTHS1
"""

from __future__ import annotations

import struct
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

# --- Frames ---------------------------------------------------------------

START_BYTE = 0xA5        # frame without timestamp
START_BYTE_TS = 0xA6     # frame with uint32 send timestamp after the CMD

# --- Commands the Jetson sends ---
CMD_TIME_SYNC = 0xB0
CMD_STAMP_MODE = 0xB2

# --- Commands the ESP sends, with their payload length ---
# The protocol has no length field, so this table must be right; a wrong value
# swallows the next packet as well.
RX_PAYLOAD_LEN: Dict[int, int] = {
    0x42: 11,   # CAL_RSP
    0x70: 1,    # BUTTON
    0x82: 12,   # PID_RSP
    0x84: 1,    # PID_SAVED
    0x93: 6,    # MOVE_DONE
    0x94: 11,   # PROGRESS_RSP
    0xA1: 6,    # BATTERY_RSP
    0xA2: 6,    # BATTERY_WARN
    0xB1: 17,   # TIME_RSP  : seq + int64 t_rx + int64 t_tx
    0xB3: 1,    # STAMP_RSP
    0xC1: 12,   # TELEMETRY : int32 pos + int32 speed + int16 duty + int16 mA
}

BITS_PER_BYTE = 10       # 8N1: start + 8 data + stop
U32 = 1 << 32


def monotonic() -> float:
    """Time base of the Jetson.

    Monotonic on purpose: time.time() would shift the offset if NTP jumps
    in the middle of a measurement. On Linux this is CLOCK_MONOTONIC.

    The link to the wall clock is only made at the very end, with a distance
    measured once:  wall_clock = mono_time + (time.time() - monotonic())
    """
    return time.monotonic()


def unwrap_u32(stamp32: int, last_full: int) -> int:
    """Extend the 32-bit frame stamp to the full ESP clock.

    ``last_full`` is the last known full value (from 0xB1). The 32-bit
    counter wraps every 71.6 minutes; as long as two known points are less
    than 35.8 minutes apart, the mapping is unambiguous.
    """
    coarse = (last_full & ~(U32 - 1)) | stamp32
    for cand in (coarse - U32, coarse, coarse + U32):
        if abs(cand - last_full) < U32 // 2:
            return cand
    return coarse


@dataclass
class Frame:
    """A received packet."""

    cmd: int
    payload: bytes
    #: Raw 32-bit send stamp of the ESP, only in 0xA6 frames.
    esp_tx_raw: Optional[int] = None
    #: Jetson clock (monotonic, seconds) when the last byte was read.
    rx_mono: float = 0.0

    @property
    def stamped(self) -> bool:
        return self.esp_tx_raw is not None


class FrameParser:
    """Splits the RX stream into packets.

    Tolerates three things that are normal on this line: ASCII status lines
    from the ESP, bytes after a loss of sync, and unknown CMDs. All of them
    end up in ``stray`` or ``unknown`` and only lead to a resync.
    """

    def __init__(self) -> None:
        self._buf = bytearray()
        self.stray = 0
        self.unknown = 0
        self.text: List[bytes] = []      # collected ASCII lines

    def feed(self, data: bytes, now: Optional[float] = None) -> List[Frame]:
        """Put new bytes in, get finished packets out.

        ``now`` is the Jetson clock when this block was read; every packet
        completed in it gets this time. Without per-byte timestamps from the
        driver it cannot be more exact - the error is the length of one read
        cycle and goes away with a blocking ``read(1)``.
        """
        if now is None:
            now = monotonic()
        self._buf += data
        out: List[Frame] = []

        while True:
            frame = self._try_one(now)
            if frame is None:
                return out
            out.append(frame)

    def _try_one(self, now: float) -> Optional[Frame]:
        buf = self._buf

        while True:
            # 1. Sync to a start byte, keep what came before as text.
            start = 0
            while start < len(buf) and buf[start] not in (START_BYTE, START_BYTE_TS):
                start += 1
            if start:
                self.stray += start
                self._collect_text(bytes(buf[:start]))
                del buf[:start]
            if len(buf) < 2:
                return None

            stamped = buf[0] == START_BYTE_TS
            head = 2 + (4 if stamped else 0)      # start + CMD (+ stamp)
            cmd = buf[1]
            length = RX_PAYLOAD_LEN.get(cmd)
            if length is None:
                # Unknown CMD: drop only the start byte, nothing more - the
                # next real packet could start right behind it.
                self.unknown += 1
                del buf[:1]
                continue

            if len(buf) < head + length:
                return None

            esp_tx = int.from_bytes(buf[2:6], "big") if stamped else None
            payload = bytes(buf[head:head + length])
            del buf[:head + length]
            return Frame(cmd=cmd, payload=payload, esp_tx_raw=esp_tx, rx_mono=now)

    def _collect_text(self, chunk: bytes) -> None:
        printable = bytes(b for b in chunk if 32 <= b < 127 or b in (10, 13))
        for line in printable.replace(b"\r", b"\n").split(b"\n"):
            if line.strip():
                self.text.append(line.strip())


@dataclass
class SyncSample:
    """One sync round."""

    esp_s: float      # ESP clock in the middle of the round, seconds
    offset: float     # ESP clock minus Jetson clock, seconds
    rtt: float        # round trip without the ESP's processing time


class EspClock:
    """Clock offset and rate drift between ESP and Jetson.

    The offset from a single round is only as good as the symmetry of the
    outbound and return path. The ESP reads its UART from ``loop()``, so the
    receive time varies by milliseconds. But every such delay also makes the
    round trip longer - so the round with the smallest ``rtt`` counts, and
    rounds with a clearly longer round trip are dropped from the drift fit.
    """

    #: How much longer than the fastest round a measurement may take and
    #: still go into the line fit. **Additive**, not a factor: a factor
    #: breaks as soon as the best round trip is negative - then the bound
    #: would be smaller than the best value and nothing would get through.
    RTT_MARGIN = 0.002

    def __init__(self, window: int = 32) -> None:
        self.window = window
        self.samples: List[SyncSample] = []
        self.last_full_us: int = 0     # last known full ESP counter
        self._a: Optional[float] = None   # offset at _ref
        self._b: float = 0.0              # rate drift (s/s)
        self._ref: float = 0.0            # reference point in ESP seconds
        self.boot_count = 0               # detected ESP reboots
        self.rejected = 0                 # rejected rounds in the window

    # --- Feeding in measurements ------------------------------------------

    def add_round(self, t1: float, t2_us: int, t3_us: int, t4: float) -> SyncSample:
        """Process one round. All four times refer to the last byte of the
        respective frame: t1/t4 in Jetson seconds, t2/t3 in ESP us."""
        if t2_us < self.last_full_us - 1_000_000:
            # The ESP clock runs from boot. If it jumps back, there was a
            # reset - everything learned so far is wrong then.
            self.reset(keep_boot_count=True)
            self.boot_count += 1

        self.last_full_us = max(self.last_full_us, t3_us)

        t2 = t2_us / 1e6
        t3 = t3_us / 1e6
        offset = ((t2 - t1) + (t3 - t4)) / 2.0
        rtt = (t4 - t1) - (t3 - t2)

        sample = SyncSample(esp_s=(t2 + t3) / 2.0, offset=offset, rtt=rtt)
        self.samples.append(sample)
        del self.samples[:-self.window]
        self._fit()
        return sample

    def reset(self, keep_boot_count: bool = False) -> None:
        self.samples.clear()
        self.last_full_us = 0
        self._a = None
        self._b = 0.0
        self._ref = 0.0
        if not keep_boot_count:
            self.boot_count = 0

    # --- Evaluation -------------------------------------------------------

    def _fit(self) -> None:
        good = self._good_samples()
        self.rejected = len(self.samples) - len(good)

        if not good:
            # Better no timestamp at all than a wrong one: valid becomes False,
            # the bridge falls back to the read time and reports it.
            self._a = None
            self._b = 0.0
            return

        self._ref = good[-1].esp_s

        if len(good) < 4:
            # Too few points for a line: best single value, no drift.
            self._a = min(good, key=lambda s: s.rtt).offset
            self._b = 0.0
            return

        # Least squares over offset(esp_s). Taking the ESP time as the
        # independent variable makes to_jetson() directly computable, without
        # having to know the Jetson time first.
        n = len(good)
        xs = [s.esp_s - self._ref for s in good]
        ys = [s.offset for s in good]
        mx = sum(xs) / n
        my = sum(ys) / n
        sxx = sum((x - mx) ** 2 for x in xs)
        if sxx <= 0:
            self._a = my
            self._b = 0.0
            return
        sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
        self._b = sxy / sxx
        self._a = my + self._b * (0.0 - mx)

    def _good_samples(self) -> List[SyncSample]:
        """The usable sync rounds.

        A **negative round trip is physically impossible** - it means that
        one of the four times was measured wrong (typically: t1 came back
        from the driver too late). Such rounds are dropped instead of going
        into the fit; ``min()`` used to pick exactly them as the "best",
        because they carried the smallest number.
        """
        usable = [s for s in self.samples if s.rtt >= 0.0]
        if not usable:
            return []
        best = min(s.rtt for s in usable)
        return [s for s in usable if s.rtt <= best + self.RTT_MARGIN]

    @property
    def valid(self) -> bool:
        return self._a is not None

    @property
    def offset(self) -> float:
        """Current offset in seconds (ESP clock minus Jetson clock)."""
        if self._a is None:
            raise RuntimeError("not synchronised yet")
        return self._a

    @property
    def drift_ppm(self) -> float:
        """Rate drift of the ESP relative to the Jetson in ppm."""
        return self._b * 1e6

    @property
    def best_rtt(self) -> float:
        """Fastest usable round. NaN as long as none is usable."""
        return min((s.rtt for s in self.samples if s.rtt >= 0.0),
                   default=float("nan"))

    def offset_at(self, esp_us: int) -> float:
        if self._a is None:
            raise RuntimeError("not synchronised yet")
        return self._a + self._b * (esp_us / 1e6 - self._ref)

    # --- Conversion -------------------------------------------------------

    def to_jetson(self, esp_us: int) -> float:
        """Full ESP microseconds -> Jetson monotonic in seconds."""
        return esp_us / 1e6 - self.offset_at(esp_us)

    def stamp_to_jetson(self, stamp32: int) -> float:
        """32-bit frame stamp -> Jetson monotonic in seconds."""
        return self.to_jetson(unwrap_u32(stamp32, self.last_full_us))

    def frame_time(self, frame: Frame) -> Optional[float]:
        """Send time of a packet in Jetson time, or None for an
        unstamped frame."""
        if frame.esp_tx_raw is None or not self.valid:
            return None
        return self.stamp_to_jetson(frame.esp_tx_raw)

    def latency(self, frame: Frame) -> Optional[float]:
        """How long the packet took from "sent" to "read"."""
        sent = self.frame_time(frame)
        return None if sent is None else frame.rx_mono - sent


class TimeSync:
    """Runs the ping-pong 0xB0/0xB1 and maintains an ``EspClock``.

    Deliberately without its own thread and without a port: ``send`` writes a
    frame and waits until it is really out; the replies are passed in by the
    bridge's RX path via ``handle``. That way it fits into a threaded bridge
    as well as an asyncio one.
    """

    def __init__(self, send: Callable[[bytes], float], clock: Optional[EspClock] = None):
        #: send(frame) -> Jetson time at which the last byte was out.
        self._send = send
        self.clock = clock or EspClock()
        self._seq = 0
        self._pending: Dict[int, float] = {}    # seq -> t1
        self.lost = 0

    def request(self) -> int:
        """Start one round. Returns the seq used."""
        self._seq = (self._seq + 1) & 0xFF
        seq = self._seq
        if len(self._pending) > 8:
            # Replies are not coming - do not drag old entries along forever.
            self.lost += len(self._pending)
            self._pending.clear()
        self._pending[seq] = self._send(bytes([START_BYTE, CMD_TIME_SYNC, seq]))
        return seq

    def handle(self, frame: Frame) -> Optional[SyncSample]:
        """Offer a received packet. Returns the sync round if it was a
        matching 0xB1 reply, otherwise None."""
        if frame.cmd != 0xB1:
            return None
        seq = frame.payload[0]
        t1 = self._pending.pop(seq, None)
        if t1 is None:
            return None     # straggler after a timeout
        t2_us, t3_us = struct.unpack(">qq", frame.payload[1:17])
        return self.clock.add_round(t1, t2_us, t3_us, frame.rx_mono)

    @staticmethod
    def stamp_mode_frame(on: bool) -> bytes:
        """Frame that switches stamping of the ESP packets on or off."""
        return bytes([START_BYTE, CMD_STAMP_MODE, 1 if on else 0])


# ==========================================================================
# Operation on the real port
# ==========================================================================

def _serial_sender(ser) -> Callable[[bytes], float]:
    """Writes a frame in one go and returns the time at which the last byte
    left the line.

    This time is **computed, not measured**: take the clock before the frame
    goes into the driver and add the transmission time (10 bits per byte) -
    the same calculation the ESP does for its side.

    The obvious way, putting a ``flush()`` after ``write()`` and then reading
    the clock, looks cleaner but is not: ``flush()`` ends up in
    ``tcdrain()``, and depending on the driver (reliably on the Jetson's
    Tegra UART) that returns clearly later than the last byte leaves the
    line. Then t1 is too late and the round trip comes out negative.

    The ``flush()`` before stays: it makes sure nothing old is left in the
    buffer and our frame really goes out immediately.
    """

    def send(frame: bytes) -> float:
        ser.flush()
        started = monotonic()
        ser.write(frame)
        return started + len(frame) * BITS_PER_BYTE / ser.baudrate

    return send


def read_available(ser) -> bytes:
    """Read everything that is there - and do it immediately, as soon as the
    first byte arrives.

    ``ser.read(n)`` with n > 1 waits until n bytes are together or the
    timeout runs out. With a packet stream that means: several packets land
    in one block and all get the same receive time - that of the last one.
    For time sync that is deadly, the error becomes as large as the timeout.

    ``read(1)`` only blocks until the first byte; the rest follows without
    waiting.
    """
    data = ser.read(1)
    if data and ser.in_waiting:
        data += ser.read(ser.in_waiting)
    return data


def run_live(port: str, baud: int = 115200, rounds: int = 12,
             interval: float = 10.0, duration: float = 60.0) -> None:
    """Synchronise, switch stamps on, print incoming packets with send time
    and latency."""
    import serial   # only imported here so the self-test works without it

    with serial.Serial(port, baud, timeout=0.05) as ser:
        parser = FrameParser()
        sync = TimeSync(_serial_sender(ser))

        def pump(seconds: float) -> List[Frame]:
            frames: List[Frame] = []
            end = monotonic() + seconds
            while monotonic() < end:
                data = read_available(ser)
                if data:
                    frames.extend(parser.feed(data))
            return frames

        print(f"-> {rounds} sync rounds ...")
        for _ in range(rounds):
            sync.request()
            for frame in pump(0.02):
                sync.handle(frame)

        clock = sync.clock
        if not clock.valid:
            raise SystemExit("no reply to 0xB0 - check wiring/baud rate")
        print(f"   offset {clock.offset * 1e3:+.3f} ms | "
              f"best round trip {clock.best_rtt * 1e3:.3f} ms | "
              f"drift {clock.drift_ppm:+.1f} ppm")

        ser.write(TimeSync.stamp_mode_frame(True))
        ser.flush()
        print("-> send timestamps switched on, listening ...")

        next_sync = monotonic() + interval
        end = monotonic() + duration
        while monotonic() < end:
            for frame in pump(0.05):
                if sync.handle(frame) is not None:
                    continue
                lat = clock.latency(frame)
                when = clock.frame_time(frame)
                if when is None:
                    print(f"   cmd=0x{frame.cmd:02X} (no stamp)")
                else:
                    print(f"   cmd=0x{frame.cmd:02X} sent at t={when:.6f} "
                          f"| latency {lat * 1e3:.2f} ms")
            for line in parser.text:
                print(f"   [esp] {line.decode('ascii', 'replace')}")
            parser.text.clear()
            if monotonic() >= next_sync:
                sync.request()
                next_sync = monotonic() + interval


# ==========================================================================
# Self-test - simulated ESP, no hardware needed
# ==========================================================================

@dataclass
class _FakeEsp:
    """Simulates the ESP side including clock offset, drift and
    varying processing time."""

    offset: float = 123.456789     # ESP clock minus Jetson clock, seconds
    drift_ppm: float = 40.0
    baud: int = 115200
    jitter: List[float] = field(default_factory=list)   # delay per round
    _round: int = 0

    def esp_us(self, jetson_s: float) -> int:
        return int(round((jetson_s + self.offset * (1 + self.drift_ppm / 1e6)) * 1e6))

    def _wire(self, nbytes: int) -> float:
        return nbytes * BITS_PER_BYTE / self.baud

    def answer(self, seq: int, t1: float) -> tuple:
        """Reply to a request that was completely out at time t1.
        Returns (frame, t4) - t4 is the Jetson time at the last byte."""
        delay = self.jitter[self._round % len(self.jitter)] if self.jitter else 0.0
        self._round += 1

        # t2: arrival at the ESP. The line has practically no delay, but the
        # ESP reads from loop() - that is the jitter.
        t2_jetson = t1 + delay
        t2_us = self.esp_us(t2_jetson)

        # The ESP replies immediately; t3 includes the transmission time.
        frame = bytes([START_BYTE, 0xB1, seq]) + struct.pack(">qq", t2_us, 0)
        t3_jetson = t2_jetson + self._wire(len(frame))
        t3_us = self.esp_us(t3_jetson)
        frame = bytes([START_BYTE, 0xB1, seq]) + struct.pack(">qq", t2_us, t3_us)
        return frame, t3_jetson

    def stamped(self, cmd: int, payload: bytes, jetson_s: float) -> bytes:
        stamp = self.esp_us(jetson_s) & (U32 - 1)
        return bytes([START_BYTE_TS, cmd]) + stamp.to_bytes(4, "big") + payload


def _selftest() -> int:
    failures = 0

    def check(name: str, ok: bool, detail: str = "") -> None:
        nonlocal failures
        print(f"  [{'ok ' if ok else 'FAIL'}] {name}{'  ' + detail if detail else ''}")
        if not ok:
            failures += 1

    print("Parser")
    # --- unstamped, stamped, ASCII in between, unknown CMD ---
    p = FrameParser()
    stream = (b"System Ready. blah\r\n"
              + bytes([START_BYTE, 0x70, 0x01])
              + bytes([START_BYTE_TS, 0x93, 0x00, 0x01, 0x02, 0x03,
                       0x07, 0x00, 0x00, 0x00, 0x03, 0x84])
              + bytes([START_BYTE, 0x55])          # unknown CMD
              + bytes([START_BYTE, 0x84, 0x00]))
    frames = p.feed(stream, now=1.0)
    check("packet count", len(frames) == 3, f"{len(frames)}")
    check("BUTTON unstamped", frames[0].cmd == 0x70 and not frames[0].stamped)
    check("MOVE_DONE stamped",
          frames[1].cmd == 0x93 and frames[1].esp_tx_raw == 0x00010203)
    check("MOVE_DONE payload", frames[1].payload == bytes([0x07, 0, 0, 0, 3, 0x84]))
    check("unknown CMD dropped", p.unknown == 1)
    check("ASCII collected", p.text and p.text[0].startswith(b"System Ready"))

    # --- byte-by-byte delivery must give the same result ---
    p2 = FrameParser()
    got: List[Frame] = []
    for i in range(len(stream)):
        got += p2.feed(stream[i:i + 1], now=1.0)
    check("byte by byte identical",
          [(f.cmd, f.payload, f.esp_tx_raw) for f in got]
          == [(f.cmd, f.payload, f.esp_tx_raw) for f in frames])

    print("Unwrapping")
    check("no wrap", unwrap_u32(1000, 900) == 1000)
    check("wrap forward",
          unwrap_u32(10, U32 - 10) == U32 + 10,
          f"{unwrap_u32(10, U32 - 10)}")
    check("wrap backward",
          unwrap_u32(U32 - 10, U32 + 10) == U32 - 10)

    print("Clock sync")
    esp = _FakeEsp(offset=123.456789, drift_ppm=40.0,
                   jitter=[0.0004, 0.0031, 0.0009, 0.0002, 0.0055, 0.0012])
    clock = EspClock()
    t_now = [1000.0]

    def fake_send(frame: bytes) -> float:
        # time for shifting out the request
        t_now[0] += len(frame) * BITS_PER_BYTE / 115200
        return t_now[0]

    sync = TimeSync(fake_send, clock)
    for _ in range(16):
        seq = sync.request()
        answer, t4 = esp.answer(seq, t_now[0])
        t_now[0] = t4
        frames = FrameParser().feed(answer, now=t4)
        check_sample = sync.handle(frames[0])
        assert check_sample is not None
        t_now[0] += 0.02        # 20 ms pause until the next round

    actual = clock.offset
    target = esp.offset * (1 + esp.drift_ppm / 1e6)
    check("offset hit", abs(actual - target) < 1e-3,
          f"error {(actual - target) * 1e6:+.1f} us")
    check("best round trip plausible", 0 <= clock.best_rtt < 0.002,
          f"{clock.best_rtt * 1e6:.0f} us")

    print("Send time of a packet")
    sent_at = t_now[0] + 0.5                     # Jetson time of sending
    frame = esp.stamped(0x70, b"\x01", sent_at)
    parsed = FrameParser().feed(frame, now=sent_at + 0.0012)[0]
    recovered = clock.frame_time(parsed)
    check("send time reconstructed", abs(recovered - sent_at) < 1e-3,
          f"error {(recovered - sent_at) * 1e6:+.1f} us")
    check("latency plausible", abs(clock.latency(parsed) - 0.0012) < 1e-3,
          f"{clock.latency(parsed) * 1e3:.2f} ms")

    print("Unusable rounds")
    # A negative round trip means: one of the four time measurements was wrong.
    # min() used to pick exactly this round as the "best", and the factor
    # filter (best * 2.0) with a negative best rejected everything and then
    # fell back to *all* samples - the garbage ended up in the fit.
    dirty = EspClock()
    for i in range(6):
        t1 = 100.0 + i
        dirty.add_round(t1, int((t1 + 0.010) * 1e6), int((t1 + 0.012) * 1e6),
                        t1 + 0.001)          # t4 before the ESP processing
    check("negative rounds detected", dirty.rejected == 6, f"{dirty.rejected}")
    check("no stamp from garbage", not dirty.valid)
    check("best_rtt is NaN", dirty.best_rtt != dirty.best_rtt)

    # A good round in between must win against the broken ones.
    good_offset = 50.0
    for i in range(6):
        t1 = 200.0 + i
        t2 = t1 + good_offset + 0.0004
        t3 = t2 + 0.0018
        dirty.add_round(t1, int(t2 * 1e6), int(t3 * 1e6), t1 + 0.0025)
    check("good rounds win", dirty.valid)
    check("offset hit despite garbage",
          dirty.valid and abs(dirty.offset - good_offset) < 1e-3,
          f"{(dirty.offset - good_offset) * 1e6:+.0f} us" if dirty.valid else "-")
    check("round trip positive", dirty.best_rtt >= 0, f"{dirty.best_rtt * 1e3:.3f} ms")

    print("Reading without block wait")

    class _FakePort:
        """Serial port that delivers bytes in small chunks."""

        def __init__(self, chunks):
            self.chunks = list(chunks)
            self.baudrate = 115200
            self.reads = 0

        @property
        def in_waiting(self):
            # What is still pending from the started block. read(1) has
            # already taken the first byte.
            return len(self.chunks[0]) if self.chunks else 0

        def read(self, n):
            self.reads += 1
            if not self.chunks:
                return b""
            head = self.chunks[0]
            take, rest = head[:n], head[n:]
            if rest:
                self.chunks[0] = rest
            else:
                self.chunks.pop(0)
            return take

    button = bytes([0xA5, 0x70, 0x01])       # BUTTON
    saved = bytes([0xA5, 0x84, 0x00])        # PID_SAVED
    port = _FakePort([button, saved])
    first = read_available(port)
    check("first block complete at once", first == button, first.hex(" "))
    second = read_available(port)
    check("second block separate", second == saved, second.hex(" "))
    check("empty port does not block", read_available(port) == b"")

    print("ESP reset")
    before = clock.boot_count
    clock.add_round(t_now[0], 5_000, 5_200, t_now[0] + 0.001)
    check("reboot detected", clock.boot_count == before + 1)

    print()
    print("Self-test FAILED" if failures else "Self-test passed")
    return 1 if failures else 0


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selftest", action="store_true",
                    help="check maths and parser without hardware")
    ap.add_argument("--port", help="serial port to the ESP, e.g. /dev/ttyTHS1")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--listen-time", type=float, default=60.0,
                    help="seconds to listen (default 60)")
    args = ap.parse_args()

    if args.selftest:
        return _selftest()
    if args.port:
        run_live(args.port, args.baud, duration=args.listen_time)
        return 0
    ap.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
