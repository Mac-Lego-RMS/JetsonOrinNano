#!/usr/bin/env python3
"""Zeitsynchronisation der Jetson-Bridge gegen den ESP32-S3-Controller.

Beantwortet die Frage "wann wurde dieses Paket losgeschickt?" in der Uhr des
Jetson. Zwei Bausteine, siehe Abschnitt 5 in JETSON_BRIDGE.md:

  * ``EspClock``   - haelt den Uhrenversatz und die Gangabweichung zwischen
                     ESP-Uhr (esp_timer, us seit Boot) und CLOCK_MONOTONIC des
                     Jetson. Rechnet ESP-Zeitstempel in Jetson-Zeit um.
  * ``FrameParser`` - zerlegt den RX-Strom in Pakete und versteht dabei beide
                     Rahmenformen, 0xA5 (ohne) und 0xA6 (mit Sendezeitstempel).

Und darueber ``TimeSync``, das den Ping-Pong 0xB0/0xB1 fuehrt.

Das Modul haengt nur fuer den echten Portbetrieb an pyserial; Parser, Rechnung
und Selbsttest laufen ohne Hardware:

    python3 timesync_jetson.py --selftest
    python3 timesync_jetson.py --port /dev/ttyTHS1
"""

from __future__ import annotations

import struct
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

# --- Rahmen ---------------------------------------------------------------

START_BYTE = 0xA5        # Rahmen ohne Zeitstempel
START_BYTE_TS = 0xA6     # Rahmen mit uint32-Sendezeitstempel hinter dem CMD

# --- Befehle, die der Jetson sendet ---
CMD_TIME_SYNC = 0xB0
CMD_STAMP_MODE = 0xB2

# --- Befehle, die der ESP sendet, mit ihrer Nutzlastlaenge ---
# Ohne Laengenfeld im Protokoll muss diese Tabelle stimmen; ein falscher Wert
# frisst das naechste Paket mit auf.
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
    0xC1: 12,   # TELEMETRY : int32 pos + int32 tempo + int16 duty + int16 mA
}

BITS_PER_BYTE = 10       # 8N1: Start + 8 Daten + Stopp
U32 = 1 << 32


def monotonic() -> float:
    """Zeitbasis des Jetson.

    Bewusst monoton: time.time() wuerde bei einem NTP-Sprung mitten in der
    Messung den Versatz verschieben. Unter Linux ist das CLOCK_MONOTONIC.

    Der Bezug zur Wanduhr wird erst ganz am Ende hergestellt, mit einem einmal
    gemessenen Abstand:  wanduhr = mono_zeit + (time.time() - monotonic())
    """
    return time.monotonic()


def unwrap_u32(stamp32: int, last_full: int) -> int:
    """Den 32-Bit-Rahmenstempel auf die volle ESP-Uhr hochziehen.

    ``last_full`` ist der letzte bekannte volle Wert (aus 0xB1). Der 32-Bit-
    Zaehler laeuft alle 71,6 Minuten ueber; solange zwischen zwei bekannten
    Punkten weniger als 35,8 Minuten liegen, ist die Zuordnung eindeutig.
    """
    coarse = (last_full & ~(U32 - 1)) | stamp32
    for cand in (coarse - U32, coarse, coarse + U32):
        if abs(cand - last_full) < U32 // 2:
            return cand
    return coarse


@dataclass
class Frame:
    """Ein empfangenes Paket."""

    cmd: int
    payload: bytes
    #: Roher 32-Bit-Sendestempel des ESP, nur bei 0xA6-Rahmen.
    esp_tx_raw: Optional[int] = None
    #: Jetson-Uhr (monoton, Sekunden), als das letzte Byte gelesen wurde.
    rx_mono: float = 0.0

    @property
    def stamped(self) -> bool:
        return self.esp_tx_raw is not None


class FrameParser:
    """Zerlegt den RX-Strom in Pakete.

    Toleriert dreierlei, was auf dieser Leitung normal ist: ASCII-Statuszeilen
    des ESP, Bytes nach einem Sync-Verlust und unbekannte CMDs. Alles davon
    landet in ``stray`` bzw. ``unknown`` und fuehrt nur zum Resync.
    """

    def __init__(self) -> None:
        self._buf = bytearray()
        self.stray = 0
        self.unknown = 0
        self.text: List[bytes] = []      # aufgesammelte ASCII-Zeilen

    def feed(self, data: bytes, now: Optional[float] = None) -> List[Frame]:
        """Neue Bytes einwerfen, fertige Pakete herausbekommen.

        ``now`` ist die Jetson-Uhr beim Lesen dieses Blocks; alle darin
        vollstaendig gewordenen Pakete bekommen diesen Zeitpunkt. Genauer geht
        es ohne Byte-Zeitstempel des Treibers nicht - der Fehler ist die Dauer
        eines Lesezyklus und faellt bei blockierendem ``read(1)`` weg.
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
            # 1. Auf ein Startbyte aufsynchronisieren, Vorlauf als Text merken.
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
            head = 2 + (4 if stamped else 0)      # Start + CMD (+ Stempel)
            cmd = buf[1]
            length = RX_PAYLOAD_LEN.get(cmd)
            if length is None:
                # Unbekanntes CMD: nur das Startbyte verwerfen, nicht mehr - das
                # naechste echte Paket koennte direkt dahinter anfangen.
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
    """Eine Messrunde."""

    esp_s: float      # ESP-Uhr in der Mitte der Runde, Sekunden
    offset: float     # ESP-Uhr minus Jetson-Uhr, Sekunden
    rtt: float        # Umlaufzeit ohne die Bearbeitungszeit des ESP


class EspClock:
    """Uhrenversatz und Gangabweichung zwischen ESP und Jetson.

    Der Versatz aus einer einzelnen Runde ist nur so gut wie die Symmetrie von
    Hin- und Rueckweg. Der ESP liest seinen UART aus ``loop()``, also schwankt
    die Empfangszeit um Millisekunden. Jede solche Verzoegerung verlaengert aber
    auch den Umlauf - deshalb zaehlt die Runde mit dem kleinsten ``rtt``, und
    Runden mit deutlich groesserem Umlauf fliegen aus der Driftschaetzung.
    """

    #: Wie viel laenger als die schnellste Runde eine Messung dauern darf,
    #: um noch in die Geradenschaetzung einzugehen. **Additiv**, nicht als
    #: Faktor: ein Faktor kippt, sobald der beste Umlauf negativ ist - dann
    #: waere die Schranke kleiner als der Bestwert und nichts kaeme durch.
    RTT_MARGIN = 0.002

    def __init__(self, window: int = 32) -> None:
        self.window = window
        self.samples: List[SyncSample] = []
        self.last_full_us: int = 0     # letzter bekannter voller ESP-Zaehler
        self._a: Optional[float] = None   # Versatz bei _ref
        self._b: float = 0.0              # Gangabweichung (s/s)
        self._ref: float = 0.0            # Bezugspunkt in ESP-Sekunden
        self.boot_count = 0               # erkannte ESP-Neustarts
        self.rejected = 0                 # verworfene Runden im Fenster

    # --- Messung einspeisen ---------------------------------------------

    def add_round(self, t1: float, t2_us: int, t3_us: int, t4: float) -> SyncSample:
        """Eine Runde verrechnen. Alle vier Zeitpunkte meinen das letzte Byte
        des jeweiligen Rahmens: t1/t4 in Jetson-Sekunden, t2/t3 in ESP-us."""
        if t2_us < self.last_full_us - 1_000_000:
            # Die ESP-Uhr laeuft ab Boot. Springt sie zurueck, war ein Reset -
            # alles Gelernte ist dann falsch.
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

    # --- Auswertung -------------------------------------------------------

    def _fit(self) -> None:
        good = self._good_samples()
        self.rejected = len(self.samples) - len(good)

        if not good:
            # Lieber gar kein Zeitstempel als ein falscher: valid wird False,
            # die Bridge faellt auf die Lesezeit zurueck und meldet es.
            self._a = None
            self._b = 0.0
            return

        self._ref = good[-1].esp_s

        if len(good) < 4:
            # Zu wenige Punkte fuer eine Gerade: bester Einzelwert, keine Drift.
            self._a = min(good, key=lambda s: s.rtt).offset
            self._b = 0.0
            return

        # Kleinste Quadrate ueber offset(esp_s). Die ESP-Zeit als unabhaengige
        # Groesse zu nehmen macht to_jetson() direkt auswertbar, ohne die
        # Jetson-Zeit vorher schon zu kennen.
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
        """Die brauchbaren Messrunden.

        Ein **negativer Umlauf ist physikalisch unmoeglich** - er heisst, dass
        einer der vier Zeitpunkte falsch gemessen wurde (typisch: t1 kam vom
        Treiber zu spaet zurueck). Solche Runden fliegen raus statt in den Fit
        einzugehen; frueher hat ``min()`` ausgerechnet sie als "beste"
        ausgesucht, weil sie die kleinste Zahl trugen.
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
        """Aktueller Versatz in Sekunden (ESP-Uhr minus Jetson-Uhr)."""
        if self._a is None:
            raise RuntimeError("noch nicht synchronisiert")
        return self._a

    @property
    def drift_ppm(self) -> float:
        """Gangabweichung des ESP gegenueber dem Jetson in ppm."""
        return self._b * 1e6

    @property
    def best_rtt(self) -> float:
        """Schnellste brauchbare Runde. NaN, solange keine taugt."""
        return min((s.rtt for s in self.samples if s.rtt >= 0.0),
                   default=float("nan"))

    def offset_at(self, esp_us: int) -> float:
        if self._a is None:
            raise RuntimeError("noch nicht synchronisiert")
        return self._a + self._b * (esp_us / 1e6 - self._ref)

    # --- Umrechnen --------------------------------------------------------

    def to_jetson(self, esp_us: int) -> float:
        """Volle ESP-Mikrosekunden -> Jetson-Monotonic in Sekunden."""
        return esp_us / 1e6 - self.offset_at(esp_us)

    def stamp_to_jetson(self, stamp32: int) -> float:
        """32-Bit-Rahmenstempel -> Jetson-Monotonic in Sekunden."""
        return self.to_jetson(unwrap_u32(stamp32, self.last_full_us))

    def frame_time(self, frame: Frame) -> Optional[float]:
        """Sendezeitpunkt eines Pakets in Jetson-Zeit, oder None bei einem
        ungestempelten Rahmen."""
        if frame.esp_tx_raw is None or not self.valid:
            return None
        return self.stamp_to_jetson(frame.esp_tx_raw)

    def latency(self, frame: Frame) -> Optional[float]:
        """Wie lange das Paket von "abgeschickt" bis "gelesen" gebraucht hat."""
        sent = self.frame_time(frame)
        return None if sent is None else frame.rx_mono - sent


class TimeSync:
    """Fuehrt den Ping-Pong 0xB0/0xB1 und pflegt eine ``EspClock``.

    Bewusst ohne eigenen Thread und ohne Port: ``send`` schreibt einen Rahmen
    und wartet, bis er wirklich draussen ist; die Antworten wirft der
    RX-Pfad der Bridge mit ``handle`` herein. So passt das sowohl in eine
    Thread- als auch in eine asyncio-Bridge.
    """

    def __init__(self, send: Callable[[bytes], float], clock: Optional[EspClock] = None):
        #: send(rahmen) -> Jetson-Zeit, zu der das letzte Byte draussen war.
        self._send = send
        self.clock = clock or EspClock()
        self._seq = 0
        self._pending: Dict[int, float] = {}    # seq -> t1
        self.lost = 0

    def request(self) -> int:
        """Eine Runde anstossen. Gibt die verwendete seq zurueck."""
        self._seq = (self._seq + 1) & 0xFF
        seq = self._seq
        if len(self._pending) > 8:
            # Antworten bleiben aus - alte Eintraege nicht ewig mitschleppen.
            self.lost += len(self._pending)
            self._pending.clear()
        self._pending[seq] = self._send(bytes([START_BYTE, CMD_TIME_SYNC, seq]))
        return seq

    def handle(self, frame: Frame) -> Optional[SyncSample]:
        """Ein empfangenes Paket anbieten. Liefert die Messrunde, wenn es eine
        passende 0xB1-Antwort war, sonst None."""
        if frame.cmd != 0xB1:
            return None
        seq = frame.payload[0]
        t1 = self._pending.pop(seq, None)
        if t1 is None:
            return None     # Nachzuegler nach einem Timeout
        t2_us, t3_us = struct.unpack(">qq", frame.payload[1:17])
        return self.clock.add_round(t1, t2_us, t3_us, frame.rx_mono)

    @staticmethod
    def stamp_mode_frame(on: bool) -> bytes:
        """Rahmen, der das Stempeln der ESP-Pakete ein- oder ausschaltet."""
        return bytes([START_BYTE, CMD_STAMP_MODE, 1 if on else 0])


# ==========================================================================
# Betrieb am echten Port
# ==========================================================================

def _serial_sender(ser) -> Callable[[bytes], float]:
    """Schreibt einen Rahmen in einem Rutsch und gibt den Zeitpunkt zurueck, zu
    dem das letzte Byte die Leitung verlassen hat.

    Dieser Zeitpunkt wird **gerechnet, nicht gemessen**: Uhr nehmen, bevor der
    Rahmen in den Treiber geht, und die Uebertragungsdauer (10 Bit je Byte)
    dazuzaehlen - dieselbe Rechnung, die der ESP fuer seine Seite macht.

    Der naheliegende Weg, nach ``write()`` ein ``flush()`` zu setzen und dann
    die Uhr zu lesen, sieht sauberer aus, ist es aber nicht: ``flush()``
    laeuft auf ``tcdrain()`` hinaus, und das kehrt je nach Treiber (auf dem
    Tegra-UART des Jetson zuverlaessig) deutlich spaeter zurueck als das
    letzte Byte die Leitung verlaesst. Dann wird t1 zu spaet und der Umlauf
    rechnerisch negativ.

    Das ``flush()`` vorher bleibt: es sorgt dafuer, dass nichts Altes mehr im
    Puffer steht und unser Rahmen wirklich sofort losgeht.
    """

    def send(frame: bytes) -> float:
        ser.flush()
        started = monotonic()
        ser.write(frame)
        return started + len(frame) * BITS_PER_BYTE / ser.baudrate

    return send


def read_available(ser) -> bytes:
    """Alles lesen, was da ist - und zwar sofort, sobald das erste Byte kommt.

    ``ser.read(n)`` mit n > 1 wartet, bis n Byte beisammen sind oder der
    Timeout ablaeuft. Bei einem Paketstrom heisst das: mehrere Pakete landen
    in einem Block und bekommen alle denselben Empfangszeitpunkt - den des
    letzten. Fuer den Zeitabgleich ist das toedlich, der Fehler wird so gross
    wie der Timeout.

    ``read(1)`` blockiert nur bis zum ersten Byte; der Rest kommt ohne Warten
    hinterher.
    """
    data = ser.read(1)
    if data and ser.in_waiting:
        data += ser.read(ser.in_waiting)
    return data


def run_live(port: str, baud: int = 115200, rounds: int = 12,
             interval: float = 10.0, duration: float = 60.0) -> None:
    """Synchronisieren, Stempel einschalten, eintreffende Pakete mit
    Sendezeitpunkt und Laufzeit ausgeben."""
    import serial   # nur hier importieren, damit der Selbsttest ohne auskommt

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

        print(f"-> {rounds} Messrunden ...")
        for _ in range(rounds):
            sync.request()
            for frame in pump(0.02):
                sync.handle(frame)

        clock = sync.clock
        if not clock.valid:
            raise SystemExit("keine Antwort auf 0xB0 - Verkabelung/Baudrate pruefen")
        print(f"   Versatz {clock.offset * 1e3:+.3f} ms | "
              f"bester Umlauf {clock.best_rtt * 1e3:.3f} ms | "
              f"Drift {clock.drift_ppm:+.1f} ppm")

        ser.write(TimeSync.stamp_mode_frame(True))
        ser.flush()
        print("-> Sendezeitstempel eingeschaltet, lausche ...")

        next_sync = monotonic() + interval
        end = monotonic() + duration
        while monotonic() < end:
            for frame in pump(0.05):
                if sync.handle(frame) is not None:
                    continue
                lat = clock.latency(frame)
                when = clock.frame_time(frame)
                if when is None:
                    print(f"   cmd=0x{frame.cmd:02X} (ohne Stempel)")
                else:
                    print(f"   cmd=0x{frame.cmd:02X} gesendet bei t={when:.6f} "
                          f"| Laufzeit {lat * 1e3:.2f} ms")
            for line in parser.text:
                print(f"   [esp] {line.decode('ascii', 'replace')}")
            parser.text.clear()
            if monotonic() >= next_sync:
                sync.request()
                next_sync = monotonic() + interval


# ==========================================================================
# Selbsttest - simulierter ESP, keine Hardware noetig
# ==========================================================================

@dataclass
class _FakeEsp:
    """Simuliert die ESP-Seite inklusive Uhrenversatz, Drift und
    schwankender Bearbeitungszeit."""

    offset: float = 123.456789     # ESP-Uhr minus Jetson-Uhr, Sekunden
    drift_ppm: float = 40.0
    baud: int = 115200
    jitter: List[float] = field(default_factory=list)   # Verzoegerung je Runde
    _round: int = 0

    def esp_us(self, jetson_s: float) -> int:
        return int(round((jetson_s + self.offset * (1 + self.drift_ppm / 1e6)) * 1e6))

    def _wire(self, nbytes: int) -> float:
        return nbytes * BITS_PER_BYTE / self.baud

    def answer(self, seq: int, t1: float) -> tuple:
        """Antwort auf eine Anfrage, die zum Zeitpunkt t1 komplett drausssen war.
        Liefert (rahmen, t4) - t4 ist die Jetson-Zeit beim letzten Byte."""
        delay = self.jitter[self._round % len(self.jitter)] if self.jitter else 0.0
        self._round += 1

        # t2: Ankunft beim ESP. Leitung ist quasi verzoegerungsfrei, aber der
        # ESP liest aus loop() - das ist der Jitter.
        t2_jetson = t1 + delay
        t2_us = self.esp_us(t2_jetson)

        # Der ESP antwortet sofort; t3 rechnet die Uebertragungsdauer ein.
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
    # --- ungestempelt, gestempelt, ASCII dazwischen, unbekanntes CMD ---
    p = FrameParser()
    stream = (b"System Ready. blah\r\n"
              + bytes([START_BYTE, 0x70, 0x01])
              + bytes([START_BYTE_TS, 0x93, 0x00, 0x01, 0x02, 0x03,
                       0x07, 0x00, 0x00, 0x00, 0x03, 0x84])
              + bytes([START_BYTE, 0x55])          # unbekanntes CMD
              + bytes([START_BYTE, 0x84, 0x00]))
    frames = p.feed(stream, now=1.0)
    check("Paketzahl", len(frames) == 3, f"{len(frames)}")
    check("BUTTON ungestempelt", frames[0].cmd == 0x70 and not frames[0].stamped)
    check("MOVE_DONE gestempelt",
          frames[1].cmd == 0x93 and frames[1].esp_tx_raw == 0x00010203)
    check("MOVE_DONE Nutzlast", frames[1].payload == bytes([0x07, 0, 0, 0, 3, 0x84]))
    check("unbekanntes CMD verworfen", p.unknown == 1)
    check("ASCII aufgesammelt", p.text and p.text[0].startswith(b"System Ready"))

    # --- byteweise Zustellung muss dasselbe ergeben ---
    p2 = FrameParser()
    got: List[Frame] = []
    for i in range(len(stream)):
        got += p2.feed(stream[i:i + 1], now=1.0)
    check("byteweise identisch",
          [(f.cmd, f.payload, f.esp_tx_raw) for f in got]
          == [(f.cmd, f.payload, f.esp_tx_raw) for f in frames])

    print("Unwrapping")
    check("kein Ueberlauf", unwrap_u32(1000, 900) == 1000)
    check("Ueberlauf vorwaerts",
          unwrap_u32(10, U32 - 10) == U32 + 10,
          f"{unwrap_u32(10, U32 - 10)}")
    check("Ueberlauf rueckwaerts",
          unwrap_u32(U32 - 10, U32 + 10) == U32 - 10)

    print("Uhrenabgleich")
    esp = _FakeEsp(offset=123.456789, drift_ppm=40.0,
                   jitter=[0.0004, 0.0031, 0.0009, 0.0002, 0.0055, 0.0012])
    clock = EspClock()
    t_now = [1000.0]

    def fake_send(frame: bytes) -> float:
        # Zeit fuers Rausschieben der Anfrage
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
        t_now[0] += 0.02        # 20 ms Pause bis zur naechsten Runde

    ist = clock.offset
    soll = esp.offset * (1 + esp.drift_ppm / 1e6)
    check("Versatz getroffen", abs(ist - soll) < 1e-3,
          f"Fehler {(ist - soll) * 1e6:+.1f} us")
    check("bester Umlauf plausibel", 0 <= clock.best_rtt < 0.002,
          f"{clock.best_rtt * 1e6:.0f} us")

    print("Sendezeitpunkt eines Pakets")
    sent_at = t_now[0] + 0.5                     # Jetson-Zeit des Absendens
    frame = esp.stamped(0x70, b"\x01", sent_at)
    parsed = FrameParser().feed(frame, now=sent_at + 0.0012)[0]
    rueck = clock.frame_time(parsed)
    check("Sendezeit rekonstruiert", abs(rueck - sent_at) < 1e-3,
          f"Fehler {(rueck - sent_at) * 1e6:+.1f} us")
    check("Laufzeit plausibel", abs(clock.latency(parsed) - 0.0012) < 1e-3,
          f"{clock.latency(parsed) * 1e3:.2f} ms")

    print("Unbrauchbare Runden")
    # Ein negativer Umlauf heisst: eine der vier Zeitmessungen war falsch.
    # Frueher hat min() genau diese Runde als "beste" ausgesucht und der
    # Faktor-Filter (best * 2.0) hat mit negativem best alles verworfen und
    # dann auf *alle* Proben zurueckgefallen - der Muell landete im Fit.
    dreck = EspClock()
    for i in range(6):
        t1 = 100.0 + i
        dreck.add_round(t1, int((t1 + 0.010) * 1e6), int((t1 + 0.012) * 1e6),
                        t1 + 0.001)          # t4 vor der ESP-Bearbeitung
    check("negative Runden erkannt", dreck.rejected == 6, f"{dreck.rejected}")
    check("kein Stempel aus Muell", not dreck.valid)
    check("best_rtt ist NaN", dreck.best_rtt != dreck.best_rtt)

    # Eine gute Runde dazwischen muss sich gegen die kaputten durchsetzen.
    gut_offset = 50.0
    for i in range(6):
        t1 = 200.0 + i
        t2 = t1 + gut_offset + 0.0004
        t3 = t2 + 0.0018
        dreck.add_round(t1, int(t2 * 1e6), int(t3 * 1e6), t1 + 0.0025)
    check("gute Runden setzen sich durch", dreck.valid)
    check("Versatz trotz Muell getroffen",
          dreck.valid and abs(dreck.offset - gut_offset) < 1e-3,
          f"{(dreck.offset - gut_offset) * 1e6:+.0f} us" if dreck.valid else "-")
    check("Umlauf positiv", dreck.best_rtt >= 0, f"{dreck.best_rtt * 1e3:.3f} ms")

    print("Lesen ohne Blockwartezeit")

    class _FakePort:
        """Serieller Port, der Bytes haeppchenweise liefert."""

        def __init__(self, chunks):
            self.chunks = list(chunks)
            self.baudrate = 115200
            self.reads = 0

        @property
        def in_waiting(self):
            # Was vom angefangenen Block noch aussteht. read(1) hat das erste
            # Byte schon abgezogen.
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
    check("erster Block sofort komplett", first == button, first.hex(" "))
    second = read_available(port)
    check("zweiter Block getrennt", second == saved, second.hex(" "))
    check("leerer Port blockiert nicht", read_available(port) == b"")

    print("ESP-Reset")
    before = clock.boot_count
    clock.add_round(t_now[0], 5_000, 5_200, t_now[0] + 0.001)
    check("Neustart erkannt", clock.boot_count == before + 1)

    print()
    print("Selbsttest fehlgeschlagen" if failures else "Selbsttest bestanden")
    return 1 if failures else 0


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selftest", action="store_true",
                    help="Rechnung und Parser ohne Hardware pruefen")
    ap.add_argument("--port", help="serieller Port zum ESP, z.B. /dev/ttyTHS1")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--dauer", type=float, default=60.0,
                    help="Sekunden lauschen (Default 60)")
    args = ap.parse_args()

    if args.selftest:
        return _selftest()
    if args.port:
        run_live(args.port, args.baud, duration=args.dauer)
        return 0
    ap.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
