#!/usr/bin/env python3
"""ROS 2 bridge to the ESP32-S3 controller.

Maps the UART protocol from docs/JETSON_BRIDGE.md completely onto ROS 2 -
every function of the ESP is reachable through a topic or a service - and
stamps every ESP message with the time it was *sent*, not the time the
Jetson happened to read it. The clock sync for this lives in
``timesync_jetson.py`` (section 5 of the spec).

Structure
---------
``EspLink``        Protocol, timesync, heartbeat, reader thread. Knows no ROS
                   and runs without rclpy - so it can be tested without the robot.
``EspBridgeNode``  rclpy node. Only wiring: topics and services onto the
                   methods of ``EspLink``, ESP packets onto publishers.

Start
-----
    ros2 run <package> esp_serial_bridge --ros-args -p port:=/dev/ttyTHS1
    python3 esp_serial_bridge.py --selftest      # without ROS, without hardware
"""

from __future__ import annotations

import argparse
import importlib.util
import math
import re
import struct
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Deque, Dict, List, Optional, Tuple
from nav_msgs.msg import Odometry

# ==========================================================================
# Pull in timesync_jetson.py
# ==========================================================================
# In the repo the file lives under docs/, in a ROS package it belongs next to
# this one. Cover both cases instead of letting the user fail on
# PYTHONPATH.

#: What is needed from timesync_jetson. Checked after every import
#: attempt - an empty placeholder or an unrelated file of the same name must
#: not slip through and only show up ten lines later as a meaningless
#: AttributeError.
_TIMESYNC_NAMES = ("Frame", "FrameParser", "EspClock", "TimeSync",
                   "monotonic", "START_BYTE")


def _timesync_missing(module) -> List[str]:
    return [name for name in _TIMESYNC_NAMES if not hasattr(module, name)]


def _load_timesync():
    tried: List[str] = []

    def accept(module, source: str):
        missing = _timesync_missing(module)
        if not missing:
            return module
        tried.append(f"{source}: incomplete, missing {', '.join(missing)}"
                     f" (file: {getattr(module, '__file__', '?')})")
        return None

    # 1. As part of the same package - the normal case in a ROS package, where
    #    both files are installed side by side.
    if __package__:
        name = f"{__package__}.timesync_jetson"
        try:
            found = accept(importlib.import_module(name), name)
            if found:
                return found
        except ImportError as exc:
            tried.append(f"{name}: {exc}")

    # 2. Anywhere on the search path.
    try:
        import timesync_jetson as module
        found = accept(module, "timesync_jetson (sys.path)")
        if found:
            return found
    except ImportError as exc:
        tried.append(f"timesync_jetson (sys.path): {exc}")

    # 3. Load directly as a file.
    here = Path(__file__).resolve().parent
    for path in (here / "timesync_jetson.py",
                 here.parent / "timesync_jetson.py",
                 here.parent / "docs" / "timesync_jetson.py",
                 here / "docs" / "timesync_jetson.py"):
        if not path.exists():
            tried.append(f"{path}: not present")
            continue
        try:
            spec = importlib.util.spec_from_file_location("timesync_jetson", path)
            module = importlib.util.module_from_spec(spec)
            # Must come before exec_module: @dataclass looks up the globals
            # of its class through sys.modules[cls.__module__] and
            # otherwise fails with "'NoneType' object has no attribute
            # '__dict__'". The import docs say the same.
            sys.modules["timesync_jetson"] = module
            spec.loader.exec_module(module)
        except Exception as exc:
            sys.modules.pop("timesync_jetson", None)
            tried.append(f"{path}: {exc}")
            continue
        found = accept(module, str(path))
        if found:
            return found
        # Do not let a broken module occupy the name.
        sys.modules.pop("timesync_jetson", None)

    raise ImportError(
        "timesync_jetson.py is missing or incomplete.\n"
        "The file belongs next to this one - in a ROS package that is\n"
        "the same folder as esp_serial_bridge.py, then rebuild.\n"
        "If an empty placeholder file sits there, it overrides the\n"
        "real one.\nTried:\n  " + "\n  ".join(tried))


_ts = _load_timesync()

Frame = _ts.Frame
FrameParser = _ts.FrameParser
EspClock = _ts.EspClock
TimeSync = _ts.TimeSync
monotonic = _ts.monotonic
START_BYTE = _ts.START_BYTE
BITS_PER_BYTE = _ts.BITS_PER_BYTE
read_available = _ts.read_available

# ==========================================================================
# Protocol
# ==========================================================================

# --- Jetson -> ESP ---
CMD_MOTOR = 0x10
CMD_SERVO = 0x20
CMD_LED = 0x30
CMD_PIXEL = 0x31            # RGBW LEDs, all
CMD_PIXEL_ONE = 0x32        # RGBW LEDs, one (index in front)
CMD_CALIBRATE = 0x40        # identical to CMD_CAL with action "start"
CMD_CAL = 0x41
CMD_TORQUE = 0x50
CMD_TRIM = 0x60
CMD_PID_SET = 0x80
CMD_PID_GET = 0x81
CMD_PID_SAVE = 0x83
CMD_MOVE = 0x90
CMD_MOVE_ABORT = 0x91
CMD_PROGRESS = 0x92
CMD_BATTERY = 0xA0
CMD_TIME_SYNC = 0xB0        # sent in timesync_jetson.py
CMD_STAMP_MODE = 0xB2
CMD_TELEM_RATE = 0xC0
CMD_EMERGENCY = 0xFF

# --- ESP -> Jetson ---
CMD_CAL_RSP = 0x42
CMD_BUTTON = 0x70
CMD_PID_RSP = 0x82
CMD_PID_SAVED = 0x84
CMD_MOVE_DONE = 0x93
CMD_PROGRESS_RSP = 0x94
CMD_BATTERY_RSP = 0xA1
CMD_BATTERY_WARN = 0xA2
CMD_TIME_RSP = 0xB1         # evaluated in timesync_jetson.py
CMD_STAMP_RSP = 0xB3
CMD_TELEMETRY = 0xC1

DUTY_MAX = 1023
TELEMETRY_MS_MIN = 10

MOVE_OK, MOVE_TIMEOUT, MOVE_ABORTED = 0x00, 0x01, 0x02
MOVE_STATUS_TEXT = {MOVE_OK: "ok", MOVE_TIMEOUT: "timeout", MOVE_ABORTED: "aborted"}

# PID parameters: index on the wire -> name. All values go out as
# int32 x 1000, the integer ones too (most common mistake, see spec).
PID_PARAMS = ["kp", "ki", "kd", "ilimit", "maxduty", "tol_deg", "settle_ms",
              "timeout_ms", "minduty"]

# Calibration actions in CMD_CAL
CAL_ACTIONS = {
    "start": 0x00, "minus": 0x01, "plus": 0x02, "center": 0x03,
    "left": 0x04, "right": 0x05, "save": 0x06, "abort": 0x07,
    "free": 0x08, "hold": 0x09, "goto_center": 0x0A, "step": 0x0B,
    "status": 0x0C,
}

# RGBW LED chain (SK6812): animation modes and named colours, the same as on
# the ESP's USB console ("px"). Colours are R, G, B, W - pure white uses the
# separate white chip.
PIXEL_MODES = {
    "off": 0, "solid": 1, "blink": 2, "breathe": 3,
    "rainbow": 4, "strobe": 5, "heart": 6,
}
PIXEL_COLOURS = {
    "red": (255, 0, 0, 0), "green": (0, 255, 0, 0), "blue": (0, 0, 255, 0),
    "white": (0, 0, 0, 255), "yellow": (255, 160, 0, 0),
    "orange": (255, 60, 0, 0), "cyan": (0, 255, 255, 0),
    "magenta": (255, 0, 255, 0), "purple": (120, 0, 255, 0),
    "warm": (255, 80, 0, 180),
}

# The ESP counts travel in 1/10 degree of the output shaft, ROS in radians.
DEG_TO_RAD = math.pi / 180.0


def _clamp(value: float, low: float, high: float) -> float:
    return low if value < low else high if value > high else value


# ==========================================================================
# Result data types
# ==========================================================================

@dataclass
class MoveDone:
    move_id: int
    status: int
    position_deg: float
    stamp: Optional[float]        # send time (Jetson clock), if stamped

    @property
    def ok(self) -> bool:
        return self.status == MOVE_OK


@dataclass
class Progress:
    move_id: int
    active: bool
    percent: int
    position_deg: float
    target_deg: float


@dataclass
class Telemetry:
    """Drive state that the ESP sends on its own at the configured rate."""

    #: Position of the output shaft, absolute since ESP boot. Signed.
    position_deg: float
    #: Rotational speed. **Signed**, negative = reverse.
    speed_deg_s: float
    #: What is applied to the H-bridge, -1023..+1023. Signed.
    duty: int
    #: Motor current - **always positive**. The VNH5019 only reports the
    #: magnitude, the direction is in ``duty``.
    current_a: float

    @property
    def speed_rad_s(self) -> float:
        return self.speed_deg_s * DEG_TO_RAD


@dataclass
class Battery:
    pack_v: float
    cell_v: float
    warning: bool


@dataclass
class CalState:
    active: bool
    have_center: bool
    have_left: bool
    have_right: bool
    torque_free: bool
    status: int
    pos: int
    center: int
    left: int
    right: int


CAL_STATUS_TEXT = {
    0x00: "done", 0x01: "saved", 0x02: "rejected",
    0x03: "servo not responding", 0x04: "send 0x40 first",
    0x05: "end of range reached",
}


# ==========================================================================
# EspLink - protocol without ROS
# ==========================================================================

class EspLink:
    """Serial link to the ESP: send, receive, sync the clocks.

    The reader thread takes everything the ESP sends on its own, feeds the
    clock sync and calls the registered callbacks. For replies that someone
    is waiting for (PID, calibration) there is also ``wait_for``.

    **The heartbeat is mandatory.** The ESP lets the motor coast when no
    command arrives for 5 s. ``tick()`` therefore resends the last motor
    command periodically - on purpose not during a position move, that one
    may run longer.
    """

    def __init__(self, port: str, baud: int = 115200, servo_id: int = 1,
                 log: Optional[Callable[[str], None]] = None) -> None:
        import serial          # only here, so the self-test works without it

        self._ser = serial.Serial(port, baud, timeout=0.05)
        self._log = log or (lambda msg: None)
        self.servo_id = servo_id

        self._write_lock = threading.Lock()
        self._parser = FrameParser()
        self.clock = EspClock()
        self.sync = TimeSync(self._send_timed, self.clock)

        self._callbacks: Dict[int, List[Callable[[Frame], None]]] = {}
        self._waiters: Dict[int, List[threading.Event]] = {}
        self._last_payload: Dict[int, bytes] = {}

        # Motor heartbeat
        self._motor_frame: Optional[bytes] = None
        self._last_motor_tx = 0.0
        self._move_active = False
        self._next_move_id = 1

        self.stamp_mode = False
        self.console_lines: Deque[str] = deque(maxlen=200)
        self.rx_frames = 0
        self.tx_frames = 0

        self._stop = threading.Event()
        self._reader = threading.Thread(target=self._read_loop, daemon=True,
                                        name="esp-rx")

    # --- Life cycle ------------------------------------------------------

    def start(self, sync_rounds: int = 12, stamp: bool = True) -> None:
        """Start the reader thread, sync the clocks, switch stamping on.

        The order matters: without the clock offset a timestamp is worthless,
        so measure first, then turn on stamping.
        """
        self._reader.start()

        for _ in range(sync_rounds):
            self.sync.request()
            time.sleep(0.02)
        time.sleep(0.05)

        if self.clock.valid:
            self._log(f"Clocks synced: offset {self.clock.offset * 1e3:+.3f} ms, "
                      f"round trip {self.clock.best_rtt * 1e3:.3f} ms, "
                      f"Drift {self.clock.drift_ppm:+.1f} ppm")
        else:
            self._log("WARNING: no reply to TIME_SYNC - timestamps stay empty")

        if stamp:
            self.set_stamp_mode(True)

    def close(self) -> None:
        self._stop.set()
        if self._reader.is_alive():
            self._reader.join(timeout=1.0)
        try:
            self.motor_coast()
        except Exception:
            pass
        self._ser.close()

    # --- Sending ---------------------------------------------------------

    def _send_timed(self, frame: bytes) -> float:
        """Send a frame and return when its last byte was out.

        The time is **computed, not measured**: clock before the write plus
        transfer time (10 bits per byte). Calling ``flush()`` afterwards
        would be the obvious way, but it does not work - ``tcdrain()`` returns
        on the Jetson's Tegra UART much later than the last byte goes out,
        and then the measured round trip turns negative.

        The ``flush()`` before it stays: it empties the buffer so our frame
        goes out at once and the calculation holds.
        """
        with self._write_lock:
            self._ser.flush()
            started = monotonic()
            self._ser.write(frame)
            self.tx_frames += 1
        return started + len(frame) * BITS_PER_BYTE / self._ser.baudrate

    def send(self, cmd: int, payload: bytes = b"") -> None:
        """Send one packet. Always in a single ``write()`` - the ESP
        drops a packet if more than 100 ms pass between two bytes."""
        self._send_timed(bytes([START_BYTE, cmd]) + payload)

    # --- Drive commands --------------------------------------------------

    def motor(self, duty: int) -> None:
        """Open-loop motor control, -1023..+1023. 0 = let it coast.

        Active braking only through ``emergency()``.
        """
        duty = int(_clamp(duty, -DUTY_MAX, DUTY_MAX))
        reverse = 1 if duty < 0 else 0
        speed = abs(duty)
        self._motor_frame = bytes([START_BYTE, CMD_MOTOR, reverse]) + \
            struct.pack(">H", speed)
        self._send_timed(self._motor_frame)
        self._last_motor_tx = monotonic()
        # Every motor command replaces a running position move - duty 0
        # too. The ESP then acks the old move with "aborted".
        self._move_active = False

    def motor_coast(self) -> None:
        self.motor(0)

    def steer(self, percent: float) -> None:
        """Steering, -100 (right) .. +100 (left). 0 = straight ahead."""
        pct = int(round(_clamp(percent, -100, 100)))
        self.send(CMD_SERVO, bytes([self.servo_id]) + struct.pack(">h", pct))

    def emergency(self) -> None:
        """Emergency halt with active braking. Also aborts a position move."""
        self._motor_frame = None
        self._move_active = False
        self.send(CMD_EMERGENCY)

    def led(self, on: bool) -> None:
        self.send(CMD_LED, bytes([1 if on else 0]))

    def pixel(self, cmd: "PixelCommand") -> None:
        """RGBW LEDs: ``cmd.led = None`` -> all (0x31), else one (0x32).

        No reply - an LED index the chain does not have is silently
        dropped by the ESP.
        """
        def byte(value: int) -> int:
            return int(_clamp(int(value), 0, 255))

        payload = bytes([byte(cmd.mode), byte(cmd.r), byte(cmd.g), byte(cmd.b),
                         byte(cmd.w), byte(cmd.brightness)]) \
            + struct.pack(">H", int(_clamp(int(cmd.period_ms), 0, 0xFFFF))) \
            + bytes([byte(cmd.count)])
        if cmd.led is None:
            self.send(CMD_PIXEL, payload)
        else:
            self.send(CMD_PIXEL_ONE, bytes([byte(cmd.led)]) + payload)

    def trim(self, action: int) -> None:
        """0 = centre to the left, 1 = to the right, 2 = save."""
        self.send(CMD_TRIM, bytes([action & 0xFF]))

    def torque_report(self) -> None:
        """Print the servo load on the ESP's USB console. No UART reply."""
        self.send(CMD_TORQUE)

    # --- Position move ---------------------------------------------------

    def move(self, degrees: float) -> int:
        """Turn on by ``degrees`` (relative!). Returns the move_id.

        Only one move runs at a time; a new one replaces the old one, and the
        old one acks with status "aborted".
        """
        move_id = self._next_move_id
        self._next_move_id = self._next_move_id % 255 + 1   # avoid 0
        deg10 = int(round(degrees * 10.0))
        # Set BEFORE sending. The node runs with a ReentrantCallbackGroup
        # in a MultiThreadedExecutor; if this line came after, the speed
        # controller in the other thread would still see "no move", send
        # duty 0 right after, and the ESP would ack the move it had just
        # started as aborted. Exactly that was seen on the robot:
        # 14 ms between CMD_MOVE and MOVE_ABORTED.
        self._move_active = True
        self.send(CMD_MOVE, bytes([move_id]) + struct.pack(">i", deg10))
        return move_id

    def move_abort(self) -> None:
        self.send(CMD_MOVE_ABORT)

    def request_progress(self) -> None:
        self.send(CMD_PROGRESS)

    # --- PID -------------------------------------------------------------

    def pid_set(self, param: int | str, value: float) -> None:
        """Set one controller parameter. Volatile only - ``pid_save()`` writes
        the whole set to NVS."""
        index = PID_PARAMS.index(param) if isinstance(param, str) else int(param)
        raw = int(round(value * 1000.0))
        self.send(CMD_PID_SET, bytes([index]) + struct.pack(">i", raw))

    def pid_get(self, timeout: float = 1.0) -> Optional[Tuple[float, float, float]]:
        payload = self.request(CMD_PID_GET, CMD_PID_RSP, timeout)
        if payload is None:
            return None
        kp, ki, kd = struct.unpack(">iii", payload)
        return kp / 1000.0, ki / 1000.0, kd / 1000.0

    def pid_save(self, timeout: float = 2.0) -> Optional[bool]:
        payload = self.request(CMD_PID_SAVE, CMD_PID_SAVED, timeout)
        return None if payload is None else payload[0] == 0x00

    # --- Steering calibration --------------------------------------------

    def calibrate(self, action: int | str = "start", arg: int = 0,
                  timeout: float = 1.5) -> Optional[CalState]:
        """Run one calibration action. Each one is answered with CAL_RSP.

        The servo cannot limit its torque - so on purpose there is no
        action that drives to the end stop on its own.
        """
        code = CAL_ACTIONS[action] if isinstance(action, str) else int(action)
        payload = self.request(CMD_CAL, CMD_CAL_RSP, timeout,
                               bytes([code, arg & 0xFF]))
        return None if payload is None else parse_cal_state(payload)

    # --- Battery and time ------------------------------------------------

    def request_battery(self) -> None:
        self.send(CMD_BATTERY)

    def set_stamp_mode(self, on: bool, timeout: float = 1.0) -> Optional[bool]:
        payload = self.request(CMD_STAMP_MODE, CMD_STAMP_RSP, timeout,
                               bytes([1 if on else 0]))
        if payload is not None:
            self.stamp_mode = payload[0] != 0
            return self.stamp_mode
        return None

    def set_telemetry_rate(self, period_s: float) -> float:
        """Set the rate of the drive telemetry. 0 switches it off.

        Returns the period actually set, in seconds - the ESP accepts
        nothing below 20 ms, because it only updates the speed every
        100 ms anyway. The position is fresh in every packet.
        """
        ms = 0 if period_s <= 0 else max(TELEMETRY_MS_MIN,
                                         min(60000, int(round(period_s * 1000))))
        self.send(CMD_TELEM_RATE, struct.pack(">H", ms))
        return ms / 1000.0

    def resync(self) -> None:
        """One round of clock sync. Must happen regularly, otherwise the
        crystal drift runs away (~0.1 ms per second)."""
        self.sync.request()

    # --- Heartbeat -------------------------------------------------------

    def tick(self, heartbeat_period: float = 0.2) -> None:
        """Call regularly. Resends the last motor command so that the
        ESP's 5 s watchdog does not trip."""
        if self._motor_frame is None or self._move_active:
            return
        if monotonic() - self._last_motor_tx >= heartbeat_period:
            self._send_timed(self._motor_frame)
            self._last_motor_tx = monotonic()

    # --- Receiving -------------------------------------------------------

    def on(self, cmd: int, callback: Callable[[Frame], None]) -> None:
        """Register a callback for a packet type. Runs in the reader thread."""
        self._callbacks.setdefault(cmd, []).append(callback)

    def request(self, cmd: int, answer: int, timeout: float,
                payload: bytes = b"") -> Optional[bytes]:
        """Send a command and wait for the matching reply.

        Replies arrive in the reader thread, so the caller may block.
        """
        event = threading.Event()
        self._waiters.setdefault(answer, []).append(event)
        self.send(cmd, payload)
        if not event.wait(timeout):
            try:
                self._waiters[answer].remove(event)
            except (KeyError, ValueError):
                pass          # the reader thread was faster
            self._log(f"Timeout: no reply 0x{answer:02X} to 0x{cmd:02X}")
            return None
        return self._last_payload.get(answer)

    def _read_loop(self) -> None:
        while not self._stop.is_set():
            try:
                # Not read(64): that waits until 64 bytes are together, and
                # then all packets in it carry the same receive time.
                data = read_available(self._ser)
            except Exception as exc:            # port gone (USB unplugged or similar)
                self._log(f"Read error: {exc}")
                break
            if not data:
                continue

            for frame in self._parser.feed(data):
                self._dispatch(frame)

            if self._parser.text:
                for line in self._parser.text:
                    self.console_lines.append(line.decode("ascii", "replace"))
                self._parser.text.clear()

    def _dispatch(self, frame: Frame) -> None:
        self.rx_frames += 1

        # Clock sync first: its reply must not wait anywhere.
        if self.sync.handle(frame) is not None:
            return

        if frame.cmd in (CMD_MOVE_DONE, CMD_PROGRESS_RSP):
            if frame.cmd == CMD_MOVE_DONE:
                self._move_active = False
            elif frame.payload[1] == 0:
                self._move_active = False

        # Wake up waiting callers
        waiters = self._waiters.pop(frame.cmd, None)
        if waiters:
            self._last_payload[frame.cmd] = frame.payload
            for event in waiters:
                event.set()

        for callback in self._callbacks.get(frame.cmd, ()):
            try:
                callback(frame)
            except Exception as exc:
                self._log(f"Callback for 0x{frame.cmd:02X} failed: {exc}")

    # --- State -----------------------------------------------------------

    @property
    def move_active(self) -> bool:
        """Is a position move running? While it runs, the heartbeat
        pauses - moves may take longer than the 5 s limit."""
        return self._move_active

    @property
    def parser(self) -> FrameParser:
        return self._parser

    # --- Timestamps ------------------------------------------------------

    def sent_at(self, frame: Frame) -> Optional[float]:
        """When the packet was sent, in the Jetson clock."""
        return self.clock.frame_time(frame)

    def latency(self, frame: Frame) -> Optional[float]:
        """How long it took from "sent" to "read"."""
        return self.clock.latency(frame)


# ==========================================================================
# Unpacking payloads
# ==========================================================================

def parse_move_done(payload: bytes, stamp: Optional[float] = None) -> MoveDone:
    move_id, status = payload[0], payload[1]
    deg10 = struct.unpack(">i", payload[2:6])[0]
    return MoveDone(move_id, status, deg10 / 10.0, stamp)


def parse_progress(payload: bytes) -> Progress:
    move_id, active, percent = payload[0], payload[1], payload[2]
    pos, target = struct.unpack(">ii", payload[3:11])
    return Progress(move_id, bool(active), percent, pos / 10.0, target / 10.0)


def parse_telemetry(payload: bytes) -> Telemetry:
    pos_deg10, speed_deg10_s = struct.unpack(">ii", payload[0:8])
    duty, current_ma = struct.unpack(">hh", payload[8:12])
    return Telemetry(pos_deg10 / 10.0, speed_deg10_s / 10.0, duty,
                     current_ma / 1000.0)


def parse_battery(payload: bytes, warning: bool) -> Battery:
    pack_mv = struct.unpack(">i", payload[0:4])[0]
    cell_mv = struct.unpack(">h", payload[4:6])[0]
    return Battery(pack_mv / 1000.0, cell_mv / 1000.0, warning)


def parse_cal_state(payload: bytes) -> CalState:
    active, flags, status = payload[0], payload[1], payload[2]
    pos, center, left, right = struct.unpack(">hhhh", payload[3:11])
    return CalState(bool(active), bool(flags & 0x01), bool(flags & 0x02),
                    bool(flags & 0x04), bool(flags & 0x08),
                    status, pos, center, left, right)


# ==========================================================================
# RGBW LEDs - plain-text commands
# ==========================================================================

@dataclass
class PixelCommand:
    """One packet for the LED chain. ``led = None`` means all LEDs."""
    led: Optional[int] = None
    mode: int = PIXEL_MODES["solid"]
    r: int = 0
    g: int = 0
    b: int = 0
    w: int = 0
    brightness: int = 64
    period_ms: int = 0       # 0 = default of the mode on the ESP
    count: int = 0           # 0 = persistent, n = one-shot over n periods


def parse_pixel(text: str, brightness: int = 64) -> List[PixelCommand]:
    """Plain text -> LED packets. Several commands separated by ``;``.

    One command::

        [<led>:] [<mode>] [<colour>] [bri=<0-255>] [ms=<period>] [x=<count>]

    * ``<led>:`` only this LED (0 = nearest to the ESP), without it: all
    * ``<mode>`` off, solid, blink, breathe, rainbow, strobe, heart
    * ``<colour>`` a name (red, green, ... see PIXEL_COLOURS), ``#RRGGBB``,
      ``#RRGGBBWW`` or ``rgbw=R,G,B[,W]``
    * ``x=<n>`` makes it a one-shot: n periods, then the previous state
      returns on the ESP by itself

    Stateless on purpose: whatever is not given takes its default (mode
    solid, colour white, brightness from the parameter, period of the mode).
    So every message means the same no matter what was sent before - the
    bridge does not know what the ESP console did in the meantime anyway.

    Examples: ``red``, ``2: breathe blue ms=2000``,
    ``0: strobe red ms=300 x=3``, ``0: red; 1: green; 2: blue; 3: white``.

    Raises ``ValueError`` with a readable message on anything unknown.
    """
    commands: List[PixelCommand] = []
    for part in text.split(";"):
        part = part.strip()
        if not part:
            continue
        cmd = PixelCommand(brightness=brightness)
        colour: Optional[Tuple[int, int, int, int]] = None
        mode: Optional[int] = None

        if ":" in part:
            head, part = part.split(":", 1)
            head = head.strip().lower()
            if head != "all":
                if not head.isdigit():
                    raise ValueError(f"LED index '{head}' is not a number")
                cmd.led = int(head)

        # rgbw= first: its value contains commas/spaces and would otherwise
        # fall apart into several words.
        match = re.search(r"rgbw\s*=\s*([0-9][0-9 ,]*)", part, re.IGNORECASE)
        if match:
            values = [int(n) for n in re.split(r"[ ,]+", match.group(1).strip())]
            if len(values) not in (3, 4):
                raise ValueError("rgbw= needs 3 or 4 numbers")
            colour = tuple((values + [0])[:4])
            part = part[:match.start()] + " " + part[match.end():]

        for token in part.split():
            low = token.lower()
            if "=" in low:
                key, value = low.split("=", 1)
                if not value.isdigit():
                    raise ValueError(f"'{token}': value is not a number")
                if key in ("bri", "brightness"):
                    cmd.brightness = int(value)
                elif key in ("ms", "period"):
                    cmd.period_ms = int(value)
                elif key in ("x", "count"):
                    cmd.count = int(value)
                else:
                    raise ValueError(f"unknown key '{key}' (bri, ms, x, rgbw)")
            elif low in PIXEL_MODES:
                mode = PIXEL_MODES[low]
            elif low in PIXEL_COLOURS:
                colour = PIXEL_COLOURS[low]
            elif low.startswith("#") and len(low) in (7, 9):
                try:
                    v = int(low[1:], 16)
                except ValueError:
                    raise ValueError(f"'{token}' is not a hex colour") from None
                if len(low) == 7:
                    v <<= 8
                colour = ((v >> 24) & 0xFF, (v >> 16) & 0xFF, (v >> 8) & 0xFF, v & 0xFF)
            else:
                raise ValueError(
                    f"unknown word '{token}'. Modes: {', '.join(PIXEL_MODES)}; "
                    f"colours: {', '.join(PIXEL_COLOURS)}, #RRGGBB[WW], rgbw=R,G,B[,W]")

        cmd.mode = PIXEL_MODES["solid"] if mode is None else mode
        cmd.r, cmd.g, cmd.b, cmd.w = colour if colour is not None else PIXEL_COLOURS["white"]
        for name in ("brightness", "count"):
            if not 0 <= getattr(cmd, name) <= 255:
                raise ValueError(f"{name} must be 0..255")
        if not 0 <= cmd.period_ms <= 0xFFFF:
            raise ValueError("ms must be 0..65535")
        if any(not 0 <= c <= 255 for c in (cmd.r, cmd.g, cmd.b, cmd.w)):
            raise ValueError("colour values must be 0..255")
        if cmd.led is not None and cmd.led > 255:
            raise ValueError("LED index must be 0..255")
        commands.append(cmd)

    if not commands:
        raise ValueError("empty LED command")
    return commands


# ==========================================================================
# ROS 2 node
# ==========================================================================

def _build_node_class():
    """Build the node class only once rclpy is there - so the self-test
    also runs on a machine without ROS."""

    import rclpy
    from rclpy.callback_groups import ReentrantCallbackGroup
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
    from rclpy.time import Time

    from builtin_interfaces.msg import Time as TimeMsg
    from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue
    from geometry_msgs.msg import Twist
    from sensor_msgs.msg import BatteryState, JointState
    from std_msgs.msg import Bool, Empty, Float32, Float32MultiArray, Float64
    from std_msgs.msg import Header
    from std_msgs.msg import Int32, Int32MultiArray, String
    from std_srvs.srv import SetBool, Trigger

    from esp_bridge.steer_lut import SteerLUT

    class EspBridgeNode(Node):
        """All functions of the ESP as topics and services.

        Split: what forms a stream (drive commands, telemetry) is a
        topic, what has an ack (save PID, calibrate) is a service. All
        messages with a header carry the **ESP's send time** as ``stamp``,
        converted into the ROS clock.
        """

        def __init__(self) -> None:
            super().__init__("esp_serial_bridge")
            group = ReentrantCallbackGroup()

            # --- Parameters ---
            self.declare_parameter("port", "/dev/ttyTHS1")
            self.declare_parameter("baud", 115200)
            self.declare_parameter("servo_id", 1)
            self.declare_parameter("stamp_mode", True)
            self.declare_parameter("sync_rounds", 12)
            self.declare_parameter("sync_interval", 10.0)
            self.declare_parameter("heartbeat_period", 0.2)
            self.declare_parameter("cmd_vel_timeout", 0.5)
            self.declare_parameter("battery_period", 5.0)
            self.declare_parameter("progress_period", 0.2)
            # Rate at which the ESP sends position and speed on its own.
            # Can be changed at runtime:
            #   ros2 param set /esp_serial_bridge telemetry_period 0.1
            self.declare_parameter("telemetry_period", 0.01)  # 0 = off
            # cmd_vel is open-loop: the ESP does not control the speed.
            # These two values are the conversion and have to be measured
            # on the car.
            self.declare_parameter("v_max", 1.648)         # m/s at PWM 1.0
            self.declare_parameter("pwm_deadband", 0.076)  # PWM fraction at which it breaks away
            self.declare_parameter("v_eps", 0.01)          # below this: standstill
            self.declare_parameter("max_angular", 1.0)     # rad/s at full lock
            self.declare_parameter("vel_accel", 0.8)

            # Ackermann steering: angular.z (rad/s) -> steering angle -> servo percent.
            # delta = atan(L*omega/v); curve servo = (delta - b)/a, per side.
            self.declare_parameter("wheelbase", 0.10)         # L [m]
            self.declare_parameter("steer_a_left", 0.3643)    # rad per servo unit (CCW)
            self.declare_parameter("steer_b_left", -0.01985)   # rad Offset (CCW)
            self.declare_parameter("steer_a_right", 0.2962)   # rad per servo unit (CW)
            self.declare_parameter("steer_b_right", 0.00377)  # rad Offset (CW)
            self.declare_parameter('steer_calib_path', '/workspace/src/esp_bridge/esp_bridge/steer_calib.json')
            self.declare_parameter("steer_v_min", 0.05)       # below: clamp delta at v_min
            self.declare_parameter("steer_raw_bypass", False)

            self.declare_parameter("vel_kp", 200.0)      # duty per (m/s) error
            self.declare_parameter("vel_ki", 800.0)      # duty per (m/s * s)
            self.declare_parameter("vel_i_limit", 600.0) # anti-windup limit (duty)
            self.declare_parameter("vel_control_rate", 50.0)
            self.declare_parameter("odom_stale_s", 0.15) # after that: feedforward only
            self.declare_parameter("odom_stop_s", 0.50)  # after that: stop the motor

            self.declare_parameter("steer_center_servo", -0.04)

            # RGBW LEDs: brightness for ~/pixel when the text gives none.
            self.declare_parameter("pixel_brightness", 64)

            self._p = lambda name: self.get_parameter(name).value
            self._cmd_vel_timeout = float(self._p("cmd_vel_timeout"))
            self._last_cmd_vel = 0.0
            self._v_target = 0.0
            self._v_actual = 0.0
            self._v_ramp = 0.0
            self._last_odom = 0.0
            # Serialises everything that writes the motor command. Without it
            # the lock on move_active is not enough: between the check and
            # the send the other thread can start the move.
            self._motor_lock = threading.Lock()
            self._vel_integral = 0.0
            self._ctrl_dt = 1.0 / (float(self._p("vel_control_rate")) or 50.0)

            # --- Connection ---
            port = self._p("port")
            self.get_logger().info(f"opening {port} @ {self._p('baud')} baud")
            self.link = EspLink(port, int(self._p("baud")),
                                int(self._p("servo_id")),
                                log=self.get_logger().info)

            # ESP packets arrive in the reader thread; from there only into a
            # queue, publishing happens on the executor thread.
            self._inbox: Deque[Tuple[Frame, Time]] = deque(maxlen=500)
            for cmd in (CMD_BUTTON, CMD_MOVE_DONE, CMD_PROGRESS_RSP,
                        CMD_BATTERY_RSP, CMD_BATTERY_WARN, CMD_CAL_RSP,
                        CMD_PID_RSP, CMD_STAMP_RSP, CMD_TELEMETRY):
                self.link.on(cmd, self._enqueue)

            self._make_publishers()
            self._make_subscribers(group)
            self._make_services(group)

            self.link.start(sync_rounds=int(self._p("sync_rounds")),
                            stamp=bool(self._p("stamp_mode")))

            self.steer_lut = SteerLUT(
                self.get_parameter('steer_calib_path').value,
                logger=self.get_logger())

            # Switch on only after the clock sync - otherwise the first
            # telemetry packets would arrive without a usable timestamp.
            self._apply_telemetry_period(float(self._p("telemetry_period")))
            self.add_on_set_parameters_callback(self._on_set_parameters)

            # --- Timer ---
            self.create_timer(0.01, self._drain, callback_group=group)
            self.create_timer(float(self._p("heartbeat_period")) / 2.0,
                              self._heartbeat, callback_group=group)
            self.create_timer(float(self._p("sync_interval")),
                              self._resync, callback_group=group)
            self.create_timer(float(self._p("battery_period")),
                              lambda: self.link.request_battery(),
                              callback_group=group)
            self.create_timer(float(self._p("progress_period")),
                              self._poll_progress, callback_group=group)
            self.create_timer(1.0, self._publish_link_status, callback_group=group)
            self.create_timer(self._ctrl_dt, self._velocity_control, callback_group=group)

            self.get_logger().info("Bridge ready")

        # --- Setup -------------------------------------------------------

        def _make_publishers(self) -> None:
            # Warnings and states should still reach a subscriber that
            # starts late.
            latched = QoSProfile(depth=1,
                                 reliability=ReliabilityPolicy.RELIABLE,
                                 durability=DurabilityPolicy.TRANSIENT_LOCAL)

            self.pub_button = self.create_publisher(Header, "~/button", 10)
            self.pub_joints = self.create_publisher(JointState, "~/joint_states", 10)
            self.pub_move_done = self.create_publisher(Int32MultiArray, "~/move_done", 10)
            self.pub_progress = self.create_publisher(Float32, "~/move_progress", 10)
            self.pub_battery = self.create_publisher(BatteryState, "~/battery", 10)
            self.pub_battery_low = self.create_publisher(Bool, "~/battery_low", latched)
            self.pub_cal = self.create_publisher(Int32MultiArray, "~/cal_state", 10)
            self.pub_pid = self.create_publisher(Float32MultiArray, "~/pid", latched)
            self.pub_speed = self.create_publisher(Float32, "~/speed", 10)
            self.pub_motor_state = self.create_publisher(
                Float32MultiArray, "~/motor_state", 10)
            self.pub_console = self.create_publisher(String, "~/console", 20)
            self.pub_status = self.create_publisher(DiagnosticArray, "/diagnostics", 10)

            # Numbers instead of text - so Foxglove can plot them.
            # /diagnostics carries the same values, but as strings.
            self.pub_latency = self.create_publisher(Float32, "~/latency_ms", 50)
            self.pub_rtt = self.create_publisher(Float32, "~/rtt_ms", 10)
            # The offset is several million ms - float32 only has 1 ms
            # steps there and the drift would be invisible.
            self.pub_offset = self.create_publisher(Float64, "~/offset_ms", 10)
            self.pub_drift = self.create_publisher(Float32, "~/drift_ppm", 10)

        def _make_subscribers(self, group) -> None:
            def sub(msg_type, name, handler):
                return self.create_subscription(msg_type, name, handler, 10, callback_group=group)
            self.create_subscription(Odometry, "/ekf/odom", self._odom_cb, 10, callback_group=group)    

            sub(Twist, "/cmd_vel", self._on_cmd_vel)
            sub(Int32, "~/motor", lambda m: self._drive(m.data))
            sub(Float32, "~/steer", lambda m: self.link.steer(m.data))
            sub(Bool, "~/led", lambda m: self.link.led(m.data))
            sub(String, "~/pixel", self._on_pixel)
            sub(Int32MultiArray, "~/pixel_raw", self._on_pixel_raw)
            sub(Float32, "~/move", self._on_move)
            sub(Int32, "~/trim", lambda m: self.link.trim(m.data))
            sub(Empty, "~/emergency", lambda _m: self._on_emergency())
            sub(Float32MultiArray, "~/pid_set", self._on_pid_set)
            sub(Int32MultiArray, "~/cal", self._on_cal)
            sub(String, "~/cal_action", self._on_cal_action)

        def _make_services(self, group) -> None:
            def srv(srv_type, name, handler):
                return self.create_service(srv_type, name, handler,
                                           callback_group=group)

            srv(Trigger, "~/emergency_stop", self._srv_emergency)
            srv(Trigger, "~/move_abort", self._srv_move_abort)
            srv(Trigger, "~/pid_get", self._srv_pid_get)
            srv(Trigger, "~/pid_save", self._srv_pid_save)
            srv(Trigger, "~/calibrate_start", self._srv_cal_start)
            srv(Trigger, "~/calibrate_save", self._srv_cal_save)
            srv(Trigger, "~/trim_save", self._srv_trim_save)
            srv(Trigger, "~/torque_report", self._srv_torque)
            srv(Trigger, "~/resync", self._srv_resync)
            srv(SetBool, "~/set_led", self._srv_led)
            srv(SetBool, "~/set_stamp_mode", self._srv_stamp)
            srv(SetBool, "~/servo_torque_free", self._srv_torque_free)

        # --- Timestamps --------------------------------------------------

        def _enqueue(self, frame: Frame) -> None:
            """Runs in the reader thread. Only record the ROS time of the
            read here, ``_drain`` does the rest."""
            self._inbox.append((frame, self.get_clock().now()))

        def _stamp(self, frame: Frame, read_at: Time) -> TimeMsg:
            """Send time of the packet as ROS time.

            The latency is measured in the monotonic clock (that is where the
            sync with the ESP lives) and subtracted from the ROS time of the
            read. So the stamp stays right, whether ROS runs on system time or
            on simulation time.
            """
            latency = self.link.latency(frame)
            if latency is None or not 0.0 <= latency < 1.0:
                return read_at.to_msg()      # unstamped or implausible
            self.pub_latency.publish(Float32(data=latency * 1e3))
            return Time(nanoseconds=read_at.nanoseconds - int(latency * 1e9)).to_msg()

        def _header(self, frame: Frame, read_at: Time, frame_id: str = "esp") -> Header:
            header = Header()
            header.stamp = self._stamp(frame, read_at)
            header.frame_id = frame_id
            return header

        # --- Incoming packets --------------------------------------------

        def _drain(self) -> None:
            while self._inbox:
                frame, read_at = self._inbox.popleft()
                try:
                    self._publish(frame, read_at)
                except Exception as exc:
                    self.get_logger().error(
                        f"Packet 0x{frame.cmd:02X} not processed: {exc}")

            while self.link.console_lines:
                self.pub_console.publish(String(data=self.link.console_lines.popleft()))

        def _publish(self, frame: Frame, read_at: Time) -> None:
            cmd, payload = frame.cmd, frame.payload

            if cmd == CMD_BUTTON:
                self.pub_button.publish(self._header(frame, read_at))

            elif cmd == CMD_MOVE_DONE:
                result = parse_move_done(payload)
                self.pub_move_done.publish(Int32MultiArray(
                    data=[result.move_id, result.status,
                          int(round(result.position_deg * 10))]))
                self._publish_joint(result.position_deg, frame, read_at)
                self.get_logger().info(
                    f"Move {result.move_id}: {MOVE_STATUS_TEXT.get(result.status, '?')} "
                    f"at {result.position_deg:+.1f} deg")

            elif cmd == CMD_PROGRESS_RSP:
                progress = parse_progress(payload)
                self.pub_progress.publish(Float32(data=float(progress.percent)))
                self._publish_joint(progress.position_deg, frame, read_at)

            elif cmd in (CMD_BATTERY_RSP, CMD_BATTERY_WARN):
                self._publish_battery(parse_battery(payload, cmd == CMD_BATTERY_WARN),
                                      frame, read_at)

            elif cmd == CMD_CAL_RSP:
                state = parse_cal_state(payload)
                self.pub_cal.publish(Int32MultiArray(data=[
                    int(state.active), int(state.have_center), int(state.have_left),
                    int(state.have_right), int(state.torque_free), state.status,
                    state.pos, state.center, state.left, state.right]))

            elif cmd == CMD_PID_RSP:
                kp, ki, kd = struct.unpack(">iii", payload)
                self.pub_pid.publish(Float32MultiArray(
                    data=[kp / 1000.0, ki / 1000.0, kd / 1000.0]))

            elif cmd == CMD_TELEMETRY:
                telemetry = parse_telemetry(payload)
                self.pub_speed.publish(Float32(data=telemetry.speed_deg_s))
                self.pub_motor_state.publish(Float32MultiArray(
                    data=[float(telemetry.duty), telemetry.current_a]))
                self._publish_joint(telemetry.position_deg, frame, read_at,
                                    velocity=telemetry.speed_rad_s)

            elif cmd == CMD_STAMP_RSP:
                self.get_logger().info(
                    f"Send timestamps {'on' if payload[0] else 'off'}")

        def _publish_joint(self, position_deg: float, frame: Frame,
                           read_at: Time,
                           velocity: Optional[float] = None) -> None:
            """Position of the output shaft, optionally with speed.

            MOVE_DONE and PROGRESS_RSP only know the position; the speed is
            only in CMD_TELEMETRY. An empty ``velocity`` means "not measured"
            in ROS - better than a zero written down.
            """
            msg = JointState()
            msg.header = self._header(frame, read_at)
            msg.name = ["drive_axle"]
            msg.position = [position_deg * DEG_TO_RAD]
            if velocity is not None:
                msg.velocity = [velocity]
            self.pub_joints.publish(msg)

        def _publish_battery(self, battery: Battery, frame: Frame,
                             read_at: Time) -> None:
            msg = BatteryState()
            msg.header = self._header(frame, read_at)
            msg.voltage = battery.pack_v
            msg.cell_voltage = [battery.cell_v] * 4
            msg.present = True
            # Rough estimate from the cell voltage. Without current measurement
            # and resting voltage it cannot be more exact - under load it sags.
            msg.percentage = float(_clamp((battery.cell_v - 3.3) / (4.2 - 3.3), 0.0, 1.0))
            msg.power_supply_status = BatteryState.POWER_SUPPLY_STATUS_DISCHARGING
            msg.power_supply_health = (
                BatteryState.POWER_SUPPLY_HEALTH_UNSPEC_FAILURE if battery.warning
                else BatteryState.POWER_SUPPLY_HEALTH_GOOD)
            self.pub_battery.publish(msg)
            self.pub_battery_low.publish(Bool(data=battery.warning))
            if battery.warning:
                self.get_logger().warn(
                    f"Undervoltage: {battery.cell_v:.3f} V/cell "
                    f"({battery.pack_v:.2f} V) - the ESP does NOT switch off")

        # --- Outgoing commands -------------------------------------------

        def _drive(self, duty: int) -> None:
            self.link.motor(int(duty))
            self._last_cmd_vel = monotonic()
        
        @staticmethod
        def _speed_to_duty(v: float, v_max: float, pwm_deadband: float,
                           v_eps: float) -> int:
            """m/s -> signed duty, with a deadband jump over the
            breakaway threshold. Standstill stays standstill."""
            if abs(v) < v_eps:
                return 0
            frac = min(abs(v) / v_max, 1.0)                      # 0..1 of the usable range
            pwm_frac = pwm_deadband + frac * (1.0 - pwm_deadband)
            duty = int(round(pwm_frac * DUTY_MAX))
            return duty if v > 0 else -duty

        def _on_cmd_vel(self, msg: Twist) -> None:
            """Buffer the Twist. The motor is set by the control timer;
            here only cache the target speed and send the steering directly
            (that one is not closed-loop, open-loop control)."""
            self._v_target = float(msg.linear.x)
            steer = self._omega_to_servo(float(msg.angular.z))
            self.link.steer(steer)
            self._last_cmd_vel = monotonic()

        def _omega_to_servo(self, omega: float) -> float:
            if bool(self._p("steer_raw_bypass")):
                return float(_clamp(omega, -1.0, 1.0) * 100.0)
            return float(self.steer_lut.servo_for(omega, self._v_actual) * 100.0)

        def _odom_cb(self, msg) -> None:
            """Actual speed (scalar, forward) from the EKF."""
            self._v_actual = float(msg.twist.twist.linear.x)
            self._last_odom = monotonic()

        def _velocity_control(self) -> None:
            """Fixed rate: feedforward + PI on v_target - v_actual -> duty.
            Sole writer of the motor command."""
            now = monotonic()
            if not self._motor_lock.acquire(blocking=False):
                return                      # someone is starting a move right now
            try:
                self._velocity_control_locked(now)
            finally:
                self._motor_lock.release()

        def _velocity_control_locked(self, now: float) -> None:
            # A running position move belongs to the ESP. EVERY motor command
            # from here replaces it -- also the duty 0 from the timeout branch
            # right below, because EspLink.motor() resets _move_active.
            # Without this lock ~/move is dead after cmd_vel_timeout
            # (0.5 s) at the latest, no matter how long the move is.
            if self.link.move_active:
                self._vel_integral = 0.0
                self._v_ramp = 0.0
                return

            # no current /cmd_vel -> stop, reset the integrator AND the
            # ramp. Without the ramp, after a run that ended mid-drive
            # v_ramp stayed at ~0.35 m/s; the first /cmd_vel of the next
            # run (v = 0) then first ramped down from 0.35 and drove the
            # car 7-13 cm forward doing so -- out of the parking bay, before
            # the first unpark move (runs parken_test_21, 22, 25).
            if (self._cmd_vel_timeout > 0 and
                    (self._last_cmd_vel == 0.0 or
                     now - self._last_cmd_vel > self._cmd_vel_timeout)):
                self._vel_integral = 0.0
                self._v_ramp = 0.0
                self.link.motor(0)
                return

            a_max = float(self._p("vel_accel"))          # m/s^2, new parameter
            dv = a_max * self._ctrl_dt
            target = self._v_target
            if target > self._v_ramp:
                self._v_ramp = min(target, self._v_ramp + dv)
            else:
                self._v_ramp = max(target, self._v_ramp - dv)
            v_target = self._v_ramp

            # Standstill: hold the integrator, duty 0 (do not fight the noise)
            if abs(v_target) < float(self._p("v_eps")):
                self.link.motor(0)
                return

            v_max = float(self._p("v_max")) or 1.0
            pwm_deadband = float(self._p("pwm_deadband"))
            v_eps = float(self._p("v_eps"))
            duty_ff = self._speed_to_duty(v_target, v_max, pwm_deadband, v_eps)

            odom_age = now - self._last_odom if self._last_odom else 1e9

            # no feedback for too long -> stop
            if odom_age > float(self._p("odom_stop_s")):
                self._vel_integral = 0.0
                self.link.motor(0)
                self.get_logger().warn(
                    "no /ekf/odom - speed controller stops",
                    throttle_duration_sec=2.0)
                return

            # feedback briefly gone -> feedforward only, freeze the integrator
            if odom_age > float(self._p("odom_stale_s")):
                duty = duty_ff + self._vel_integral
            else:
                error = v_target - self._v_actual
                self._vel_integral += float(self._p("vel_ki")) * error * self._ctrl_dt
                i_limit = float(self._p("vel_i_limit"))
                self._vel_integral = _clamp(self._vel_integral, -i_limit, i_limit)
                duty = duty_ff + float(self._p("vel_kp")) * error + self._vel_integral

            self.link.motor(int(round(_clamp(duty, -DUTY_MAX, DUTY_MAX))))

        def _on_move(self, msg: Float32) -> None:
            with self._motor_lock:
                move_id = self.link.move(float(msg.data))
            self.get_logger().info(f"Move {move_id}: {msg.data:+.1f} deg (relative)")

        def _on_emergency(self) -> None:
            self.link.emergency()
            self.get_logger().warn("EMERGENCY HALT")

        def _on_pid_set(self, msg: Float32MultiArray) -> None:
            """[paramId, value] or the whole set at once [kp, ki, kd, ...]."""
            data = list(msg.data)
            if len(data) == 2:
                self.link.pid_set(int(data[0]), data[1])
            elif len(data) == len(PID_PARAMS):
                for index, value in enumerate(data):
                    self.link.pid_set(index, value)
            else:
                self.get_logger().error(
                    f"pid_set: expected 2 or {len(PID_PARAMS)} values, "
                    f"got {len(data)}")
                return
            self.get_logger().info("PID set (volatile - ~/pid_save to keep it)")

        def _on_cal_action(self, msg: String) -> None:
            """Calibrate in plain text, e.g. "plus", "left", "save".

            Meant for manual calibration on the command line:
                ros2 topic pub --once <node>/cal_action std_msgs/String "data: plus"
            """
            name = msg.data.strip().lower()
            if name not in CAL_ACTIONS:
                self.get_logger().error(
                    f"unknown action '{name}'. Possible: "
                    + ", ".join(sorted(CAL_ACTIONS)))
                return
            self._run_cal(CAL_ACTIONS[name], 0)

        def _on_pixel(self, msg: String) -> None:
            """RGBW LEDs in plain text, syntax see ``parse_pixel``:
                ros2 topic pub --once <node>/pixel std_msgs/String "data: '2: breathe blue'"
            """
            try:
                commands = parse_pixel(msg.data,
                                       int(self._p("pixel_brightness")))
            except ValueError as exc:
                self.get_logger().error(f"pixel: {exc}")
                return
            for cmd in commands:
                self.link.pixel(cmd)

        def _on_pixel_raw(self, msg: Int32MultiArray) -> None:
            """[led, mode, r, g, b, w, brightness, period_ms, count],
            led = -1 for all LEDs. For programs - no text parsing."""
            data = [int(v) for v in msg.data]
            if len(data) != 9:
                self.get_logger().error(
                    "pixel_raw: expected [led, mode, r, g, b, w, brightness, "
                    f"period_ms, count], got {len(data)} values")
                return
            led, mode, r, g, b, w, bri, period, count = data
            if not 0 <= mode < len(PIXEL_MODES):
                self.get_logger().error(f"pixel_raw: unknown mode {mode}")
                return
            self.link.pixel(PixelCommand(None if led < 0 else led, mode,
                                         r, g, b, w, bri, period, count))

        def _on_cal(self, msg: Int32MultiArray) -> None:
            """[action, arg] - action codes see CAL_ACTIONS."""
            data = list(msg.data) + [0, 0]
            self._run_cal(int(data[0]), int(data[1]))

        def _run_cal(self, action: int, arg: int) -> None:
            state = self.link.calibrate(action, arg)
            if state is None:
                self.get_logger().error("Calibration: no reply from ESP")
                return
            if state.status != 0x00:
                self.get_logger().warn(
                    f"Calibration: {CAL_STATUS_TEXT.get(state.status, '?')}")
            self.get_logger().info(
                f"cal: pos={state.pos} centre={state.center} left={state.left} "
                f"right={state.right} "
                f"set={'C' if state.have_center else '-'}"
                f"{'L' if state.have_left else '-'}"
                f"{'R' if state.have_right else '-'}"
                f"{' free' if state.torque_free else ''}")

        # --- Services -----------------------------------------------------

        @staticmethod
        def _reply(response, ok: bool, message: str):
            response.success = ok
            response.message = message
            return response

        def _srv_emergency(self, _req, res):
            self.link.emergency()
            return self._reply(res, True, "Emergency halt triggered")

        def _srv_move_abort(self, _req, res):
            self.link.move_abort()
            return self._reply(res, True, "Move aborted")

        def _srv_pid_get(self, _req, res):
            values = self.link.pid_get()
            if values is None:
                return self._reply(res, False, "no reply from ESP")
            return self._reply(res, True,
                               "kp={:.3f} ki={:.3f} kd={:.3f}".format(*values))

        def _srv_pid_save(self, _req, res):
            ok = self.link.pid_save()
            if ok is None:
                return self._reply(res, False, "no reply from ESP")
            return self._reply(res, ok,
                               "written to NVS" if ok else "NVS error")

        def _srv_cal_start(self, _req, res):
            state = self.link.calibrate("start")
            if state is None:
                return self._reply(res, False, "no reply from ESP")
            return self._reply(res, True,
                               "calibration mode running - now use ~/cal")

        def _srv_cal_save(self, _req, res):
            state = self.link.calibrate("save")
            if state is None:
                return self._reply(res, False, "no reply from ESP")
            ok = state.status == 0x01
            return self._reply(res, ok, CAL_STATUS_TEXT.get(state.status, "?"))

        def _srv_trim_save(self, _req, res):
            self.link.trim(2)
            return self._reply(res, True, "Trim offset saved")

        def _srv_torque(self, _req, res):
            self.link.torque_report()
            return self._reply(res, True, "Output on the ESP's USB console")

        def _srv_resync(self, _req, res):
            self.link.resync()
            time.sleep(0.1)
            if not self.link.clock.valid:
                return self._reply(res, False, "no reply to TIME_SYNC")
            return self._reply(res, True,
                               f"Offset {self.link.clock.offset * 1e3:+.3f} ms, "
                               f"Drift {self.link.clock.drift_ppm:+.1f} ppm")

        def _srv_led(self, req, res):
            self.link.led(req.data)
            return self._reply(res, True, "LED " + ("on" if req.data else "off"))

        def _srv_stamp(self, req, res):
            state = self.link.set_stamp_mode(req.data)
            if state is None:
                return self._reply(res, False, "no reply from ESP")
            return self._reply(res, True,
                               "Timestamps " + ("on" if state else "off"))

        def _srv_torque_free(self, req, res):
            """Switch the servo torque off to move the steering by hand.

            Only works while calibration is running - outside of it the ESP
            rejects with status 0x04. Call ~/calibrate_start first.
            """
            state = self.link.calibrate("free" if req.data else "hold")
            if state is None:
                return self._reply(res, False, "no reply from ESP")
            if state.status == 0x04:
                return self._reply(res, False,
                                   "only in calibration mode - ~/calibrate_start first")
            return self._reply(res, True,
                               "Servo " + ("released" if req.data else "holding"))

        # --- Telemetry rate ----------------------------------------------

        def _apply_telemetry_period(self, period: float) -> None:
            actual = self.link.set_telemetry_rate(period)
            if actual <= 0:
                self.get_logger().info("Drive telemetry off")
                return
            if abs(actual - period) > 1e-6:
                self.get_logger().warn(
                    f"Telemetry period limited to {actual * 1e3:.0f} ms "
                    f"(requested {period * 1e3:.0f} ms, minimum "
                    f"{TELEMETRY_MS_MIN} ms)")
            self.get_logger().info(
                f"Drive telemetry every {actual * 1e3:.0f} ms "
                f"({1.0 / actual:.1f} Hz) - the ESP updates the speed at 10 Hz")

        def _on_set_parameters(self, params):
            """Pass on ros2 param set ... telemetry_period 0.1."""
            from rcl_interfaces.msg import SetParametersResult

            for param in params:
                if param.name == "telemetry_period":
                    try:
                        self._apply_telemetry_period(float(param.value))
                    except Exception as exc:
                        return SetParametersResult(successful=False,
                                                   reason=str(exc))
            return SetParametersResult(successful=True)

        # --- Timer --------------------------------------------------------

        def _heartbeat(self) -> None:
            """Keep the motor alive and stop it when /cmd_vel stays away.

            During a position move it does NOT stop: tick() holds back the
            motor command there anyway, because the ESP controls the move
            to the end itself -- a motor_coast() from here would abort
            it."""
            with self._motor_lock:
                if (self._cmd_vel_timeout > 0 and self._last_cmd_vel
                        and not self.link.move_active
                        and monotonic() - self._last_cmd_vel > self._cmd_vel_timeout):
                    self._last_cmd_vel = 0.0
                    self.link.motor_coast()
                    self.get_logger().warn(
                        f"no /cmd_vel for {self._cmd_vel_timeout:.1f} s - motor off")
                # tick() resends the last motor command and checks move_active
                # itself while doing so -- the same check-then-send gap,
                # so it belongs under the same lock.
                self.link.tick(float(self._p("heartbeat_period")))

        def _resync(self) -> None:
            self.link.resync()

        def _poll_progress(self) -> None:
            if self.link.move_active:
                self.link.request_progress()

        def _publish_link_status(self) -> None:
            clock = self.link.clock
            status = DiagnosticStatus()
            status.name = "esp_serial_bridge: link"
            status.hardware_id = str(self._p("port"))

            if not clock.valid:
                status.level = DiagnosticStatus.WARN
                status.message = "Clocks not synced"
            elif clock.rejected:
                status.level = DiagnosticStatus.WARN
                status.message = (f"{clock.rejected} of {len(clock.samples)} "
                                  "sync rounds unusable")
            elif clock.best_rtt > 0.05:
                status.level = DiagnosticStatus.WARN
                status.message = f"Round trip {clock.best_rtt * 1e3:.1f} ms - link slow"
            else:
                status.level = DiagnosticStatus.OK
                status.message = "all good"

            def kv(key, value):
                return KeyValue(key=key, value=str(value))

            status.values = [
                kv("offset_ms", f"{clock.offset * 1e3:+.3f}" if clock.valid else "-"),
                kv("drift_ppm", f"{clock.drift_ppm:+.1f}" if clock.valid else "-"),
                kv("rtt_ms", f"{clock.best_rtt * 1e3:.3f}" if clock.valid else "-"),
                kv("timestamps", "on" if self.link.stamp_mode else "off"),
                kv("telemetry_ms", int(float(self._p("telemetry_period")) * 1000)),
                kv("esp_reboots", clock.boot_count),
                kv("packets_rx", self.link.rx_frames),
                kv("packets_tx", self.link.tx_frames),
                kv("sync_lost", self.link.sync.lost),
                kv("sync_rejected", clock.rejected),
                kv("unknown_packets", self.link.parser.unknown),
            ]

            msg = DiagnosticArray()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.status = [status]
            self.pub_status.publish(msg)

            if clock.valid:
                self.pub_rtt.publish(Float32(data=clock.best_rtt * 1e3))
                self.pub_offset.publish(Float64(data=clock.offset * 1e3))
                self.pub_drift.publish(Float32(data=float(clock.drift_ppm)))

        # --- Shutdown -----------------------------------------------------

        def destroy_node(self) -> bool:
            try:
                self.link.close()
            except Exception:
                pass
            return super().destroy_node()

    return rclpy, EspBridgeNode


def main(args=None) -> None:
    rclpy, node_class = _build_node_class()
    from rclpy.executors import MultiThreadedExecutor

    rclpy.init(args=args)
    node = node_class()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


# ==========================================================================
# Self-test - protocol without ROS and without hardware
# ==========================================================================

def _selftest() -> int:
    failures = 0

    def check(name: str, ok: bool, detail: str = "") -> None:
        nonlocal failures
        print(f"  [{'ok ' if ok else 'FAIL'}] {name}{'  ' + detail if detail else ''}")
        if not ok:
            failures += 1

    class _FakeLink(EspLink):
        """EspLink without a serial port - writes into a list."""

        def __init__(self):
            self.sent: List[bytes] = []
            self._write_lock = threading.Lock()
            self._parser = FrameParser()
            self.clock = EspClock()
            self.sync = TimeSync(self._send_timed, self.clock)
            self._callbacks, self._waiters, self._last_payload = {}, {}, {}
            self._motor_frame, self._last_motor_tx = None, 0.0
            self._move_active, self._next_move_id = False, 1
            self.servo_id, self.stamp_mode = 1, False
            self.console_lines = deque(maxlen=200)
            self.rx_frames = self.tx_frames = 0
            self._log = lambda msg: None

        def _send_timed(self, frame: bytes) -> float:
            self.sent.append(frame)
            self.tx_frames += 1
            return monotonic()

    print("Encoding commands")
    link = _FakeLink()

    link.motor(700)
    check("Motor forward", link.sent[-1] == bytes([0xA5, 0x10, 0x00, 0x02, 0xBC]),
          link.sent[-1].hex(" "))
    link.motor(-700)
    check("Motor reverse", link.sent[-1] == bytes([0xA5, 0x10, 0x01, 0x02, 0xBC]))
    link.motor(9999)
    check("Motor clamped", link.sent[-1] == bytes([0xA5, 0x10, 0x00, 0x03, 0xFF]))

    link.steer(-100)
    check("Steering right", link.sent[-1] == bytes([0xA5, 0x20, 0x01, 0xFF, 0x9C]),
          link.sent[-1].hex(" "))
    link.steer(250)
    check("Steering clamped", link.sent[-1] == bytes([0xA5, 0x20, 0x01, 0x00, 0x64]))

    move_id = link.move(90.0)
    check("Move 90 deg",
          link.sent[-1] == bytes([0xA5, 0x90, move_id, 0x00, 0x00, 0x03, 0x84]),
          link.sent[-1].hex(" "))
    link.move(-45.0)
    check("Move negative",
          link.sent[-1][3:] == struct.pack(">i", -450), link.sent[-1].hex(" "))
    check("move_id counts up", link.sent[-1][2] == move_id % 255 + 1)

    # Order in move(): the lock must be set BEFORE the frame goes out.
    # The other way round, the speed controller in the other thread still
    # sees "no move" and sends duty 0 right after -- the ESP then acks the
    # move it just started as aborted. Measured on the robot: 14 ms.
    seen = {}
    real_send = link.send

    def _spy(cmd, payload=b""):
        seen[cmd] = link.move_active
        return real_send(cmd, payload)

    link.send = _spy
    link._move_active = False
    link.move(90.0)
    link.send = real_send
    check("Lock already set while sending", seen.get(CMD_MOVE) is True)

    # The most common mistake according to the spec: the x1000 encoding also
    # applies to the integer parameters. maxDuty=700 must go out as 700000.
    link.pid_set("maxduty", 700)
    check("PID maxDuty x1000",
          link.sent[-1] == bytes([0xA5, 0x80, 0x04]) + struct.pack(">i", 700000),
          link.sent[-1].hex(" "))
    link.pid_set("kp", 4.25)
    check("PID kp x1000",
          link.sent[-1] == bytes([0xA5, 0x80, 0x00]) + struct.pack(">i", 4250))

    link.emergency()
    check("Emergency halt", link.sent[-1] == bytes([0xA5, 0xFF]))
    link.led(True)
    check("LED", link.sent[-1] == bytes([0xA5, 0x30, 0x01]))
    link.trim(2)
    check("Trim save", link.sent[-1] == bytes([0xA5, 0x60, 0x02]))

    # The example packets from the spec, byte for byte.
    for cmd in parse_pixel("strobe red bri=255 ms=300 x=3"):
        link.pixel(cmd)
    check("PIXEL all, one-shot",
          link.sent[-1] == bytes([0xA5, 0x31, 0x05, 0xFF, 0x00, 0x00, 0x00,
                                  0xFF, 0x01, 0x2C, 0x03]),
          link.sent[-1].hex(" "))
    for cmd in parse_pixel("2: strobe red bri=255 ms=300 x=3"):
        link.pixel(cmd)
    check("PIXEL_ONE LED 2",
          link.sent[-1] == bytes([0xA5, 0x32, 0x02, 0x05, 0xFF, 0x00, 0x00, 0x00,
                                  0xFF, 0x01, 0x2C, 0x03]),
          link.sent[-1].hex(" "))
    cmds = parse_pixel("0: red; 1: #00ff0010; 3: breathe rgbw=1,2,3", brightness=40)
    check("PIXEL several, hex with W, rgbw=",
          [(c.led, c.mode, c.r, c.g, c.b, c.w, c.brightness) for c in cmds]
          == [(0, 1, 255, 0, 0, 0, 40), (1, 1, 0, 255, 0, 16, 40),
              (3, 3, 1, 2, 3, 0, 40)], f"{cmds}")
    only_mode = parse_pixel("rainbow")[0]
    check("PIXEL mode alone -> all LEDs, white",
          (only_mode.led, only_mode.mode, only_mode.w) == (None, 4, 255))
    for bad in ("blurple", "x: red", "red bri=300", "foo=1", ""):
        try:
            parse_pixel(bad)
            check(f"PIXEL rejects '{bad}'", False)
        except ValueError:
            check(f"PIXEL rejects '{bad}'", True)

    actual = link.set_telemetry_rate(0.05)
    check("Telemetry period 50 ms",
          link.sent[-1] == bytes([0xA5, 0xC0, 0x00, 0x32]) and actual == 0.05,
          link.sent[-1].hex(" "))
    actual = link.set_telemetry_rate(0.001)
    check("Period clamped to minimum",
          link.sent[-1] == bytes([0xA5, 0xC0, 0x00, TELEMETRY_MS_MIN])
          and actual == TELEMETRY_MS_MIN / 1000.0,
          f"{actual * 1e3:.0f} ms")
    actual = link.set_telemetry_rate(0)
    check("Telemetry can be switched off",
          link.sent[-1] == bytes([0xA5, 0xC0, 0x00, 0x00]) and actual == 0.0)

    print("Unpacking replies")
    done = parse_move_done(bytes([3, 0]) + struct.pack(">i", 905))
    check("MOVE_DONE", (done.move_id, done.ok, done.position_deg) == (3, True, 90.5),
          f"{done}")
    prog = parse_progress(bytes([3, 1, 42]) + struct.pack(">ii", 450, 905))
    check("PROGRESS", (prog.percent, prog.target_deg) == (42, 90.5))
    telemetry = parse_telemetry(struct.pack(">ii", 905, 1800)
                                + struct.pack(">hh", -700, 2500))
    check("TELEMETRY",
          (telemetry.position_deg, telemetry.speed_deg_s, telemetry.duty,
           telemetry.current_a) == (90.5, 180.0, -700, 2.5), f"{telemetry}")
    check("TELEMETRY in rad/s", abs(telemetry.speed_rad_s - math.pi) < 1e-9,
          f"{telemetry.speed_rad_s:.6f}")

    # Reverse drive. The bytes are worked out by hand from the two's
    # complement, not made with struct.pack - otherwise the test would only
    # check that Python agrees with itself, and a sign error on the ESP
    # side would go unnoticed.
    #   -905 = 0xFFFFFC77   -1800 = 0xFFFFF8F8   -700 = 0xFD44
    backwards = bytes([0xFF, 0xFF, 0xFC, 0x77,
                         0xFF, 0xFF, 0xF8, 0xF8,
                         0xFD, 0x44,
                         0x09, 0xC4])
    back = parse_telemetry(backwards)
    check("TELEMETRY negative",
          (back.position_deg, back.speed_deg_s, back.duty, back.current_a)
          == (-90.5, -180.0, -700, 2.5), f"{back}")
    check("TELEMETRY negative in rad/s", abs(back.speed_rad_s + math.pi) < 1e-9,
          f"{back.speed_rad_s:.6f}")
    check("Current stays positive", back.current_a > 0)

    batt = parse_battery(struct.pack(">i", 15200) + struct.pack(">h", 3800), True)
    check("BATTERY", (batt.pack_v, batt.cell_v, batt.warning) == (15.2, 3.8, True))
    cal = parse_cal_state(bytes([1, 0x0B, 0]) + struct.pack(">hhhh", 512, 500, 800, 200))
    check("CAL_RSP", cal.active and cal.have_center and cal.have_left
          and not cal.have_right and cal.torque_free and cal.center == 500)

    print("Heartbeat")
    link.sent.clear()
    link.motor(500)
    link._last_motor_tx = monotonic() - 1.0
    link.tick(0.2)
    check("resends", len(link.sent) == 2)
    link.tick(0.2)
    check("not too often", len(link.sent) == 2)
    link._move_active = True
    link._last_motor_tx = monotonic() - 1.0
    link.tick(0.2)
    check("pauses during the move", len(link.sent) == 2)

    print("Receiving and timestamps")
    link._move_active = True
    got: List[Frame] = []
    link.on(CMD_MOVE_DONE, got.append)
    link._dispatch(Frame(cmd=CMD_MOVE_DONE,
                         payload=bytes([1, 0]) + struct.pack(">i", 900),
                         esp_tx_raw=None, rx_mono=monotonic()))
    check("Callback called", len(got) == 1)
    check("Move finished", link._move_active is False)
    check("no stamp without sync", link.sent_at(got[0]) is None)

    print()
    print("Self-test failed" if failures else "Self-test passed")
    return 1 if failures else 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--selftest", action="store_true",
                        help="check the protocol without ROS and without hardware")
    known, rest = parser.parse_known_args()
    if known.selftest:
        raise SystemExit(_selftest())
    main()
