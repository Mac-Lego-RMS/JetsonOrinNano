#!/usr/bin/env python3
"""ROS-2-Bridge zum ESP32-S3-Controller.

Setzt das UART-Protokoll aus docs/JETSON_BRIDGE.md vollstaendig auf ROS 2 um -
jede Funktion des ESP ist ueber ein Topic oder einen Service erreichbar - und
versieht jede Meldung des ESP mit dem Zeitpunkt, zu dem sie *abgeschickt*
wurde, nicht mit dem, zu dem der Jetson sie zufaellig gelesen hat. Der
Uhrenabgleich dafuer steckt in ``timesync_jetson.py`` (Abschnitt 5 der Spec).

Aufbau
------
``EspLink``        Protokoll, Timesync, Heartbeat, Lese-Thread. Kennt kein ROS
                   und laeuft ohne rclpy - damit ohne Roboter testbar.
``EspBridgeNode``  rclpy-Node. Nur Verdrahtung: Topics und Services auf die
                   Methoden von ``EspLink``, Pakete des ESP auf Publisher.

Start
-----
    ros2 run <paket> esp_serial_bridge --ros-args -p port:=/dev/ttyTHS1
    python3 esp_serial_bridge.py --selftest      # ohne ROS, ohne Hardware
"""

from __future__ import annotations

import argparse
import importlib.util
import math
import struct
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Deque, Dict, List, Optional, Tuple

# ==========================================================================
# timesync_jetson.py einbinden
# ==========================================================================
# Im Repo liegt die Datei unter docs/, in einem ROS-Paket gehoert sie neben
# diese hier. Beide Faelle abdecken, statt den Anwender an PYTHONPATH
# scheitern zu lassen.

#: Was aus timesync_jetson gebraucht wird. Wird nach jedem Importversuch
#: geprueft - ein leerer Platzhalter oder eine fremde Datei gleichen Namens
#: soll nicht durchrutschen und erst zehn Zeilen spaeter als nichtssagender
#: AttributeError auffallen.
_TIMESYNC_NAMES = ("Frame", "FrameParser", "EspClock", "TimeSync",
                   "monotonic", "START_BYTE")


def _timesync_missing(module) -> List[str]:
    return [name for name in _TIMESYNC_NAMES if not hasattr(module, name)]


def _load_timesync():
    tried: List[str] = []

    def accept(module, quelle: str):
        missing = _timesync_missing(module)
        if not missing:
            return module
        tried.append(f"{quelle}: unvollstaendig, es fehlen {', '.join(missing)}"
                     f" (Datei: {getattr(module, '__file__', '?')})")
        return None

    # 1. Als Teil desselben Pakets - der Normalfall in einem ROS-Paket, wo
    #    beide Dateien nebeneinander installiert sind.
    if __package__:
        name = f"{__package__}.timesync_jetson"
        try:
            found = accept(importlib.import_module(name), name)
            if found:
                return found
        except ImportError as exc:
            tried.append(f"{name}: {exc}")

    # 2. Irgendwo auf dem Suchpfad.
    try:
        import timesync_jetson as module
        found = accept(module, "timesync_jetson (sys.path)")
        if found:
            return found
    except ImportError as exc:
        tried.append(f"timesync_jetson (sys.path): {exc}")

    # 3. Direkt als Datei laden.
    here = Path(__file__).resolve().parent
    for path in (here / "timesync_jetson.py",
                 here.parent / "timesync_jetson.py",
                 here.parent / "docs" / "timesync_jetson.py",
                 here / "docs" / "timesync_jetson.py"):
        if not path.exists():
            tried.append(f"{path}: nicht vorhanden")
            continue
        try:
            spec = importlib.util.spec_from_file_location("timesync_jetson", path)
            module = importlib.util.module_from_spec(spec)
            # Muss vor exec_module stehen: @dataclass schlaegt die Globals
            # ihrer Klasse ueber sys.modules[cls.__module__] nach und
            # scheitert sonst mit "'NoneType' object has no attribute
            # '__dict__'". So steht es auch in der Import-Doku.
            sys.modules["timesync_jetson"] = module
            spec.loader.exec_module(module)
        except Exception as exc:
            sys.modules.pop("timesync_jetson", None)
            tried.append(f"{path}: {exc}")
            continue
        found = accept(module, str(path))
        if found:
            return found
        # Kaputtes Modul den Namen nicht belegen lassen.
        sys.modules.pop("timesync_jetson", None)

    raise ImportError(
        "timesync_jetson.py fehlt oder ist unvollstaendig.\n"
        "Die Datei gehoert neben diese hier - in einem ROS-Paket also in\n"
        "denselben Ordner wie esp_serial_bridge.py, danach neu bauen.\n"
        "Steht dort eine leere Platzhalterdatei, ueberschreibt sie den\n"
        "richtigen Fund.\nVersucht wurde:\n  " + "\n  ".join(tried))


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
# Protokoll
# ==========================================================================

# --- Jetson -> ESP ---
CMD_MOTOR = 0x10
CMD_SERVO = 0x20
CMD_LED = 0x30
CMD_CALIBRATE = 0x40        # identisch zu CMD_CAL mit Aktion "start"
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
CMD_TIME_SYNC = 0xB0        # wird in timesync_jetson.py gesendet
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
CMD_TIME_RSP = 0xB1         # wird in timesync_jetson.py ausgewertet
CMD_STAMP_RSP = 0xB3
CMD_TELEMETRY = 0xC1

DUTY_MAX = 1023
TELEMETRY_MS_MIN = 10

MOVE_OK, MOVE_TIMEOUT, MOVE_ABORTED = 0x00, 0x01, 0x02
MOVE_STATUS_TEXT = {MOVE_OK: "ok", MOVE_TIMEOUT: "timeout", MOVE_ABORTED: "abgebrochen"}

# PID-Parameter: Index auf der Leitung -> Name. Alle Werte gehen als
# int32 x 1000 raus, auch die ganzzahligen (haeufigste Fehlerquelle, s. Spec).
PID_PARAMS = ["kp", "ki", "kd", "ilimit", "maxduty", "tol_deg", "settle_ms",
              "timeout_ms", "minduty"]

# Kalibrier-Aktionen in CMD_CAL
CAL_ACTIONS = {
    "start": 0x00, "minus": 0x01, "plus": 0x02, "center": 0x03,
    "left": 0x04, "right": 0x05, "save": 0x06, "abort": 0x07,
    "free": 0x08, "hold": 0x09, "goto_center": 0x0A, "step": 0x0B,
    "status": 0x0C,
}

# Der ESP rechnet Wege in 1/10 Grad der Ausgangswelle, ROS in Radiant.
DEG_TO_RAD = math.pi / 180.0


def _clamp(value: float, low: float, high: float) -> float:
    return low if value < low else high if value > high else value


# ==========================================================================
# Ergebnis-Datentypen
# ==========================================================================

@dataclass
class MoveDone:
    move_id: int
    status: int
    position_deg: float
    stamp: Optional[float]        # Sendezeitpunkt (Jetson-Uhr), falls gestempelt

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
    """Fahrzustand, den der ESP im eingestellten Takt von sich aus schickt."""

    #: Stellung der Ausgangswelle, absolut seit ESP-Boot. Vorzeichenbehaftet.
    position_deg: float
    #: Drehgeschwindigkeit. **Vorzeichenbehaftet**, negativ = rueckwaerts.
    speed_deg_s: float
    #: Was an der Bruecke anliegt, -1023..+1023. Vorzeichenbehaftet.
    duty: int
    #: Motorstrom - **immer positiv**. Der VNH5019 meldet nur den Betrag, die
    #: Richtung steht in ``duty``.
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
    0x00: "ausgefuehrt", 0x01: "gespeichert", 0x02: "abgelehnt",
    0x03: "servo antwortet nicht", 0x04: "erst 0x40 senden",
    0x05: "bereichsende erreicht",
}


# ==========================================================================
# EspLink - Protokoll ohne ROS
# ==========================================================================

class EspLink:
    """Serielle Verbindung zum ESP: senden, empfangen, Uhren abgleichen.

    Der Lese-Thread nimmt alles entgegen, was der ESP von sich aus schickt,
    fuettert den Zeitabgleich und ruft die angemeldeten Callbacks. Fuer
    Antworten, auf die jemand wartet (PID, Kalibrierung), gibt es zusaetzlich
    ``wait_for``.

    **Der Heartbeat ist Pflicht.** Der ESP laesst den Motor auslaufen, wenn
    5 s lang kein Befehl kommt. ``tick()`` schickt deshalb den letzten
    Motorbefehl zyklisch nach - waehrend einer Positionsfahrt bewusst nicht,
    die darf laenger laufen.
    """

    def __init__(self, port: str, baud: int = 115200, servo_id: int = 1,
                 log: Optional[Callable[[str], None]] = None) -> None:
        import serial          # nur hier, damit der Selbsttest ohne auskommt

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

        # Motor-Heartbeat
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

    # --- Lebenszyklus ----------------------------------------------------

    def start(self, sync_rounds: int = 12, stamp: bool = True) -> None:
        """Lese-Thread starten, Uhren abgleichen, Stempel einschalten.

        Reihenfolge ist wichtig: ohne Uhrenversatz ist ein Zeitstempel wertlos,
        also erst messen, dann stempeln lassen.
        """
        self._reader.start()

        for _ in range(sync_rounds):
            self.sync.request()
            time.sleep(0.02)
        time.sleep(0.05)

        if self.clock.valid:
            self._log(f"Uhren abgeglichen: Versatz {self.clock.offset * 1e3:+.3f} ms, "
                      f"Umlauf {self.clock.best_rtt * 1e3:.3f} ms, "
                      f"Drift {self.clock.drift_ppm:+.1f} ppm")
        else:
            self._log("WARNUNG: keine Antwort auf TIME_SYNC - Zeitstempel bleiben leer")

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

    # --- Senden ----------------------------------------------------------

    def _send_timed(self, frame: bytes) -> float:
        """Rahmen rausschicken und zurueckgeben, wann das letzte Byte draussen
        war.

        Der Zeitpunkt wird **gerechnet, nicht gemessen**: Uhr vor dem Schreiben
        plus Uebertragungsdauer (10 Bit je Byte). ``flush()`` danach abzufragen
        waere naheliegend, taugt aber nicht - ``tcdrain()`` kehrt auf dem
        Tegra-UART des Jetson deutlich spaeter zurueck als das letzte Byte
        rausgeht, und dann wird der gemessene Umlauf negativ.

        Das ``flush()`` davor bleibt: es raeumt den Puffer, damit unser Rahmen
        sofort losgeht und die Rechnung stimmt.
        """
        with self._write_lock:
            self._ser.flush()
            started = monotonic()
            self._ser.write(frame)
            self.tx_frames += 1
        return started + len(frame) * BITS_PER_BYTE / self._ser.baudrate

    def send(self, cmd: int, payload: bytes = b"") -> None:
        """Ein Paket abschicken. Immer in einem einzigen ``write()`` - der
        ESP verwirft ein Paket, wenn zwischen zwei Byte >100 ms liegen."""
        self._send_timed(bytes([START_BYTE, cmd]) + payload)

    # --- Fahrbefehle -----------------------------------------------------

    def motor(self, duty: int) -> None:
        """Offene Motorsteuerung, -1023..+1023. 0 = auslaufen lassen.

        Aktives Bremsen gibt es nur ueber ``emergency()``.
        """
        duty = int(_clamp(duty, -DUTY_MAX, DUTY_MAX))
        reverse = 1 if duty < 0 else 0
        speed = abs(duty)
        self._motor_frame = bytes([START_BYTE, CMD_MOTOR, reverse]) + \
            struct.pack(">H", speed)
        self._send_timed(self._motor_frame)
        self._last_motor_tx = monotonic()
        # Jeder Motorbefehl loest eine laufende Positionsfahrt ab - auch
        # duty 0. Der ESP quittiert die alte Fahrt dann mit "abgebrochen".
        self._move_active = False

    def motor_coast(self) -> None:
        self.motor(0)

    def steer(self, percent: float) -> None:
        """Lenkung, -100 (rechts) .. +100 (links). 0 = geradeaus."""
        pct = int(round(_clamp(percent, -100, 100)))
        self.send(CMD_SERVO, bytes([self.servo_id]) + struct.pack(">h", pct))

    def emergency(self) -> None:
        """Nothalt mit aktiver Bremse. Bricht auch eine Positionsfahrt ab."""
        self._motor_frame = None
        self._move_active = False
        self.send(CMD_EMERGENCY)

    def led(self, on: bool) -> None:
        self.send(CMD_LED, bytes([1 if on else 0]))

    def trim(self, action: int) -> None:
        """0 = Mitte nach links, 1 = nach rechts, 2 = speichern."""
        self.send(CMD_TRIM, bytes([action & 0xFF]))

    def torque_report(self) -> None:
        """Servolast auf die USB-Konsole des ESP ausgeben. Ohne UART-Antwort."""
        self.send(CMD_TORQUE)

    # --- Positionsfahrt --------------------------------------------------

    def move(self, degrees: float) -> int:
        """Um ``degrees`` weiterdrehen (relativ!). Gibt die move_id zurueck.

        Es laeuft immer nur eine Fahrt; eine neue loest die alte ab, und die
        alte quittiert mit Status "abgebrochen".
        """
        move_id = self._next_move_id
        self._next_move_id = self._next_move_id % 255 + 1   # 0 vermeiden
        deg10 = int(round(degrees * 10.0))
        self.send(CMD_MOVE, bytes([move_id]) + struct.pack(">i", deg10))
        self._move_active = True
        return move_id

    def move_abort(self) -> None:
        self.send(CMD_MOVE_ABORT)

    def request_progress(self) -> None:
        self.send(CMD_PROGRESS)

    # --- PID -------------------------------------------------------------

    def pid_set(self, param: int | str, value: float) -> None:
        """Einen Regelparameter setzen. Nur fluechtig - ``pid_save()`` schreibt
        den ganzen Satz ins NVS."""
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

    # --- Lenkungs-Kalibrierung -------------------------------------------

    def calibrate(self, action: int | str = "start", arg: int = 0,
                  timeout: float = 1.5) -> Optional[CalState]:
        """Eine Kalibrieraktion ausfuehren. Jede wird mit CAL_RSP beantwortet.

        Der Servo kann sein Drehmoment nicht begrenzen - es gibt deshalb
        bewusst keine Aktion, die selbsttaetig bis zum Anschlag faehrt.
        """
        code = CAL_ACTIONS[action] if isinstance(action, str) else int(action)
        payload = self.request(CMD_CAL, CMD_CAL_RSP, timeout,
                               bytes([code, arg & 0xFF]))
        return None if payload is None else parse_cal_state(payload)

    # --- Batterie und Zeit -----------------------------------------------

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
        """Takt der Fahrtelemetrie setzen. 0 schaltet sie ab.

        Gibt den tatsaechlich gesetzten Takt in Sekunden zurueck - der ESP
        nimmt nichts unter 20 ms an, weil er die Geschwindigkeit ohnehin nur
        alle 100 ms neu bildet. Die Position ist bei jedem Paket frisch.
        """
        ms = 0 if period_s <= 0 else max(TELEMETRY_MS_MIN,
                                         min(60000, int(round(period_s * 1000))))
        self.send(CMD_TELEM_RATE, struct.pack(">H", ms))
        return ms / 1000.0

    def resync(self) -> None:
        """Eine Runde Uhrenabgleich. Muss regelmaessig passieren, sonst laeuft
        der Quarzdrift weg (~0,1 ms pro Sekunde)."""
        self.sync.request()

    # --- Heartbeat -------------------------------------------------------

    def tick(self, heartbeat_period: float = 0.2) -> None:
        """Regelmaessig aufrufen. Schickt den letzten Motorbefehl nach, damit
        der 5-s-Watchdog des ESP nicht zuschlaegt."""
        if self._motor_frame is None or self._move_active:
            return
        if monotonic() - self._last_motor_tx >= heartbeat_period:
            self._send_timed(self._motor_frame)
            self._last_motor_tx = monotonic()

    # --- Empfang ---------------------------------------------------------

    def on(self, cmd: int, callback: Callable[[Frame], None]) -> None:
        """Callback fuer einen Pakettyp anmelden. Laeuft im Lese-Thread."""
        self._callbacks.setdefault(cmd, []).append(callback)

    def request(self, cmd: int, answer: int, timeout: float,
                payload: bytes = b"") -> Optional[bytes]:
        """Befehl senden und auf die passende Antwort warten.

        Antworten kommen im Lese-Thread an, der Aufrufer darf also blockieren.
        """
        event = threading.Event()
        self._waiters.setdefault(answer, []).append(event)
        self.send(cmd, payload)
        if not event.wait(timeout):
            try:
                self._waiters[answer].remove(event)
            except (KeyError, ValueError):
                pass          # der Lese-Thread war schneller
            self._log(f"Timeout: keine Antwort 0x{answer:02X} auf 0x{cmd:02X}")
            return None
        return self._last_payload.get(answer)

    def _read_loop(self) -> None:
        while not self._stop.is_set():
            try:
                # Nicht read(64): das wartet, bis 64 Byte beisammen sind, und
                # dann tragen alle Pakete darin denselben Empfangszeitpunkt.
                data = read_available(self._ser)
            except Exception as exc:            # Port weg (USB gezogen o.ae.)
                self._log(f"Lesefehler: {exc}")
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

        # Zeitabgleich zuerst: die Antwort darf nirgends warten.
        if self.sync.handle(frame) is not None:
            return

        if frame.cmd in (CMD_MOVE_DONE, CMD_PROGRESS_RSP):
            if frame.cmd == CMD_MOVE_DONE:
                self._move_active = False
            elif frame.payload[1] == 0:
                self._move_active = False

        # Wartende Aufrufer wecken
        waiters = self._waiters.pop(frame.cmd, None)
        if waiters:
            self._last_payload[frame.cmd] = frame.payload
            for event in waiters:
                event.set()

        for callback in self._callbacks.get(frame.cmd, ()):
            try:
                callback(frame)
            except Exception as exc:
                self._log(f"Callback fuer 0x{frame.cmd:02X} fehlgeschlagen: {exc}")

    # --- Zustand ---------------------------------------------------------

    @property
    def move_active(self) -> bool:
        """Laeuft gerade eine Positionsfahrt? Solange sie laeuft, pausiert der
        Heartbeat - Fahrten duerfen laenger als die 5-s-Grenze dauern."""
        return self._move_active

    @property
    def parser(self) -> FrameParser:
        return self._parser

    # --- Zeitstempel -----------------------------------------------------

    def sent_at(self, frame: Frame) -> Optional[float]:
        """Wann das Paket abgeschickt wurde, in der Jetson-Uhr."""
        return self.clock.frame_time(frame)

    def latency(self, frame: Frame) -> Optional[float]:
        """Wie lange es von "abgeschickt" bis "gelesen" gebraucht hat."""
        return self.clock.latency(frame)


# ==========================================================================
# Nutzlast auspacken
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
# ROS-2-Node
# ==========================================================================

def _build_node_class():
    """Node-Klasse erst bauen, wenn rclpy da ist - so laeuft der Selbsttest
    auch auf einem Rechner ohne ROS."""

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

    class EspBridgeNode(Node):
        """Alle Funktionen des ESP als Topics und Services.

        Aufteilung: was einen Strom bildet (Fahrbefehle, Telemetrie) ist ein
        Topic, was eine Quittung hat (PID speichern, kalibrieren) ist ein
        Service. Alle Nachrichten mit Header tragen als ``stamp`` den
        **Sendezeitpunkt des ESP**, umgerechnet in die ROS-Uhr.
        """

        def __init__(self) -> None:
            super().__init__("esp_serial_bridge")
            group = ReentrantCallbackGroup()

            # --- Parameter ---
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
            # Takt, in dem der ESP Position und Geschwindigkeit von sich aus
            # schickt. Zur Laufzeit aenderbar:
            #   ros2 param set /esp_serial_bridge telemetry_period 0.1
            self.declare_parameter("telemetry_period", 0.01)  # 0 = aus
            # cmd_vel ist offene Steuerung: der ESP regelt die Drehzahl nicht.
            # Diese beiden Werte sind die Umrechnung und muessen am Fahrzeug
            # ausgemessen werden.
            self.declare_parameter("max_linear", 1.0)     # m/s bei Volldampf
            self.declare_parameter("max_angular", 1.0)    # rad/s bei Vollausschlag

            self._p = lambda name: self.get_parameter(name).value
            self._cmd_vel_timeout = float(self._p("cmd_vel_timeout"))
            self._last_cmd_vel = 0.0

            # --- Verbindung ---
            port = self._p("port")
            self.get_logger().info(f"oeffne {port} @ {self._p('baud')} Baud")
            self.link = EspLink(port, int(self._p("baud")),
                                int(self._p("servo_id")),
                                log=self.get_logger().info)

            # Pakete des ESP kommen im Lese-Thread an; von dort nur in eine
            # Queue, veroeffentlicht wird auf dem Executor-Thread.
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

            # Erst nach dem Uhrenabgleich einschalten - sonst kaemen die ersten
            # Telemetriepakete ohne brauchbaren Zeitstempel an.
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

            self.get_logger().info("Bridge bereit")

        # --- Aufbau ------------------------------------------------------

        def _make_publishers(self) -> None:
            # Warnungen und Zustaende sollen auch ein spaet gestarteter
            # Abonnent noch sehen.
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

            # Zahlen statt Text - damit Foxglove sie plotten kann.
            # /diagnostics traegt dieselben Werte, aber als String.
            self.pub_latency = self.create_publisher(Float32, "~/latency_ms", 50)
            self.pub_rtt = self.create_publisher(Float32, "~/rtt_ms", 10)
            # Der Versatz ist mehrere Millionen ms gross - float32 hat dort
            # nur noch 1-ms-Schritte und der Drift waere unsichtbar.
            self.pub_offset = self.create_publisher(Float64, "~/offset_ms", 10)
            self.pub_drift = self.create_publisher(Float32, "~/drift_ppm", 10)

        def _make_subscribers(self, group) -> None:
            def sub(msg_type, name, handler):
                return self.create_subscription(msg_type, name, handler, 10,
                                                callback_group=group)

            sub(Twist, "/cmd_vel", self._on_cmd_vel)
            sub(Int32, "~/motor", lambda m: self._drive(m.data))
            sub(Float32, "~/steer", lambda m: self.link.steer(m.data))
            sub(Bool, "~/led", lambda m: self.link.led(m.data))
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

        # --- Zeitstempel --------------------------------------------------

        def _enqueue(self, frame: Frame) -> None:
            """Laeuft im Lese-Thread. Hier nur die ROS-Zeit des Lesens
            festhalten, alles Weitere macht ``_drain``."""
            self._inbox.append((frame, self.get_clock().now()))

        def _stamp(self, frame: Frame, read_at: Time) -> TimeMsg:
            """Sendezeitpunkt des Pakets als ROS-Zeit.

            Die Laufzeit wird in der monotonen Uhr gemessen (dort steckt der
            Abgleich mit dem ESP) und von der ROS-Zeit des Lesens abgezogen.
            So bleibt der Stempel richtig, egal ob ROS auf Systemzeit oder
            Simulationszeit laeuft.
            """
            latency = self.link.latency(frame)
            if latency is None or not 0.0 <= latency < 1.0:
                return read_at.to_msg()      # ungestempelt oder unplausibel
            self.pub_latency.publish(Float32(data=latency * 1e3))
            return Time(nanoseconds=read_at.nanoseconds - int(latency * 1e9)).to_msg()

        def _header(self, frame: Frame, read_at: Time, frame_id: str = "esp") -> Header:
            header = Header()
            header.stamp = self._stamp(frame, read_at)
            header.frame_id = frame_id
            return header

        # --- Eingehende Pakete --------------------------------------------

        def _drain(self) -> None:
            while self._inbox:
                frame, read_at = self._inbox.popleft()
                try:
                    self._publish(frame, read_at)
                except Exception as exc:
                    self.get_logger().error(
                        f"Paket 0x{frame.cmd:02X} nicht verarbeitet: {exc}")

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
                    f"Fahrt {result.move_id}: {MOVE_STATUS_TEXT.get(result.status, '?')} "
                    f"bei {result.position_deg:+.1f} grad")

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
                    f"Sendezeitstempel {'an' if payload[0] else 'aus'}")

        def _publish_joint(self, position_deg: float, frame: Frame,
                           read_at: Time,
                           velocity: Optional[float] = None) -> None:
            """Stellung der Ausgangswelle, optional mit Geschwindigkeit.

            MOVE_DONE und PROGRESS_RSP kennen nur die Position; die
            Geschwindigkeit steht ausschliesslich in CMD_TELEMETRY. Ein leeres
            ``velocity`` heisst in ROS "nicht gemessen" - besser als eine
            hingeschriebene Null.
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
            # Grobe Schaetzung ueber die Zellspannung. Ohne Strommessung und
            # Ruhespannung geht es nicht genauer - unter Last sackt der Wert ab.
            msg.percentage = float(_clamp((battery.cell_v - 3.3) / (4.2 - 3.3), 0.0, 1.0))
            msg.power_supply_status = BatteryState.POWER_SUPPLY_STATUS_DISCHARGING
            msg.power_supply_health = (
                BatteryState.POWER_SUPPLY_HEALTH_UNSPEC_FAILURE if battery.warning
                else BatteryState.POWER_SUPPLY_HEALTH_GOOD)
            self.pub_battery.publish(msg)
            self.pub_battery_low.publish(Bool(data=battery.warning))
            if battery.warning:
                self.get_logger().warn(
                    f"Unterspannung: {battery.cell_v:.3f} V/Zelle "
                    f"({battery.pack_v:.2f} V) - der ESP schaltet NICHT ab")

        # --- Ausgehende Befehle -------------------------------------------

        def _drive(self, duty: int) -> None:
            self.link.motor(int(duty))
            self._last_cmd_vel = monotonic()

        def _on_cmd_vel(self, msg: Twist) -> None:
            """Twist auf offene Steuerung abbilden.

            Der ESP hat keine Drehzahlregelung - ``linear.x`` wird direkt auf
            PWM umgerechnet. Die Zuordnung ist so gut wie ``max_linear``
            ausgemessen ist.
            """
            max_linear = float(self._p("max_linear")) or 1.0
            max_angular = float(self._p("max_angular")) or 1.0

            duty = int(round(_clamp(msg.linear.x / max_linear, -1.0, 1.0) * DUTY_MAX))
            steer = _clamp(msg.angular.z / max_angular, -1.0, 1.0) * 100.0

            self.link.motor(duty)
            self.link.steer(steer)
            self._last_cmd_vel = monotonic()

        def _on_move(self, msg: Float32) -> None:
            move_id = self.link.move(float(msg.data))
            self.get_logger().info(f"Fahrt {move_id}: {msg.data:+.1f} grad (relativ)")

        def _on_emergency(self) -> None:
            self.link.emergency()
            self.get_logger().warn("NOTHALT")

        def _on_pid_set(self, msg: Float32MultiArray) -> None:
            """[paramId, wert] oder gleich der ganze Satz [kp, ki, kd, ...]."""
            data = list(msg.data)
            if len(data) == 2:
                self.link.pid_set(int(data[0]), data[1])
            elif len(data) == len(PID_PARAMS):
                for index, value in enumerate(data):
                    self.link.pid_set(index, value)
            else:
                self.get_logger().error(
                    f"pid_set: erwarte 2 oder {len(PID_PARAMS)} Werte, "
                    f"bekam {len(data)}")
                return
            self.get_logger().info("PID gesetzt (fluechtig - ~/pid_save zum Sichern)")

        def _on_cal_action(self, msg: String) -> None:
            """Kalibrieren im Klartext, z.B. "plus", "left", "save".

            Fuer die Handkalibrierung an der Kommandozeile gedacht:
                ros2 topic pub --once <node>/cal_action std_msgs/String "data: plus"
            """
            name = msg.data.strip().lower()
            if name not in CAL_ACTIONS:
                self.get_logger().error(
                    f"unbekannte Aktion '{name}'. Moeglich: "
                    + ", ".join(sorted(CAL_ACTIONS)))
                return
            self._run_cal(CAL_ACTIONS[name], 0)

        def _on_cal(self, msg: Int32MultiArray) -> None:
            """[aktion, arg] - Aktionscodes siehe CAL_ACTIONS."""
            data = list(msg.data) + [0, 0]
            self._run_cal(int(data[0]), int(data[1]))

        def _run_cal(self, action: int, arg: int) -> None:
            state = self.link.calibrate(action, arg)
            if state is None:
                self.get_logger().error("Kalibrierung: keine Antwort vom ESP")
                return
            if state.status != 0x00:
                self.get_logger().warn(
                    f"Kalibrierung: {CAL_STATUS_TEXT.get(state.status, '?')}")
            self.get_logger().info(
                f"cal: pos={state.pos} mitte={state.center} links={state.left} "
                f"rechts={state.right} "
                f"gesetzt={'M' if state.have_center else '-'}"
                f"{'L' if state.have_left else '-'}"
                f"{'R' if state.have_right else '-'}"
                f"{' frei' if state.torque_free else ''}")

        # --- Services -----------------------------------------------------

        @staticmethod
        def _reply(response, ok: bool, message: str):
            response.success = ok
            response.message = message
            return response

        def _srv_emergency(self, _req, res):
            self.link.emergency()
            return self._reply(res, True, "Nothalt ausgeloest")

        def _srv_move_abort(self, _req, res):
            self.link.move_abort()
            return self._reply(res, True, "Fahrt abgebrochen")

        def _srv_pid_get(self, _req, res):
            values = self.link.pid_get()
            if values is None:
                return self._reply(res, False, "keine Antwort vom ESP")
            return self._reply(res, True,
                               "kp={:.3f} ki={:.3f} kd={:.3f}".format(*values))

        def _srv_pid_save(self, _req, res):
            ok = self.link.pid_save()
            if ok is None:
                return self._reply(res, False, "keine Antwort vom ESP")
            return self._reply(res, ok,
                               "ins NVS geschrieben" if ok else "NVS-Fehler")

        def _srv_cal_start(self, _req, res):
            state = self.link.calibrate("start")
            if state is None:
                return self._reply(res, False, "keine Antwort vom ESP")
            return self._reply(res, True,
                               "Kalibriermodus laeuft - jetzt ~/cal benutzen")

        def _srv_cal_save(self, _req, res):
            state = self.link.calibrate("save")
            if state is None:
                return self._reply(res, False, "keine Antwort vom ESP")
            ok = state.status == 0x01
            return self._reply(res, ok, CAL_STATUS_TEXT.get(state.status, "?"))

        def _srv_trim_save(self, _req, res):
            self.link.trim(2)
            return self._reply(res, True, "Trim-Offset gespeichert")

        def _srv_torque(self, _req, res):
            self.link.torque_report()
            return self._reply(res, True, "Ausgabe auf der USB-Konsole des ESP")

        def _srv_resync(self, _req, res):
            self.link.resync()
            time.sleep(0.1)
            if not self.link.clock.valid:
                return self._reply(res, False, "keine Antwort auf TIME_SYNC")
            return self._reply(res, True,
                               f"Versatz {self.link.clock.offset * 1e3:+.3f} ms, "
                               f"Drift {self.link.clock.drift_ppm:+.1f} ppm")

        def _srv_led(self, req, res):
            self.link.led(req.data)
            return self._reply(res, True, "LED " + ("an" if req.data else "aus"))

        def _srv_stamp(self, req, res):
            state = self.link.set_stamp_mode(req.data)
            if state is None:
                return self._reply(res, False, "keine Antwort vom ESP")
            return self._reply(res, True,
                               "Zeitstempel " + ("an" if state else "aus"))

        def _srv_torque_free(self, req, res):
            """Servo stromlos stellen, um die Lenkung von Hand zu bewegen.

            Geht nur bei laufender Kalibrierung - ausserhalb lehnt der ESP mit
            Status 0x04 ab. Erst ~/calibrate_start aufrufen.
            """
            state = self.link.calibrate("free" if req.data else "hold")
            if state is None:
                return self._reply(res, False, "keine Antwort vom ESP")
            if state.status == 0x04:
                return self._reply(res, False,
                                   "nur im Kalibriermodus - erst ~/calibrate_start")
            return self._reply(res, True,
                               "Servo " + ("frei" if req.data else "haelt"))

        # --- Telemetrietakt -----------------------------------------------

        def _apply_telemetry_period(self, period: float) -> None:
            actual = self.link.set_telemetry_rate(period)
            if actual <= 0:
                self.get_logger().info("Fahrtelemetrie aus")
                return
            if abs(actual - period) > 1e-6:
                self.get_logger().warn(
                    f"Telemetrietakt auf {actual * 1e3:.0f} ms begrenzt "
                    f"(angefragt {period * 1e3:.0f} ms, Minimum "
                    f"{TELEMETRY_MS_MIN} ms)")
            self.get_logger().info(
                f"Fahrtelemetrie alle {actual * 1e3:.0f} ms "
                f"({1.0 / actual:.1f} Hz) - Tempo bildet der ESP mit 10 Hz")

        def _on_set_parameters(self, params):
            """ros2 param set ... telemetry_period 0.1 durchreichen."""
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
            """Motor am Leben halten und bei ausbleibendem /cmd_vel stoppen."""
            if (self._cmd_vel_timeout > 0 and self._last_cmd_vel
                    and monotonic() - self._last_cmd_vel > self._cmd_vel_timeout):
                self._last_cmd_vel = 0.0
                self.link.motor_coast()
                self.get_logger().warn(
                    f"kein /cmd_vel seit {self._cmd_vel_timeout:.1f} s - Motor aus")
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
                status.message = "Uhren nicht abgeglichen"
            elif clock.rejected:
                status.level = DiagnosticStatus.WARN
                status.message = (f"{clock.rejected} von {len(clock.samples)} "
                                  "Messrunden unbrauchbar")
            elif clock.best_rtt > 0.05:
                status.level = DiagnosticStatus.WARN
                status.message = f"Umlauf {clock.best_rtt * 1e3:.1f} ms - Link traege"
            else:
                status.level = DiagnosticStatus.OK
                status.message = "in Ordnung"

            def kv(key, value):
                return KeyValue(key=key, value=str(value))

            status.values = [
                kv("versatz_ms", f"{clock.offset * 1e3:+.3f}" if clock.valid else "-"),
                kv("drift_ppm", f"{clock.drift_ppm:+.1f}" if clock.valid else "-"),
                kv("umlauf_ms", f"{clock.best_rtt * 1e3:.3f}" if clock.valid else "-"),
                kv("zeitstempel", "an" if self.link.stamp_mode else "aus"),
                kv("telemetrie_ms", int(float(self._p("telemetry_period")) * 1000)),
                kv("esp_neustarts", clock.boot_count),
                kv("pakete_rx", self.link.rx_frames),
                kv("pakete_tx", self.link.tx_frames),
                kv("sync_verloren", self.link.sync.lost),
                kv("sync_verworfen", clock.rejected),
                kv("unbekannte_pakete", self.link.parser.unknown),
            ]

            msg = DiagnosticArray()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.status = [status]
            self.pub_status.publish(msg)

            if clock.valid:
                self.pub_rtt.publish(Float32(data=clock.best_rtt * 1e3))
                self.pub_offset.publish(Float64(data=clock.offset * 1e3))
                self.pub_drift.publish(Float32(data=float(clock.drift_ppm)))

        # --- Ende ---------------------------------------------------------

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
# Selbsttest - Protokoll ohne ROS und ohne Hardware
# ==========================================================================

def _selftest() -> int:
    failures = 0

    def check(name: str, ok: bool, detail: str = "") -> None:
        nonlocal failures
        print(f"  [{'ok ' if ok else 'FAIL'}] {name}{'  ' + detail if detail else ''}")
        if not ok:
            failures += 1

    class _FakeLink(EspLink):
        """EspLink ohne seriellen Port - schreibt in eine Liste."""

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

    print("Befehle kodieren")
    link = _FakeLink()

    link.motor(700)
    check("Motor vorwaerts", link.sent[-1] == bytes([0xA5, 0x10, 0x00, 0x02, 0xBC]),
          link.sent[-1].hex(" "))
    link.motor(-700)
    check("Motor rueckwaerts", link.sent[-1] == bytes([0xA5, 0x10, 0x01, 0x02, 0xBC]))
    link.motor(9999)
    check("Motor begrenzt", link.sent[-1] == bytes([0xA5, 0x10, 0x00, 0x03, 0xFF]))

    link.steer(-100)
    check("Lenkung rechts", link.sent[-1] == bytes([0xA5, 0x20, 0x01, 0xFF, 0x9C]),
          link.sent[-1].hex(" "))
    link.steer(250)
    check("Lenkung begrenzt", link.sent[-1] == bytes([0xA5, 0x20, 0x01, 0x00, 0x64]))

    move_id = link.move(90.0)
    check("Fahrt 90 grad",
          link.sent[-1] == bytes([0xA5, 0x90, move_id, 0x00, 0x00, 0x03, 0x84]),
          link.sent[-1].hex(" "))
    link.move(-45.0)
    check("Fahrt negativ",
          link.sent[-1][3:] == struct.pack(">i", -450), link.sent[-1].hex(" "))
    check("move_id zaehlt", link.sent[-1][2] == move_id % 255 + 1)

    # Der haeufigste Fehler laut Spec: die x1000-Kodierung gilt auch fuer die
    # ganzzahligen Parameter. maxDuty=700 muss als 700000 rausgehen.
    link.pid_set("maxduty", 700)
    check("PID maxDuty x1000",
          link.sent[-1] == bytes([0xA5, 0x80, 0x04]) + struct.pack(">i", 700000),
          link.sent[-1].hex(" "))
    link.pid_set("kp", 4.25)
    check("PID kp x1000",
          link.sent[-1] == bytes([0xA5, 0x80, 0x00]) + struct.pack(">i", 4250))

    link.emergency()
    check("Nothalt", link.sent[-1] == bytes([0xA5, 0xFF]))
    link.led(True)
    check("LED", link.sent[-1] == bytes([0xA5, 0x30, 0x01]))
    link.trim(2)
    check("Trim speichern", link.sent[-1] == bytes([0xA5, 0x60, 0x02]))

    actual = link.set_telemetry_rate(0.05)
    check("Telemetrietakt 50 ms",
          link.sent[-1] == bytes([0xA5, 0xC0, 0x00, 0x32]) and actual == 0.05,
          link.sent[-1].hex(" "))
    actual = link.set_telemetry_rate(0.001)
    check("Takt auf Minimum begrenzt",
          link.sent[-1] == bytes([0xA5, 0xC0, 0x00, 0x14]) and actual == 0.020,
          f"{actual * 1e3:.0f} ms")
    actual = link.set_telemetry_rate(0)
    check("Telemetrie abschaltbar",
          link.sent[-1] == bytes([0xA5, 0xC0, 0x00, 0x00]) and actual == 0.0)

    print("Antworten auspacken")
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

    # Rueckwaertsfahrt. Die Bytes sind von Hand aus dem Zweierkomplement
    # gerechnet, nicht mit struct.pack erzeugt - sonst wuerde der Test nur
    # pruefen, dass Python zu sich selbst passt, und ein Vorzeichenfehler auf
    # der ESP-Seite bliebe unentdeckt.
    #   -905 = 0xFFFFFC77   -1800 = 0xFFFFF8F8   -700 = 0xFD44
    rueckwaerts = bytes([0xFF, 0xFF, 0xFC, 0x77,
                         0xFF, 0xFF, 0xF8, 0xF8,
                         0xFD, 0x44,
                         0x09, 0xC4])
    back = parse_telemetry(rueckwaerts)
    check("TELEMETRY negativ",
          (back.position_deg, back.speed_deg_s, back.duty, back.current_a)
          == (-90.5, -180.0, -700, 2.5), f"{back}")
    check("TELEMETRY negativ in rad/s", abs(back.speed_rad_s + math.pi) < 1e-9,
          f"{back.speed_rad_s:.6f}")
    check("Strom bleibt positiv", back.current_a > 0)

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
    check("schickt nach", len(link.sent) == 2)
    link.tick(0.2)
    check("nicht zu oft", len(link.sent) == 2)
    link._move_active = True
    link._last_motor_tx = monotonic() - 1.0
    link.tick(0.2)
    check("pausiert waehrend der Fahrt", len(link.sent) == 2)

    print("Empfang und Zeitstempel")
    link._move_active = True
    got: List[Frame] = []
    link.on(CMD_MOVE_DONE, got.append)
    link._dispatch(Frame(cmd=CMD_MOVE_DONE,
                         payload=bytes([1, 0]) + struct.pack(">i", 900),
                         esp_tx_raw=None, rx_mono=monotonic()))
    check("Callback gerufen", len(got) == 1)
    check("Fahrt beendet", link._move_active is False)
    check("ohne Abgleich kein Stempel", link.sent_at(got[0]) is None)

    print()
    print("Selbsttest fehlgeschlagen" if failures else "Selbsttest bestanden")
    return 1 if failures else 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--selftest", action="store_true",
                        help="Protokoll ohne ROS und ohne Hardware pruefen")
    known, rest = parser.parse_known_args()
    if known.selftest:
        raise SystemExit(_selftest())
    main()
