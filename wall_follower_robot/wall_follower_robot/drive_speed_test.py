"""Geradeausfahrt mit fester Dauer, dabei die Encoder-Drehzahl mitschreiben.

Zum Ausmessen von ``max_linear`` der Bruecke: der Roboter faehrt eine
einstellbare Zeit mit einem festen Anteil der Hoechstgeschwindigkeit
geradeaus. Die ersten und letzten Zehntelsekunden werden verworfen -
dort beschleunigt bzw. bremst der Antrieb noch und die Werte wuerden die
Varianz verfaelschen. Am Ende stehen Mittelwert und Varianz der
gemessenen Achsdrehzahl auf der Konsole - dazu der gesamte zurueckgelegte
Drehwinkel der Achse, den der ESP als absolute Stellung seit seinem Boot
mitliefert (``JointState.position``). Der Weg wird ueber die ganze Fahrt
gezaehlt, nicht nur im Messfenster, und schliesst das Nachlaufen nach dem
Stopp mit ein (``report_delay``).

Echte Encoder-Rohticks stehen nicht im UART-Protokoll - der ESP schickt
Zehntelgrad. Mit ``ticks_per_rev`` werden sie zurueckgerechnet.

    ros2 run wall_follower_robot drive_speed_test \
        --ros-args -p duration:=5.0 -p speed_fraction:=0.4 \
                   -p ticks_per_rev:=1440.0
"""

import math
import statistics
import time

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Twist
from sensor_msgs.msg import JointState

RAD_TO_DEG = 180.0 / math.pi


class DriveSpeedTest(Node):

    def __init__(self) -> None:
        super().__init__("drive_speed_test")

        # --- Parameter ---
        self.declare_parameter("duration", 5.0)          # Fahrzeit in s
        self.declare_parameter("speed_fraction", 0.4)    # Anteil von max_linear
        # Muss zu max_linear der Bruecke passen, sonst stimmt der Anteil nicht.
        self.declare_parameter("max_linear", 1.0)        # m/s bei Volldampf
        self.declare_parameter("settle_start", 0.5)      # Anlauf verwerfen
        self.declare_parameter("settle_end", 0.5)        # Bremsen verwerfen
        self.declare_parameter("publish_rate", 20.0)     # Hz, < cmd_vel_timeout
        self.declare_parameter("joint_topic",
                               "/esp_serial_bridge/joint_states")
        self.declare_parameter("joint_name", "drive_axle")
        # Nachlauf abwarten, sonst fehlt der ausgerollte Rest im Gesamtweg.
        self.declare_parameter("report_delay", 0.5)      # s nach dem Stopp
        # 0 = unbekannt, dann keine Tick-Ausgabe. Bezieht sich auf eine
        # Umdrehung der *Ausgangswelle*, nicht auf die Motorwelle.
        self.declare_parameter("ticks_per_rev", 0.0)

        self.duration = float(self._p("duration"))
        self.settle_start = float(self._p("settle_start"))
        self.settle_end = float(self._p("settle_end"))
        self.joint_name = str(self._p("joint_name"))
        self.speed = (float(self._p("speed_fraction"))
                      * float(self._p("max_linear")))

        window = self.duration - self.settle_start - self.settle_end
        if window <= 0.0:
            raise ValueError(
                f"duration ({self.duration:.2f} s) ist zu kurz fuer "
                f"settle_start + settle_end "
                f"({self.settle_start + self.settle_end:.2f} s)")

        # --- Schnittstellen ---
        self.pub_cmd_vel = self.create_publisher(Twist, "/cmd_vel", 10)
        self.create_subscription(JointState, str(self._p("joint_topic")),
                                 self._on_joint, 50)

        self.samples: list[float] = []
        # Absolutstellung der Achse: erster/letzter Wert der ganzen Fahrt und
        # des Messfensters. None, solange noch nichts angekommen ist.
        self.pos_first: float | None = None
        self.pos_last: float | None = None
        self.pos_window_start: float | None = None
        self.pos_window_end: float | None = None
        self.recording = True
        self.finished = False
        self.start_time = time.monotonic()
        self.create_timer(1.0 / float(self._p("publish_rate")), self._tick)

        self.get_logger().info(
            f"Fahre {self.duration:.1f} s mit {self.speed:.3f} m/s "
            f"({float(self._p('speed_fraction')) * 100:.0f} %), "
            f"Messfenster {window:.1f} s")

    def _p(self, name: str):
        return self.get_parameter(name).value

    # --- Ablauf --------------------------------------------------------

    def _tick(self) -> None:
        """Fahrbefehl nachlegen - ohne Nachschub schaltet der ESP ab."""
        if self.finished:
            return

        elapsed = time.monotonic() - self.start_time
        if elapsed >= self.duration:
            self._stop()
            # Nicht sofort auswerten: der Antrieb rollt noch aus und die
            # letzten Telemetriepakete stehen noch aus. Erst danach steht der
            # Gesamtweg fest.
            delay = max(0.0, float(self._p("report_delay")))
            self.coast_timer = self.create_timer(delay, self._finish)
            return

        cmd = Twist()
        cmd.linear.x = self.speed
        self.pub_cmd_vel.publish(cmd)

    def _measuring(self) -> bool:
        elapsed = time.monotonic() - self.start_time
        return (self.settle_start <= elapsed
                <= self.duration - self.settle_end)

    def _on_joint(self, msg: JointState) -> None:
        if not self.recording:
            return
        try:
            index = list(msg.name).index(self.joint_name)
        except ValueError:
            index = 0

        # Die Stellung laeuft ueber die ganze Fahrt mit - auch waehrend Anlauf,
        # Bremsen und Nachlauf, sonst fehlt sie im Gesamtweg.
        if index < len(msg.position):
            position = float(msg.position[index])
            if self.pos_first is None:
                self.pos_first = position
            self.pos_last = position
            if self._measuring():
                if self.pos_window_start is None:
                    self.pos_window_start = position
                self.pos_window_end = position

        # Die Drehzahl dagegen nur im Messfenster - und nur, wenn das Paket
        # ueberhaupt eine mitbringt (MOVE_DONE/PROGRESS tun das nicht).
        if (not self.finished and msg.velocity and self._measuring()
                and index < len(msg.velocity)):
            self.samples.append(float(msg.velocity[index]))

    def _stop(self) -> None:
        """Mehrfach halten - ein einzelnes Paket kann verloren gehen."""
        self.finished = True
        for _ in range(5):
            self.pub_cmd_vel.publish(Twist())
            time.sleep(0.02)

    def _finish(self) -> None:
        """Nachlauf ist vorbei: Mitschreiben beenden und auswerten."""
        self.coast_timer.cancel()
        self.recording = False
        self._report()
        rclpy.shutdown()

    # --- Auswertung ----------------------------------------------------

    def _travel(self, start, end) -> float:
        """Zurueckgelegter Drehwinkel in rad, oder ``nan`` ohne Messwerte."""
        if start is None or end is None:
            return math.nan
        return end - start

    def _report(self) -> None:
        print("\n--- Messfahrt ---")
        self._report_speed()
        self._report_travel()
        print("-----------------\n")

    def _report_speed(self) -> None:
        if len(self.samples) < 2:
            self.get_logger().error(
                f"Nur {len(self.samples)} Messpunkte auf "
                f"{self._p('joint_topic')} - laeuft die Bruecke?")
            return

        mean = statistics.fmean(self.samples)
        # Populationsvarianz: die Stichprobe ist die ganze Messfahrt.
        var = statistics.pvariance(self.samples, mu=mean)
        sigma = math.sqrt(var)

        print(f"Messpunkte:  {len(self.samples)}")
        print(f"Sollwert:    {self.speed:.3f} m/s")
        print(f"Mittelwert:  {mean:.4f} rad/s  ({mean * RAD_TO_DEG:.2f} grad/s)")
        print(f"Varianz:     {var:.6f} rad^2/s^2")
        print(f"Standardabw: {sigma:.4f} rad/s  ({sigma * RAD_TO_DEG:.2f} grad/s)")

    def _report_travel(self) -> None:
        """Gesamtweg der Achse aus der Absolutstellung des ESP.

        Der ESP zaehlt die Stellung seit seinem Boot durch; die Differenz
        zwischen erstem und letztem Paket ist der gefahrene Drehwinkel.
        """
        total = self._travel(self.pos_first, self.pos_last)
        if math.isnan(total):
            print("Gesamtweg:   -- (keine Position empfangen)")
            return

        window = self._travel(self.pos_window_start, self.pos_window_end)
        ticks_per_rev = float(self._p("ticks_per_rev"))

        def line(label: str, radians: float) -> None:
            degrees = radians * RAD_TO_DEG
            text = (f"{label:<13}{radians:+.3f} rad  ({degrees:+.1f} grad, "
                    f"{degrees / 360.0:+.2f} U")
            if ticks_per_rev > 0.0:
                text += f", {degrees / 360.0 * ticks_per_rev:+.0f} Ticks"
            print(text + ")")

        line("Gesamtweg:", total)
        if not math.isnan(window):
            line("im Fenster:", window)
        if ticks_per_rev <= 0.0:
            print("             (ticks_per_rev:=<Ticks je Achsumdrehung> "
                  "setzen fuer Tickzahlen)")


def main(args=None) -> None:
    rclpy.init(args=args)
    node = DriveSpeedTest()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node._stop()
        node.recording = False
        node._report()
    finally:
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
