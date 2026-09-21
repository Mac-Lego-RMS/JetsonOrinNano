#!/usr/bin/env python3
"""
Ausparksequenz hin und zurueck fahren, um ihren Fehler zu MESSEN.

    ros2 run ekf ausparken_test

Zuerst wird die Folge vorwaerts gefahren, dann dieselbe Folge rueckwaerts:
Schritte in umgekehrter Reihenfolge, jeder mit derselben Lenkung und
negativer Strecke. Kinematisch hebt das den Hinweg exakt auf -- der Roboter
muesste wieder genau dort stehen, wo er losgefahren ist.

Was uebrig bleibt, ist der RUECKKEHRFEHLER, und der ist das Messergebnis.
Er braucht keine Lueckenmasse und keine Karte, nur die Odometrie, und er
trennt zwei Dinge, die sich sonst vermischen:

  * Ein Fehler, der sich auf dem Rueckweg AUFHEBT (Hinweg und Rueckweg weichen
    gleich ab), steckt im Modell -- Wendekreis, Radstand, Trimm.
  * Ein Fehler, der BLEIBT, steckt in der Mechanik -- Schlupf, Spiel in der
    Lenkung, Nachlauf des Positionsreglers.

Der Vergleich zwischen gefahrener und gerechneter Endpose des Hinwegs zeigt
zusaetzlich, wie gut die Kennlinie aus steer_calib.json gerade passt.

ACHTUNG: dieser Knoten FAEHRT. Er zaehlt vor dem Start herunter, und waehrend
einer Positionsfahrt wirkt /cmd_vel nicht -- der Nothalt geht ueber
/esp_serial_bridge/emergency.
"""
import math

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from std_msgs.msg import Float32, Float32MultiArray, Int32, Int32MultiArray

from ekf.ausparken import (bahn, cm_zu_grad, schritte_aus_flach, schritte_fuer,
                           spiegeln, LENK_QUELLE, SCHRITTE_STANDARD,
                           wenderadius)


def yaw_from_quaternion(q):
    siny = 2.0 * (q.w * q.z + q.x * q.y)
    cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny, cosy)


def wrap(a):
    return math.atan2(math.sin(a), math.cos(a))


def umkehren(schritte):
    """Die Folge, die den Hinweg aufhebt: rueckwaerts durch die Liste, jede
    Strecke negiert, jede Lenkung unveraendert."""
    return [(lenk, -cm) for lenk, cm in reversed(schritte)]


def im_startrahmen(start, pose):
    """Abweichung von ``start`` in dessen eigenem Rahmen: (laengs, quer, gier).

    Laengs und quer sind aussagekraeftiger als x/y im Odom-Rahmen -- "3 cm zu
    weit" und "3 cm daneben" sind verschiedene Fehler.
    """
    xs, ys, ths = start
    dx, dy = pose[0] - xs, pose[1] - ys
    c, s = math.cos(ths), math.sin(ths)
    return (c * dx + s * dy, -s * dx + c * dy, wrap(pose[2] - ths))


class Fahrer:
    """Faehrt eine Schrittfolge ueber die Bruecke ab.

    Erst lenken, dann die Positionsfahrt ausloesen, dann auf die Quittung
    warten. Kein /cmd_vel dazwischen: die Bruecke wuerde die Lenkung neu
    stellen, und ein Motorbefehl loest die laufende Fahrt ab.
    """

    def __init__(self, node, schritte, name):
        self.node = node
        self.schritte = schritte
        self.name = name
        self.i = 0
        self.phase = 'lenken'
        self.gesendet = False
        self.t0 = 0.0
        self.quittung = None
        self.los_t = None
        self.pose0 = None
        self.fehler = None

    @property
    def fertig(self):
        return self.i >= len(self.schritte)

    def takt(self, jetzt, pose):
        if self.fertig:
            return True
        lenk, cm = self.schritte[self.i]

        if self.phase == 'lenken':
            if not self.gesendet:
                self.gesendet = True
                self.t0 = jetzt
            # Waehrend der ganzen Wartezeit wiederholen: der allererste Befehl
            # auf einer frischen Verbindung geht in der DDS-Erkennung verloren.
            self.node.pub_steer.publish(Float32(data=float(lenk)))
            if jetzt - self.t0 < self.node.lenk_wartezeit:
                return False
            self.quittung = None
            self.los_t = jetzt
            self.pose0 = pose
            self.node.pub_move.publish(Float32(data=float(cm_zu_grad(cm))))
            self.phase = 'fahren'
            self.node.get_logger().info(
                "%s Zug %d/%d: Lenkung %+.0f %%, %+.1f cm"
                % (self.name, self.i + 1, len(self.schritte), lenk, cm))
            return False

        q = self.quittung
        if q is not None and q[0] >= self.los_t:
            status = q[1]
            plan = bahn((0.0, 0.0, 0.0), [(lenk, cm)])[-1][0]
            soll_dreh = math.degrees(plan[2])
            soll_weg = math.hypot(plan[0], plan[1])
            ist_dreh = math.degrees(wrap(pose[2] - self.pose0[2]))
            ist_weg = math.hypot(pose[0] - self.pose0[0], pose[1] - self.pose0[1])
            self.node.get_logger().info(
                "%s Zug %d fertig: Drehung %+.1f grad (Modell %+.1f), "
                "Weg %.1f cm (Modell %.1f)%s"
                % (self.name, self.i + 1, ist_dreh, soll_dreh,
                   ist_weg * 100, soll_weg * 100,
                   '' if status == 0 else '  [Status %d]' % status))
            if status == 2:
                self.fehler = ("Zug %d wurde von einem Motorbefehl abgeloest"
                               % (self.i + 1))
                return True
            if status == 1 and abs(ist_weg - soll_weg) > \
                    self.node.weg_toleranz_cm / 100.0:
                self.fehler = ("Zug %d: Zeitueberschreitung UND %.1f cm zu "
                               "kurz" % (self.i + 1, (soll_weg - ist_weg) * 100))
                return True
            self.i += 1
            self.phase = 'lenken'
            self.gesendet = False
            return self.fertig

        if jetzt - self.los_t > self.node.zug_timeout:
            self.fehler = ("Zug %d ohne Quittung nach %.0f s -- laeuft der "
                           "esp_serial_bridge?" % (self.i + 1, self.node.zug_timeout))
            return True
        return False


class AusparkTest(Node):

    def __init__(self):
        super().__init__('ausparken_test')
        arr = self._array_typ()
        self.declare_parameter('schritte', list(SCHRITTE_STANDARD), arr)
        self.declare_parameter('richtung', 'CW')
        self.declare_parameter('wiederholungen', 1)
        self.declare_parameter('pause_s', 2.0)
        self.declare_parameter('countdown_s', 3.0)
        self.declare_parameter('lenk_wartezeit', 0.6)
        self.declare_parameter('zug_timeout', 15.0)
        self.declare_parameter('weg_toleranz_cm', 1.0)
        self.declare_parameter('pid', [4.0, 140.0, 8.0, 90.0], arr)
        self.declare_parameter('pid_nachher', [4.0, 1023.0], arr)

        self.lenk_wartezeit = float(self.get_parameter('lenk_wartezeit').value)
        self.zug_timeout = float(self.get_parameter('zug_timeout').value)
        self.weg_toleranz_cm = float(self.get_parameter('weg_toleranz_cm').value)
        self.pause_s = float(self.get_parameter('pause_s').value)
        self.countdown_s = float(self.get_parameter('countdown_s').value)
        self.runden = max(1, int(self.get_parameter('wiederholungen').value))

        richtung = str(self.get_parameter('richtung').value).strip().upper()
        if richtung not in ('CW', 'CCW'):
            raise ValueError('richtung muss CW oder CCW sein, nicht "%s"'
                             % richtung)
        roh, herkunft = schritte_fuer(
            richtung, gemeinsam=list(self.get_parameter('schritte').value))
        self.hin = spiegeln(schritte_aus_flach(roh), richtung == 'CCW')
        self.zurueck = umkehren(self.hin)

        self.pub_steer = self.create_publisher(Float32, '/esp_serial_bridge/steer', 10)
        self.pub_move = self.create_publisher(Float32, '/esp_serial_bridge/move', 10)
        self.pub_pid = self.create_publisher(
            Float32MultiArray, '/esp_serial_bridge/pid_set', 10)
        self.pub_motor = self.create_publisher(Int32, '/esp_serial_bridge/motor', 10)
        self.create_subscription(Int32MultiArray, '/esp_serial_bridge/move_done',
                                 self.move_done_cb, 10)
        self.create_subscription(Odometry, '/ekf/odom', self.odom_cb, 10)

        self.pose = None
        self.zustand = 'WARTEN'
        self.t0 = self.now_s()
        self.fahrer = None
        self.runde = 0
        self.start_pose = None
        self.wende_pose = None
        self.protokoll = []

        ende = bahn((0.0, 0.0, 0.0), self.hin)[-1][0]
        self.get_logger().info(
            ">>> Ausparktest: %s, %s, %d Zuege, %.0f cm hin und dieselbe "
            "Strecke zurueck, %dx <<<"
            % (richtung, herkunft, len(self.hin),
               sum(abs(cm) for _l, cm in self.hin), self.runden))
        self.get_logger().info(
            "Lenkung: %s, Vollausschlag R = %.3f m. Modell sagt fuer den "
            "Hinweg: %.1f cm voraus, %.1f cm zur Seite, %+.1f grad."
            % (LENK_QUELLE or 'NOTNAGEL (steer_calib.json nicht gefunden!)',
               wenderadius(100.0), ende[0] * 100, ende[1] * 100,
               math.degrees(ende[2])))
        self.create_timer(1.0 / 30.0, self.control_loop)

    @staticmethod
    def _array_typ():
        from rcl_interfaces.msg import ParameterDescriptor, ParameterType
        return ParameterDescriptor(type=ParameterType.PARAMETER_DOUBLE_ARRAY)

    def now_s(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def odom_cb(self, msg):
        p = msg.pose.pose
        self.pose = (p.position.x, p.position.y,
                     yaw_from_quaternion(p.orientation))

    def move_done_cb(self, msg):
        if len(msg.data) >= 3 and self.fahrer is not None:
            self.fahrer.quittung = (self.now_s(), int(msg.data[1]),
                                    msg.data[2] / 10.0)

    def _pid(self, werte):
        werte = list(werte)
        for i in range(0, len(werte) - 1, 2):
            self.pub_pid.publish(
                Float32MultiArray(data=[float(werte[i]), float(werte[i + 1])]))

    def _abbruch(self, grund):
        self.pub_motor.publish(Int32(data=0))
        self._pid(self.get_parameter('pid_nachher').value)
        self.get_logger().error("Abgebrochen: %s" % grund)
        self.zustand = 'ENDE'

    def control_loop(self):
        if self.pose is None:
            self.get_logger().warn("warte auf /ekf/odom -- laeuft ekf_node?",
                                   throttle_duration_sec=2.0)
            return
        jetzt = self.now_s()

        if self.zustand == 'WARTEN':
            fehlt = [n for n, pub in (('steer', self.pub_steer),
                                      ('move', self.pub_move),
                                      ('pid_set', self.pub_pid),
                                      ('motor', self.pub_motor))
                     if pub.get_subscription_count() == 0]
            if fehlt:
                self.get_logger().info("warte auf die Bruecke (%s)"
                                       % ', '.join(fehlt),
                                       throttle_duration_sec=1.0)
                if jetzt - self.t0 > 15.0:
                    self._abbruch("Bruecke hoert nicht zu (%s)"
                                  % ', '.join(fehlt))
                return
            self.zustand = 'COUNTDOWN'
            self.t0 = jetzt
            return

        if self.zustand == 'COUNTDOWN':
            rest = self.countdown_s - (jetzt - self.t0)
            if rest > 0.0:
                self.get_logger().warn("Start in %.0f s -- er FAEHRT gleich."
                                       % math.ceil(rest),
                                       throttle_duration_sec=0.9)
                return
            self._pid(self.get_parameter('pid').value)
            self._neue_runde()
            return

        if self.zustand in ('HIN', 'ZURUECK'):
            if self.fahrer.takt(jetzt, self.pose):
                if self.fahrer.fehler:
                    self._abbruch(self.fahrer.fehler)
                    return
                self._abschnitt_fertig(jetzt)
            return

        if self.zustand == 'PAUSE':
            if jetzt - self.t0 < self.pause_s:
                return
            self.fahrer = Fahrer(self, self.zurueck, 'RUECKWEG')
            self.zustand = 'ZURUECK'
            return

        if self.zustand == 'ENDE':
            self._bericht()
            raise SystemExit(0)

    def _neue_runde(self):
        self.runde += 1
        self.start_pose = self.pose
        self.fahrer = Fahrer(self, self.hin, 'HINWEG')
        self.zustand = 'HIN'
        self.get_logger().info(
            "--- Runde %d/%d, Start bei (%.3f, %.3f, %+.1f grad) ---"
            % (self.runde, self.runden, self.pose[0], self.pose[1],
               math.degrees(self.pose[2])))

    def _abschnitt_fertig(self, jetzt):
        if self.zustand == 'HIN':
            self.wende_pose = self.pose
            laengs, quer, gier = im_startrahmen(self.start_pose, self.pose)
            modell = bahn((0.0, 0.0, 0.0), self.hin)[-1][0]
            self.get_logger().info(
                "HINWEG fertig: %.1f cm voraus, %.1f cm zur Seite, %+.1f grad "
                "| Modell: %.1f / %.1f / %+.1f"
                % (laengs * 100, quer * 100, math.degrees(gier),
                   modell[0] * 100, modell[1] * 100, math.degrees(modell[2])))
            self.zustand = 'PAUSE'
            self.t0 = jetzt
            return

        laengs, quer, gier = im_startrahmen(self.start_pose, self.pose)
        self.protokoll.append((laengs, quer, gier))
        self.get_logger().info(
            "RUECKKEHRFEHLER Runde %d: %+.1f cm laengs, %+.1f cm quer, "
            "%+.1f grad  (Abstand %.1f cm)"
            % (self.runde, laengs * 100, quer * 100, math.degrees(gier),
               math.hypot(laengs, quer) * 100))
        if self.runde < self.runden:
            self._neue_runde()
        else:
            self._pid(self.get_parameter('pid_nachher').value)
            self.zustand = 'ENDE'

    def _bericht(self):
        if not self.protokoll:
            self.get_logger().warn("Keine vollstaendige Runde gefahren.")
            return
        self.get_logger().info("=== Ergebnis ueber %d Runde(n) ==="
                               % len(self.protokoll))
        for i, (l, q, g) in enumerate(self.protokoll, 1):
            self.get_logger().info(
                "  Runde %d: %+6.1f cm laengs  %+6.1f cm quer  %+6.1f grad"
                % (i, l * 100, q * 100, math.degrees(g)))
        n = len(self.protokoll)
        ml = sum(p[0] for p in self.protokoll) / n
        mq = sum(p[1] for p in self.protokoll) / n
        mg = sum(p[2] for p in self.protokoll) / n
        self.get_logger().info(
            "  Mittel:  %+6.1f cm laengs  %+6.1f cm quer  %+6.1f grad"
            % (ml * 100, mq * 100, math.degrees(mg)))
        self.get_logger().info(
            "Ein Fehler, der sich hier aufhebt, steckt im Modell "
            "(Wendekreis, Radstand); was uebrig bleibt, ist Mechanik "
            "(Schlupf, Lenkspiel, Nachlauf).")


def main(args=None):
    rclpy.init(args=args)
    node = AusparkTest()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        try:
            node.pub_motor.publish(Int32(data=0))
        except Exception:
            pass
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
