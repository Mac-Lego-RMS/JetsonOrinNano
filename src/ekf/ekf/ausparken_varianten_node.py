#!/usr/bin/env python3
"""
NUR ausparken -- eine der sechs Varianten, zum Einmessen und Anpassen.

    python3 -m ekf.ausparken_varianten_node --ros-args -p lage:=mitte
    python3 -m ekf.ausparken_varianten_node --ros-args -p lage:=innen -p richtung:=CW
    python3 -m ekf.ausparken_varianten_node --ros-args -p lage:=aussen \\
        -p schritte:="[0.0,0.0, 100.0,6.0, -100.0,-4.5, 100.0,9.6, 0.0,5.0, -100.0,24.0, 0.0,0.0]"

Die Varianten stehen in ausparken.py: SCHRITTE_{CW,CCW}_{INNEN,MITTE,AUSSEN}.
Ablauf:
  1. Fahrtrichtung aus dem Scan messen, so wie der Regler es tut (oder mit
     richtung:=CW/CCW vorgeben). Damit ist die Variante festgelegt.
  2. Trockenlauf gegen die Lueckenmasse: Kollision? Endlage laut Modell?
  3. Countdown, dann die Zuege abfahren -- ueber denselben Zugfahrer wie der
     Regler, also genau so, wie es spaeter im Lauf passiert.
  4. Am Ende stehen bleiben und mit dem Lidar messen, wo er in der Spur steht:
     Abstand zur Aussenbande und Kurs zur Bande.
  5. Die gefahrene Liste zum Hineinkopieren nach ausparken.py ausgeben.

Mit schritte:=[...] faehrt er statt der Tabelle diese Liste -- zum Probieren,
ohne ausparken.py anzufassen. Passt sie, kommt sie per Hand in die Tabelle.

Zurueck in die Luecke stellt ihr ihn von Hand; es gibt keinen Rueckweg.

Beim Start werden EKF und scan_processor frisch gestartet (Fenster 8/9, ueber
den Neustart-Waechter in Fenster 11, siehe ekf/schaetzung_neustart.py) -- wie
beim Regler. Abschalten: -p schaetzung_neustart:=false.

Laufen muessen esp_serial_bridge, IMU und Lidar (also start_robot.sh). Der
Regler darf NICHT laufen: waehrend eines Zuges wirkt /cmd_vel nicht, und der
Regler wuerde dazwischenfunken. Der Nothalt geht ueber
/esp_serial_bridge/emergency.
"""
import math

import numpy as np
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float32, Float32MultiArray, Int32, Int32MultiArray

from ekf.ausparken import (LAGEN, LENK_QUELLE, richtung_aus_scan,
                           schritte_aus_flach, schritte_variante, simuliere,
                           spiegeln, startpose, wenderadius)
from ekf.schaetzung_neustart import neu_starten
from ekf.wall_extraction import LIDAR_OFFSET_X, scan_to_points
from ekf.zugfahrer import Fahrer, pose_text, wrap

SPUR_BREITE = 1.00        # Aussenbande bis Innenbande auf der Startgeraden [m]


def yaw_from_quaternion(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                      1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def im_startrahmen(start, pose):
    """(laengs, quer, gier) von ``pose`` im Rahmen von ``start``."""
    dx, dy = pose[0] - start[0], pose[1] - start[1]
    c, s = math.cos(start[2]), math.sin(start[2])
    return c * dx + s * dy, -s * dx + c * dy, wrap(pose[2] - start[2])


def wand_messen(punkte, seite_links, max_abstand=0.9, halbwinkel_grad=55.0,
                nur_vorn=True):
    """Gerade durch die Wandpunkte auf einer Seite (Hauptachse).

    ``punkte`` im base_link-Rahmen. Rueckgabe (Abstand von base_link [m],
    Winkel der Wand gegen die Fahrzeuglaengsachse [rad], Punktzahl) oder None.
    Der Winkel ist positiv, wenn die Wand nach vorn hin naeher kommt -- dann
    zeigt die Nase auf die Wand zu.
    """
    mitte = math.pi / 2.0 if seite_links else -math.pi / 2.0
    w = np.arctan2(punkte[:, 1], punkte[:, 0])
    r = np.hypot(punkte[:, 0], punkte[:, 1])
    d = np.abs(np.arctan2(np.sin(w - mitte), np.cos(w - mitte)))
    # Aussenbande nur vor der Hinterachse (nur_vorn): schraeg dahinter steht
    # nach dem Ausparken die vordere Buchtwand, quer zur Bande -- sie wuerde
    # die Gerade verdrehen. Die Innenbande dagegen endet oft schon knapp hinter
    # dem Roboter (CCW: Inselecke bei x=0,25, er steht bei 0,29) und ist nur
    # schraeg hinten zu sehen.
    k = (d <= math.radians(halbwinkel_grad)) & (r <= max_abstand)
    if nur_vorn:
        k &= punkte[:, 0] >= 0.0
    p = punkte[k]
    if len(p) < 8:
        return None
    for _ in range(2):
        # zweimal: anpassen, Ausreisser (Pylone, Buchtreste) > 3 cm verwerfen
        m = p.mean(axis=0)
        _u, _s, vt = np.linalg.svd(p - m)
        n = np.array([-vt[0][1], vt[0][0]])
        rest = np.abs((p - m) @ n)
        if (rest > 0.03).sum() == 0 or (rest <= 0.03).sum() < 8:
            break
        p = p[rest <= 0.03]
    m = p.mean(axis=0)
    _u, _s, vt = np.linalg.svd(p - m)
    richtung = vt[0]
    if richtung[0] < 0:
        richtung = -richtung
    normale = np.array([-richtung[1], richtung[0]])
    abstand = abs(float(normale @ m))
    winkel = math.atan2(richtung[1], richtung[0])     # Wand gegen +x
    # Rechts (y<0): kommt die Wand nach vorn naeher, steigt y -> winkel > 0.
    # Links spiegelbildlich.
    zu_wand = winkel if not seite_links else -winkel
    return abstand, zu_wand, len(p)


class AusparkVarianten(Node):

    def __init__(self):
        super().__init__('ausparken_varianten')
        from rcl_interfaces.msg import ParameterDescriptor, ParameterType
        arr = ParameterDescriptor(type=ParameterType.PARAMETER_DOUBLE_ARRAY)
        self.declare_parameter('lage', 'mitte')
        self.declare_parameter('richtung', '')
        self.declare_parameter('schritte', [0.0], arr)   # [0.0] = aus der Tabelle
        self.declare_parameter('scans', 5)
        self.declare_parameter('sektor_grad', 20.0)
        self.declare_parameter('richtung_timeout', 8.0)
        self.declare_parameter('countdown_s', 3.0)
        self.declare_parameter('lenk_wartezeit', 0.6)
        self.declare_parameter('zug_timeout', 15.0)
        self.declare_parameter('weg_toleranz_cm', 1.0)
        self.declare_parameter('mess_scans', 10)
        self.declare_parameter('pid', [4.0, 140.0, 8.0, 90.0], arr)
        self.declare_parameter('pid_nachher', [4.0, 1023.0], arr)

        self.lage = str(self.get_parameter('lage').value).strip().lower()
        if self.lage not in LAGEN:
            raise ValueError('lage muss %s sein, nicht "%s"'
                             % (' / '.join(LAGEN), self.lage))
        self.vorgabe = str(self.get_parameter('richtung').value).strip().upper()
        if self.vorgabe not in ('', 'CW', 'CCW'):
            raise ValueError('richtung muss leer, CW oder CCW sein')
        eigene = [float(v) for v in self.get_parameter('schritte').value]
        self.eigene = eigene if len(eigene) >= 2 else None

        self.scans = max(1, int(self.get_parameter('scans').value))
        self.sektor_grad = float(self.get_parameter('sektor_grad').value)
        self.richtung_timeout = float(self.get_parameter('richtung_timeout').value)
        self.countdown_s = float(self.get_parameter('countdown_s').value)
        self.lenk_wartezeit = float(self.get_parameter('lenk_wartezeit').value)
        self.zug_timeout = float(self.get_parameter('zug_timeout').value)
        self.weg_toleranz_cm = float(self.get_parameter('weg_toleranz_cm').value)
        self.mess_scans = max(3, int(self.get_parameter('mess_scans').value))

        self.pub_steer = self.create_publisher(Float32, '/esp_serial_bridge/steer', 10)
        self.pub_move = self.create_publisher(Float32, '/esp_serial_bridge/move', 10)
        self.pub_pid = self.create_publisher(
            Float32MultiArray, '/esp_serial_bridge/pid_set', 10)
        self.pub_motor = self.create_publisher(Int32, '/esp_serial_bridge/motor', 10)
        self.create_subscription(Int32MultiArray, '/esp_serial_bridge/move_done',
                                 self.move_done_cb, 10)
        self.create_subscription(Odometry, '/ekf/odom', self.odom_cb, 10)
        self.create_subscription(LaserScan, '/scan', self.scan_cb, 10)

        self.pose = None
        self.zustand = 'WARTEN'
        self.t0 = self.now_s()
        self.stimmen = []
        self.letzter_grund = None
        self.richtung = None
        self.roh = None
        self.name = None
        self.folge = None
        self.fahrer = None
        self.start_pose = None
        self.messungen = []
        self.modell_ende = None

        self.get_logger().info(
            '>>> Ausparken, Variante "%s". Fahrtrichtung: %s <<<'
            % (self.lage, self.vorgabe + ' (vorgegeben)' if self.vorgabe
               else 'wird aus dem Scan gemessen'))
        self.get_logger().info(
            'Lenkung: %s, Vollausschlag R = %.3f m.'
            % (LENK_QUELLE or 'NOTNAGEL (steer_calib.json nicht gefunden!)',
               wenderadius(100.0)))
        self.create_timer(1.0 / 30.0, self.control_loop)

    # ------------------------------------------------------------ Eingaenge
    def now_s(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def odom_cb(self, msg):
        p = msg.pose.pose
        self.pose = (p.position.x, p.position.y, yaw_from_quaternion(p.orientation))

    def move_done_cb(self, msg):
        if len(msg.data) >= 3 and self.fahrer is not None:
            self.fahrer.quittung = (self.now_s(), int(msg.data[1]), msg.data[2] / 10.0)

    def scan_cb(self, msg):
        if self.zustand == 'RICHTUNG':
            e = richtung_aus_scan(scan_to_points(msg), halbwinkel_grad=self.sektor_grad)
            self.letzter_grund = e['grund']
            if not e['sicher']:
                self.stimmen = []
                return
            if self.stimmen and self.stimmen[-1] != e['richtung']:
                self.stimmen = []
            self.stimmen.append(e['richtung'])
        elif self.zustand == 'MESSEN':
            p = scan_to_points(msg)
            p = np.column_stack((p[:, 0] + LIDAR_OFFSET_X, p[:, 1]))   # -> base_link
            self.messungen.append(p)

    def _pid(self, werte):
        werte = list(werte)
        for i in range(0, len(werte) - 1, 2):
            self.pub_pid.publish(Float32MultiArray(data=[float(werte[i]), float(werte[i + 1])]))

    def _abbruch(self, grund):
        self.pub_motor.publish(Int32(data=0))
        self._pid(self.get_parameter('pid_nachher').value)
        self.get_logger().error('Abgebrochen: %s' % grund)
        if self.pose is not None:
            self.get_logger().error('           steht bei  %s' % pose_text(self.pose))
        raise SystemExit(1)

    # ---------------------------------------------------------- Vorbereitung
    def _folge_festlegen(self, richtung, quelle):
        self.richtung = richtung
        if self.eigene:
            self.roh, self.name = list(self.eigene), 'schritte:= (Kommandozeile)'
        else:
            self.roh, self.name = schritte_variante(richtung, self.lage)
        roh_paare = schritte_aus_flach(self.roh)
        self.folge = spiegeln(roh_paare, richtung == 'CCW')
        log = self.get_logger()
        log.info('Fahrtrichtung %s (%s) -> %s, %d Zuege, %.0f cm Weg.'
                 % (richtung, quelle, self.name, len(self.folge),
                    sum(abs(cm) for _l, cm in self.folge)))

        # Trockenlauf im Lueckenrahmen: Aussenbande y=0, offene Seite +y.
        e = simuliere(spiegeln(roh_paare, True))
        x0 = startpose()[0]
        ende = e['endpose']
        self.modell_ende = ende
        log.info('Trockenlauf: %s, engster Abstand %.0f mm zu Magenta, %.0f mm '
                 'zur Aussenbande, %s.'
                 % ('KOLLISION in Zug %d' % e['bei_schritt'] if e['kollision']
                    else 'keine Kollision',
                    e['magenta_abstand_m'] * 1000, e['wand_abstand_m'] * 1000,
                    'am Ende frei' if e['frei'] else 'am Ende NOCH IN DER LUECKE'))
        log.info('Modell-Endlage: base_link %.1f cm von der Aussenbande (Spurmitte '
                 '= %.0f cm), %.1f cm voraus, Kurs %+.1f grad.'
                 % (ende[1] * 100, SPUR_BREITE * 50, (ende[0] - x0) * 100,
                    math.degrees(ende[2])))
        if e['kollision']:
            log.warn('Die Folge geht rechnerisch nicht auf -- sie wird trotzdem '
                     'gefahren, das Modell kann von der Luecke abweichen. '
                     'Hand an den Nothalt.')

    # ---------------------------------------------------------------- Takt
    def control_loop(self):
        if self.pose is None:
            self.get_logger().warn('warte auf /ekf/odom -- laeuft ekf_node?',
                                   throttle_duration_sec=2.0)
            return
        jetzt = self.now_s()

        if self.zustand == 'WARTEN':
            fehlt = [n for n, pub in (('steer', self.pub_steer), ('move', self.pub_move),
                                      ('pid_set', self.pub_pid), ('motor', self.pub_motor))
                     if pub.get_subscription_count() == 0]
            if fehlt:
                self.get_logger().info('warte auf die Bruecke (%s)' % ', '.join(fehlt),
                                       throttle_duration_sec=1.0)
                if jetzt - self.t0 > 15.0:
                    self._abbruch('Bruecke hoert nicht zu (%s)' % ', '.join(fehlt))
                return
            self.t0 = jetzt
            if self.vorgabe:
                self._folge_festlegen(self.vorgabe, 'vorgegeben')
                self.zustand = 'COUNTDOWN'
            else:
                self.zustand = 'RICHTUNG'
                self.get_logger().info('suche die offene Seite, %d einige Scans noetig.'
                                       % self.scans)
            return

        if self.zustand == 'RICHTUNG':
            if len(self.stimmen) < self.scans:
                if jetzt - self.t0 > self.richtung_timeout:
                    self._abbruch('keine eindeutige Fahrtrichtung in %.0f s -- zuletzt: %s. '
                                  'Steht er in der Luecke? Sonst richtung:=CW oder CCW.'
                                  % (self.richtung_timeout,
                                     self.letzter_grund or 'kein /scan empfangen'))
                return
            self._folge_festlegen(self.stimmen[-1], 'aus dem Scan gemessen')
            self.zustand = 'COUNTDOWN'
            self.t0 = jetzt
            return

        if self.zustand == 'COUNTDOWN':
            rest = self.countdown_s - (jetzt - self.t0)
            if rest > 0.0:
                self.get_logger().warn('Start in %.0f s -- er FAEHRT gleich.' % math.ceil(rest),
                                       throttle_duration_sec=0.9)
                return
            self._pid(self.get_parameter('pid').value)
            self.start_pose = self.pose
            self.get_logger().info('Start bei  %s' % pose_text(self.pose))
            self.fahrer = Fahrer(self, self.folge, 'AUSPARKEN')
            self.zustand = 'FAHREN'
            return

        if self.zustand == 'FAHREN':
            if self.fahrer.takt(jetzt, self.pose):
                if self.fahrer.fehler:
                    self._abbruch(self.fahrer.fehler)
                self._pid(self.get_parameter('pid_nachher').value)
                self.messungen = []
                self.zustand = 'MESSEN'
                self.t0 = jetzt
            return

        if self.zustand == 'MESSEN':
            # erst kurz ausruhen lassen, dann ein paar Scans sammeln
            if jetzt - self.t0 < 0.5:
                self.messungen = []
                return
            if len(self.messungen) < self.mess_scans and jetzt - self.t0 < 4.0:
                return
            self._bericht()
            raise SystemExit(0)

    # -------------------------------------------------------------- Bericht
    def _bericht(self):
        log = self.get_logger()
        laengs, quer, gier = im_startrahmen(self.start_pose, self.pose)
        ende = self.modell_ende
        log.info('=== %s, Fahrtrichtung %s ===' % (self.name, self.richtung))
        log.info('Odometrie ab Start: %.1f cm voraus, %.1f cm zur offenen Seite, '
                 'Kurs %+.1f grad zur Startlage'
                 % (laengs * 100, (quer if self.richtung == 'CCW' else -quer) * 100,
                    math.degrees(gier if self.richtung == 'CCW' else -gier)))
        log.info('           steht bei  %s' % pose_text(self.pose))

        offen_links = self.richtung == 'CCW'
        if self.messungen:
            p = np.vstack(self.messungen)
            aussen = wand_messen(p, seite_links=not offen_links)
            # Die Innenbande ist hier nicht verlaesslich zu messen: der Roboter
            # steht nach dem Ausparken genau neben der Inselecke (Lauf 16:
            # "Spur 1,28 m"). Die Spur ist im Hindernisrennen aber immer
            # SPUR_BREITE breit -- die Lage folgt aus der Aussenbande allein.
            if aussen:
                a, w, n = aussen
                log.info('Lidar: Aussenbande %.1f cm von base_link (Modell %.1f), '
                         'Kurs %+.1f grad zur Bande (%s)  [%d Punkte]'
                         % (a * 100, ende[1] * 100, math.degrees(w),
                            'Nase zur Bande' if w > 0 else 'Nase zur Spurmitte', n))
                log.info('Lage in der Spur: %.0f %% von aussen (0 = Aussenbande, '
                         '50 = Mitte, 100 = Innenbande), bei %.2f m Spurbreite.'
                         % (100 * a / SPUR_BREITE, SPUR_BREITE))
            else:
                log.warn('Lidar: Aussenbande nicht gefunden (zu nah oder verdeckt).')
        else:
            log.warn('Keine Scans fuer die Endmessung empfangen.')

        # zum Hineinkopieren
        paare = ',\n'.join('    %6.1f, %5.1f' % (self.roh[i], self.roh[i + 1])
                           for i in range(0, len(self.roh), 2))
        tabelle = ('SCHRITTE_%s_%s' % (self.richtung, self.lage.upper())
                   if not self.name.startswith('SCHRITTE') else self.name)
        log.info('Gefahrene Liste fuer ausparken.py:\n%s = [\n%s,\n]' % (tabelle, paare))


def main(args=None):
    rclpy.init(args=args)
    # Vor dem eigenen Knoten: sonst kaemen die gelatchten Topics noch vom
    # alten scan_processor.
    neu_starten('ausparken_varianten')
    node = AusparkVarianten()
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
