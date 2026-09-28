"""EKF und scan_processor frisch starten, bevor ein Fahrknoten loslegt.

Die Karte haengt an der Pose, bei der ekf_node startet, und der scan_processor
misst Richtung und Lage in der Bucht nur beim Start (start_from_bay). Nach dem
Wieder-Hinstellen in die Luecke gehoeren also beide neu gestartet -- vor jedem
Lauf und vor jedem Ausparkversuch. Das passiert hier automatisch, sobald
round1_controller oder ausparken_varianten_node mit ros2 run starten.

Die beiden Knoten laufen in den tmux-Fenstern 8 und 9 auf dem Jetson, und an
tmux kommt man aus dem Container nicht heran. Deshalb geht der Neustart ueber
einen Waechter auf dem Jetson (Fenster 11, src/schaetzung_waechter.sh):

  1. Dieser Knoten legt eine Anfrage mit einer Kennung in den gemeinsamen
     Workspace (/workspace = ~/ros2_ws).
  2. Der Waechter startet 8 und 9 ueber schaetzung_neustart.sh neu -- die Logs
     bleiben in ihren Fenstern -- und antwortet mit derselben Kennung.
  3. Hier wird gewartet, bis Gyro ok, Bucht erkannt und Lokalisierung ok sind.
     Klappt das nicht, startet der Fahrknoten gar nicht.

Aufrufen VOR dem Anlegen des eigenen Knotens: dessen gelatchte Abos
(/corner_geometry, ...) bekaemen sonst noch die Werte des alten scan_processor.

Abschalten fuer einen einzelnen Start: --ros-args -p schaetzung_neustart:=false
"""
import os
import sys
import time
import uuid

import rclpy
import rclpy.logging
from rclpy.qos import DurabilityPolicy, QoSProfile
from std_msgs.msg import Bool, String

from robot_msgs.msg import CornerGeometry

WORKSPACE = os.environ.get('ROBOT_WORKSPACE', '/workspace')
ANFRAGE = os.path.join(WORKSPACE, '.schaetzung_neustart_anfrage')
ANTWORT = os.path.join(WORKSPACE, '.schaetzung_neustart_antwort')

ANTWORT_TIMEOUT = 25.0     # Waechter: Knoten beenden (bis ~8 s) und neu starten
BEREIT_TIMEOUT = 40.0      # danach: Gyro, Bucht, Lokalisierung


def _abgeschaltet(argv):
    for a in argv:
        k = a.replace(' ', '').lower()
        if k in ('schaetzung_neustart:=false', 'schaetzung_neustart:=0'):
            return True
    return False


def _entfernen(pfad):
    try:
        os.remove(pfad)
    except FileNotFoundError:
        pass


def neu_starten(knoten, argv=None, scan_args=''):
    """Neustart anstossen und warten, bis beide bereit sind.

    Beendet den Prozess (SystemExit 1), wenn der Waechter nicht antwortet oder
    Gyro/Bucht nicht rechtzeitig stehen -- dann soll der Fahrknoten nicht los.

    scan_args: zusaetzliche ROS-Argumente fuer den scan_processor (z. B.
    '-p start_from_bay:=false -p start_gerade:=CCW' fuer den Einpark-Test).
    Der Waechter laesst nur harmlose Zeichen durch.
    """
    argv = sys.argv if argv is None else argv
    log = rclpy.logging.get_logger(knoten)
    if _abgeschaltet(argv):
        log.warn('schaetzung_neustart:=false -- EKF und scan_processor laufen '
                 'weiter wie sie sind (Karte vom letzten Start!).')
        return

    kennung = uuid.uuid4().hex[:8]
    _entfernen(ANTWORT)
    with open(ANFRAGE + '.tmp', 'w') as f:
        f.write(kennung + '\n' + scan_args.replace('\n', ' ') + '\n')
    os.replace(ANFRAGE + '.tmp', ANFRAGE)
    log.info('EKF und scan_processor werden neu gestartet (Fenster 8/9) ...')

    t0 = time.monotonic()
    antwort = None
    while time.monotonic() - t0 < ANTWORT_TIMEOUT:
        try:
            with open(ANTWORT) as f:
                teile = f.read().split()
        except FileNotFoundError:
            teile = []
        if len(teile) >= 2 and teile[1] == kennung:
            antwort = teile[0]
            break
        time.sleep(0.1)
    _entfernen(ANTWORT)
    if antwort is None:
        _entfernen(ANFRAGE)
        log.fatal('Der Neustart-Waechter antwortet nicht (tmux-Fenster 11 '
                  '"neustart", src/schaetzung_waechter.sh). Nicht gestartet. '
                  'Ohne Neustart: -p schaetzung_neustart:=false')
        raise SystemExit(1)
    if antwort != 'ok':
        log.fatal('Neustart von EKF/scan_processor fehlgeschlagen -- siehe '
                  'Fenster 11. Nicht gestartet.')
        raise SystemExit(1)

    # Erst JETZT abonnieren: die alten Knoten sind beendet, ihre gelatchten
    # Nachrichten mit ihnen verschwunden.
    n = rclpy.create_node(knoten + '_neustart_warten')
    q = QoSProfile(depth=1)
    q.durability = DurabilityPolicy.TRANSIENT_LOCAL
    st = {'gyro': None, 'karte': False, 'lok': None}
    n.create_subscription(Bool, '/ekf/gyro_ok', lambda m: st.update(gyro=m.data), q)
    n.create_subscription(CornerGeometry, '/corner_geometry',
                          lambda m: st.update(karte=True), q)
    n.create_subscription(String, '/localization_state',
                          lambda m: st.update(lok=m.data), q)
    try:
        t0 = time.monotonic()
        while time.monotonic() - t0 < BEREIT_TIMEOUT:
            rclpy.spin_once(n, timeout_sec=0.2)
            if st['gyro'] and st['karte'] and st['lok'] == 'ok':
                log.info('Bereit nach %.0f s: Gyro ok, Bucht erkannt, Lokalisierung ok.'
                         % (time.monotonic() - t0))
                return
        log.fatal('NICHT bereit nach %.0f s: Gyro %s, Karte %s, Lokalisierung %s. '
                  'Nicht gestartet.'
                  % (BEREIT_TIMEOUT,
                     {None: 'keine Meldung', True: 'ok', False: 'AUSGEFALLEN'}[st['gyro']],
                     'da' if st['karte'] else 'fehlt (steht er in der Luecke?)',
                     st['lok'] or '-'))
        raise SystemExit(1)
    finally:
        n.destroy_node()
