#!/bin/bash
# EKF und scan_processor frisch starten -- vor JEDEM Lauf.
#
# Warum: die Karte haengt an der Pose, bei der ekf_node startet, und der
# scan_processor misst Richtung und Lage in der Bucht nur beim Start
# (start_from_bay). Wer den Roboter nach dem letzten Lauf wieder in die Luecke
# stellt, braucht also beide neu. Von Hand ging das mehrmals schief: ein
# zweiter scan_processor lief neben dem alten weiter (parken_test_14, Karte
# mitten in der Kurve neu gelatcht), oder der Neustart wurde vergessen.
#
# Ablauf:
#   1. Ctrl-C in die Fenster 8 und 9, danach alles beenden, was sonst noch als
#      ekf_node oder scan_processor im Container laeuft (auch von Hand
#      gestartete).
#   2. Beide in Fenster 8 und 9 neu starten, EKF zuerst.
#   3. Warten, bis der Gyro ok meldet und der scan_processor die Bucht erkannt
#      hat (/corner_geometry da, Lokalisierung ok). Erst dann Rueckgabe 0 --
#      in Fenster 6 startet der Regler also nur, wenn beides steht.
#
# Aufruf:
#   schaetzung_neustart.sh                   Neustart und warten (vor dem Lauf)
#   schaetzung_neustart.sh --ohne-warten     nur neu starten
#   schaetzung_neustart.sh --verzoegerung 10 Knoten erst nach 10 s starten
#                                            (beim Hochfahren, Hardware zuerst)
# Umgebung: RACE_MODE (obstacle), AUSPARKEN (true) -- setzt start_robot.sh.

set -u
SESSION=${SESSION:-robot_session}
CONTAINER=${CONTAINER:-yolo_dev}
RACE_MODE=${RACE_MODE:-obstacle}
AUSPARKEN=${AUSPARKEN:-true}
ROS_SETUP="source /opt/ros/humble/setup.bash && source /workspace/install/setup.bash"
WARTEN=1
VERZ=0
BEREIT_TIMEOUT=30

while [ $# -gt 0 ]; do
    case "$1" in
        --ohne-warten) WARTEN=0; shift ;;
        --verzoegerung) VERZ="$2"; shift 2 ;;
        *) echo "Unbekannte Option: $1"; exit 2 ;;
    esac
done

# Die Startbefehle stehen NUR hier -- start_robot.sh ruft dieses Skript auf.
EKF_CMD="ros2 run ekf ekf_node"
SCAN_EXTRA=${SCAN_EXTRA:-}   # vom Waechter, z. B. fuer den Einpark-Test
SCAN_CMD="ros2 run ekf scan_processor --ros-args -p race_mode:=$RACE_MODE -p start_from_bay:=$AUSPARKEN $SCAN_EXTRA"
MUSTER='lib/ekf/(ekf_node|scan_processor)( |$)'

laeuft() { docker exec "$CONTAINER" pgrep -f "$MUSTER" >/dev/null 2>&1; }

fenster() {   # Fenster anlegen, falls es fehlt
    tmux list-windows -t "$SESSION" -F '#I' 2>/dev/null | grep -qx "$1" \
        || tmux new-window -d -t "$SESSION:$1" -n "$2"
}

# --- 1. stoppen ----------------------------------------------------------
if laeuft; then
    echo "EKF/scan_processor beenden ..."
    for w in 8 9; do tmux send-keys -t "$SESSION:$w" C-c 2>/dev/null; done
    for _ in $(seq 40); do laeuft || break; sleep 0.1; done
    if laeuft; then
        # von Hand gestartet (nicht in 8/9) oder haengt: gezielt nachhelfen
        docker exec "$CONTAINER" pkill -INT -f "$MUSTER"
        for _ in $(seq 30); do laeuft || break; sleep 0.1; done
    fi
    if laeuft; then
        docker exec "$CONTAINER" pkill -KILL -f "$MUSTER"
        sleep 0.5
    fi
fi

# --- 2. starten ----------------------------------------------------------
fenster 8 ekf
fenster 9 scan
VOR=""
[ "$VERZ" != "0" ] && VOR="sleep $VERZ && "
tmux send-keys -t "$SESSION:8" \
    "docker exec -it $CONTAINER bash -c '$ROS_SETUP && $VOR$EKF_CMD'" C-m
# scan_processor 2 s nach dem EKF: er soll dessen Startpose schon sehen
tmux send-keys -t "$SESSION:9" \
    "docker exec -it $CONTAINER bash -c '$ROS_SETUP && ${VOR}sleep 2 && $SCAN_CMD'" C-m
echo "EKF (Fenster 8) und scan_processor (Fenster 9) gestartet."

[ "$WARTEN" -eq 1 ] || exit 0

# --- 3. warten, bis beide stehen -----------------------------------------
echo "Warte auf Gyro und Buchterkennung (hoechstens ${BEREIT_TIMEOUT} s) ..."
docker exec -i "$CONTAINER" bash -c "$ROS_SETUP && python3 - $BEREIT_TIMEOUT" <<'PY'
import sys, time
import rclpy
from rclpy.qos import QoSProfile, DurabilityPolicy
from std_msgs.msg import Bool, String
from robot_msgs.msg import CornerGeometry

timeout = float(sys.argv[1])
rclpy.init()
n = rclpy.create_node('schaetzung_bereit')
q = QoSProfile(depth=1)
q.durability = DurabilityPolicy.TRANSIENT_LOCAL
st = {'gyro': None, 'karte': False, 'lok': None}
n.create_subscription(Bool, '/ekf/gyro_ok', lambda m: st.update(gyro=m.data), q)
n.create_subscription(CornerGeometry, '/corner_geometry', lambda m: st.update(karte=True), q)
n.create_subscription(String, '/localization_state', lambda m: st.update(lok=m.data), q)
t0 = time.monotonic()
while time.monotonic() - t0 < timeout:
    rclpy.spin_once(n, timeout_sec=0.2)
    if st['gyro'] and st['karte'] and st['lok'] == 'ok':
        print('Bereit: Gyro ok, Bucht erkannt, Lokalisierung ok.')
        sys.exit(0)
print('NICHT bereit nach %.0f s: Gyro %s, Karte %s, Lokalisierung %s.'
      % (timeout, {None: 'keine Meldung', True: 'ok', False: 'AUSGEFALLEN'}[st['gyro']],
         'da' if st['karte'] else 'fehlt (steht er in der Luecke?)', st['lok'] or '-'))
sys.exit(1)
PY
