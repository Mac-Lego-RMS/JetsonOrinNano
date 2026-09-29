#!/bin/bash
# Restart EKF and scan_processor fresh -- before EVERY run.
#
# Why: the map is tied to the pose at which ekf_node starts, and the
# scan_processor measures direction and pose in the bay only at start-up
# (start_from_bay). Whoever puts the robot back into the bay after the last
# run therefore needs both restarted. By hand this went wrong several times: a
# second scan_processor kept running next to the old one (parken_test_14, map
# re-latched in the middle of the corner), or the restart was forgotten.
#
# Sequence:
#   1. Ctrl-C into windows 8 and 9, then stop everything else that still runs
#      as ekf_node or scan_processor in the container (also ones started by
#      hand).
#   2. Restart both in windows 8 and 9, EKF first.
#   3. Wait until the gyro reports ok and the scan_processor has detected the
#      bay (/corner_geometry there, localisation ok). Only then return 0 --
#      so the controller in window 6 only starts when both are up.
#
# Usage:
#   estimation_restart.sh                   restart and wait (before the run)
#   estimation_restart.sh --no-wait         only restart
#   estimation_restart.sh --delay 10        start the nodes only after 10 s
#                                           (at boot, hardware first)
# Environment: RACE_MODE (obstacle), UNPARK (true) -- set by start_robot.sh.

set -u
SESSION=${SESSION:-robot_session}
CONTAINER=${CONTAINER:-yolo_dev}
RACE_MODE=${RACE_MODE:-obstacle}
UNPARK=${UNPARK:-true}
# OPENBLAS_NUM_THREADS=1: otherwise numpy (OpenBLAS) starts a worker thread
# already for np.linalg.inv on a 2x2 matrix, which then busy-waits. In the
# ekf_node (inv on every wall hit, ~14 Hz) it permanently burned 0.9
# cores -- measured at standstill, for no benefit at all. With 1 thread: 0 %.
ROS_SETUP="export OPENBLAS_NUM_THREADS=1 && source /opt/ros/humble/setup.bash && source /workspace/install/setup.bash"
WAIT=1
DELAY=0
READY_TIMEOUT=30

while [ $# -gt 0 ]; do
    case "$1" in
        --no-wait) WAIT=0; shift ;;
        --delay) DELAY="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 2 ;;
    esac
done

# The start commands live ONLY here -- start_robot.sh calls this script.
EKF_CMD="ros2 run ekf ekf_node"
SCAN_EXTRA=${SCAN_EXTRA:-}   # from the watchdog, e.g. for the parking test
SCAN_CMD="ros2 run ekf scan_processor --ros-args -p race_mode:=$RACE_MODE -p start_from_bay:=$UNPARK $SCAN_EXTRA"
PATTERN='lib/ekf/(ekf_node|scan_processor)( |$)'

running() { docker exec "$CONTAINER" pgrep -f "$PATTERN" >/dev/null 2>&1; }

window() {   # create the window if it is missing
    tmux list-windows -t "$SESSION" -F '#I' 2>/dev/null | grep -qx "$1" \
        || tmux new-window -d -t "$SESSION:$1" -n "$2"
}

# --- 1. stop -------------------------------------------------------------
if running; then
    echo "Stopping EKF/scan_processor ..."
    for w in 8 9; do tmux send-keys -t "$SESSION:$w" C-c 2>/dev/null; done
    for _ in $(seq 40); do running || break; sleep 0.1; done
    if running; then
        # started by hand (not in 8/9) or hanging: help it along directly
        docker exec "$CONTAINER" pkill -INT -f "$PATTERN"
        for _ in $(seq 30); do running || break; sleep 0.1; done
    fi
    if running; then
        docker exec "$CONTAINER" pkill -KILL -f "$PATTERN"
        sleep 0.5
    fi
fi

# --- 2. start ------------------------------------------------------------
window 8 ekf
window 9 scan
PREFIX=""
[ "$DELAY" != "0" ] && PREFIX="sleep $DELAY && "
tmux send-keys -t "$SESSION:8" \
    "docker exec -it $CONTAINER bash -c '$ROS_SETUP && $PREFIX$EKF_CMD'" C-m
# scan_processor 2 s after the EKF: it should already see the EKF's start pose
tmux send-keys -t "$SESSION:9" \
    "docker exec -it $CONTAINER bash -c '$ROS_SETUP && ${PREFIX}sleep 2 && $SCAN_CMD'" C-m
echo "EKF (window 8) and scan_processor (window 9) started."

[ "$WAIT" -eq 1 ] || exit 0

# --- 3. wait until both are up -------------------------------------------
echo "Waiting for gyro and bay detection (at most ${READY_TIMEOUT} s) ..."
docker exec -i "$CONTAINER" bash -c "$ROS_SETUP && python3 - $READY_TIMEOUT" <<'PY'
import sys, time
import rclpy
from rclpy.qos import QoSProfile, DurabilityPolicy
from std_msgs.msg import Bool, String
from robot_msgs.msg import CornerGeometry

timeout = float(sys.argv[1])
rclpy.init()
n = rclpy.create_node('estimation_ready')
q = QoSProfile(depth=1)
q.durability = DurabilityPolicy.TRANSIENT_LOCAL
st = {'gyro': None, 'map': False, 'loc': None}
n.create_subscription(Bool, '/ekf/gyro_ok', lambda m: st.update(gyro=m.data), q)
n.create_subscription(CornerGeometry, '/corner_geometry', lambda m: st.update(map=True), q)
n.create_subscription(String, '/localization_state', lambda m: st.update(loc=m.data), q)
t0 = time.monotonic()
while time.monotonic() - t0 < timeout:
    rclpy.spin_once(n, timeout_sec=0.2)
    if st['gyro'] and st['map'] and st['loc'] == 'ok':
        print('Ready: gyro ok, bay detected, localisation ok.')
        sys.exit(0)
print('NOT ready after %.0f s: gyro %s, map %s, localisation %s.'
      % (timeout, {None: 'no message', True: 'ok', False: 'FAILED'}[st['gyro']],
         'present' if st['map'] else 'missing (is it standing in the bay?)', st['loc'] or '-'))
sys.exit(1)
PY
