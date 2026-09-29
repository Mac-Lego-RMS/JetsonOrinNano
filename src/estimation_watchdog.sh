#!/bin/bash
# Restart watchdog: runs on the Jetson (tmux window 11) and carries out
# restart requests from the container.
#
# round1_controller and unpark_variants_node want fresh EKF and
# scan_processor nodes at start-up (see ekf/estimation_restart.py). Those run
# in windows 8 and 9, and the container cannot reach tmux. So they put a
# request into the shared workspace, and this script restarts the two through
# estimation_restart.sh and replies.
#
# Environment: WORKSPACE, RACE_MODE, UNPARK, SESSION, CONTAINER (set by
# start_robot.sh).

set -u
WORKSPACE=${WORKSPACE:-/home/macjetson/ros2_ws}
REQUEST_FILE="$WORKSPACE/.estimation_restart_request"
REPLY_FILE="$WORKSPACE/.estimation_restart_reply"
HERE=$(dirname "$(readlink -f "$0")")

rm -f "$REQUEST_FILE" "$REPLY_FILE"
echo "Restart watchdog ready -- waiting for requests from round1_controller /"
echo "unpark_variants_node ($REQUEST_FILE)."

while true; do
    if [ -f "$REQUEST_FILE" ]; then
        REQ_ID=$(head -1 "$REQUEST_FILE")
        # Line 2: extra arguments for the scan_processor (parking test).
        # Only harmless characters -- they end up in a tmux command line.
        EXTRA=$(sed -n 2p "$REQUEST_FILE" | tr -cd 'A-Za-z0-9_:=.+ -')
        rm -f "$REQUEST_FILE"
        echo
        echo "$(date +%T) Restart requested ($REQ_ID)${EXTRA:+ -- scan_processor: $EXTRA}"
        if SCAN_EXTRA="$EXTRA" "$HERE/estimation_restart.sh" --no-wait; then
            RESULT=ok
        else
            RESULT=error
        fi
        echo "$RESULT $REQ_ID" > "$REPLY_FILE.tmp"
        mv "$REPLY_FILE.tmp" "$REPLY_FILE"
        echo "$(date +%T) $RESULT -- the node now waits for gyro and bay itself."
    fi
    sleep 0.2
done
