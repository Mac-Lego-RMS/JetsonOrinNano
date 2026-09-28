#!/bin/bash
# Neustart-Waechter: laeuft auf dem Jetson (tmux-Fenster 11) und fuehrt
# Neustart-Anfragen aus dem Container aus.
#
# round1_controller und ausparken_varianten_node wollen beim Start frische
# EKF- und scan_processor-Knoten (siehe ekf/schaetzung_neustart.py). Die laufen
# in den Fenstern 8 und 9, und an tmux kommt der Container nicht heran. Also
# legen sie eine Anfrage in den gemeinsamen Workspace, und dieses Skript startet
# die beiden ueber schaetzung_neustart.sh neu und antwortet.
#
# Umgebung: WORKSPACE, RACE_MODE, AUSPARKEN, SESSION, CONTAINER (setzt
# start_robot.sh).

set -u
WORKSPACE=${WORKSPACE:-/home/macjetson/ros2_ws}
ANFRAGE="$WORKSPACE/.schaetzung_neustart_anfrage"
ANTWORT="$WORKSPACE/.schaetzung_neustart_antwort"
HIER=$(dirname "$(readlink -f "$0")")

rm -f "$ANFRAGE" "$ANTWORT"
echo "Neustart-Waechter bereit -- wartet auf Anfragen von round1_controller /"
echo "ausparken_varianten_node ($ANFRAGE)."

while true; do
    if [ -f "$ANFRAGE" ]; then
        KENNUNG=$(head -1 "$ANFRAGE")
        # Zeile 2: Zusatzargumente fuer den scan_processor (Einpark-Test).
        # Nur harmlose Zeichen -- sie landen in einer tmux-Befehlszeile.
        EXTRA=$(sed -n 2p "$ANFRAGE" | tr -cd 'A-Za-z0-9_:=.+ -')
        rm -f "$ANFRAGE"
        echo
        echo "$(date +%T) Neustart angefragt ($KENNUNG)${EXTRA:+ -- scan_processor: $EXTRA}"
        if SCAN_EXTRA="$EXTRA" "$HIER/schaetzung_neustart.sh" --ohne-warten; then
            ERG=ok
        else
            ERG=fehler
        fi
        echo "$ERG $KENNUNG" > "$ANTWORT.tmp"
        mv "$ANTWORT.tmp" "$ANTWORT"
        echo "$(date +%T) $ERG -- der Knoten wartet jetzt selbst auf Gyro und Bucht."
    fi
    sleep 0.2
done
