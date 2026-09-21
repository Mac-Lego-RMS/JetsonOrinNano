#!/bin/bash
#
# Startet den kompletten Roboter: Container hoch, dann jeder Node in einem
# eigenen tmux-Fenster.
#
#   tmux attach -t robot_session     # reinschauen
#   Strg-b n / Strg-b <nummer>       # Fenster wechseln
#   ./start_robot.sh --calib         # zusaetzlich die Kamera-Kalibrierung
#
# Fenster: 0 lidar  1 imu  2 esp  3 camera  4 foxglove  5 overlay
#          7 fusion  8 ekf  9 scan   [10 calib, nur mit --calib]
#          6 round1 -- Kommando steht fertig da, faehrt aber erst los,
#                      wenn du dort Enter drueckst.  Strg-b 6
#
# Strg-b <ziffer> kann nur EINE Ziffer -- deshalb hat der Regler eine
# einstellige Nummer. Die Fensterliste gibt es mit Strg-b w.

set -u

# ------------------------------------------------------------------ #
# Konfiguration
# ------------------------------------------------------------------ #
CONTAINER=yolo_dev
IMAGE=my_robot_base_yolo2
SESSION=robot_session
WORKSPACE=/home/macjetson/ros2_ws

# Kamera: 1280x960 ist der native Modus der 360-Grad-USB-Kamera. NICHT auf 720
# stellen -- der Fisheye-Bildkreis ist 903 px im Durchmesser und wuerde oben und
# unten abgeschnitten. Die Kalibrierung in config/fisheye_calib.yaml gilt fuer
# genau diese Aufloesung; wer sie aendert, muss neu kalibrieren.
# Fester Kameraname per udev (/etc/udev/rules.d/99-picam.rules), genau wie
# beim rplidar. Die USB-Kamera meldet sich im Betrieb neu an und wandert dabei
# zwischen /dev/video0 und /dev/video1. Ohne festen Namen findet video_source
# sie nicht wieder UND das v4l2-ctl weiter unten scheitert still -- dann laeuft
# die Kamera mit Autobelichtung und die Farberkennung bricht weg.
#
# ACHTUNG, hier steckte bis 18.09.2026 ein Fehler: video_source (jetson-utils)
# parst aus einem "v4l2://"-URI eine NUMERISCHE Geraete-ID. Ein Symlink ohne
# Ziffer im Namen scheitert deshalb hart:
#     URI -- failed to parse V4L2 device ID from /dev/picam
#     [video] videoOptions -- failed to parse input resource URI
# Der Symlink taugt also zum FINDEN des richtigen Knotens, nicht als URI --
# fuer video_source wird er mit readlink -f auf /dev/videoN aufgeloest.
#
# Dass es am 16.09. trotzdem lief, war Zufall: der Container startete, bevor
# udev den Symlink angelegt hatte, also griff die Rueckfallebene unten. Nach
# dem naechsten Reboot existierte /dev/picam von Anfang an -- und die Kamera
# kam nicht mehr hoch.
if [ -e /dev/picam ]; then
    CAM_DEV=/dev/picam
else
    # Rueckfallebene: den Bild-Knoten (index 0) selbst suchen. index 1 ist der
    # Metadaten-Knoten derselben Kamera und liefert kein Video.
    CAM_DEV=""
    for _d in /dev/video*; do
        [ -e "$_d" ] || continue
        if [ "$(cat /sys/class/video4linux/$(basename "$_d")/index 2>/dev/null)" = "0" ]; then
            CAM_DEV="$_d"; break
        fi
    done
    CAM_DEV=${CAM_DEV:-/dev/video0}
    echo "HINWEIS: /dev/picam fehlt, benutze $CAM_DEV."
    echo "  Fuer einen festen Namen: /etc/udev/rules.d/99-picam.rules anlegen."
fi
# Symlink -> echter Geraeteknoten. v4l2-ctl ist der Pfad egal, video_source
# nicht. Schlaegt readlink fehl, bleibt der Originalpfad stehen.
CAM_NODE=$(readlink -f "$CAM_DEV" 2>/dev/null || echo "$CAM_DEV")
case "$CAM_NODE" in
    /dev/video[0-9]*) ;;
    *) echo "WARNUNG: $CAM_DEV zeigt auf '$CAM_NODE', nicht auf /dev/videoN --"
       echo "  video_source wird das nicht oeffnen koennen."
       ;;
esac
CAM_RESOURCE=v4l2://$CAM_NODE
CAM_WIDTH=1280
CAM_HEIGHT=960
CAM_FPS=15.0

# Belichtung der Fisheye-Kamera. exposure_time_absolute zaehlt in 100-us-
# Schritten, 500 sind also 50 ms. Muss unter die Frameperiode passen, sonst
# faellt die Kamera eine Stufe runter -- siehe Fenster 3.
CAM_EXPOSURE=500
CAM_SATURATION=128
CAM_GAIN=20
# Weissabgleich festnageln. Gleiche Begruendung wie bei der Belichtung: der
# Treiber steht nach Reboot oder Neuanstecken wieder auf Automatik, und der
# Farbton ist das Hauptmerkmal fuer gruen (hue 40..72). Regelt der Weissabgleich
# frei mit, wandert der Farbton mit der Szene -- und genau an der dunklen Bande,
# wo er ohnehin instabil ist, entscheidet das ueber gruen oder nicht.
CAM_WB_TEMP=4600
# NACHGEMESSEN am 16.09.2026 (Bag wb_test, 266 Frames): bei 4600 K liefert die
# WEISSE Matte B=167 G=212 R=194, also (G-R)/max = +0.084 statt 0. Der
# Nullpunkt der rg_kennzahl liegt damit im Gruenen, und die symmetrischen
# Schwellen +-rg_z_min sind in Wahrheit unsymmetrisch:
#     gruen braucht einen Farbhub von 0.150 - 0.084 = 0.066
#     rot   braucht einen Farbhub von 0.150 + 0.084 = 0.234
# Deshalb kam rot auf Entfernung als gruen heraus. Der Wert bleibt trotzdem
# stehen, korrigiert wird in Software (siehe FUSION_WEISSPUNKT weiter unten).
#
# Warum nicht einfach die Kelvinzahl richtig stellen: der Stich ist nicht
# rundum gleich, sondern laeuft ueber den Azimut von +0.046 bis +0.121 (Spanne
# 0.075, also die halbe Schwelle). Eine einzelne Kelvinzahl kann das gar nicht
# treffen -- sie wuerde nur den Mittelwert verschieben und die halbe Korrektur
# verschenken. Und sie wuerde alle anderen eingefahrenen Schwellen mitziehen,
# vor allem rg_s_min und die HSV-Bereiche fuer magenta, die dann neu
# abgestimmt werden muessten.
#
# Die Softwarekorrektur verschiebt dagegen nur die MESSUNG um den gemessenen
# Nullpunkt, nicht die Pixel -- rg_z_min, rg_s_min und rg_d_min behalten ihre
# Bedeutung und ihre Abstimmung. Sie kostet damit auch kein Rauschen.
#
# Ein Kelvin-Sweep bleibt sinnvoll, wenn der Stich kleiner werden soll (kleinere
# Korrektur = mehr Reserve). Dann: WB-Wert setzen, Matte messen, wiederholen.
# Erst danach aendern, nicht auf Verdacht.

# ------------------------------------------------------------------ #
# Lidar-Kamera-Fusion (Fenster 7). Die Werte stammen aus Messreihen am
# Aufbau, die Begruendungen stehen in src/camera_lidar_fusion/README.md.
#
# FUSION_DEBUG im Wettkampflauf auf false: das Debug-Bild ist mit Abstand der
# teuerste Posten (38 Prozent eines Kerns, obwohl es nur mit 5 Hz laeuft --
# Zeichnen und Polar-Entzerrung auf 1280x960 kosten). Ohne Debug bleiben rund
# 56 Prozent fuer Klassifikation und Bandenerkennung.
FUSION_DEBUG=true
FUSION_ZONE_FRAC=0.5      # Stimmenanteil; bei 0.5 blieben genau die echten
                          # Pylonen uebrig, bei 0.2 waren es 7 statt 2 Cluster
FUSION_RANGE_MIN=0.15     # darunter sieht das Lidar den eigenen Aufbau
FUSION_LABELS="[rot,gruen]"   # magenta erst zuschalten, wenn die Parkzone dran ist
FUSION_BAND_STEPS=360     # 180 spart 10 Prozent CPU, 2 Grad reichen fuer eine Bande
# Saettigungsschwelle relativ zur Umgebung statt absolut. Absolut ueberlappen
# Pylone und Bande hoffnungslos -- am Aufbau gemessen (106 Wolken, raw-Wolke):
#     Pylone        S=73  bei Umgebung 40  -> Verhaeltnis 1.82
#     Bande hell    S=48  bei Umgebung 50  -> 0.96
#     Bande dunkel  S=61  bei Umgebung 55  -> 1.09
#     Eigenaufbau   S=48  bei Umgebung 60  -> 0.78
# Mit 1.4 liegt die Schwelle sauber im Graben. Gegengerechnet an derselben
# Aufnahme: Fehlalarme 304 -> 69 bei 2286 von 2413 erhaltenen Pylonenpunkten,
# also Fehlerquote 11.2 -> 2.9 Prozent. Hoeher NICHT einstellen: bei 1.8 bricht
# die Pylone auf 1345 Punkte ein, weil sie selbst bei 1.82 liegt.
# Bandensuche AUS. Sie ueberschreibt die AEUSSERE Zonenkante mit der gemessenen
# Bandenunterkante -- die Pylone steht aber VOR der Bande und reicht radial
# weiter nach aussen als diese. Die Suche schnitt der Pylone also unten etwas
# ab. Das Zonenmodell in fisheye_calib.yaml ist am Rohbild ausgemessen:
#   Oberkante  r = 396.0        konstant, Objektiv auf Bandenoberkante
#   Unterkante r = 396.5 + 15.0/rho
# Gemessene Farbsignale: 0.39 m -> r bis 435, 0.85 m -> 413, 1.24 m -> 411,
# 1.93 m -> 403. Das Modell trifft alle vier.
# Absolutes Tor auf den Kanalunterschied |G-R| in Zaehlwerten. Die beiden
# relativen Tore (Saettigung, Verhaeltnis) sind gegen einen Farbstich ueber das
# Fischauge blind: ein dunkles, fast neutrales Bandenpixel BGR(30,35,25) hat
# rechnerisch S=73 und z=+0.29, obwohl der Kanalunterschied nur 10 Zaehlwerte
# betraegt. Genau daran wurden in Lauf 7 die Ostwand rot und die Westwand gruen.
# Am Bild gemessen: 87.8 % aller Abgriffe liegen bei |G-R| <= 10, die echte
# Pylone bei > 60. Mit 20 fallen drei Phantomhindernisse weg, alle echten
# bleiben; ab 30 verliert man die gruene Pylone auf 1.24 m.
# 18.09.2026 von 20 auf 16 gesenkt, ZUSAMMEN mit FUSION_WEISSPUNKT.
# Der Grund: die 20 waren selbst eine Kruecke gegen den Farbstich. Ein
# neutrales Mattenpixel bei mx=200 hat roh |G-R| = 0.09*200 = 18 -- das Tor
# musste also knapp darueber liegen. Mit abgezogenem Neutralpunkt hat ein
# neutrales Pixel |G-R| ~ 0, und das Tor darf tiefer.
#
# Bei 20 UND aktivem Weisspunkt fallen dagegen dunkle gruene Pylonen raus:
# am Aufbau gemessen B=35 G=52 R=30, also |G-R| = 22 roh, korrigiert aber
# 22 - 0.102*52 = 16.7 -- knapp unter 20.
#
# Gegen den Bag wb_test durchgefahren (219 Scans, Cluster je Scan):
#     ohne Weisspunkt, 20 : rot 3.00 | gruen echt 3.05, gestreut 0.40
#     mit  Weisspunkt, 20 : rot 3.00 | gruen echt 2.38, gestreut 0.73  <- Gruen kaputt
#     mit  Weisspunkt, 16 : rot 3.00 | gruen echt 2.99, gestreut 0.18  <- hier
#     mit  Weisspunkt, 12 : rot 3.00 | gruen echt 3.02, gestreut 0.24
# Bei 16 bleibt das echte Gruen vollstaendig erhalten und die gestreuten
# Cluster (die Phantomhindernisse) halbieren sich.
FUSION_RG_DMIN=16
FUSION_ZONE_ADAPTIV=0.0
# --- Neutralpunkt der Kamera je Azimutsektor an der weissen Matte messen ---
# Zieht den Weissabgleich-Versatz von der rg_kennzahl ab, damit eine farblose
# Flaeche wirklich z=0 ergibt und die Schwellen +-rg_z_min wieder symmetrisch
# wirken. Gemessen wird in einem Pixelring auf der Matte, knapp ausserhalb der
# Bande (automatisch: zone_r0_out + 12 px bis Bildkreisrand - 15 px).
#
# Gegen den Bag wb_test geprueft, 219 Scans mit 2 roten und 3 gruenen Pylonen:
#     rot   unveraendert 3 stabile Cluster, 22.4 -> 22.8 Punkte
#     die rote Pylone auf 2 m lieferte 13.1 Punkte faelschlich als gruen,
#     jetzt noch 5.7
#     echte gruene Pylone auf 2.55 m: 11.5 -> 12.5 Punkte
#     echte gruene Pylone auf 0.56 m: 16.1 -> 10.6 Punkte (Preis der Korrektur)
# Zum Abschalten auf false setzen -- dann verhaelt sich alles wie vorher.
FUSION_WEISSPUNKT=true
# 12 Sektoren = 30 Grad. Feiner macht die Messung je Sektor rauschiger, gröber
# verschenkt den Gang ueber den Azimut (Spanne 0.075 ueber 12 Sektoren).
FUSION_WEISSPUNKT_SEKTOREN=12
# Gruen liess bisher ab V=12 durch, waehrend alles bis V=45 als "schwarz" gilt.
# Ein per Definition schwarzes Pixel konnte also gruen gewinnen -- und der
# Schwarz-Test laeuft erst, wenn keine Farbe gewonnen hat. Die echte Pylone
# liegt bei V=91, die dunkle Bande bei V=38.
FUSION_GRUEN_VMIN=50
# Bewegungskompensation: die Lidarpunkte werden vor dem Farbabgriff in die Lage
# ZUM BILDZEITPUNKT zurueckgerechnet. Noetig, weil die Kamera im Fahrbetrieb von
# 15 auf 3 Hz einbricht und das zugeordnete Bild dann 100-700 ms alt ist. In
# Lauf 20 gemessen: Farbausbeute auf einer Pylone 38 Prozent im Stand, aber 6
# Prozent ab 0.5 rad/s und 2 Prozent ab 1 rad/s -- eine Pylone ist auf 1.6 m nur
# 1.6 Grad breit, der Peilfehler omega*dt betraegt dort schon 8.6 Grad.
# Braucht /ekf/odom. Fehlt die Pose, faerbt die Node unkorrigiert weiter und
# schreibt das in die Sync-Zeile.
FUSION_BEWEGUNGSKOMP=true

# Lidar-Topic. sllidar_s3_launch.py publiziert auf /scan.
SCAN_TOPIC=/scan

# Wettbewerbsmodus fuer scan_processor:
#   obstacle  Hinderniswettbewerb -- volle Karte fuer die erkannte
#             Startposition, Pylonenerkennung aus der Farbwolke.
#   open      offener Wettbewerb -- reduzierte 3-Wand-Startkarte, die
#             Spurbreiten werden gelernt. Keine Pylonenerkennung.
RACE_MODE=obstacle

# Wie viele Ecken der Regler faehrt, bevor er anhaelt. 12 sind drei
# Runden. Nur fuer die vorbereitete Zeile in Fenster 6.
N_CORNERS=12

# Startet der Roboter in der Parkluecke?
#
# EIN Schalter fuer zwei Knoten, absichtlich: er setzt den Regler auf
# Ausparken UND sagt dem scan_processor, dass er seine Fahrtrichtung nicht
# selbst latchen soll. Aus der Luecke heraus sieht der naemlich keine
# brauchbare Ecke, liefert aber trotzdem ein Ergebnis -- in einem Lauf stand
# dort CCW, waehrend das Ausparken CW gemessen hatte. Wer sich irrt, faehrt
# die ganze Runde andersherum.
#
# Warum der scan_processor das als PARAMETER braucht und nicht per Topic
# erfaehrt: er startet hier beim Hochfahren, der Regler erst, wenn du in
# Fenster 6 Enter drueckst. Ein "warte mal" von ihm kaeme immer zu spaet.
# Die Richtung selbst kommt dann sehr wohl ueber ein Topic
# (/parking_direction), nur eben das Warten nicht.
AUSPARKEN=true

# Kalibrier-Node nur auf Wunsch (--calib). Im normalen Lauf nicht gebraucht.
START_CALIB=0
[ "${1:-}" = "--calib" ] && START_CALIB=1

# Farbe der Kalibrierpylone: rot, gruen oder magenta. Ohne diese Vorgabe nimmt
# die Node den groessten Farbfleck im Bild -- und das ist im moeblierten Raum
# fast nie die Pylone.
CALIB_TARGET_COLOR=gruen

ROS_SETUP="source /opt/ros/humble/setup.bash && source /workspace/install/setup.bash"

# ------------------------------------------------------------------ #
# Hilfsfunktion: ein Kommando in einem eigenen tmux-Fenster im Container
# ------------------------------------------------------------------ #
run_window() {
    local index="$1" name="$2" cmd="$3"

    if [ "$index" -eq 0 ]; then
        tmux rename-window -t "$SESSION:0" "$name"
    else
        tmux new-window -t "$SESSION:$index" -n "$name"
    fi
    tmux send-keys -t "$SESSION:$index" \
        "docker exec -it $CONTAINER bash -c '$ROS_SETUP && $cmd'" C-m
}

# Wie run_window, aber das Kommando wird nur in die Zeile GESCHRIEBEN und
# nicht abgeschickt. Fuer alles, was den Roboter in Bewegung setzt: erst
# hinsehen, dann Enter.
arm_window() {
    local index="$1" name="$2" cmd="$3"

    tmux new-window -t "$SESSION:$index" -n "$name"
    # Kurz warten, bis die Shell ihre Eingabezeile hat. Ohne das echot das
    # Terminal den Text einmal roh und bash danach noch einmal -- das
    # Kommando stuende dann doppelt im Fenster.
    sleep 0.7
    tmux send-keys -t "$SESSION:$index" C-l
    tmux send-keys -t "$SESSION:$index" \
        "docker exec -it $CONTAINER bash -c '$ROS_SETUP && $cmd'"
}

# ------------------------------------------------------------------ #
# 1. Maximale Leistung erzwingen (verhindert UART-Latenzen durch CPU-Sleep)
# jetson_clocks braucht root. Ohne sudo-Rechte NICHT nachfragen, sonst haengt
# ein automatischer Start am Passwort-Prompt -- stattdessen deutlich sagen,
# dass die Taktbremse aktiv bleibt.
# ------------------------------------------------------------------ #
if [ "$(id -u)" -eq 0 ]; then
    /usr/bin/jetson_clocks
elif sudo -n /usr/bin/jetson_clocks 2>/dev/null; then
    echo "jetson_clocks gesetzt (via sudo)."
else
    echo "HINWEIS: jetson_clocks uebersprungen (braucht root). Fuer volle Leistung:"
    echo "  sudo /usr/bin/jetson_clocks"
    echo "  oder dauerhaft:  echo \"$USER ALL=(root) NOPASSWD: /usr/bin/jetson_clocks\" | sudo tee /etc/sudoers.d/jetson_clocks"
fi

# ------------------------------------------------------------------ #
# 2. Alte Container & alte Terminals aufraeumen
# ------------------------------------------------------------------ #
docker rm -f "$CONTAINER" 2>/dev/null
tmux kill-session -t "$SESSION" 2>/dev/null

# ------------------------------------------------------------------ #
# 2b. Auf die Systemuhr warten
# Ohne gueltige RTC-Zeit startet die Jetson bei 1970. Springt die Uhr dann
# per NTP mitten im Betrieb 56 Jahre nach vorn, ist die Foxglove-Timeline
# zerrissen: die Panels zeigen nichts mehr, obwohl jedes Topic sauber
# publiziert. Also erst die Zeit, dann die Nodes.
# Nicht ewig warten: ohne Netz (Wettkampf) kommt nie ein NTP-Sync, und
# fahren muss der Roboter trotzdem.
# ------------------------------------------------------------------ #
synced() { [ "$(timedatectl show -p NTPSynchronized --value 2>/dev/null)" = "yes" ]; }

if synced; then
    echo "Uhr synchron: $(date '+%F %T')."
else
    echo -n "Warte auf Zeitsynchronisation"
    for _ in $(seq 30); do
        synced && break
        echo -n "."
        sleep 1
    done
    if synced; then
        echo " -- Uhr steht auf $(date '+%F %T')."
    else
        echo
        echo "HINWEIS: keine NTP-Synchronisation, Systemzeit ist $(date '+%F %T')."
        echo "  Springt die Uhr spaeter, Foxglove neu verbinden und diesen"
        echo "  Stack neu starten -- sonst bleiben die Panels leer."
    fi
fi

# ------------------------------------------------------------------ #
# 3. Container starten
# ------------------------------------------------------------------ #
/usr/local/bin/jetson-containers run -d \
  --name "$CONTAINER" \
  --privileged \
  -v /dev:/dev \
  --volume /tmp/argus_socket:/tmp/argus_socket \
  --shm-size=2g \
  --runtime nvidia \
  -v "$WORKSPACE":/workspace \
  -w /workspace \
  "$IMAGE" \
  sleep infinity

# Warten bis der Container wirklich antwortet, statt blind zu schlafen.
echo -n "Warte auf Container"
for _ in $(seq 30); do
    if docker exec "$CONTAINER" true 2>/dev/null; then
        echo " -- da."
        break
    fi
    echo -n "."
    sleep 1
done

# Pruefen, ob der Workspace gebaut ist -- sonst starten alle Fenster ins Leere.
if ! docker exec "$CONTAINER" test -f /workspace/install/setup.bash 2>/dev/null; then
    echo "FEHLER: /workspace/install/setup.bash fehlt. Erst bauen:"
    echo "  docker exec -it $CONTAINER bash -c 'source /opt/ros/humble/setup.bash && cd /workspace && colcon build --symlink-install'"
    exit 1
fi

# ------------------------------------------------------------------ #
# 3b. jetson-stats im Container nachruesten -- Quelle fuer die /jtop/*-Topics
# der Foxglove-Overlay-Node (Kernauslastung, Temperaturen, GPU, RAM, Watt).
#
# Zwei Dinge muessen dafuer stimmen, beide deckt dieser Block ab:
#   * Das Python-Paket im Container. Der hat keinen Netzzugang, also offline
#     aus tools/jtop_wheels -- dauert etwa eine Sekunde.
#   * Der Socket /run/jtop.sock. Den mountet jetson-containers/run.sh von
#     selbst, ABER nur wenn er beim Containerstart schon existiert. Also
#     immer erst jtop.service, dann dieses Skript.
# ------------------------------------------------------------------ #
if [ -S /run/jtop.sock ]; then
    if docker exec "$CONTAINER" pip3 install -q --no-index \
            --find-links /workspace/tools/jtop_wheels jetson-stats 2>/dev/null; then
        echo 'jetson-stats im Container bereit -- /jtop/* wird publiziert.'
    else
        echo 'HINWEIS: jetson-stats liess sich nicht installieren.'
        echo '  Wheels liegen in tools/jtop_wheels; /jtop/* bleibt solange still.'
    fi
else
    echo 'HINWEIS: /run/jtop.sock fehlt -- jtop.service laeuft nicht.'
    echo '  sudo systemctl start jtop.service   und dieses Skript neu starten,'
    echo '  sonst bleiben die /jtop/*-Topics aus (der Rest laeuft normal).'
fi

# Hardware Initialisierungszeit geben
sleep 10

# ------------------------------------------------------------------ #
# 4. Virtuelle Terminals (tmux) starten
# ------------------------------------------------------------------ #
tmux new-session -d -s "$SESSION"

# Fenster 0: Lidar
run_window 0 lidar "ros2 launch sllidar_ros2 sllidar_s3_launch.py"

# Fenster 1: IMU
run_window 1 imu "ros2 run bno055 bno055 --ros-args --params-file /workspace/bno055_params.yaml"

# Fenster 2: ESP Serial
run_window 2 esp "ros2 run esp_bridge esp_serial_bridge"

# Fenster 3: Kamera (USB UVC, 360-Grad-Fisheye)
run_window 3 camera \
    "/workspace/install/ros_deep_learning/lib/ros_deep_learning/video_source --ros-args -p resource:=$CAM_RESOURCE -p width:=$CAM_WIDTH -p height:=$CAM_HEIGHT -p framerate:=$CAM_FPS"

# Belichtung festnageln -- MUSS bei jedem Start neu passieren. V4L2-Controls
# leben im Kerneltreiber und stehen nach Reboot oder Neuanstecken wieder auf
# Werkseinstellung (auto_exposure=3, also Auto).
#
# Mit Autobelichtung regelt die Kamera auf das Hellste im Bild, meist ein
# Fenster. Der Bildrand -- und damit genau der Horizontring, auf dem
# lidar_pixel_mapper abgreift -- saeuft dabei ab: gemessen V-Median 26 von 255.
# Eine gruene Pylone kam dort auf V=37 und fiel damit unter die Schwelle von
# 45; find_color_blob() fand ueberhaupt nichts mehr.
#
# Feste 50 ms heben den Ring von V=26 auf V=55 -- der Blob wird mit den
# Standardschwellen wieder sauber erkannt (0 -> rund 1350 px, die Mindestflaeche
# liegt bei 300). saturation hoch, weil Aufhellen Farbe kostet.
#
# gamma bleibt bewusst neutral: es hebt zwar die Helligkeit, frisst aber die
# Saettigung (gamma=180 drueckte die Saettigung einer Pylone von 103 auf 47)
# und macht die Farberkennung damit kaputt.
#
# gain=20 dagegen zahlt sich aus -- am fertigen colored_scan gemessen, Median
# gruener Punkte je Scan ueber jeweils 111 Scans:
#     gain  0 -> 6     gain 20 -> 16     gain 30 -> 16     gain 40 -> 16
# Ueber 20 bringt es nichts mehr, weit darueber brennt das Bild aus und frisst
# dann doch die Saettigung. Rot bleibt ueber den ganzen Bereich stabil bei 21
# bis 23 Punkten. gain kostet keine Framerate, anders als die Belichtungszeit.
#
# WARUM GENAU 50 ms: die Belichtung MUSS unter die Frameperiode passen. UVC
# kennt nur feste Stufen (30/15/10/7.5/5 fps); passt die Zeit nicht mehr in
# 1/15 s = 66.7 ms, faellt die Kamera auf 7.5 fps. Gemessen bei CAM_FPS=15:
#     70 ms -> 8.1 fps     60 ms -> 8.4 fps     50 ms -> 13.2 fps
# Nach unten ist bei 30 ms Schluss, da findet die Farberkennung nichts mehr
# (Blob 0 px). Das Fenster ist also schmal -- wer mehr Licht braucht, macht
# mehr Licht in den Raum, statt an gain oder gamma zu drehen.
#
# Im Fahrbetrieb verschmieren 50 ms die Pylonen weiterhin etwas. Wenn das
# stoert: CAM_EXPOSURE runter UND fuer echte Beleuchtung sorgen.
#
# Erst nach dem Start von video_source setzen, damit das Geraet offen ist.
sleep 5
if docker exec "$CONTAINER" v4l2-ctl -d "$CAM_NODE" \
        -c auto_exposure=1 \
        -c exposure_time_absolute=$CAM_EXPOSURE \
        -c saturation=$CAM_SATURATION \
        -c white_balance_automatic=0 \
        -c white_balance_temperature=$CAM_WB_TEMP \
        -c gamma=100 -c gain=$CAM_GAIN -c contrast=32 -c brightness=0 2>/dev/null; then
    echo "Kamera ($CAM_NODE): feste Belichtung $((CAM_EXPOSURE / 10)) ms, Saettigung $CAM_SATURATION, gain $CAM_GAIN, Weissabgleich fest $CAM_WB_TEMP K."
else
    echo "HINWEIS: v4l2-ctl fehlgeschlagen -- Kamera laeuft mit Autobelichtung."
    echo "  Der Bildrand saeuft dann ab und die Farberkennung findet keine Pylonen."
fi

# Fenster 4: Foxglove
run_window 4 foxglove "ros2 run foxglove_bridge foxglove_bridge"
run_window 5 foxglove "ros2 run ekf foxglove_overlay"



# Fenster 7: Lidar-Kamera-Fusion (Farbe je Lidar-Punkt)
# Fenster 6 bleibt frei -- dort startet --calib die Kalibrier-Node.
#
# zone_from_band: die Abgriffszone haengt an der live erkannten Unterkante
# der schwarzen Bande, die Oberkante bleibt konstant (das Objektiv sitzt auf
# ihrer Hoehe). Wo keine Bande gefunden wird, greift die Zonenkurve aus
# config/fisheye_calib.yaml -- die Suche kann also nur verbessern.
#
# Die Zonenkurve muss einmal kalibriert werden ("zone" und "zonefit" in
# rotation_calibration, siehe README). Fehlt sie, rechnet die Node aus cam_z
# und Brennweite -- das trifft nicht gut genug, weil das equidistante Modell
# am Bildrand um 9 bis 12 px danebenliegt.
#
# CSV auf Zuruf:  ros2 topic pub --once /camera_lidar/capture std_msgs/msg/Empty {}
# Anzeige aus:    ros2 param set /lidar_pixel_mapper debug false
run_window 7 fusion \
    "sleep 8 && ros2 run camera_lidar_fusion lidar_pixel_mapper --ros-args \
       -p scan_topic:=$SCAN_TOPIC \
       -p zone_from_band:=false \
       -p sample_zone_min_frac:=$FUSION_ZONE_FRAC \
       -p range_min_m:=$FUSION_RANGE_MIN \
       -p active_labels:=$FUSION_LABELS \
       -p band_steps:=$FUSION_BAND_STEPS \
       -p sample_zone_adaptiv:=$FUSION_ZONE_ADAPTIV \
       -p rg_d_min:=$FUSION_RG_DMIN \
       -p weisspunkt:=$FUSION_WEISSPUNKT \
       -p weisspunkt_sektoren:=$FUSION_WEISSPUNKT_SEKTOREN \
       -p color.gruen.v_min:=$FUSION_GRUEN_VMIN \
       -p motion_compensation:=$FUSION_BEWEGUNGSKOMP \
       -p debug:=$FUSION_DEBUG \
       -p csv_mode:=trigger"

# Fenster 10: Kamera-Rotationskalibrierung -- nur mit ./start_robot.sh --calib
# Ablauf: circle -> mehrmals sample -> solve -> verify -> save
#   ros2 topic pub --once /camera_lidar/calib_cmd std_msgs/msg/String "data: circle"
if [ "$START_CALIB" -eq 1 ]; then
    run_window 10 calib \
        "sleep 8 && ros2 run camera_lidar_fusion rotation_calibration --ros-args -p scan_topic:=$SCAN_TOPIC -p target_label:=$CALIB_TARGET_COLOR"
fi

# ------------------------------------------------------------------ #
# Fenster 8/9: Zustandsschaetzung und Wahrnehmung
# ------------------------------------------------------------------ #
# Ohne ekf_node steht der Roboter: die Brueckenregelung braucht
# /ekf/odom als Ist-Geschwindigkeit und schaltet den Motor ab, wenn die
# Pose aelter als odom_stop_s (0,5 s) ist. Auch die Bewegungskompensation
# der Fusion (Fenster 7) haengt daran.
#
# Beide erst nach der Hardware starten -- ekf_node will Radstellungen von
# der Bruecke und Drehraten von der IMU, scan_processor den Lidar.
run_window 8 ekf   "sleep 10 && ros2 run ekf ekf_node"
run_window 9 scan  "sleep 12 && ros2 run ekf scan_processor --ros-args -p race_mode:=$RACE_MODE -p wait_for_parking:=$AUSPARKEN"

# ------------------------------------------------------------------ #
# Fenster 6: der Regler -- vorbereitet, aber NICHT gestartet
# ------------------------------------------------------------------ #
# Der round1_controller faehrt los, sobald er Eingaben hat. Deshalb steht
# sein Kommando hier nur fertig in der Zeile: hinsehen, ob der Roboter
# richtig steht, dann Enter. Einstellige Fensternummer, damit Strg-b 6
# hinfuehrt.
arm_window 6 round1 \
    "ros2 run ekf round1_controller --ros-args -p n_corners:=$N_CORNERS -p ausparken:=$AUSPARKEN"

# ------------------------------------------------------------------ #
# Optionale Fahr-Nodes -- bei Bedarf einkommentieren
# ------------------------------------------------------------------ #
#run_window 7 obstacle "ros2 run robot_vision obstacle_run"
#run_window 7 wallfollower "ros2 run wall_follower_robot wall_follower_logic"

echo
echo "Alles gestartet. Reinschauen mit:  tmux attach -t $SESSION"
echo
echo "Fenster 6 (round1) haelt das Reglerkommando bereit, ausgefuehrt ist es"
echo "NICHT. Zum Losfahren:  tmux attach -t $SESSION, dann Strg-b 6,"
echo "hinsehen, Enter.  (Strg-b w zeigt alle Fenster.)"
echo
echo "Anhalten von aussen:"
echo "  ros2 topic pub --once /cmd_vel geometry_msgs/msg/Twist \"{linear: {x: 0.0}, angular: {z: 0.0}}\""
echo "Waehrend einer AUSPARK-Fahrt wirkt /cmd_vel nicht -- die Fahrt gehoert"
echo "dann dem ESP. Dort hilft nur der Nothalt:"
echo "  ros2 topic pub --once /esp_serial_bridge/emergency std_msgs/msg/Empty \"{}\""
tmux list-windows -t "$SESSION" -F "  Fenster #{window_index}: #{window_name}"
