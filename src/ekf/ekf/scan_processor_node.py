#!/usr/bin/env python3
"""
scan_processor node: start detection, map management, and perception outputs
for both challenges. race_mode ('obstacle' | 'open') is a ROS parameter.

  obstacle: commits the full generated field map for the detected position
            (direction defaults to CW, resolved at the first corner). Detects
            traffic signs from the colour-classified cloud.
  open:     inner-band geometry is unknown, so it commits a reduced 3-wall
            start map, computes the exact field start pose from the measured
            distances at the direction latch, learns each straight's lane width
            and reconstructs the inner band. No obstacle detection.

Two layers of obstacle output, mirroring the wall outputs:
  /obstacles_live  raw, per scan, base_link frame -- for REACTING. Needs no map,
                   so it works from the first scan.
  /obstacles       snapped to the seat grid, accumulated, map frame -- for
                   PLANNING. The grid needs the start pose, so it only exists
                   after the direction latch; detections from before that are
                   BUFFERED with their pose and replayed when the grid is
                   built, so start-straight obstacles are not lost.

Die Fahrtrichtung kommt normalerweise aus der Eckengeometrie. Wird aus einer
Parkluecke gestartet, ist sie DORT sicherer zu bestimmen (die nahe Seite ist
die Aussenwand, die ferne das Spielfeld); der Regler schickt sie dann auf
/parking_direction, und die gilt. Kommt dort nichts, bleibt alles beim Alten.

Subscribes: /scan, /ekf/odom, /round1_controller/lap_state (latched),
            /camera_lidar/colored_scan (obstacle mode only),
            /parking_direction (latched, optional)
Publishes:  /wall_matches
            /wall_distances    live [left, right] side distances, NaN if unseen
            /obstacles_live    raw obstacles, base_link frame, every scan
            /front_wall_x      (latched) front wall x in the map frame
            /race_direction    (latched) CW / CCW, latched once, then frozen
            /corner_geometry   (latched) outer box, at the direction latch
            /inner_geometry    (latched) inner band
            /obstacles         (latched) accumulated obstacle set, map frame
"""
import numpy as np
import time
from collections import Counter, deque

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan, PointCloud2
from nav_msgs.msg import Odometry

from robot_msgs.msg import (WallMatch, WallMatchArray, CornerGeometry, WallHNF,
                            Obstacle, ObstacleArray)

from std_msgs.msg import Float64, String, Int32MultiArray, Float64MultiArray
from rclpy.qos import QoSProfile, DurabilityPolicy

from geometry_msgs.msg import Point

from ekf.ekf import wrap
from ekf.direction_detection import detect_direction
from ekf.obstacle_detection import (detect_obstacles, mask_sectors,
                                    sector_from_robot_point, PILLAR_HALF_WIDTH)
from ekf.obstacle_map import ObstacleMap, MIN_SEAT_VOTES
from ekf.wall_extraction import (
    LIDAR_OFFSET_X, BLOCK_ANGLE,
    scan_to_points, cluster_points, merge_wraparound, split_at_corners,
    fit_wall_hnf, lidar_to_base_link, match_walls,
)
from ekf.field_map import (
    generate_map, start_map_3wall, outer_box_map, outer_walls_map,
    inner_walls_map, inner_band_from_widths, obstacle_seats_map,
    seat_group_to_wall_index, START_POSES_CW, START_POSES_CCW,
)
from ekf.start_detection import detect_start_obstacle, detect_start_open

START_VOTES = 5                # scans to vote over before committing the map
DIRECTION_VOTES = 5            # confident, agreeing scans before latching
LANE_NOMINALS = (0.60, 1.00)   # plausible lane widths (open challenge)
LANE_PLAUS_TOL = 0.15          # measurement must be within this of a nominal
MIN_WIDTH_SAMPLES = 10         # driving samples per straight before it counts
SIDE_ALPHA_TOL = np.radians(25.0)

# The map switch is jump-free for the whole start straight: the CW and CCW maps
# describe the SAME three walls there and differ only in which side is the inner
# band, which first matters at the corner. So the guard runs up to just short of
# the corner -- driving far, or swerving around an obstacle, must not block it.
MAP_SWITCH_CORNER_MARGIN = 0.40
MAP_SWITCH_FALLBACK_X = 0.30

# Detections taken before the seat grid exists are kept with their pose and
# replayed once it does. Bounded so a long pre-latch phase cannot grow without
# limit (15 Hz -> 60 s of scans).
PENDING_MAX = 900

# A seat is masked out of the wall extraction far earlier than it is reported
# as an obstacle: one vote is already reason enough to keep those directions
# out of a wall fit, while reporting still needs the full threshold.
MASK_MIN_VOTES = max(1, MIN_SEAT_VOTES // 4)


OUTER_HALF = 1.5               # outer wall position in the field frame
INNER_HALF = 0.5               # inner band (obstacle challenge, fixed)

# Start aus der Parkluecke. Die Bucht steht immer an der Aussenbande; die liegt
# dann ~5 cm neben dem Roboter, unter der Untergrenze der Fusion, und ist
# unsichtbar. Sichtbar sind die Innenbande quer ueber die Gasse (~0,9 m) und die
# Frontwand am Ende der Geraden. Daraus folgen Richtung UND Pose:
#   Innenbande rechts -> CW,  Innenbande links -> CCW
BAY_INNER_MIN = 0.75           # Innenbande muss in diesem Abstand liegen ...
BAY_INNER_MAX = 1.05
BAY_OUTER_MAX = 0.25           # ... und die andere Seite leer oder ganz nah sein
BAY_DEPTH = 0.20               # Buchtwaende ragen 20 cm von der Aussenbande ein
BAY_CLEAR_MARGIN = 0.03        # LiDAR muss so weit aus der Bucht heraus sein
BAY_VIEW_MIN_LAT = 0.20        # in der Bucht: Hindernisse nur so weit seitlich
                               # zur Oeffnung hin (Buchtwaende ~0,04, Pylonen ~0,44)

# Startgerade aus dem Stand abtasten: pro Sitz entscheiden, ob dort etwas steht,
# nachweislich nichts steht, oder es (noch) nicht zu sagen ist. Ueber den rohen
# /scan, unabhaengig von der Farberkennung: geht der Strahl in Richtung des
# Sitzes UEBER ihn hinaus, ist er frei; trifft er AM Sitz, steht dort etwas;
# trifft er DAVOR, ist der Sitz verdeckt.
SEAT_RANGE_TOL = 0.10          # Treffer innerhalb +-10 cm um den Sitz = belegt
SEAT_FREE_SCANS = 5            # so viele Durchblicke, bis ein Sitz als frei gilt
# Durchblick-Freigabe im ganzen Lauf: ein Sitz mit Stimmen wird wieder frei,
# wenn der LiDAR mehrmals in Folge klar durch ihn hindurchsieht. Sonst bleibt
# ein Phantom bis zum Schluss (parken_test_22: beim Start stand etwas neben der
# Bucht, wurde als gruene #21 gewertet und nie wieder geprueft, obwohl der
# Strahl danach dutzende Male durch den Sitz ging). Vorsichtig, weil eine
# faelschlich freigegebene echte Pylone schlimmer ist als ein Phantom:
#   - Kegel = Pylonenbreite + SEAT_CLEAR_MARGIN (Posenfehler) -- KEIN Strahl
#     darin darf am Sitz enden, und keiner davor (verdeckt = keine Aussage)
#   - nur bis SEAT_CLEAR_MAX_DIST, nur bei Lokalisierung 'ok' und kleiner
#     Gierrate (ein Scan dauert 66 ms -- in der Kurve verzerrt er um Grade)
#   - SEAT_CLEAR_SCANS Durchblicke in Folge UND insgesamt mindestens doppelt
#     so viele Durchblicke wie Treffer am Sitz
# Nachgespielt an parken_test_18..22: nur #21 (Phantom, 22) und #2 (Geist neben
# #3, 22) werden frei, keine der echten Pylonen.
SEAT_CLEAR_SCANS = 6
SEAT_CLEAR_MAX_DIST = 1.20
SEAT_CLEAR_MARGIN = 0.05
SEAT_CLEAR_MAX_YAWRATE = 0.5   # rad/s
SEAT_CLEAR_MIN_RANGE = 0.12    # naeher: eigener Aufbau, kein Verdecker
# Farbe nur bei ruhigem Kurs werten, Belegung immer. In der Drehung passen Bild
# und Scan nicht zusammen (Versatz bis ~0,5 s): parken_test_42, rote #17 aus
# 0,3-0,5 m bei 1,5-1,75 rad/s fuenfmal GRUEN, sonst immer rot -- die Farbe
# kippte, der Pfad sprang von innen nach aussen (Steigung 3,2).
FARBE_MAX_GIERRATE = 0.6       # rad/s
# Nach Runde 1 keine Stimmen und keine Freigaben mehr -- wie der Regler
# (obs_freeze_lap): alles Relevante ist in Runde 1 gesehen, was danach neu
# auftaucht, kann nur falsch sein. So zeigt auch Foxglove beim Einparken
# keine neuen Pylonen mehr.
OBS_FREEZE_LAP = 1
START_SCAN_TIMEOUT = 2.0       # s nach dem Commit; danach 'incomplete' melden
START_SCAN_MAX_ALONG = 0.75    # nur Sitze bis so weit laengs der Geraden (voraus
                               # oder zurueck) von der Hinterachse. Der ferne Sitz
                               # (~1 m) ist aus der Bucht nicht einsehbar und fuers
                               # Ausparken egal -- ihn erkennt die normale Fahrt.

# Ausparken: Stimmen ruhen, sobald der Roboter losfaehrt, bis er aus der Bucht
# ist UND wieder parallel zur Geraden steht. Aus der Bucht heraus eingeschwenkt,
# teils noch zwischen den Waenden -- da entstehen nur schlechte Stimmen.
BAY_MOVE_DIST = 0.03           # Losfahren erkannt ab so viel Weg ...
BAY_MOVE_ANGLE = np.radians(3.0)   # ... oder so viel Drehung seit dem Commit
EXIT_HEADING_TOL = np.radians(15.0)  # Kurs gilt als parallel zur Geraden
EXIT_MAX_DIST = 1.20           # Notausgang: so weit gefahren -> sicher draussen
FRONT_ALPHA_TOL = np.radians(25.0)
FRONT_MIN_LEN = 0.50           # Frontwand ist 3 m lang, die Buchtwand 0,20 m
FRONT_MIN_DIST = 0.60          # die Bucht steht nie direkt an der Ecke

# Parkbucht aus der Wandextraktion maskieren. Die Buchtwaende sind im Wandmodell
# nicht enthalten, und in der letzten Kurve fuehrt der Weg genau an ihnen
# vorbei: in einem Lauf lagen dort alle Frontmessungen 0,2-1,0 m kuerzer als die
# Karte, die Zuordnung riss ab, und der Filter fing sich nie wieder.
# Rein geometrisch um die gemessene Startpose -- Magenta wird nicht gebraucht.
# Grosszuegig, weil die Pose ausgerechnet in der letzten Kurve, wenn die Bucht
# wieder in Sicht kommt, 10-30 cm daneben liegen kann. Die Buchtwaende reichen
# bis Feld-y = 1,30; der Kasten bis 1,20 laesst also 10 cm quer Luft.
BAY_BOX_HALF_LEN = 0.40        # laengs, um die Hinterachse beim Start
BAY_BOX_INNER_Y = 1.20         # quer, ab hier bis zur Aussenbande (Nordgasse)

# Rueckweg fuer die Wandzuordnung. Das feste Tor ohne Rueckweg divergierte: ein
# Fehler knapp ueber 12 cm schloss es, danach gab es keine Korrektur mehr, der
# Fehler wuchs, und das Tor blieb fuer immer zu -- 30 s Blindflug. Die
# EKF-Kovarianz taugt als Tor NICHT: sie wuchs in diesen 30 s nur von 0,2 auf
# 5,8 cm, waehrend der echte Fehler auf Meter anwuchs. Also zaehlen statt
# vertrauen: nach ein paar leeren Scans stufenweise aufweiten.
#
# Obergrenze 0,35 m: unter dem halben Abstand paralleler Waende (1 m), damit das
# weite Tor nicht auf die falsche Bande springt. Einen Meter Fehler faengt das
# nicht mehr ein -- deshalb muss es SCHNELL greifen, solange der Fehler beim
# Abriss noch bei 15-30 cm liegt.
GATE_LEVELS = [                # (d_tol m, alpha_tol)
    (0.12, np.radians(20.0)),  # eingerastet, wie bisher
    (0.20, np.radians(25.0)),
    (0.28, np.radians(30.0)),
    (0.35, np.radians(35.0)),
]
# Laengspruefung (match_walls overlap_tol): wie weit ein gemessenes Wandstueck
# ueber das Ende der Kartenwand hinaus liegen darf. Gerechnet mit der EKF-Pose --
# ist die laengs verlaufen, liegt auch ein echtes Wandende scheinbar daneben.
# Deshalb waechst die Toleranz mit dem Tor: wer das Tor quer aufweitet, weil die
# Zuordnung abgerissen ist, traut der Pose auch laengs weniger.
GATE_OVERLAP = [0.15, 0.25, 0.35, 0.45]
# Punkte naeher am LiDAR gehoeren nie zu einer Bande, an der er vorbeifaehrt
# (Fahrgasse >= 0,19 m, Bucht maskiert): das ist eine angestossene oder
# mitgeschobene Pylone oder der eigene Aufbau. Nur fuer die Wandextraktion.
WALL_MIN_RANGE = 0.10
GATE_EMPTY_SCANS = 4           # enge Stufe: Scans ohne Treffer vor dem Oeffnen
GATE_LEVEL_SCANS = 3           # weite Stufe: Scans ohne Einrasten vor der naechsten
GATE_SETTLE_SCANS = 5          # Scans mit kleiner Innovation vor dem Zurueckgehen
# Worst Case bis zur weitesten Stufe: 4 + 3 + 3 = 10 Scans, ~0,7 s bei 15 Hz.
# Vorher wurden auf den weiten Stufen nur LEERE Scans gezaehlt: vereinzelte
# Treffer setzten den Zaehler zurueck, ohne je ruhig genug zum Einrasten zu
# sein -- Stufe 2 -> 3 dauerte in einem Lauf 2,6 s, 1,2 m Blindfahrt in eine
# Wand.
GATE_WIDE_MIN_MATCHES = 2      # im weiten Tor: eine einzelne Wand reicht nicht

PERF_PERIOD = 5.0              # s zwischen zwei Laufzeit-Zeilen im Log
# Warnschwelle pro Strom. Die Farbwolke hat schon ab Werk ~155 ms Latenz (Kamera
# holen, fusionieren, einfaerben im camera_lidar-Node) -- die eigene Rechenzeit
# liegt bei ~13 ms. Eine gemeinsame Schwelle meldete deshalb dauernd "haelt
# nicht mit", obwohl der Node mithielt.
PERF_LAT_WARN_MS = {'scan': 150.0, 'color': 400.0}
POSE_HIST_LEN = 300            # ~3 s Posenhistorie bei ~100 Hz Odometrie

COLOR_CODE = {'red': Obstacle.COLOR_RED, 'green': Obstacle.COLOR_GREEN}


def yaw_from_quaternion(q):
    siny = 2.0 * (q.w * q.z + q.x * q.y)
    cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return np.arctan2(siny, cosy)


class ScanProcessor(Node):
    def __init__(self):
        super().__init__('scan_processor')
        self.race_mode = self.declare_parameter(
            'race_mode', 'obstacle').get_parameter_value().string_value
        self.parking_lot_present = self.declare_parameter(
            'parking_lot_present', False).get_parameter_value().bool_value
        # Wird aus der Parkluecke gestartet? Dann NICHT selbst latchen: aus der
        # Luecke heraus ist keine brauchbare Ecke zu sehen, die Erkennung
        # liefert trotzdem ein Ergebnis, und das war in einem Lauf falsch
        # herum. Die Richtung kommt stattdessen auf /parking_direction.
        #
        # Warum ein Parameter und kein Topic: dieser Knoten startet beim
        # Hochfahren, der Regler Minuten spaeter von Hand. Ein "warte mal" von
        # ihm kaeme grundsaetzlich zu spaet -- gelatcht waere laengst.
        self.wait_for_parking = self.declare_parameter(
            'wait_for_parking', False).get_parameter_value().bool_value
        # Start aus der Parkluecke, mit Richtung und Pose IN DER BUCHT gemessen.
        # Ersetzt wait_for_parking: die Karte steht ab dem ersten Scans, die
        # Richtung kommt nicht vom Regler, und das Sitzraster existiert von
        # Anfang an. Nur im Hindernisrennen.
        self.start_from_bay = self.declare_parameter(
            'start_from_bay', False).get_parameter_value().bool_value
        if self.start_from_bay:
            self.parking_lot_present = True    # Buchtstart heisst: Luecke steht
        # Einpark-Test: der Roboter steht auf der STARTGERADEN (Nordgasse), in
        # Fahrtrichtung start_gerade (CCW/CW), z. B. am Anfang der letzten
        # Geraden. Frontwand und Aussenbande messen -> Feldpose, dann Karte wie
        # beim Buchtstart. Die Luecke selbst sieht er nicht: ihre Lage fuer die
        # Maske kommt aus test_bucht_front / test_bucht_q (Mittel frueherer
        # Laeufe, 0 = Standard je Richtung). Setzt round1_controller mit
        # einparken_test:=true selbst beim Neustart.
        self.start_gerade = self.declare_parameter(
            'start_gerade', '').get_parameter_value().string_value.strip().upper()
        self.test_bucht_front = self.declare_parameter(
            'test_bucht_front', 0.0).get_parameter_value().double_value
        self.test_bucht_q = self.declare_parameter(
            'test_bucht_q', 0.159).get_parameter_value().double_value
        if self.start_gerade not in ('', 'CW', 'CCW'):
            self.get_logger().error(
                f'start_gerade={self.start_gerade!r} -- nur CW oder CCW. Aus.')
            self.start_gerade = ''
        if self.start_gerade:
            self.start_from_bay = False
            self.parking_lot_present = True
        self.test_pose_field = None  # Feldpose beim Start auf der Geraden
        self.gerade_votes = []
        self.bay_votes = []          # (richtung, front_d, d_innen) pro Scan
        self.bay_pose_field = None   # Feldpose des Roboters, in der Bucht gemessen
        self.bay_left = False        # LiDAR hat die Bucht verlassen (klebt)
        # None (vor dem Commit) | 'parked' | 'exiting' | 'clear'
        self.bay_phase = None
        self.start_scan_state = None # None | 'scanning' | 'complete' | 'incomplete'
        self.start_scan_t0 = None
        self.seat_free = {}          # Sitz-ID -> Scans, die ueber den Sitz hinaussahen
        self.seat_hit = {}           # Sitz-ID -> Scans mit Treffer am Sitz
        # Durchblick-Freigabe (Index in obstacle_map.seats):
        self.clear_run = {}          # Durchblicke in Folge
        self.clear_durch = {}        # Durchblicke insgesamt
        self.clear_treffer = {}      # Treffer am Sitz insgesamt
        self.yaw_rate = 0.0          # aus /ekf/odom, fuer die Freigabe
        self.bay_odo_weg = 0.0       # Eigenbewegung seit dem Commit in der Bucht
        self.bay_odo_dreh = 0.0
        self.bay_odo_t = None

        # Rueckweg der Wandzuordnung
        self.gate_level = 0
        self.gate_empty = 0          # enge Stufe: Scans ohne Treffer in Folge
        self.gate_level_scans = 0    # weite Stufe: Scans seit Eintritt
        self.gate_settle = 0         # ruhige Scans in Folge (im weiten Tor)
        self.loc_state = None        # zuletzt publizierter Zustand
        self.perf = {}               # Laufzeitmessung pro Callback
        self.pose_hist = deque(maxlen=POSE_HIST_LEN)   # (stamp, pose)
        self.perf_t = time.monotonic()

        self.pose = (0.0, 0.0, 0.0)
        self.map_walls = None
        self.front_wall_x = None
        self.votes = []
        self.position = None
        self.lane_width = None

        self.left_d = None
        self.right_d = None
        self.front_d_meas = None

        self.direction = None
        self.dir_votes = []
        # Pose des Roboters, als die Startposition erkannt wurde. Bei einem
        # normalen Start (0,0,0); nach dem Ausparken die Pose in der Spur.
        self.commit_pose = (0.0, 0.0, 0.0)
        # Richtung aus der Parkluecke, solange die Karte noch nicht steht.
        self.parking_direction = None

        # --- lane-width learning (open mode) ---
        self.lap_state = None        # [corner_idx, corner_count, lap]
        self.width_samples = {}
        self.width_fixed = {}
        self.inner_walls = None

        # --- obstacles (obstacle mode) ---
        self.obstacle_map = None     # built at the direction latch
        self.seat_wall_idx = None
        self.start_seat_group = None  # Sitzgruppe der Startgeraden
        self.start_wall_idx = None
        self.obstacle_state = None
        self.pending_dets = deque(maxlen=PENDING_MAX)   # (detections, pose)
        # angular sectors of the pillars seen in the most recent colour scan.
        # The wall extraction masks these out: a pillar merged into a wall
        # corrupts its fit, and near a corner that delays the direction latch
        # by seconds -- which in turn delays the seat grid and the obstacle map.
        self.obstacle_sectors = []

        latched = QoSProfile(depth=1)
        latched.durability = DurabilityPolicy.TRANSIENT_LOCAL

        # Tiefe 1: immer nur den NEUESTEN Scan verarbeiten. Mit Tiefe 10 staute
        # sich bei einem langsamen Callback eine Reihe alter Scans, die der Node
        # brav nacheinander abarbeitete -- Wandkorrekturen kamen 0,35-0,6 s
        # nach ihrem Scan beim EKF an, in einer 90-Grad/s-Kurve 30-50 Grad zu
        # spaet. Ein verworfener Scan kostet nichts, ein veralteter zieht den
        # Kurs zurueck.
        latest = QoSProfile(depth=1)
        self.create_subscription(LaserScan, '/scan', self.scan_cb, latest)
        self.create_subscription(Odometry, '/ekf/odom', self.pose_cb, 10)
        self.create_subscription(Int32MultiArray,
                                 '/round1_controller/lap_state',
                                 self.lap_state_cb, latched)
        self.create_subscription(PointCloud2, '/camera_lidar/colored_scan',
                                 self.colored_scan_cb, latest)
        # Fahrtrichtung aus der Parkluecke, falls dort ausgeparkt wird.
        # BEWUSST NICHT latched, und der Regler sendet auch nicht latched:
        # eine latched Nachricht ueberlebt den Lauf, der sie erzeugt hat, und
        # hat diesen Knoten schon einmal mit der Richtung aus dem VORIGEN Lauf
        # entsperrt. Der Regler wiederholt sie stattdessen waehrend des ganzen
        # Scan-Halts. (Ausserdem passen TRANSIENT_LOCAL-Abonnent und
        # VOLATILE-Publisher in DDS nicht zusammen -- sie faenden sich nicht.)
        self.create_subscription(String, '/parking_direction',
                                 self.parking_direction_cb, 10)

        self.pub = self.create_publisher(WallMatchArray, '/wall_matches', 10)
        self.wall_dist_pub = self.create_publisher(
            Float64MultiArray, '/wall_distances', 10)
        self.obstacle_live_pub = self.create_publisher(
            ObstacleArray, '/obstacles_live', 10)
        self.front_wall_pub = self.create_publisher(Float64, '/front_wall_x', latched)
        self.direction_pub = self.create_publisher(String, '/race_direction', latched)
        self.corner_pub = self.create_publisher(CornerGeometry, '/corner_geometry', latched)
        self.inner_pub = self.create_publisher(CornerGeometry, '/inner_geometry', latched)
        self.obstacle_pub = self.create_publisher(ObstacleArray, '/obstacles', latched)
        # 'ok' | 'recovering' | 'lost' -- damit der Regler weiss, wann er blind
        # faehrt (langsamer werden, nicht einparken, keinen Erfolg melden)
        self.loc_pub = self.create_publisher(String, '/localization_state', latched)
        # Startgerade aus der Bucht abgetastet? 'scanning' -> 'complete' |
        # 'incomplete'. Erst dann ist /obstacles fuer die Startgerade vollstaendig.
        self.start_scan_pub = self.create_publisher(String, '/start_scan_state', latched)

        self.get_logger().info(
            f'start detection running (mode={self.race_mode}, '
            f'parking_lot={self.parking_lot_present})...')
        if self.wait_for_parking:
            self.get_logger().info(
                'wait_for_parking: die Fahrtrichtung kommt aus der Parkluecke '
                'ueber /parking_direction. Die Eckengeometrie latcht NICHT '
                'von selbst -- ohne diese Nachricht bleibt der Knoten ohne '
                'Richtung und damit ohne Sitzraster.')

    # ------------------------------------------------------------------ #
    # callbacks
    # ------------------------------------------------------------------ #

    def pose_cb(self, msg):
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        theta = yaw_from_quaternion(msg.pose.pose.orientation)
        self.pose = (x, y, theta)
        self.yaw_rate = float(msg.twist.twist.angular.z)
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        self.pose_hist.append((t, self.pose))
        # Losfahren in der Bucht: Weg und Drehung aus der EIGENBEWEGUNG
        # (Encoder, Gyro), nicht aus der Pose -- die springt, wenn der
        # Wandabgleich nach dem Kartencommit nachzieht (parken_test_29, CW:
        # 4-9 cm im Stand, als "losgefahren" gewertet, Startabtastung
        # abgebrochen, rote Pylone vor der Luecke unbekannt).
        if self.bay_phase == 'parked':
            if self.bay_odo_t is not None and 0.0 < t - self.bay_odo_t < 0.5:
                dt = t - self.bay_odo_t
                self.bay_odo_weg += float(msg.twist.twist.linear.x) * dt
                self.bay_odo_dreh += self.yaw_rate * dt
            self.bay_odo_t = t

    def _pose_at(self, stamp):
        """Pose zum Zeitpunkt einer Messung statt der aktuellen.

        Die Farbwolke kommt ~155 ms nach ihrem Scan an. In einer Kurve mit
        90 Grad/s hat sich der Roboter bis dahin um 14 Grad weitergedreht; mit
        der aktuellen Pose verrechnet, laege ein Hindernis in 1 m Abstand 25 cm
        daneben -- auf dem falschen Sitz. Linear interpoliert zwischen den zwei
        naechsten Odometrie-Nachrichten; ausserhalb der Historie die naechste.
        """
        t = stamp.sec + stamp.nanosec * 1e-9
        h = self.pose_hist
        if not h:
            return self.pose
        if t <= h[0][0]:
            return h[0][1]
        if t >= h[-1][0]:
            return h[-1][1]
        lo, hi = 0, len(h) - 1
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if h[mid][0] <= t:
                lo = mid
            else:
                hi = mid
        (t0, p0), (t1, p1) = h[lo], h[hi]
        w = (t - t0) / (t1 - t0) if t1 > t0 else 0.0
        dth = np.arctan2(np.sin(p1[2] - p0[2]), np.cos(p1[2] - p0[2]))
        return (p0[0] + w * (p1[0] - p0[0]), p0[1] + w * (p1[1] - p0[1]),
                p0[2] + w * dth)

    def lap_state_cb(self, msg):
        prev = self.lap_state
        self.lap_state = list(msg.data)
        if self.lap_state[1] == 0 and self.start_wall_idx is None:
            self.start_wall_idx = self._current_outer_wall_index()
        if prev is not None and self.lap_state[2] > prev[2]:
            self._maybe_commit_inner_band(verbose=True)

    # ------------------------------------------------------------------ #
    # Laufzeit: Latenz und Rechenzeit sichtbar machen
    # ------------------------------------------------------------------ #

    def scan_cb(self, msg):
        self._timed('scan', msg, self._scan_cb)

    def colored_scan_cb(self, msg):
        self._timed('color', msg, self._colored_scan_cb)

    def _timed(self, kind, msg, fn):
        """Callback ausfuehren und Latenz (Scanstempel -> Bearbeitungsbeginn)
        sowie Rechenzeit sammeln. Alle PERF_PERIOD Sekunden eine Zeile ins
        Log -- so sieht man sofort, ob der Node mit dem LiDAR mithaelt."""
        t_start = time.perf_counter()
        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        now = self.get_clock().now().nanoseconds * 1e-9
        fn(msg)
        st = self.perf.setdefault(kind, {'lat': [], 'cpu': []})
        st['lat'].append(now - stamp)
        st['cpu'].append(time.perf_counter() - t_start)
        if time.monotonic() - self.perf_t >= PERF_PERIOD:
            self._report_perf()

    def _report_perf(self):
        self.perf_t = time.monotonic()
        teile, zu_langsam = [], False
        for kind, st in self.perf.items():
            if not st['lat']:
                continue
            lat_ms = 1000 * np.array(st['lat'])
            cpu_ms = 1000 * np.array(st['cpu'])
            teile.append(f"{kind}: Latenz {np.median(lat_ms):.0f}/{lat_ms.max():.0f} ms, "
                         f"Rechenzeit {np.median(cpu_ms):.1f}/{cpu_ms.max():.1f} ms")
            zu_langsam |= lat_ms.max() > PERF_LAT_WARN_MS.get(kind, 150.0)
            st['lat'].clear()
            st['cpu'].clear()
        if not teile:
            return
        text = 'Laufzeit (Median/Max) -- ' + ' | '.join(teile)
        # getrennte Aufrufstellen (rclpy: eine Stufe pro Zeile)
        if zu_langsam:
            self.get_logger().warn(text + ' -- Node haelt nicht mit!')
        else:
            self.get_logger().info(text)

    def _scan_cb(self, msg):
        measured = self._extract(msg)
        self._publish_wall_distances(measured)

        if self.map_walls is None:
            if self.start_gerade and self.race_mode == 'obstacle':
                self._gerade_start_step(measured)
                return
            if self.start_from_bay and self.race_mode == 'obstacle':
                self._bay_start_step(measured)
                return
            if self.wait_for_parking and self.parking_direction is None:
                # Noch in der Parkluecke. Von dort aus misst die
                # Startpositionserkennung Unsinn: in einem Lauf hat sie pos1
                # gewaehlt (Frontwand 1,45 m), waehrend der Roboter 2 m
                # entfernt stand -- also pos2. Die Karte haengt danach einen
                # halben Meter daneben, und der Regler sieht im ERSTEN
                # Regelschritt 1,67 m Querabweichung.
                return
            res = self._detect(measured)
            if res['valid']:
                self.votes.append((res['position'], res['front_dist'],
                                   res.get('left_d'), res.get('right_d')))
            if len(self.votes) >= START_VOTES:
                self._commit()
                if self.parking_direction and self.direction is None:
                    self._latch_direction(self.parking_direction, 'parking')
            return

        matches = self._match_with_recovery(measured)
        self._update_direction(measured)
        self._learn_lane_width(measured)
        self._update_bay_phase()
        self._scan_start_seats(msg)
        self._seats_durchblick(msg)

        out = WallMatchArray()
        out.header = msg.header
        for m in matches:
            wm = WallMatch()
            wm.header = msg.header
            wm.alpha_meas = float(m['measured'][0])
            wm.d_meas = float(m['measured'][1])
            wm.alpha_map = float(m['map'][0])
            wm.d_map = float(m['map'][1])
            out.matches.append(wm)
        self.pub.publish(out)

    def _colored_scan_cb(self, msg):
        """Obstacle detection, obstacle mode only.

        Raw detections go out every scan on /obstacles_live -- no map needed.
        For the seat grid: if it does not exist yet (before the direction
        latch), the detections are buffered WITH the pose they were taken at
        and replayed when the grid is built. The pose is start-anchored and
        valid from the first scan, so a replayed detection snaps exactly as it
        would have live.
        """
        if self.race_mode != 'obstacle':
            return
        dets = detect_obstacles(msg)
        # Hintere Sperrzone des LiDAR (Platine, |Rohwinkel| <= BLOCK_ANGLE, im
        # Roboterframe also 180 +- 60 grad): die Wandextraktion wirft sie weg,
        # die Farbscan-Detektion bisher nicht. Dort entstand eine gruene
        # Phantom-Pylone, immer an derselben Stelle RELATIV zum Roboter
        # (base_link -0,18/+0,29, 135 grad vom LiDAR) -- Sitz #21 in
        # parken_test_22 und 24.
        dets = [d for d in dets
                if abs(np.arctan2(d['y'], d['x'] - LIDAR_OFFSET_X)) < np.pi - BLOCK_ANGLE]

        # hand the pillar directions to the wall extraction. Set every scan,
        # including the empty case, so the mask clears once a pillar is passed.
        # Colour scan and /scan come from the same LiDAR at the same rate, so
        # the sectors are at most one scan interval old -- well inside the
        # margin they are widened by.
        self.obstacle_sectors = [d['sector'] for d in dets]

        self._publish_obstacles_live(dets, msg.header.stamp)
        if not dets:
            return

        if self.start_from_bay:
            if self.bay_phase is None:
                return                     # vor dem Commit: Richtung unbekannt
            if self.bay_phase == 'parked':
                # Im Stand: eingeschlossen ist der Roboter nur nach vorn, hinten
                # und zur Aussenbande; zur Oeffnung hin ist der Blick frei. Die
                # Pylonen der Startgeraden stehen genau dort, und von ihnen
                # haengt die Wahl des Ausparkmanoevers ab.
                dets = self._bay_opening_filter(dets)
                if not dets:
                    return
            elif self.bay_phase == 'exiting':
                return                     # beim Ausparken ruhen die Stimmen
            # 'clear': alles zaehlt
        elif self._in_parking_bay():
            # Parkluecken-Start ohne Buchtmessung (wait_for_parking): die Pose
            # haengt noch nicht am endgueltigen Frame -- nichts werten.
            return

        if self.obstacle_map is None:
            self.pending_dets.append((dets, self._pose_at(msg.header.stamp)))
            return
        if self._hindernisse_eingefroren():
            return

        if abs(self.yaw_rate) > FARBE_MAX_GIERRATE:
            # nur Belegung zaehlen: dist=inf -> Stimme 'far' (obstacle_map)
            dets = [dict(d, dist=float('inf')) for d in dets]
        self.obstacle_map.add_detections(dets, self._pose_at(msg.header.stamp),
                                         allowed=self._seat_allowed)
        self._publish_obstacles_if_changed()

    # ------------------------------------------------------------------ #
    # extraction / start detection
    # ------------------------------------------------------------------ #

    def _map_obstacle_sectors(self):
        """Sectors for pillars whose position is already known, recomputed from
        the current pose.

        This is the layer that carries a close pass. /obstacles_live cannot:
        it arrives at ~6 Hz, and at 0.1-0.2 m the bearing sweeps ~25 deg
        between messages while the pillar is ~20 deg wide, so the previous
        sector no longer overlaps. The fusion also drops everything below its
        0.15 m range floor, so the pillar vanishes from the live topic exactly
        in the window where it breaks the wall fit. The map position does not
        vanish, and the pose is available at scan rate.
        """
        if self.obstacle_map is None:
            return []
        px, py, th = self.pose
        c, s = np.cos(th), np.sin(th)
        out = []
        for p in self.obstacle_map.seats_for_mask(MASK_MIN_VOTES):
            dx, dy = p[0] - px, p[1] - py
            xr = c * dx + s * dy          # map -> robot frame
            yr = -s * dx + c * dy
            sec = sector_from_robot_point(xr, yr)
            if sec is not None:
                out.append(sec)
        return out

    def _extract(self, msg):
        pts = scan_to_points(msg)
        if self.map_walls is not None and len(pts):
            pts = pts[np.hypot(pts[:, 0], pts[:, 1]) >= WALL_MIN_RANGE]
        # drop the directions occupied by pillars, so they cannot end up inside
        # a wall cluster. A missing slice of wall is harmless (gap clustering
        # splits it, both parts still match the same map wall); a pillar inside
        # a wall is not. Two sources: known positions from the map (fast, works
        # at any range) and the live topic (for pillars not yet mapped).
        pts = mask_sectors(pts, self.obstacle_sectors + self._map_obstacle_sectors())
        pts = self._mask_parking_bay(pts)
        clusters = merge_wraparound(cluster_points(pts))
        split = []
        for c in clusters:
            split.extend(split_at_corners(c))
        measured = []
        for c in split:
            hnf = fit_wall_hnf(c)
            if hnf is not None:
                measured.append(lidar_to_base_link(*hnf))
        return measured

    @staticmethod
    def _side_distances(measured):
        """Nearest wall distance on each side, in metres (positive).
        Left is alpha ~ -90 (+y), right is alpha ~ +90 (-y)."""
        left = right = None
        for w in measured:
            a, d = w[0], w[1]
            if abs(wrap(a + np.radians(90.0))) < SIDE_ALPHA_TOL:
                if left is None or abs(d) < left:
                    left = abs(d)
            elif abs(wrap(a - np.radians(90.0))) < SIDE_ALPHA_TOL:
                if right is None or abs(d) < right:
                    right = abs(d)
        return left, right

    def _publish_wall_distances(self, measured):
        left, right = self._side_distances(measured)
        msg = Float64MultiArray()
        msg.data = [float(left) if left is not None else float('nan'),
                    float(right) if right is not None else float('nan')]
        self.wall_dist_pub.publish(msg)

    def _detect(self, measured):
        if self.race_mode == 'open':
            return detect_start_open(measured)
        return detect_start_obstacle(measured)

    def _commit(self):
        positions = [v[0] for v in self.votes]
        winner, _ = Counter(positions).most_common(1)[0]
        win = [v for v in self.votes if v[0] == winner]

        if self.race_mode == 'open':
            front_d = float(np.mean([v[1] for v in win]))
            left_d = float(np.mean([v[2] for v in win]))
            right_d = float(np.mean([v[3] for v in win]))
            self._commit_open(winner, front_d, left_d, right_d)
        else:
            self._commit_obstacle(winner)

    def _verankert(self, feldpose):
        """Feldpose des Roboters JETZT -> Feldpose des ODOM-URSPRUNGS.

        generate_map() und alles danach erwarten die Pose des Punktes, an dem
        die Odometrie genullt wurde -- nicht die des Roboters. Bei einem
        normalen Start ist das dasselbe: der Roboter steht still, bis erkannt
        ist. Nach dem Ausparken liegen 50 cm dazwischen, und ohne diese
        Verrechnung waere die Karte genau um die Ausparkstrecke versetzt.

        Steht der Roboter beim Erkennen im Ursprung, liefert das exakt die
        uebergebene Pose zurueck -- der normale Fall bleibt unveraendert.
        """
        xf, yf, thf = feldpose
        xo, yo, tho = self.commit_pose
        th0 = wrap(thf - tho)
        c, sn = np.cos(th0), np.sin(th0)
        return (xf - (c * xo - sn * yo), yf - (sn * xo + c * yo), th0)

    def _commit_obstacle(self, position):
        self.position = position
        self.lane_width = 1.0
        self.commit_pose = self.pose
        start_pose = self._verankert(START_POSES_CW[f'pos{position}'])
        self.map_walls = generate_map(start_pose)
        self.front_wall_x = self._front_wall_x_from_map(self.map_walls)
        self.get_logger().info(
            f'[obstacle] start position {position} -> map committed '
            f'({len(self.map_walls)} walls, CW default)')
        self._publish_front_wall_x()

    def _commit_open(self, position, front_d, left_d, right_d):
        self.position = position
        self.commit_pose = self.pose
        self.lane_width = left_d + right_d
        self.left_d = left_d
        self.right_d = right_d
        self.front_d_meas = front_d
        self.map_walls = start_map_3wall(front_d, left_d, right_d)
        self.front_wall_x = front_d
        self.get_logger().info(
            f'[open] start position {position} -> 3-wall map committed '
            f'(front={front_d:.2f}, left={left_d:.2f}, right={right_d:.2f})')
        self._publish_front_wall_x()

    # ------------------------------------------------------------------ #
    # direction latch + map switch
    # ------------------------------------------------------------------ #

    def _update_direction(self, measured):
        if self.direction is not None:
            return                                # already latched -> frozen

        res = detect_direction(measured, lane_width=self.lane_width)
        if not res['confident']:
            return
        self.dir_votes.append(res['direction'])
        if len(self.dir_votes) > DIRECTION_VOTES:
            self.dir_votes.pop(0)
        if len(self.dir_votes) == DIRECTION_VOTES and len(set(self.dir_votes)) == 1:
            if self.wait_for_parking:
                # Stimmen weiter sammeln, aber nicht festnageln. Sobald
                # /parking_direction kommt, wird verglichen -- so faellt auf,
                # wenn die beiden Quellen sich widersprechen.
                self.get_logger().info(
                    f'corner geometry would say {self.dir_votes[0]}, '
                    f'waiting for /parking_direction',
                    throttle_duration_sec=5.0)
                return
            self._latch_direction(self.dir_votes[0], 'corner geometry')

    def parking_direction_cb(self, msg):
        """Fahrtrichtung aus der Parkluecke.

        Beim Ausparken ist die Richtung sicher bestimmbar: der Roboter steht
        an der Aussenwand, die nahe Seite IST die Wand und die ferne das
        Spielfeld. Die Eckengeometrie hier kann das nicht besser wissen -- sie
        sieht aus der Luecke heraus ueberhaupt keine brauchbare Ecke, latcht
        aber trotzdem und war in einem Lauf nachweislich falsch herum
        (Ausparken CW, Latch CCW).

        Kommt nichts auf diesem Topic, bleibt alles wie bisher.
        """
        richtung = msg.data.strip().upper()
        if richtung not in ('CW', 'CCW'):
            self.get_logger().warn(
                f'/parking_direction: "{msg.data}" ist weder CW noch CCW')
            return
        if self.parking_direction is None and self.pending_dets:
            # Harte Grenze, auch falls wait_for_parking nicht gesetzt ist:
            # alles vor dieser Nachricht wurde in der Bucht aufgenommen.
            self.get_logger().info(
                f'{len(self.pending_dets)} gepufferte Scans aus der Parkluecke '
                f'verworfen')
            self.pending_dets.clear()
        self.parking_direction = richtung
        if self.map_walls is None:
            # Die Karte steht noch nicht -- sie wartet ja gerade auf diese
            # Nachricht. Erst die Startposition erkennen (jetzt darf sie das,
            # der Roboter ist aus der Luecke heraus), dann latchen; siehe
            # scan_cb. Andersherum ginge es nicht: _start_pose_for_direction
            # braucht self.position aus dem Commit.
            self.get_logger().info(
                f'Parkrichtung {richtung} vorgemerkt -- erst die Startposition '
                f'erkennen, dann latchen.')
            return
        if self.direction is None:
            aus_ecken = (self.dir_votes[0]
                         if len(self.dir_votes) == DIRECTION_VOTES
                         and len(set(self.dir_votes)) == 1 else None)
            if aus_ecken and aus_ecken != richtung:
                self.get_logger().warn(
                    f'/parking_direction sagt {richtung}, die Eckengeometrie '
                    f'haette {aus_ecken} gesagt. Das Parken gilt -- aus der '
                    f'Luecke heraus ist keine brauchbare Ecke zu sehen.')
            self._latch_direction(richtung, 'parking')
            return
        if self.direction != richtung:
            self.get_logger().error(
                f'/parking_direction meldet {richtung}, gelatcht ist aber '
                f'{self.direction}. Der Latch bleibt -- Karte, Eckengeometrie '
                f'und Sitzraster haengen daran. Wenn das Parken recht hat, '
                f'startet den scan_processor NACH dem Ausparken.')

    def _latch_direction(self, direction, quelle):
        """Richtung festnageln und alles daran Haengende aufbauen."""
        self.direction = direction
        self.direction_pub.publish(String(data=self.direction))
        self.get_logger().info(
            f'race direction latched: {self.direction} (from {quelle})')

        self._switch_map_to_direction()

        start_pose = self._start_pose_for_direction()
        self._publish_corner_geometry(start_pose)
        if self.race_mode == 'obstacle':
            inner, corners = inner_walls_map(start_pose)
            self._publish_inner_geometry(inner, corners)
            self._init_obstacle_map(start_pose)

    def _start_pose_for_direction(self):
        if self.race_mode == 'open':
            return self._open_start_pose()
        if self.test_pose_field is not None:
            return self._verankert(self.test_pose_field)
        if self.bay_pose_field is not None:
            return self._verankert(self.bay_pose_field)
        poses = START_POSES_CW if self.direction == 'CW' else START_POSES_CCW
        # Dieselbe Verankerung wie beim Commit, sonst laegen Eckengeometrie,
        # Innenband und Sitzraster gegenueber der Matching-Karte versetzt.
        return self._verankert(poses[f'pos{self.position}'])

    def _map_switch_limit(self):
        """How far along the straight the map may still be switched: up to just
        short of the corner, since both maps hold the same three walls until
        the inner band ends."""
        if self.front_wall_x is None:
            return MAP_SWITCH_FALLBACK_X
        return max(MAP_SWITCH_FALLBACK_X,
                   self.front_wall_x - MAP_SWITCH_CORNER_MARGIN)

    def _switch_map_to_direction(self):
        limit = self._map_switch_limit()
        if abs(self.pose[0]) > limit:
            self.get_logger().warn(
                f'direction latched at x={self.pose[0]:.2f} m, past the '
                f'{limit:.2f} m limit (corner) -- NOT switching map to avoid '
                f'a pose jump')
            return

        start_pose = self._start_pose_for_direction()
        if self.race_mode == 'open':
            self.get_logger().info(
                f'open start pose (from measurements): '
                f'({start_pose[0]:+.3f}, {start_pose[1]:+.3f}, '
                f'{np.degrees(start_pose[2]):+.1f} deg)')
            self.map_walls = outer_walls_map(start_pose)
        else:
            self.map_walls = generate_map(start_pose)

        self.front_wall_x = self._front_wall_x_from_map(self.map_walls)
        self.get_logger().info(
            f'matching map switched to {self.direction} '
            f'({len(self.map_walls)} walls, at x={self.pose[0]:.2f} m)')

    def _open_start_pose(self):
        """Exact field start pose for the open challenge, from the measured
        distances. The OUTER wall is the only fixed reference (always +-1.5);
        it is on the left for CW and on the right for CCW.

            CW  (faces +x): x = 1.5 - front_d,  y = 1.5 - left_d,   theta = 0
            CCW (faces -x): x = front_d - 1.5,  y = 1.5 - right_d,  theta = pi
        """
        f = self.front_d_meas
        if self.direction == 'CW':
            return (OUTER_HALF - f, OUTER_HALF - self.left_d, 0.0)
        return (f - OUTER_HALF, OUTER_HALF - self.right_d, np.pi)

    # ------------------------------------------------------------------ #
    # obstacles
    # ------------------------------------------------------------------ #

    def _publish_obstacles_live(self, dets, stamp):
        """Raw detections, base_link frame, every scan. No map needed, so this
        works from the first scan on. id and wall_idx are -1: without a map
        there is no seat to assign."""
        msg = ObstacleArray()
        msg.header.stamp = stamp
        msg.header.frame_id = 'base_link'
        for d in dets:
            o = Obstacle()
            o.id = -1
            o.position = Point(x=float(d['x']), y=float(d['y']), z=0.0)
            o.color = COLOR_CODE.get(d['color'], Obstacle.COLOR_UNKNOWN)
            o.wall_idx = -1
            msg.obstacles.append(o)
        self.obstacle_live_pub.publish(msg)

    # ------------------------------------------------------------------ #
    # Buchtstart: Phasen und Abtasten der Startgeraden
    # ------------------------------------------------------------------ #

    def _update_bay_phase(self):
        """parked -> exiting (losgefahren) -> clear (draussen und parallel)."""
        if not self.start_from_bay or self.bay_phase in (None, 'clear'):
            return
        cx, cy, cth = self.commit_pose
        px, py, th = self.pose
        weg = float(np.hypot(px - cx, py - cy))
        dth = abs(float(np.arctan2(np.sin(th - cth), np.cos(th - cth))))

        if self.bay_phase == 'parked':
            if (abs(self.bay_odo_weg) > BAY_MOVE_DIST
                    or abs(self.bay_odo_dreh) > BAY_MOVE_ANGLE):
                self.bay_phase = 'exiting'
                self.get_logger().info(
                    'Ausparken beginnt -- Hindernisstimmen ruhen bis zum Ende')
                if self.start_scan_state == 'scanning':
                    self._finish_start_scan('incomplete',
                                            'losgefahren, bevor alles entschieden war')
            return

        # exiting
        draussen = self._bay_cleared()
        parallel = dth < EXIT_HEADING_TOL
        if (draussen and parallel) or weg > EXIT_MAX_DIST:
            self.bay_phase = 'clear'
            grund = ('aus der Bucht und parallel zur Geraden'
                     if draussen and parallel else f'{weg:.2f} m gefahren')
            self.get_logger().info(
                f'Ausparken beendet ({grund}) -- Hindernisse zaehlen wieder, '
                f'in alle Richtungen')

    def _start_seats(self):
        """Die Sitze der Startgeraden, die aus der Bucht geprueft werden:
        [(sitz_id, map-Punkt)].

        Nur Sitze, die belegt sein duerfen (mit Parkluecke: innere Spalte), und
        nur bis START_SCAN_MAX_ALONG laengs der Geraden. Der ferne Sitz am Ende
        der Geraden ist aus der Bucht nicht einsehbar -- bliebe er in der
        Liste, stuende er ewig auf "offen", und jede Abtastung endete mit
        'incomplete'. Fuers Ausparken zaehlt er nicht; auf der Geraden hat die
        normale Erkennung ihn laengst, bevor der Roboter dort ist.
        """
        if self.obstacle_map is None or self.start_seat_group is None:
            return []
        cx, cy, cth = self.commit_pose
        ux, uy = np.cos(cth), np.sin(cth)       # Richtung der Geraden beim Start
        out = []
        for sid, (si, k, sp, col, row) in enumerate(self.obstacle_map.seats):
            if si != self.start_seat_group or not self._seat_allowed(si, col):
                continue
            laengs = (sp[0] - cx) * ux + (sp[1] - cy) * uy
            # nur VOR dem Roboter: nur dort kann eine Pylone das Ausparken
            # beeinflussen (CCW ein Sitz, CW zwei). Was dahinter steht, sieht
            # er erst auf der Runde -- beim Start stand dort schon ein Phantom.
            if 0.0 < laengs <= START_SCAN_MAX_ALONG:
                out.append((sid, sp))
        return out

    def _scan_start_seats(self, msg):
        """Im Stand: jeden Sitz der Startgeraden als belegt / frei / offen
        entscheiden, und 'complete' melden, sobald alle entschieden sind.

        Frei ist ein Sitz nur mit Beleg: SEAT_FREE_SCANS Scans, in denen der
        Strahl in seine Richtung weiter reichte als bis zum Sitz. Belegt ist
        er, wenn die Hinderniskarte ihn mit Farbe fuehrt. Alles andere bleibt
        offen -- ein Sitz, den man nicht sieht, ist nicht frei.
        """
        if self.bay_phase != 'parked' or self.start_scan_state != 'scanning':
            return
        seats = self._start_seats()
        if not seats:
            return

        pts = scan_to_points(msg)              # roh, OHNE Masken: die Pylonen-
        if len(pts) == 0:                      # maske wuerde genau diese Strahlen
            return                             # wegschneiden
        ang = np.arctan2(pts[:, 1], pts[:, 0])
        rng = np.hypot(pts[:, 0], pts[:, 1])
        px, py, th = self._pose_at(msg.header.stamp)
        lx = px + LIDAR_OFFSET_X * np.cos(th)
        ly = py + LIDAR_OFFSET_X * np.sin(th)

        for sid, sp in seats:
            dx, dy = sp[0] - lx, sp[1] - ly
            d = float(np.hypot(dx, dy))
            if d < 1e-3:
                continue
            b = np.arctan2(dy, dx) - th        # Peilung im Scanframe
            half = np.arctan2(PILLAR_HALF_WIDTH, d) + np.radians(1.0)
            im_kegel = np.abs((ang - b + np.pi) % (2 * np.pi) - np.pi) < half
            if not im_kegel.any():
                continue                       # kein Strahl (Abschattung o. ae.)
            r = float(rng[im_kegel].min())
            if r > d + SEAT_RANGE_TOL:
                self.seat_free[sid] = self.seat_free.get(sid, 0) + 1
            elif r >= d - SEAT_RANGE_TOL:
                self.seat_hit[sid] = self.seat_hit.get(sid, 0) + 1
            # sonst: verdeckt, zaehlt nicht

        belegt = {self._seat_id(o): o['color']
                  for o in self.obstacle_map.occupied_seats()}
        offen = []
        for sid, _ in seats:
            if belegt.get(sid) in ('red', 'green'):
                continue
            if (self.seat_free.get(sid, 0) >= SEAT_FREE_SCANS
                    and self.seat_hit.get(sid, 0) * 4 <= self.seat_free.get(sid, 0)):
                continue
            offen.append(sid)

        if not offen:
            self._finish_start_scan('complete')
        elif time.monotonic() - self.start_scan_t0 > START_SCAN_TIMEOUT:
            self._finish_start_scan('incomplete', f'Zeitlimit {START_SCAN_TIMEOUT:.1f} s')

    def _hindernisse_eingefroren(self):
        """Nach Runde OBS_FREEZE_LAP (lap_state vom Regler) steht die Karte."""
        if self.lap_state is None or len(self.lap_state) < 3:
            return False
        if self.lap_state[2] >= OBS_FREEZE_LAP:
            if not getattr(self, '_frost_gemeldet', False):
                self._frost_gemeldet = True
                self.get_logger().info(
                    f'Runde {OBS_FREEZE_LAP} vorbei -- Hinderniskarte eingefroren '
                    f'(keine neuen Stimmen, keine Freigaben).')
            return True
        return False

    def _seats_durchblick(self, msg):
        """Sitze mit Stimmen wieder freigeben, wenn der LiDAR durch sie
        hindurchsieht (Konstanten und Begruendung bei SEAT_CLEAR_SCANS)."""
        om = self.obstacle_map
        if om is None or not om.votes or self.loc_state != 'ok':
            return
        if self._hindernisse_eingefroren():
            return
        if abs(self.yaw_rate) > SEAT_CLEAR_MAX_YAWRATE:
            return
        kand = [sid for sid, keys in om.votes.items() if sum(keys.values()) > 0]
        if not kand:
            return
        pts = scan_to_points(msg)              # roh: die Pylonenmaske wuerde
        if len(pts) == 0:                      # genau diese Strahlen schneiden
            return
        ang = np.arctan2(pts[:, 1], pts[:, 0])
        rng = np.hypot(pts[:, 0], pts[:, 1])
        px, py, th = self._pose_at(msg.header.stamp)
        lx = px + LIDAR_OFFSET_X * np.cos(th)
        ly = py + LIDAR_OFFSET_X * np.sin(th)
        frei = []
        for sid in kand:
            sp = om.seats[sid][2]
            dx, dy = sp[0] - lx, sp[1] - ly
            d = float(np.hypot(dx, dy))
            if d > SEAT_CLEAR_MAX_DIST or d < 0.15:
                continue
            b = np.arctan2(dy, dx) - th
            half = np.arctan2(PILLAR_HALF_WIDTH + SEAT_CLEAR_MARGIN, d)
            kegel = ((np.abs((ang - b + np.pi) % (2 * np.pi) - np.pi) < half)
                     & (rng > SEAT_CLEAR_MIN_RANGE))
            if kegel.sum() < 3:
                continue                       # kein Strahl (Sperrzone, ins Leere)
            r = rng[kegel]
            if (r < d - SEAT_RANGE_TOL).any():
                continue                       # verdeckt: keine Aussage
            if (r <= d + SEAT_RANGE_TOL).any():
                self.clear_run[sid] = 0
                self.clear_treffer[sid] = self.clear_treffer.get(sid, 0) + 1
                continue
            self.clear_run[sid] = self.clear_run.get(sid, 0) + 1
            self.clear_durch[sid] = self.clear_durch.get(sid, 0) + 1
            if (self.clear_run[sid] >= SEAT_CLEAR_SCANS
                    and self.clear_durch[sid] >= 2 * self.clear_treffer.get(sid, 0)):
                frei.append((sid, d))
        if not frei:
            return
        for sid, d in frei:
            stimmen = dict(om.votes.pop(sid))
            self.clear_run[sid] = 0
            si, k, sp, col, row = om.seats[sid]
            nr = self._seat_id({'straight': si, 'row': row, 'column': col})
            self.get_logger().warn(
                f'Sitz #{nr} freigegeben: der LiDAR sieht aus {d:.2f} m '
                f'{self.clear_durch[sid]}x hindurch (Treffer '
                f'{self.clear_treffer.get(sid, 0)}), Stimmen waren {stimmen}')
        self._publish_obstacles_if_changed()

    def _finish_start_scan(self, state, grund=''):
        self.start_scan_state = state
        self.start_scan_pub.publish(String(data=state))
        belegt = {} if self.obstacle_map is None else {
            self._seat_id(o): o['color'] for o in self.obstacle_map.occupied_seats()}
        teile, n_belegt = [], 0
        for sid, _ in self._start_seats():
            farbe = belegt.get(sid)
            if farbe in ('red', 'green'):
                teile.append(f'#{sid} {"rot" if farbe == "red" else "gruen"}')
                n_belegt += 1
            elif (self.seat_free.get(sid, 0) >= SEAT_FREE_SCANS
                  and self.seat_hit.get(sid, 0) * 4 <= self.seat_free.get(sid, 0)):
                teile.append(f'#{sid} frei')
            elif self.seat_hit.get(sid, 0) > 0:
                teile.append(f'#{sid} belegt, Farbe unbekannt')
                n_belegt += 1
            else:
                teile.append(f'#{sid} offen')
        text = f'Startgerade {state}: ' + ', '.join(teile)
        if grund:
            text += f' ({grund})'
        # Mehr als zwei kann nicht sein (Regel: 1-2 pro Gerade). Null ist
        # moeglich: die Pylone kann auf dem fernen, nicht geprueften Sitz stehen.
        if n_belegt > 2:
            text += f' -- laut Regel hoechstens 2 Pylonen, erkannt {n_belegt}'
        elif n_belegt == 0:
            text += ' -- keine in den geprueften Sitzen (ferner Sitz folgt auf der Geraden)'
        # getrennte Aufrufstellen (rclpy: eine Stufe pro Zeile)
        if state == 'complete' and n_belegt <= 2:
            self.get_logger().info(text)
        else:
            self.get_logger().warn(text)

    def _bay_opening_filter(self, dets):
        """Nur Detektionen, die durch die Oeffnung der Bucht gesehen werden.

        Die Buchtwaende enden bei Feld-y = 1,30, der LiDAR steht bei ~1,34 --
        er schaut also schon ueber die Wandenden hinaus. Zur Innenbande hin
        ist ein Faecher von rund 146 Grad frei (Wandenden bei etwa -17 und
        -163 Grad fuer CW). Die Oeffnung liegt bei CW rechts (y < 0 im
        Roboterframe), bei CCW links.

        Mindestens BAY_VIEW_MIN_LAT seitlich: die Buchtwaende liegen ~4 cm
        neben dem LiDAR, eine fehlklassifizierte Buchtwand kommt also nicht
        durch. Die Pylonen der Startgeraden (innere Spalte, Feld-y = 0,9)
        liegen ~0,44 m seitlich.
        """
        if self.direction == 'CW':
            seite = -1.0
        elif self.direction == 'CCW':
            seite = +1.0
        else:
            return []                      # Richtung noch nicht bekannt
        # und nur VOR dem Roboter (x > 0 im base_link, Hinterachse): hinter
        # ihm steht kein Sitz, der das Ausparken betrifft (siehe _start_seats)
        return [d for d in dets
                if seite * d['y'] >= BAY_VIEW_MIN_LAT and d['x'] > 0.0]

    def _in_parking_bay(self):
        """Steht der Roboter noch in der Parkluecke?

        Buchtstart: geometrisch aus der Pose -- sie stimmt ab dem ersten Scan,
        also kann die Perzeption selbst sehen, wann der LiDAR draussen ist.
        Kein Signal vom Regler noetig.

        Sonst (wait_for_parking): geparkt, bis der Regler /parking_direction
        meldet -- dieselbe Bedingung, die den Kartencommit zurueckhaelt.
        """
        if self.start_from_bay:
            return not self._bay_cleared()
        return self.wait_for_parking and self.parking_direction is None

    def _bay_cleared(self):
        """Hat der LiDAR die Bucht verlassen? Einmal draussen, bleibt es so.

        Die Bucht reicht 20 cm von der Aussenbande in die Gasse. Solange der
        LiDAR in diesem Streifen steht, sieht er zwischen den Buchtwaenden
        hindurch; ausserhalb ist der Blick quer ueber die Gasse frei. Das
        Kriterium ist nur quer, nicht laengs -- konservativ: wer nur
        vorwaerts aus der Bucht rollt, gilt noch als drin.
        """
        if self.bay_left:
            return True
        if self.bay_pose_field is None:
            return False
        xs, ys, ths = self._start_pose_for_direction()   # Feldpose Odom-Ursprung
        px, py, th = self.pose
        lx = px + LIDAR_OFFSET_X * np.cos(th)            # LiDAR im map-Frame
        ly = py + LIDAR_OFFSET_X * np.sin(th)
        c, sn = np.cos(ths), np.sin(ths)
        y_feld = ys + sn * lx + c * ly                   # map -> Feld, nur y
        if y_feld < OUTER_HALF - BAY_DEPTH - BAY_CLEAR_MARGIN:
            self.bay_left = True
            self.get_logger().info(
                f'LiDAR hat die Parkluecke verlassen (Feld-y={y_feld:.2f}) -- '
                f'warte auf Kurs parallel zur Geraden')
        return self.bay_left

    # ------------------------------------------------------------------ #
    # Wandzuordnung mit Rueckweg
    # ------------------------------------------------------------------ #

    def _match_with_recovery(self, measured):
        """match_walls mit einem Tor, das sich oeffnet, wenn die Zuordnung
        abreisst, und schliesst, sobald sie sauber wieder eingerastet ist.

        Enge Stufe: nach GATE_EMPTY_SCANS Scans ohne Treffer oeffnen.
        Weite Stufe: die Bedingung ist NICHT "leer", sondern "nicht
        eingerastet" -- nach GATE_LEVEL_SCANS Scans ohne Einrasten eine Stufe
        weiter, auch wenn zwischendurch einzelne Treffer kamen. Nur ein
        laufendes Einrasten haelt die Eskalation an. Scans ganz ohne Waende
        zaehlen mit: blind ist blind.
        """
        d_tol, a_tol = GATE_LEVELS[self.gate_level]
        matches = match_walls(measured, self.map_walls, self.pose,
                              alpha_tol=a_tol, d_tol=d_tol,
                              overlap_tol=GATE_OVERLAP[self.gate_level])

        # im weiten Tor ist eine einzelne Wand zu wenig -- das kann genauso
        # ein Rest von Pylone oder Bucht sein wie die richtige Bande
        if self.gate_level > 0 and len(matches) < GATE_WIDE_MIN_MATCHES:
            matches = []

        top = len(GATE_LEVELS) - 1
        base_d = GATE_LEVELS[0][0]

        if self.gate_level == 0:
            if matches:
                self.gate_empty = 0
            else:
                self.gate_empty += 1
                if self.gate_empty >= GATE_EMPTY_SCANS:
                    self._gate_escalate()
        else:
            self.gate_level_scans += 1
            ruhig = bool(matches) and max(abs(m['innov_d']) for m in matches) < base_d
            self.gate_settle = self.gate_settle + 1 if ruhig else 0

            if self.gate_settle >= GATE_SETTLE_SCANS:
                self.get_logger().info(
                    f'Wandzuordnung wieder eingefangen (von Stufe '
                    f'{self.gate_level}) -- Tor zurueck auf {base_d:.2f} m')
                self.gate_level = 0
                self.gate_empty = 0
                self.gate_level_scans = 0
                self.gate_settle = 0
            elif (self.gate_settle == 0
                  and self.gate_level_scans >= GATE_LEVEL_SCANS
                  and self.gate_level < top):
                self._gate_escalate()

        self._publish_loc_state()
        return matches

    def _gate_escalate(self):
        self.gate_level += 1
        self.gate_empty = 0
        self.gate_level_scans = 0
        self.gate_settle = 0
        d, a = GATE_LEVELS[self.gate_level]
        self.get_logger().warn(
            f'Wandzuordnung abgerissen -- Tor aufgeweitet auf {d:.2f} m / '
            f'{np.degrees(a):.0f} deg (Stufe {self.gate_level})')

    def _publish_loc_state(self):
        top = len(GATE_LEVELS) - 1
        if self.gate_level == 0:
            state = 'ok' if self.gate_empty < GATE_EMPTY_SCANS else 'recovering'
        elif (self.gate_level == top and self.gate_settle == 0
              and self.gate_level_scans >= GATE_LEVEL_SCANS):
            state = 'lost'
        else:
            state = 'recovering'
        if state != self.loc_state:
            self.loc_state = state
            self.loc_pub.publish(String(data=state))
            # getrennte Aufrufstellen: rclpy merkt sich die Stufe pro Zeile und
            # wirft, wenn an derselben Stelle mal info und mal warn kommt
            if state == 'ok':
                self.get_logger().info(f'Lokalisierung: {state}')
            else:
                self.get_logger().warn(f'Lokalisierung: {state}')

    # ------------------------------------------------------------------ #
    # Parkbucht aus der Wandextraktion maskieren
    # ------------------------------------------------------------------ #

    def _mask_parking_bay(self, pts):
        """Scanpunkte an der Parkbucht verwerfen, bevor sie zu Waenden werden.

        Die Buchtwaende sind im Wandmodell nicht enthalten. Kommt die Bucht in
        der letzten Kurve wieder in Sicht, liegen dort Messungen, die die Karte
        nicht kennt -- in einem Lauf riss daran die Wandzuordnung ab. Daher ein
        Kasten im Feldframe um die beim Start gemessene Buchtpose. Er schneidet
        auch ein Stueck Aussenbande weg, was harmlos ist: die restliche Bande
        und die anderen Waende tragen die Lokalisierung weiter.

        Nur beim Start aus der Bucht aktiv -- nur dann ist ihre Lage bekannt.
        """
        if len(pts) == 0 or self.bay_pose_field is None:
            return pts

        px, py, th = self.pose
        c, sn = np.cos(th), np.sin(th)
        bx = pts[:, 0] + LIDAR_OFFSET_X                 # Scan -> base_link
        by = pts[:, 1]
        mx = px + c * bx - sn * by                      # base_link -> map
        my = py + sn * bx + c * by

        xs, ys, ths = self._start_pose_for_direction()  # Feldpose Odom-Ursprung
        cs, ss = np.cos(ths), np.sin(ths)
        fx = xs + cs * mx - ss * my                     # map -> Feld
        fy = ys + ss * mx + cs * my
        drin = ((np.abs(fx - self.bay_pose_field[0]) < BAY_BOX_HALF_LEN)
                & (fy > BAY_BOX_INNER_Y))
        return pts[~drin]

    # ------------------------------------------------------------------ #
    # Start aus der Parkluecke
    # ------------------------------------------------------------------ #

    @staticmethod
    def _front_distance(measured, kreuzen=False):
        """Abstand zur Frontwand (alpha ~ +-180), oder None.

        NICHT einfach die naechste Wand in Fahrtrichtung: aus der Bucht heraus
        steht die vordere Buchtwand ~13 cm vor dem LiDAR, ebenfalls quer, und
        auf /scan ist sie sichtbar (nur die Fusion blendet unter 0,15 m aus).
        Die naechste quer stehende Wand war deshalb in einem Lauf die
        Buchtwand -- Karte 0,9 m laengs versetzt.

        Die echte Frontwand ist die Aussenbande der naechsten Seite, 3 m lang;
        die Buchtwand ist 20 cm lang. Also nur lange Segmente, und zusaetzlich
        eine Mindestdistanz: die Bucht steht nie direkt an der Ecke.
        """
        best = None
        for w in measured:
            if abs(wrap(w[0] - np.pi)) >= FRONT_ALPHA_TOL:
                continue
            laenge = float(np.hypot(*(np.asarray(w[3]) - np.asarray(w[2]))))
            if laenge < FRONT_MIN_LEN:
                continue                     # Buchtwand oder Fragment
            d = abs(w[1])
            if d < FRONT_MIN_DIST:
                continue
            # kreuzen: das Stueck muss die Fahrlinie (y = 0) schneiden. Nur fuer
            # den Start auf der Geraden (Einpark-Test): am Anfang der
            # Startgeraden stand die Westseite der Innenbande 0,66 m voraus --
            # quer, 1 m lang, aber schraeg rechts vor ihm; als Frontwand
            # genommen lag die Karte 2 m daneben. Aus der BUCHT geht das nicht:
            # dort verdeckt die vordere Buchtwand genau die Fahrlinie, sichtbar
            # ist die Frontwand erst ~0,4 m seitlich.
            if kreuzen and float(w[2][1]) * float(w[3][1]) > 0.0:
                continue
            if best is None or d < best:
                best = d
        return best

    def _bay_vote(self, measured):
        """Ein Scan aus der Bucht -> (richtung, front_d, d_innen) oder None.

        Auf der Seite der Aussenbande ist nichts zu sehen (zu nah fuer die
        Fusion); die Seite mit einer Wand bei ~0,9 m ist die Innenbande.
        Stehen auf BEIDEN Seiten Waende in Innenbanden-Abstand, ist der
        Roboter nicht in der Bucht -- dann keine Stimme.
        """
        left, right = self._side_distances(measured)

        def innen(d):
            return d is not None and BAY_INNER_MIN <= d <= BAY_INNER_MAX

        def aussen_frei(d):
            return d is None or d < BAY_OUTER_MAX

        if innen(right) and aussen_frei(left):
            richtung, d_innen, seite = 'CW', right, -1.0
        elif innen(left) and aussen_frei(right):
            richtung, d_innen, seite = 'CCW', left, 1.0
        else:
            return None
        front = self._front_distance(measured)
        if front is None:
            front = self._front_ueber_innenbande(measured, seite)
        if front is None:
            front = self._front_stueck(measured)
        if front is None:
            return None
        return (richtung, front, d_innen)

    @staticmethod
    def _front_stueck(measured):
        """Letzter Rueckfall aus der Bucht: ein kurzes Stueck Frontwand
        (>= 0,25 m, quer, mindestens FRONT_MIN_DIST voraus).

        In CW sieht er die Ostwand nur durch ein schmales Fenster neben der
        Innenbande; steht dort eine Pylone, bleiben ~45 cm Wand, 1,4-1,9 m
        seitlich. Den HNF-Abstand eines so kurzen, seitlich weit entfernten
        Stuecks verdreht schon ein Fitfehler von 3-4 grad um 10 cm -- also die
        Lage der Stueckmitte in Fahrtrichtung. Der Roboter steht in der Bucht
        parallel; der Rest faellt spaeter beim Wandabgleich heraus.
        Die Buchtwaende (0,20 m, ~0,24 m voraus) fallen ueber den Abstand weg."""
        best = None
        for w in measured:
            if abs(wrap(w[0] - np.pi)) >= FRONT_ALPHA_TOL:
                continue
            p1, p2 = np.asarray(w[2]), np.asarray(w[3])
            if float(np.hypot(*(p2 - p1))) < 0.25:
                continue
            x_mitte = 0.5 * float(p1[0] + p2[0])
            if x_mitte < FRONT_MIN_DIST:
                continue
            if best is None or x_mitte < best:
                best = x_mitte
        return best

    @staticmethod
    def _front_ueber_innenbande(measured, seite):
        """Rueckfall, wenn die Frontwand aus der Bucht nicht zu sehen ist:
        das vordere Ende der Innenbande neben dem Roboter liegt immer
        OUTER_HALF - INNER_HALF (1,0 m) vor der Frontwand.

        Aus der Bucht verdeckt die vordere Buchtwand die Fahrlinie; die
        Frontwand ist nur durch ein Fenster zwischen Buchtwand und Innenbande
        zu sehen. In CW stand dort eine Pylone -- uebrig blieben 45 cm Wand,
        als Stueck zu kurz. Die Innenbande dagegen liegt in voller Laenge da.
        Nur wenn sie praktisch ganz sichtbar ist (>= 0,9 m): dann sind beide
        Enden echte Enden und keine Abschattung."""
        best = None
        for w in measured:
            # Seitenwand auf der Innenseite: alpha ~ -90 (links) / +90 (rechts)
            if abs(wrap(w[0] + seite * np.pi / 2.0)) >= SIDE_ALPHA_TOL:
                continue
            if not (BAY_INNER_MIN <= abs(w[1]) <= BAY_INNER_MAX):
                continue
            p1, p2 = np.asarray(w[2]), np.asarray(w[3])
            if float(np.hypot(*(p2 - p1))) < 0.9:
                continue
            vorn = max(float(p1[0]), float(p2[0]))
            if vorn <= 0.0:
                continue
            f = vorn + (OUTER_HALF - INNER_HALF)
            if best is None or f < best:
                best = f
        return best

    def _bay_start_step(self, measured):
        """Im Stand in der Bucht abstimmen, dann Karte, Richtung und Sitzraster
        in einem Zug aufbauen."""
        v = self._bay_vote(measured)
        if v is not None:
            self.bay_votes.append(v)
        if len(self.bay_votes) < START_VOTES:
            return

        richtung, n = Counter(b[0] for b in self.bay_votes).most_common(1)[0]
        if n < START_VOTES:
            # uneinig -- noch nicht festlegen, weiter sammeln
            self.bay_votes = self.bay_votes[-START_VOTES:]
            return
        win = [b for b in self.bay_votes if b[0] == richtung]
        front_d = float(np.mean([b[1] for b in win]))
        d_innen = float(np.mean([b[2] for b in win]))

        # Nordgasse, Innenbande bei y = 0.5:
        #   CW  blickt +x, Frontwand bei x = +1.5
        #   CCW blickt -x, Frontwand bei x = -1.5
        y = INNER_HALF + d_innen
        if richtung == 'CW':
            self.bay_pose_field = (OUTER_HALF - front_d, y, 0.0)
        else:
            self.bay_pose_field = (front_d - OUTER_HALF, y, np.pi)

        self.position = 'bay'
        self.lane_width = 1.0
        self.commit_pose = self.pose
        xf, yf, thf = self.bay_pose_field
        self.get_logger().info(
            f'[obstacle] Start aus der Parkluecke: {richtung}, '
            f'Front={front_d:.3f} m, Innenbande={d_innen:.3f} m -> Feldpose '
            f'({xf:+.3f}, {yf:+.3f}, {np.degrees(thf):+.0f} deg), '
            f'Abstand Aussenbande {OUTER_HALF - yf:.3f} m, '
            f'Abstand Frontwand {front_d:.3f} m')

        if self.parking_direction and self.parking_direction != richtung:
            self.get_logger().warn(
                f'/parking_direction sagt {self.parking_direction}, die Bucht '
                f'zeigt {richtung}. Es gilt die Messung.')

        # Karte, Eckengeometrie, Innenband, Sitzraster -- alles an der Richtung
        self._latch_direction(richtung, 'bay')
        self.get_logger().info(
            f'Hindernisse aus der Bucht: nur zur Oeffnung '
            f'({"rechts" if richtung == "CW" else "links"}), mindestens '
            f'{BAY_VIEW_MIN_LAT:.2f} m seitlich')
        self.bay_phase = 'parked'
        self.bay_odo_weg = 0.0
        self.bay_odo_dreh = 0.0
        self.bay_odo_t = None
        self.start_scan_state = 'scanning'
        self.start_scan_t0 = time.monotonic()
        self.start_scan_pub.publish(String(data='scanning'))
        self._publish_front_wall_x()

    def _gerade_front(self, measured, richtung):
        """Abstand zur Frontwand beim Start auf der Startgeraden, und woher.

        1. Frontwand: quer, lang, kreuzt die Fahrlinie.
        2. Sonst die nahe Stirnseite der Innenbande: quer, ganz auf der
           Innenseite (CW rechts, CCW links), 0,35-1,8 m seitlich. Ihre nahe
           Stirnseite steht immer 2,0 m vor der Frontwand (Innenbande +-0,5,
           Bande +-1,5). Am
           Anfang der Geraden ist sie oft die einzige sichtbare Querwand --
           im CW-Test lag etwas quer auf der Spur und verdeckte die Ostwand.
        (None, None), wenn keins von beiden.
        """
        innen = -1.0 if richtung == 'CW' else 1.0
        front = stirn = None
        for w in measured:
            if abs(wrap(w[0] - np.pi)) >= FRONT_ALPHA_TOL:
                continue
            d = abs(w[1])
            if d < 0.3:
                continue
            laenge = float(np.hypot(*(np.asarray(w[3]) - np.asarray(w[2]))))
            y1, y2 = float(w[2][1]), float(w[3][1])
            if y1 * y2 <= 0.0:
                if laenge >= FRONT_MIN_LEN and d >= FRONT_MIN_DIST:
                    front = d if front is None else min(front, d)
                continue
            if (innen * y1 > 0.0 and laenge >= 0.40
                    and min(abs(y1), abs(y2)) >= 0.35 and max(abs(y1), abs(y2)) <= 1.8):
                stirn = d if stirn is None else min(stirn, d)
        if front is not None:
            return front, 'Frontwand'
        if stirn is not None:
            return stirn + OUTER_HALF + INNER_HALF, 'Innenbande'
        return None, None

    def _gerade_start_step(self, measured):
        """Einpark-Test: im Stand auf der Startgeraden Frontwand und
        Aussenbande messen, dann Karte, Richtung und Sitzraster aufbauen."""
        richtung = self.start_gerade
        links, rechts = self._side_distances(measured)
        aussen = rechts if richtung == 'CCW' else links
        front, quelle = self._gerade_front(measured, richtung)
        if front is None or aussen is None or not (0.10 <= aussen <= 0.95):
            return
        if front < 1.0:
            # Am Anfang der Startgeraden liegen 2,3-2,7 m vor ihm. So nah: er
            # steht am Ende der Geraden oder falsch herum.
            self.get_logger().warn(
                f"[Einpark-Test] Frontwand nur {front:.2f} m voraus -- er muss am "
                f"ANFANG der Startgeraden stehen, Nase Richtung Luecke "
                f"({richtung}). Warte.", throttle_duration_sec=2.0)
            return
        self.gerade_votes.append((front, aussen, quelle))
        if len(self.gerade_votes) < START_VOTES:
            return
        front_d = float(np.median([v[0] for v in self.gerade_votes]))
        d_aussen = float(np.median([v[1] for v in self.gerade_votes]))
        quellen = '/'.join(sorted({v[2] for v in self.gerade_votes}))
        y = OUTER_HALF - d_aussen
        if richtung == 'CCW':
            self.test_pose_field = (front_d - OUTER_HALF, y, np.pi)
        else:
            self.test_pose_field = (OUTER_HALF - front_d, y, 0.0)
        self.position = 'gerade'
        self.lane_width = 1.0
        self.commit_pose = self.pose
        # Luecke fuer die Maske (sonst werden die Magenta-Waende zu Waenden)
        fb = self.test_bucht_front if self.test_bucht_front > 0.0 else (
            1.245 if richtung == 'CCW' else 1.96)
        yb = OUTER_HALF - self.test_bucht_q
        self.bay_pose_field = ((fb - OUTER_HALF, yb, np.pi) if richtung == 'CCW'
                               else (OUTER_HALF - fb, yb, 0.0))
        xf, yf, thf = self.test_pose_field
        self.get_logger().info(
            f'[Einpark-Test] Start auf der Startgeraden: {richtung}, '
            f'Front={front_d:.3f} m (aus {quellen}), Aussenbande={d_aussen:.3f} m -> Feldpose '
            f'({xf:+.3f}, {yf:+.3f}, {np.degrees(thf):+.0f} deg); Luecke '
            f'angenommen {fb:.3f} m vor der Frontwand, {self.test_bucht_q:.3f} m '
            f'von der Aussenbande')
        self._latch_direction(richtung, 'Einpark-Test')
        self._publish_front_wall_x()

    def _init_obstacle_map(self, start_pose):
        """Build the seat grid, then replay everything seen before it existed."""
        seats = obstacle_seats_map(start_pose)
        self.seat_wall_idx = seat_group_to_wall_index(start_pose)
        # Startgerade = die Sitzgruppe, die dem Commit-Punkt am naechsten liegt.
        # Geometrisch statt aus lap_state: dann gilt die Parkluecken-Regel ab
        # dem ersten Scan, auch bevor der Regler etwas gemeldet hat.
        cx, cy = self.commit_pose[0], self.commit_pose[1]
        self.start_seat_group = int(np.argmin([
            np.hypot(*(np.mean([q['p'] for q in g], axis=0) - (cx, cy)))
            for g in seats]))
        self.obstacle_map = ObstacleMap(seats)
        self.get_logger().info(
            f'obstacle seat grid ready (24 seats, groups -> walls '
            f'{self.seat_wall_idx}, Startgerade = Gruppe '
            f'{self.start_seat_group}'
            f'{", aeussere Spalte gesperrt" if self.parking_lot_present else ""})')

        if self.pending_dets:
            n = sum(len(d) for d, _ in self.pending_dets)
            for dets, pose in self.pending_dets:
                self.obstacle_map.add_detections(dets, pose,
                                                 allowed=self._seat_allowed)
            self.get_logger().info(
                f'replayed {n} buffered detections from '
                f'{len(self.pending_dets)} scans taken before the latch')
            self.pending_dets.clear()
            self._publish_obstacles_if_changed()

    def _seat_allowed(self, seat_group, column):
        """Parkluecken-Regel: steht eine Parkluecke, rueckt das Reglement alle
        Zeichen der Startgeraden nach innen -- dort ist nur die innere Spalte
        belegt. Die aeussere wird gesperrt, damit dort nichts eingerastet
        wird. Eine Detektion an einem gesperrten Aussensitz rastet auch nicht
        auf den inneren um: der liegt 0,2 m entfernt, die Einrastgrenze ist
        0,12 m -- sie wird schlicht verworfen.
        """
        if not self.parking_lot_present or self.start_seat_group is None:
            return True
        if seat_group != self.start_seat_group:
            return True
        return column == 'inner'

    @staticmethod
    def _seat_id(seat):
        return seat['straight'] * 6 + seat['row'] * 2 + \
            (0 if seat['column'] == 'outer' else 1)

    def _publish_obstacles_if_changed(self):
        occupied = self.obstacle_map.occupied_seats()
        state = tuple(sorted((self._seat_id(s), s['color']) for s in occupied))
        if state == self.obstacle_state:
            return
        self.obstacle_state = state

        msg = ObstacleArray()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'map'
        for s in occupied:
            o = Obstacle()
            o.id = self._seat_id(s)
            o.position = Point(x=float(s['p'][0]), y=float(s['p'][1]), z=0.0)
            o.color = COLOR_CODE.get(s['color'], Obstacle.COLOR_UNKNOWN)
            o.wall_idx = int(self.seat_wall_idx[s['straight']])
            msg.obstacles.append(o)
        self.obstacle_pub.publish(msg)

        txt = ', '.join(
            f"#{self._seat_id(s)}({s['color'][0]},w{self.seat_wall_idx[s['straight']]})"
            for s in sorted(occupied, key=self._seat_id))
        self.get_logger().info(f'obstacles: {len(occupied)} [{txt}]')
        self.get_logger().info('votes: ' + self.obstacle_map.vote_summary())

    # ------------------------------------------------------------------ #
    # lane-width learning (open mode)
    # ------------------------------------------------------------------ #

    def _current_outer_wall_index(self):
        """Outer wall the robot is driving along. corner_idx names the corner
        AHEAD; CCW came from corner k-1 (wall k-1), CW from k+1 (wall k)."""
        if self.lap_state is None or self.direction is None:
            return None
        k = self.lap_state[0]
        return (k - 1) % 4 if self.direction == 'CCW' else k % 4

    def _learn_lane_width(self, measured):
        """The START straight's width comes from the stationary start detection.
        The others are sampled while driving. Learning continues past round 1
        until the inner band is committed."""
        if self.race_mode != 'open' or self.inner_walls is not None:
            return
        wall_idx = self._current_outer_wall_index()
        if wall_idx is None:
            return

        if self.lap_state[1] == 0 and wall_idx not in self.width_fixed:
            self.width_fixed[wall_idx] = float(self.lane_width)
            self.get_logger().info(
                f'start straight {wall_idx}: lane width '
                f'{self.lane_width:.3f} taken from start detection')
            self._maybe_commit_inner_band()
            return
        if wall_idx in self.width_fixed:
            return

        left, right = self._side_distances(measured)
        if left is None or right is None:
            return
        width = left + right
        if min(abs(width - n) for n in LANE_NOMINALS) > LANE_PLAUS_TOL:
            return

        self.width_samples.setdefault(wall_idx, []).append(width)
        self._maybe_commit_inner_band()

    def _maybe_commit_inner_band(self, verbose=False):
        """Commit + publish the inner band once every straight's width is known.
        Publishes nothing while one is missing: better no inner geometry than a
        wrong one."""
        if self.race_mode != 'open' or self.inner_walls is not None:
            return
        widths = {}
        for i in range(4):
            if i in self.width_fixed:
                widths[i] = self.width_fixed[i]
                continue
            s = self.width_samples.get(i, [])
            if len(s) < MIN_WIDTH_SAMPLES:
                if verbose:
                    self.get_logger().warn(
                        f'lane width for straight {i}: only {len(s)} samples '
                        f'-> inner band not committed yet, will keep measuring')
                return
            widths[i] = float(np.median(s))

        result = inner_band_from_widths(self._open_start_pose(), widths)
        if result is None:
            self.get_logger().warn('inner band reconstruction failed (degenerate)')
            return
        inner_walls, inner_corners = result

        self.inner_walls = inner_walls
        self.map_walls = list(self.map_walls) + inner_walls
        wtxt = ', '.join(f'{i}:{widths[i]:.3f}' for i in range(4))
        where = (f'lap {self.lap_state[2]}, corner {self.lap_state[0]}'
                 if self.lap_state is not None else 'lap unknown')
        self.get_logger().info(
            f'inner band learned ({wtxt}) at {where} -> map extended to '
            f'{len(self.map_walls)} walls')
        self._publish_inner_geometry(inner_walls, inner_corners)

    # ------------------------------------------------------------------ #
    # publishing
    # ------------------------------------------------------------------ #

    def _publish_front_wall_x(self):
        if self.front_wall_x is not None:
            self.front_wall_pub.publish(Float64(data=float(self.front_wall_x)))
            self.get_logger().info(
                f'published front_wall_x = {self.front_wall_x:.3f}')

    def _publish_corner_geometry(self, start_pose):
        corners, walls, edge = outer_box_map(start_pose)
        msg = CornerGeometry()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'map'
        for i in range(4):
            msg.corners[i] = Point(x=float(corners[i][0]),
                                   y=float(corners[i][1]), z=0.0)
            w = WallHNF()
            w.nx, w.ny, w.d = walls[i]
            msg.walls[i] = w
        msg.edge_length = float(edge)
        self.corner_pub.publish(msg)
        self.get_logger().info('published corner_geometry (outer box)')

    def _publish_inner_geometry(self, walls, corners):
        msg = CornerGeometry()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'map'
        for i in range(4):
            msg.corners[i] = Point(x=float(corners[i][0]),
                                   y=float(corners[i][1]), z=0.0)
            w = WallHNF()
            w.nx = float(np.cos(walls[i]['alpha']))
            w.ny = float(np.sin(walls[i]['alpha']))
            w.d = float(walls[i]['d'])
            msg.walls[i] = w
        msg.edge_length = 0.0     # inner band is a rectangle: no single edge
        self.inner_pub.publish(msg)
        self.get_logger().info('published inner_geometry')

    # ------------------------------------------------------------------ #
    # helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _front_wall_x_from_map(map_walls):
        """Map-frame x of the front wall (alpha ~ +-180), as |d|."""
        for w in map_walls:
            alpha = w['alpha'] if isinstance(w, dict) else w[0]
            d = w['d'] if isinstance(w, dict) else w[1]
            if abs(abs(alpha) - np.pi) < np.radians(30.0):
                return abs(d)
        return None


def main():
    rclpy.init()
    rclpy.spin(ScanProcessor())


if __name__ == '__main__':
    main()