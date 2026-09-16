#!/usr/bin/env python3
"""
Round-1 controller -- multi-corner (full lap).

State machine:
  [AUSPARKEN -> AUSPARK_SCAN] -> WAIT_INPUTS -> [WAIT_BUTTON] -> APPROACH ->
  TURN -> EXIT -> (loop) -> FINISHING -> DONE

  AUSPARKEN  Optional (Parameter ausparken). Der Roboter steht laengs in der
             Startluecke und muss SEITLICH heraus -- mit Ackermann geht das nur
             ueber Rangieren. Die Zuege laufen als Positionsfahrten auf dem ESP
             (Encoder), nicht ueber /cmd_vel: der Lidar sieht unter 0,15 m
             nichts, und in der Luecke ist die naechste Wand genau dort.
             Siehe ekf/ausparken.py. Mit nur_ausparken haelt der Regler danach
             an, statt das Rennen zu fahren.

  APPROACH   Drive the current straight, centred against the target line (outer
             wall of the current edge, offset inward by o_out). Watch for the
             turn-in point of the current corner.
  TURN       Pose-native arc tracking through the current corner (cross-track to
             the planned circle + heading to the tangent + speed-honest
             feedforward, blended out near the target). theta-based completion.
  EXIT       Stanley path-following onto the exit line for a short settle
             distance, then advance to the next corner (APPROACH) -- or, after a
             full lap, to FINISHING.
  FINISHING  Ramp speed down to a smooth stop on the finish straight.

Corners come from /corner_geometry: 4 outer-box corners + 4 outer walls,
edge-synchronous (walls[i] = edge corners[i]->corners[i+1]), CCW-indexed,
index 0 = largest x. The two walls at corner idx are walls[idx] and
walls[(idx-1)%4]. Direction step through the index: CCW -> +1, CW -> -1.
/corner_geometry is ALWAYS the 4 outer walls (both modes); the EKF's internal
8-wall matching map is separate and not used here.

Command convention: REP 103 (linear.x m/s fwd, angular.z rad/s CCW=left). The
esp_bridge does the calibrated Ackermann inverse and speed control.
"""

import collections
import math

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy
from rcl_interfaces.msg import SetParametersResult
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Bool, Float64, String, Int32MultiArray, Float64MultiArray
from std_msgs.msg import Float32, Float32MultiArray, Header, Int32

from ekf.ausparken import (bahn, cm_zu_grad, richtung_aus_scan,
                           schritte_aus_flach, simuliere, spiegeln,
                           SCHRITTE_STANDARD)
from ekf.wall_extraction import scan_to_points


# Farbcodes aus robot_msgs/Obstacle.msg und die halbe Klotzbreite aus
# obstacle_path.py -- hier gespiegelt, damit der Startgeraden-Zweig ohne
# zusaetzlichen Import auskommt.
OBST_UNBEKANNT, OBST_ROT, OBST_GRUEN = 0, 1, 2
BLOCK_HALB = 0.022          # 44 mm / 2


def yaw_from_quaternion(q):
    siny = 2.0 * (q.w * q.z + q.x * q.y)
    cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny, cosy)


def wrap(a):
    return math.atan2(math.sin(a), math.cos(a))


def line_from_points(p0, p1):
    """HNF (nx,ny,d) of the line through p0,p1, unit normal. Sign arbitrary."""
    dx, dy = p1[0] - p0[0], p1[1] - p0[1]
    n = math.hypot(dx, dy)
    if n < 1e-9:
        return None
    nx, ny = -dy / n, dx / n
    d = nx * p0[0] + ny * p0[1]
    return (nx, ny, d)


def line_intersect(l1, l2):
    n1x, n1y, d1 = l1
    n2x, n2y, d2 = l2
    det = n1x * n2y - n1y * n2x
    if abs(det) < 1e-9:
        return None
    x = (d1 * n2y - d2 * n1y) / det
    y = (n1x * d2 - n2x * d1) / det
    return (x, y)


class Round1Controller(Node):

    # param_name -> (attribute_name, converter). Table declares AND loads, so an
    # entry can never be half-present (declared but not read, or vice versa).
    _PARAMS = {
        'nose_offset':   ('nose_offset',   0.14,  float),
        'stop_gap':      ('stop_gap',      0.35,  float),
        'o_in':          ('o_in',          0.50,  float),
        'o_out':         ('o_out',         0.50,  float),
        'inner_clearance': ('inner_clearance', 0.25, float),  # target gap to the INNER band (racing line)
        'racing_line':     ('racing_line',     0.0, lambda v: bool(float(v))),  # 0 = drive lane CENTRE when free
        'use_auto_offset': ('use_auto_offset', 1.0, lambda v: bool(float(v))),  # 0 -> use o_in_list/o_out_list
        # obstacle avoidance
        'obs_clear_before': ('obs_clear_before', 0.20, float),  # on the new offset this far BEFORE the block
        'obs_clear_after':  ('obs_clear_after',  0.05, float),  # hold it this far AFTER (small -> swap earlier)
        'obs_transition_pref': ('obs_transition_pref', 0.70, float),  # lane-change length
        'obs_anchor_early': ('obs_anchor_early', 1.0, lambda v: bool(float(v))),  # swap right after the block
        'obs_skew':         ('obs_skew',         0.0,  float),   # 0=smooth S, 1=front-loaded (kink at start)
        'obs_freeze_lap':   ('obs_freeze_lap',   1,    int),     # ignore /obstacles after this many laps
        'obs_transition_min':  ('obs_transition_min',  0.40, float),  # measured limit ~0.40 m @0.45 m/s
        'obs_wall_margin':  ('obs_wall_margin',  0.12, float),  # never plan closer than this to a wall
        'v_obstacle':       ('v_obstacle',       0.35, float),  # speed on straights with obstacles
        'obs_slope_slow':   ('obs_slope_slow',   0.80, float),  # above this slope -> v_obstacle_steep
        'v_obstacle_steep': ('v_obstacle_steep', 0.35, float),  # speed for steep lane changes
        'parking_lot_present': ('parking_lot_present', 0.0, lambda v: bool(float(v))),
        # planning against the ACTUAL pose (not the ideal line)
        'arc_shrink':       ('arc_shrink',       1.0, lambda v: bool(float(v))),  # shrink R if the run-up is too short
        'min_turn_radius':  ('min_turn_radius',  0.30, float),
        'max_settle_slope': ('max_settle_slope', 0.80, float),  # lateral m per longitudinal m we trust
        'turn_in_lat_warn': ('turn_in_lat_warn', 0.10, float),  # warn above this lateral error at turn-in
        # do not start the arc while still correcting laterally (0.2 s steering dead time)
        'turn_in_lat_gate': ('turn_in_lat_gate', 0.03, float),  # settled below this lateral error
        'turn_in_om_gate':  ('turn_in_om_gate',  0.50, float),  # ... and below this commanded omega
        'turn_in_delay_max':('turn_in_delay_max',0.25, float),  # max distance to wait past T_A
        # scan pause at the end of each straight (lap 1 only -- after that the
        # seat grid is filled and standing still would only cost time)
        'scan_pause':       ('scan_pause',       1.0, lambda v: bool(float(v))),
        'scan_pause_s':     ('scan_pause_s',     1.5, float),   # how long to stand still [s]
        'scan_front_dist':  ('scan_front_dist',  1.18, float),  # ALWAYS stop this far from the front wall (pose)
        'scan_pause_laps':  ('scan_pause_laps',  1,    int),    # pause only during the first N laps
        'v_start':       ('v_start',       0.35,  float),   # speed on the start straight (before direction latch)
        'start_stop_gap': ('start_stop_gap', 0.50, float),  # stop this far from the front wall if direction never comes
        'start_lane_min': ('start_lane_min', 0.45, float),  # plausibility band for d_left+d_right
        'start_lane_max': ('start_lane_max', 1.30, float),
        # --- Ausweichen auf der Startgeraden ---------------------------------
        # Das Sitzraster und die Eckengeometrie gibt es erst beim Richtungs-
        # Latch, und der kann nicht frueher kommen: bis zum Ende des Innen-
        # blocks bei x=0.95 messen beide Wandabstaende 0.50 -- die Fahrtrichtung
        # ist bis dahin geometrisch nicht bestimmbar (Lauf 22: rechts oeffnet
        # bei x=0.84, Latch bei x=1.02, Pylone steht bei x=0.95). Gepuffert
        # wurde alles korrekt, es kommt nur zu spaet. /obstacles_live liefert
        # die Pylone dagegen schon 0.5 s VOR dem Losfahren, durchgehend und in
        # der richtigen Farbe -- und die Regel "rot rechts vorbei, gruen links
        # vorbei" gilt im ROBOTERframe ohne jede Fahrtrichtung.
        'start_dodge':      ('start_dodge',      1.0, lambda v: bool(float(v))),
        'start_dodge_look': ('start_dodge_look', 1.20, float),  # nur Hindernisse so weit voraus [m]
        'start_dodge_back': ('start_dodge_back', 0.25, float),  # Versatz noch so weit dahinter halten [m]
        'start_dodge_lane': ('start_dodge_lane', 0.35, float),  # seitliches Fenster um die Spurmitte [m]
        'start_dodge_margin': ('start_dodge_margin', 0.12, float),  # nie naeher an eine Wand planen [m]
        'start_dodge_votes': ('start_dodge_votes', 3, int),     # Sichtungen, bevor gelenkt wird
        'start_dodge_window_s': ('start_dodge_window_s', 1.0, float),  # ueber diesen Zeitraum gezaehlt
        'turn_radius':   ('R',             0.50,  float),
        'sweep_tol_deg': ('sweep_tol',     3.0,   lambda v: math.radians(float(v))),
        'ff_blend_deg':  ('ff_blend',      7.0,  lambda v: math.radians(float(v))),
        'k_ct':          ('k_ct',          8.0,   float),
        'k_th':          ('k_th',          2.5,   float),
        'k_stanley':     ('k_stanley',     1.2,   float),
        'k_stanley_i':   ('k_stanley_i',   0.0,   float),   # cross-track integral gain
        'k_heading':     ('k_heading',     1.0,   float),   # Stanley heading-term weight (damping)
        'k_heading_v_ref': ('k_heading_v_ref', 0.45, float),
	    'stanley_v_ref': ('stanley_v_ref', 0.0,   float),   # >0: fixed v for cross-track gain (speed-indep.)
        'i_ct_limit':    ('i_ct_limit',    math.radians(15.0), lambda v: math.radians(float(v))),  # anti-windup [deg->rad]
        'max_steer_deg': ('max_steer',     25.0,  lambda v: math.radians(float(v))),
        'wheelbase':     ('wheelbase',     0.10,  float),
        'max_yaw_rate':  ('max_yaw_rate',  3.0,   float),
        # speed profile (distance-based)
        'v_drive':       ('v_drive',       0.35,  float),   # straight cruise
        'v_turn':        ('v_turn',        0.35,  float),   # through the arc
        'accel_dist':    ('accel_dist',    0.2,   float),   # ramp v_turn->v_drive after a corner
        'brake_dist':    ('brake_dist',    0.2,   float),   # ramp v_drive->v_turn before T_A
        # lap / finish
        'n_corners':     ('n_corners',     4,     int),
        'finish_front_dist': ('finish_front_dist', 1.5, float),
        'finish_decel':  ('finish_decel',  0.8,   float),   # look-ahead brake decel [m/s^2]
        'finish_lead_time': ('finish_lead_time', 0.15, float),  # reaction lead [s] -> stops on point
        'v_finish_min':  ('v_finish_min',  0.15,  float),   # DRIVABLE crawl, just above deadband
        'finish_tol':    ('finish_tol',    0.04,  float),   # stop tolerance on front_dist
        # --- Ausparken aus der Startluecke -----------------------------------
        # Die beiden Magenta-Waende stehen senkrecht auf dem Aussenwall und
        # ragen 20 cm ins Feld; die Luecke ist der 26,25 cm breite Spalt
        # dazwischen. Der Roboter steht laengs darin und muss quer heraus.
        # Die ganze Rechnerei steckt in ekf/ausparken.py, hier nur die Schalter.
        # ausparken, nur_ausparken und ausparken_richtung_invertieren sind
        # ECHTE Bool-Parameter und stehen weiter unten bei require_button --
        # damit "-p nur_ausparken:=true" tut, was man erwartet. Alles in
        # dieser Tabelle ist DOUBLE und wollte "1.0" statt "true".
        'ausparken_sektor_grad':  ('ausparken_sektor_grad',  20.0, float),
        'ausparken_scans':        ('ausparken_scans',        5,    int),
        'ausparken_richtung_timeout': ('ausparken_richtung_timeout', 8.0, float),
        # Der Servo braucht Zeit bis zum Anschlag -- erst danach losfahren,
        # sonst faehrt der erste Zentimeter mit halbem Einschlag.
        'ausparken_lenk_wartezeit': ('ausparken_lenk_wartezeit', 0.6, float),
        'ausparken_zug_timeout':  ('ausparken_zug_timeout',  15.0, float),
        # Wieviel der ESP am Sollweg fehlen darf, bevor eine
        # Zeitueberschreitung als Fehler gilt. Der ESP meldet Status 1, wenn
        # er nicht einschwingt -- die letzten Millimeter schafft er oft nicht,
        # weil der Stellwert dort unter die Losbrechschwelle faellt. Fuer uns
        # zaehlt der gefahrene Weg, nicht das Einschwingen: 2 mm bei 8 mm
        # Reserve sind kein Grund, die Sequenz abzubrechen.
        'ausparken_weg_toleranz_cm': ('ausparken_weg_toleranz_cm', 1.0, float),
        # Standzeit nach dem Ausparken, bevor das Rennen beginnt. Die
        # Wahrnehmung braucht sie: die Farbausbeute der Fusion liegt im
        # Stillstand bei 38 Prozent und faellt ab 1 rad/s auf 2 Prozent. Der
        # Roboter steht hier zum ersten Mal in der Spur und schaut die ganze
        # Startgerade entlang -- das ist der beste Blick auf die Pylonen, den
        # er im ganzen Lauf bekommt. 0 schaltet die Pause ab.
        # Nicht "ausparken_scan_s" nennen: das unterscheidet sich nur um ein
        # Zeichen von ausparken_scans (Stimmen fuer die Richtung) und meint
        # etwas voellig anderes.
        'ausparken_halt_s': ('ausparken_halt_s', 2.0, float),
        'debug':         ('debug',         1.0,   lambda v: bool(float(v))),
    }

    def __init__(self):
        super().__init__('round1_controller')

        for name, (attr, default, conv) in self._PARAMS.items():
            self.declare_parameter(name, default)
        # structural (read once)
        self.declare_parameter('require_button', False)
        # Ausparken aus der Startluecke. nur_ausparken haelt danach an, statt
        # das Rennen zu fahren -- zum Einstellen der Schrittfolge.
        self.declare_parameter('ausparken', False)
        self.declare_parameter('nur_ausparken', False)
        # Falls meine Herleitung der offenen Seite doch falsch herum ist:
        # ein Schalter statt einer Codeaenderung.
        self.declare_parameter('ausparken_richtung_invertieren', False)
        self.declare_parameter('control_rate', 30.0)
        self.declare_parameter('odom_timeout', 0.5)   # bridge past short EKF gaps

        # per-corner overrides (index = corner_idx). Empty -> use the global scalar
        # (o_in / o_out / turn_radius). Set a 4-element list to override per corner,
        # e.g. o_in_list:=[0.5,0.3,0.5,0.3]. o_out[N] and o_in[N+1] need NOT match
        # (asymmetric racing line is allowed; Stanley drives the transition smoothly).
        from rcl_interfaces.msg import ParameterDescriptor, ParameterType
        arr = ParameterDescriptor(type=ParameterType.PARAMETER_DOUBLE_ARRAY)
        self.declare_parameter('o_in_list', [0.35, 0.35, 0.35, 0.35], arr)
        self.declare_parameter('o_out_list', [0.35, 0.35, 0.35, 0.35], arr)
        self.declare_parameter('turn_radius_list', [0.5, 0.5, 0.5, 0.5], arr)

        # Ausparksequenz als flache Liste [lenkung_%, cm, lenkung_%, cm, ...].
        # Positive Lenkung heisst ZUR OFFENEN SEITE, negative cm rueckwaerts --
        # die Tabelle ist dadurch richtungsfrei und wird erst beim Ausfuehren
        # gespiegelt. Pruefen ohne Roboter:
        #     python3 src/ekf/ekf/ausparken.py 100 5.9 -100 -4.4 ...
        self.declare_parameter('ausparken_schritte', list(SCHRITTE_STANDARD), arr)
        # Regelparameter des ESP fuer die Dauer der Sequenz. Flach als
        # [index, wert, ...], Reihenfolge der Indizes siehe PID_PARAMS in
        # esp_serial_bridge.py: 0 kp, 1 ki, 2 kd, 3 ilimit, 4 maxduty,
        # 5 tol_deg, 6 settle_ms, 7 timeout_ms, 8 minduty.
        # Ohne Begrenzung steht der Stellwert bei mehreren hundert Grad
        # Sollweg bis kurz vors Ziel am Anschlag -- in einer 26 cm langen
        # Luecke ist das zu schnell.
        # 4 = maxduty, 8 = minduty. Das Paar ist bewusst ENG gewaehlt.
        #
        # Der ESP kennt keine Anlauframpe -- in PID_PARAMS gibt es keinen
        # solchen Wert. Am Anfang jedes Zuges ist der Regelfehler riesig (Zug 1
        # sind 225 Grad Welle), kp mal Fehler also weit ueber jeder Grenze, und
        # der Stellwert springt in einem einzigen Takt auf maxduty. Bei 300
        # drehen die Raeder durch, erst recht bei vollem Lenkeinschlag, wo sie
        # zusaetzlich radieren.
        #
        # Fuers Ausparken wollen wir aber gar kein Geschwindigkeitsprofil,
        # sondern gleichmaessiges Kriechen ueber wenige Zentimeter. Liegt
        # maxduty nur knapp ueber minduty, laeuft der Motor die ganze Strecke
        # auf fast konstantem niedrigem Stellwert: kein Sprung am Anfang, und
        # unten haelt minduty ihn ueber der Losbrechschwelle (pwm_deadband
        # 0.076, also rund 78 duty), damit die letzten Millimeter nicht
        # liegenbleiben und der ESP nicht in seine Zeitgrenze laeuft.
        #
        # Zu langsam? minduty und maxduty gemeinsam anheben, den Abstand
        # zwischen beiden aber klein lassen.
        self.declare_parameter('ausparken_pid', [4.0, 140.0, 8.0, 90.0], arr)
        self.declare_parameter('ausparken_pid_nachher', [4.0, 1023.0], arr)

        self._load_params()
        self.require_button = bool(self.get_parameter('require_button').value)
        self.ausparken = bool(self.get_parameter('ausparken').value)
        self.nur_ausparken = bool(self.get_parameter('nur_ausparken').value)
        self.ausparken_richtung_invertieren = bool(
            self.get_parameter('ausparken_richtung_invertieren').value)
        self.control_rate = float(self.get_parameter('control_rate').value)
        self.odom_timeout = float(self.get_parameter('odom_timeout').value)
        self.add_on_set_parameters_callback(self._on_params)

        self.ausparken_schritte = list(
            self.get_parameter('ausparken_schritte').value)
        self.ausparken_pid = list(self.get_parameter('ausparken_pid').value)
        self.ausparken_pid_nachher = list(
            self.get_parameter('ausparken_pid_nachher').value)
        # Ein Schalter soll reichen: wer nur ausparken will, meint auch ausparken.
        if self.nur_ausparken and not self.ausparken:
            self.ausparken = True

        # --- state ---
        self.state = 'AUSPARK_BUTTON' if self.ausparken else 'WAIT_INPUTS'
        # Ausparken: Stimmen fuer die Richtung, Stelle in der Schrittfolge,
        # letzte Quittung der Bruecke.
        self.ausp_stimmen = []
        self.ausp_letzter_grund = None
        self.ausp_schritte = None
        self.ausp_richtung = None
        self.ausp_index = 0
        self.ausp_phase = 'lenken'
        self.ausp_lenk_gesendet = False
        self.ausp_t0 = 0.0
        self.ausp_gesendet_t = None
        self.ausp_move_done = None
        self.ausp_theta0 = 0.0
        self.ausp_pose0 = None
        self.pose = None
        self.v_ist = 0.0
        self.front_wall_x = None
        self.race_direction = None        # 'CW' | 'CCW'
        self.corners = None               # [(x,y)] * 4
        self.walls = None                 # [(nx,ny,d)] * 4
        self.inner_walls = None           # [(nx,ny,d)] * 4 from /inner_geometry (lap 2+)
        self.lane_width = None            # [m] * 4, per straight, from outer<->inner distance
        self.wall_dist = None             # (d_left, d_right) live, for the start straight
        self.start_center_y = None        # map-frame y of the lane centre, held once computed
        # Rohdetektionen der Startgeraden: (t, map_x, map_y, farbe). Im MAP-Frame
        # abgelegt, damit die Eigenbewegung zwischen zwei Meldungen die Abstimmung
        # nicht verfaelscht.
        self.live_obs = collections.deque(maxlen=60)
        self.start_dodge_aktiv = None     # zuletzt gewaehlter Versatz, nur fuer das Log
        self._start_halt = None           # (obst_x, ziel_y, info) bis zum Passieren
        self.obstacles = None             # full current stand from /obstacles
        self.obs_path = None              # planned polyline [(x,y)] for this straight
        self.obs_max_slope = 0.0          # steepest lane change in the current plan
        self.obs_path_end_q = None        # lateral offset the path ends on (= corner entry)
        self._path_idx = 0                # nearest-segment cursor for path following
        self.scan_done_this_straight = False   # scan pause fires once per straight
        self.scan_pause_t0 = 0.0
        self.last_odom_time = None
        self.button_pressed = False
        self.v_cmd = 0.0
        self.arc = None
        self.drive_start_xy = (0.0, 0.0)  # for the post-corner accel ramp
        self.ct_integral = 0.0            # Stanley cross-track integrator (reset per straight)
        self.corner_idx = None            # index of the corner currently targeted
        self.corner_count = 0             # corners completed
        self.last_cmd = (0.0, 0.0)        # (v, omega) held during short odom gaps

        latched = QoSProfile(depth=1)
        latched.durability = DurabilityPolicy.TRANSIENT_LOCAL
        self.create_subscription(Odometry, '/ekf/odom', self.odom_cb, 10)
        self.create_subscription(Float64, '/front_wall_x', self.front_wall_cb, latched)
        self.create_subscription(String, '/race_direction', self.direction_cb, latched)
        self.create_subscription(self._corner_msg_type(), '/corner_geometry',
                                 self.corner_cb, latched)
        self.create_subscription(self._corner_msg_type(), '/inner_geometry',
                                 self.inner_cb, latched)
        self.create_subscription(Float64MultiArray, '/wall_distances',
                                 self.wall_dist_cb, 10)
        try:
            from robot_msgs.msg import ObstacleArray
            self.create_subscription(ObstacleArray, '/obstacles',
                                     self.obstacles_cb, latched)
            # Rohdetektionen, ungerastert und ohne Fahrtrichtung. Genau das
            # braucht die Startgerade: das Sitzraster und die Eckengeometrie
            # entstehen erst beim Richtungs-Latch, und der kann geometrisch
            # nicht frueher kommen (siehe _start_ausweich_y).
            self.create_subscription(ObstacleArray, '/obstacles_live',
                                     self.obstacles_live_cb, 10)
        except ImportError:
            self.get_logger().warn("robot_msgs/ObstacleArray nicht verfuegbar -- "
                                   "Hindernisplanung inaktiv.")
        if self.require_button:
            self.create_subscription(Header, '/esp_serial_bridge/button', self.button_cb, 10)
        self.pub_cmd = self.create_publisher(Twist, '/cmd_vel', 10)
        # debug: Stanley errors for live plotting in Foxglove
        self.pub_e_ct = self.create_publisher(Float64, '~/dbg/e_ct', 10)
        self.pub_e_th = self.create_publisher(Float64, '~/dbg/e_theta_deg', 10)
        self.pub_delta = self.create_publisher(Float64, '~/dbg/delta_deg', 10)
        self.pub_k_h = self.create_publisher(Float64, '~/dbg/k_h_eff', 10)
        self.pub_arc_dist = self.create_publisher(Float64, '~/dbg/arc_dist', 10)
        self.pub_arc_R = self.create_publisher(Float64, '~/dbg/arc_R', 10)
        # lap state for the perception side (round-1 learning): which corner is
        # being approached, how many corners done, which lap.
        # data = [corner_idx, corner_count, lap]  (lap = corner_count // 4)
        # latched: a later-starting perception node still gets the current state.
        self.pub_lap = self.create_publisher(Int32MultiArray, '~/lap_state', latched)

        # Nur fuers Ausparken. Bewusst nicht immer angelegt -- sonst haengt der
        # Regler ohne Not am /scan und an vier weiteren Bruecken-Topics.
        if self.ausparken:
            self.pub_steer = self.create_publisher(
                Float32, '/esp_serial_bridge/steer', 10)
            self.pub_move = self.create_publisher(
                Float32, '/esp_serial_bridge/move', 10)
            self.pub_pid = self.create_publisher(
                Float32MultiArray, '/esp_serial_bridge/pid_set', 10)
            # Ein Motorbefehl bricht eine laufende Positionsfahrt ab -- das ist
            # unser Notausstieg. Ueber /cmd_vel geht das NICHT: der
            # Geschwindigkeitsregler der Bruecke schweigt waehrend einer Fahrt.
            self.pub_motor = self.create_publisher(
                Int32, '/esp_serial_bridge/motor', 10)
            self.create_subscription(Int32MultiArray,
                                     '/esp_serial_bridge/move_done',
                                     self.ausparken_move_done_cb, 10)
            self.create_subscription(LaserScan, '/scan',
                                     self.ausparken_scan_cb, 10)

        self.dt = 1.0 / self.control_rate
        self.create_timer(self.dt, self.control_loop)
        if self.ausparken:
            self.get_logger().info(
                ">>> Round1Controller bereit. Erst AUSPARKEN%s. <<<"
                % (", danach anhalten (nur_ausparken)" if self.nur_ausparken
                   else ", danach das Rennen"))
        else:
            self.get_logger().info(">>> Round1Controller (multi-corner) bereit. Warte auf Eingaben... <<<")

    def _corner_msg_type(self):
        from robot_msgs.msg import CornerGeometry
        return CornerGeometry

    # ------------------------------------------------------------- params
    def _load_params(self):
        for name, (attr, _default, conv) in self._PARAMS.items():
            setattr(self, attr, conv(self.get_parameter(name).value))
        self._load_lists()

    def _load_lists(self):
        self.o_in_list = [float(v) for v in self.get_parameter('o_in_list').value]
        self.o_out_list = [float(v) for v in self.get_parameter('o_out_list').value]
        self.R_list = [float(v) for v in self.get_parameter('turn_radius_list').value]

    def _on_params(self, params):
        for p in params:
            if p.name in self._PARAMS:
                attr, _default, conv = self._PARAMS[p.name]
                setattr(self, attr, conv(p.value))
            elif p.name == 'o_in_list':
                self.o_in_list = [float(v) for v in p.value]
            elif p.name == 'o_out_list':
                self.o_out_list = [float(v) for v in p.value]
            elif p.name == 'turn_radius_list':
                self.R_list = [float(v) for v in p.value]
        return SetParametersResult(successful=True)

    def _entry_wall_idx(self, idx):
        """Wall index of the straight the robot is currently ON (entering corner idx).

        walls[i] is the edge corners[i]->corners[i+1]. At corner k the two edges
        walls[k-1] and walls[k] meet. Which one the robot is driving depends on the
        direction it walks the indices:
          CCW (dir_step +1): comes from k-1  -> current straight = walls[k-1]
          CW  (dir_step -1): comes from k+1  -> current straight = walls[k]
        """
        return (idx - 1) % 4 if self.dir_step() > 0 else idx % 4

    def _exit_wall_idx(self, idx):
        """Wall index of the straight AFTER corner idx (the exit straight)."""
        return idx % 4 if self.dir_step() > 0 else (idx - 1) % 4

    def _lane_default_offset(self, wall_idx):
        """Offset for a straight with NO obstacle on it.

        Default is the LANE CENTRE -- safest, and it keeps the corner entry clean
        (no lateral settling needed). The tight racing line (inner_clearance from
        the inner band) is opt-in via `racing_line`, because deriving it from
        "have we seen /obstacles yet" was fragile: before the first obstacle
        message arrives that test is false and the robot hugged the inner band.
        """
        if self.lane_width is None or wall_idx >= len(self.lane_width):
            return None
        w = self.lane_width[wall_idx]
        if self.racing_line:
            return max(w - self.inner_clearance, 0.05)
        return 0.5 * w

    def _obstacle_offset_near_corner(self, wall_idx, corner_pt):
        """Pass-by offset for the obstacle on `wall_idx` CLOSEST to `corner_pt`.

        Used twice: for o_out it is the FIRST obstacle after the corner, for o_in
        the LAST one before it -- in both cases the one nearest that corner.
        """
        if not self.obstacles or self.lane_width is None or self.walls is None:
            return None
        mine = [o for o in self.obstacles if o['wall'] == wall_idx]
        if not mine:
            return None
        nx, ny, d = self.walls[wall_idx]
        near = min(mine, key=lambda o: (o['x'] - corner_pt[0]) ** 2
                                       + (o['y'] - corner_pt[1]) ** 2)
        q_block = (nx * near['x'] + ny * near['y']) - d
        return self._obs_planner_for_wall(wall_idx).pass_offset(
            q_block, near['color'], self.dir_step() > 0)

    def corner_o_in(self, idx):
        w = self._entry_wall_idx(idx)
        if self.corners is not None:
            q = self._obstacle_offset_near_corner(w, self.corners[idx])
            if q is not None:
                return q                      # last obstacle before the corner
        auto = self._lane_default_offset(w) if self.use_auto_offset else None
        if auto is not None:
            return auto
        return self.o_in_list[idx] if idx < len(self.o_in_list) else self.o_in

    def corner_o_out(self, idx):
        w = self._exit_wall_idx(idx)
        if self.corners is not None:
            q = self._obstacle_offset_near_corner(w, self.corners[idx])
            if q is not None:
                return q                      # first obstacle after the corner
        auto = self._lane_default_offset(w) if self.use_auto_offset else None
        if auto is not None:
            return auto
        return self.o_out_list[idx] if idx < len(self.o_out_list) else self.o_out

    def _obs_planner_for_wall(self, wall_idx):
        from ekf.obstacle_path import ObstaclePathPlanner
        w = self.lane_width[wall_idx]
        return ObstaclePathPlanner(lane_width=w, wall_margin=self.obs_wall_margin)

    def corner_R(self, idx):
        return self.R_list[idx] if idx < len(self.R_list) else self.R

    # ------------------------------------------------------------- callbacks
    def odom_cb(self, msg):
        p = msg.pose.pose
        self.pose = (p.position.x, p.position.y, yaw_from_quaternion(p.orientation))
        self.v_ist = float(msg.twist.twist.linear.x)
        self.last_odom_time = self.get_clock().now()

    def front_wall_cb(self, msg):
        if self.front_wall_x is None:
            self.get_logger().info(f"/front_wall_x empfangen: {msg.data:.3f} m.")
        self.front_wall_x = float(msg.data)

    def direction_cb(self, msg):
        if self.race_direction is None:
            self.get_logger().info(f"/race_direction empfangen: {msg.data}.")
        self.race_direction = msg.data

    def corner_cb(self, msg):
        corners = [(p.x, p.y) for p in msg.corners]
        walls = [(w.nx, w.ny, w.d) for w in msg.walls]
        if self.corners is None:
            self.get_logger().info(f"/corner_geometry empfangen: {len(corners)} Ecken.")
            self._assert_edge_convention(corners, walls)
        self.corners = corners
        self.walls = walls
        # inner band may have arrived FIRST (both topics are latched) -- then the
        # widths could not be computed yet. Do it now.
        if self.inner_walls is not None and self.lane_width is None:
            self._compute_lane_widths()

    def inner_cb(self, msg):
        """Inner band from round-1 learning (open mode, once at lap 0->1).

        Index convention matches /corner_geometry: inner walls[i] belongs to the
        SAME straight as outer walls[i]. From the pair we get that straight's lane
        width, and from then on the offsets are derived as
            offset_from_outer = lane_width - inner_clearance
        so the robot keeps a CONSTANT distance to the inner band on every straight,
        adapted to the real measured width (which is NOT rounded to 60/100 cm).
        """
        inner = [(w.nx, w.ny, w.d) for w in msg.walls]
        self.inner_walls = inner
        if self.debug:
            self.get_logger().info(
                "  [INNER] " + " | ".join(f"({nx:+.2f},{ny:+.2f},{d:+.2f})"
                                          for nx, ny, d in inner))
        # /corner_geometry and /inner_geometry are BOTH latched and arrive within
        # milliseconds -- the order at the subscriber is NOT guaranteed. So never
        # drop the inner band; just remember it and compute the widths as soon as
        # the outer walls are there (see corner_cb).
        if self.walls is None:
            self.get_logger().info("/inner_geometry empfangen (vor corner_geometry) "
                                   "-- Breiten werden nachgerechnet.")
            return
        self._compute_lane_widths()

    def _compute_lane_widths(self):
        """Lane width per straight, outer wall i <-> the PARALLEL inner wall.

        We do NOT trust the index convention here: pairing outer[i] with inner[i]
        silently produced nonsense (perpendicular walls -> |d_i - d_o| is
        meaningless, e.g. 1.95 m in a 1.0 m lane). Matching by parallelism is
        unambiguous whatever the indexing does.
        """
        if self.walls is None or self.inner_walls is None:
            return
        widths = []
        for i, (onx, ony, od) in enumerate(self.walls):
            # For every outer wall there are TWO parallel inner walls (the near
            # side of the inner box and the far one). Take the parallel one with
            # the SMALLEST gap -- that is the inner band of THIS straight.
            cands = []
            for w in self.inner_walls:
                if abs(onx * w[0] + ony * w[1]) > 0.9:        # parallel
                    cands.append((self._wall_gap(self.walls[i], w), w))
            if not cands:
                self.get_logger().error(
                    f"Keine parallele Innenwand zu Aussenwand {i}. Breiten verworfen.")
                return
            widths.append(min(cands)[0])

        # plausibility: a lane is never wider than the box or narrower than the car
        if not all(self.start_lane_min <= w <= self.start_lane_max for w in widths):
            self.get_logger().error(
                "Unplausible Gassenbreiten " +
                ", ".join(f"{w:.3f}" for w in widths) +
                f" (erwartet {self.start_lane_min:.2f}..{self.start_lane_max:.2f} m). "
                "Verworfen -- fahre mit o_in/o_out-Parametern weiter.")
            return

        self.lane_width = widths
        self.get_logger().info(
            "/inner_geometry empfangen. Gassenbreiten [m]: " +
            ", ".join(f"{w:.3f}" for w in widths) +
            f" -> Offsets (Breite - {self.inner_clearance:.2f}): " +
            ", ".join(f"{max(w - self.inner_clearance, 0.05):.3f}" for w in widths))

        # Re-plan the UPCOMING corner, but KEEP the entry line of the straight we
        # are already driving: the robot must finish this straight on its current
        # line and change offset only THROUGH the corner. Re-planning LA as well
        # would make Stanley pull over mid-straight and enter the corner skewed.
        if self.state == 'DRIVE' and self.arc is not None and self.pose is not None:
            keep_o_in = self.arc.get('o_in')
            self.arc = None
            self.plan_arc(self.pose[2], o_in_override=keep_o_in)
            self.get_logger().info(
                f"Bogen neu geplant: Eintritt bleibt {keep_o_in:.2f}, "
                f"Austritt auf neuen Offset.")

    @staticmethod
    def _wall_gap(outer, inner):
        """Perpendicular distance between two (near-)parallel HNF lines.

        Outer normals point inward, inner normals point outward (toward the lane),
        so the two normals are roughly opposite. Flip the inner one to compare, then
        the gap is the difference of the offsets along the common normal.
        """
        onx, ony, od = outer
        inx, iny, ind = inner
        if onx * inx + ony * iny < 0.0:      # opposite normals -> align them
            inx, iny, ind = -inx, -iny, -ind
        # both normals now point the same way; gap = |d_inner - d_outer|
        return abs(ind - od)

    def obstacles_cb(self, msg):
        """Full current obstacle stand (latched). Replan the current straight's
        path -- also mid-drive, because a late detection MUST still be avoided."""
        # Freeze after the scanning lap(s): everything relevant was seen in lap 1
        # (we stop at every corner for that). A "new" block appearing in lap 2 or 3
        # can only be a false positive -- and acting on it would wreck a good run.
        if (self.corner_count // 4) >= self.obs_freeze_lap:
            if self.obstacles is not None and len(msg.obstacles) != len(self.obstacles):
                self.get_logger().warn(
                    f"/obstacles nach Runde {self.obs_freeze_lap} ignoriert "
                    f"({len(msg.obstacles)} statt {len(self.obstacles)} gemeldet) "
                    f"-- Hindernisse sind eingefroren.")
            return

        obs = []
        for o in msg.obstacles:
            obs.append(dict(id=int(o.id), x=float(o.position.x), y=float(o.position.y),
                            color=int(o.color), wall=int(o.wall_idx)))
        changed = (self.obstacles is None or
                   {(o['id'], o['color']) for o in obs} !=
                   {(o['id'], o['color']) for o in self.obstacles})
        self.obstacles = obs
        if changed:
            self.get_logger().info(
                f"/obstacles: {len(obs)} Hindernisse " +
                ", ".join(f"id{o['id']}(w{o['wall']},"
                          f"{'gruen' if o['color']==2 else 'rot'})" for o in obs))
        if self.state in ('DRIVE', 'SCAN_PAUSE') and self.arc is not None:
            self.plan_obstacle_path()

    def plan_obstacle_path(self):
        """Plan the (x,y) polyline for the straight we are currently driving.

        Works in lane coordinates (s along the straight, q from the OUTER wall),
        then maps to map frame using the entry wall's geometry. Handles late
        detections by starting the plan at the robot's current position.
        """
        self.obs_path = None
        self.obs_path_end_q = None
        self.obs_max_slope = 0.0
        self._path_idx = 0
        if self.arc is None or self.pose is None or self.obstacles is None:
            return
        if self.lane_width is None:
            return                              # need the inner band for offsets

        idx = self.corner_idx
        w_entry = self._entry_wall_idx(idx)
        mine = [o for o in self.obstacles if o['wall'] == w_entry]
        if not mine:
            return                              # no obstacle -> normal LA line

        # lane frame: origin at the projection of the robot onto the entry wall,
        # +s along travel, +q away from the outer wall (into the lane)
        tx, ty = self.arc['travel']
        nx, ny, d = self.walls[w_entry]          # outer wall, normal points INWARD
        x, y, _ = self.pose

        def to_lane(px, py):
            s = (px - x) * tx + (py - y) * ty            # ahead of the robot
            q = (nx * px + ny * py) - d                  # distance from outer wall
            return s, q

        def to_map(s, q):
            # start from the robot's foot point on the wall, walk s along travel
            # and q along the inward normal
            fx = x - ((nx * x + ny * y) - d) * nx
            fy = y - ((nx * x + ny * y) - d) * ny
            return (fx + tx * s + nx * q, fy + ty * s + ny * q)

        obs_lane = []
        for o in mine:
            s_o, q_o = to_lane(o['x'], o['y'])
            obs_lane.append((s_o, q_o, o['color']))
        # only what is still ahead of us (plus a little behind for hysteresis)
        obs_lane = [t for t in obs_lane if t[0] > -0.10]
        if not obs_lane:
            return

        _, q_now = to_lane(x, y)
        # how far the straight still runs: up to the turn-in point T_A
        tA = self.arc['T_A']
        s_end = (tA[0] - x) * tx + (tA[1] - y) * ty
        s_end = max(s_end, max(t[0] for t in obs_lane) + 0.3)

        planner = self._obs_planner()
        pts = planner.plan(obs_lane, s_end, self.dir_step() > 0,
                           q_start=q_now, s_start=0.0,
                           q_default=(self._lane_default_offset(w_entry)
                                      or self.corner_o_in(idx)))
        self.obs_max_slope = planner.max_slope(pts)
        dense = planner.densify(pts, 0.05, skew=self.obs_skew)
        self.obs_path = [to_map(s, q) for (s, q) in dense]
        self.obs_path_end_q = dense[-1][1]
        self.get_logger().info(
            f"Hindernis-Pfad geplant (Gerade w{w_entry}, {len(obs_lane)} Hindernisse, "
            f"steilster Wechsel {self.obs_max_slope:.2f}, {len(self.obs_path)} Punkte, "
            f"Ende bei q={self.obs_path_end_q:.2f}).")

        # --- reconcile the two planners -------------------------------------
        # The obstacle path holds its pass-by offset to the end of the straight;
        # the arc was planned with its own o_in. If they disagree, the robot
        # arrives somewhere the turn-in point is not -- that is exactly the
        # "lat=0.65 -> NOTSTOP" case. Re-plan the arc onto the path's END offset
        # so the corner starts where the robot really is.
        arc_o_in = self.arc.get('o_in')
        if arc_o_in is not None and abs(arc_o_in - self.obs_path_end_q) > 0.03:
            self.get_logger().info(
                f"Bogen an Pfadende angeglichen: o_in {arc_o_in:.2f} -> "
                f"{self.obs_path_end_q:.2f} (Hindernis am Geradenende).")
            self.plan_arc(self.pose[2], o_in_override=self.obs_path_end_q)

    def _obs_planner(self):
        from ekf.obstacle_path import ObstaclePathPlanner
        w = self.lane_width[self._entry_wall_idx(self.corner_idx)]
        # the start straight is 20 cm narrower when a parking lot is placed
        if self.parking_lot_present and self.corner_count == 0:
            w = max(w - 0.20, 0.40)
        return ObstaclePathPlanner(
            lane_width=w,
            clear_before=self.obs_clear_before,
            clear_after=self.obs_clear_after,
            transition_pref=self.obs_transition_pref,
            transition_min=self.obs_transition_min,
            wall_margin=self.obs_wall_margin,
            anchor_early=self.obs_anchor_early)

    def _assert_edge_convention(self, corners, walls):
        """Verify walls[i] lies on the line through corners[i]->corners[i+1]."""
        ok = True
        for i in range(4):
            p0, p1 = corners[i], corners[(i + 1) % 4]
            nx, ny, d = walls[i]
            e0 = abs(nx * p0[0] + ny * p0[1] - d)
            e1 = abs(nx * p1[0] + ny * p1[1] - d)
            if e0 > 0.02 or e1 > 0.02:
                ok = False
                self.get_logger().error(
                    f"ASSERT: wall[{i}] passt nicht zu corners[{i}]->[{i+1}] "
                    f"(Abw {e0:.3f}/{e1:.3f} m). Kanten-Ecken-Konvention verletzt!")
        if ok:
            self.get_logger().info("Kanten-Ecken-Konvention verifiziert (walls<->corners).")

    def wall_dist_cb(self, msg):
        """Live side-wall distances [left, right] -- used ONLY on the start straight,
        before /race_direction and /corner_geometry exist. From them we derive the
        lane centre in the map frame so the robot can pull to the middle without
        knowing the drive direction (the middle needs no direction).

        Rejects implausible readings (a wall not seen -> outlier) so a single bad
        sample cannot yank the robot sideways.
        """
        if len(msg.data) < 2 or self.pose is None:
            return
        d_l, d_r = float(msg.data[0]), float(msg.data[1])
        width = d_l + d_r
        if not (self.start_lane_min <= width <= self.start_lane_max):
            return                      # implausible -> keep the last good centre
        self.wall_dist = (d_l, d_r)
        # lateral error to the lane centre: >0 means the robot is RIGHT of centre
        # (left gap bigger than right) and must move left (+y in its own frame).
        e_lat = 0.5 * (d_l - d_r)
        x, y, th = self.pose
        # left-of-travel unit normal at the current heading
        lx, ly = -math.sin(th), math.cos(th)
        # centre point = robot position shifted by e_lat to the LEFT
        self.start_center_y = (x + e_lat * lx, y + e_lat * ly)

    def obstacles_live_cb(self, msg):
        """Rohdetektionen (Roboterframe) fuer die Startgerade sammeln.

        Sofort in den MAP-Frame umgerechnet und mit Zeitstempel abgelegt: der
        Roboter faehrt zwischen zwei Meldungen rund 6 cm, im Roboterframe waere
        dieselbe Pylone also jedes Mal woanders und liesse sich nicht abstimmen.
        Nach dem Latch wird der Puffer nicht mehr gebraucht -- dann planen
        /obstacles und der Hindernispfad.
        """
        if self.pose is None or self.geometry_ready():
            return
        x, y, th = self.pose
        c, s = math.cos(th), math.sin(th)
        jetzt = self.get_clock().now().nanoseconds * 1e-9
        for o in msg.obstacles:
            self.live_obs.append((jetzt,
                                  x + c * o.position.x - s * o.position.y,
                                  y + s * o.position.x + c * o.position.y,
                                  int(o.color)))

    def _start_ausweich_y(self, cy, breite):
        """Ziel-y auf der Startgeraden, wenn ein Hindernis davor steht.

        Gibt ``(ziel_y, info)`` zurueck; ``ziel_y`` ist ``cy``, wenn nichts zu
        umfahren ist. Braucht KEINE Fahrtrichtung: die Regel lautet im
        Roboterframe "rot rechts vorbei, gruen links vorbei", und die
        Startgerade zeigt per Definition nach map +x, links ist also +y.

        Gefahren wird -- wie im Rennbetrieb, siehe ObstaclePathPlanner.
        pass_offset -- mittig zwischen Klotz und der Wand, an der vorbeigefahren
        wird. Fuer die gruene Pylone auf (0.95,-0.10) ergibt das y=+0.21, exakt
        den Wert, den der Hindernispfad spaeter selbst plant.
        """
        if not self.start_dodge or self.pose is None:
            return cy, None
        x, y, _th = self.pose
        jetzt = self.get_clock().now().nanoseconds * 1e-9
        frisch = [o for o in self.live_obs
                  if jetzt - o[0] <= self.start_dodge_window_s]
        if not frisch:
            return self._start_ausweich_halten(x, cy)

        # Kandidaten: voraus im Fenster und seitlich in der Gasse. Der Bereich
        # reicht bewusst ein Stueck nach HINTEN, damit der Versatz beim
        # Vorbeifahren gehalten und nicht mitten neben dem Klotz zurueck-
        # geschnappt wird.
        kand = [o for o in frisch
                if -self.start_dodge_back <= (o[1] - x) <= self.start_dodge_look
                and abs(o[2] - cy) <= self.start_dodge_lane]
        if not kand:
            return self._start_ausweich_halten(x, cy)

        # Abstimmen: raeumlich gruppieren und die naechstliegende Gruppe nehmen,
        # die genug Sichtungen hat. Eine einzelne Fehldetektion lenkt so nicht.
        kand.sort(key=lambda o: o[1] - x)
        gruppe, farben = [], []
        for o in kand:
            if not gruppe or abs(o[1] - gruppe[0][1]) < 0.12:
                if not gruppe or abs(o[2] - gruppe[0][2]) < 0.12:
                    gruppe.append(o)
                    farben.append(o[3])
                    continue
            if len(gruppe) >= self.start_dodge_votes:
                break
            gruppe, farben = [o], [o[3]]
        if len(gruppe) < self.start_dodge_votes:
            return self._start_ausweich_halten(x, cy)

        ox = sum(o[1] for o in gruppe) / len(gruppe)
        oy = sum(o[2] for o in gruppe) / len(gruppe)
        farbe = max(set(farben), key=farben.count)

        y_links = cy + 0.5 * breite
        y_rechts = cy - 0.5 * breite
        if farbe == OBST_GRUEN:
            ziel = 0.5 * ((oy + BLOCK_HALB) + y_links)      # links am Klotz vorbei
            seite = 'links'
        elif farbe == OBST_ROT:
            ziel = 0.5 * (y_rechts + (oy - BLOCK_HALB))     # rechts vorbei
            seite = 'rechts'
        else:
            # Farbe unklar: nicht raten, sondern auf die Seite mit mehr Platz.
            if (y_links - oy) >= (oy - y_rechts):
                ziel = 0.5 * ((oy + BLOCK_HALB) + y_links)
                seite = 'links (Farbe unklar)'
            else:
                ziel = 0.5 * (y_rechts + (oy - BLOCK_HALB))
                seite = 'rechts (Farbe unklar)'
        ziel = min(max(ziel, y_rechts + self.start_dodge_margin),
                   y_links - self.start_dodge_margin)
        info = (ox - x, oy, farbe, seite, ziel, len(gruppe))
        # Festhalten, bis der Klotz sicher hinter uns ist. Ohne das faellt
        # der Versatz genau beim Vorbeifahren weg -- dort verliert
        # /obstacles_live die Pylone, weil unter 0.17 m Laserentfernung
        # range_min_m greift -- und der Roboter zoege mitten neben dem
        # Klotz zurueck zur Spurmitte.
        self._start_halt = (ox, ziel, info)
        return ziel, info

    def _start_ausweich_halten(self, x, cy):
        """Den zuletzt bestimmten Versatz halten, solange der Klotz noch
        nicht passiert ist. Danach zurueck auf die Spurmitte."""
        halt = self._start_halt
        if halt is None:
            return cy, None
        ox, ziel, info = halt
        if x > ox + self.start_dodge_back:
            self._start_halt = None
            return cy, None
        return ziel, info

    # ------------------------------------------------------------ Ausparken
    #
    # Ablauf: AUSPARK_BUTTON -> AUSPARK_RICHTUNG -> AUSPARK_FAHREN -> weiter.
    # Waehrend AUSPARK_FAHREN wird KEIN /cmd_vel veroeffentlicht: die Bruecke
    # wuerde daraufhin die Lenkung neu stellen, und ein Motorbefehl bricht die
    # laufende Positionsfahrt ab.

    def ausparken_scan_cb(self, msg):
        """Eine Stimme fuer die Fahrtrichtung. Laeuft nur waehrend der Suche."""
        if self.state != 'AUSPARK_RICHTUNG':
            return
        ergebnis = richtung_aus_scan(
            scan_to_points(msg), halbwinkel_grad=self.ausparken_sektor_grad)
        self.ausp_letzter_grund = ergebnis['grund']
        if not ergebnis['sicher']:
            self.ausp_stimmen = []
            return
        # Nur EINIGE Stimmen zaehlen, keine Mehrheit: ein einziger Widerspruch
        # setzt zurueck. Wer den Roboter waehrend der Suche anfasst, bekommt
        # keine Entscheidung statt einer knappen.
        if self.ausp_stimmen and self.ausp_stimmen[-1] != ergebnis['richtung']:
            self.ausp_stimmen = []
        self.ausp_stimmen.append(ergebnis['richtung'])

    def ausparken_move_done_cb(self, msg):
        """Quittung der Bruecke: [move_id, status, position_zehntelgrad]."""
        if len(msg.data) >= 3:
            self.ausp_move_done = (self.now_s(), int(msg.data[0]),
                                   int(msg.data[1]), msg.data[2] / 10.0)

    def _ausparken_pid(self, flach):
        """Regelparameter des ESP setzen. Fluechtig, nicht ins NVS."""
        werte = list(flach)
        if len(werte) % 2 != 0:
            self.get_logger().warn(
                "Ausparken: PID-Liste braucht Paare aus Index und Wert, "
                "bekam %d Werte -- uebersprungen." % len(werte))
            return
        namen = {0: 'kp', 1: 'ki', 2: 'kd', 3: 'ilimit', 4: 'maxduty',
                 5: 'tol_deg', 6: 'settle_ms', 7: 'timeout_ms', 8: 'minduty'}
        gesetzt = []
        for i in range(0, len(werte), 2):
            self.pub_pid.publish(
                Float32MultiArray(data=[float(werte[i]), float(werte[i + 1])]))
            gesetzt.append('%s=%g' % (namen.get(int(werte[i]), '?%d' % werte[i]),
                                      werte[i + 1]))
        self.get_logger().info("Ausparken: Regelparameter %s"
                               % ', '.join(gesetzt))

    def _ausparken_abbruch(self, grund):
        """Fahrt abbrechen, Regelparameter zuruecksetzen, stehen bleiben."""
        self.pub_motor.publish(Int32(data=0))      # loest die Positionsfahrt ab
        self._ausparken_pid(self.ausparken_pid_nachher)
        self.publish_stop()
        self.state = 'DONE'
        self.get_logger().error("Ausparken abgebrochen: %s" % grund)

    def _ausparken_planen(self):
        """Tabelle auf die erkannte Seite drehen und den Trockenlauf mitloggen."""
        richtung = self.ausp_stimmen[-1]
        if self.ausparken_richtung_invertieren:
            richtung = 'CW' if richtung == 'CCW' else 'CCW'
            self.get_logger().warn(
                "Ausparken: Richtung per Parameter invertiert.")
        offen_links = (richtung == 'CCW')
        try:
            tabelle = schritte_aus_flach(self.ausparken_schritte)
        except ValueError as fehler:
            self._ausparken_abbruch("Schrittliste unbrauchbar: %s" % fehler)
            return
        if not tabelle:
            self._ausparken_abbruch("Schrittliste ist leer")
            return

        self.ausp_schritte = spiegeln(tabelle, offen_links)
        self.ausp_richtung = richtung

        # Trockenlauf zum Mitschreiben, immer in der Lage "offen links"
        # gerechnet: die Luecke ist spiegelsymmetrisch, die Lenkung nur fast
        # (R 0,306 m links gegen 0,312 m rechts). Fuer die Warnung reicht das.
        probe = simuliere(spiegeln(tabelle, True))
        self.get_logger().info(
            "Ausparken: %s -- offene Seite %s (%s). %d Zuege, %.0f cm Weg."
            % (richtung, 'links' if offen_links else 'rechts',
               self.ausp_letzter_grund, len(tabelle),
               sum(abs(cm) for _l, cm in tabelle)))
        self.get_logger().info(
            "Ausparken: Trockenlauf -- %s, engster Abstand %.0f mm zur "
            "Magenta-Wand, am Ende %s."
            % ('KOLLISION in Zug %s' % probe['bei_schritt'] if probe['kollision']
               else 'kollisionsfrei',
               probe['magenta_abstand_m'] * 1000,
               'frei' if probe['frei'] else 'NOCH IN DER LUECKE'))
        if probe['kollision'] or not probe['frei']:
            self.get_logger().warn(
                "Ausparken: die Schrittfolge geht rechnerisch nicht auf. "
                "Sie wird trotzdem gefahren -- die Masse der Luecke koennen "
                "von meinem Modell abweichen. Hand an den Nothalt.")

        self._ausparken_pid(self.ausparken_pid)
        self.state = 'AUSPARK_FAHREN'
        self.ausp_index = 0
        self.ausp_phase = 'lenken'
        self.ausp_lenk_gesendet = False

    def _ausparken_schritt(self, x, y, theta):
        jetzt = self.now_s()

        # --- auf den Taster warten -------------------------------------
        if self.state == 'AUSPARK_BUTTON':
            if self.require_button and not self.button_pressed:
                self.publish_stop()
                return
            self.state = 'AUSPARK_RICHTUNG'
            self.ausp_t0 = jetzt
            self.get_logger().info(
                "Ausparken: suche die offene Seite, %d einige Scans noetig."
                % self.ausparken_scans)
            return

        # --- Fahrtrichtung aus dem rohen Scan --------------------------
        if self.state == 'AUSPARK_RICHTUNG':
            self.publish_stop()
            if len(self.ausp_stimmen) < self.ausparken_scans:
                if jetzt - self.ausp_t0 > self.ausparken_richtung_timeout:
                    self._ausparken_abbruch(
                        "keine eindeutige Richtung in %.0f s -- zuletzt: %s"
                        % (self.ausparken_richtung_timeout,
                           self.ausp_letzter_grund or 'kein Scan empfangen'))
                else:
                    self.get_logger().info(
                        "Ausparken: %d/%d Stimmen -- %s"
                        % (len(self.ausp_stimmen), self.ausparken_scans,
                           self.ausp_letzter_grund or 'warte auf /scan'),
                        throttle_duration_sec=1.0)
                return
            # Erst senden, wenn die Bruecke die Topics auch abonniert hat.
            # Die allerersten Nachrichten auf einer frisch angelegten
            # Verbindung gehen in der DDS-Erkennung verloren -- und das sind
            # hier ausgerechnet pid_set (maxduty, also das Tempo) und der
            # erste Lenkbefehl. Ohne diese Pruefung faehrt Zug 1 ungebremst
            # und ohne Lenkeinschlag.
            fehlt = [name for name, pub in (
                ('steer', self.pub_steer), ('move', self.pub_move),
                ('pid_set', self.pub_pid), ('motor', self.pub_motor))
                if pub.get_subscription_count() == 0]
            if fehlt:
                if jetzt - self.ausp_t0 > self.ausparken_richtung_timeout:
                    self._ausparken_abbruch(
                        "Bruecke hoert nicht zu (%s ohne Abonnent) -- laeuft "
                        "der esp_serial_bridge?" % ', '.join(fehlt))
                else:
                    self.get_logger().info(
                        "Ausparken: warte auf die Bruecke (%s)"
                        % ', '.join(fehlt), throttle_duration_sec=1.0)
                return
            self._ausparken_planen()
            return

        # --- Scan-Halt nach dem Ausparken ------------------------------
        if self.state == 'AUSPARK_SCAN':
            self._ausparken_scanhalt(x, y, theta)
            return

        # --- die Zuege abfahren ----------------------------------------
        if self.state == 'AUSPARK_FAHREN':
            if self.ausp_index >= len(self.ausp_schritte):
                self._ausparken_fertig(x, y, theta)
                return
            lenk, cm = self.ausp_schritte[self.ausp_index]

            # Erst lenken, dann fahren. Der Servo braucht bis zum Anschlag
            # laenger als ein Regelzyklus.
            if self.ausp_phase == 'lenken':
                if not self.ausp_lenk_gesendet:
                    self.ausp_lenk_gesendet = True
                    self.ausp_t0 = jetzt
                # Waehrend der ganzen Wartezeit wiederholen statt einmalig:
                # ein verlorener Lenkbefehl faellt sonst erst auf, wenn der
                # Zug ohne Einschlag gefahren ist. Die Pakete sind 5 Byte.
                self.pub_steer.publish(Float32(data=float(lenk)))
                if jetzt - self.ausp_t0 < self.ausparken_lenk_wartezeit:
                    return
                grad = cm_zu_grad(cm)
                self.ausp_move_done = None
                self.ausp_gesendet_t = jetzt
                self.ausp_theta0 = theta
                self.ausp_pose0 = (x, y)
                self.pub_move.publish(Float32(data=float(grad)))
                self.ausp_phase = 'fahren'
                self.get_logger().info(
                    "Ausparken Zug %d/%d: Lenkung %+.0f %%, %+.1f cm "
                    "(%+.0f grad Welle)."
                    % (self.ausp_index + 1, len(self.ausp_schritte),
                       lenk, cm, grad))
                return

            # --- auf die Quittung warten --------------------------------
            quittung = self.ausp_move_done
            if quittung is not None and quittung[0] >= self.ausp_gesendet_t:
                _t, _mid, status, _pos = quittung
                plan = bahn((0.0, 0.0, 0.0), [(lenk, cm)])[-1][0]
                soll = plan[2]
                erwartet = math.hypot(plan[0], plan[1])
                ist = wrap(theta - self.ausp_theta0)
                gefahren = math.hypot(x - self.ausp_pose0[0],
                                      y - self.ausp_pose0[1])
                # Gegen die SEHNE vergleichen, nicht gegen die Bogenlaenge:
                # ueber Grund misst die Pose den direkten Abstand, und bei
                # 37 cm Vollkreisbogen sind das schon 2 cm Unterschied.
                fehlt = abs(gefahren - erwartet)

                if status == 1 and fehlt <= self.ausparken_weg_toleranz_cm / 100.0:
                    # Der ESP hat nicht eingeschwungen, ist aber weit genug
                    # gekommen. Fuer uns zaehlt der Weg.
                    self.get_logger().warn(
                        "Ausparken Zug %d: ESP meldet Zeitueberschreitung, "
                        "Weg stimmt aber (%.1f statt %.1f cm) -- weiter. "
                        "Wenn das bei jedem Zug passiert, minduty erhoehen."
                        % (self.ausp_index + 1, gefahren * 100, erwartet * 100))
                elif status != 0:
                    # MOVE_OK/TIMEOUT/ABORTED aus esp_serial_bridge.py. Status 2
                    # heisst: irgendetwas hat einen Motorbefehl geschickt und
                    # die Fahrt damit abgeloest -- der haeufigste Fall ist eine
                    # Bruecke ohne die move-Sperre in _velocity_control.
                    bedeutung = {1: "Zeitueberschreitung im ESP, und der Weg "
                                    "fehlt auch",
                                 2: "von einem Motorbefehl abgeloest -- laeuft "
                                    "die Bruecke mit der move-Sperre?"}
                    self.get_logger().error(
                        "Ausparken Zug %d NICHT ausgefuehrt: Status %d (%s). "
                        "%.1f cm ueber Grund statt %.1f cm."
                        % (self.ausp_index + 1, status,
                           bedeutung.get(status, "unbekannt"),
                           gefahren * 100, erwartet * 100))
                    self._ausparken_abbruch(
                        "Zug %d quittiert mit Status %d" % (self.ausp_index + 1, status))
                    return
                else:
                    self.get_logger().info(
                        "Ausparken Zug %d fertig: Kurs %+.1f grad (geplant "
                        "%+.1f), %.1f cm ueber Grund (geplant %.1f)."
                        % (self.ausp_index + 1, math.degrees(ist),
                           math.degrees(soll), gefahren * 100, erwartet * 100))
                self.ausp_index += 1
                self.ausp_phase = 'lenken'
                self.ausp_lenk_gesendet = False
                return

            if jetzt - self.ausp_gesendet_t > self.ausparken_zug_timeout:
                self._ausparken_abbruch(
                    "Zug %d ohne Quittung nach %.0f s -- laeuft der "
                    "esp_serial_bridge mit der move-Sperre?"
                    % (self.ausp_index + 1, self.ausparken_zug_timeout))
            return

    def _ausparken_fertig(self, x, y, theta):
        self._ausparken_pid(self.ausparken_pid_nachher)
        self.publish_stop()
        self.get_logger().info(
            "Ausparken fertig: Pose (%.2f, %.2f), Kurs %+.1f grad."
            % (x, y, math.degrees(theta)))
        if self.nur_ausparken:
            self.state = 'DONE'
            self.get_logger().info(
                "nur_ausparken gesetzt -- Regler haelt hier an.")
            return
        if self.ausparken_halt_s > 0.0:
            self.state = 'AUSPARK_SCAN'
            self.ausp_t0 = self.now_s()
            self.get_logger().info(
                "Scan-Halt: %.1f s stehen bleiben, bevor es losgeht."
                % self.ausparken_halt_s)
            return
        self._ausparken_uebergeben()

    def _ausparken_scanhalt(self, x, y, theta):
        """Stillstehen, damit die Wahrnehmung die Startgerade aufnehmen kann.

        Der Roboter steht hier zum ersten Mal in der Spur und schaut sie
        entlang. Fahrend bricht die Bildrate der Kamera von 15,5 auf 2,5 Hz
        ein und die Farbausbeute von 38 auf 2 Prozent -- die Pylonen der
        Startgeraden sind jetzt besser zu sehen als spaeter im Lauf.
        """
        self.publish_stop()
        rest = self.ausparken_halt_s - (self.now_s() - self.ausp_t0)
        if rest > 0.0:
            self.get_logger().info(
                "Scan-Halt, noch %.1f s." % rest, throttle_duration_sec=0.5)
            return
        self._ausparken_uebergeben()

    def _ausparken_uebergeben(self):
        """An die normale Zustandsmaschine abgeben."""
        # Stimmt die Richtung, die wir in der Luecke gemessen haben, mit der
        # ueberein, die die Wahrnehmung gelatcht hat? Weichen sie ab, faehrt
        # der Regler die Runde andersherum als geplant -- das muss auffallen.
        if self.race_direction and self.ausp_richtung \
                and self.race_direction != self.ausp_richtung:
            self.get_logger().error(
                "WIDERSPRUCH: beim Ausparken %s gemessen, /race_direction "
                "meldet %s. Der Latch stammt vermutlich noch aus der Zeit IN "
                "der Luecke, wo die Startpositionserkennung nichts Sinnvolles "
                "sehen kann. Der Regler folgt /race_direction."
                % (self.ausp_richtung, self.race_direction))
        elif self.ausp_richtung:
            self.get_logger().info(
                "Fahrtrichtung %s, bestaetigt durch %s."
                % (self.ausp_richtung,
                   '/race_direction' if self.race_direction else
                   'nichts weiter -- /race_direction fehlt noch'))
        # Der Taster ist bereits gedrueckt worden, sonst waeren wir nicht hier.
        self.button_pressed = True
        self.state = 'WAIT_INPUTS'
        self.get_logger().info("Weiter zum Rennen. Warte auf Eingaben...")

    def publish_lap_state(self):
        """Publish [corner_idx, corner_count, lap] for the perception side.
        corner_idx = corner currently being approached (0..3, box index)
        corner_count = corners completed so far
        lap = corner_count // 4  (0 = first lap)"""
        if self.corner_idx is None:
            return
        msg = Int32MultiArray()
        msg.data = [int(self.corner_idx), int(self.corner_count),
                    int(self.corner_count // 4)]
        self.pub_lap.publish(msg)

    def button_cb(self, msg):
        """Die Bridge publiziert bei jedem Tastendruck einen Header (kein Bool).
        Die Nachricht selbst IST das Ereignis."""
        self.button_pressed = True

    # ------------------------------------------------------------- helpers
    def publish_stop(self):
        self.pub_cmd.publish(Twist())
        self.v_cmd = 0.0
        self.last_cmd = (0.0, 0.0)

    def publish_cmd(self, v, omega):
        # Clamp by the STEERING ANGLE, not just the yaw rate: omega = v*tan(d)/L,
        # so a fixed yaw-rate limit allows physically impossible steering at low
        # speed (3 rad/s at 0.35 m/s would need 42 deg, mechanical limit is 25).
        v_eff = max(abs(v), 0.05)
        omega_steer_max = v_eff * math.tan(self.max_steer) / self.wheelbase
        limit = min(self.max_yaw_rate, omega_steer_max)
        if abs(omega) > limit:
            self.get_logger().warn(
                f"omega {omega:+.2f} auf {math.copysign(limit, omega):+.2f} begrenzt "
                f"(Lenkwinkelgrenze {math.degrees(self.max_steer):.0f} deg bei v={v_eff:.2f}).",
                throttle_duration_sec=1.0)
        omega = max(-limit, min(limit, omega))
        cmd = Twist()
        cmd.linear.x = float(v)
        cmd.angular.z = float(omega)
        self.pub_cmd.publish(cmd)
        self.last_cmd = (float(v), float(omega))

    def republish_last(self):
        """Hold the last command during a short odom gap (don't stop mid-manoeuvre)."""
        v, omega = self.last_cmd
        cmd = Twist()
        cmd.linear.x = float(v)
        cmd.angular.z = float(omega)
        self.pub_cmd.publish(cmd)

    def odom_is_stale(self):
        if self.last_odom_time is None:
            return True
        age = (self.get_clock().now() - self.last_odom_time).nanoseconds / 1e9
        return age > self.odom_timeout

    def inputs_ready(self):
        """Enough to START driving. The corner geometry and the drive direction
        only latch once the robot is CLOSE to the first corner -- so we must be
        able to drive the start straight without them (see _drive_start)."""
        return self.front_wall_x is not None

    def geometry_ready(self):
        """Everything needed for corner planning."""
        return (self.corners is not None and self.walls is not None
                and self.race_direction in ('CW', 'CCW'))

    def dir_step(self):
        return 1 if self.race_direction == 'CCW' else -1

    def pick_first_corner(self, x, y, theta):
        """Which corner is the robot heading toward.

        A corner ahead has positive projection on the travel direction. But two
        corners can share the same forward projection while one is far to the
        side -- so among the corners ahead, pick the one with the SMALLEST
        lateral offset from the travel line (closest to straight ahead).
        """
        tx, ty = math.cos(theta), math.sin(theta)
        px, py = -ty, tx                                  # left-perpendicular
        best_i, best_lat = None, 1e9
        for i, c in enumerate(self.corners):
            fwd = (c[0] - x) * tx + (c[1] - y) * ty       # along travel (ahead > 0)
            if fwd <= 0.1:
                continue
            lat = abs((c[0] - x) * px + (c[1] - y) * py)  # sideways distance
            if lat < best_lat:
                best_lat = lat
                best_i = i
        return best_i

    # ------------------------------------------------------------- arc planning
    def plan_arc(self, theta, o_in_override=None):
        """Plan the inscribed arc for the current corner_idx from the box walls.

        o_in_override: keep the entry line of the straight we are ALREADY driving
        (used when re-planning mid-straight after /inner_geometry arrives -- the
        robot must finish the straight on its current line and only change offset
        THROUGH the corner, otherwise it swerves right before turning in)."""
        s = float(self.dir_step())
        idx = self.corner_idx

        # the two walls meeting at corner idx
        wall_a = self.walls[(idx - 1) % 4]   # edge ending at corner idx (entry side)
        wall_b = self.walls[idx]             # edge starting at corner idx (exit side)

        # Orient normals inward (toward box centre) so offsetting is consistent.
        cx = sum(c[0] for c in self.corners) / 4.0
        cy = sum(c[1] for c in self.corners) / 4.0
        A = self._inward(wall_a, cx, cy)
        B = self._inward(wall_b, cx, cy)

        # Decide which is the "entry" (roughly parallel to current travel) and
        # which is the "exit" (roughly perpendicular / ahead). Entry wall's normal
        # is perpendicular to travel; exit wall's normal opposes travel.
        tx, ty = math.cos(theta), math.sin(theta)
        if abs(A[0] * tx + A[1] * ty) > abs(B[0] * tx + B[1] * ty):
            A, B = B, A   # ensure A = entry (normal perp to travel), B = exit (normal along -travel)

        o_in = self.corner_o_in(idx) if o_in_override is None else o_in_override
        o_out = self.corner_o_out(idx)
        R = self.corner_R(idx)

        # --- feasibility against the ACTUAL pose, not the ideal line ---------
        # The turn-in point sits at (corner - o_out - R) along travel. If the robot
        # is still far off the entry line, it needs longitudinal room to settle:
        # lateral error / room must stay under the slope the car can actually do.
        # Shrinking R moves T_A FORWARD and buys that room.
        if self.pose is not None and self.arc_shrink:
            for _ in range(12):
                LA_t = (A[0], A[1], A[2] + o_in)
                LB_t = (B[0], B[1], B[2] + o_out)
                P_t = line_intersect(LA_t, LB_t)
                if P_t is None:
                    break
                C_t = (P_t[0] + R * (A[0] + B[0]), P_t[1] + R * (A[1] + B[1]))
                TA_t = (C_t[0] - R * A[0], C_t[1] - R * A[1])
                px, py, _ = self.pose
                room = (TA_t[0] - px) * tx + (TA_t[1] - py) * ty
                lat_err = abs((A[0] * px + A[1] * py) - LA_t[2])
                if room <= 0.01:
                    break                      # already past it -- cannot help
                if lat_err / room <= self.max_settle_slope or R <= self.min_turn_radius:
                    break
                R = max(R - 0.05, self.min_turn_radius)
            if R < self.corner_R(idx) - 1e-6:
                self.get_logger().warn(
                    f"Anlauf zu kurz fuer Ecke {idx}: Radius {self.corner_R(idx):.2f} "
                    f"-> {R:.2f} m verkleinert, um den Einlenkpunkt erreichbar zu machen.")

        LA = (A[0], A[1], A[2] + o_in)
        LB = (B[0], B[1], B[2] + o_out)
        P = line_intersect(LA, LB)
        if P is None:
            self.get_logger().error("Eintritts-/Austrittslinie parallel -- kann Bogen nicht planen.")
            return False

        C = (P[0] + R * (A[0] + B[0]), P[1] + R * (A[1] + B[1]))
        T_A = (C[0] - R * A[0], C[1] - R * A[1])
        T_B = (C[0] - R * B[0], C[1] - R * B[1])
        a0 = math.atan2(T_A[1] - C[1], T_A[0] - C[0])

        # travel direction along THIS straight = parallel to entry wall A, sign
        # chosen to match the current heading. Derived from the box geometry, NOT
        # from the current theta -- otherwise a small heading error at plan time
        # accumulates from corner to corner (theta_target drifts over the lap).
        thx, thy = math.cos(theta), math.sin(theta)
        wa1 = (-A[1], A[0])
        travel = wa1 if (wa1[0] * thx + wa1[1] * thy) >= 0 else (A[1], -A[0])

        # exit travel direction = parallel to exit wall B, sign = the turn outcome
        u_B = (-s * (T_B[1] - C[1]) / R, s * (T_B[0] - C[0]) / R)
        # theta_target = heading of the exit straight, absolute from wall B
        wb1 = (-B[1], B[0])
        u_exit = wb1 if (wb1[0] * u_B[0] + wb1[1] * u_B[1]) >= 0 else (B[1], -B[0])
        theta_target = math.atan2(u_exit[1], u_exit[0])
        u_B = u_exit   # keep exit travel consistent with theta_target

        tx, ty = travel
        self.arc = dict(C=C, s=s, R=R, o_in=o_in, T_A=T_A, T_B=T_B, a0=a0, travel=travel,
                        LA=LA, LB=LB, u_B=u_B, theta_target=theta_target)
        corner = self.corners[idx]
        self.get_logger().info(
            f"DECIDE Ecke {self.corner_count+1}/{self.n_corners} (idx {idx}, {self.race_direction}): "
            f"Eckpunkt=({corner[0]:.2f},{corner[1]:.2f}) o_in={o_in:.2f} o_out={o_out:.2f} R={R:.2f} "
            f"[Gerade ein=w{self._entry_wall_idx(idx)} aus=w{self._exit_wall_idx(idx)}"
            + (f", Breiten {self.lane_width[self._entry_wall_idx(idx)]:.2f}/"
               f"{self.lane_width[self._exit_wall_idx(idx)]:.2f}"
               if self.lane_width is not None else "") + "] "
            f"T_A=({T_A[0]:.2f},{T_A[1]:.2f}) T_B=({T_B[0]:.2f},{T_B[1]:.2f}) "
            f"theta_target={math.degrees(theta_target):.1f}.")
        if self.debug:
            self.get_logger().info(
                f"  [GEO] travel=({tx:+.2f},{ty:+.2f}) "
                f"A(entry)=({A[0]:+.2f},{A[1]:+.2f},{A[2]:+.2f}) "
                f"B(exit)=({B[0]:+.2f},{B[1]:+.2f},{B[2]:+.2f}) "
                f"C=({C[0]:.2f},{C[1]:.2f}) "
                f"LA=({LA[0]:+.2f},{LA[1]:+.2f},{LA[2]:+.2f}) "
                f"LB=({LB[0]:+.2f},{LB[1]:+.2f},{LB[2]:+.2f})")
        return True

    @staticmethod
    def _inward(wall, cx, cy):
        """Return wall HNF with normal pointing toward (cx,cy)."""
        nx, ny, d = wall
        # signed distance of centre; if negative, flip so centre is on +normal side
        if nx * cx + ny * cy - d < 0:
            return (-nx, -ny, -d)
        return (nx, ny, d)

    # ------------------------------------------------------------- main loop
    def control_loop(self):
        if self.pose is None:
            return
        x, y, theta = self.pose

        # Vor allem anderen, und bewusst VOR der odom-Alterspruefung: die
        # Ausparkzuege laufen auf dem ESP und brauchen keine frische Pose.
        if self.state.startswith('AUSPARK'):
            self._ausparken_schritt(x, y, theta)
            return

        if self.state == 'WAIT_INPUTS':
            if not self.inputs_ready():
                return
            self.state = 'WAIT_BUTTON' if self.require_button else 'DRIVE'
            if self.state == 'DRIVE':
                self._enter_drive(x, y, theta)
            self.get_logger().info("Eingaben da. " +
                                   ("Warte auf Button..." if self.require_button else "Fahre los."))
            return

        if self.state == 'WAIT_BUTTON':
            if self.button_pressed:
                self.state = 'DRIVE'
                self._enter_drive(x, y, theta)
                self.get_logger().info("Start.")
            else:
                self.publish_stop()
            return

        if self.state == 'DONE':
            self.publish_stop()
            return

        # --- odom-stale handling: hold last cmd through short gaps, stop on long ---
        if self.odom_is_stale():
            if self.state in ('TURN', 'DRIVE'):
                self.republish_last()   # bridge past the gap; bridge watchdog is the backstop
            else:
                self.publish_stop()
            return

        if self.state == 'DRIVE':
            self._drive(x, y, theta)
        elif self.state == 'SCAN_PAUSE':
            self._scan_pause(x, y, theta)
        elif self.state == 'TURN':
            self._turn(x, y, theta)

    def _scan_pause(self, x, y, theta):
        """Stand still at the end of a straight so the perception can accumulate
        scans without motion blur. Only during the first lap(s).

        Afterwards the plan is REDONE: a block seen only during the pause changes
        o_out of this corner (and the next straight's path). We keep the entry
        line (we are on it) and hand back to DRIVE instead of turning in blindly --
        DRIVE then either covers the remaining bit to T_A or turns in at once if
        the new T_A already lies behind us.
        """
        self.publish_stop()
        left = self.scan_pause_s - (self.now_s() - self.scan_pause_t0)
        if left > 0.0:
            self.get_logger().info(
                f"SCAN-HALT ({left:.1f}s verbleibend) bei ({x:.2f},{y:.2f}).",
                throttle_duration_sec=0.5)
            return

        keep_o_in = self.arc.get('o_in') if self.arc else None
        old_TA = self.arc['T_A'] if self.arc else None
        self.arc = None
        self.plan_arc(theta, o_in_override=keep_o_in)
        self.plan_obstacle_path()
        if self.arc is not None and old_TA is not None:
            tr = self.arc['travel']
            new_TA = self.arc['T_A']
            shift = ((new_TA[0] - old_TA[0]) * tr[0] + (new_TA[1] - old_TA[1]) * tr[1])
            if abs(shift) > 0.02:
                self.get_logger().info(
                    f"Nach SCAN-HALT neu geplant: Einlenkpunkt um {shift:+.2f} m "
                    f"verschoben (o_out={self.corner_o_out(self.corner_idx):.2f}).")
        self.state = 'DRIVE'
        self.get_logger().info("SCAN-HALT fertig, Plan aktualisiert.")

    def now_s(self):
        return self.get_clock().now().nanoseconds * 1e-9

    # ------------------------------------------------------------- states
    def _enter_drive(self, x, y, theta):
        """Enter DRIVE. If the corner geometry / direction are not latched yet, we
        just drive the start straight (see _drive_start) and plan later."""
        self.drive_start_xy = (x, y)
        self.ct_integral = 0.0
        if not self.geometry_ready():
            self.get_logger().info(
                "Start ohne Kartengeometrie: fahre mittig geradeaus bis Richtung erkannt.")
            return
        if self.corner_idx is None:
            self.corner_idx = self.pick_first_corner(x, y, theta)
            if self.corner_idx is None:
                self.get_logger().warn("Keine Ecke voraus gefunden -- nehme idx 0.")
                self.corner_idx = 0
        if self.arc is None:
            self.plan_arc(theta)
        self.publish_lap_state()

    def _drive_start(self, x, y, theta):
        """Drive the start straight before the direction/geometry are latched.

        The lane centre comes from /wall_distances (left/right gaps) -- it needs NO
        drive direction, which is exactly why this works before the latch. We build
        a virtual target line through the computed centre, along the start heading,
        and feed it to the SAME verified Stanley controller.

        Safety: if the direction never latches, stop start_stop_gap before the front
        wall instead of driving into it.
        """
        # front-wall safety stop (front_wall_x is available from the very start)
        front_dist = self.front_wall_x - x - self.nose_offset
        if front_dist <= self.start_stop_gap:
            self.publish_stop()
            self.get_logger().warn(
                f"Startgerade: {front_dist:.2f} m vor Frontwand, aber keine Fahrtrichtung "
                f"erkannt. Stoppe.", throttle_duration_sec=2.0)
            return

        if self.start_center_y is None:
            # no usable wall reading yet -> hold the start heading, drive slowly on
            self.publish_cmd(self.v_start, 0.0)
            return

        # virtual line: through the lane centre, along the start heading (theta~0).
        cx, cy = self.start_center_y
        # Steht eine Pylone davor, wird die Ziellinie zur Seite geschoben --
        # gleiche Regel und gleicher Abstand wie spaeter im Hindernispfad.
        dl, dr = self.wall_dist if self.wall_dist else (float('nan'), float('nan'))
        breite = dl + dr if self.wall_dist else 1.0
        ziel_y, info = self._start_ausweich_y(cy, breite)
        if info is not None and self.start_dodge_aktiv is None:
            self.get_logger().info(
                f"Startgerade: Hindernis {info[3]} vorbei "
                f"({'gruen' if info[2] == OBST_GRUEN else 'rot' if info[2] == OBST_ROT else 'Farbe unklar'}, "
                f"{info[0]:.2f} m voraus, {info[5]} Sichtungen) -- "
                f"Ziellinie {cy:+.3f} -> {ziel_y:+.3f} m.")
        self.start_dodge_aktiv = info

        ux, uy = math.cos(0.0), math.sin(0.0)     # start straight = map +x by definition
        nx, ny = -uy, ux                           # left normal
        d = nx * cx + ny * ziel_y
        omega = self._stanley_steer(x, y, theta, (nx, ny, d), (ux, uy))

        if self.debug:
            aus = f" AUSWEICH->{ziel_y:+.3f}" if info is not None else ""
            self.get_logger().info(
                f"[START] pos=({x:+.2f},{y:+.2f}) th={math.degrees(theta):+.1f} "
                f"links={dl:.2f} rechts={dr:.2f} mitte_y={cy:+.3f}{aus} "
                f"front={front_dist:.2f} om={omega:+.2f}",
                throttle_duration_sec=0.3)

        self.publish_cmd(self.v_start, omega)

    def _speed_profile(self, dist_to_TA, dist_since_corner):
        """Distance-based speed: accelerate v_turn->v_drive over accel_dist after a
        corner, cruise v_drive, brake v_drive->v_turn over brake_dist before T_A.
        The lower of the two ramps wins (handles short straights)."""
        # acceleration ramp (grows from v_turn to v_drive over accel_dist)
        if self.accel_dist > 1e-3:
            ra = max(0.0, min(1.0, dist_since_corner / self.accel_dist))
        else:
            ra = 1.0
        v_acc = self.v_turn + ra * (self.v_drive - self.v_turn)
        # braking ramp (falls from v_drive to v_turn as dist_to_TA -> 0)
        if self.brake_dist > 1e-3:
            rb = max(0.0, min(1.0, dist_to_TA / self.brake_dist))
        else:
            rb = 1.0
        v_brk = self.v_turn + rb * (self.v_drive - self.v_turn)
        return min(v_acc, v_brk)

    def _drive(self, x, y, theta):
        """Lane-following on the current straight (Stanley holds the centre line).
        Watches the turn-in point T_A; at the last corner, stops mid-lane at
        finish_front_dist instead of turning in."""
        # --- start straight: no map geometry / direction yet -> hold lane centre ---
        if self.arc is None and not self.geometry_ready():
            self._drive_start(x, y, theta)
            return
        if self.arc is None:
            # geometry just arrived -> set up the corner now
            self._enter_drive(x, y, theta)
            if self.arc is None:
                return

        tr = self.arc['travel']
        tA = self.arc['T_A']
        # hold the entry line of THIS straight (LA); Stanley keeps us centred
        # follow the planned obstacle path if there is one, else the plain line
        omega = None
        if self.obs_path:
            omega = self._stanley_follow_path(x, y, theta, self.obs_path)
        if omega is None:
            omega = self._stanley_steer(x, y, theta, self.arc['LA'], tr)

        # signed distance to T_A along travel (positive = T_A still ahead)
        to_TA = (tA[0] - x) * tr[0] + (tA[1] - y) * tr[1]
        px, py = -tr[1], tr[0]
        lateral = abs((x - tA[0]) * px + (y - tA[1]) * py)
        # distance travelled since the corner start (for the accel ramp)
        dsc = math.hypot(x - self.drive_start_xy[0], y - self.drive_start_xy[1])

        # --- final straight: stop mid-lane instead of turning in ---
        if self.corner_count >= self.n_corners:
            # corner_idx already points at the corner ahead on THIS straight
            # (advanced at the end of the last turn); its front wall is the goal.
            fc = self.corners[self.corner_idx]
            front_dist = (fc[0] - x) * tr[0] + (fc[1] - y) * tr[1]
            if self.debug:
                self.get_logger().info(
                    f"[FINISH] pos=({x:+.2f},{y:+.2f}) th={math.degrees(theta):+.1f} "
                    f"front_dist={front_dist:+.2f} (Ziel {self.finish_front_dist:.2f}) om={omega:+.2f}",
                    throttle_duration_sec=0.2)
            # remaining distance to the STOP point, compensated for the reaction
            # lead (a tick + motor/vehicle latency): stop when the robot will be
            # AT the target after it coasts through the lead, not when it first
            # crosses the line -- otherwise it overshoots, worse at higher speed.
            v_now = max(abs(self.v_ist), 0.0)
            lead = v_now * self.finish_lead_time
            remain = front_dist - self.finish_front_dist - lead

            if remain <= self.finish_tol:
                self.state = 'DONE'
                self.publish_stop()
                self.get_logger().info(
                    f"ZIEL ({self.corner_count} Ecken, {front_dist:.2f} m vor Frontwand, "
                    f"v={v_now:.2f}). STOP.")
                return

            # look-ahead braking: v = sqrt(2*a*remain) reaches 0 exactly at the
            # target under constant decel a. Clamp to v_drive above, and to a
            # drivable crawl below so it never starves short of the point.
            v_brake = math.sqrt(2.0 * self.finish_decel * max(remain, 0.0))
            v = min(self.v_drive, v_brake)
            v = max(v, self.v_finish_min)
            self.publish_cmd(v, omega)
            return

        if self.debug:
            self.get_logger().info(
                f"[DRIVE idx{self.corner_idx}] pos=({x:+.2f},{y:+.2f}) th={math.degrees(theta):+.1f} "
                f"to_TA={to_TA:+.2f} lat={lateral:+.2f} om={omega:+.2f}",
                throttle_duration_sec=0.25)

        # --- scan pause: stand still at the end of the straight ---------------
        # ALWAYS at the same distance to the front wall, so every scan is taken
        # from the same geometry. The turn-in is SUPPRESSED until the pause has
        # happened -- otherwise T_A (which moves with o_out and R) would trigger
        # first and the stopping distance would vary from corner to corner.
        # Only in the first lap(s); from lap 2 the seat grid is filled.
        scan_pending = (self.scan_pause and not self.scan_done_this_straight
                        and (self.corner_count // 4) < self.scan_pause_laps)
        if scan_pending:
            corner = self.corners[self.corner_idx]
            front_dist = (corner[0] - x) * tr[0] + (corner[1] - y) * tr[1]
            if front_dist <= self.scan_front_dist:
                self.scan_done_this_straight = True
                self.scan_pause_t0 = self.now_s()
                self.state = 'SCAN_PAUSE'
                self.publish_stop()
                self.get_logger().info(
                    f"SCAN-HALT Start (Runde {self.corner_count // 4 + 1}): "
                    f"{self.scan_pause_s:.1f}s, Frontwand {front_dist:.2f} m "
                    f"(Soll {self.scan_front_dist:.2f}), to_TA {to_TA:+.2f} m.")
                return

        # --- turn-in when pose crosses T_A (never before the scan pause) ---
        if to_TA <= 0.0 and not scan_pending:
            if lateral > 0.6:
                self.state = 'DONE'
                self.publish_stop()
                self.get_logger().error(
                    f"NOTSTOP: Einlenkpunkt seitlich verfehlt (lat={lateral:.2f}). "
                    f"Falsche Ecke? idx {self.corner_idx}.")
                return

            # Do NOT start the arc while Stanley is still fighting a lateral error:
            # the steering has ~0.2 s dead time, so a counter-steer commanded just
            # before turn-in keeps acting INTO the first part of the arc and throws
            # the heading the wrong way. Wait until the car runs settled -- but only
            # for a limited distance, then commit anyway (geometry must not run away).
            om_last = abs(self.last_cmd[1])
            unsettled = (lateral > self.turn_in_lat_gate
                         or om_last > self.turn_in_om_gate)
            overshoot = -to_TA
            if unsettled and overshoot < self.turn_in_delay_max:
                self.get_logger().info(
                    f"Einlenken verzoegert: lat={lateral:.3f} om={om_last:.2f} "
                    f"(ueber T_A hinaus {overshoot:.2f}/{self.turn_in_delay_max:.2f} m).",
                    throttle_duration_sec=0.3)
                v = self._speed_profile(0.0, dsc)     # creep at turn speed
                self.publish_cmd(v, omega)
                return

            if unsettled:
                self.get_logger().warn(
                    f"Einlenken trotz Unruhe (lat={lateral:.3f}, om={om_last:.2f}) "
                    f"-- Verzoegerungsfenster {self.turn_in_delay_max:.2f} m aufgebraucht.")
            if overshoot > 0.03:
                # we drifted past T_A while settling -> re-plan the arc from HERE
                keep_o_in = self.arc.get('o_in')
                self.arc = None
                self.plan_arc(theta, o_in_override=keep_o_in)
                self.get_logger().info(
                    f"Bogen nach {overshoot:.2f} m Verzoegerung neu geplant.")
                if self.arc is None:
                    return
            self.state = 'TURN'
            if lateral > self.turn_in_lat_warn:
                self.get_logger().warn(
                    f"Einlenken mit Querfehler {lateral:.2f} m (> {self.turn_in_lat_warn:.2f}) "
                    f"-- dieser Fehler wandert durch die ganze Kurve.")
            self.get_logger().info(
                f"TURN: Einlenken bei ({x:.2f},{y:.2f}, {math.degrees(theta):.1f}).")
            return

        v = self._speed_profile(to_TA, dsc)
        if self.obs_path:
            # safety before speed on obstacle straights; steeper swap -> slower
            v_cap = (self.v_obstacle_steep
                     if self.obs_max_slope >= self.obs_slope_slow
                     else self.v_obstacle)
            v = min(v, v_cap)
        self.publish_cmd(v, omega)

    def _turn(self, x, y, theta):
        C = self.arc['C']; s = self.arc['s']; R = self.arc['R']
        rx, ry = x - C[0], y - C[1]
        dist = math.hypot(rx, ry) or 1e-6
        r_hat = (rx / dist, ry / dist)
        e_ct = dist - R
        t_hat = (-s * r_hat[1], s * r_hat[0])
        e_th = wrap(math.atan2(t_hat[1], t_hat[0]) - theta)
        theta_err = wrap(self.arc['theta_target'] - theta)

        v_meas = abs(self.v_ist) if abs(self.v_ist) > 0.05 else self.v_turn
        blend = max(0.0, min(1.0, abs(theta_err) / self.ff_blend)) if self.ff_blend > 1e-6 else 1.0
        omega = s * (v_meas / R) * blend + s * self.k_ct * e_ct + self.k_th * e_th

        # debug: arc cross-track on the SAME topic as the straight -> continuous plot.
        # e_ct = dist-R : >0 = robot OUTSIDE the planned circle (turning too wide).
        # arc_dist vs arc_R shows the REAL radius against the planned one.
        self.pub_e_ct.publish(Float64(data=float(e_ct)))
        self.pub_arc_dist.publish(Float64(data=float(dist)))
        self.pub_arc_R.publish(Float64(data=float(R)))
        self.pub_e_th.publish(Float64(data=float(math.degrees(e_th))))
        delta_cmd = math.atan(self.wheelbase * omega / max(abs(self.v_ist), 0.05))
        self.pub_delta.publish(Float64(data=float(math.degrees(delta_cmd))))

        if s * theta_err <= self.sweep_tol:
            # corner done: advance index, plan next arc, back to DRIVE (no stop)
            self.corner_count += 1
            self.get_logger().info(
                f"TURN fertig Ecke {self.corner_count} (theta={math.degrees(theta):.1f}, "
                f"ziel={math.degrees(self.arc['theta_target']):.1f}).")
            self.corner_idx = (self.corner_idx + self.dir_step()) % 4
            self.publish_lap_state()
            self.arc = None
            self.drive_start_xy = (x, y)
            self.ct_integral = 0.0        # fresh cross-track integrator for the new straight
            self.obs_path = None          # new straight -> plan its obstacle path below
            self.scan_done_this_straight = False
            self.plan_arc(theta)
            self.plan_obstacle_path()     # obstacles of the NEW straight
            self.state = 'DRIVE'
            return
        if self.debug:
            self.get_logger().info(
                f"[TURN idx{self.corner_idx}] pos=({x:+.2f},{y:+.2f}) th={math.degrees(theta):+.1f} "
                f"distC-R={e_ct:+.3f} th_err={math.degrees(theta_err):+.1f} blend={blend:.2f} om={omega:+.2f}",
                throttle_duration_sec=0.2)
        self.publish_cmd(self.v_turn, omega)

    # ------------------------------------------------------------- Stanley
    def _stanley_follow_path(self, x, y, theta, path_xy):
        """Follow a POLYLINE with the existing, verified Stanley controller.

        Finds the nearest segment, turns it into a line (HNF + direction) and
        hands that to _stanley_steer. So the path can bend around obstacles while
        the proven line-following maths stays untouched.

        path_xy: list of (x, y) in map frame, in driving order.
        """
        if not path_xy or len(path_xy) < 2:
            return None
        # nearest segment (search forward from the last index -- the robot only
        # moves forward, so this stays cheap)
        best_i, best_d2 = self._path_idx, float('inf')
        n = len(path_xy) - 1
        start = max(0, self._path_idx - 2)
        for i in range(start, n):
            ax, ay = path_xy[i]
            bx, by = path_xy[i + 1]
            dx, dy = bx - ax, by - ay
            seg2 = dx * dx + dy * dy
            if seg2 < 1e-12:
                continue
            t = ((x - ax) * dx + (y - ay) * dy) / seg2
            t = max(0.0, min(1.0, t))
            px, py = ax + t * dx, ay + t * dy
            d2 = (x - px) ** 2 + (y - py) ** 2
            if d2 < best_d2:
                best_d2, best_i = d2, i
        self._path_idx = best_i

        ax, ay = path_xy[best_i]
        bx, by = path_xy[best_i + 1]
        dx, dy = bx - ax, by - ay
        L = math.hypot(dx, dy) or 1e-9
        ux, uy = dx / L, dy / L
        nx, ny = -uy, ux                      # left normal of the segment
        d = nx * ax + ny * ay
        return self._stanley_steer(x, y, theta, (nx, ny, d), (ux, uy))

    def _stanley_steer(self, x, y, theta, target_line, u_dir):
        """Stanley path-following -> yaw rate.

        Cross-track is defined explicitly as the robot's offset to the LEFT of
        the travel line (positive = robot is left of the line), independent of
        the arbitrary sign of the HNF normal. A left offset needs a RIGHT
        (negative) steer to return, hence the minus on the cross-track term.

        Convention: positive angular.z / delta = LEFT (confirmed).
        """
        ux, uy = u_dir
        un = math.hypot(ux, uy) or 1e-9
        ux, uy = ux / un, uy / un
        # left-of-travel unit normal
        lx, ly = -uy, ux
        # foot of the line: any point on it. Use the HNF: closest point to origin
        # is (nx*d, ny*d); signed lateral offset of robot from line, measured
        # positive to the LEFT of travel.
        nx, ny, d = target_line
        # signed distance from robot to line along the HNF normal:
        dist_along_n = (nx * x + ny * y) - d
        # component of the HNF normal in the left-of-travel direction:
        n_dot_left = nx * lx + ny * ly
        # robot's left-offset from the line = -(signed distance) projected so that
        # +e_ct means "robot is left of the line"
        e_ct = -dist_along_n * (1.0 if n_dot_left >= 0 else -1.0)

        heading_line = math.atan2(uy, ux)
        e_theta = wrap(heading_line - theta)

        # integral of cross-track over this straight -> closes the residual that a
        # pure Stanley (P-like) leaves standing on short straights. Reset at each
        # corner exit (see _turn) so it never accumulates across the lap.
        self.ct_integral += self.k_stanley_i * e_ct * self.dt
        self.ct_integral = max(-self.i_ct_limit, min(self.i_ct_limit, self.ct_integral))

        # speed used everywhere: clamped against EKF spikes/dropouts
        v = min(max(abs(self.v_ist), 0.2), 1.2)

        # cross-track: real v in the denominator keeps the closed loop
        # speed-independent (e_ct decays with time constant 1/k_stanley).
        v_gain = self.stanley_v_ref if self.stanley_v_ref > 1e-3 else v

        # heading: scale k_heading ~ 1/v so the heading loop's time constant
        # L/(v*k_h_eff) stays constant. 0 -> no scaling.
        v_ref_h = self.k_heading_v_ref if self.k_heading_v_ref > 1e-3 else v
        k_h_eff = self.k_heading * (v_ref_h / v)

        delta = (k_h_eff * e_theta + math.atan2(self.k_stanley * e_ct, v_gain) + self.ct_integral)
        delta = max(-self.max_steer, min(self.max_steer, delta))
        omega = v * math.tan(delta) / self.wheelbase

        # debug publish for Foxglove
        #self.pub_e_ct.publish(Float64(data=float(e_ct)))
        self.pub_e_th.publish(Float64(data=float(math.degrees(e_theta))))
        self.pub_delta.publish(Float64(data=float(math.degrees(delta))))
        self.pub_k_h.publish(Float64(data=float(k_h_eff)))
        return omega


def main(args=None):
    rclpy.init(args=args)
    node = Round1Controller()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.publish_stop()
        except Exception:
            pass
        if node.context.ok():
            node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()