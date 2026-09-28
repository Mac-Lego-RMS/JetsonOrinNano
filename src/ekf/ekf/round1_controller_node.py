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
             Siehe ekf/ausparken.py. Mit ausparken_nur haelt der Regler danach
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
import sys
import time
import math

import rclpy
import rclpy.logging
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy
from rcl_interfaces.msg import SetParametersResult
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Bool, Float64, String, Int32MultiArray, Float64MultiArray
from std_msgs.msg import Float32, Float32MultiArray, Header, Int32

from ekf.schaetzung_neustart import neu_starten
from ekf.ausparken import (FZ_BREITE, FZ_HECK, FZ_NASE, LUECKE_LAENGE, LUECKE_TIEFE,
                           LENK_KENNLINIE, RADSTAND,
                           bahn, cm_zu_grad, grad_zu_cm, richtung_aus_scan,
                           schritte_aus_flach, schritte_fuer, simuliere,
                           spiegeln, einparkfolge)
from ekf.wall_extraction import scan_to_points
import ekf.ausparken as _ausparken_modul
# Folgen einzeln und robust holen: ein fehlender Name in ausparken.py soll den
# Regler nicht am Start hindern (so geschehen, als SCHRITTE_CW/CCW beim
# Einfuehren der Varianten entfernt wurden). Die normalen Folgen sind die
# Referenz fuers Einparken -- Ersatz: CW -> CW_AUSSEN, CCW -> STANDARD.
SCHRITTE_STANDARD = list(getattr(_ausparken_modul, 'SCHRITTE_STANDARD', []))
_cw_name = next((n for n in ('SCHRITTE_CW', 'SCHRITTE_CW_AUSSEN', 'SCHRITTE_STANDARD')
                 if getattr(_ausparken_modul, n, None)), None)
_ccw_name = next((n for n in ('SCHRITTE_CCW', 'SCHRITTE_STANDARD')
                  if getattr(_ausparken_modul, n, None)), None)
SCHRITTE_CW = list(getattr(_ausparken_modul, _cw_name)) if _cw_name else []
SCHRITTE_CCW = list(getattr(_ausparken_modul, _ccw_name)) if _ccw_name else []
# Innen-Folgen sind neu -- eine aeltere ausparken.py ohne sie laeuft trotzdem.
# Ausparkfolgen je nach Pylone in der mittleren Reihe der Startgeraden:
# innen / aussen / mitte (Reihe frei). Fehlt eine, faehrt er die normale.
AUSPARK_VARIANTEN = {
    (r, v): list(getattr(_ausparken_modul, 'SCHRITTE_%s_%s' % (r, v.upper()), []))
    for r in ('CW', 'CCW') for v in ('innen', 'aussen', 'mitte')
}


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
        'v_obstacle':       ('v_obstacle',       0.55, float),  # speed on straights with obstacles
        'obs_slope_slow':   ('obs_slope_slow',   0.80, float),  # above this slope -> v_obstacle_steep
        'v_obstacle_steep': ('v_obstacle_steep', 0.55, float),  # speed for steep lane changes
        'parking_lot_present': ('parking_lot_present', 0.0, lambda v: bool(float(v))),
        # planning against the ACTUAL pose (not the ideal line)
        'arc_shrink':       ('arc_shrink',       1.0, lambda v: bool(float(v))),  # shrink R if the run-up is too short
        'min_turn_radius':  ('min_turn_radius',  0.30, float),
        'max_settle_slope': ('max_settle_slope', 0.80, float),  # lateral m per longitudinal m we trust
        'turn_in_lat_warn': ('turn_in_lat_warn', 0.10, float),  # warn above this lateral error at turn-in
        # do not start the arc while still correcting laterally (0.2 s steering dead time)
        # 3 -> 7 cm: mit 3 cm bremste er auf der Startgeraden und auf w0 bei
        # fast jeder Runde 0,5 m vor der Ecke auf v_settle (5-7 cm neben der
        # Linie nach Spurwechsel/Kurvenausgang, parken_test_24). Solche
        # Einfahrten faengt inzwischen die Verankerung des Bogens ab.
        'turn_in_lat_gate': ('turn_in_lat_gate', 0.07, float),  # settled below this lateral error
        'turn_in_om_gate':  ('turn_in_om_gate',  0.50, float),  # ... and below this commanded omega
        'turn_in_settle_window':('turn_in_settle_window',0.50, float),  # slow down within this distance to T_A
        'v_settle':         ('v_settle',         0.25, float),  # speed while settling before the corner
        'turn_in_past_max': ('turn_in_past_max', 0.50, float),  # Plausibilitaet: weiter hinter T_A -> Nothalt
        # Bogen an der Ist-Pose: so nah an der Aussenbande darf er hoechstens
        # auf die naechste Gerade kommen, wenn der kleinste Radius noetig ist.
        'turn_anchor_min_out': ('turn_anchor_min_out', 0.20, float),
        # Pylonen am Kurvenein-/ausgang: den Bogen gegen den Fahrzeugumriss
        # pruefen und den Radius so waehlen, dass mindestens dieser Abstand
        # bleibt (parken_test_20: R 0,50 liess 3 mm zur roten Pylone am
        # Kurvenausgang, er hat sie mitgeschoben). Gesucht wird zwischen
        # min_turn_radius und bogen_pylonen_r_max, am naechsten am geplanten R --
        # ob kleiner oder groesser hilft, haengt von Farbe und Drehrichtung ab.
        # 4 -> 8 cm: gefahren liegt der Bogen einige cm enger als geplant
        # (parken_test_23: geplant 5 cm zu #19, gefahren 0,1-0,5 cm, gestreift)
        'bogen_pylonen_abstand': ('bogen_pylonen_abstand', 0.08, float),
        'bogen_pylonen_r_max':   ('bogen_pylonen_r_max',   0.70, float),
        # Auch beim puenktlichen Einlenken verankern, wenn die Einfahrt gestoert
        # ist (Kursfehler zur Kreistangente oder Querversatz ueber diesen Werten).
        'turn_anchor_puenktlich': ('turn_anchor_puenktlich', 1.0, lambda v: bool(float(v))),
        'turn_anchor_kurs_deg': ('turn_anchor_kurs', 3.0, lambda v: math.radians(float(v))),
        'turn_anchor_quer':     ('turn_anchor_quer', 0.02, float),
        # Totzeit der Lenkung: 241 ms vom /cmd_vel-Befehl bis zur Gierrate
        # (Kreuzkorrelation, Lenkverstaerkung 0,84). Der kurze Radstand macht
        # den Kursintegrator schnell (3,5 rad/s Gierrate je rad Lenkwinkel);
        # mit 241 ms bleiben nur ~28 grad Phasenreserve -- jede Anregung
        # klingelt ueber Sekunden aus (Schwingdauer ~4 x Totzeit = 1 s).
        # Stanley rechnet deshalb mit der Pose bei Wirkbeginn. Simuliert: statt
        # +-18 grad Lenkungspendeln +-2 grad, kein Klingeln. k_heading NICHT
        # senken -- dann uebernimmt der Querterm und es wird schlechter.
        # ACHTUNG: nicht zusaetzlich pose_extrapolate_s im Fusionsknoten
        # setzen, sonst wird die Totzeit doppelt vorausgerechnet. 0 = aus.
        'steer_dead_time':  ('steer_dead_time',  0.260, float),   # Lauf 2: 260 ms (Lauf 1: 241)
        'steer_gain_pred':  ('steer_gain_pred',  0.84, float),
        # Lenk-Pose: die Korrekturen der Lokalisierung (Wandabgleich, 1-1,5 cm
        # bzw. ~1 grad, mehrmals pro Sekunde) gingen als Sprung in den
        # Lenkbefehl -- 1 cm quer = 2-3 grad Lenkschritt, das hektische Zucken
        # auf ruhigen Geraden (parken_test_21, 51,0 s). Fuer das Lenkgesetz
        # werden sie ueber lenk_pose_tau eingeblendet; die Bewegung selbst
        # (v, Gierrate aus /ekf/odom) geht ohne Verzoegerung durch, also keine
        # zusaetzliche Totzeit im Regelkreis. Groessere Spruenge (Kartenwechsel)
        # werden sofort uebernommen. Ausloeser (T_A, Halte) nutzen weiter die
        # ungeglaettete Pose. 0 = aus.
        'lenk_pose_tau':    ('lenk_pose_tau',    0.30, float),
        'lenk_pose_sprung': ('lenk_pose_sprung', 0.10, float),
        'lenk_pose_sprung_grad': ('lenk_pose_sprung_grad', 8.0, lambda v: math.radians(float(v))),
        # Hindernispfad: Kruemmung vorsteuern (delta_ff = atan(L*kappa)) und die
        # Tangente zwischen den Pfadpunkten stufenlos interpolieren. Ohne
        # Vorsteuerung folgte Stanley jedem Spurwechsel nur ueber den Fehler --
        # mit 0,26 s Totzeit ueberschwingt das und muss zurueckgelenkt werden.
        # Faktor auf die Vorsteuerung, 0 = aus.
        'pfad_vorsteuerung': ('pfad_vorsteuerung', 1.0, float),
        # Ecke 1 direkt nach dem Ausparken: liegt der Einlenkpunkt mehr als
        # erste_ecke_zurueck_min HINTER ihm, erst gerade zuruecksetzen (ESP-
        # Positionsfahrt) statt den Bogen mit dem kleinsten Radius an der
        # Ist-Pose zu verankern -- der kam 25 cm zu weit aussen heraus, und vor
        # einer gruenen Pylone gab das einen Haken bis 47 grad ueber die Gerade
        # (parken_test_21). Nur wenn der Rueckweg frei ist.
        # Einpark-Test: der Roboter steht am Anfang der letzten Geraden (=
        # Startgerade), kein Ausparken, keine Runden -- nur Zielgerade und
        # einparken. Die Luecke kennt er nicht aus einer Buchtmessung: ihre Lage
        # kommt aus test_bucht_front (Abstand Hinterachse in der Luecke zur
        # Frontwand, 0 = Mittel frueherer Laeufe: CCW 1,245 m, CW 1,96 m) und
        # test_bucht_q (Abstand zur Aussenbande, Mittel 0,159 m). Die Einpark-
        # Startpose daraus wie im Rennen (park_std_*), dazu einpark_versatz_*_cw/_ccw.
        # Richtung: test_richtung. Startet den scan_processor passend neu.
        'einparken_test':   ('einparken_test',   0.0, lambda v: bool(float(v))),
        'test_bucht_front': ('test_bucht_front', 0.0, float),
        'test_bucht_q':     ('test_bucht_q',     0.159, float),
        'erste_ecke_zurueck':     ('erste_ecke_zurueck',     1.0, lambda v: bool(float(v))),
        'erste_ecke_zurueck_min': ('erste_ecke_zurueck_min', 0.10, float),
        'erste_ecke_zurueck_max': ('erste_ecke_zurueck_max', 0.80, float),
        # ... und nur, wenn der Bogen an der Ist-Pose so weit aussen herauskaeme,
        # dass der Rueckweg auf die Spur vor der ersten Pylone der naechsten
        # Geraden steiler wuerde als das (quer/laengs). parken_test_23: 15 cm
        # auf 65 cm = 0,23 haette gereicht, er setzte trotzdem zurueck.
        'erste_ecke_zurueck_steigung': ('erste_ecke_zurueck_steigung', 0.50, float),
        # Einlenken um die Totzeit frueher: der Befehl wirkt erst nach
        # steer_dead_time, das Lenkgesetz rechnet schon mit der Pose von dann.
        # Beim Umschalten genau an T_A lag diese Pose 9 cm auf der Geraden
        # hinter dem Bogenanfang, ~10 grad hinter der Kreistangente -- er schlug
        # erst ~20 grad ein und ging dann auf die 13 grad des Radius zurueck.
        'einlenk_vorhalt': ('einlenk_vorhalt', 1.0, lambda v: bool(float(v))),
        # scan pause at the end of each straight (lap 1 only -- after that the
        # seat grid is filled and standing still would only cost time)
        'scan_pause':       ('scan_pause',       1.0, lambda v: bool(float(v))),
        'scan_pause_s':     ('scan_pause_s',     1.5, float),   # how long to stand still [s]
        'scan_front_dist':  ('scan_front_dist',  1.10, float),  # ALWAYS stop this far from the front wall (pose)
        # So weit rollt er nach dem Haltbefehl noch (gemessen ~13 cm). Der Halt
        # wird um diesen Weg frueher ausgeloest, sonst steht er genau am
        # Einlenkpunkt und beschleunigt erst in der Kurve.
        'scan_nachlauf':    ('scan_nachlauf',    0.13, float),
        # Der Nachlauf haengt am Tempo (parken_test_38-42, Ausloesung immer bei
        # 1,22 m): 0,25 m/s -> 8 cm, 0,37 -> 17 cm, 0,45 -> 21 cm; er stand
        # zwischen 1,00 und 1,15 m. Modell: Totzeit + Bremsweg,
        #   Nachlauf = v * scan_nachlauf_t + v^2 / (2 * scan_brems_a)
        # (bei 0,35 m/s = 14 cm). scan_brems_a <= 0 -> fest scan_nachlauf.
        'scan_nachlauf_t':  ('scan_nachlauf_t',  0.10, float),
        'scan_brems_a':     ('scan_brems_a',     0.57, float),
        # Vorausschauend auf den Haltepunkt abbremsen (wie am Ziel): kommt er
        # so immer mit ~0,15 m/s an, ist der Nachlauf klein und gleich. 0 = aus.
        'scan_verzoegerung': ('scan_verzoegerung', 0.50, float),   # m/s^2
        'scan_pause_laps':  ('scan_pause_laps',  1,    int),    # pause only during the first N laps
        'v_start':       ('v_start',       0.35,  float),   # speed on the start straight (before direction latch)
        'start_stop_gap': ('start_stop_gap', 0.50, float),  # stop this far from the front wall if direction never comes
        'start_lane_min': ('start_lane_min', 0.45, float),  # plausibility band for d_left+d_right
        # Steht der Roboter beim Losfahren dichter als das an einer Wand, ist
        # das keine Spur mehr. In der Parkluecke misst er 0.15 zur Aussenwand
        # gegen 0.83 ins Feld -- die Summe liegt im Plausibilitaetsband, die
        # Aufteilung nicht. Nur eine Warnung: er koennte auch schief stehen.
        'start_wand_warn': ('start_wand_warn', 0.30, float),
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
        # Vorsteuerung ueber die letzten so viel Grad ausblenden. 7 grad waren
        # bei 1,5 rad/s nur 80 ms -- weniger als die 260 ms Totzeit, er drehte
        # nach dem Kurvenende 13-34 grad weiter. Simuliert: 20 grad -> 1 statt 6,5.
        'ff_blend_deg':  ('ff_blend',      20.0, lambda v: math.radians(float(v))),
        # Kurvenregler mit der Pose bei Wirkbeginn rechnen (wie auf der Geraden)
        # und das Kurvenende am vorausberechneten Kurs festmachen.
        'turn_praediktion': ('turn_praediktion', 1.0, lambda v: bool(float(v))),
        # Kurvenbefehl kruemmungsbasiert und passend zur Umrechnung der Bruecke
        # (siehe _turn). 0 = alte Formel.
        'turn_kruemmung':   ('turn_kruemmung',   1.0, lambda v: bool(float(v))),
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
        'v_drive':       ('v_drive',       0.75,  float),   # straight cruise
        'v_turn':        ('v_turn',        0.55,  float),   # through the arc
        'accel_dist':    ('accel_dist',    0.2,   float),   # ramp v_turn->v_drive after a corner
        'brake_dist':    ('brake_dist',    0.2,   float),   # ramp v_drive->v_turn before T_A
        # lap / finish
        'n_corners':     ('n_corners',     4,     int),
        'finish_front_dist': ('finish_front_dist', 1.5, float),
        'finish_decel':  ('finish_decel',  0.8,   float),   # look-ahead brake decel [m/s^2]
        'finish_lead_time': ('finish_lead_time', 0.15, float),  # reaction lead [s] -> stops on point
        'v_finish_min':  ('v_finish_min',  0.15,  float),   # DRIVABLE crawl, just above deadband
        # Letzte Kurve und Zielgerade langsamer: dort wird der Haltepunkt
        # getroffen und danach eingeparkt, und ein Fehler ist nicht mehr
        # aufzuholen. 0 schaltet die Kappe ab.
        'v_ziel':        ('v_ziel',        0.30,  float),
        'finish_tol':    ('finish_tol',    0.04,  float),   # stop tolerance on front_dist
        # --- Ausparken aus der Startluecke -----------------------------------
        # Die beiden Magenta-Waende stehen senkrecht auf dem Aussenwall und
        # ragen 20 cm ins Feld; die Luecke ist der 26,25 cm breite Spalt
        # dazwischen. Der Roboter steht laengs darin und muss quer heraus.
        # Die ganze Rechnerei steckt in ekf/ausparken.py, hier nur die Schalter.
        # ausparken, ausparken_nur und ausparken_richtung_invertieren sind
        # ECHTE Bool-Parameter und stehen weiter unten bei require_button --
        # damit "-p ausparken_nur:=true" tut, was man erwartet. Alles in
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
        # --- Einparken am Ende --------------------------------------------
        # Das Regelwerk verlangt nach drei Runden 3 s Stillstand, erst danach
        # darf eingeparkt werden. Reserve drauf, damit ein langsamer Takt die
        # 3 s nicht knapp unterschreitet.
        # 0 = KEIN Halt: die Zeit laeuft bis er in der Luecke steht, eine
        # Pflichtpause gibt es laut Regelwerk nicht. > 0 = altes Verhalten
        # (anhalten, so lange warten, dann einparken).
        'einparken_halt_s':      ('einparken_halt_s',      0.0,  float),
        # Ab hier sind die Seiten an den Pylonen frei (drei Runden vorbei):
        # base_link so weit vor der Frontwand -- das Heck ist dann an der
        # Pylonenreihe am Anfang der Startgeraden (2 m) vorbei.
        'seiten_frei_ab':        ('seiten_frei_ab',        1.915, float),
        # Wie genau die Einpark-Startpose getroffen sein muss, bevor die
        # umgekehrte Ausparkfolge startet. Ein Kursfehler dreht die GANZE
        # Folge mit -- 2 Grad sind ueber ~50 cm Rangierweg schon 1,7 cm.
        'einparken_quer_tol':    ('einparken_quer_tol',    0.025, float),
        'einparken_kurs_tol_grad': ('einparken_kurs_tol_grad', 2.5, float),
        # Plausibilitaet: laenger darf die gerade Anfahrt nicht sein.
        'einparken_max_anfahrt': ('einparken_max_anfahrt', 1.20, float),
        # Steilster Rueckschwenk auf die Parklinie nach dem letzten Hindernis
        # (Meter quer pro Meter laengs). Gemessen sauber: ~1,0.
        'einparken_rueck_steigung_max': ('einparken_rueck_steigung_max', 0.90, float),
        # Anfahrt zur Startpose: so genau muss sie laengs stimmen, so viele
        # gerade Korrekturzuege sind erlaubt, so lange wird vor dem
        # Nachmessen gewartet (EKF soll ruhen).
        'einparken_laengs_tol':  ('einparken_laengs_tol',  0.010, float),
        'einparken_anfahrt_max_zuege': ('einparken_anfahrt_max_zuege', 3, int),
        'einparken_nachmess_s':  ('einparken_nachmess_s',  0.3,  float),
        # So lange wird im Stand gemittelt (nach der Beruhigungszeit).
        'einparken_mittel_s':    ('einparken_mittel_s',    0.5,  float),
        # Rueckfall fuer die Parklinie, falls /wall_distances im Scan-Halt
        # nichts Brauchbares liefert. Handmessung base_link -> Aussenbande am
        # Ende des Ausparkens (Radnabe + halbe Spurweite).
        'einparken_linie_cw':    ('einparken_linie_cw',    0.370, float),
        'einparken_linie_ccw':   ('einparken_linie_ccw',   0.345, float),
        # Nur wenn Ecke 1 nach dem Ausparken NAEHER als das liegt, wird am
        # Ausparkende gescannt (voller Halt ausparken_halt_s, ersetzt den
        # Scan-Stopp vor Ecke 1). Sonst faehrt er gleich los und scannt
        # regulaer am Ende der Geraden -- ein Scan-Stopp pro Gerade.
        'ausparken_scan_ersetzt_bis': ('ausparken_scan_ersetzt_bis', 1.10, float),
        # Mindestens so lange steht er nach dem Ausparken trotzdem: die
        # Parklinie wird hier im Stand per Lidar gemessen.
        'ausparken_mess_s':      ('ausparken_mess_s',      0.5,  float),
        # So lange wird hoechstens auf die Eckengeometrie gewartet. Ohne sie
        # faehrt er blind zur Spurmitte -- mit Parkluecke stehen die Zeichen
        # der Startgeraden innen bei 0,60 m, das waeren ~2 cm Luft.
        'ausparken_geo_timeout_s': ('ausparken_geo_timeout_s', 4.0, float),
        # --- Start aus der Bucht (Perzeption mit start_from_bay) ------------
        # Erst ausparken, wenn /race_direction da ist: die Perzeption mittelt
        # die Pose ueber 5 Scans im Stand und committet dann. Faehrt er vorher
        # los, passen Messung und Kartenverankerung nicht zusammen. Kommt nach
        # dieser Zeit nichts, gilt die eigene Messung (Perzeption ohne
        # start_from_bay).
        'ausparken_warte_auf_richtung_s': ('ausparken_warte_auf_richtung_s', 3.0, float),
        # Auf /start_scan_state 'complete' warten, bevor er losfaehrt -- jede
        # Bewegung bricht die Abtastung der Startgeraden ab. Kommt das Topic
        # nicht (Perzeption ohne start_from_bay), nach dieser Zeit weiter.
        'ausparken_warte_scan_s': ('ausparken_warte_scan_s', 3.0, float),
        # Die Ausparkfolge richtet sich nach der NAECHSTEN Pylone VOR dem
        # geparkten Roboter, so weit voraus (ab base_link). NICHT nach einer
        # festen Reihe: die Luecke liegt je nach Aufbau woanders (Starts bei
        # 1,965 / 1,728 / 1,247 m vor der Frontwand gesehen) -- mit fester Reihe
        # wurde eine gruene Pylone 28 cm vor ihm uebersehen und er parkte auf der
        # falschen Seite aus. Bis 0,75 m prueft die Perzeption aus der Luecke.
        # Was neben oder hinter ihm steht, zaehlt nicht.
        'ausparken_entscheid_ab':  ('ausparken_entscheid_ab',  0.10, float),
        'ausparken_entscheid_bis': ('ausparken_entscheid_bis', 0.75, float),
        # Endlage der NORMALEN Ausparkfolge relativ zur Startlage in der Luecke
        # (gemessen). Daraus die Einpark-Startpose, wenn die Innen-Folge gefahren
        # wurde -- eingeparkt wird immer mit der umgekehrten normalen Folge.
        'park_std_laengs_cw':    ('park_std_laengs_cw',    0.315, float),
        'park_std_quer_cw':      ('park_std_quer_cw',      0.227, float),
        'park_std_laengs_ccw':   ('park_std_laengs_ccw',   0.276, float),
        'park_std_quer_ccw':     ('park_std_quer_ccw',     0.202, float),
        # Handversatz der Einpark-Startpose [m], nach allem anderen
        # draufgerechnet -- egal ob normal oder mit einer Variante ausgeparkt
        # wurde. laengs: + = weiter in Fahrtrichtung (zur naechsten Ecke hin).
        # quer: + = weiter weg von der Aussenbande (ins Feld). Quer wandert die
        # Parklinie mit, an der die Anfahrt ausgerichtet wird.
        # je Richtung: CCW (bisheriger Wert) und CW
        'einpark_versatz_laengs_ccw': ('einpark_versatz_laengs_ccw', 0.06, float),
        'einpark_versatz_quer_ccw':   ('einpark_versatz_quer_ccw',   -0.05, float),
        'einpark_versatz_laengs_cw':  ('einpark_versatz_laengs_cw',  0.0, float),
        'einpark_versatz_quer_cw':    ('einpark_versatz_quer_cw',    -0.03, float),
        # --- /localization_state -------------------------------------------
        # Bei 'recovering'/'lost' hoechstens so schnell (Kruemmung bleibt gleich).
        'v_lok_unsicher':        ('v_lok_unsicher',        0.20, float),
        # So lange darf 'lost' beim Fahren anhalten, dann Stopp. Die
        # Perzeption faengt Fehler bis ~0,35 m wieder ein; bei 0,20 m/s sind
        # 2 s schon 0,40 m Blindflug. 0 = nie stoppen.
        'lok_lost_stopp_s':      ('lok_lost_stopp_s',      2.0,  float),
        # So lange wartet das Einparken im Stand auf 'ok', dann kein Einparken.
        'einparken_lok_warte_s': ('einparken_lok_warte_s', 3.0,  float),
        # Kurskorrektur beim Einparken ueber die Laenge der Vollanschlag-Boegen.
        # Kleine Lenkwinkel verschluckt das Lenkspiel (2-5 grad), der Anschlag
        # nicht. Nur solange die Sensoren reichen: Lokalisierung 'ok' und der
        # Lidar noch aus der Bucht heraus (mehr als so weit von der Aussenbande).
        'einparken_korr_min_abstand': ('einparken_korr_min_abstand', 0.23, float),
        # Bogen hoechstens um diesen Anteil laenger/kuerzer.
        'einparken_korr_max':    ('einparken_korr_max',    0.40, float),
        # Kursabweichung zur Ausparkbahn ab der gewarnt wird (grad).
        'einparken_kurs_warn_grad': ('einparken_kurs_warn_grad', 3.0, float),
        # Haltepunkt am Ziel moeglichst an die Einpark-Startpose legen, damit
        # die Anfahrt danach kurz ist. Nur innerhalb der erlaubten Zone.
        'ziel_an_parkstart':     ('ziel_an_parkstart',     1.0, lambda v: bool(float(v))),
        # Pflichthalt so FRUEH wie erlaubt: am hinteren Zonenrand, gleich nach
        # der letzten Kurve. Danach ist die Seite an den Pylonen frei, und die
        # Fahrt zur Startpose laeuft geregelt. Hat Vorrang vor ziel_an_parkstart.
        'ziel_frueh':            ('ziel_frueh',            1.0, lambda v: bool(float(v))),
        'ziel_zone_min':         ('ziel_zone_min',         1.00, float),  # m zur Frontwand
        'ziel_zone_max':         ('ziel_zone_max',         2.00, float),
        'ziel_zone_rand':        ('ziel_zone_rand',        0.05, float),  # Sicherheitsrand
        # Zone gilt fuers GANZE Fahrzeug (Nase bis Heck), nicht nur base_link.
        'ziel_zone_ganzes_fz':   ('ziel_zone_ganzes_fz',   1.0, lambda v: bool(float(v))),
        # Anfahrt zur Startpose ab dieser Strecke vorwaerts GEREGELT (Stanley)
        # statt als blinder gerader ESP-Zug.
        'einparken_fahrt_ab':    ('einparken_fahrt_ab',    0.08, float),
        'v_park_fahrt':          ('v_park_fahrt',          0.30, float),   # Zeit zaehlt bis eingeparkt
        # Ab dem Ausgang der letzten Kurve bis zur Einpark-Startpose (ohne
        # Pflichthalt). Mit 0.30 m/s blieben nach der Kurve nur ~0.5 m zum
        # Einschwingen: Kurs pendelte auf +10..+15 grad und kam schief an.
        'v_park_anfahrt':        ('v_park_anfahrt',        0.15, float),
        'steiler_pfad_ab':       ('steiler_pfad_ab',       0.60, float),   # quer je laengs
        'v_steiler_pfad':        ('v_steiler_pfad',        0.35, float),
        # Startpose um so viel UEBERFAHREN, anhalten und geregelt rueckwaerts
        # (PARK_RUECK) zurueck. Vorwaerts bleiben nach der letzten Kurve nur
        # ~0,3 m bis zur Startpose -- zu wenig, um Kurs und Querlage
        # einzuschwingen (Lauf 35: +20 grad an der Startpose). Rueckwaerts
        # regelt er Kurs UND Querlage stetig nach. 0 = aus (wie bisher).
        'einparken_ueberfahren': ('einparken_ueberfahren', 0.30, float),
        # Die geregelte Anfahrt haelt so weit VOR der Startpose an. Nach dem
        # Stopp rollt er noch nach (Totzeit + Bremsweg) und lag in den Laeufen
        # 9/15/16 dadurch 1,7-4 cm HINTER der Startpose -- der Restzug in
        # PARK_NACHMESSEN war immer ein Stueck rueckwaerts. Mit Vorhalt bleibt
        # ein kurzer VORWAERTS-Rest, den der ESP als Positionsfahrt auf den
        # Millimeter faehrt (nachgemessen: jeder Zug +-0,1 cm).
        'einparken_vorhalt':     ('einparken_vorhalt',     0.15, float),
        # Gerader ESP-Zug bei der Anfahrt: die ACHSE trifft auf 0,1 mm, der
        # Wagen faehrt ueber Grund aber weiter -- rund 6 Prozent (R_EFF) und
        # vorwaerts noch ~1 cm Ueberschuss. parken_test_18 per Frontwand im
        # Lidar: +5,74 cm befohlen -> 7,1 cm gefahren, -1,47 -> 1,8 cm. Der
        # Befehl wird darum verkuerzt: vorwaerts (d - ueberschuss) / skala,
        # rueckwaerts d / skala. Sonst folgt auf jeden Anfahrtszug ein
        # Rueckwaertszug.
        'einparken_zug_skala':      ('einparken_zug_skala',      1.06,  float),
        'einparken_zug_ueberschuss': ('einparken_zug_ueberschuss', 0.010, float),
        # --- Rueckwaerts zur Startpose, stetig geregelt ----------------------
        # Der ESP faehrt die Strecke als EINEN Zug, der Controller lenkt dabei
        # laufend ueber die Rohlenkung nach (Hinterachse auf der Parklinie).
        # Gesetz rueckwaerts: delta = k_kurs*psi - k_quer*e (stabil, auf die
        # Strecke bezogen tempounabhaengig, Einschwinglaenge gut 10 cm).
        # Simuliert mit Totzeit und 4 grad Lenkspiel: nach 45 cm <= 0,4 cm /
        # 1 grad; blinder gerader Zug: 3,4 cm / 8,6 grad.
        'einparken_rueck_ab':    ('einparken_rueck_ab',    0.08, float),
        'rueck_k_kurs':          ('rueck_k_kurs',          1.5,  float),
        'rueck_k_quer':          ('rueck_k_quer',          6.7,  float),
        'rueck_max_lenk_grad':   ('rueck_max_lenk_grad',   18.0, float),
        # Totzeit-Vorausberechnung rueckwaerts: bei Rangiertempo nur ~4 cm
        # Weg, simuliert eher schaedlich -> aus.
        'rueck_praed_s':         ('rueck_praed_s',         0.0,  float),
        # Einparkfolge = umgekehrte Ausparkfolge. Die ESP-Zuege fahren aber
        # vorwaerts ~1,1 cm, rueckwaerts nur ~0,4 cm weiter als befohlen -- die
        # Umkehr hebt das NICHT auf, er steht beim Einparken ~2 cm weiter vorn
        # als beim Ausparken und schlaegt beim einzigen Vorwaertszug an (im
        # Lauf: 3,3 von 4,5 cm, dann Wand). Nur fuers Einparken, die
        # abgestimmten Ausparkfolgen bleiben unberuehrt.
        'einparken_vor_korr_cm': ('einparken_vor_korr_cm', -1.5, float),
        'einparken_rueck_korr_cm': ('einparken_rueck_korr_cm', 0.0, float),
        'einparken_leerzuege_weg': ('einparken_leerzuege_weg', 1.0, lambda v: bool(float(v))),
        'debug':         ('debug',         1.0,   lambda v: bool(float(v))),
    }

    def __init__(self):
        super().__init__('round1_controller')

        for name, (attr, default, conv) in self._PARAMS.items():
            self.declare_parameter(name, default)
        # structural (read once)
        self.declare_parameter('require_button', False)
        # ausparken ist DER Schalter. ausparken_nur ist eine Unteroption
        # davon: nach der Sequenz anhalten, statt das Rennen zu fahren -- zum
        # Einstellen der Schrittfolge.
        #
        # Frueher hiess sie nur_ausparken, und das war eine Falle:
        # "-p nur_ausparken:=false" liest sich wie "ausparken und dann
        # fahren", schaltet aber gar nichts ein. Der Name sagt jetzt, wozu
        # sie gehoert.
        self.declare_parameter('ausparken', False)
        self.declare_parameter('ausparken_nur', False)
        # Einparken am Ende. Greift nur, wenn vorher ausgeparkt wurde -- ohne
        # Ausparken gibt es keine aufgezeichnete Startpose, und der Regler
        # haelt wie bisher am Ziel an (Eroeffnungsrennen).
        self.declare_parameter('einparken', True)
        # Falls meine Herleitung der offenen Seite doch falsch herum ist:
        # ein Schalter statt einer Codeaenderung.
        self.declare_parameter('ausparken_richtung_invertieren', False)
        # Wer bestimmt die Fahrtrichtung, wenn ausgeparkt wird? In der Luecke
        # ist sie sicher messbar: die nahe Seite IST die Aussenwand, die ferne
        # das Spielfeld. Der scan_processor kann das aus der Luecke heraus
        # nicht besser wissen -- er sieht dort keine brauchbare Ecke, latcht
        # aber trotzdem und lag in einem Lauf nachweislich falsch herum.
        # Aus: dann gilt weiter /race_direction aus der Eckengeometrie.
        self.declare_parameter('ausparken_setzt_richtung', True)
        self.declare_parameter('test_richtung', 'CCW')     # Einpark-Test
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
        # Je Fahrtrichtung eine eigene Folge, falls der Roboter in der Luecke
        # unterschiedlich steht. Spiegeln allein reicht dann nicht: es sind
        # andere WEGE, nicht nur andere Vorzeichen. Leer = die gemeinsame
        # Folge oben gilt, so dass man nur die Richtung fuellen muss, die
        # wirklich abweicht.
        self.declare_parameter('ausparken_schritte_cw', list(SCHRITTE_CW), arr)
        self.declare_parameter('ausparken_schritte_ccw', list(SCHRITTE_CCW), arr)
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
        self.declare_parameter('ausparken_pid', [4.0, 140.0, 8.0, 90.0, 7.0, 4000.0], arr)
        self.declare_parameter('ausparken_pid_nachher', [4.0, 1023.0], arr)
        # Feinabstimmung einzelner Einparkzuege in cm (+ = laenger), ein Wert je
        # Zug der Einparkfolge; leer = nur die Richtungskorrekturen oben.
        self.declare_parameter('einparken_zug_korr_cm', [0.0], arr)

        self._load_params()
        self.require_button = bool(self.get_parameter('require_button').value)
        self.ausparken = bool(self.get_parameter('ausparken').value)
        self.ausparken_nur = bool(self.get_parameter('ausparken_nur').value)
        self.einparken = bool(self.get_parameter('einparken').value)
        self.ausparken_richtung_invertieren = bool(
            self.get_parameter('ausparken_richtung_invertieren').value)
        self.ausparken_setzt_richtung = bool(
            self.get_parameter('ausparken_setzt_richtung').value)
        self.control_rate = float(self.get_parameter('control_rate').value)
        self.odom_timeout = float(self.get_parameter('odom_timeout').value)
        self.add_on_set_parameters_callback(self._on_params)

        self.ausparken_schritte = list(
            self.get_parameter('ausparken_schritte').value)
        self.ausparken_schritte_cw = list(
            self.get_parameter('ausparken_schritte_cw').value)
        self.ausparken_schritte_ccw = list(
            self.get_parameter('ausparken_schritte_ccw').value)
        self.ausparken_pid = list(self.get_parameter('ausparken_pid').value)
        self.ausparken_pid_nachher = list(
            self.get_parameter('ausparken_pid_nachher').value)
        self.einparken_zug_korr = [float(v) for v in
                                   self.get_parameter('einparken_zug_korr_cm').value]
        # Ein Schalter soll reichen: wer nur ausparken will, meint auch ausparken.
        if self.ausparken_nur and not self.ausparken:
            self.ausparken = True
        self.test_richtung = str(self.get_parameter('test_richtung').value).strip().upper()
        if self.einparken_test:
            # nur Zielgerade und einparken
            self.ausparken = False
            self.einparken = True
            self.n_corners = 0
            self.get_logger().warn(
                "EINPARK-TEST (%s): kein Ausparken, keine Runden -- faehrt die "
                "Startgerade als Zielgerade und parkt ein." % self.test_richtung)

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
        # 'aus' = Ausparkfolge am Start, 'ein' = Einparkfolge am Ende. Beide
        # laufen durch DENSELBEN Ausfuehrer (AUSPARK_FAHREN), damit er genau so
        # einparkt, wie er ausgeparkt hat.
        self.ausp_modus = 'aus'
        self.park_ursprung = None     # Pose vor dem Ausparken (Lueckenlage)
        self.park_start = None        # Pose NACH dem Ausparken = Einpark-Start
        self.ausp_ende_pose = None    # Pose direkt nach dem letzten Ausparkzug
        # Wellenstellung aus der letzten Quittung (0,1-grad-Zaehler des ESP,
        # absolut seit Boot). Innerhalb einer Zugfolge dreht sich die Welle
        # zwischen zwei Zuegen nicht -- die Differenz ist dann EXAKT die
        # Drehung dieses Zuges, unabhaengig vom EKF.
        self.ausp_pos_prev = None
        self.anfahrt_iter = 0
        self.park_einpark = []
        self.park_mittel = []         # Posen zum Mitteln im Stand
        # Parklinie WANDBEZOGEN: Abstand base_link -> Aussenbande am Ende des
        # Ausparkens, per Lidar gemessen (/wall_distances). Nicht aus der
        # gemerkten Pose: die Karte wird beim Start aus der Lueckenlage heraus
        # ~14 cm / 3,4 grad versetzt platziert (CCW-Einparktest), und das
        # gleicht der EKF erst waehrend der Runde aus.
        self.park_q = None
        self.park_q_luecke = None     # erwarteter Abstand eingeparkt
        self.park_q_proben = []
        # Erste Ecke nach dem Ausparken: steht er schon nah davor, ersetzt der
        # Halt am Ausparkende den Scan-Stopp, und Ecke 1 wird aus der Lage
        # geplant, in der er steht (kein Querversatz auf kurzem Anlauf).
        self.erste_ecke_pruefen = False
        self.erste_ecke_idx = None
        self.ausp_scan_hier = None    # None = offen, True = am Ausparkende scannen
        self.ausp_richtung_wart_t0 = None   # seit wann auf /race_direction gewartet wird
        self.start_scan_state = None  # /start_scan_state: scanning | complete | incomplete
        self.ausp_scan_wart_t0 = None
        self.ausp_variante = 'normal' # 'normal' oder 'innen'
        self.ausp_schritte_std = []   # normale Folge (Leitungswerte) -- daraus wird eingeparkt
        self.lok_state = None         # /localization_state: None = nie empfangen (wie 'ok')
        self.lok_lost_t0 = None
        self.lok_warte_t0 = None      # Einparken wartet auf 'ok'
        self.bucht = None             # /parking_bay: gemessene Buchtwaende
        self.ausp_bahn = []           # Posen an allen Zuggrenzen des Ausparkens
        self.park_lok_unsicher = False  # parkt ohne Korrekturen weiter
        self._ziel_gemeldet = False
        self.seiten_frei = False      # nach dem Pflichthalt: Pylonen auf beliebiger Seite
        self.park_fahrt_ziel_f = None # Ueberfahren: vorwaerts bis hierhin (Abstand Frontwand)
        self.park_ueber = 0.0         # einparken_ueberfahren, begrenzt durch Pylonen
        self.rueck_phase = 'lenken'
        self.rueck_t0 = 0.0
        self.rueck_delta = 0.0
        self.rueck_versuche = 0
        self.erste_ecke_q = None
        self.park_t0 = 0.0
        self.pose = None
        self.v_ist = 0.0
        self.pose_lenk = None             # geglaettete Pose fuers Lenkgesetz
        self.pose_lenk_t = None
        self.odo_weg_vorher = 0.0         # Encoderweg seit EKF-Start, bis das Ausparken beginnt
        self.odo_weg_t = None
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
        self.obstacles_roh = None         # ungefiltert, fuer Einfrieren und Neufiltern
        self._phantom_ids = set()         # schon gemeldete Phantome (nur einmal loggen)
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
        self.cmd_hist = collections.deque(maxlen=200)   # (t, omega) der letzten Befehle

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
        self.create_subscription(String, '/localization_state',
                                 self.lok_state_cb, latched)
        self.create_subscription(String, '/start_scan_state',
                                 self.start_scan_cb, latched)
        try:
            from robot_msgs.msg import ParkingBay
            self.create_subscription(ParkingBay, '/parking_bay',
                                     self.bucht_cb, latched)
        except ImportError:
            self.get_logger().warn(
                "robot_msgs/ParkingBay nicht gebaut -- /parking_bay wird nicht "
                "gelesen (Einparken laeuft trotzdem, nur ohne Buchtpruefung).")
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

        # Nur fuers Ausparken (und den Einpark-Test, der dieselben ESP-Zuege
        # faehrt). Bewusst nicht immer angelegt -- sonst haengt der Regler ohne
        # Not am /scan und an vier weiteren Bruecken-Topics.
        if self.ausparken or self.einparken_test:
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
            # BEWUSST NICHT latched. Eine latched Nachricht ueberlebt den
            # Lauf, der sie erzeugt hat: ein frisch gestarteter
            # scan_processor bekam sie noch aus dem VORIGEN Lauf zugestellt
            # und hat seine Startpositionserkennung entsperrt, bevor
            # ueberhaupt ausgeparkt war. Stattdessen wird sie waehrend des
            # ganzen Scan-Halts wiederholt -- wer dann zuhoert, bekommt sie.
            self.pub_park_dir = self.create_publisher(
                String, '/parking_direction', 10)

        self.dt = 1.0 / self.control_rate
        self.create_timer(self.dt, self.control_loop)
        # Aktive Regelwerte ins Log -- sonst muss jede Analyse raten, ob z.B.
        # die Totzeit-Vorausberechnung lief.
        self.get_logger().info(
            'Regler: Totzeit-Vorausberechnung %.3f s (Verstaerkung %.2f), '
            'k_heading %.2f, k_stanley %.2f, k_ct %.1f, k_th %.1f, '
            'Bogen verankern bei gestoerter Einfahrt: %s.'
            % (self.steer_dead_time, self.steer_gain_pred, self.k_heading,
               self.k_stanley, self.k_ct, self.k_th,
               'an' if self.turn_anchor_puenktlich else 'aus'))
        self.get_logger().info(
            'Normale Ausparkfolgen (Referenz fuers Einparken): CW aus %s (%d Zuege), '
            'CCW aus %s (%d Zuege).'
            % (_cw_name, len(SCHRITTE_CW) // 2, _ccw_name, len(SCHRITTE_CCW) // 2))
        # Laut und deutlich: ohne require_button startet er SELBST. Beim
        # Ausparken ist der erste echte Zug +6,9 cm vorwaerts -- das sah aus
        # wie 'kriecht vor dem Tastendruck los'.
        self._log(not self.require_button,
            'Taster: %s' % ('erforderlich -- wartet auf Druck.' if self.require_button
                            else 'NICHT erforderlich (require_button:=false) -- '
                                 'startet OHNE Tastendruck!'))
        if self.ausparken and self.ausparken_nur:
            self.get_logger().info(
                ">>> Round1Controller bereit: AUSPARKEN, danach ANHALTEN "
                "(ausparken_nur). <<<")
        elif self.ausparken:
            self.get_logger().info(
                ">>> Round1Controller bereit: AUSPARKEN, danach das RENNEN. <<<")
        else:
            self.get_logger().info(
                ">>> Round1Controller bereit: KEIN Ausparken -- er faehrt los, "
                "sobald die Eingaben da sind. Zum Ausparken: "
                "-p ausparken:=true <<<")

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

    def _obstacle_offset_near_corner(self, wall_idx, corner_pt, min_frontabstand=None):
        """Pass-by offset for the obstacle on `wall_idx` CLOSEST to `corner_pt`.

        Used twice: for o_out it is the FIRST obstacle after the corner, for o_in
        the LAST one before it -- in both cases the one nearest that corner.

        min_frontabstand: nur Pylonen, die mindestens so weit vor dem ANDEREN
        Ende der Geraden (ihrer Frontwand) stehen. Fuer die letzte Kurve vor dem
        Einparken: was hinter dem Umschaltpunkt steht, darf auf beliebiger Seite
        passiert werden und soll die Kurve nicht nach innen ziehen.
        """
        if not self.obstacles or self.lane_width is None or self.walls is None:
            return None
        mine = [o for o in self.obstacles if o['wall'] == wall_idx]
        if mine and min_frontabstand is not None and self.corners is not None:
            # Wand i verbindet corners[i] -> corners[i+1]; das andere Ende ist die Frontwand
            a = self.corners[wall_idx]
            b = self.corners[(wall_idx + 1) % len(self.corners)]
            da = (a[0] - corner_pt[0]) ** 2 + (a[1] - corner_pt[1]) ** 2
            db = (b[0] - corner_pt[0]) ** 2 + (b[1] - corner_pt[1]) ** 2
            ende = b if da < db else a
            laenge = math.hypot(ende[0] - corner_pt[0], ende[1] - corner_pt[1]) or 1e-6
            ux = (ende[0] - corner_pt[0]) / laenge
            uy = (ende[1] - corner_pt[1]) / laenge
            mine = [o for o in mine
                    if laenge - ((o['x'] - corner_pt[0]) * ux + (o['y'] - corner_pt[1]) * uy)
                    > min_frontabstand]
        if not mine:
            return None
        nx, ny, d = self.walls[wall_idx]
        near = min(mine, key=lambda o: (o['x'] - corner_pt[0]) ** 2
                                       + (o['y'] - corner_pt[1]) ** 2)
        q_block = (nx * near['x'] + ny * near['y']) - d
        return self._obs_planner_for_wall(wall_idx).pass_offset(
            q_block, near['color'], self.dir_step() > 0)

    def _parklinie_festlegen(self):
        """Abstand zur Aussenbande am Ende des Ausparkens -- gemessen, sonst
        Rueckfallwert. Dazu der erwartete Abstand eingeparkt (Parklinie minus
        seitlicher Weg des Ausparkens, der ist relativ und damit verlaesslich)."""
        if not self.ausp_richtung or self.park_start is None:
            return
        ccw = self.ausp_richtung == 'CCW'
        std_laengs = self.park_std_laengs_ccw if ccw else self.park_std_laengs_cw
        std_quer = self.park_std_quer_ccw if ccw else self.park_std_quer_cw

        if self.ausp_variante != 'normal':
            # Andere Folge gefahren: er steht NICHT dort, wo die umgekehrte
            # normale Folge beginnt. Startpose und Parklinie aus der Startlage in
            # der Luecke und der gemessenen Endlage der normalen Folge. Keine
            # Referenzbahn fuer die Bogenkorrektur -- die gefahrene war eine andere.
            if self.park_ursprung is None or self.walls is None:
                self.get_logger().error(
                    "%s-Folge gefahren, aber Startlage oder Waende fehlen -- "
                    "Einparken ohne verlaessliche Startpose." % self.ausp_variante)
                return
            ux, uy, uth = self.park_ursprung
            nx, ny, dw = self.walls[self._start_wand()]
            self.park_start = (ux + std_laengs * math.cos(uth) + std_quer * nx,
                               uy + std_laengs * math.sin(uth) + std_quer * ny, uth)
            self.park_q_luecke = (nx * ux + ny * uy) - dw
            self.park_q = self.park_q_luecke + std_quer
            self.ausp_bahn = []
            self.get_logger().info(
                "%s-Folge gefahren: Einpark-Startpose aus der normalen Folge "
                "(%.1f cm laengs, %.1f cm quer ab Startlage), Parklinie %.3f m, "
                "eingeparkt erwartet %.3f m. Ohne Bogenkorrektur."
                % (self.ausp_variante, std_laengs * 100, std_quer * 100,
                   self.park_q, self.park_q_luecke))
            return

        # Normale Folge gefahren: gemessene Endlage ausgeben -- damit lassen
        # sich die park_std_*-Werte fuer den Innen-Fall nachschaerfen.
        if self.park_ursprung is not None and self.walls is not None:
            ux, uy, uth = self.park_ursprung
            nx, ny, dw = self.walls[self._start_wand()]
            laengs = ((self.park_start[0] - ux) * math.cos(uth)
                      + (self.park_start[1] - uy) * math.sin(uth))
            quer = (nx * self.park_start[0] + ny * self.park_start[1]) - (nx * ux + ny * uy)
            self.get_logger().info(
                "Normale Ausparkfolge endete %.1f cm laengs, %.1f cm quer ab Startlage "
                "(Parameter park_std_laengs_%s=%.3f, park_std_quer_%s=%.3f)."
                % (laengs * 100, quer * 100, 'ccw' if ccw else 'cw', std_laengs,
                   'ccw' if ccw else 'cw', std_quer))

        rueck = (self.einparken_linie_ccw if self.ausp_richtung == 'CCW'
                 else self.einparken_linie_cw)
        proben = sorted(v for v in self.park_q_proben if 0.15 <= v <= 0.90)
        if len(proben) >= 3:
            self.park_q = proben[len(proben) // 2]
            quelle = "gemessen (%d Proben)" % len(proben)
            if abs(self.park_q - rueck) > 0.05:
                self.get_logger().warn(
                    "Parklinie gemessen %.3f m, Handmessung %.3f m -- %.1f cm "
                    "Unterschied. Lidar-Bezugspunkt pruefen."
                    % (self.park_q, rueck, (self.park_q - rueck) * 100))
        else:
            self.park_q = rueck
            quelle = "Rueckfallwert (nur %d brauchbare Proben)" % len(proben)
        # Seitlicher Weg des Ausparkens im Startrahmen -- RELATIV, also auch
        # dann richtig, wenn die Karte versetzt liegt.
        if self.park_ursprung is not None:
            ux, uy, uth = self.park_ursprung
            dx, dy = self.park_start[0] - ux, self.park_start[1] - uy
            seitlich = abs(-dx * math.sin(uth) + dy * math.cos(uth))
            self.park_q_luecke = self.park_q - seitlich
        self.get_logger().info(
            "Parklinie %.3f m zur Aussenbande (%s)%s."
            % (self.park_q, quelle,
               ", eingeparkt erwartet %.3f m" % self.park_q_luecke
               if self.park_q_luecke is not None else ""))

    def _park_versatz_anwenden(self):
        """Einpark-Startpose um den Handversatz verschieben
        (einpark_versatz_laengs/quer_ccw bzw. _cw, je nach Fahrtrichtung)."""
        if self.ausp_richtung == 'CW':
            dl, dq = self.einpark_versatz_laengs_cw, self.einpark_versatz_quer_cw
        else:
            dl, dq = self.einpark_versatz_laengs_ccw, self.einpark_versatz_quer_ccw
        if self.park_start is None or (dl == 0.0 and dq == 0.0):
            return
        x, y, th = self.park_start
        if self.walls is not None:
            nx, ny, _dw = self.walls[self._start_wand()]   # zeigt ins Feld
        elif self.ausp_richtung == 'CCW':                  # Feld links
            nx, ny = -math.sin(th), math.cos(th)
        else:                                              # Feld rechts
            nx, ny = math.sin(th), -math.cos(th)
        self.park_start = (x + dl * math.cos(th) + dq * nx,
                           y + dl * math.sin(th) + dq * ny, th)
        if self.park_q is not None:
            self.park_q += dq
        if self.park_q_luecke is not None:
            self.park_q_luecke += dq
        self.get_logger().info(
            "Einpark-Startpose von Hand verschoben (%s): %+.1f cm laengs, %+.1f cm quer "
            "-> (%.3f, %.3f)%s."
            % (self.ausp_richtung, dl * 100, dq * 100, self.park_start[0], self.park_start[1],
               ', Parklinie %.3f m' % self.park_q if self.park_q is not None else ''))

    def _park_aktiv(self):
        return self.einparken and self.park_start is not None

    def _park_offset_for_wall(self, wall_idx):
        """Parklinie: Abstand der aufgezeichneten Einpark-Startpose zur
        Aussenbande dieser Geraden. Gemessen vom Roboter selbst am Ende des
        Ausparkens -- keine Konstanten, die fuer CW und CCW verschieden waeren
        (CW 37,0 cm, CCW 34,5 cm, je nach Schlussbogen).

        None, wenn nicht eingeparkt wird oder der Wert unplausibel ist (dann
        faehrt die Zielgerade wie bisher die Mitte).
        """
        if not self._park_aktiv() or self.park_q is None:
            return None
        q = self.park_q
        breite = (self.lane_width[wall_idx]
                  if self.lane_width is not None and wall_idx < len(self.lane_width)
                  else 1.0)
        if not (0.15 <= q <= breite - 0.10):
            self.get_logger().warn(
                "Parklinie %.2f m auf Gerade w%d unplausibel -- ist das die "
                "Startgerade? Zielgerade faehrt ohne Parklinie." % (q, wall_idx),
                throttle_duration_sec=5.0)
            return None
        return q

    def _zielgerade_folgt(self):
        """Wird gerade die LETZTE Ecke geplant (ihr Ausgang ist die Zielgerade)?"""
        return self.corner_count + 1 == self.n_corners

    def _auf_zielgerade(self):
        return self.corner_count >= self.n_corners

    def corner_o_in(self, idx):
        w = self._entry_wall_idx(idx)
        # Auf der Zielgeraden ist die Ecke dahinter nur noch Rechengroesse --
        # sie wird nie gefahren. Ihre Eintrittslinie IST die Zielgerade, und die
        # soll auf der Parklinie liegen. Hindernisse davor erledigt der
        # Hindernispfad, der danach auf die Parklinie zurueckschwenkt.
        if self._auf_zielgerade():
            q = self._park_offset_for_wall(w)
            if q is not None:
                return q
        if self.corners is not None:
            q = self._obstacle_offset_near_corner(w, self.corners[idx])
            if q is not None:
                return q                      # last obstacle before the corner
        # Erste Ecke direkt nach dem Ausparken: auf der Linie bleiben, auf der
        # er steht. Ein Hindernis davor hat oben schon Vorrang bekommen.
        if (self.erste_ecke_q is not None and self.corner_count == 0
                and idx == self.erste_ecke_idx):
            return self.erste_ecke_q
        auto = self._lane_default_offset(w) if self.use_auto_offset else None
        if auto is not None:
            return auto
        return self.o_in_list[idx] if idx < len(self.o_in_list) else self.o_in

    def corner_o_out(self, idx):
        w = self._exit_wall_idx(idx)
        if self.corners is not None:
            # Letzte Kurve vor dem Einparken ohne Pflichthalt: nur Pylonen vor dem
            # Umschaltpunkt zaehlen. Eine Pylone am Ende der Startgeraden (hinter
            # dem Punkt, ab dem die Seite frei ist) zog die Kurve sonst auf die
            # Innenlinie 0,81 -- und danach musste er quer zurueck zur Parklinie.
            grenze = (self.seiten_frei_ab
                      if (self._zielgerade_folgt() and self._park_aktiv()
                          and self.einparken_halt_s <= 0.0) else None)
            q = self._obstacle_offset_near_corner(w, self.corners[idx], grenze)
            if q is not None:
                return q                      # first obstacle after the corner
        # Letzte Ecke ohne Hindernis dahinter: direkt auf die Parklinie
        # aus der Kurve kommen, dann muss auf der Zielgeraden nichts mehr
        # rangiert werden.
        if self._zielgerade_folgt():
            q = self._park_offset_for_wall(w)
            if q is not None:
                return q
        auto = self._lane_default_offset(w) if self.use_auto_offset else None
        if auto is not None:
            return auto
        return self.o_out_list[idx] if idx < len(self.o_out_list) else self.o_out

    def _obs_planner_for_wall(self, wall_idx):
        from ekf.obstacle_path import ObstaclePathPlanner
        w = self.lane_width[wall_idx]
        return ObstaclePathPlanner(lane_width=w, wall_margin=self.obs_wall_margin,
                                   outer_margin=self._aussen_rand(wall_idx))

    def _aussen_rand(self, wall_idx):
        """Hindernis an der Aussenbande dieser Geraden (Magenta-Waende der
        Parkluecke auf der Startgeraden), fuer die Vorbeifahrt aussen.
        Nicht auf der Zielgeraden beim Einparken -- dort faehrt er bewusst
        auf der Parklinie, die eigene Logik haelt Abstand."""
        if not (self.parking_lot_present or self.park_ursprung is not None):
            return 0.0
        if self._park_aktiv() and (self._auf_zielgerade() or self._zielgerade_folgt()):
            return 0.0
        sw = self._start_wand()
        if sw is None and self.corner_count == 0 and self.corner_idx is not None:
            sw = self._entry_wall_idx(self.corner_idx)
        return LUECKE_TIEFE if wall_idx == sw else 0.0

    def corner_R(self, idx):
        return self.R_list[idx] if idx < len(self.R_list) else self.R

    # ------------------------------------------------------------- callbacks
    def odom_cb(self, msg):
        p = msg.pose.pose
        self.pose = (p.position.x, p.position.y, yaw_from_quaternion(p.orientation))
        self.v_ist = float(msg.twist.twist.linear.x)
        self.last_odom_time = self.get_clock().now()
        t_odo = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        if self.park_ursprung is None:
            if self.odo_weg_t is not None and 0.0 < t_odo - self.odo_weg_t < 0.5:
                self.odo_weg_vorher += abs(self.v_ist) * (t_odo - self.odo_weg_t)
            self.odo_weg_t = t_odo
        self._lenk_pose_nachfuehren(msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9,
                                    float(msg.twist.twist.angular.z))

    def _lenk_pose_nachfuehren(self, t, w):
        """Lenk-Pose: Eigenbewegung sofort, Korrekturen ueber lenk_pose_tau
        (siehe Parameter)."""
        roh = self.pose
        alt_t, self.pose_lenk_t = self.pose_lenk_t, t
        if (self.lenk_pose_tau <= 0.0 or self.pose_lenk is None or alt_t is None
                or not (0.0 < t - alt_t < 0.2)):
            self.pose_lenk = roh
            return
        dt = t - alt_t
        xl, yl, tl = self.pose_lenk
        thm = tl + 0.5 * w * dt
        xp = xl + self.v_ist * dt * math.cos(thm)
        yp = yl + self.v_ist * dt * math.sin(thm)
        tp = tl + w * dt
        ex, ey, et = roh[0] - xp, roh[1] - yp, wrap(roh[2] - tp)
        if math.hypot(ex, ey) > self.lenk_pose_sprung or abs(et) > self.lenk_pose_sprung_grad:
            self.pose_lenk = roh            # Kartenwechsel o. ae.: sofort
            return
        k = min(1.0, dt / self.lenk_pose_tau)
        self.pose_lenk = (xp + k * ex, yp + k * ey, wrap(tp + k * et))

    def front_wall_cb(self, msg):
        if self.front_wall_x is None:
            self.get_logger().info(f"/front_wall_x empfangen: {msg.data:.3f} m.")
        self.front_wall_x = float(msg.data)

    def start_scan_cb(self, msg):
        neu = msg.data.strip().lower()
        if neu == self.start_scan_state:
            return
        self.start_scan_state = neu
        if neu == 'incomplete':
            self.get_logger().warn(
                "Startgerade nicht vollstaendig abgetastet (incomplete) -- "
                "ein Sitz ist offen, eine Pylone dort kennt er nicht.")
        else:
            self.get_logger().info("Startgerade: %s." % neu)

    def _log(self, warnung, text):
        """Je nach Bedingung warn oder info.

        NICHT als (warn if c else info)(...) schreiben: rclpy merkt sich den
        Level pro Aufrufstelle und wirft 'Logger severity cannot be changed
        between calls', sobald dieselbe Zeile einmal info und einmal warn
        loggt. Hier stehen beide in eigenen Zeilen."""
        if warnung:
            self.get_logger().warn(text)
        else:
            self.get_logger().info(text)

    def lok_state_cb(self, msg):
        """/localization_state: ok | recovering | lost (latched, bei Wechsel)."""
        neu = msg.data.strip().lower()
        if neu == self.lok_state:
            return
        alt = self.lok_state
        self.lok_state = neu
        if neu == 'lost':
            self.lok_lost_t0 = self.now_s()
        else:
            self.lok_lost_t0 = None
        self._log(neu != 'ok',
            "Lokalisierung: %s -> %s%s." % (
                alt or '-', neu,
                '' if neu == 'ok' else
                ' (hoechstens %.2f m/s, kein Einparken)' % self.v_lok_unsicher))

    def _lok_ok(self):
        """Nie empfangen zaehlt als ok -- sonst liefe ohne das Topic nichts."""
        return self.lok_state in (None, 'ok')

    def bucht_cb(self, msg):
        """/parking_bay: gemessene Buchtwaende (einmal, latched).

        Nur zur PRUEFUNG. Eingeparkt wird mit der umgekehrten Ausparkfolge,
        und die landet dort, wo er gestartet ist -- das war nachweislich in der
        Bucht. Die Messung zeigt, ob diese Lage zwischen den Waenden liegt.
        Bei schraegem Blick koennen Wandenden fehlen (Luecke eher zu gross),
        deshalb bricht hier nichts ab."""
        if not msg.detected:
            self.get_logger().warn("/parking_bay: keine Bucht gefunden.")
            self.bucht = None
            return
        a = ((msg.wall_a_start.x, msg.wall_a_start.y), (msg.wall_a_end.x, msg.wall_a_end.y))
        b = ((msg.wall_b_start.x, msg.wall_b_start.y), (msg.wall_b_end.x, msg.wall_b_end.y))
        self.bucht = dict(a=a, b=b)
        la = math.hypot(a[1][0] - a[0][0], a[1][1] - a[0][1])
        lb = math.hypot(b[1][0] - b[0][0], b[1][1] - b[0][1])
        text = "Buchtwaende %.3f / %.3f m (Soll %.2f)" % (la, lb, LUECKE_TIEFE)
        if self.park_ursprung is not None:
            r = self._bucht_abstaende(*self.park_ursprung)
            if r is not None:
                heck, front, luecke = r
                text += (", Luecke %.3f m (Soll %.4f), Startlage: Heck %.1f cm, "
                         "Front %.1f cm Luft" % (luecke, LUECKE_LAENGE,
                                                  heck * 100, front * 100))
        self.get_logger().info("/parking_bay: " + text + ".")

    def _bucht_abstaende(self, x, y, theta):
        """Luft von Heck und Front zu den Buchtwaenden fuer eine Pose in der
        Bucht, gemessen entlang des Kurses. Rueckgabe (heck, front, luecke) in m,
        oder None, wenn keine Bucht bekannt ist oder sie nicht um die Pose liegt."""
        if self.bucht is None:
            return None
        ax, ay = math.cos(theta), math.sin(theta)
        def laengs(seg):
            mx = 0.5 * (seg[0][0] + seg[1][0])
            my = 0.5 * (seg[0][1] + seg[1][1])
            return (mx - x) * ax + (my - y) * ay
        sa, sb = laengs(self.bucht['a']), laengs(self.bucht['b'])
        hinten, vorn = min(sa, sb), max(sa, sb)
        if not (hinten < 0.0 < vorn):
            return None
        return (FZ_HECK - hinten, vorn - FZ_NASE, vorn - hinten)

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
            if self.obstacles_roh is not None and len(msg.obstacles) != len(self.obstacles_roh):
                self.get_logger().warn(
                    f"/obstacles nach Runde {self.obs_freeze_lap} ignoriert "
                    f"({len(msg.obstacles)} statt {len(self.obstacles_roh)} gemeldet) "
                    f"-- Hindernisse sind eingefroren.")
            return

        obs = []
        for o in msg.obstacles:
            obs.append(dict(id=int(o.id), x=float(o.position.x), y=float(o.position.y),
                            color=int(o.color), wall=int(o.wall_idx)))
        self.obstacles_roh = list(obs)
        obs = self._phantome_filtern(obs)
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

    def _start_wand(self):
        """Index der Startgeraden: die Wand, an der er in der Luecke stand."""
        if self.walls is None or self.park_ursprung is None:
            return None
        ux, uy, _ = self.park_ursprung
        return min(range(len(self.walls)),
                   key=lambda i: abs(self.walls[i][0] * ux + self.walls[i][1] * uy
                                     - self.walls[i][2]))

    def _phantome_filtern(self, obs):
        """Regelwerk: mit Parkluecke stehen auf der Startgeraden NUR Zeichen in
        der inneren Spalte. Eine Meldung in der aeusseren Spalte dort kann nur
        falsch sein -- im CCW-Einparktest war es vermutlich eine Magenta-
        Parkwand, als Rot gelesen (id20, kein echtes Hindernis auf der Bahn).
        Sitzkodierung: id = Gruppe*6 + k, gerades k = aeussere Spalte."""
        if self.ausp_richtung is None and not self.parking_lot_present:
            return obs
        w = self._start_wand()
        if w is None:
            return obs
        nx, ny, dw = self.walls[w]
        breite = (self.lane_width[w] if self.lane_width is not None
                  and w < len(self.lane_width) else 1.0)
        behalten, weg = [], []
        for o in obs:
            # ZWEI Bedingungen, beide muessen stimmen: Sitzkodierung sagt aussen
            # UND die Lage liegt wirklich in der aeusseren Spurhaelfte. Ein
            # echtes inneres Zeichen zu verwerfen hiesse hineinfahren -- falls
            # die Kodierung einmal nicht stimmt, haelt die Geometrie es fest.
            q = (nx * o['x'] + ny * o['y']) - dw
            aussen = (o['id'] % 6) % 2 == 0 and q < 0.5 * breite
            (weg if (o['wall'] == w and aussen) else behalten).append(o)
        neu = {o['id'] for o in weg} - self._phantom_ids
        if neu:
            self._phantom_ids |= neu
            self.get_logger().warn(
                "Phantom verworfen: %s auf der Startgeraden w%d in der AEUSSEREN "
                "Spalte -- mit Parkluecke stehen dort nur innere Zeichen. "
                "(Magenta-Parkwand als Rot gelesen?)"
                % (', '.join('id%d(%s)' % (o['id'], 'gruen' if o['color'] == 2 else 'rot')
                             for o in weg if o['id'] in neu), w))
        return behalten

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

        # Zielgerade mit Einparken: nur Hindernisse, an denen er VOR dem
        # Haltepunkt vorbeikommt. Was dahinter steht, faehrt er nie an --
        # darauf auszuweichen hiesse, neben der Parklinie anzuhalten.
        # Nach dem Pflichthalt: die Seite ist frei. Alle Pylonen auf der
        # AUSSENseite, dort liegt die Parklinie. Pylonen, die gerade neben dem
        # Wagen stehen, nicht "ueberholen" -- Spur halten, bis sie hinter dem
        # Heck liegen, sonst schwenkt er seitlich in sie hinein.
        seiten_frei = self.seiten_frei and self._auf_zielgerade()
        s_halten = 0.0
        alle_voraus = []
        if seiten_frei:
            self.park_fahrt_ziel_f = None
            ccw_ = self.dir_step() > 0
            c_aussen = OBST_ROT if ccw_ else OBST_GRUEN     # Farbe -> Aussenseite
            c_innen = OBST_GRUEN if ccw_ else OBST_ROT
            neben = [t for t in obs_lane if t[0] < self.obs_clear_before]
            if neben:
                s_halten = max(0.0, max(t[0] for t in neben)
                               + self.obs_clear_after - FZ_HECK)
            # Seite je Pylone: aussen (Parklinie), wenn der Wechsel dorthin
            # fahrbar ist -- sonst auf der Seite bleiben, auf der er gerade ist.
            # "Alle aussen" um jeden Preis plante nach einem Halt auf der
            # Innenseite einen unfahrbar steilen Wechsel quer durch die Pylonen.
            pl_ = self._obs_planner()
            q_cur = to_lane(x, y)[1]
            s_cur = s_halten
            neu = []
            for (so, qo, _c) in sorted(obs_lane, key=lambda t: t[0]):
                if so < self.obs_clear_before:
                    continue
                q_a = pl_.pass_offset(qo, c_aussen, ccw_)
                platz = so - self.obs_clear_before - s_cur
                if abs(q_a - q_cur) < 0.02 or (
                        platz > 0.0 and abs(q_a - q_cur) / platz
                        <= self.einparken_rueck_steigung_max):
                    farbe = c_aussen
                else:
                    farbe = c_aussen if q_cur < qo else c_innen
                    self.get_logger().info(
                        "Seite frei: Pylone bei %.2f m auf der %s passiert -- "
                        "Wechsel nach aussen waere zu steil."
                        % (so, 'Aussenseite' if farbe == c_aussen else 'Innenseite'))
                q_cur = pl_.pass_offset(qo, farbe, ccw_)
                s_cur = so + self.obs_clear_after
                neu.append((so, qo, farbe))
            obs_lane = neu
            alle_voraus = list(neu)           # auch jenseits der Startpose
            # Alle aussen und die Parklinie selbst hat an jeder genug Luft?
            # Dann kein Ausweichpfad -- einfach die Parklinie fahren, statt
            # mittig in die Luecke (q~0,29) und kurz vor der Startpose zurueck.
            q_pl = self._park_offset_for_wall(w_entry)
            if (q_pl is not None and all(f == c_aussen for (_so, _qo, f) in neu)
                    and all(q_pl + 0.06 + 0.05 <= qo - BLOCK_HALB for (_so, qo, _f) in neu)):
                obs_lane = []

        q_park = None
        s_stop = None
        if self._auf_zielgerade():
            q_park = self._park_offset_for_wall(w_entry)
            if q_park is not None:
                fc = self.corners[self.corner_idx]
                vorn = (fc[0] - x) * tx + (fc[1] - y) * ty
                # vor dem Halt: bis zum Haltepunkt; danach: bis zur Startpose
                s_stop = vorn - ((self._park_front_abstand() - self.park_ueber)
                                 if seiten_frei else self._ziel_abstand())
                obs_lane = [t for t in obs_lane if t[0] < s_stop + 0.10]
        if not obs_lane and not (seiten_frei and q_park is not None):
            return

        _, q_now = to_lane(x, y)
        # how far the straight still runs: up to the turn-in point T_A
        tA = self.arc['T_A']
        s_end = (tA[0] - x) * tx + (tA[1] - y) * ty
        if obs_lane:                           # nach dem Halt ggf. leer
            s_end = max(s_end, max(t[0] for t in obs_lane) + 0.3)

        planner = self._obs_planner()
        pts = planner.plan(obs_lane, s_end, self.dir_step() > 0,
                           q_start=q_now, s_start=s_halten,
                           q_default=(q_park if seiten_frei else
                                      (self._lane_default_offset(w_entry)
                                       or self.corner_o_in(idx))))
        if s_halten > 0.0:
            pts = [(0.0, q_now)] + pts        # bis s_halten die Spur halten
        # Zielgerade: nach dem letzten Hindernis zurueck auf die Parklinie,
        # und zwar fertig VOR dem Haltepunkt. Reicht der Platz nicht, bleibt
        # er auf dem Vorbeifahr-Offset -- die Anfahrtspruefung nach dem Halt
        # faengt das ab und parkt dann nicht, statt schief einzuparken.
        if (q_park is not None and s_stop is not None
                and (seiten_frei or self.einparken_halt_s > 0.0)):
            letzte = (max(t[0] for t in obs_lane) + self.obs_clear_after
                      if obs_lane else s_halten)
            behalten = [(sv, qv) for (sv, qv) in pts if sv <= letzte + 1e-6]
            if not behalten:
                behalten = [pts[0]]
            q_letzt = behalten[-1][1]
            # Laenge, die der Rueckschwenk mindestens braucht, um nicht
            # steiler als einparken_rueck_steigung_max zu werden. Gemessen
            # schafft er ~1,0 (40 cm quer auf 40 cm laengs bei 0,45 m/s).
            noetig = abs(q_letzt - q_park) / max(self.einparken_rueck_steigung_max, 0.1)
            s_zurueck = min(letzte + max(self.obs_transition_pref, noetig),
                            s_stop - 0.05)
            s_bis = max(s_end, s_stop + 0.20)
            # Ueberfahren: passt der Rueckschwenk nicht mehr vor die Startpose,
            # nach dem Pflichthalt JENSEITS davon auf die Parklinie wechseln --
            # fertig vor der naechsten Pylone -- und danach geregelt rueckwaerts
            # zur Startpose (PARK_RUECK). Nach dem Halt zaehlt die Zeit nicht.
            s_ueber = None
            if (seiten_frei and abs(q_letzt - q_park) >= 0.01
                    and s_zurueck - letzte < noetig):
                weiter = [so for (so, _qo, _f) in alle_voraus if so > letzte]
                grenze = (min(weiter) - self.obs_clear_before) if weiter \
                    else letzte + 2.0
                s_zu2 = min(letzte + max(self.obs_transition_pref, noetig), grenze)
                if s_zu2 - letzte >= noetig:
                    s_ueber = s_zu2

            if abs(q_letzt - q_park) < 0.01:
                behalten.append((s_bis, q_park))
            elif s_zurueck - letzte >= noetig:
                behalten += [(letzte, q_letzt), (s_zurueck, q_park), (s_bis, q_park)]
                self.get_logger().info(
                    "Zielgerade: nach dem letzten Hindernis zurueck auf die "
                    "Parklinie (q %.2f -> %.2f ueber %.2f m)."
                    % (q_letzt, q_park, s_zurueck - letzte))
            elif s_ueber is not None:
                behalten += [(letzte, q_letzt), (s_ueber, q_park),
                             (s_ueber + 0.30, q_park)]
                self.park_fahrt_ziel_f = vorn - (s_ueber + 0.10)
                self.get_logger().info(
                    "Seite frei: Startpose zu nah fuer den Wechsel auf die "
                    "Parklinie -- wechselt bis %.2f m voraus, haelt %.2f m vor "
                    "der Frontwand und setzt dann %.0f cm zurueck."
                    % (s_ueber, self.park_fahrt_ziel_f,
                       (self._park_front_abstand() - self.park_fahrt_ziel_f) * 100))
            else:
                behalten.append((s_bis, q_letzt))
                self.get_logger().warn(
                    "Zielgerade: zu wenig Platz zwischen letztem Hindernis und "
                    "Haltepunkt (%.2f m, noetig %.2f m) -- er haelt neben der "
                    "Parklinie (q %.2f statt %.2f), Einparken wird dann ausfallen."
                    % (max(s_zurueck - letzte, 0.0), noetig, q_letzt, q_park))
            # stabil nach s sortieren, gleiche s nur einmal (der erste bleibt)
            pts = []
            for sv, qv in sorted(behalten, key=lambda pq: pq[0]):
                if pts and abs(sv - pts[-1][0]) < 1e-9:
                    continue
                pts.append((sv, qv))
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

    def _rueckfuehr_pfad(self):
        """Nach der Kurve ohne Hindernispfad: sanfter Pfad von der Ist-Lage
        auf die Spurlinie der neuen Geraden (Eintrittslinie LA).

        Die Kurve endet einige cm neben dieser Linie. Stanley zielte bisher
        sofort auf sie -- der Lenkbefehl sprang am Kurvenende in einem Takt
        von der Kurvenlenkung in die Gegenrichtung (parken_test_28: -4,5 ->
        +9,8, -8,6 -> +6,6, -7,2 -> +12 grad). Mit dem Pfad beginnt der
        Querfehler bei null und wird ueber mindestens 0,5 m abgebaut
        (hoechstens 0,2 quer je laengs), mit Kruemmungsvorsteuerung."""
        if self.arc is None or self.pose is None or self.walls is None:
            return
        tx, ty = self.arc['travel']
        nx, ny, d = self.walls[self._entry_wall_idx(self.corner_idx)]
        x, y, _ = self.pose
        q_ist = (nx * x + ny * y) - d
        q_ziel = self.arc['o_in']
        dq = q_ziel - q_ist
        if abs(dq) < 0.02:
            return
        tA = self.arc['T_A']
        s_ende = (tA[0] - x) * tx + (tA[1] - y) * ty
        lang = max(0.50, abs(dq) / 0.20)
        if s_ende < lang + 0.05:
            return                      # zu kurz: dann lieber direkt wie bisher
        fx = x - q_ist * nx
        fy = y - q_ist * ny             # Fusspunkt auf der Aussenbande
        from ekf.obstacle_path import ObstaclePathPlanner
        dicht = ObstaclePathPlanner.densify([(0.0, q_ist), (lang, q_ziel), (s_ende + 0.3, q_ziel)], 0.05)
        self.obs_path = [(fx + tx * s + nx * q, fy + ty * s + ny * q) for (s, q) in dicht]
        self.obs_path_end_q = q_ziel
        self.obs_max_slope = abs(dq) / lang
        self._path_idx = 0
        self.get_logger().info(
            "Kurvenausgang %.1f cm neben der Spurlinie -- Rueckfuehrpfad ueber %.2f m."
            % (abs(dq) * 100.0, lang))

    def _obs_planner(self):
        from ekf.obstacle_path import ObstaclePathPlanner
        w_idx = self._entry_wall_idx(self.corner_idx)
        w = self.lane_width[w_idx]
        # Startgerade mit Parkluecke: die Magenta-Waende ragen LUECKE_TIEFE von
        # der AUSSENbande ins Feld. Frueher wurde die Spur innen um 0,20
        # verschmaelert -- auf der falschen Seite; an Rot (CCW aussen) fuhr er
        # mittig zwischen Bande und Pylone, 1-4 cm an der Luecke vorbei.
        return ObstaclePathPlanner(
            outer_margin=self._aussen_rand(w_idx),
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
        # Hindernisse, die VOR der Geometrie kamen, jetzt nachfiltern: vorher
        # war die Startwand unbekannt.
        if self.obstacles_roh is not None:
            self.obstacles = self._phantome_filtern(list(self.obstacles_roh))

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
        self.get_logger().error("%s abgebrochen: %s"
                                % ('Einparken' if self.ausp_modus == 'ein'
                                   else 'Ausparken', grund))

    def _auspark_variante(self, richtung):
        """Welche Ausparkfolge? Entscheidend ist die NAECHSTE Pylone VOR dem
        geparkten Roboter (ausparken_entscheid_ab..bis voraus): Farbe verlangt
        innen (CW rot, CCW gruen) -> 'innen', andere Farbe -> 'aussen', keine
        -> 'mitte'. Relativ zum Roboter statt zu einer festen Reihe, weil die
        Luecke je nach Aufbau an anderer Stelle der Geraden liegt.
        Rueckgabe (variante, Pylone|None)."""
        if self.obstacles is None or self.walls is None or self.pose is None:
            return 'mitte', None
        x, y, th = self.pose
        vorn = self._abstand_erste_ecke(x, y, th)
        w = min(range(len(self.walls)),
                key=lambda i: abs(self.walls[i][0] * x + self.walls[i][1] * y
                                  - self.walls[i][2]))
        c, sn = math.cos(th), math.sin(th)
        nx, ny, dw = self.walls[w]
        breite = (self.lane_width[w] if self.lane_width is not None
                  and w < len(self.lane_width) else 1.0)
        naechste = None
        for o in self.obstacles:
            if o['wall'] != w:
                continue
            # Zweite, unabhaengige Bedingung: die Pylone muss auch SEITLICH in der
            # Startspur liegen. Steht die Frontwand nah (Start bei 1,25 m), liegt
            # die innere Spalte der NAECHSTEN Geraden nur ~0,65 m vor ihm -- in
            # Reichweite. Sie steht aber ~1,0 m von der Aussenbande; die Sitze der
            # Startgeraden bei 0,4 / 0,6 m. Faellt der Wandindex nahe der Ecke
            # einmal falsch aus, haelt die Geometrie sie trotzdem heraus.
            q_o = (nx * o['x'] + ny * o['y']) - dw
            if not (0.05 < q_o < breite - 0.10):
                self.get_logger().warn(
                    "Startgerade: Pylone #%d liegt %.2f m von der Aussenbande -- "
                    "nicht in der Startspur, zaehlt nicht fuer das Ausparken."
                    % (o['id'], q_o))
                continue
            s_o = (o['x'] - x) * c + (o['y'] - y) * sn          # + = voraus
            self.get_logger().info(
                "Startgerade: Pylone #%d %s, %.2f m %s%s." % (
                    o['id'], 'gruen' if o['color'] == OBST_GRUEN else
                    'rot' if o['color'] == OBST_ROT else '?', abs(s_o),
                    'voraus' if s_o >= 0 else 'zurueck',
                    '' if vorn is None else ' (%.2f m vor der Frontwand)' % (vorn - s_o)))
            if (self.ausparken_entscheid_ab < s_o < self.ausparken_entscheid_bis
                    and (naechste is None or s_o < naechste[1])):
                naechste = (o, s_o)
        if self.start_scan_state != 'complete':
            self.get_logger().warn(
                "Startgerade nicht vollstaendig abgetastet -- vor ihm kann eine "
                "Pylone unentdeckt sein.")
        if naechste is None:
            return 'mitte', None
        o = naechste[0]
        innen = OBST_ROT if richtung == 'CW' else OBST_GRUEN
        aussen = OBST_GRUEN if richtung == 'CW' else OBST_ROT
        if o['color'] == innen:
            return 'innen', o
        if o['color'] == aussen:
            return 'aussen', o
        return 'mitte', o

    def _ausparken_planen(self):
        """Tabelle auf die erkannte Seite drehen und den Trockenlauf mitloggen."""
        richtung = self.ausp_stimmen[-1]
        if self.ausparken_richtung_invertieren:
            richtung = 'CW' if richtung == 'CCW' else 'CCW'
            self.get_logger().warn(
                "Ausparken: Richtung per Parameter invertiert.")
        offen_links = (richtung == 'CCW')
        try:
            roh, herkunft = schritte_fuer(
                richtung, gemeinsam=self.ausparken_schritte,
                cw=self.ausparken_schritte_cw,
                ccw=self.ausparken_schritte_ccw)
            tabelle = schritte_aus_flach(roh)
        except ValueError as fehler:
            self._ausparken_abbruch("Schrittliste unbrauchbar: %s" % fehler)
            return
        if not tabelle:
            self._ausparken_abbruch("Schrittliste ist leer")
            return

        self.ausp_schritte = spiegeln(tabelle, offen_links)
        self.ausp_richtung = richtung
        # Die NORMALE Folge immer merken: eingeparkt wird mit ihrer Umkehrung,
        # auch wenn gleich die Innen-Folge gefahren wird.
        self.ausp_schritte_std = list(self.ausp_schritte)
        self.ausp_variante = 'normal'
        variante, pyl = self._auspark_variante(richtung)
        liste = AUSPARK_VARIANTEN[(richtung, variante)]
        grund = ("keine Pylone vor ihm" if pyl is None else
                 "Pylone #%d %s vor ihm" % (pyl['id'], 'rot' if pyl['color'] == OBST_ROT
                                            else 'gruen' if pyl['color'] == OBST_GRUEN else '?'))
        tab_v = None
        if liste:
            try:
                tab_v = schritte_aus_flach(list(liste))
            except ValueError as fehler:
                self.get_logger().error("SCHRITTE_%s_%s unbrauchbar: %s"
                                        % (richtung, variante.upper(), fehler))
        if tab_v and tab_v == tabelle:
            # identisch mit der normalen Folge: als normal behandeln, dann
            # behaelt das Einparken Bogenkorrektur und gemessene Startpose
            self.get_logger().info(
                "Ausparken %s: %s -> %s-Folge (identisch mit der normalen)."
                % (richtung, grund, variante))
        elif tab_v:
            tabelle = tab_v
            self.ausp_schritte = spiegeln(tab_v, offen_links)
            self.ausp_variante = variante
            herkunft = '%s-%s-Folge' % (richtung, variante)
            self.get_logger().info(
                "Ausparken %s: %s -> %s-Folge." % (richtung, grund, variante))
        elif variante == 'innen':
            self.get_logger().error(
                "Ausparken %s: %s verlangt die Innenseite, aber SCHRITTE_%s_INNEN ist "
                "leer. Faehrt die normale Folge -- die Pylone wird dann auf der "
                "FALSCHEN Seite passiert." % (richtung, grund, richtung))
        else:
            self.get_logger().info(
                "Ausparken %s: %s -> %s-Folge, SCHRITTE_%s_%s ist leer: faehrt die "
                "normale." % (richtung, grund, variante, richtung, variante.upper()))

        # Trockenlauf zum Mitschreiben, immer in der Lage "offen links"
        # gerechnet: die Luecke ist spiegelsymmetrisch, die Lenkung nur fast
        # (R 0,306 m links gegen 0,312 m rechts). Fuer die Warnung reicht das.
        probe = simuliere(spiegeln(tabelle, True))
        self.get_logger().info(
            "Ausparken: %s -- offene Seite %s (%s). %s: %d Zuege, %.0f cm Weg."
            % (richtung, 'links' if offen_links else 'rechts',
               self.ausp_letzter_grund, herkunft, len(tabelle),
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
        self.ausp_modus = 'aus'
        # Hier soll er am Ende wieder stehen. Hat er sich seit dem EKF-Start
        # nicht bewegt (Encoder), ist das der Kartenursprung: die Karte ist so
        # verankert, dass er beim Commit genau dort steht. Die EKF-POSITION
        # dagegen springt in CW im Stand um bis zu +-5 cm (Wandabgleich mit
        # kaum sichtbarer Frontwand) -- genommen im Moment des ersten Zugs
        # wanderte die berechnete Einpark-Startpose von Lauf zu Lauf um 6 cm
        # (parken_test_27-29). Den Kurs misst der Abgleich an Aussen- und
        # Innenbande dagegen gut: der bleibt aus dem EKF.
        if self.odo_weg_vorher < 0.01 and self.pose is not None:
            ex, ey, eth = self.pose
            self.park_ursprung = (0.0, 0.0, eth)
            self.get_logger().info(
                "Lueckenlage = Kartenursprung (Encoderweg seit Start %.1f cm); EKF "
                "stand bei %+.1f / %+.1f cm, Kurs %+.1f grad uebernommen."
                % (self.odo_weg_vorher * 100, ex * 100, ey * 100, math.degrees(eth)))
        else:
            self.park_ursprung = self.pose
            self.get_logger().warn(
                "Lueckenlage aus dem EKF: er ist seit dem Start schon %.1f cm "
                "gefahren -- steht er nicht mehr, wo die Karte verankert wurde?"
                % (self.odo_weg_vorher * 100))
        self.ausp_bahn = [self.park_ursprung]  # dazu die Pose nach jedem Zug
        self.ausp_pos_prev = None
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
            # Eigene Messung steht. Jetzt auf die Perzeption warten und
            # abgleichen: mit start_from_bay misst sie Richtung und Pose im
            # Stand und committet, BEVOR er losfahren darf.
            eigene = self.ausp_stimmen[-1]
            if self.ausparken_richtung_invertieren:
                eigene = 'CW' if eigene == 'CCW' else 'CCW'
            if self.race_direction not in ('CW', 'CCW'):
                if self.ausp_richtung_wart_t0 is None:
                    self.ausp_richtung_wart_t0 = jetzt
                gewartet = jetzt - self.ausp_richtung_wart_t0
                if gewartet < self.ausparken_warte_auf_richtung_s:
                    self.get_logger().info(
                        "Ausparken: eigene Messung %s, warte auf /race_direction "
                        "(%.1f s)." % (eigene, gewartet), throttle_duration_sec=0.5)
                    return
                self.get_logger().warn(
                    "Ausparken: nach %.1f s kein /race_direction -- faehrt mit "
                    "der eigenen Messung %s (Perzeption ohne start_from_bay?)."
                    % (gewartet, eigene), throttle_duration_sec=60.0)
            elif self.race_direction != eigene:
                # Auf der falschen Seite ausparken heisst in die Buchtwand
                # fahren. Lieber gar nicht.
                self._ausparken_abbruch(
                    "Richtung widerspruechlich: eigene Messung %s (offene Seite), "
                    "Perzeption %s. Er faehrt nicht los." % (eigene, self.race_direction))
                return
            # Abtastung der Startgeraden abwarten: sie laeuft im Stand, jede
            # Bewegung bricht sie ab. Erst mit 'complete' ist /obstacles fuer die
            # Startgerade vollstaendig -- ein fehlender Sitz ist dann GEMESSEN
            # frei, nicht uebersehen. Davon haengt die Wahl der Ausparkfolge ab.
            if self.start_scan_state not in ('complete', 'incomplete'):
                if self.ausp_scan_wart_t0 is None:
                    self.ausp_scan_wart_t0 = jetzt
                gewartet = jetzt - self.ausp_scan_wart_t0
                if gewartet < self.ausparken_warte_scan_s:
                    self.get_logger().info(
                        "Ausparken: warte auf die Abtastung der Startgeraden "
                        "(%s, %.1f s)." % (self.start_scan_state or 'noch nichts', gewartet),
                        throttle_duration_sec=0.5)
                    return
                self.get_logger().warn(
                    "Ausparken: nach %.1f s kein Ergebnis der Startgeraden-Abtastung "
                    "(%s) -- faehrt mit dem, was bekannt ist."
                    % (gewartet, self.start_scan_state or 'kein /start_scan_state'),
                    throttle_duration_sec=60.0)
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
                if self.ausp_modus == 'ein':
                    self._einparken_fertig(x, y, theta)
                elif self.ausp_modus == 'anfahrt':
                    self.state = 'PARK_NACHMESSEN'
                    self.park_t0 = self.now_s()
                    self.park_mittel = []
                elif self.ausp_modus == 'zurueck':
                    self._erste_ecke_zurueck_fertig(x, y, theta)
                else:
                    self._ausparken_fertig(x, y, theta)
                return
            lenk, cm = self.ausp_schritte[self.ausp_index]

            # Einparken: Leerzuege (0 cm, nur Lenkung) ueberspringen. Sie kosten
            # je 0,6 s Lenkwartezeit plus Quittung, und der naechste echte Zug
            # stellt seine Lenkung ohnehin selbst ein. Der Index laeuft weiter,
            # damit die Zuordnung zur Ausparkbahn stimmt. Ausparken unveraendert.
            if (self.ausp_modus == 'ein' and self.einparken_leerzuege_weg
                    and abs(cm) < 0.05 and self.ausp_phase == 'lenken'):
                self.ausp_index += 1
                self.ausp_lenk_gesendet = False
                return

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
                if self.ausp_modus == 'ein':
                    cm = self._park_bogen_korrigieren(self.ausp_index, lenk, cm,
                                                      x, y, theta)
                    self.ausp_schritte[self.ausp_index] = (lenk, cm)
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
                # Besser: der Encoder. Die Differenz zweier Quittungen ist die
                # Wellendrehung dieses Zuges -- ein Sprung der Lokalisierung
                # kann den Zug dann nicht mehr faelschlich abbrechen (so
                # geschehen im CCW-Ausparktest). Nur fuer den ersten Zug einer
                # Folge fehlt der Vorgaenger, dort bleibt es bei der Pose.
                if self.ausp_pos_prev is not None:
                    gedreht = _pos - self.ausp_pos_prev
                    fehlt = abs(grad_zu_cm(gedreht - cm_zu_grad(cm))) / 100.0
                self.ausp_pos_prev = _pos

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
                if self.ausp_modus == 'aus':
                    self.ausp_bahn.append((x, y, theta))
                elif self.ausp_modus == 'ein':
                    self._park_zug_abweichung(self.ausp_index, theta)
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
        self.ausp_ende_pose = (x, y, theta)   # Vergleich nach dem Scan-Halt
        self.get_logger().info(
            "Ausparken fertig: Pose (%.2f, %.2f), Kurs %+.1f grad."
            % (x, y, math.degrees(theta)))
        if self.ausparken_nur:
            self.state = 'DONE'
            self.get_logger().info(
                "ausparken_nur gesetzt -- Regler haelt hier an.")
            return
        # Die Richtung JETZT bekanntgeben, nicht erst nach dem Halt: die
        # Wahrnehmung braucht sie, um ueberhaupt mit der
        # Startpositionserkennung anzufangen -- und die soll im Stillstand
        # laufen, also genau waehrend des Halts.
        self._ausparken_richtung_uebernehmen()
        if self.ausparken_halt_s > 0.0:
            self.state = 'AUSPARK_SCAN'
            self.ausp_t0 = self.now_s()
            self.ausp_scan_hier = None
            self.get_logger().info(
                "Nach dem Ausparken: warte auf die Eckengeometrie, dann entscheide, "
                "ob hier gescannt wird.")
            return
        self._ausparken_uebergeben()

    def _park_gerade(self):
        """Aussenwand (einwaerts gerichtete HNF) und Fahrtrichtung der
        Zielgeraden = Startgerade."""
        w = self._entry_wall_idx(self.corner_idx)
        nx, ny, dw = self.walls[w]
        tx, ty = self.arc['travel']
        return nx, ny, dw, tx, ty

    def _park_lage(self, x, y, theta):
        """Wo liegt die Einpark-Startpose relativ zum Roboter -- WANDBEZOGEN.

        d    laengs, entlang der Geraden (+ voraus). Aus der gemerkten Pose:
             laengs passen Startrahmen und Karte (front_wall_x und Eckpunkt
             stimmen ueberein).
        quer Abstand zur Aussenbande minus Parklinie (+ = zu weit innen).
             NICHT aus der gemerkten Pose -- die liegt quer ~14 cm daneben.
        kurs gegen die Richtung der Geraden; das Ausparken endet parallel.
        """
        nx, ny, dw, tx, ty = self._park_gerade()
        px, py, _pth = self.park_start
        d = (px * tx + py * ty) - (x * tx + y * ty)
        quer = ((nx * x + ny * y) - dw) - self.park_q
        kurs = wrap(theta - math.atan2(ty, tx))
        return d, quer, kurs

    def _park_mittelpose(self):
        """Mittel der im Stand gesammelten Posen. Ein Einzelwert traegt das
        EKF-Rauschen voll in den Korrekturzug -- und weil jeder Vorwaertszug
        ~1 cm Ueberschuss hat, pendelt die Anfahrt dann hin und her. Simuliert
        mit dem gemessenen Fahrmodell: Einzelwert 89 Prozent, gemittelt 100."""
        n = len(self.park_mittel)
        mx = sum(p[0] for p in self.park_mittel) / n
        my = sum(p[1] for p in self.park_mittel) / n
        mth = math.atan2(sum(math.sin(p[2]) for p in self.park_mittel),
                         sum(math.cos(p[2]) for p in self.park_mittel))
        return mx, my, mth

    def _park_lage_ok(self, d, quer, kurs, max_anfahrt):
        gruende = []
        if abs(quer) > self.einparken_quer_tol:
            gruende.append("%.1f cm seitlich (Grenze %.1f)"
                           % (quer * 100, self.einparken_quer_tol * 100))
        if abs(math.degrees(kurs)) > self.einparken_kurs_tol_grad:
            gruende.append("Kurs %+.1f grad (Grenze %.1f)"
                           % (math.degrees(kurs), self.einparken_kurs_tol_grad))
        if abs(d) > max_anfahrt:
            gruende.append("Anfahrt %.2f m (Grenze %.2f)" % (abs(d), max_anfahrt))
        return gruende

    def _park_zuege_starten(self, schritte, modus):
        self.ausp_schritte = schritte
        self.ausp_modus = modus
        self.state = 'AUSPARK_FAHREN'
        self.ausp_index = 0
        self.ausp_phase = 'lenken'
        self.ausp_lenk_gesendet = False

    def _park_halt(self, x, y, theta):
        """Pflichtstillstand nach drei Runden, danach zur Startpose anfahren.

        Eingeparkt wird mit der UMGEKEHRTEN Ausparkfolge, die er zu Beginn
        selbst gefahren ist -- am Roboter nachgemessen trifft der Rueckweg die
        Lueckenlage auf rund 2 cm, Achsen parallel. Das gilt aber nur, wenn die
        Folge exakt an der Pose beginnt, an der das Ausparken endete.
        """
        self.publish_stop()
        jetzt = self.now_s()
        rest = self.einparken_halt_s - (jetzt - self.park_t0)
        # Nur bei eingerasteter Lokalisierung messen: die Startpose der
        # Umkehrfolge muss auf 1-2 cm stimmen. Solange sie unsicher ist,
        # verwerfen, und danach die Mittelung neu beginnen.
        if not self._lok_ok() and not self.park_lok_unsicher:
            self.park_mittel = []
            if self.lok_warte_t0 is None:
                self.lok_warte_t0 = jetzt
            if rest <= 0.0 and jetzt - max(self.lok_warte_t0, self.park_t0
                                             + self.einparken_halt_s) \
                    > self.einparken_lok_warte_s:
                # Nichts zu verlieren: trotzdem einparken, aber ohne die
                # posenbasierten Korrekturen -- die Pose ist gerade nicht belastbar.
                self.park_lok_unsicher = True
                self.lok_warte_t0 = None
                self.get_logger().warn(
                    "Lokalisierung '%s' -- parkt trotzdem ein, aber ohne "
                    "Korrekturen (Pose nicht belastbar)." % self.lok_state)
            else:
                self.get_logger().warn("Einparken wartet auf Lokalisierung 'ok' (jetzt '%s')."
                                       % self.lok_state, throttle_duration_sec=0.5)
                return
        if self.lok_warte_t0 is not None:
            # gerade wieder 'ok': Mittelungsfenster von vorn
            self.lok_warte_t0 = None
            if rest < self.einparken_mittel_s:
                self.park_t0 = jetzt - (self.einparken_halt_s - self.einparken_mittel_s)
                rest = self.einparken_mittel_s
        if rest <= self.einparken_mittel_s:
            self.park_mittel.append((x, y, theta))
        if rest > 0.0:
            self.get_logger().info("Pflichtstillstand, noch %.1f s." % rest,
                                   throttle_duration_sec=0.5)
            return

        d, quer, kurs = self._park_lage(*self._park_mittelpose())
        self.park_mittel = []
        self.get_logger().info(
            "Einparken: Startpose %.1f cm %s, %.1f cm seitlich, Kursfehler "
            "%+.1f grad." % (abs(d) * 100, 'voraus' if d >= 0 else 'zurueck',
                             quer * 100, math.degrees(kurs)))
        gruende = self._park_lage_ok(d, quer, kurs, self.einparken_max_anfahrt)
        if gruende:
            # Nichts zu verlieren: melden und trotzdem einparken. Den Kurs
            # holen die Boegen wieder rein, solange die Sensoren reichen.
            self.get_logger().warn(
                "Einparken: Startlage ungenau (%s) -- parkt trotzdem ein."
                % '; '.join(gruende))
        if abs(d) > self.einparken_max_anfahrt:
            # Nur die Anfahrt deckeln: mehr als das faehrt Richtung Frontwand.
            d = math.copysign(self.einparken_max_anfahrt, d)

        # Buchtpruefung: landet die Umkehrfolge (= Startlage) zwischen den
        # gemessenen Waenden? Nur Warnung -- die Startlage WAR in der Bucht,
        # eine Abweichung spricht eher fuer eine unvollstaendige Messung.
        if self.park_ursprung is not None:
            r = self._bucht_abstaende(*self.park_ursprung)
            if r is None:
                self.get_logger().info(
                    "Einparken: keine Buchtmessung zur Pruefung%s."
                    % ('' if self.bucht is None else ' (Bucht liegt nicht um die Startlage)'))
            else:
                heck, front, luecke = r
                self._log(min(heck, front) < 0.005,
                    "Einparken: laut Buchtmessung erwartet Heck %.1f cm, Front "
                    "%.1f cm Luft (Luecke %.1f cm)." % (heck * 100, front * 100, luecke * 100))

        # Die Ausparkfolge JETZT sichern und umkehren: der Ausfuehrer bekommt
        # gleich die Anfahrtszuege, und die ueberschreiben ausp_schritte.
        self.park_einpark = self._einparkfolge_bilden()
        self.park_korr_angewandt = False
        self.ausp_pos_prev = None       # Welle hat sich in den Runden gedreht
        self.anfahrt_iter = 0
        if d > self.einparken_fahrt_ab and not self.park_lok_unsicher:
            # Pflichthalt ist vorbei: ab hier duerfen die Pylonen auf beliebiger
            # Seite passiert werden. Pfad dafuer neu planen.
            self.seiten_frei = True
            self._park_ueber_festlegen()
            self.plan_obstacle_path()
            self.state = 'PARK_FAHRT'
            self.get_logger().info(
                "Einparken: %.1f cm vorwaerts GEREGELT zur Startpose%s."
                % (d * 100, ', mit Hindernispfad' if self.obs_path else
                   ' auf der Parklinie'))
            return
        self._ausparken_pid(self.ausparken_pid)
        self._park_anfahren(d)

    def _park_anfahren(self, d):
        """Ein gerader Zug um d. Danach wird nachgemessen (PARK_NACHMESSEN):
        die Umrechnung Grad -> cm (R_EFF) liegt real rund 6 Prozent daneben,
        dazu ~1 cm Ueberschuss je Vorwaertszug. Blind gefahren laege er bei 1 m
        Anfahrt 7 cm daneben -- nachgemessen nach 1-3 Zuegen auf 1 cm."""
        if abs(d) < self.einparken_laengs_tol:
            self._park_folge_starten()
            return
        self.anfahrt_iter += 1
        offen_links = (self.ausp_richtung == 'CCW')
        # Um den bekannten Fehler ueber Grund verkuerzen (einparken_zug_skala).
        # Nie unter die Haelfte kuerzen: ein ganz kurzer Rest soll nicht zu
        # einem Null-Zug werden.
        rest = abs(d)
        if d > 0.0:
            rest = max(rest - self.einparken_zug_ueberschuss, 0.5 * rest)
        befehl = math.copysign(rest / max(self.einparken_zug_skala, 0.5), d)
        self.get_logger().info(
            "Einparken: Anfahrt %d, %+.1f cm gerade (befohlen %+.1f cm Radweg)."
            % (self.anfahrt_iter, d * 100, befehl * 100))
        self._park_zuege_starten(spiegeln([(0.0, befehl * 100.0)], offen_links), 'anfahrt')

    def _park_nachmessen(self, x, y, theta):
        """Nach einem Anfahrtszug kurz ruhen lassen, dann Rest bestimmen."""
        if self.park_lok_unsicher:
            self._park_folge_starten()      # Pose nicht belastbar: nicht nachmessen
            return
        if not self._lok_ok():
            jetzt = self.now_s()
            self.park_mittel = []
            if self.lok_warte_t0 is None:
                self.lok_warte_t0 = jetzt
            if jetzt - self.lok_warte_t0 > self.einparken_lok_warte_s:
                self.park_lok_unsicher = True
                self.lok_warte_t0 = None
                self.get_logger().warn(
                    "Lokalisierung '%s' beim Nachmessen -- startet die Einparkfolge "
                    "trotzdem, ohne Korrekturen." % self.lok_state)
                self._park_folge_starten()
                return
            self.park_t0 = jetzt        # nach 'ok' Beruhigung + Mittelung von vorn
            return
        self.lok_warte_t0 = None
        t = self.now_s() - self.park_t0
        if t < self.einparken_nachmess_s:
            return                      # Fahrzeug und EKF beruhigen lassen
        self.park_mittel.append((x, y, theta))
        if t < self.einparken_nachmess_s + self.einparken_mittel_s:
            return
        d, quer, kurs = self._park_lage(*self._park_mittelpose())
        self.park_mittel = []
        # Der erste Vollanschlag-Bogen holt den Kursfehler ueber seine Laenge
        # rein -- dabei verschiebt sich der ganze Bogen laengs (Lauf 34: -6,3
        # grad -> 2,7 cm kuerzer -> Nase in der Buchtwand). Die Startpose
        # wandert deshalb um genau diesen Betrag mit.
        d_kurs = self._park_kurs_laengs(kurs)
        d += d_kurs
        if abs(d) < self.einparken_laengs_tol:
            self.get_logger().info(
                "Einparken: Startpose erreicht (%.1f cm, %.1f cm seitlich, "
                "%+.1f grad%s) nach %d Anfahrtszug/-zuegen."
                % (d * 100, quer * 100, math.degrees(kurs),
                   '' if abs(d_kurs) < 0.002 else
                   ', Startpose wegen Kurs um %+.1f cm verschoben' % (d_kurs * 100),
                   self.anfahrt_iter))
            self._park_folge_starten()
            return
        if abs(d_kurs) >= 0.002:
            self.get_logger().info(
                "Einparken: Kurs %+.1f grad -> Startpose %+.1f cm laengs verschoben "
                "(der erste Bogen wird zum Kursausgleich %s)."
                % (math.degrees(kurs), d_kurs * 100,
                   'kuerzer' if d_kurs < 0 else 'laenger'))
        if (d < -self.einparken_rueck_ab and not self.park_lok_unsicher
                and self.rueck_versuche < 2):
            self._park_rueck_starten(-d)
            return
        if self.anfahrt_iter >= self.einparken_anfahrt_max_zuege:
            self.get_logger().warn(
                "Einparken: nach %d Anfahrtszuegen noch %.1f cm laengs daneben "
                "-- startet die Folge trotzdem." % (self.anfahrt_iter, d * 100))
            self._park_folge_starten()
            return
        # Quer- und Kursfehler kann ein gerader Zug nicht beheben -- nur melden.
        # Den Kurs holt der erste Bogen der Einparkfolge wieder rein.
        for g in self._park_lage_ok(0.0, quer, kurs, 1.0):
            self.get_logger().warn("Einparken: %s -- wird im Bogen korrigiert." % g)
        self._park_anfahren(max(-0.30, min(0.30, d)))

    def _park_kurs_laengs(self, kurs):
        """Um wie viel die Startpose laengs wandern muss (+ = weiter voraus),
        damit der erste Vollanschlag-Bogen trotz Kursfehler ``kurs`` (rad, zur
        Geraden) dort endet, wo er ohne Fehler geendet haette. Gerechnet wie
        _park_bogen_korrigieren: der Bogen wird auf den Soll-Endkurs
        verlaengert/verkuerzt, Grenze einparken_korr_max."""
        folge = list(getattr(self, 'park_einpark', None) or [])
        if not folge or abs(kurs) < math.radians(0.5):
            return 0.0
        k = next((i for i, (lenk, cm) in enumerate(folge)
                  if abs(lenk) >= 50.0 and abs(cm) >= 1.0), None)
        if k is None:
            return 0.0
        vor = folge[:k]
        lenk, cm = folge[k]
        th_vor = bahn((0.0, 0.0, 0.0), vor)[-1][0][2] if vor else 0.0
        soll = bahn((0.0, 0.0, 0.0), vor + [(lenk, cm)])[-1][0]
        geplant = wrap(soll[2] - th_vor)
        if abs(math.degrees(geplant)) < 5.0:
            return 0.0
        f = wrap(soll[2] - (th_vor + kurs)) / geplant
        f = max(1.0 - self.einparken_korr_max, min(1.0 + self.einparken_korr_max, f))
        ist = bahn((0.0, 0.0, kurs), vor + [(lenk, cm * f)])[-1][0]
        return soll[0] - ist[0]

    def _einparkfolge_bilden(self):
        """Einparkfolge (Leitungswerte, in Fahrreihenfolge): SCHRITTE_EINPARKEN_CW
        bzw. _CCW aus ausparken.py. Fehlt die Liste oder ist sie unbrauchbar, die
        Umkehrung der normalen Ausparkfolge wie bisher.

        Weicht sie von dieser Umkehrung ab, passt die Ausparkbahn nicht mehr
        Zug fuer Zug (_park_ziel_kurse nimmt an: Einparkzug k = Ausparkzug
        n-1-k rueckwaerts) -- dann ohne Bogenkorrektur."""
        umkehr = [(lenk, -cm) for lenk, cm in
                  reversed(self.ausp_schritte_std or self.ausp_schritte)]
        try:
            flach, name = einparkfolge(self.ausp_richtung)
            folge = spiegeln(schritte_aus_flach(flach), self.ausp_richtung == 'CCW')
        except (ValueError, KeyError) as fehler:
            self.get_logger().warn(
                "Einparkfolge: %s -- nehme die Umkehrung der Ausparkfolge." % fehler)
            return umkehr
        gleich = (len(folge) == len(umkehr) and all(
            abs(a[0] - b[0]) < 1e-6 and abs(a[1] - b[1]) < 1e-6
            for a, b in zip(folge, umkehr)))
        if gleich:
            self.get_logger().info(
                "Einparkfolge aus %s (%d Zuege, = Umkehrung der Ausparkfolge)."
                % (name, len(folge)))
        else:
            self.get_logger().info(
                "Einparkfolge aus %s (%d Zuege, weicht von der Umkehrung der "
                "Ausparkfolge ab -- ohne Bogenkorrektur): %s"
                % (name, len(folge), ", ".join("%+.0f%%/%+.1fcm" % z for z in folge)))
            self.ausp_bahn = []
        return folge

    def _park_ziel_kurse(self, k):
        """Kurs am Anfang und am Ende von Einparkzug k laut Ausparkbahn.
        Einparkzug k faehrt Ausparkzug j = n-1-k rueckwaerts: er beginnt, wo
        dieser endete, und endet, wo dieser begann."""
        n = len(self.park_einpark)
        if len(self.ausp_bahn) != n + 1 or not (0 <= k < n):
            return None
        j = n - 1 - k
        return self.ausp_bahn[j + 1][2], self.ausp_bahn[j][2]

    def _park_sensoren_reichen(self, x, y):
        """Korrigieren nur, solange die Pose traegt: Lokalisierung 'ok' und
        der Lidar noch nicht in der Bucht (dort sieht er die nahe Wand nicht)."""
        if self.park_lok_unsicher or not self._lok_ok():
            return False
        try:
            nx, ny, dw, _tx, _ty = self._park_gerade()
        except Exception:
            return False
        return (nx * x + ny * y) - dw >= self.einparken_korr_min_abstand

    def _park_bogen_korrigieren(self, k, lenk, cm, x, y, theta):
        """Vollanschlag-Bogen so verlaengern/verkuerzen, dass er mit dem Kurs
        endet, den die Ausparkbahn an dieser Stelle hatte.

        Den Kursfehler bringen vor allem die GERADEN Zuege mit: bei -2 Prozent
        bleiben die Raeder durch das Lenkspiel 2-5 grad auf der Seite stehen,
        von der sie kamen. Kleine Lenkkorrekturen verschluckt dasselbe Spiel --
        der Anschlag dagegen hat keins. Also Kurs ueber die Bogenlaenge."""
        if abs(lenk) < 50.0 or abs(cm) < 1.0:
            return cm                       # gerader Zug: nicht korrigierbar
        ziel = self._park_ziel_kurse(k)
        if ziel is None:
            return cm
        th_start, th_ende = ziel
        geplant = wrap(th_ende - th_start)
        noetig = wrap(th_ende - theta)
        abweichung = math.degrees(wrap(theta - th_start))
        if abs(math.degrees(geplant)) < 5.0:
            return cm
        if not self._park_sensoren_reichen(x, y):
            if abs(abweichung) > self.einparken_kurs_warn_grad:
                self.get_logger().warn(
                    "Einparken Zug %d: %+.1f grad neben der Ausparkbahn, Sensoren "
                    "reichen hier nicht mehr zum Korrigieren -- faehrt wie geplant."
                    % (k + 1, abweichung))
            return cm
        f = noetig / geplant
        f_k = max(1.0 - self.einparken_korr_max, min(1.0 + self.einparken_korr_max, f))
        cm_neu = cm * f_k
        if abs(cm_neu - cm) >= 0.3:
            self._log(f_k != f,
                "Einparken Zug %d: Kurs %+.1f grad neben der Ausparkbahn -> Bogen "
                "%.1f statt %.1f cm%s." % (
                    k + 1, abweichung, abs(cm_neu), abs(cm),
                    '' if f_k == f else ' (Korrektur begrenzt, voll waeren %.1f cm)'
                    % abs(cm * f)))
        return cm_neu

    def _park_zug_abweichung(self, k, theta):
        """Nach jedem Einparkzug melden, wie weit der Kurs neben der
        Ausparkbahn liegt -- ohne abzubrechen."""
        ziel = self._park_ziel_kurse(k)
        if ziel is None:
            return
        dev = math.degrees(wrap(theta - ziel[1]))
        self._log(abs(dev) > self.einparken_kurs_warn_grad,
            "Einparken Zug %d fertig: Kurs %+.1f grad neben der Ausparkbahn."
            % (k + 1, dev))

    def _park_uebergang(self, x, y, theta):
        """Drei Runden fertig, OHNE anzuhalten ins Einparken uebergehen: die
        Seite an den Pylonen ist ab hier frei, der Pfad wird dafuer neu
        geplant, und er faehrt geregelt zur Einpark-Startpose weiter."""
        self.get_logger().info(
            "Drei Runden fertig (%d Ecken) -- Seiten frei, faehrt ohne Halt zur "
            "Einpark-Startpose." % self.corner_count)
        if not self._lok_ok():
            self.park_lok_unsicher = True
            self.get_logger().warn(
                "Lokalisierung '%s' beim Uebergang -- parkt ohne posenbasierte "
                "Korrekturen." % self.lok_state)
        self.park_einpark = self._einparkfolge_bilden()
        self.park_korr_angewandt = False
        self.ausp_pos_prev = None
        self.anfahrt_iter = 0
        self.rueck_versuche = 0
        self.seiten_frei = True
        self._park_ueber_festlegen()
        self.plan_obstacle_path()
        self.state = 'PARK_FAHRT'

    def _park_ueber_festlegen(self):
        """Ueberfahrweite fuer diese Anfahrt: einparken_ueberfahren, aber nur
        bis vor eine Pylone, die AUF der Parklinie steht (daran vorbei geht es
        nicht, und rueckwaerts faehrt er stur die Parklinie). Unter der
        Rueckfahr-Schwelle lohnt es nicht -- dann wie bisher."""
        self.park_ueber = 0.0
        weite = self.einparken_ueberfahren
        if weite <= 0.0 or self.park_start is None or self.park_lok_unsicher:
            return
        try:
            nx, ny, dw, tx, ty = self._park_gerade()
        except Exception:
            return
        w = self._entry_wall_idx(self.corner_idx)
        s_ps = self.park_start[0] * tx + self.park_start[1] * ty
        halb = FZ_BREITE / 2.0 + BLOCK_HALB + 0.05
        grund = ''
        for o in (self.obstacles or []):
            if o.get('wall') != w:
                continue
            s_o = o['x'] * tx + o['y'] * ty - s_ps
            q_o = (nx * o['x'] + ny * o['y']) - dw
            if s_o <= 0.0 or abs(q_o - self.park_q) >= halb:
                continue
            frei = s_o - FZ_NASE - BLOCK_HALB - 0.05
            if frei < weite:
                weite = frei
                grund = ' (Pylone %.2f m hinter der Startpose auf der Parklinie)' % s_o
        if weite < self.einparken_rueck_ab + 0.02:
            self.get_logger().info(
                "Einparken: kein Ueberfahren%s -- Anfahrt wie bisher." % (grund or ''))
            return
        self.park_ueber = weite
        self.get_logger().info(
            "Einparken: faehrt %.0f cm ueber die Startpose hinaus und setzt dann "
            "geregelt zurueck%s." % (weite * 100, grund))

    def _park_front_abstand(self):
        """Abstand der Einpark-Startpose zur Frontwand der Zielgeraden."""
        if self.park_start is None or self.corners is None or self.arc is None:
            return None
        fc = self.corners[self.corner_idx]
        tx, ty = self.arc['travel']
        return (fc[0] - self.park_start[0]) * tx + (fc[1] - self.park_start[1]) * ty

    def _ziel_abstand(self):
        """Haltepunkt am Ziel (base_link zur Frontwand).

        Mit Einparken so nah an der Einpark-Startpose wie erlaubt: dann ist
        die Anfahrt danach kurz. Eine lange blinde Anfahrt hat einmal 55 cm
        mit 7,6 grad Kursfehler gefahren und kam 12 cm seitlich und 17 grad
        schief an. Die Zone gilt, wenn ziel_zone_ganzes_fz, fuers ganze
        Fahrzeug -- dann muss auch die Nase noch drin sein."""
        if self._park_aktiv() and self.einparken_halt_s <= 0.0:
            if not self._ziel_gemeldet:
                self._ziel_gemeldet = True
                f_ps = self._park_front_abstand()
                self.get_logger().info(
                    "Kein Pflichthalt: Pylonen bis %.2f m vor der Frontwand nach "
                    "Regel, danach Seite frei; Einpark-Startpose bei %s m."
                    % (self.seiten_frei_ab, '%.2f' % f_ps if f_ps is not None else '?'))
            return self.seiten_frei_ab
        if not (self.ziel_an_parkstart and self._park_aktiv()):
            return self.finish_front_dist
        f_ps = self._park_front_abstand()
        if f_ps is None:
            return self.finish_front_dist
        nase = FZ_NASE if self.ziel_zone_ganzes_fz else 0.0
        heck = -FZ_HECK if self.ziel_zone_ganzes_fz else 0.0
        lo = self.ziel_zone_min + nase + self.ziel_zone_rand
        hi = self.ziel_zone_max - heck - self.ziel_zone_rand
        # frueh: hinterer Zonenrand -- die Startpose liegt dann immer voraus,
        # und die Anfahrt kann immer vorwaerts geregelt gefahren werden
        ziel = hi if self.ziel_frueh else max(lo, min(hi, f_ps))
        if not self._ziel_gemeldet:
            self._ziel_gemeldet = True
            self.get_logger().info(
                "Haltepunkt %.2f m vor der Frontwand (Einpark-Startpose bei %.2f m, "
                "erlaubt %.2f..%.2f m) -> danach %.0f cm Anfahrt."
                % (ziel, f_ps, lo, hi, abs(ziel - f_ps) * 100))
        return ziel

    def _park_fahrt(self, x, y, theta):
        """Vorwaerts zur Startpose, GEREGELT: Stanley auf der Parklinie im
        Kriechtempo, Zielbremsung auf den Abstand der Startpose. Kurs und
        Querlage werden unterwegs korrigiert -- ein gerader ESP-Zug kann das
        nicht, und das Lenkspiel dreht ihn dabei noch weiter weg."""
        fc = self.corners[self.corner_idx]
        tr = self.arc['travel']
        vorn = (fc[0] - x) * tr[0] + (fc[1] - y) * tr[1]
        lead = abs(self.v_ist) * self.finish_lead_time
        vorhalt = self.einparken_vorhalt
        if self.park_fahrt_ziel_f is not None:
            ziel_f = self.park_fahrt_ziel_f
        elif self.park_ueber > 0.0:
            ziel_f = self._park_front_abstand() - self.park_ueber
            vorhalt = 0.0               # zurueck faehrt ohnehin PARK_RUECK
        else:
            ziel_f = self._park_front_abstand()
        rest = vorn - ziel_f - lead - vorhalt
        if rest <= self.einparken_laengs_tol:
            self.publish_stop()
            if self.park_ueber > 0.0 and self.park_fahrt_ziel_f is None:
                self.get_logger().info(
                    "Einparken: %.1f cm ueber die Startpose gefahren -- misst nach "
                    "und setzt geregelt zurueck." % ((self.park_ueber - rest) * 100))
            else:
                self.get_logger().info(
                    "Einparken: geregelte Anfahrt fertig, %.1f cm vor der Startpose "
                    "(Vorhalt %.0f cm, den Rest faehrt der ESP als Positionsfahrt)."
                    % ((rest + vorhalt) * 100, vorhalt * 100))
            self._ausparken_pid(self.ausparken_pid)
            self.state = 'PARK_NACHMESSEN'
            self.park_t0 = self.now_s()
            self.park_mittel = []
            return
        px_, py_, pth_ = self._pose_nach_totzeit(x, y, theta)
        omega = None
        if self.obs_path:
            omega = self._stanley_follow_path(px_, py_, pth_, self.obs_path)
        if omega is None:
            omega = self._stanley_steer(px_, py_, pth_, self.arc['LA'], tr)
        v = min(self.v_park_fahrt, self.v_park_anfahrt,
                math.sqrt(2.0 * self.finish_decel * max(rest, 0.0)))
        self.publish_cmd(max(v, self.v_finish_min), omega)

    def _lenk_prozent(self, delta):
        """Lenkwinkel (rad, + = links) -> Servoprozent, ueber die gemessene
        Kennlinie aus steer_calib.json (dieselbe wie beim Ausparken)."""
        paare = sorted((g, p) for p, g in LENK_KENNLINIE)
        g = math.degrees(delta)
        if g <= paare[0][0]:
            return paare[0][1]
        if g >= paare[-1][0]:
            return paare[-1][1]
        for (g0, p0), (g1, p1) in zip(paare, paare[1:]):
            if g0 <= g <= g1:
                return p0 + (p1 - p0) * (g - g0) / (g1 - g0) if g1 > g0 else p0
        return paare[-1][1]

    def _rueck_lenkwinkel(self, x, y, theta):
        """Stetiges Rueckwaerts-Gesetz, Hinterachse auf der Parklinie:
            delta = k_kurs * psi - k_quer * e
        e = Versatz nach LINKS der Fahrtrichtung, psi = Kursfehler. Rueckwaerts
        kehrt sich die Wirkung der Lenkung auf den Kurs um (dpsi = -u tan d / L)
        -- deshalb NICHT das Stanley-Gesetz der Vorwaertsfahrt."""
        nx, ny, dw, tx, ty = self._park_gerade()
        seite = 1.0 if (nx * -ty + ny * tx) >= 0.0 else -1.0
        e = seite * (((nx * x + ny * y) - dw) - self.park_q)
        psi = wrap(theta - math.atan2(ty, tx))
        if self.rueck_praed_s > 0.0:
            u, T = abs(self.v_ist), self.rueck_praed_s
            e -= u * T * math.sin(psi)
            psi -= u * T * math.tan(self.rueck_delta) / RADSTAND
        d = self.rueck_k_kurs * psi - self.rueck_k_quer * e
        grenze = math.radians(self.rueck_max_lenk_grad)
        return max(-grenze, min(grenze, d)), e, psi

    def _park_rueck_starten(self, strecke):
        self.rueck_versuche += 1
        self.rueck_strecke = strecke
        self.rueck_phase = 'lenken'
        self.rueck_t0 = self.now_s()
        self.rueck_delta_max = 0.0
        self.state = 'PARK_RUECK'
        self.get_logger().info(
            "Einparken: %.1f cm rueckwaerts zur Startpose, stetig geregelt "
            "(Versuch %d)." % (strecke * 100, self.rueck_versuche))

    def _park_rueck(self, x, y, theta):
        """Der ESP faehrt die Strecke als EINEN Zug; waehrenddessen lenkt der
        Controller in jedem Takt ueber die Rohlenkung nach. So bleibt die
        Regelung stetig, ohne von der Lenkumrechnung der Bruecke fuer negative
        Geschwindigkeit abzuhaengen -- die ist nur vorwaerts kalibriert."""
        jetzt = self.now_s()
        delta, e, psi = self._rueck_lenkwinkel(x, y, theta)
        self.rueck_delta = delta
        self.rueck_delta_max = max(self.rueck_delta_max, abs(delta))
        self.pub_steer.publish(Float32(data=float(self._lenk_prozent(delta))))
        if self.rueck_phase == 'lenken':
            if jetzt - self.rueck_t0 < self.ausparken_lenk_wartezeit:
                return
            self.ausp_move_done = None
            self.rueck_gesendet = jetzt
            self.pub_move.publish(Float32(data=float(cm_zu_grad(-self.rueck_strecke * 100.0))))
            self.rueck_phase = 'fahren'
            return
        q = self.ausp_move_done
        fertig = q is not None and q[0] >= self.rueck_gesendet
        if not fertig and jetzt - self.rueck_gesendet > self.ausparken_zug_timeout:
            self.pub_motor.publish(Int32(data=0))
            self.get_logger().warn("Einparken rueckwaerts: keine Quittung -- abgeloest.")
            fertig = True
        if not fertig:
            return
        if q is not None:
            self.ausp_pos_prev = q[3]        # Encoder-Bezug fuer den naechsten Zug
        self.get_logger().info(
            "Einparken rueckwaerts fertig: %.1f cm neben der Parklinie, Kurs %+.1f "
            "grad, groesster Lenkwinkel %.1f grad."
            % (e * 100, math.degrees(psi), math.degrees(self.rueck_delta_max)))
        self.state = 'PARK_NACHMESSEN'
        self.park_t0 = jetzt
        self.park_mittel = []

    def _park_folge_starten(self):
        if getattr(self, 'park_korr_angewandt', False):
            self._park_zuege_starten(list(self.park_einpark), 'ein')   # schon korrigiert
            return
        self.park_korr_angewandt = True
        folge = []
        for k, (lenk, cm) in enumerate(self.park_einpark):
            korr = 0.0
            if abs(cm) >= 0.5:                      # Nullzuege (nur Lenkung) bleiben
                korr = (self.einparken_vor_korr_cm if cm > 0.0
                        else self.einparken_rueck_korr_cm)
            if k < len(self.einparken_zug_korr):
                korr += self.einparken_zug_korr[k]
            if korr != 0.0 and abs(cm) >= 0.5:
                betrag = max(0.5, abs(cm) + korr)   # Richtung bleibt, Mindestweg 0,5 cm
                neu = math.copysign(betrag, cm)
                self.get_logger().info(
                    "Einparken Zug %d: %+.1f statt %+.1f cm (Korrektur %+.1f)."
                    % (k + 1, neu, cm, korr))
                cm = neu
            folge.append((lenk, cm))
        self.park_einpark = folge
        if len(self.ausp_bahn) != len(folge) + 1:
            self._park_referenz_aus_folge(folge)
        self.get_logger().info(
            "Einparken: %d Zuege aus der umgekehrten Ausparkfolge." % len(folge))
        self._park_zuege_starten(list(folge), 'ein')

    def _park_referenz_aus_folge(self, folge):
        """Soll-Kurse fuer die Bogenkorrektur, wenn es keine gemessene
        Ausparkbahn gibt (eigene Einparkfolge, innen/mitte-Folge, Einparktest).

        Die Folge wird mit dem Fahrmodell ab Kurs der Geraden abgefahren; die
        Kurse an den Zuggrenzen sind die Sollwerte. Ohne das blieb ein
        Kursfehler der Startlage (gerader Anfahrtszug mit Lenkspiel: bis zu
        4-5 grad) unkorrigiert bis in die Luecke. ausp_bahn wird in der
        Reihenfolge des Ausparkens abgelegt (_park_ziel_kurse: Einparkzug k =
        Ausparkzug n-1-k rueckwaerts)."""
        try:
            _nx, _ny, _dw, tx, ty = self._park_gerade()
            th0 = math.atan2(ty, tx)
        except Exception:
            if self.park_ursprung is None:
                return
            th0 = self.park_ursprung[2]
        grenzen = {}
        for pose, nr in bahn((0.0, 0.0, th0), folge):
            grenzen[nr] = pose
        posen = [grenzen.get(k) for k in range(len(folge) + 1)]
        for k in range(1, len(posen)):          # Zug ohne Weg: Pose bleibt
            if posen[k] is None:
                posen[k] = posen[k - 1]
        self.ausp_bahn = list(reversed(posen))
        self.get_logger().info(
            "Einparken: Soll-Kurse aus dem Fahrmodell (keine gemessene Ausparkbahn), "
            "Bogenkorrektur aktiv: %s, Ende %+.1f grad zur Geraden."
            % (", ".join("%+.1f" % math.degrees(wrap(p[2] - th0)) for p in posen[1:]),
               math.degrees(wrap(posen[-1][2] - th0))))

    def _einparken_fertig(self, x, y, theta):
        # Wie beim Abbruch die Positionsfahrt ausdruecklich abloesen: /cmd_vel
        # erreicht den ESP waehrend einer Positionsfahrt nicht. Haelt er sonst
        # seine letzte Zielposition weiter, faehrt er beim Zurueckstellen fuer
        # den naechsten Lauf dorthin zurueck. Hier gefahrlos -- danach DONE.
        self.pub_motor.publish(Int32(data=0))
        self._ausparken_pid(self.ausparken_pid_nachher)
        self.publish_stop()
        self.state = 'DONE'
        r = self._bucht_abstaende(x, y, theta)
        if r is not None:
            heck, front, _ = r
            self._log(min(heck, front) < 0.0,
                "Endlage laut Buchtmessung: Heck %.1f cm, Front %.1f cm Luft%s."
                % (heck * 100, front * 100,
                   '' if min(heck, front) >= 0.0 else ' -- steht NICHT ganz in der Bucht'))
        if not self._lok_ok():
            self.get_logger().warn(
                "Lokalisierung beim Einparken '%s' -- die Endlage ist NICHT gesichert, "
                "die Zahlen hier koennen falsch sein." % self.lok_state)
        # Wandbezogen berichten -- genau das, was man mit dem Lineal nachmisst.
        try:
            nx, ny, dw, tx, ty = self._park_gerade()
            q = (nx * x + ny * y) - dw
            kurs = math.degrees(wrap(theta - math.atan2(ty, tx)))
            achsdiff = 0.105 * math.sin(math.radians(kurs)) * 100   # Radstand
            self.get_logger().info(
                "EINGEPARKT. base_link %.1f cm von der Aussenbande (erwartet "
                "%s), Kurs %+.1f grad zur Bande = %.1f cm Achsdifferenz "
                "(Regel: hoechstens 2 cm)."
                % (q * 100,
                   '%.1f cm' % (self.park_q_luecke * 100)
                   if self.park_q_luecke is not None else '?',
                   kurs, abs(achsdiff)))
        except Exception:
            self.get_logger().info("EINGEPARKT bei (%.2f, %.2f)." % (x, y))

    def _ausparken_scanhalt(self, x, y, theta):
        """Stillstehen, damit die Wahrnehmung die Startgerade aufnehmen kann.

        Der Roboter steht hier zum ersten Mal in der Spur und schaut sie
        entlang. Fahrend bricht die Bildrate der Kamera von 15,5 auf 2,5 Hz
        ein und die Farbausbeute von 38 auf 2 Prozent -- die Pylonen der
        Startgeraden sind jetzt besser zu sehen als spaeter im Lauf.
        """
        self.publish_stop()
        # Parklinie messen: Abstand zur Aussenbande, im Stand gemittelt.
        # CCW -> Aussenbande rechts, CW -> links.
        if self.wall_dist is not None and self.ausp_richtung:
            d_l, d_r = self.wall_dist
            self.park_q_proben.append(d_r if self.ausp_richtung == 'CCW' else d_l)
        # Wiederholen, solange gehalten wird: der Publisher ist nicht latched,
        # und wer erst jetzt zuhoert, soll sie trotzdem bekommen.
        if self.ausparken_setzt_richtung and self.ausp_richtung:
            self.pub_park_dir.publish(String(data=self.ausp_richtung))
        t = self.now_s() - self.ausp_t0
        if self.geometry_ready():
            if self.ausp_scan_hier is None:
                vorn = self._abstand_erste_ecke(x, y, theta)
                self.ausp_scan_hier = (vorn is not None
                                       and vorn < self.ausparken_scan_ersetzt_bis)
                self.get_logger().info(
                    "Ecke 1 liegt %s voraus -> %s." % (
                        '%.2f m' % vorn if vorn is not None else '?',
                        'hier scannen (%.1f s Halt)' % self.ausparken_halt_s
                        if self.ausp_scan_hier else
                        'gleich losfahren, Scan-Stopp am Ende der Geraden'))
            halt = self.ausparken_halt_s if self.ausp_scan_hier else self.ausparken_mess_s
            if t < halt:
                self.get_logger().info("Halt nach dem Ausparken, noch %.1f s."
                                       % (halt - t), throttle_duration_sec=0.5)
                return
            self._ausparken_uebergeben()
            return
        if t < self.ausparken_geo_timeout_s:
            self.get_logger().info("Warte auf die Eckengeometrie (%.1f s)." % t,
                                   throttle_duration_sec=0.5)
            return
        self.get_logger().warn(
            "Nach %.1f s noch keine Eckengeometrie -- fahre trotzdem los "
            "(Startmodus, Spurmitte)." % t)
        self._ausparken_uebergeben()

    def _abstand_erste_ecke(self, x, y, theta):
        """Abstand entlang des Kurses bis zum Eckpunkt der ersten Ecke."""
        idx = self.pick_first_corner(x, y, theta)
        if idx is None:
            return None
        c = self.corners[idx]
        return (c[0] - x) * math.cos(theta) + (c[1] - y) * math.sin(theta)

    def _ausparken_richtung_uebernehmen(self):
        """Wer hat bei der Fahrtrichtung das letzte Wort?

        Beim Parken ist die Richtung sicher messbar, aus der Eckengeometrie
        heraus nicht: der scan_processor sieht aus der Luecke keine brauchbare
        Ecke, latcht aber trotzdem. In einem Lauf standen dort CW aus dem
        Ausparken und CCW aus der Wahrnehmung. Mit ausparken_setzt_richtung
        gilt deshalb das Parken, sonst weiter /race_direction.
        """
        if not self.ausp_richtung:
            return
        passt = (self.race_direction is None
                 or self.race_direction == self.ausp_richtung)

        if not self.ausparken_setzt_richtung:
            if passt:
                self.get_logger().info(
                    "Fahrtrichtung %s (aus der Eckengeometrie), das Ausparken "
                    "kam auf dasselbe." % (self.race_direction or self.ausp_richtung))
            else:
                self.get_logger().error(
                    "WIDERSPRUCH: Ausparken %s, /race_direction %s. "
                    "ausparken_setzt_richtung ist aus, also gilt "
                    "/race_direction -- der Lauf geht damit vermutlich "
                    "andersherum als geplant."
                    % (self.ausp_richtung, self.race_direction))
            return

        # Das Parken hat das Wort.
        self.pub_park_dir.publish(String(data=self.ausp_richtung))
        if passt:
            self.get_logger().info(
                "Fahrtrichtung %s aus der Parkluecke%s."
                % (self.ausp_richtung,
                   ", bestaetigt durch die Eckengeometrie" if self.race_direction
                   else " (die Eckengeometrie hat noch nicht gelatcht)"))
        else:
            self.get_logger().warn(
                "Fahrtrichtung %s aus der Parkluecke -- /race_direction meldet "
                "%s. Der Latch stammt aus der Zeit IN der Luecke, wo die "
                "Eckenerkennung nichts Sinnvolles sehen kann. Das Parken "
                "gilt; der scan_processor bekommt sie ueber "
                "/parking_direction." % (self.ausp_richtung, self.race_direction))
        self.race_direction = self.ausp_richtung

    def _ausparken_uebergeben(self):
        """An die normale Zustandsmaschine abgeben.

        Die Richtung ist zu diesem Zeitpunkt schon gesetzt und verschickt --
        das passiert in _ausparken_fertig, vor dem Scan-Halt.
        """
        # Einpark-Startpose JETZT aufzeichnen, nicht direkt nach dem letzten
        # Zug: dazwischen liegt der Scan-Halt, in dem die Richtung an den
        # scan_processor geht und der die Karte umschaltet. Ein Sprung der
        # Pose beim Umschalten ist damit schon eingerechnet -- genau dieser
        # Sprung hat den CCW-Ausparktest verfaelscht. Der Roboter hat sich
        # seit dem letzten Zug nicht bewegt.
        if self.pose is not None and self.ausp_schritte:
            self.park_start = self.pose
            # Selbstdiagnose Kartenwechsel: seit dem letzten Zug steht er still,
            # jede Posenaenderung ist also ein Sprung der Lokalisierung. Die
            # Lueckenlage wurde VOR dem Sprung gemerkt und wird mitverschoben,
            # sonst misst die Schlussmeldung den Sprung statt der Parkgenauigkeit.
            if self.ausp_ende_pose is not None:
                ax, ay, ath = self.ausp_ende_pose
                bx, by, bth = self.park_start
                dth = wrap(bth - ath)
                sprung = math.hypot(bx - ax, by - ay)
                self._log(sprung > 0.01 or abs(math.degrees(dth)) > 1.0,
                    "Kartenwechsel im Scan-Halt: Pose um %.1f cm / %+.1f grad "
                    "gesprungen (Roboter stand still)."
                    % (sprung * 100, math.degrees(dth)))
                if self.park_ursprung is not None:
                    # starre Transformation alt -> neu auf die Lueckenlage
                    ux, uy, uth = self.park_ursprung
                    c, sn = math.cos(dth), math.sin(dth)
                    rx, ry = ux - ax, uy - ay
                    self.park_ursprung = (bx + c * rx - sn * ry,
                                          by + sn * rx + c * ry,
                                          wrap(uth + dth))
                neu = []
                c, sn = math.cos(dth), math.sin(dth)
                for (px_, py_, pth_) in self.ausp_bahn:
                    rx, ry = px_ - ax, py_ - ay
                    neu.append((bx + c * rx - sn * ry, by + sn * rx + c * ry,
                                wrap(pth_ + dth)))
                self.ausp_bahn = neu
            if self.ausparken_halt_s < 1.0:
                self.get_logger().warn(
                    "Einpark-Startpose ohne Beruhigungszeit aufgezeichnet "
                    "(ausparken_halt_s=%.1f) -- springt die Pose beim "
                    "Kartenwechsel, stimmt sie nicht." % self.ausparken_halt_s)
            self.get_logger().info(
                "Einpark-Startpose gemerkt: (%.3f, %.3f), Kurs %+.1f grad."
                % (self.park_start[0], self.park_start[1],
                   math.degrees(self.park_start[2])))
        self._parklinie_festlegen()
        self._park_versatz_anwenden()
        self.erste_ecke_pruefen = True
        # Der Taster ist bereits gedrueckt worden, sonst waeren wir nicht hier.
        self.button_pressed = True
        if self._erste_ecke_zuruecksetzen():
            return
        self.state = 'WAIT_INPUTS'
        self.get_logger().info("Weiter zum Rennen. Warte auf Eingaben...")

    def _einpark_test_vorbereiten(self):
        """Einpark-Test: Luecke, Einpark-Startpose, Parklinie und Einparkfolge
        so setzen, als waere er ausgeparkt. False = Geometrie fehlt noch."""
        if not self.geometry_ready() or self.pose is None:
            return False
        richtung = self.test_richtung
        if richtung not in ('CW', 'CCW'):
            self.get_logger().error("test_richtung=%r -- CW oder CCW." % richtung)
            self.state = 'DONE'
            return False
        if self.race_direction != richtung:
            self.get_logger().error(
                "Einpark-Test %s, /race_direction meldet %s -- steht er richtig "
                "herum auf der Startgeraden? Abbruch." % (richtung, self.race_direction))
            self.state = 'DONE'
            return False
        x, y, th = self.pose
        idx = self.pick_first_corner(x, y, th)
        if idx is None:
            return False
        nx, ny, dw = self.walls[self._entry_wall_idx(idx)]     # Aussenbande, n ins Feld
        tx, ty = -ny, nx
        if tx * math.cos(th) + ty * math.sin(th) < 0.0:
            tx, ty = -tx, -ty                                  # Fahrtrichtung
        cx, cy = self.corners[idx]                             # Ecke an der Frontwand
        f_b = self.test_bucht_front if self.test_bucht_front > 0.0 else (
            1.245 if richtung == 'CCW' else 1.96)
        q_b = self.test_bucht_q
        # Lueckenlage: f_b vor der Frontwand, q_b von der Aussenbande
        bx, by = cx - f_b * tx, cy - f_b * ty
        k = q_b - ((nx * bx + ny * by) - dw)
        bx, by = bx + k * nx, by + k * ny
        bth = math.atan2(ty, tx)
        self.park_ursprung = (bx, by, bth)
        ccw = richtung == 'CCW'
        std_laengs = self.park_std_laengs_ccw if ccw else self.park_std_laengs_cw
        std_quer = self.park_std_quer_ccw if ccw else self.park_std_quer_cw
        self.park_start = (bx + std_laengs * tx + std_quer * nx,
                           by + std_laengs * ty + std_quer * ny, bth)
        self.park_q_luecke = q_b
        self.park_q = q_b + std_quer
        self.ausp_richtung = richtung
        self.ausp_variante = 'test'
        self.ausp_bahn = []                    # keine Referenzbahn -> ohne Bogenkorrektur
        try:
            roh, herkunft = schritte_fuer(
                richtung, gemeinsam=self.ausparken_schritte,
                cw=self.ausparken_schritte_cw, ccw=self.ausparken_schritte_ccw)
            tabelle = schritte_aus_flach(roh)
        except ValueError as fehler:
            self.get_logger().error("Einpark-Test: Schrittliste unbrauchbar: %s" % fehler)
            self.state = 'DONE'
            return False
        self.ausp_schritte_std = spiegeln(tabelle, ccw)
        self.ausp_schritte = list(self.ausp_schritte_std)
        vorn = (cx - x) * tx + (cy - y) * ty
        bis_start = (self.park_start[0] - x) * tx + (self.park_start[1] - y) * ty
        if bis_start < 0.10:
            self.get_logger().error(
                "Einpark-Test: die Einpark-Startpose laege %.2f m %s ihm -- er muss "
                "VOR der Luecke stehen (Anfang der Startgeraden, Nase Richtung "
                "Luecke). Steht er %.2f m vor der Frontwand? test_richtung und "
                "test_bucht_front pruefen. Abbruch, faehrt nicht."
                % (abs(bis_start), hinter if bis_start < 0 else vor, vorn))
            self.park_start = None
            self.state = DONE
            self.publish_stop()
            return False
        self.get_logger().info(
            "Einpark-Test %s: steht %.2f m vor der Frontwand, %.3f m von der "
            "Aussenbande. Luecke angenommen %.3f m vor der Frontwand, %.3f m von "
            "der Aussenbande; Einpark-Startpose %.1f cm laengs, %.1f cm quer davon "
            "-> (%.3f, %.3f), Parklinie %.3f m. Einparkfolge: %s (%d Zuege)."
            % (richtung, vorn, (nx * x + ny * y) - dw, f_b, q_b, std_laengs * 100,
               std_quer * 100, self.park_start[0], self.park_start[1], self.park_q,
               herkunft, len(tabelle)))
        self._park_versatz_anwenden()
        return True

    def _erste_ecke_zuruecksetzen(self):
        """Liegt der Einlenkpunkt von Ecke 1 hinter ihm, gerade dorthin
        zuruecksetzen (siehe Parameter erste_ecke_zurueck). True = Zug laeuft."""
        if (not self.erste_ecke_zurueck or self.pose is None
                or not self.geometry_ready() or not self.ausp_richtung):
            return False
        x, y, th = self.pose
        idx = self.pick_first_corner(x, y, th)
        if idx is None:
            return False
        # Probehalber planen (derselbe Bogen wie gleich beim Losfahren, inkl.
        # Pylonen-Radius), danach den alten Zustand wiederherstellen: nach dem
        # Zuruecksetzen wird aus der neuen Lage frisch geplant.
        alt = (self.corner_idx, self.arc)
        self.corner_idx = idx
        try:
            ok = self.plan_arc(th)
            arc = self.arc
        finally:
            self.corner_idx, self.arc = alt
        if not ok or arc is None:
            return False
        tx, ty = arc['travel']
        tA = arc['T_A']
        to_TA = (tA[0] - x) * tx + (tA[1] - y) * ty
        if to_TA > -self.erste_ecke_zurueck_min:
            return False
        # Waere der Bogen an der Ist-Pose gut genug? (wie _bogen_an_pose_verankern)
        sg = arc['s']
        bx, by, lb = arc['LB']
        nenner = 1.0 - sg * (bx * -math.sin(th) + by * math.cos(th))
        abst = (bx * x + by * y) - lb
        if nenner >= 0.2 and abst > 0.0:
            r_a = abst / nenner
            verschub = max(0.0, (self.min_turn_radius - r_a) * nenner)
            ux, uy = arc['u_B']
            s_aus = (x * ux + y * uy) + max(r_a, self.min_turn_radius)
            w_aus = self._exit_wall_idx(idx)
            voraus = [(o['x'] * ux + o['y'] * uy) - s_aus
                      for o in (self.obstacles or []) if o['wall'] == w_aus]
            voraus = [v for v in voraus if v > 0.0]
            platz = (min(voraus) - self.obs_clear_before) if voraus else 2.0
            steig = verschub / platz if platz > 0.01 else float('inf')
            if verschub <= 0.02 or steig <= self.erste_ecke_zurueck_steigung:
                self.get_logger().info(
                    "Ecke 1: Einlenkpunkt %.2f m hinter ihm, Bogen an der Ist-Pose "
                    "kommt %.0f cm weiter aussen heraus -- bis zur naechsten Pylone "
                    "%.2f m, Steigung %.2f: kein Zuruecksetzen."
                    % (-to_TA, verschub * 100, platz, steig))
                return False
        weg = -to_TA + 0.02
        if weg > self.erste_ecke_zurueck_max:
            self.get_logger().warn(
                "Ecke 1: Einlenkpunkt %.2f m hinter ihm -- mehr als %.2f m, "
                "setzt nicht zurueck (Bogen wird an der Ist-Pose verankert)."
                % (-to_TA, self.erste_ecke_zurueck_max))
            return False
        # Rueckweg frei? Umriss entlang der Geraden gegen die bekannten Pylonen.
        for o in (self.obstacles or []):
            for k in range(11):
                sv = -weg * k / 10.0
                p = (x + sv * math.cos(th), y + sv * math.sin(th), th)
                luft = self._abstand_umriss(p, o['x'], o['y']) - BLOCK_HALB
                if luft < 0.03:
                    self.get_logger().warn(
                        "Ecke 1: Einlenkpunkt %.2f m hinter ihm, aber Pylone "
                        "#%d im Rueckweg (%.1f cm) -- setzt nicht zurueck."
                        % (-to_TA, o['id'], luft * 100))
                    return False
        befehl = -weg / max(self.einparken_zug_skala, 0.5)
        self.get_logger().info(
            "Ecke 1: Einlenkpunkt %.2f m hinter ihm -- setzt %.1f cm gerade "
            "zurueck (befohlen %.1f cm Radweg), statt mit dem kleinsten Radius "
            "zu verankern." % (-to_TA, weg * 100, befehl * 100))
        self.ausp_pos_prev = None
        # Die Ausparkfolge ist die Vorlage fuers Einparken (umgekehrt) -- der
        # Rueckwaertszug darf sie nicht ueberschreiben.
        self.ausp_schritte_vor_zurueck = list(self.ausp_schritte)
        self._ausparken_pid(self.ausparken_pid)
        self._park_zuege_starten(
            spiegeln([(0.0, befehl * 100.0)], self.ausp_richtung == 'CCW'), 'zurueck')
        return True

    def _erste_ecke_zurueck_fertig(self, x, y, theta):
        self.ausp_schritte = self.ausp_schritte_vor_zurueck
        self.ausp_pos_prev = None
        self._ausparken_pid(self.ausparken_pid_nachher)
        self.publish_stop()
        self.arc = None                  # aus der neuen Lage frisch planen
        self.corner_idx = None
        self.state = 'WAIT_INPUTS'
        self.get_logger().info(
            "Zurueckgesetzt: Pose (%.2f, %.2f), Kurs %+.1f grad. Weiter zum Rennen."
            % (x, y, math.degrees(theta)))

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

    def _scan_brems_kappe(self):
        """Tempo-Obergrenze vor dem Scan-Halt: v = sqrt(2 a Rest), Rest bis
        zum Ausloesepunkt (Soll + Nachlauf bei Ankunftstempo v_finish_min)."""
        if (self.scan_verzoegerung <= 0.0 or self.state != 'DRIVE'
                or not self.scan_pause or self.scan_done_this_straight
                or (self.corner_count // 4) >= self.scan_pause_laps
                or self.corner_count >= self.n_corners
                or self.pose is None or self.arc is None or self.corners is None):
            return None
        x, y, _ = self.pose
        corner = self.corners[self.corner_idx]
        tr = self.arc['travel']
        front_dist = (corner[0] - x) * tr[0] + (corner[1] - y) * tr[1]
        v0 = self.v_finish_min
        nachlauf = (v0 * self.scan_nachlauf_t + v0 * v0 / (2.0 * max(self.scan_brems_a, 0.1))
                    if self.scan_brems_a > 0.0 else self.scan_nachlauf)
        rest = front_dist - self.scan_front_dist - nachlauf
        return max(self.v_finish_min,
                   math.sqrt(2.0 * self.scan_verzoegerung * max(rest, 0.0)))

    def _v_kappe(self):
        """Obergrenze fuers Tempo in der aktuellen Phase, oder None."""
        kappe = None
        if not self._lok_ok() and self.state in ('DRIVE', 'TURN'):
            kappe = self.v_lok_unsicher
        vb = self._scan_brems_kappe()
        if vb is not None:
            kappe = vb if kappe is None else min(kappe, vb)
        # Steiler Ausweichpfad (Pylone spaet gesehen): langsamer, sonst
        # ueberschwingt er weit ueber die Pfadrichtung (parken_test_42: Pfad
        # 1,22 quer je laengs = 51 grad, gefahren bis 82 grad neben der Geraden).
        if (self.state == 'DRIVE' and self.obs_path
                and getattr(self, 'obs_max_slope', 0.0) > self.steiler_pfad_ab):
            kappe = (self.v_steiler_pfad if kappe is None
                     else min(kappe, self.v_steiler_pfad))
        if self.v_ziel > 0.0:
            letzte_kurve = (self.state == 'TURN'
                            and self.corner_count + 1 >= self.n_corners)
            if letzte_kurve or self._auf_zielgerade():
                kappe = self.v_ziel if kappe is None else min(kappe, self.v_ziel)
        return kappe

    def publish_cmd(self, v, omega):
        # Langsamer fahren, aber mit GLEICHER Kruemmung: omega = v * kappa --
        # nur v zu kappen hiesse enger zu lenken als geplant.
        kappe = self._v_kappe()
        if kappe is not None and abs(v) > kappe:
            k = kappe / abs(v)
            v, omega = v * k, omega * k
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
        self.cmd_hist.append((self.now_s(), float(omega)))

    def _pose_nach_totzeit(self, x, y, theta):
        """Pose, die der Wagen hat, wenn der JETZT berechnete Befehl wirkt.

        Die Befehle der letzten steer_dead_time Sekunden sind schon unterwegs:
        sie bestimmen, wie er in dieser Zeit giert. Aufintegriert (mit der
        gemessenen Lenkverstaerkung) ergibt das Kurs und Lage bei Wirkbeginn."""
        # Lenkgesetz mit der geglaetteten Pose (Aufrufer uebergeben self.pose;
        # beide stammen aus derselben /ekf/odom-Nachricht)
        if self.pose_lenk is not None and self.pose is not None and (x, y, theta) == self.pose:
            x, y, theta = self.pose_lenk
        T = self.steer_dead_time
        if T <= 0.0 or not self.cmd_hist:
            return x, y, theta
        jetzt = self.now_s()
        t0 = jetzt - T
        dth = 0.0
        eintraege = [e for e in self.cmd_hist if e[0] >= t0 - 0.2]
        for i, (t, om) in enumerate(eintraege):
            ende = eintraege[i + 1][0] if i + 1 < len(eintraege) else jetzt
            a, b = max(t, t0), min(ende, jetzt)
            if b > a:
                dth += om * (b - a)
        dth *= self.steer_gain_pred
        v = max(abs(self.v_ist), 0.0)
        th_mid = theta + 0.5 * dth
        return (x + v * T * math.cos(th_mid),
                y + v * T * math.sin(th_mid),
                theta + dth)

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
        if self.race_direction in ('CW', 'CCW'):
            # Aus der Kartengeometrie, NICHT aus dem Ist-Kurs: nach einem steilen
            # Ausweichschwenk stand er bei -172 statt -90 grad, die Neuplanung
            # vertauschte Ein- und Ausfahrtswand und plante die falsche Ecke
            # (parken_test_42: Scan-Halt bei "Frontwand -0.51 m", dann NOTSTOP).
            A = self._inward(self.walls[self._entry_wall_idx(idx)], cx, cy)
            B = self._inward(self.walls[self._exit_wall_idx(idx)], cx, cy)
            tx, ty = -B[0], -B[1]          # Fahrtrichtung = gegen die Normale der Frontwand
            theta = math.atan2(ty, tx)
        else:
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

        R = self._radius_fuer_pylonen(idx, A, B, o_in, o_out, R, theta)

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
        self.arc = dict(C=C, s=s, R=R, o_in=o_in, o_out=o_out, T_A=T_A, T_B=T_B, a0=a0,
                        travel=travel, LA=LA, LB=LB, u_B=u_B, theta_target=theta_target)
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

    def _bogen_posen(self, A, B, o_in, o_out, R, theta):
        """Hinterachs-Posen entlang Einfahrt (letzte 0,15 m), Bogen und Ausfahrt
        (erste 0,25 m) fuer den Radius R -- dieselbe Geometrie wie plan_arc."""
        LA = (A[0], A[1], A[2] + o_in)
        LB = (B[0], B[1], B[2] + o_out)
        P = line_intersect(LA, LB)
        if P is None:
            return None, None
        s = float(self.dir_step())
        C = (P[0] + R * (A[0] + B[0]), P[1] + R * (A[1] + B[1]))
        TA = (C[0] - R * A[0], C[1] - R * A[1])
        TB = (C[0] - R * B[0], C[1] - R * B[1])
        a0 = math.atan2(TA[1] - C[1], TA[0] - C[0])
        dphi = wrap(math.atan2(TB[1] - C[1], TB[0] - C[0]) - a0)
        th_ein = a0 + s * math.pi / 2.0
        posen = []
        for k in range(4):
            d = -0.15 + 0.05 * k
            posen.append((TA[0] + d * math.cos(th_ein), TA[1] + d * math.sin(th_ein), th_ein))
        n = max(4, int(abs(dphi) / math.radians(4.0)))
        for k in range(n + 1):
            phi = a0 + dphi * k / n
            posen.append((C[0] + R * math.cos(phi), C[1] + R * math.sin(phi),
                          phi + s * math.pi / 2.0))
        th_aus = a0 + dphi + s * math.pi / 2.0
        for k in range(1, 6):
            d = 0.05 * k
            posen.append((TB[0] + d * math.cos(th_aus), TB[1] + d * math.sin(th_aus), th_aus))
        return posen, C

    def _arc_pylonen_abstand(self, a):
        """Kleinster Abstand Fahrzeugkante -> Pylonenkante fuer einen fertigen
        Bogen (plan_arc oder verankert): Einfahrt 0,15 m, Bogen, Ausfahrt 0,25 m.
        (None, None) ohne Pylonen in der Naehe."""
        if not self.obstacles or a is None:
            return None, None
        C, R, sg = a['C'], a['R'], a['s']
        TA, TB = a['T_A'], a['T_B']
        pyl = [o for o in self.obstacles
               if math.hypot(o['x'] - C[0], o['y'] - C[1]) < R + 0.6]
        if not pyl:
            return None, None
        a0 = math.atan2(TA[1] - C[1], TA[0] - C[0])
        dphi = wrap(math.atan2(TB[1] - C[1], TB[0] - C[0]) - a0)
        if sg * dphi < 0.0:
            dphi += sg * 2.0 * math.pi
        th_e = a0 + sg * math.pi / 2.0
        th_a = a0 + dphi + sg * math.pi / 2.0
        posen = [(TA[0] + d * math.cos(th_e), TA[1] + d * math.sin(th_e), th_e)
                 for d in (-0.15, -0.10, -0.05)]
        n = max(4, int(abs(dphi) / math.radians(4.0)))
        for k in range(n + 1):
            phi = a0 + dphi * k / n
            posen.append((C[0] + R * math.cos(phi), C[1] + R * math.sin(phi),
                          phi + sg * math.pi / 2.0))
        posen += [(TB[0] + d * math.cos(th_a), TB[1] + d * math.sin(th_a), th_a)
                  for d in (0.05, 0.10, 0.15, 0.20, 0.25)]
        best, wer = float('inf'), None
        for o in pyl:
            for p in posen:
                dd = self._abstand_umriss(p, o['x'], o['y']) - BLOCK_HALB
                if dd < best:
                    best, wer = dd, o
        return best, wer

    @staticmethod
    def _abstand_umriss(pose, px, py):
        """Abstand Pylonenmitte -> Fahrzeugrechteck (Hinterachse = Pose)."""
        x, y, th = pose
        c, sn = math.cos(th), math.sin(th)
        dx, dy = px - x, py - y
        lx, ly = c * dx + sn * dy, -sn * dx + c * dy
        hb = 0.5 * FZ_BREITE
        ax = max(FZ_HECK - lx, 0.0, lx - FZ_NASE)
        ay = max(-hb - ly, 0.0, ly - hb)
        return math.hypot(ax, ay)

    def _bogen_pylonen_abstand(self, A, B, o_in, o_out, R, theta, pylonen):
        """Kleinster Abstand Fahrzeugkante -> Pylonenkante ueber den Bogen, und
        die Pylone dazu. (None, None) ohne Geometrie."""
        posen, _C = self._bogen_posen(A, B, o_in, o_out, R, theta)
        if posen is None:
            return None, None
        best, wer = float('inf'), None
        for o in pylonen:
            for p in posen:
                d = self._abstand_umriss(p, o['x'], o['y']) - BLOCK_HALB
                if d < best:
                    best, wer = d, o
        return best, wer

    def _anlauf_reicht(self, A, B, o_in, o_out, R, theta):
        """Wie die Schrumpfschleife in plan_arc: ist der Einlenkpunkt fuer
        diesen Radius von der Ist-Pose aus noch sauber erreichbar?"""
        if self.pose is None:
            return True
        LA = (A[0], A[1], A[2] + o_in)
        LB = (B[0], B[1], B[2] + o_out)
        P = line_intersect(LA, LB)
        if P is None:
            return False
        C = (P[0] + R * (A[0] + B[0]), P[1] + R * (A[1] + B[1]))
        TA = (C[0] - R * A[0], C[1] - R * A[1])
        px, py, _ = self.pose
        tx, ty = math.cos(theta), math.sin(theta)
        room = (TA[0] - px) * tx + (TA[1] - py) * ty
        lat_err = abs((A[0] * px + A[1] * py) - LA[2])
        return room > 0.01 and lat_err / room <= self.max_settle_slope

    def _radius_fuer_pylonen(self, idx, A, B, o_in, o_out, R, theta):
        """Radius so waehlen, dass der Bogen an Pylonen am Kurvenein- und
        -ausgang mit bogen_pylonen_abstand vorbeikommt.

        o_in/o_out legen nur fest, auf welcher Seite er VOR und NACH der Kurve
        faehrt. Steht eine Pylone kurz hinter der Ecke, liegt sie mitten im
        Bogen: in CCW eine rote (rechts = aussen vorbei) innerhalb des Kreises,
        dann hilft ein KLEINERER Radius; in CW eine rote (rechts = innen)
        ausserhalb, dann ein GROESSERER. Statt der Regel wird gerechnet:
        Kandidaten von min_turn_radius bis bogen_pylonen_r_max, der naechste am
        geplanten R mit genug Abstand gewinnt. Groessere Radien nur, wenn der
        Einlenkpunkt noch sauber erreichbar ist."""
        if not self.obstacles or self.bogen_pylonen_abstand <= 0.0:
            return R
        posen, C = self._bogen_posen(A, B, o_in, o_out, R, theta)
        if posen is None:
            return R
        pylonen = [o for o in self.obstacles
                   if math.hypot(o['x'] - C[0], o['y'] - C[1]) < R + 0.6]
        if not pylonen:
            return R
        soll = self.bogen_pylonen_abstand
        ab0, wer0 = self._bogen_pylonen_abstand(A, B, o_in, o_out, R, theta, pylonen)
        if ab0 is None or ab0 >= soll:
            return R
        kand = {round(R, 3)}
        r = self.min_turn_radius
        while r <= self.bogen_pylonen_r_max + 1e-6:
            kand.add(round(r, 3))
            r += 0.05
        beste = (ab0, R)
        for r in sorted(kand, key=lambda v: (abs(v - R), v)):
            if r > R + 1e-6 and not self._anlauf_reicht(A, B, o_in, o_out, r, theta):
                continue
            ab, _w = self._bogen_pylonen_abstand(A, B, o_in, o_out, r, theta, pylonen)
            if ab is None:
                continue
            if ab >= soll:
                beste = (ab, r)
                break
            if ab > beste[0]:
                beste = (ab, r)
        ab, r = beste
        farbe = {OBST_ROT: 'rot', OBST_GRUEN: 'gruen'}.get(wer0['color'], '?')
        if ab >= soll:
            self.get_logger().warn(
                f"Ecke {self.corner_count + 1}: Pylone #{wer0['id']} ({farbe}) im Bogen -- Radius "
                f"{R:.2f} -> {r:.2f} m, Abstand {ab0*100:.1f} -> {ab*100:.1f} cm.")
        else:
            self.get_logger().error(
                f"Ecke {self.corner_count + 1}: Pylone #{wer0['id']} ({farbe}) im Bogen, kein Radius "
                f"{self.min_turn_radius:.2f}-{self.bogen_pylonen_r_max:.2f} m haelt "
                f"{soll*100:.0f} cm -- nehme {r:.2f} m mit {ab*100:.1f} cm "
                f"(geplant {R:.2f} m: {ab0*100:.1f} cm).")
        return r

    def _bogen_an_pose_verankern(self, x, y, theta, r_max=None):
        """Bogen so legen, dass er HIER tangential zum Ist-Kurs beginnt und
        tangential auf der Austrittslinie endet.

        plan_arc kennt kein 'hier': T_A und C kommen aus der Kastengeometrie.
        Steht der Wagen schon hinter T_A, liegt der Kreis hinter ihm, und die
        Schrumpfschleife bricht dann sofort ab (room <= 0). Hier dagegen:
            C = P + R*s*n_links,   Abstand(C, LB) = R
            ->  R = (B.P - LB) / (1 - s*(B.n_links))
        Genau an T_A ergibt das den alten Radius, delta dahinter R - delta.
        Unter min_turn_radius: mit dem kleinsten Radius fahren und dafuer
        frueher, naeher an der Aussenbande, auf die naechste Gerade kommen.
        Rueckgabe (fahrbar, Beschreibung).
        """
        a = self.arc
        s = a['s']
        bx, by, lb = a['LB']
        nlx, nly = -math.sin(theta), math.cos(theta)
        nenner = 1.0 - s * (bx * nlx + by * nly)
        abst = (bx * x + by * y) - lb            # > 0: Austrittslinie noch voraus
        if nenner < 0.2 or abst <= 0.0:
            return False, ("Austrittslinie nicht mehr erreichbar (Abstand %.2f m, "
                           "Kurs passt nicht)" % abst)
        R = abst / nenner
        if r_max is not None and R > r_max:
            return False, ("verankerter Radius %.2f m > %.2f -- Kurs zeigt schon "
                           "stark in die Kurve" % (R, r_max))
        o_out = a.get('o_out')
        verschub = 0.0
        if R < self.min_turn_radius:
            verschub = (self.min_turn_radius - R) * nenner
            if o_out is not None and o_out - verschub < self.turn_anchor_min_out:
                return False, ("selbst mit Radius %.2f m kaeme er %.2f m vor der "
                               "Aussenbande heraus (Minimum %.2f)"
                               % (self.min_turn_radius, o_out - verschub,
                                  self.turn_anchor_min_out))
            R = self.min_turn_radius
        C = (x + R * s * nlx, y + R * s * nly)
        lb_neu = lb - verschub
        a.update(C=C, R=R, T_A=(x, y),
                 T_B=(C[0] - R * bx, C[1] - R * by),
                 a0=math.atan2(y - C[1], x - C[0]),
                 LB=(bx, by, lb_neu))
        if o_out is not None:
            a['o_out'] = o_out - verschub
        if verschub > 0.0:
            return True, ("kleinster Radius %.2f m, Austritt bei %.2f statt %.2f m"
                          % (R, o_out - verschub, o_out))
        return True, "Radius %.2f m, Austritt wie geplant" % R

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
        if self.state == 'PARK_HALT':
            self._park_halt(x, y, theta)
            return
        if self.state == 'PARK_NACHMESSEN':
            self._park_nachmessen(x, y, theta)
            return
        if self.state == 'PARK_RUECK':
            self._park_rueck(x, y, theta)
            return

        if self.state == 'WAIT_INPUTS':
            if not self.inputs_ready():
                return
            if (self.einparken_test and self.park_start is None
                    and not self._einpark_test_vorbereiten()):
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
            # Nur kurz Stopp senden, dann still sein: ein vergessener Regler
            # im Zustand DONE schickte sonst dauerhaft /cmd_vel = 0 (30 Hz),
            # und die Bruecke stellte darueber die Lenkung des NAECHSTEN
            # Laufs auf geradeaus (parken_test_36/37: Ausparken ohne Lenkung).
            jetzt = self.now_s()
            if getattr(self, '_done_seit', None) is None:
                self._done_seit = jetzt
            if jetzt - self._done_seit < 1.0:
                self.publish_stop()
            return
        self._done_seit = None

        # --- odom-stale handling: hold last cmd through short gaps, stop on long ---
        if self.odom_is_stale():
            if self.state in ('TURN', 'DRIVE'):
                self.republish_last()   # bridge past the gap; bridge watchdog is the backstop
            else:
                self.publish_stop()
            return

        if (self.state in ('DRIVE', 'TURN') and self.lok_state == 'lost'
                and self.lok_lost_stopp_s > 0.0 and self.lok_lost_t0 is not None
                and self.now_s() - self.lok_lost_t0 > self.lok_lost_stopp_s):
            self.state = 'DONE'
            self.publish_stop()
            self.get_logger().error(
                "NOTSTOP: Lokalisierung seit %.1f s 'lost' -- ueber ~0,35 m Fehler "
                "faengt sie sich nicht mehr, Weiterfahren waere Blindflug."
                % (self.now_s() - self.lok_lost_t0))
            return
        if self.state == 'PARK_FAHRT':
            self._park_fahrt(x, y, theta)
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
        self._warnen_wenn_in_der_luecke()
        if not self.geometry_ready():
            self.get_logger().info(
                "Start ohne Kartengeometrie: fahre mittig geradeaus bis Richtung erkannt.")
            return
        if self.corner_idx is None:
            self.corner_idx = self.pick_first_corner(x, y, theta)
            if self.corner_idx is None:
                self.get_logger().warn("Keine Ecke voraus gefunden -- nehme idx 0.")
                self.corner_idx = 0
        if self.erste_ecke_pruefen and self.corner_count == 0:
            self._erste_ecke_nach_ausparken(x, y, theta)
        if self.arc is None:
            self.plan_arc(theta)
        self.publish_lap_state()

    def _erste_ecke_nach_ausparken(self, x, y, theta):
        """Einmalig nach dem Ausparken: steht er schon vor Ecke 1?

        Dann hat er gerade im Stand gescannt -- genau dort, wo sonst der
        Scan-Stopp waere. Ein zweiter Halt 27 cm weiter bringt nichts, und der
        Weg dazwischen reichte nicht, um den Querversatz zur Spurmitte
        abzubauen (CCW-Test: 31 cm auf 45 cm Anlauf, Ecke 1 unruhig).
        """
        self.erste_ecke_pruefen = False
        c = self.corners[self.corner_idx]
        vorn = (c[0] - x) * math.cos(theta) + (c[1] - y) * math.sin(theta)
        # Im Halt schon entschieden? Dann dabei bleiben (er hat sich seitdem
        # nicht bewegt). Sonst (Geometrie kam erst beim Fahren) jetzt pruefen.
        hier = (self.ausp_scan_hier if self.ausp_scan_hier is not None
                else vorn < self.ausparken_scan_ersetzt_bis)
        if not hier:
            return
        self.scan_done_this_straight = True
        w = self._entry_wall_idx(self.corner_idx)
        nx, ny, dw = self.walls[w]
        q = (nx * x + ny * y) - dw
        breite = (self.lane_width[w] if self.lane_width is not None
                  and w < len(self.lane_width) else 1.0)
        if 0.15 <= q <= breite - 0.10:
            self.erste_ecke_idx = self.corner_idx
            self.erste_ecke_q = q
        self.get_logger().info(
            "Nach dem Ausparken %.2f m vor Ecke 1: Halt am Ausparkende ersetzt "
            "den Scan-Stopp%s."
            % (vorn, ", Ecke 1 aus der jetzigen Lage (q=%.2f) geplant" % q
               if self.erste_ecke_q is not None else ""))

    def _warnen_wenn_in_der_luecke(self):
        """Steht er beim Losfahren noch in der Parkluecke?

        Die bestehende Plausibilitaetspruefung (start_lane_min/max) sieht nur
        die SUMME der beiden Wandabstaende. In der Luecke sind das 0.15 + 0.83
        = 0.98 -- mitten im erlaubten Band. Die Aufteilung verraet es: in einer
        Spur steht er ungefaehr mittig, in der Luecke klebt er an der Wand.
        """
        if self.ausparken or not self.wall_dist:
            return
        links, rechts = self.wall_dist
        if not (math.isfinite(links) and math.isfinite(rechts)):
            return
        if min(links, rechts) >= self.start_wand_warn:
            return
        self.get_logger().warn(
            "Eine Seite ist nur %.2f m entfernt, die andere %.2f m -- in einer "
            "Spur stuende er mittig. Sieht nach der Parkluecke aus, und "
            "ausparken ist AUS. Falls ja: mit -p ausparken:=true starten, "
            "sonst faehrt er in die Magenta-Wand."
            % (min(links, rechts), max(links, rechts)))

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
        # Lenkgesetz mit der Pose bei Wirkbeginn (Totzeit), die Ausloeser
        # (T_A, Haltepunkt) weiter mit der echten Pose.
        px_, py_, pth_ = self._pose_nach_totzeit(x, y, theta)
        omega = None
        if self.obs_path:
            omega = self._stanley_follow_path(px_, py_, pth_, self.obs_path)
        if omega is None:
            omega = self._stanley_steer(px_, py_, pth_, self.arc['LA'], tr)

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
                    f"front_dist={front_dist:+.2f} (Ziel {self._ziel_abstand():.2f}) om={omega:+.2f}",
                    throttle_duration_sec=0.2)
            # remaining distance to the STOP point, compensated for the reaction
            # lead (a tick + motor/vehicle latency): stop when the robot will be
            # AT the target after it coasts through the lead, not when it first
            # crosses the line -- otherwise it overshoots, worse at higher speed.
            if self._park_aktiv() and self.einparken_halt_s <= 0.0:
                # Kein Pflichthalt: durchfahren und ohne Anhalten uebergehen.
                if front_dist <= self.seiten_frei_ab:
                    self._park_uebergang(x, y, theta)
                    return
                self.publish_cmd(min(self.v_drive, max(self.v_park_anfahrt,
                                                       self.v_finish_min)), omega)
                return
            v_now = max(abs(self.v_ist), 0.0)
            lead = v_now * self.finish_lead_time
            remain = front_dist - self._ziel_abstand() - lead

            if remain <= self.finish_tol:
                self.publish_stop()
                self.get_logger().info(
                    f"ZIEL ({self.corner_count} Ecken, {front_dist:.2f} m vor Frontwand, "
                    f"v={v_now:.2f}). STOP.")
                if self._park_aktiv():
                    self.state = 'PARK_HALT'
                    self.park_t0 = self.now_s()
                    self.park_mittel = []
                    self.get_logger().info(
                        "Einparken: %.1f s Pflichtstillstand, dann einparken."
                        % self.einparken_halt_s)
                else:
                    self.state = 'DONE'
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
            if self.scan_brems_a > 0.0:
                v_n = abs(self.v_ist)
                nachlauf = (v_n * self.scan_nachlauf_t
                            + v_n * v_n / (2.0 * self.scan_brems_a))
            else:
                nachlauf = self.scan_nachlauf
            if front_dist <= self.scan_front_dist + nachlauf:
                self.scan_done_this_straight = True
                self.scan_pause_t0 = self.now_s()
                self.state = 'SCAN_PAUSE'
                self.publish_stop()
                self.get_logger().info(
                    f"SCAN-HALT Start (Runde {self.corner_count // 4 + 1}): "
                    f"{self.scan_pause_s:.1f}s, Frontwand {front_dist:.2f} m "
                    f"(Soll {self.scan_front_dist:.2f}, Nachlauf {nachlauf*100:.0f} cm "
                    f"bei {abs(self.v_ist):.2f} m/s), to_TA {to_TA:+.2f} m.")
                return

        # --- turn-in when pose crosses T_A (never before the scan pause) ---
        vorhalt = 0.0
        if self.einlenk_vorhalt and self.turn_praediktion:
            vorhalt = max(self.v_ist, 0.0) * self.steer_dead_time
        if to_TA - vorhalt <= 0.0 and not scan_pending:
            # ab hier mit der Pose bei Wirkbeginn: an ihr beginnt der Bogen
            to_TA -= vorhalt
            vx_, vy_, vth_ = (self._pose_nach_totzeit(x, y, theta) if vorhalt > 0.0
                              else (x, y, theta))
            if lateral > 0.6:
                self.state = 'DONE'
                self.publish_stop()
                self.get_logger().error(
                    f"NOTSTOP: Einlenkpunkt seitlich verfehlt (lat={lateral:.2f}). "
                    f"Falsche Ecke? idx {self.corner_idx}.")
                return

            # NIE ueber T_A hinaus warten. Der Bogen ist an T_A verankert; jeder
            # Zentimeter dahinter macht den Kreis unerreichbar, und Neuplanen
            # hilft nicht (der neue Bogen beginnt dann hinter dem Wagen). Genau
            # das trug die erste Kurve in Lauf 21 geradeaus in die Wand -- und
            # im CCW-Einparktest bis auf 4 cm an die Frontwand. Beruhigt wird
            # VOR T_A durch Langsamerwerden (siehe unten).
            if -to_TA > self.turn_in_past_max:
                # nur noch Plausibilitaet -- so weit hinter T_A stimmt etwas
                # Grundsaetzliches nicht (falsche Ecke?)
                self.state = 'DONE'
                self.publish_stop()
                self.get_logger().error(
                    f"NOTSTOP: Einlenkpunkt {-to_TA:.2f} m hinter uns "
                    f"(> {self.turn_in_past_max:.2f}). Falsche Ecke?")
                return
            if -to_TA > 0.02:
                # hinter T_A: den Bogen an der Ist-Pose verankern, statt einem
                # Kreis nachzufahren, der hinter dem Wagen liegt
                ok, text = self._bogen_an_pose_verankern(vx_, vy_, vth_)
                if not ok:
                    self.state = 'DONE'
                    self.publish_stop()
                    self.get_logger().error(
                        f"NOTSTOP: {-to_TA:.2f} m hinter T_A, Bogen nicht fahrbar: {text}.")
                    return
                self.get_logger().warn(
                    f"Einlenkpunkt {-to_TA:.2f} m hinter uns -- Bogen an der "
                    f"Ist-Pose verankert: {text}.")
            elif self.turn_anchor_puenktlich:
                # Puenktlich, aber gestoert eingefahren (Kurs oder quer neben dem
                # Kreis): den Kreis so legen, dass er HIER tangential zum Ist-
                # Kurs beginnt. Der Standardbogen verlangt sofort Kreis und
                # Tangente -- bei 20 grad Kursfehler springt der Befehl in die
                # Begrenzung, und der Bogenfehler waechst bis 14 cm (Ecke idx1,
                # 61 Prozent der Kurve in der Begrenzung). Verankert simuliert:
                # 3,5-4,2 cm bei jeder Einfahrt. Scheitert es, bleibt der
                # Standardbogen -- hier kein Nothalt.
                kurs_fehler = abs(wrap(vth_ - math.atan2(tr[1], tr[0])))
                if (kurs_fehler > self.turn_anchor_kurs
                        or lateral > self.turn_anchor_quer):
                    r0 = self.arc['R']
                    sicherung = dict(self.arc)
                    ok, text = self._bogen_an_pose_verankern(
                        vx_, vy_, vth_, r_max=1.5 * r0)
                    if ok:
                        # Pylonen: der verankerte Radius ist nicht mehr frei
                        # gewaehlt. Kommt er einer Pylone naeher als der
                        # geplante Bogen und unter den Mindestabstand, lieber
                        # den geplanten (pylonengeprueften) Standardbogen.
                        ab_neu, wer = self._arc_pylonen_abstand(self.arc)
                        ab_alt, _w = self._arc_pylonen_abstand(sicherung)
                        if (ab_neu is not None and ab_neu < self.bogen_pylonen_abstand
                                and (ab_alt is None or ab_neu < ab_alt)):
                            ok, text = False, (
                                "Pylone #%d nur %.1f cm am verankerten Bogen, geplant %.1f cm"
                                % (wer['id'], ab_neu * 100,
                                   (ab_alt if ab_alt is not None else float('nan')) * 100))
                    if ok:
                        self.get_logger().info(
                            f"Einfahrt gestoert (Kurs {math.degrees(kurs_fehler):.1f} "
                            f"grad, quer {lateral:.3f} m) -- Bogen verankert: {text}.")
                    else:
                        self.arc = sicherung
                        self.get_logger().info(
                            f"Einfahrt gestoert, Verankerung verworfen ({text}) "
                            f"-- Standardbogen.")
            om_last = abs(self.last_cmd[1])
            if lateral > self.turn_in_lat_gate or om_last > self.turn_in_om_gate:
                self.get_logger().warn(
                    f"Einlenken unruhig: lat={lateral:.3f} om={om_last:.2f} "
                    f"-- trotzdem an T_A eingelenkt (Geometrie darf nicht weglaufen).")
            self.state = 'TURN'
            if lateral > self.turn_in_lat_warn:
                self.get_logger().warn(
                    f"Einlenken mit Querfehler {lateral:.2f} m (> {self.turn_in_lat_warn:.2f}) "
                    f"-- dieser Fehler wandert durch die ganze Kurve.")
            self.get_logger().info(
                f"TURN: Einlenken bei ({x:.2f},{y:.2f}, {math.degrees(theta):.1f}).")
            return

        v = self._speed_profile(to_TA, dsc)
        # Vor der Ecke beruhigen: laeuft er unruhig auf T_A zu, langsamer werden.
        # Stanley bekommt so mehr Zeit pro Meter, ohne die Geometrie zu verschieben.
        if (to_TA < self.turn_in_settle_window
                and (lateral > self.turn_in_lat_gate
                     or abs(self.last_cmd[1]) > self.turn_in_om_gate)):
            v = min(v, self.v_settle)
            self.get_logger().info(
                f"Beruhigen vor Ecke: to_TA={to_TA:.2f} lat={lateral:.3f} "
                f"om={abs(self.last_cmd[1]):.2f} -> v={v:.2f}.",
                throttle_duration_sec=0.5)
        if self.obs_path:
            # safety before speed on obstacle straights; steeper swap -> slower
            v_cap = (self.v_obstacle_steep
                     if self.obs_max_slope >= self.obs_slope_slow
                     else self.v_obstacle)
            v = min(v, v_cap)
        self.publish_cmd(v, omega)

    def _turn(self, x, y, theta):
        C = self.arc['C']; s = self.arc['s']; R = self.arc['R']
        # Mit der Pose rechnen, die der Wagen hat, wenn der Befehl wirkt (260 ms
        # Totzeit) -- wie auf der Geraden. Auch das Kurvenende haengt daran: am
        # ist-Kurs beendet, drehte er waehrend der Totzeit noch 13-34 grad weiter.
        if self.turn_praediktion:
            xp, yp, thp = self._pose_nach_totzeit(x, y, theta)
        else:
            xp, yp, thp = x, y, theta
        rx, ry = xp - C[0], yp - C[1]
        dist = math.hypot(rx, ry) or 1e-6
        r_hat = (rx / dist, ry / dist)
        e_ct = dist - R
        t_hat = (-s * r_hat[1], s * r_hat[0])
        e_th = wrap(math.atan2(t_hat[1], t_hat[0]) - thp)
        theta_err = wrap(self.arc['theta_target'] - thp)

        blend = max(0.0, min(1.0, abs(theta_err) / self.ff_blend)) if self.ff_blend > 1e-6 else 1.0
        if self.turn_kruemmung:
            # Erst die gewuenschte KRUEMMUNG, dann omega mit genau der
            # Geschwindigkeit, durch die die Bruecke wieder teilt
            # (delta = atan(L*omega/v_ist), v_ist >= 0,05). So hebt sich v heraus
            # -- wie bei Stanley auf der Geraden. Vorher ging beim Anfahren aus
            # dem Stand v_turn in die Vorsteuerung: omega 0,85 / 0,05 -> 58 grad,
            # Volleinschlag in Ecke 1 und 3. Bei 0,35 m/s kommt dasselbe heraus
            # wie vorher (Korrekturverstaerkungen sind auf v_turn bezogen).
            v_b = max(abs(self.v_ist), 0.05)
            kappa = (s * blend / R
                     + (s * self.k_ct * e_ct + self.k_th * e_th) / max(self.v_turn, 0.05))
            omega = v_b * kappa
        else:
            v_meas = abs(self.v_ist) if abs(self.v_ist) > 0.05 else self.v_turn
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
            if not self.obs_path:
                self._rueckfuehr_pfad()       # sanft aus der Kurve auf die Spurlinie
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
        if self.pfad_vorsteuerung <= 0.0:
            return self._stanley_steer(x, y, theta, (nx, ny, d), (ux, uy))
        # Stufenlose Tangente: an den Pfadpunkten der Mittelwert der beiden
        # angrenzenden Segmente, dazwischen linear. Sonst springt e_theta an
        # jedem Punkt (alle 5 cm) um den Knick -- beim Spurwechsel mehrere grad.
        t = ((x - ax) * dx + (y - ay) * dy) / (L * L)
        t = max(0.0, min(1.0, t))
        h = math.atan2(uy, ux)
        h_vor = self._pfad_richtung(path_xy, best_i - 1, h)
        h_nach = self._pfad_richtung(path_xy, best_i + 1, h)
        h_a = h + 0.5 * wrap(h_vor - h)       # am Punkt best_i
        h_b = h + 0.5 * wrap(h_nach - h)      # am Punkt best_i + 1
        h_t = h_a + t * wrap(h_b - h_a)
        # Kruemmung an den beiden Punkten, dazwischen linear
        k_a = self._pfad_kruemmung(path_xy, best_i)
        k_b = self._pfad_kruemmung(path_xy, best_i + 1)
        kappa = k_a + t * (k_b - k_a)
        delta_ff = self.pfad_vorsteuerung * math.atan(self.wheelbase * kappa)
        return self._stanley_steer(x, y, theta, (nx, ny, d),
                                   (math.cos(h_t), math.sin(h_t)), delta_ff=delta_ff)

    @staticmethod
    def _pfad_richtung(path_xy, i, sonst):
        """Richtung von Segment i (Punkt i -> i+1); ausserhalb: sonst."""
        if i < 0 or i + 1 >= len(path_xy):
            return sonst
        (ax, ay), (bx, by) = path_xy[i], path_xy[i + 1]
        if abs(bx - ax) + abs(by - ay) < 1e-9:
            return sonst
        return math.atan2(by - ay, bx - ax)

    @staticmethod
    def _pfad_kruemmung(path_xy, i):
        """Kruemmung am Pfadpunkt i (Richtungsaenderung / halbe Segmentlaengen),
        + = Linkskurve. 0 an den Enden."""
        if i <= 0 or i + 1 >= len(path_xy):
            return 0.0
        (ax, ay), (bx, by), (cx, cy) = path_xy[i - 1], path_xy[i], path_xy[i + 1]
        l1 = math.hypot(bx - ax, by - ay)
        l2 = math.hypot(cx - bx, cy - by)
        if l1 < 1e-6 or l2 < 1e-6:
            return 0.0
        dh = wrap(math.atan2(cy - by, cx - bx) - math.atan2(by - ay, bx - ax))
        return dh / (0.5 * (l1 + l2))

    def _stanley_steer(self, x, y, theta, target_line, u_dir, delta_ff=0.0):
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

        delta = (k_h_eff * e_theta + math.atan2(self.k_stanley * e_ct, v_gain) + self.ct_integral
                 + delta_ff)
        delta = max(-self.max_steer, min(self.max_steer, delta))
        omega = v * math.tan(delta) / self.wheelbase

        # debug publish for Foxglove
        #self.pub_e_ct.publish(Float64(data=float(e_ct)))
        self.pub_e_th.publish(Float64(data=float(math.degrees(e_theta))))
        self.pub_delta.publish(Float64(data=float(math.degrees(delta))))
        self.pub_k_h.publish(Float64(data=float(k_h_eff)))
        return omega


def _einpark_test_scan_args(argv):
    """Einpark-Test: der scan_processor muss statt aus der Bucht von der
    Startgeraden starten. Aus der Kommandozeile gelesen, weil der Neustart vor
    dem Anlegen des Knotens (also vor den Parametern) passiert."""
    werte = {}
    for a in argv:
        if ':=' in a:
            k, v = a.split(':=', 1)
            werte[k.strip()] = v.strip()
    if werte.get('einparken_test', '').lower() not in ('true', '1', '1.0'):
        return ''
    richtung = werte.get('test_richtung', 'CCW').upper()
    if richtung not in ('CW', 'CCW'):
        richtung = 'CCW'
    args = '-p start_from_bay:=false -p start_gerade:=%s' % richtung
    for k in ('test_bucht_front', 'test_bucht_q'):
        if k in werte:
            try:
                args += ' -p %s:=%.4f' % (k, float(werte[k]))
            except ValueError:
                pass
    return args


def _zweiter_regler_laeuft(warte_s=1.5):
    """Laeuft schon ein round1_controller? Dann faehrt dieser hier nicht los.
    Zwei Regler senden beide /cmd_vel und Lenkung; der alte meldet dazu seine
    Runden (lap_state) an den neuen scan_processor, der die Hinderniskarte
    dann sofort einfriert (parken_test_36/37)."""
    pruefer = rclpy.create_node('round1_controller_pruefer')
    try:
        ende = time.monotonic() + warte_s
        n = 0
        while time.monotonic() < ende:
            rclpy.spin_once(pruefer, timeout_sec=0.1)
            n = pruefer.count_publishers('/round1_controller/lap_state')
            if n > 0:
                break
        return n
    finally:
        pruefer.destroy_node()


def main(args=None):
    rclpy.init(args=args)
    n = _zweiter_regler_laeuft()
    if n > 0:
        rclpy.logging.get_logger('round1_controller').fatal(
            'Es laeuft schon ein round1_controller (%d Sender auf '
            '/round1_controller/lap_state) -- den erst beenden (Strg+C in seinem '
            'Fenster), sonst kaempfen zwei Regler um Lenkung und Motor. '
            'Starte NICHT.' % n)
        rclpy.try_shutdown()
        sys.exit(1)
    # EKF und scan_processor frisch starten, bevor die eigenen gelatchten Abos
    # entstehen -- die Karte haengt an der Startpose (ekf/schaetzung_neustart.py).
    neu_starten('round1_controller', scan_args=_einpark_test_scan_args(sys.argv))
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