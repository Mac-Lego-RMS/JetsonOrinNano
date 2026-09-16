#!/usr/bin/env python3
"""Ordnet jedem Lidar-Punkt den Pixel bzw. die Farbe der 360-Grad-Kamera zu.

Fuer jeden Scan wird jeder gueltige Messpunkt ueber das Fisheye-Modell ins Bild
projiziert, dort die Farbe ausgelesen und als rot/gruen/magenta/schwarz
klassifiziert. Ergebnis geht raus als

  * CSV   (Hauptausgabe -- eine Zeile pro Lidar-Punkt),
  * PointCloud2 mit RGB -- Eingang der Hinderniserkennung (scan_processor)
    und zugleich Foxglove-Ansicht; eingefaerbt entweder in kraeftigen
    Label-Farben (``cloud_color_mode: label``, Default) oder in der gemessenen
    Pixelfarbe (``raw``),
  * Debug-Bild mit den eingezeichneten Projektionen.

Wo im Bild abgegriffen wird (Parameter ``sample_mode``):

  horizon  (Default) auf Objektivhoehe. Die Hoehendifferenz zur Kamera ist dann
           null, theta exakt 90 Grad, der Bildradius konstant f*pi/2 -- es
           bleibt nur der Azimut, also eine feste Kreislinie im Bild.
           Das reicht fuer Pylonen, SOLANGE das Objektiv zwischen Matte und
           Pylonenoberkante sitzt: eine Pylone, die die waagerechte Ebene durch
           die Linse durchstoesst, liegt in JEDER Entfernung auf diesem Ring.
           Vorteil: Entfernungsfehler des Lidars und ein falsches cam_z wirken
           sich radial gar nicht mehr aus, es zaehlt nur noch yaw.
           Sitzt die Linse ueber der Pylonenoberkante, greift der Ring dagegen
           an der Pylone vorbei -- dann height nehmen.

  height   auf fester Hoehe ``sample_height_m`` ueber der Lidar-Ebene. Der
           Bildradius haengt dann an der Entfernung.

Ring nach unten kippen (``sample_depression_deg``, nur bei horizon): aus der
waagerechten Ebene wird ein Kegel. Der greift in waagerechter Entfernung rho um
rho*tan(Winkel) unter der Linse ab -- die Tiefe waechst also MIT der Entfernung.
Bei 1 Grad sind das 0.5 cm auf 0.3 m, aber 3.5 cm auf 2 m. Fuer 10-cm-Pylonen
heisst das: nur Bruchteile eines Grades sind brauchbar, und sitzt die Linse
ueber der Pylonenoberkante, gibt es GAR KEINEN Winkel, der nah und fern
gleichzeitig trifft -- dann hilft nur ``height``.

Mitteln statt ein Pixel (``sample_band_m``, ``sample_band_count``): es werden
mehrere Stuetzstellen entlang der radialen Linie durch den Punkt gelesen -- die
liegt im Fisheye laengs der Pylone -- und davon der Median genommen. Die
Bandbreite ist in Metern Pylonenhoehe angegeben und wird je Punkt aus der
Entfernung in Pixel umgerechnet, schrumpft fern also von selbst mit und bleibt
damit innerhalb der Pylone. 0 schaltet auf ein einzelnes Pixel zurueck.

Statt einer Linie eine ZONE (``sample_zone_high_m`` > ``sample_zone_low_m``):
ein einzelner Abgriffsradius trifft je nach Entfernung und Kalibrierfehler mal
die Pylone, mal die Wand dahinter, mal den Boden davor. Die Zone tastet
stattdessen ein Stueck der radialen Linie ab und zaehlt aus, welcher Anteil der
Pixel zu welcher Farbe passt; ab ``sample_zone_min_frac`` gewinnt eine Farbe.

Entscheidend ist, dass die Zone durch zwei HOEHEN aufgespannt wird und nicht
durch eine Pixelbreite. Eine Bande fester Hoehe ist im Fisheye naemlich KEIN
Kreisband konstanter Dicke:

  * Die obere Kante, wenn sie auf Objektivhoehe liegt: Hoehendifferenz null,
    theta exakt 90 Grad, Radius konstant. Sie laeuft als gerade Linie, egal in
    welcher Entfernung.
  * Die untere Kante liegt die Bandenhoehe tiefer. Ihr theta naehert sich mit
    wachsender Entfernung von oben an 90 Grad an, ihr Radius also von aussen an
    den der oberen Kante. Sie wandert mit der Entfernung nach oben.

Bei 9 cm Bandenhoehe und f=262 px/rad heisst das: die Zone ist auf 0.3 m rund
77 px dick, auf 1 m noch 24 px und auf 3 m nur 8 px. Eine konstante Pixelbreite
waere nah viel zu schmal und fern zu breit -- fern ragt sie ueber die Bande
hinaus und sammelt die helle Wand dahinter mit ein, wodurch die Punkte
faelschlich als "unbekannt" statt "schwarz" herauskommen. Genau das war am
Aufbau zu sehen: bei 2.3 m Entfernung lag die Bande bei r=402 px, bei 0.7 m
zwischen 370 und 415 px, und ein fester Ring bei 412 px las in den fernen
Richtungen V=235 statt V=25.

CSV-Modi (Parameter ``csv_mode``):
  trigger      pro Trigger eine Datei  -> ros2 topic pub --once \
                   /camera_lidar/capture std_msgs/msg/Empty '{}'
  continuous   haengt jeden Scan an eine Datei an
  off          keine CSV, nur Topics

Start:
    ros2 run camera_lidar_fusion lidar_pixel_mapper
"""

import collections
import csv
import datetime
import math
import os
import threading
import time

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import rclpy
from cv_bridge import CvBridge
from rcl_interfaces.msg import ParameterDescriptor
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image, LaserScan, PointCloud2, PointField
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Empty, String

import cv2

from camera_lidar_fusion import colors
from camera_lidar_fusion.fisheye_model import (
    FisheyeCalib, project, scan_to_points, theta_to_radius, visible_mask,
)

CLOUD_FIELDS = [
    PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
    PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
    PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
    PointField(name='rgb', offset=12, datatype=PointField.FLOAT32, count=1),
]

# Sensor-QoS mit Tiefe 1 statt der ueblichen 5: bei einer Node, die langsamer
# rechnet als das Lidar liefert, fuellt eine tiefe Queue nur einen Rueckstau.
# Mit depth=1 liegt immer der NEUESTE Scan an -- lieber einen auslassen als alle
# um fuenf Frames verspaetet zu faerben.
SCAN_QOS = QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=1,
                      reliability=ReliabilityPolicy.BEST_EFFORT)
# Bilder brauchen etwas mehr Tiefe, damit der Ringpuffer auch bei Jitter
# lueckenlos gefuellt wird -- gepuffert wird dann in der Node, nach Zeitstempel.
IMAGE_QOS = QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=5,
                       reliability=ReliabilityPolicy.BEST_EFFORT)
# Die Odometrie traegt die Bewegungskompensation. Hier ZUVERLAESSIG und mit
# Tiefe, denn eine Luecke im Posenpuffer kostet die Korrektur fuer alle Scans,
# die in die Luecke fallen. /ekf/odom sendet mit dem Standardprofil (RELIABLE),
# ein RELIABLE-Abonnent passt also dazu.
ODOM_QOS = QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=50,
                      reliability=ReliabilityPolicy.RELIABLE)

CSV_HEADER = [
    'stamp_sec', 'idx', 'angle_deg', 'range_m', 'x_m', 'y_m', 'z_m',
    'u_px', 'v_px', 'theta_deg', 'phi_deg', 'b', 'g', 'r', 'h', 's', 'v', 'label',
]


_KREIS_CACHE = {}


def _kreis_offsets(radius, ring):
    """Pixel-Offsets eines Kreises, einmal von cv2 selbst gezeichnet.

    So setzt die vektorisierte Variante exakt dieselben Pixel wie ein
    ``cv2.circle`` je Punkt -- ein selbst gerechnetes Muster trifft besonders
    den 1 px breiten Rand nicht genau.
    """
    schluessel = (int(radius), bool(ring))
    if schluessel not in _KREIS_CACHE:
        r = int(radius)
        patch = np.zeros((2 * r + 3, 2 * r + 3), np.uint8)
        cv2.circle(patch, (r + 1, r + 1), r, 255, 1 if ring else -1)
        oy, ox = np.nonzero(patch)
        _KREIS_CACHE[schluessel] = (ox.astype(np.int32) - (r + 1),
                                    oy.astype(np.int32) - (r + 1))
    return _KREIS_CACHE[schluessel]


def _scheiben(canvas, u, v, farben, radius, ring=False):
    """Kleine Scheiben (oder Ringe) an (u,v) in EINEM numpy-Zugriff setzen.

    Ersetzt die Schleife mit einem ``cv2.circle`` je Punkt. Bei 2400 Punkten
    kostete die am Aufbau 43 ms -- fast die gesamte Zeit eines Scans.

    ``farben`` ist (N,3) in BGR, ``ring=True`` zeichnet nur den Rand.
    """
    n = len(u)
    if n == 0:
        return
    hoehe, breite = canvas.shape[:2]
    ox, oy = _kreis_offsets(radius, ring)
    pu = np.rint(np.asarray(u)).astype(np.int32)[:, None] + ox[None, :]
    pv = np.rint(np.asarray(v)).astype(np.int32)[:, None] + oy[None, :]
    gut = (pu >= 0) & (pu < breite) & (pv >= 0) & (pv < hoehe)
    farben = np.asarray(farben, dtype=np.uint8).reshape(n, 1, 3)
    canvas[pv[gut], pu[gut]] = np.broadcast_to(farben, pu.shape + (3,))[gut]


def _segmente(canvas, x0, y0, x1, y1, farbe, dicke=1):
    """Viele gerade Strecken mit EINEM cv2.polylines-Aufruf statt cv2.line je
    Strecke. polylines nimmt eine Liste von Polygonzuegen -- hier je Strecke
    einer aus zwei Punkten."""
    if len(x0) == 0:
        return
    pts = np.stack([np.column_stack([x0, y0]), np.column_stack([x1, y1])], axis=1)
    cv2.polylines(canvas, np.rint(pts).astype(np.int32), False, farbe, dicke)


def _stuecke(werte):
    """Indexbloecke zusammenhaengender endlicher Werte (NaN trennt)."""
    gut = np.isfinite(werte)
    if not gut.any():
        return []
    kanten = np.flatnonzero(np.diff(gut.astype(np.int8)))
    bloecke = np.split(np.arange(len(werte)), kanten + 1)
    return [b for b in bloecke if gut[b[0]] and len(b) >= 2]


class LidarPixelMapper(Node):

    def __init__(self):
        super().__init__('lidar_pixel_mapper')

        self.declare_parameter('scan_topic', '/scan')
        self.declare_parameter('image_topic', '/video_source/raw')
        self.declare_parameter('calib_file', '/workspace/config/fisheye_calib.yaml')
        # horizon = auf dem Horizontring abgreifen (Default, siehe Modulkopf).
        # height  = auf fester Hoehe ueber der Lidar-Ebene, dann zaehlt
        #           sample_height_m. Nur noetig, wenn das Objektiv NICHT
        #           zwischen Matte und Pylonenoberkante sitzt.
        self.declare_parameter('sample_mode', 'horizon')
        self.declare_parameter('sample_height_m', 0.00)
        # Ring nach unten kippen (nur bei horizon). 0 = waagerecht durch die
        # Linse. Positiv blickt nach unten, der Ring wird groesser. ACHTUNG: der
        # Kegel greift dann in ENTFERNUNG*tan(Winkel) Tiefe -- fern also viel
        # tiefer als nah. Siehe Modulkopf.
        self.declare_parameter('sample_depression_deg', 0.0)
        # Statt eines Pixels laengs der Pylone mitteln: ueber +-sample_band_m
        # Pylonenhoehe, mit sample_band_count Stuetzstellen. 0 = ein Pixel.
        self.declare_parameter('sample_band_m', 0.03)
        self.declare_parameter('sample_band_count', 5)
        # Zonen-Abstimmung statt Band-Median. Die Zone wird durch zwei HOEHEN
        # ueber der Lidar-Ebene aufgespannt, nicht durch eine Pixelbreite --
        # eine Bande fester Hoehe erscheint im Fisheye naemlich nicht als
        # Kreisband konstanter Dicke (siehe Modulkopf). Aus den Hoehen folgt je
        # Punkt ein Radienintervall, das mit der Entfernung von selbst
        # schrumpft. sample_zone_high_m <= sample_zone_low_m schaltet ab.
        self.declare_parameter('sample_zone_low_m', 0.0)
        self.declare_parameter('sample_zone_high_m', 0.0)
        self.declare_parameter('sample_zone_steps', 13)
        self.declare_parameter('sample_zone_min_frac', 0.20)
        # Welcher Anteil der Zone wird abgetastet? 1.0 = ganze Zone,
        # 0.33 = mittleres Drittel. Sitzen die Zonengrenzen sauber, ist die
        # Mitte die sauberste Stelle -- die Raender tragen Mischpixel bei.
        self.declare_parameter('sample_zone_nutz', 1.0)
        # Welche Farben ueberhaupt gesucht werden. Wird bei jedem Scan gelesen,
        # laesst sich also im Betrieb umschalten -- anders als die Schwellen in
        # color.*, die beim Start eingefroren werden. Magenta zum Beispiel
        # produziert in der Entfernung leicht Fehltreffer und stoert nur,
        # solange die Parkzone nicht gebraucht wird.
        self.declare_parameter('active_labels', ['rot', 'gruen', 'magenta'])
        # Saettigungsschwelle relativ zur Umgebung statt absolut. 0 = aus.
        # Eine Pylone ist immer deutlich gesaettigter als die Bande neben ihr
        # (gemessen Faktor 2.4 bis 2.9), und zwar unabhaengig davon, ob sie im
        # Schatten steht. Absolute Schwellen scheitern dagegen an dunklen
        # Pylonen, weil die in S UND V mit der Bande ueberlappen.
        # Das Fenster muss breiter sein als eine Pylone, sonst hebt sie ihre
        # eigene Schwelle an.
        # Rot/Gruen ueber das Kanalverhaeltnis (G-R)/max(B,G,R) statt ueber ein
        # Farbton-Fenster. Am Aufbau gemessen trennt das die beiden Pylonen von
        # Bande, Holz und Eigenaufbau vollstaendig -- ueber alle 151 Azimut-
        # fenster des Vollkreises kein einziger Fehlalarm. Details und Zahlen in
        # colors.rg_kennzahl. 0 schaltet zurueck auf das Farbton-Fenster.
        # Harter innerer Radiusanschlag. Alles innerhalb davon ist im Fisheye
        # der RAUM -- Decke, Wand, Moebel, Holz -- und hat im Abgriff nichts
        # verloren. Am Rohbild gemessen (Radialprofil ueber alle Azimute):
        #     r 279..390  V 108..200  heller Raum
        #     r 397..419  V  38.. 84  die Bande
        #     r 427..449  V 221..255  die Matte
        # Der Uebergang Raum -> Bande liegt scharf bei r ~390. Beide Pylonen
        # standen bei r 391..412. Der Anschlag ist eine KONSTANTE: die
        # Bandenoberkante liegt auf Objektivhoehe, ihr Bildradius haengt also
        # nicht von der Entfernung ab. 0 schaltet den Anschlag ab.
        self.declare_parameter('sample_r_min_px', 0.0)
        # FESTES Abgriffsfenster statt der gefitteten Kurve. Am Rohbild mit zwei
        # Pylonen auf 0.85 m ausgemessen: beide belegen r 391..412 px, darunter
        # (kleinerer Radius) ist heller Raum, darueber die helle Matte. Mit
        # 394..412 und der G-R-Kennzahl: rot 28/28, gruen 26/27, NULL Fehlalarme
        # ueber den ganzen Kreis -- gegen 17 Fehlalarme mit Kurve + Farbton.
        #
        # Physikalisch gerechtfertigt ist vor allem die INNERE Kante: sie liegt
        # an der Bandenoberkante auf Objektivhoehe und haengt damit nicht von
        # der Entfernung ab. Die aeussere Kante wandert eigentlich mit der
        # Entfernung -- ob 412 auch auf 2..3 m traegt, ist noch nicht gemessen.
        # Beide 0 -> wie bisher ueber _zone_radien.
        self.declare_parameter('sample_r_fix_in', 0.0)
        self.declare_parameter('sample_r_fix_out', 0.0)
        self.declare_parameter('rg_z_min', 0.15)
        # Zweites Tor: Mindestsaettigung. Bande liegt bei S~36 (p95 55), die
        # gruene Pylone bei S 76..131, die rote bei 162..219.
        self.declare_parameter('rg_s_min', 60)
        # Absolutes Tor auf |G-R| in Zaehlwerten. Faengt den Farbstich ueber
        # das Fischauge ab, gegen den die relativen Tore blind sind.
        self.declare_parameter('rg_d_min', 20)
        self.declare_parameter('sample_zone_adaptiv', 0.0)
        self.declare_parameter('sample_zone_adaptiv_grad', 20.0)
        # Median-Blur ueber das GANZE Bild -- kostet auf 1280x960 rund 26 ms je
        # Scan, also bei 15 Hz gut 40 Prozent eines Kerns. Solange das Band aktiv
        # ist (sample_band_m > 0), ist der Blur ueberfluessig: der Median laengs
        # der Pylone faengt Ausreisser bereits ab. Nur hochdrehen, wenn du das
        # Band abschaltest.
        self.declare_parameter('patch_px', 1)
        self.declare_parameter('range_min_m', 0.05)
        self.declare_parameter('range_max_m', 3.0)
        self.declare_parameter('max_sync_age_s', 0.5)
        # Bilder werden mit Zeitstempel in einem Ringpuffer gehalten; zu jedem
        # Scan wird das zeitlich naechstliegende gesucht statt blind das letzte
        # zu nehmen. 8 Bilder sind bei 15 fps gut eine halbe Sekunde Historie.
        self.declare_parameter('image_buffer_len', 8)
        # Findet sich kein Bild innerhalb von max_sync_age_s, ist jede Faerbung
        # geraten: der Scan wird dann verworfen statt falsch eingefaerbt. Auf
        # false nur zum Debuggen, wenn man die schlechte Zuordnung sehen will.
        self.declare_parameter('sync_drop', True)
        # BEWEGUNGSKOMPENSATION. Das Bild zum Scan ist im Fahrbetrieb 100 bis
        # 700 ms alt (Kamera faellt unter Last von 15 auf 3 Hz). In dieser Zeit
        # hat sich der Roboter gedreht und bewegt -- der Abgriff-Azimut aus dem
        # Lidarstrahl zeigt dann im BILD woanders hin. Gemessen in Lauf 20:
        # Farbausbeute auf einer Pylone 38 Prozent im Stand, 6 Prozent ab
        # 0.5 rad/s, 2 Prozent ab 1 rad/s. Eine Pylone ist bei 1.6 m nur 1.6
        # Grad breit, 0.5 rad/s mal 0.3 s sind 8.6 Grad -- also glatt daneben.
        # Hier werden die Lidarpunkte deshalb in den Roboter-Frame ZUM
        # BILDZEITPUNKT zurueckgerechnet, bevor sie projiziert werden. Die
        # veroeffentlichte Punktwolke bleibt unveraendert bei der Scangeometrie.
        self.declare_parameter('motion_compensation', True)
        self.declare_parameter('odom_topic', '/ekf/odom')
        # Wie weit die Pose extrapoliert werden darf, wenn der Puffer den
        # Bildzeitpunkt nicht ganz abdeckt. 0 = gar nicht (dann keine Korrektur).
        self.declare_parameter('pose_extrapolate_s', 0.05)
        self.declare_parameter('pose_buffer_len', 400)
        # Lidar im base_link: der Punkt, um den sich das Lidar beim Gieren
        # dreht. Nur fuer den kleinen Translationsanteil r*dtheta noetig.
        self.declare_parameter('lidar_offset_x', 0.110)
        self.declare_parameter('lidar_offset_y', 0.0)
        self.declare_parameter('lidar_yaw_deg', 180.0)
        # Alle n Sekunden eine Zeile mit Bildrate, Scanrate und dem tatsaechlich
        # erreichten Zeitversatz. Ohne die sieht man im Feld nicht, ob die
        # Zuordnung gerade gut ist. 0 schaltet sie ab.
        self.declare_parameter('stats_period_s', 10.0)
        # Die Fusion muss NICHT mit der Lidar-Rate laufen: Pylonen bewegen sich
        # nicht, und die Farbe je Punkt ist nach ein paar Scans entschieden.
        # Das Lidar liefert 15 Hz; jeder Scan kostet hier ~30 ms Rechenzeit, im
        # Fahrbetrieb bei ausgelasteten Kernen deutlich mehr. Begrenzen entlastet
        # genau die CPU, die sonst dem Kamerapfad fehlt.
        # 0 = jeden Scan verarbeiten (altes Verhalten).
        # dynamic_typing, damit auch "fusion_rate_hz:=0" durchgeht. Ohne das
        # lehnt rclpy die 0 als INTEGER gegen den DOUBLE-Default ab und die Node
        # startet gar nicht erst.
        self.declare_parameter('fusion_rate_hz', 7.0,
                               ParameterDescriptor(dynamic_typing=True))
        self.declare_parameter('csv_mode', 'trigger')
        self.declare_parameter('csv_dir', '/workspace/lidar_color_logs')
        self.declare_parameter('csv_only_labeled', False)
        # debug schaltet NUR noch das Debug-Bild -- also das, was wirklich nur
        # zum Anschauen da ist. Im Wettkampflauf auf false setzen: dann faellt
        # das Zeichnen und Serialisieren weg, die Punktwolke bleibt aber, weil
        # scan_processor_node daraus die Hindernisse baut.
        self.declare_parameter('debug', True)
        self.declare_parameter('publish_cloud', True)
        # Womit die Punkte in /camera_lidar/colored_scan eingefaerbt werden:
        #   label  (Default) kraeftige Farbe je erkanntem Label. Rot und Gruen
        #          stechen heraus, alles Unklassifizierte bleibt dunkelgrau --
        #          die Ansicht zum Pylonensuchen. Die Werte sind exakt, also
        #          auch maschinell eindeutig auswertbar.
        #   raw    die tatsaechlich gemessene Pixelfarbe. Die braucht man zum
        #          Pruefen der Kalibrierung (stehen die roten Punkte auf dem
        #          roten Klotz?) und zum Nachziehen der Farbschwellen.
        # Wird bei jedem Scan neu gelesen, wirkt also sofort.
        self.declare_parameter('cloud_color_mode', 'label')
        self.declare_parameter('publish_debug_image', True)
        self.declare_parameter('debug_rate_hz', 5.0)
        # Polar-Entzerrung unter das runde Bild haengen: Azimut waagerecht,
        # Bildradius senkrecht. Darin liegt die Bande als waagerechtes Band und
        # der Abgriff als Linie -- man sieht also auf einen Blick, ob der
        # Abgriff die Bande trifft oder darueber bzw. darunter vorbeigreift.
        # Im runden Bild ist das kaum zu beurteilen, weil dort alles am aeusseren
        # Rand zusammengedraengt ist.
        self.declare_parameter('debug_polar', True)
        self.declare_parameter('debug_polar_height', 150)
        # Bandensuche: je Azimut von innen nach aussen laufen und die Stelle
        # suchen, an der die schwarze Bande in die helle Matte uebergeht. Das
        # ist die verlaesslichste Kante im Bild -- dahinter liegt immer die
        # Matte, also derselbe Kontrast, egal in welche Richtung. Die Oberkante
        # taugt dafuer nicht: hinter ihr ist mal Wand, mal Moebel, mal Holz
        # (an 1362 Kantenpaaren gemessen: RMS 12.8 px oben gegen 5.3 px unten).
        self.declare_parameter('band_detect', True)
        self.declare_parameter('band_steps', 360)        # Azimutschritte
        self.declare_parameter('band_r_min', 360.0)      # Suchbereich von innen
        self.declare_parameter('band_r_max', 0.0)        # 0 = bis Bildkreisrand
        self.declare_parameter('band_dark_max', 60)      # so dunkel ist die Bande
        self.declare_parameter('band_bright_min', 100)   # so hell ist die Matte
        self.declare_parameter('band_run', 4)            # so viele helle am Stueck
        # Ausreisserfilter. Die Bande ist rund 10 cm hoch und die Kamera sitzt
        # auf ihrer Oberkante -- deshalb kann die Unterkante nur in einem
        # schmalen Band liegen, und Nachbarazimute muessen sich aehneln. Ein
        # Glanzpunkt IN der Bande loest die Kante sonst zu frueh aus und zieht
        # eine Zacke nach innen. Solche Werte sind schlicht falsch.
        self.declare_parameter('band_smooth', 9)         # Median ueber n Azimute
        self.declare_parameter('band_max_dev', 12.0)     # max Abweichung davon [px]
        self.declare_parameter('band_min_dicke', 3.0)    # min Abstand zur Oberkante
        # Die Zone an die gefundene Bande koppeln, statt sie zu rechnen oder aus
        # der Kalibrierkurve zu nehmen. Die Oberkante ist dabei konstant -- das
        # Objektiv sitzt auf ihrer Hoehe, die Hoehendifferenz ist damit null und
        # der Bildradius entfernungsunabhaengig (die Zonenkalibrierung bestaetigt
        # das: zone_k_in entspricht nur 0.9 cm). Die Unterkante kommt live aus
        # dem Bild. Wo keine Kante gefunden wurde, greift die Kalibrierkurve.
        self.declare_parameter('zone_from_band', False)

        self.scan_topic = self.get_parameter('scan_topic').value
        self.image_topic = self.get_parameter('image_topic').value
        self.calib_path = self.get_parameter('calib_file').value
        self.csv_dir = self.get_parameter('csv_dir').value
        self.ranges = colors.ranges_from_params(self)

        self.calib = FisheyeCalib.load(self.calib_path, _packaged_default())
        self.bridge = CvBridge()
        # Ringpuffer statt einem einzelnen "letztes Bild": zu jedem Scan wird
        # das zeitlich passende Bild gesucht (siehe _bild_zum_scan). Der Puffer
        # wird aus dem Bild-Thread beschrieben und aus dem Scan-Thread gelesen,
        # deshalb die Sperre.
        self.image_buf = collections.deque(
            maxlen=max(2, int(self.get_parameter('image_buffer_len').value)))
        self.image_lock = threading.Lock()
        self.sync_stats = [0, 0]        # [gefaerbt, wegen Zeitversatz verworfen]
        self.n_images = 0
        self.versatz_log = collections.deque(maxlen=300)
        # Posenpuffer fuer die Bewegungskompensation: (stempel, x, y, yaw).
        # 400 Eintraege sind bei 50 Hz acht Sekunden -- reicht auch fuer die
        # seltenen 1.9-s-Ausreisser im Bildversatz.
        self.pose_buf = collections.deque(
            maxlen=max(2, int(self.get_parameter('pose_buffer_len').value)))
        self.pose_lock = threading.Lock()
        self.komp_log = collections.deque(maxlen=300)   # (|dyaw| rad, |dt| m)
        self.n_komp_ohne_pose = 0
        self._stats_letzte = None
        self._naechster_slot = 0.0
        self._letzter_scan = 0.0
        self._scan_periode = 0.0
        self.n_rate_skip = 0
        self.capture_pending = False
        self.continuous_writer = None   # (file, csv.writer) fuer csv_mode=continuous
        self.last_debug_stamp = 0.0
        self._polar_map = None          # (schluessel, map_x, map_y) fuer _polar_view
        self._capture_image = None      # Rohbild des Scans, den capture erwischt

        # Eigene Callback-Gruppen: Scan und Bild laufen im MultiThreadedExecutor
        # nebenlaeufig. Vorher hing beides am selben Thread -- solange on_scan
        # rechnete (gemessen ~130 ms), konnte on_image nicht laufen, und das
        # "letzte Bild" war entsprechend alt.
        self.cbg_scan = MutuallyExclusiveCallbackGroup()
        self.cbg_image = MutuallyExclusiveCallbackGroup()
        # Eigene Gruppe fuer die Odometrie: on_scan rechnet rund 30 ms, und der
        # Posenpuffer darf in dieser Zeit keine Luecke bekommen.
        self.cbg_odom = MutuallyExclusiveCallbackGroup()

        self.create_subscription(LaserScan, self.scan_topic, self.on_scan,
                                 SCAN_QOS, callback_group=self.cbg_scan)
        self.create_subscription(Image, self.image_topic, self.on_image,
                                 IMAGE_QOS, callback_group=self.cbg_image)
        self.create_subscription(Odometry, self.get_parameter('odom_topic').value,
                                 self.on_odom, ODOM_QOS,
                                 callback_group=self.cbg_odom)
        self.create_subscription(Empty, '/camera_lidar/capture', self.on_capture, 10)
        # Nach einem "save" in der Kalibrier-Node hier neu einlesen, statt die
        # Node neu starten zu muessen.
        self.create_subscription(Empty, '/camera_lidar/reload', self.on_reload, 10)

        self.pub_cloud = self.create_publisher(PointCloud2, '/camera_lidar/colored_scan', 5)
        self.pub_debug = self.create_publisher(Image, '/camera_lidar/debug_image', 2)
        self.pub_summary = self.create_publisher(String, '/camera_lidar/summary', 10)

        periode = float(self.get_parameter('stats_period_s').value)
        if periode > 0.0:
            self.create_timer(periode, self._log_stats)

        if self.get_parameter('csv_mode').value == 'continuous':
            self._open_continuous_csv()

        mode = self.get_parameter('sample_mode').value
        depression = self.get_parameter('sample_depression_deg').value
        if mode == 'horizon':
            radius = float(theta_to_radius(
                self.calib, np.array([np.pi / 2 + math.radians(depression)]))[0])
            abgriff = (f'horizon -- feste Kreislinie bei r={radius:.1f} px, '
                       f'entfernungsunabhaengig.\n'
                       f'    Setzt voraus, dass das Objektiv ZWISCHEN Matte und '
                       f'Pylonenoberkante sitzt. Mittig (ca. 5 cm bei 10-cm-Pylonen) '
                       f'ist der Abstand zu beiden Kanten am groessten.')
            if depression != 0.0:
                abgriff += (f'\n    Ring {depression:.2f} Grad nach unten gekippt: greift '
                            f'{math.tan(math.radians(depression)) * 30:.1f} cm unter der Linse '
                            f'ab auf 0.3 m, aber '
                            f'{math.tan(math.radians(depression)) * 200:.1f} cm auf 2 m.')
        else:
            abgriff = (f'height -- {self.get_parameter("sample_height_m").value * 100:.1f} cm '
                       f'ueber der Lidar-Ebene, Bildradius haengt an der Entfernung.')

        z_lo = self.get_parameter('sample_zone_low_m').value
        z_hi = self.get_parameter('sample_zone_high_m').value
        zone_an = z_hi > z_lo or self.calib.zone_kalibriert
        band_m = 0.0 if zone_an else self.get_parameter('sample_band_m').value
        if zone_an:
            frac = self.get_parameter('sample_zone_min_frac').value
            dicke = []
            for d in (0.3, 1.0, 3.0):
                ri, ra = self._zone_radien(np.array([d]), z_lo, z_hi)
                dicke.append(f'{d:.1f} m: {float(ri[0]):.0f}..{float(ra[0]):.0f} px')
            if self.calib.zone_kalibriert:
                quelle = (f'GEMESSEN: r_innen = {self.calib.zone_r0_in:.1f} '
                          f'{self.calib.zone_k_in:+.2f}/rho, r_aussen = '
                          f'{self.calib.zone_r0_out:.1f} {self.calib.zone_k_out:+.2f}/rho')
            else:
                quelle = (f'GERECHNET aus {z_lo * 100:.1f} bis {z_hi * 100:.1f} cm ueber '
                          f'der Lidar-Ebene (nicht kalibriert -- "zone"/"zonefit" in der '
                          f'Kalibrier-Node liefert bessere Werte)')
            abgriff = (
                f'ZONE, Abstimmung ab {frac * 100:.0f} Prozent der Pixel.\n'
                f'    {quelle}\n'
                f'    Daraus: ' + ', '.join(dicke) + '.\n'
                f'    Eine Pylone fester Hoehe ist im Fisheye eben KEIN Kreisband '
                f'konstanter Dicke -- nah ist sie breit, fern schmal.')
        elif band_m > 0.0:
            abgriff += (f'\n    Median ueber +-{band_m * 100:.1f} cm Pylonenhoehe '
                        f'({self.get_parameter("sample_band_count").value} Stuetzstellen '
                        f'laengs der Pylone).')
        else:
            patch = self.get_parameter('patch_px').value
            abgriff += f'\n    Ein einzelnes Pixel (sample_band_m = 0, patch_px = {patch}).'
            if patch <= 1:
                abgriff += (' ACHTUNG: weder Band noch Blur -- ungefiltert. '
                            'patch_px hochsetzen oder sample_band_m > 0.')

        self.get_logger().info(
            f'lidar_pixel_mapper laeuft. scan={self.scan_topic} image={self.image_topic}\n'
            f'  Kalibrierung: {self.calib_path}\n'
            f'  Bildkreis cx={self.calib.cx:.1f} cy={self.calib.cy:.1f} '
            f'r={self.calib.radius_px:.1f} FOV={self.calib.fov_deg:.0f} Grad\n'
            f'  Lage yaw={self.calib.yaw_deg:.2f} pitch={self.calib.pitch_deg:.2f} '
            f'roll={self.calib.roll_deg:.2f} (Grad), '
            f'Kamera {self.calib.cam_z * 100:.1f} cm ueber der Lidar-Ebene\n'
            f'  Abgriff: {abgriff}\n'
            f'  CSV-Modus: {self.get_parameter("csv_mode").value} -> {self.csv_dir}\n'
            f'  debug={self.get_parameter("debug").value}, '
            f'cloud_color_mode={self.get_parameter("cloud_color_mode").value} '
            f'({"kraeftige Label-Farben" if self.get_parameter("cloud_color_mode").value == "label" else "gemessene Pixelfarbe"})\n'
            + (f'  Fusionsrate: begrenzt auf {self.get_parameter("fusion_rate_hz").value:.1f} Hz'
               ' (fusion_rate_hz:=0 -> jeden Scan)\n'
               if float(self.get_parameter('fusion_rate_hz').value) > 0.0
               else '  Fusionsrate: jeder Scan (fusion_rate_hz=0)\n')
            + f'  -> Foxglove: /camera_lidar/colored_scan und /camera_lidar/debug_image'
        )

    # ---------------------------------------------------------------- #
    def on_image(self, msg: Image):
        try:
            image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        except Exception as exc:  # noqa: BLE001
            self.get_logger().warn(f'Bild nicht dekodierbar: {exc}')
            return
        with self.image_lock:
            self.image_buf.append((_stamp_sec(msg.header.stamp), image))
            self.n_images += 1

    def _log_stats(self):
        """Wie gut passt die Zuordnung gerade? Eine Zeile alle stats_period_s.

        Raten ueber time.monotonic(), nicht ueber die ROS-Uhr: die kann per NTP
        springen, und dann stimmen die Hz-Angaben nicht mehr.
        """
        jetzt = time.monotonic()
        with self.image_lock:
            stand = (jetzt, self.n_images, self.sync_stats[0], self.sync_stats[1],
                     self.n_rate_skip)
            versatz = list(self.versatz_log)
        with self.pose_lock:
            komp = list(self.komp_log)
            ohne_pose = self.n_komp_ohne_pose
            self.n_komp_ohne_pose = 0
        if self._stats_letzte is None:
            self._stats_letzte = stand
            return
        dt = stand[0] - self._stats_letzte[0]
        if dt < 1e-3:
            return
        d_img = stand[1] - self._stats_letzte[1]
        d_ok = stand[2] - self._stats_letzte[2]
        d_weg = stand[3] - self._stats_letzte[3]
        d_skip = stand[4] - self._stats_letzte[4]
        self._stats_letzte = stand
        # "zugeordnet" = Scan hat ein Bild innerhalb max_sync_age_s bekommen.
        # Das ist nicht dasselbe wie die Rate von /camera_lidar/colored_scan:
        # danach koennen noch Scans ohne Punkt im Sichtfeld herausfallen.
        text = (f'Sync: Bilder {d_img / dt:.1f} Hz, zugeordnet {d_ok / dt:.1f} Hz, '
                f'verworfen {d_weg / dt:.1f} Hz')
        rate = float(self.get_parameter('fusion_rate_hz').value)
        if rate > 0.0:
            text += (f' | Drossel {rate:.1f} Hz: von {(d_ok + d_weg + d_skip) / dt:.1f} Hz '
                     f'Scans {d_skip / dt:.1f} Hz uebersprungen')
        if versatz:
            v = np.abs(np.asarray(versatz)) * 1000.0
            text += (f' | Versatz Bild-Scan: med {np.median(v):.0f} ms, '
                     f'p90 {np.percentile(v, 90):.0f} ms, max {v.max():.0f} ms')
        if not self.get_parameter('motion_compensation').value:
            text += ' | Bewegungskompensation AUS'
        elif komp:
            k = np.asarray(komp)
            gier = np.degrees(k[:, 0])
            text += (f' | Kompensiert: Gier med {np.median(gier):.1f} Grad, '
                     f'p90 {np.percentile(gier, 90):.1f} Grad, max {gier.max():.1f} Grad; '
                     f'Versatz med {np.median(k[:, 1]) * 100:.0f} cm')
            if ohne_pose:
                text += f'; {ohne_pose} Scans ohne Pose (unkorrigiert)'
        else:
            text += (' | Bewegungskompensation ohne Wirkung: keine Pose empfangen '
                     f'({self.get_parameter("odom_topic").value} da?)')
        self.get_logger().info(text)

    def _bild_zum_scan(self, scan_stamp):
        """Das Bild aus dem Puffer, dessen Zeitstempel dem Scan am naechsten liegt.

        Rueckgabe ``(bild, bild_stempel, versatz)`` oder ``(None, None, None)``,
        wenn noch nichts da ist. ``versatz`` ist vorzeichenbehaftet:
        positiv = das Bild ist NEUER als der Scan.

        Warum ueberhaupt suchen: Lidar und Kamera laufen frei gegeneinander, und
        beide Topics werden unabhaengig gepuffert. "Das zuletzt eingetroffene
        Bild" ist deshalb mal 20 ms, mal 800 ms vom Scan entfernt -- und ein
        fester Korrekturwert hilft nicht, weil der Versatz schwankt. Ueber den
        Stempel gesucht ist die Zuordnung dagegen so gut, wie die Rate hergibt.
        """
        with self.image_lock:
            if not self.image_buf:
                return None, None, None
            kandidaten = list(self.image_buf)
        stempel, bild = min(kandidaten, key=lambda e: abs(e[0] - scan_stamp))
        return bild, stempel, stempel - scan_stamp

    # ---------------------------------------------------------------- #
    # Bewegungskompensation
    # ---------------------------------------------------------------- #
    def on_odom(self, msg: Odometry):
        q = msg.pose.pose.orientation
        yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                         1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        with self.pose_lock:
            self.pose_buf.append((_stamp_sec(msg.header.stamp),
                                  msg.pose.pose.position.x,
                                  msg.pose.pose.position.y, yaw))

    def _pose_bei(self, t, puffer):
        """Pose zum Zeitpunkt t, linear interpoliert. None, wenn zu weit weg.

        Der Gierwinkel wird ueber die DIFFERENZ interpoliert, sonst springt er
        beim Ueberlauf von +pi nach -pi mitten in der Kurve.
        """
        if len(puffer) < 2:
            return None
        rand = float(self.get_parameter('pose_extrapolate_s').value)
        if t < puffer[0][0] - rand or t > puffer[-1][0] + rand:
            return None
        # Puffer ist nach Zeit sortiert (Odometrie kommt monoton an).
        lo, hi = 0, len(puffer) - 1
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if puffer[mid][0] <= t:
                lo = mid
            else:
                hi = mid
        t0, x0, y0, th0 = puffer[lo]
        t1, x1, y1, th1 = puffer[hi]
        if t1 <= t0:
            return x0, y0, th0
        f = (t - t0) / (t1 - t0)
        dth = math.atan2(math.sin(th1 - th0), math.cos(th1 - th0))
        return x0 + (x1 - x0) * f, y0 + (y1 - y0) * f, th0 + dth * f

    def _auf_bildzeit(self, pts, scan_stamp, image_stamp):
        """Lidarpunkte vom Scan- in den Lidar-Frame zum BILDzeitpunkt drehen.

        Der Abgriff im Bild haengt allein am Azimut (``phi`` aus ``project``).
        Zwischen Bild und Scan hat sich der Roboter aber gedreht und bewegt, das
        Bild zeigt die Welt also aus einer anderen Lage. Wer den Azimut aus dem
        Scan nimmt, greift entsprechend daneben ab.

        Rueckgabe ``(pts_bild, dyaw, dtrans)``. Fehlt die Pose, kommen die
        unveraenderten Punkte und ``(0.0, 0.0)`` zurueck -- die Node laeuft dann
        wie vorher, statt mit geratenen Werten zu rechnen.
        """
        if not self.get_parameter('motion_compensation').value:
            return pts, 0.0, 0.0
        with self.pose_lock:
            puffer = list(self.pose_buf)
        p_s = self._pose_bei(scan_stamp, puffer)
        p_i = self._pose_bei(image_stamp, puffer)
        if p_s is None or p_i is None:
            self.n_komp_ohne_pose += 1
            return pts, 0.0, 0.0

        out, dth, dtrans = auf_bildzeit(
            pts, p_s, p_i,
            float(self.get_parameter('lidar_offset_x').value),
            float(self.get_parameter('lidar_offset_y').value),
            math.radians(float(self.get_parameter('lidar_yaw_deg').value)))
        self.komp_log.append((abs(dth), dtrans))
        return out, dth, dtrans

    def on_capture(self, _msg: Empty):
        self.capture_pending = True
        self.get_logger().info(
            'Capture angefordert -- naechster Scan wird als CSV + Rohbild abgelegt.')

    def on_reload(self, _msg: Empty):
        self.calib = FisheyeCalib.load(self.calib_path, _packaged_default())
        ring = float(theta_to_radius(self.calib, np.array([np.pi / 2]))[0])
        self.get_logger().info(
            f'Kalibrierung neu geladen: yaw={self.calib.yaw_deg:.2f} Grad, '
            f'Horizontring r={ring:.1f} px, '
            f'{len(self.calib.lidar_blind_sectors_deg) // 2} Blindsektoren.')

    # ---------------------------------------------------------------- #
    def _sample_z(self, rho: np.ndarray):
        """Auf welcher Hoehe (Roboter-Frame) wird der Lidar-Punkt abgegriffen?

        Bei ``horizon`` genau auf Objektivhoehe. Dann ist die Hoehendifferenz
        zur Kamera null, theta damit exakt 90 Grad und der Bildradius konstant
        f*pi/2 -- unabhaengig von der Entfernung. Es bleibt nur der Azimut, also
        eine feste Kreislinie im Bild.

        ``sample_depression_deg`` kippt den Ring nach unten. Aus der Ebene wird
        dann ein Kegel: in waagerechter Entfernung ``rho`` liegt er
        ``rho*tan(Winkel)`` unter der Linse -- fern also viel tiefer als nah.
        """
        if self.get_parameter('sample_mode').value != 'horizon':
            return self.get_parameter('sample_height_m').value

        depression = math.radians(self.get_parameter('sample_depression_deg').value)
        if depression == 0.0:
            return self.calib.cam_z
        return self.calib.cam_z - rho * math.tan(depression)

    def _zone_radien(self, rho, hoehe_unten, hoehe_oben):
        """Zwei Hoehen ueber der Lidar-Ebene -> Bildradien je Punkt.

        Der Witz an der Sache: die Dicke der Zone folgt aus der Geometrie und
        muss nicht geraten werden. Sitzt das Objektiv auf Hoehe der oberen
        Kante, ist deren Hoehendifferenz null, theta damit exakt 90 Grad und
        der Radius konstant -- die Kante laeuft als gerade Linie. Die untere
        Kante liegt tiefer, ihr theta naehert sich mit wachsender Entfernung
        von oben an 90 Grad an, ihr Radius also von aussen an den der oberen.
        Die Zone ist deshalb nah breit und fern schmal, genau wie das Objekt
        selbst im Bild.
        """
        rho = np.maximum(np.asarray(rho, dtype=float), 1e-3)
        # Ist die Zone gemessen worden ("zone" + "zonefit" in der Kalibrier-Node),
        # gilt die Messung. Sie faengt den Modellfehler am Bildrand mit auf, den
        # die Rechnung unten nicht kennen kann.
        if self.calib.zone_kalibriert:
            return self.calib.zone_radien(rho)
        dz_oben = hoehe_oben - self.calib.cam_z
        dz_unten = hoehe_unten - self.calib.cam_z
        r_innen = self.calib.focal_px * np.arctan2(rho, dz_oben)
        r_aussen = self.calib.focal_px * np.arctan2(rho, dz_unten)
        return r_innen, r_aussen

    def on_scan(self, msg: LaserScan):
        # Ratenbegrenzung ZUERST -- vor der Bildsuche und vor jeder Rechnung,
        # sonst spart der uebersprungene Scan nichts.
        rate = float(self.get_parameter('fusion_rate_hz').value)
        if rate > 0.0:
            jetzt = time.monotonic()
            # Naechster-Slot-Verfahren statt fester Mindestpause: es wird der
            # Scan genommen, der dem Zielzeitpunkt am naechsten liegt. Eine
            # feste Pause rastet sonst auf ein Vielfaches der EINGANGSperiode
            # ein -- mit 0.75/rate kamen bei 15 Hz Eingang 7.5 Hz heraus, bei
            # 10 Hz Eingang aber nur 5.0 Hz statt der gewuenschten 7.
            if self._letzter_scan:
                dt_in = jetzt - self._letzter_scan
                if 0.0 < dt_in < 1.0:
                    self._scan_periode = (dt_in if not self._scan_periode
                                          else 0.8 * self._scan_periode + 0.2 * dt_in)
            self._letzter_scan = jetzt
            periode = 1.0 / rate
            if not self._naechster_slot:
                self._naechster_slot = jetzt
            if jetzt < self._naechster_slot - 0.5 * self._scan_periode:
                self.n_rate_skip += 1
                return
            self._naechster_slot += periode
            if self._naechster_slot < jetzt:     # nach einer Luecke neu aufsetzen
                self._naechster_slot = jetzt + periode

        scan_stamp = _stamp_sec(msg.header.stamp)
        image, image_stamp, versatz = self._bild_zum_scan(scan_stamp)
        if image is None:
            self.get_logger().warn('Noch kein Kamerabild empfangen.', throttle_duration_sec=5.0)
            return
        # vor der Drop-Entscheidung mitschreiben, sonst zeigt die Statistik nur
        # die gelungenen Zuordnungen und sieht kuenstlich gut aus
        self.versatz_log.append(versatz)

        age = abs(versatz)
        if age > self.get_parameter('max_sync_age_s').value:
            self.sync_stats[1] += 1
            if self.get_parameter('sync_drop').value:
                # Faerben waere hier geraten: bei 1 rad/s Drehrate sind 0.5 s
                # bereits 29 Grad Peilfehler, die Farbe landet dann auf der
                # Bande statt auf der Pylone. Lieber diesen Scan auslassen.
                self.get_logger().warn(
                    f'Kein Bild naeher als {age:.2f} s am Scan '
                    f'({self.sync_stats[1]} von {sum(self.sync_stats) + 1} verworfen) '
                    f'-- Scan uebersprungen.',
                    throttle_duration_sec=5.0)
                return
            self.get_logger().warn(
                f'Bild ist {age:.2f} s vom Scan entfernt -- Zuordnung unsicher.',
                throttle_duration_sec=5.0)
        else:
            self.sync_stats[0] += 1

        ranges = np.asarray(msg.ranges, dtype=float)
        r_min = max(float(msg.range_min), self.get_parameter('range_min_m').value)
        r_max = min(float(msg.range_max), self.get_parameter('range_max_m').value)
        keep = np.isfinite(ranges) & (ranges >= r_min) & (ranges <= r_max)
        # Verbaute Sektoren raus (Kabel, Elektronik) -- dort misst das Lidar nur
        # sich selbst und wuerde die Kamerafarbe des eigenen Aufbaus liefern.
        _, all_angles = scan_to_points(ranges, msg.angle_min, msg.angle_increment)
        keep &= visible_mask(all_angles, self.calib.lidar_blind_sectors_deg)
        if not keep.any():
            return

        idx = np.flatnonzero(keep)
        flat, angles = scan_to_points(ranges, msg.angle_min, msg.angle_increment)
        rho = np.hypot(flat[:, 0] - self.calib.cam_x, flat[:, 1] - self.calib.cam_y)
        pts, angles = scan_to_points(ranges, msg.angle_min, msg.angle_increment,
                                     self._sample_z(rho))
        pts, angles, rho = pts[keep], angles[keep], rho[keep]

        # Projiziert wird die BILDZEIT-Geometrie, veroeffentlicht die des Scans:
        # das Bild zeigt die Welt aus der Lage von vor bis zu 0.7 s, die Wolke
        # soll aber dort liegen, wo das Lidar gerade gemessen hat.
        pts_bild, _dyaw, _dtrans = self._auf_bildzeit(pts, scan_stamp, image_stamp)
        u, v, theta, phi, in_fov = project(self.calib, pts_bild)
        rho_bild = np.hypot(pts_bild[:, 0] - self.calib.cam_x,
                            pts_bild[:, 1] - self.calib.cam_y)
        height, width = image.shape[:2]
        on_image = in_fov & (u >= 0) & (u < width) & (v >= 0) & (v < height)
        if not on_image.any():
            self.get_logger().warn(
                'Kein Lidar-Punkt landet im Bild -- Kalibrierung pruefen.',
                throttle_duration_sec=5.0)
            return

        idx, pts, angles, rho = idx[on_image], pts[on_image], angles[on_image], rho[on_image]
        rho_bild = rho_bild[on_image]
        u, v, theta, phi = u[on_image], v[on_image], theta[on_image], phi[on_image]

        # Bandbreite in px: +-sample_band_m Pylonenhoehe, aus der Entfernung
        # umgerechnet. Fern schrumpft das Band von selbst mit, bleibt also
        # automatisch innerhalb der Pylone.
        band_m = self.get_parameter('sample_band_m').value
        band_px = None
        if band_m > 0.0:
            band_px = self.calib.focal_px * np.arctan(band_m / np.maximum(rho_bild, 1e-3))

        zone_low = self.get_parameter('sample_zone_low_m').value
        zone_high = self.get_parameter('sample_zone_high_m').value
        if zone_high > zone_low or self.calib.zone_kalibriert:
            # Die beiden Zonengrenzen je Punkt in Bildradien umrechnen. Hoehere
            # Kante = kleinerer Radius (radial nach aussen heisst nach unten).
            fix_in = float(self.get_parameter('sample_r_fix_in').value)
            fix_out = float(self.get_parameter('sample_r_fix_out').value)
            if fix_in > 0.0 and fix_out > fix_in:
                r_innen = np.full(rho_bild.shape, fix_in)
                r_aussen = np.full(rho_bild.shape, fix_out)
            else:
                r_innen, r_aussen = self._zone_radien(rho_bild, zone_low, zone_high)
            r_min = float(self.get_parameter('sample_r_min_px').value)
            if r_min > 0.0:
                r_innen = np.maximum(r_innen, r_min)
                r_aussen = np.maximum(r_aussen, r_min + 2.0)
            if self.get_parameter('zone_from_band').value:
                r_innen, r_aussen = self._zone_aus_bande(image, phi, r_innen, r_aussen)
            labels, bgr, hsv = colors.classify_zone(
                image, phi, r_innen, r_aussen,
                center=(self.calib.cx, self.calib.cy),
                min_frac=self.get_parameter('sample_zone_min_frac').value,
                ranges=self._aktive_ranges(),
                steps=self.get_parameter('sample_zone_steps').value,
                nutz_anteil=self.get_parameter('sample_zone_nutz').value,
                adaptiv_faktor=self.get_parameter('sample_zone_adaptiv').value,
                adaptiv_grad=self.get_parameter('sample_zone_adaptiv_grad').value,
                rg_z_min=float(self.get_parameter('rg_z_min').value),
                rg_s_min=int(self.get_parameter('rg_s_min').value),
                rg_d_min=int(self.get_parameter('rg_d_min').value))
        else:
            bgr, hsv = colors.sample_colors(
                image, u, v, self.get_parameter('patch_px').value,
                center=(self.calib.cx, self.calib.cy), band_px=band_px,
                band_count=self.get_parameter('sample_band_count').value)
            labels = colors.classify_hsv(hsv, self._aktive_ranges())

        if self.capture_pending:
            self._capture_image = image
        self._publish_summary(labels)
        # Die Punktwolke haengt NICHT mehr an 'debug': scan_processor_node liest
        # /camera_lidar/colored_scan und baut daraus die Hinderniserkennung.
        # Mit debug:=false fiel sie vorher stillschweigend weg -- und damit die
        # Hindernisse. Abschalten geht weiterhin gezielt ueber publish_cloud.
        if self.get_parameter('publish_cloud').value:
            self._publish_cloud(msg.header, pts, bgr, labels)
        if self.get_parameter('debug').value and self.get_parameter('publish_debug_image').value:
            self._publish_debug(image, u, v, labels, bgr, rho_bild)

        self._write_csv(scan_stamp, idx, angles, np.linalg.norm(pts[:, :2], axis=1),
                        pts, u, v, theta, phi, bgr, hsv, labels)

    # ---------------------------------------------------------------- #
    def _publish_summary(self, labels):
        counts = {}
        for label in labels:
            counts[label] = counts.get(label, 0) + 1
        text = ' '.join(f'{k}={v}' for k, v in sorted(counts.items()))
        self.pub_summary.publish(String(data=f'{len(labels)} Punkte: {text}'))

    def _publish_cloud(self, header, pts, bgr, labels):
        """Punktwolke mit RGB. Farbe je nach ``cloud_color_mode``: kraeftige
        Label-Farbe (Default) oder die gemessene Pixelfarbe."""
        if self.get_parameter('cloud_color_mode').value == 'label':
            bgr = colors.label_colors(labels)
        packed = ((bgr[:, 2].astype(np.uint32) << 16)
                  | (bgr[:, 1].astype(np.uint32) << 8)
                  | bgr[:, 0].astype(np.uint32))
        rgb = packed.view(np.float32)
        # Direkt als numpy-Array uebergeben. Mit .tolist() landet create_cloud im
        # Zweig "Cast python objects to structured NumPy array (slow)" und baut
        # je Punkt ein Tupel -- am Aufbau gemessen 1.50 ms gegen 0.11 ms.
        cloud_points = np.column_stack([pts.astype(np.float32), rgb]).astype(np.float32)
        self.pub_cloud.publish(point_cloud2.create_cloud(header, CLOUD_FIELDS, cloud_points))

    def _publish_debug(self, image, u, v, labels, bgr, rho):
        now = self.get_clock().now().nanoseconds * 1e-9
        rate = self.get_parameter('debug_rate_hz').value
        if rate > 0 and now - self.last_debug_stamp < 1.0 / rate:
            return
        self.last_debug_stamp = now

        canvas = image.copy()
        center = (int(round(self.calib.cx)), int(round(self.calib.cy)))
        cv2.circle(canvas, center, int(round(self.calib.radius_px)), (255, 255, 0), 2)
        # Horizontring zur Kontrolle: liegt er auf Hoehe der Pylonen?
        ring = self.calib.focal_px * math.pi / 2.0
        mode = self.get_parameter('sample_mode').value
        z_lo = self.get_parameter('sample_zone_low_m').value
        z_hi = self.get_parameter('sample_zone_high_m').value
        zone_an = z_hi > z_lo or self.calib.zone_kalibriert
        if zone_an:
            r_innen, r_aussen = self._zone_radien(rho, z_lo, z_hi)
        else:
            r_innen = r_aussen = None
        winkel_pkt = np.arctan2(np.asarray(v) - self.calib.cy,
                                np.asarray(u) - self.calib.cx)
        # Der Horizontring als Referenz -- bei height liegt der Abgriff NICHT
        # darauf, dann ist er nur die Marke fuer "Objektivhoehe".
        cv2.circle(canvas, center, int(round(ring)), (0, 90, 160), 1)

        if zone_an and len(u):
            # Die Zone so zeichnen, wie sie tatsaechlich liegt: je eine
            # Polylinie durch die inneren und die aeusseren Kanten. Die innere
            # laeuft fast kreisrund, die aeussere wandert mit der Entfernung --
            # genau daran sieht man, ob die Zone die Bande abdeckt.
            reihe = np.argsort(winkel_pkt)
            # In den Blindsektoren fehlen Punkte. Ohne Unterbrechung zoege die
            # Polylinie eine Sehne quer durchs Bild.
            luecke = np.diff(winkel_pkt[reihe]) > math.radians(5.0)
            grenzen = np.flatnonzero(luecke) + 1
            cosw, sinw = np.cos(winkel_pkt[reihe]), np.sin(winkel_pkt[reihe])
            for radien, farbe in ((r_innen, (0, 200, 255)), (r_aussen, (0, 140, 255))):
                pu = self.calib.cx + radien[reihe] * cosw
                pv = self.calib.cy + radien[reihe] * sinw
                punkte = np.column_stack([pu, pv]).astype(np.int32)
                for teil in np.split(punkte, grenzen):
                    if len(teil) >= 2:
                        cv2.polylines(canvas, [teil.reshape(-1, 1, 2)], False, farbe, 1)

        # Alles punktweise Gezeichnete vektorisiert: erst die Abgriffsegmente
        # (ein polylines-Aufruf je Label), dann die Messfarbe als gefuellte
        # Scheibe und die Labelfarbe als Ring -- je ein numpy-Zugriff.
        labels_arr = np.asarray(labels)
        if zone_an and len(u):
            cosw, sinw = np.cos(winkel_pkt), np.sin(winkel_pkt)
            for name in ('rot', 'gruen', 'magenta'):
                m = labels_arr == name
                if not m.any():
                    continue
                _segmente(canvas,
                          self.calib.cx + r_innen[m] * cosw[m],
                          self.calib.cy + r_innen[m] * sinw[m],
                          self.calib.cx + r_aussen[m] * cosw[m],
                          self.calib.cy + r_aussen[m] * sinw[m],
                          colors.LABEL_BGR.get(name, (255, 255, 255)))
        if len(u):
            _scheiben(canvas, u, v, np.asarray(bgr), 4)
            lab_farben = np.array([colors.LABEL_BGR.get(l, (255, 255, 255))
                                   for l in labels], dtype=np.uint8)
            _scheiben(canvas, u, v, lab_farben, 4, ring=True)

        # Gefundene Bandenunterkante als durchgehende Linie. Loecher (NaN) sind
        # Stellen, an denen keine Kante gefunden wurde -- dort steht etwas davor
        # oder es gibt keine Bande. Die Linie bricht dort ab statt zu raten.
        band_winkel = band_kante = None
        if self.get_parameter('band_detect').value:
            band_winkel, band_kante = self._bande_finden(image)
            if band_kante is not None:
                gute = np.isfinite(band_kante)
                if gute.any():
                    pu = self.calib.cx + band_kante * np.cos(band_winkel)
                    pv = self.calib.cy + band_kante * np.sin(band_winkel)
                    zuege = [np.rint(np.column_stack([pu[b], pv[b]])).astype(np.int32)
                             for b in _stuecke(band_kante)]
                    if zuege:
                        cv2.polylines(canvas, zuege, False, (255, 0, 255), 2)

        # Kopfzeile: welcher Modus, und was ist dabei herausgekommen?
        zaehl = {}
        for label in labels:
            zaehl[label] = zaehl.get(label, 0) + 1
        kopf = f'{mode}'
        if zone_an:
            if self.get_parameter('zone_from_band').value:
                woher = 'zone: Bande live'
            elif self.calib.zone_kalibriert:
                woher = 'zone: kalibriert'
            else:
                woher = f'zone {z_lo * 100:.0f}..{z_hi * 100:.0f} cm gerechnet'
            adaptiv = self.get_parameter('sample_zone_adaptiv').value
            if adaptiv > 0.0:
                woher += f'  adaptiv x{adaptiv:.1f}'
            nutz = self.get_parameter('sample_zone_nutz').value
            kopf += (f'  {woher}  '
                     f'>={self.get_parameter("sample_zone_min_frac").value * 100:.0f} %'
                     + (f'  mitte {nutz * 100:.0f} %' if nutz < 0.999 else ''))
        kopf += f'  |  {len(labels)} Punkte'
        if band_kante is not None:
            gute = int(np.isfinite(band_kante).sum())
            kopf += (f'  |  Bande {gute}/{len(band_kante)} Azimute'
                     + (f', r {np.nanmin(band_kante):.0f}..{np.nanmax(band_kante):.0f}'
                        if gute else ''))
        cv2.putText(canvas, kopf, (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        spalte = 12
        for name in ('rot', 'gruen', 'magenta', 'schwarz', 'unbekannt'):
            if name not in zaehl:
                continue
            text = f'{name}={zaehl[name]}'
            cv2.putText(canvas, text, (spalte, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        colors.LABEL_BGR.get(name, (255, 255, 255)), 2)
            spalte += 22 + 13 * len(text)

        # Wo liegt "vorne"? Hilft beim Beurteilen der Yaw-Kalibrierung.
        front_u, front_v, _, _, _ = project(self.calib, np.array([[1.0, 0.0, 0.0]]))
        cv2.arrowedLine(canvas,
                        (int(round(self.calib.cx)), int(round(self.calib.cy))),
                        (int(round(front_u[0])), int(round(front_v[0]))),
                        (0, 255, 255), 2, tipLength=0.08)
        cv2.putText(canvas, 'vorne (+X)',
                    (int(round(front_u[0])) + 6, int(round(front_v[0]))),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        if self.get_parameter('debug_polar').value:
            streifen = self._polar_view(image, u, v, labels, r_innen, r_aussen,
                                        canvas.shape[1], band_winkel, band_kante)
            if streifen is not None:
                canvas = np.vstack([canvas, streifen])

        out = self.bridge.cv2_to_imgmsg(canvas, 'bgr8')
        out.header.frame_id = 'camera'
        self.pub_debug.publish(out)

    def _aktive_ranges(self):
        """Nur die Farben, die gerade gesucht werden sollen."""
        aktiv = set(self.get_parameter('active_labels').value)
        gefiltert = {k: v for k, v in self.ranges.items() if k in aktiv}
        return gefiltert or self.ranges

    def _zone_aus_bande(self, image, phi, r_innen, r_aussen):
        """Zone an die live gefundene Bandenunterkante haengen.

        Die Oberkante bleibt, wo sie ist: das Objektiv sitzt auf ihrer Hoehe,
        also ist ihr Bildradius konstant und unabhaengig von der Entfernung --
        da gibt es nichts zu suchen. Die Unterkante dagegen wandert mit der
        Entfernung und wird deshalb im Bild gemessen statt gerechnet.

        Wo keine Kante gefunden wurde (etwas steht davor, oder die Bande fehlt),
        bleibt der Wert aus der Kalibrierkurve stehen -- die Bandensuche
        verbessert also nur, wo sie etwas gefunden hat, und verschlechtert nie.
        """
        winkel, kante = self._bande_finden(image)
        if kante is None:
            return r_innen, r_aussen
        gute = np.isfinite(kante)
        if gute.sum() < 8:
            return r_innen, r_aussen
        # Zyklisch interpolieren: fuer jeden Punkt die Kante in SEINER Richtung
        w = np.concatenate([winkel[gute] - 2 * math.pi, winkel[gute],
                            winkel[gute] + 2 * math.pi])
        k = np.tile(kante[gute], 3)
        aussen_neu = np.interp(np.asarray(phi), w, k)
        # Nur uebernehmen, wo die Interpolation nicht ueber eine grosse Luecke
        # gemittelt hat -- sonst zoege eine Fehlstelle die Zone quer durchs Bild.
        naechste = np.min(np.abs(np.asarray(phi)[:, None] - w[None, :]), axis=1)
        brauchbar = naechste < math.radians(4.0)
        r_aussen = np.where(brauchbar, aussen_neu, r_aussen)
        return r_innen, np.maximum(r_aussen, r_innen + 2.0)

    def _bande_finden(self, image):
        # Je Bild nur einmal rechnen: bei zone_from_band braucht sie jeder Scan,
        # das Debug-Bild noch einmal. Ohne Cache liefe sie doppelt.
        kennung = id(image), image.shape
        if getattr(self, '_band_cache', (None,))[0] == kennung:
            return self._band_cache[1], self._band_cache[2]
        winkel, kante = self._bande_suchen(image)
        self._band_cache = (kennung, winkel, kante)
        return winkel, kante

    def _bande_suchen(self, image):
        """Sucht je Azimut die Unterkante der schwarzen Bande.

        Von innen nach aussen laufen und die erste Stelle nehmen, an der es
        dauerhaft hell wird -- das ist der Uebergang Bande -> Matte. "Dauerhaft"
        heisst ``band_run`` Pixel am Stueck, damit ein einzelner Glanzpunkt auf
        der Bande die Kante nicht vorzeitig ausloest. Davor muss mindestens ein
        dunkles Pixel gelegen haben, sonst war da gar keine Bande.

        Warum die UNTERkante und nicht die obere: dahinter liegt immer die
        Matte, also derselbe Kontrast in jeder Richtung. Hinter der Oberkante
        liegt dagegen der halbe Raum -- mal weisse Wand, mal dunkle Couch. An
        1362 Kantenpaaren gemessen: RMS 5.3 px unten gegen 12.8 px oben.

        Rueckgabe: (winkel, radien) -- Radien sind NaN, wo keine Kante gefunden
        wurde (etwa wo etwas vor der Bande steht oder sie ganz fehlt).
        """
        schritte = max(int(self.get_parameter('band_steps').value), 8)
        r_von = float(self.get_parameter('band_r_min').value)
        r_bis = float(self.get_parameter('band_r_max').value) or float(self.calib.radius_px)
        dunkel_max = int(self.get_parameter('band_dark_max').value)
        hell_min = int(self.get_parameter('band_bright_min').value)
        lauf = max(int(self.get_parameter('band_run').value), 1)
        if r_bis - r_von < lauf + 2:
            return None, None

        hoehe, breite = image.shape[:2]
        grau = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)[..., 2]
        winkel = np.linspace(-math.pi, math.pi, schritte, endpoint=False)
        radien = np.arange(r_von, r_bis)
        uu = np.clip(np.rint(self.calib.cx + radien[None, :] * np.cos(winkel)[:, None]),
                     0, breite - 1).astype(int)
        vv = np.clip(np.rint(self.calib.cy + radien[None, :] * np.sin(winkel)[:, None]),
                     0, hoehe - 1).astype(int)
        profil = grau[vv, uu].astype(np.int16)          # (Azimut, Radius)

        hell = profil >= hell_min
        # Wieviele helle Pixel liegen in einem Fenster der Laenge lauf?
        summe = np.cumsum(np.concatenate(
            [np.zeros((schritte, 1), int), hell.astype(int)], axis=1), axis=1)
        voll = (summe[:, lauf:] - summe[:, :-lauf]) >= lauf
        # Vor der Kante muss es dunkel gewesen sein
        dunkel_davor = np.cumsum((profil <= dunkel_max).astype(int), axis=1)[:, :voll.shape[1]] > 0
        treffer = voll & dunkel_davor

        gefunden = treffer.any(axis=1)
        kante = np.full(schritte, np.nan)
        kante[gefunden] = radien[np.argmax(treffer[gefunden], axis=1)]

        # --- Ausreisser raus ---------------------------------------------- #
        # Die Kante darf nicht innerhalb der Oberkante liegen: die Bande ist
        # rund 10 cm hoch, ihre Unterkante also immer ein Stueck WEITER AUSSEN
        # als die Oberkante (radial nach aussen heisst nach unten).
        oben = self.calib.zone_r0_in if self.calib.zone_kalibriert else \
            self.calib.focal_px * math.pi / 2.0
        kante[kante < oben + float(self.get_parameter('band_min_dicke').value)] = np.nan

        # Nachbarazimute muessen sich aehneln -- die Bande springt nicht. Der
        # Median ueber ein Fenster ist der robuste Erwartungswert; wer zu weit
        # davon abweicht, ist eine Fehldetektion (meist ein Glanzpunkt in der
        # Bande, der die Kante zu frueh ausloest).
        fenster = max(int(self.get_parameter('band_smooth').value), 1)
        if fenster > 1 and np.isfinite(kante).sum() >= fenster:
            halb = fenster // 2
            # zyklisch, der Azimut laeuft ja rundum. Vektorisiert ueber ein
            # Gleitfenster -- als Python-Schleife kostete genau das hier 17.7 ms
            # von 24 ms Gesamtlaufzeit, so sind es 1.1 ms.
            breit = np.concatenate([kante[-halb:], kante, kante[:halb]])
            with np.errstate(all='ignore'):
                glatt = np.nanmedian(sliding_window_view(breit, 2 * halb + 1), axis=1)
            grenze = float(self.get_parameter('band_max_dev').value)
            daneben = np.isfinite(kante) & np.isfinite(glatt) & (np.abs(kante - glatt) > grenze)
            kante[daneben] = np.nan
        return winkel, kante

    def _polar_view(self, image, u, v, labels, r_innen, r_aussen, breite,
                    band_winkel=None, band_kante=None):
        """Entzerrter Streifen: Azimut waagerecht, Bildradius senkrecht.

        Im runden Fisheye liegt alles Interessante am aeusseren Rand und ist
        dort auf wenige Pixel zusammengedraengt. Aufgerollt wird daraus ein
        Band, in dem die Schichten sauber uebereinander liegen: oben der Raum,
        darunter die schwarze Bande, ganz unten die Matte. Der Abgriff ist als
        Punktreihe eingezeichnet, die Zonengrenzen als duenne Linien -- damit
        ist sofort zu sehen, ob der Abgriff auf der Bande sitzt.
        """
        hoehe = int(self.get_parameter('debug_polar_height').value)
        if hoehe < 20 or not len(u):
            return None
        ring = self.calib.focal_px * math.pi / 2.0
        # Fenster um die Punkte legen, nicht um den Bildkreis: bei height haben
        # nahe Punkte kleine Radien, ein Fenster am Minimum rutscht viel zu weit
        # nach innen. Die Perzentile lassen einzelne Ausreisser aussen vor.
        rad_pkt = np.hypot(np.asarray(u) - self.calib.cx, np.asarray(v) - self.calib.cy)
        alle = rad_pkt if r_innen is None else np.concatenate([r_innen, r_aussen, rad_pkt])
        r_lo = max(0.0, float(np.percentile(alle, 2)) - 15.0)
        r_hi = min(float(self.calib.radius_px), float(np.percentile(alle, 98)) + 20.0)
        if r_hi - r_lo < 20.0:
            return None

        # Direkt in der Zielgroesse abtasten statt warpPolar ueber den ganzen
        # Bildkreis (Zwischenbild 1280 x r_hi) und anschliessend zu croppen,
        # transponieren und zu skalieren. Die Abtasttabelle haengt nur an
        # (r_lo, r_hi, Groesse) und wird deshalb wiederverwendet.
        schluessel = (round(r_lo, 1), round(r_hi, 1), breite, hoehe,
                      round(self.calib.cx, 1), round(self.calib.cy, 1))
        if self._polar_map is None or self._polar_map[0] != schluessel:
            winkel_sp = np.linspace(0.0, 2 * math.pi, breite, endpoint=False)
            radius_sp = np.linspace(r_lo, r_hi, hoehe)
            map_x = (self.calib.cx
                     + radius_sp[:, None] * np.cos(winkel_sp)[None, :]).astype(np.float32)
            map_y = (self.calib.cy
                     + radius_sp[:, None] * np.sin(winkel_sp)[None, :]).astype(np.float32)
            self._polar_map = (schluessel, map_x, map_y)
        _, map_x, map_y = self._polar_map
        streifen = cv2.remap(image, map_x, map_y, cv2.INTER_NEAREST,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
        skala = (hoehe - 1) / max(r_hi - r_lo, 1.0)

        winkel = np.arctan2(np.asarray(v) - self.calib.cy,
                            np.asarray(u) - self.calib.cx) % (2 * math.pi)
        x_sp = np.rint(winkel / (2 * math.pi) * breite).astype(np.int32) % breite
        if r_innen is not None:
            for rr, fb in ((r_innen, (0, 200, 255)), (r_aussen, (0, 140, 255))):
                yy = np.rint((np.asarray(rr) - r_lo) * skala).astype(np.int32)
                m = (yy >= 0) & (yy < hoehe)
                streifen[yy[m], x_sp[m]] = fb
        y_sp = np.rint((np.asarray(rad_pkt) - r_lo) * skala).astype(np.int32)
        m = (y_sp >= 0) & (y_sp < hoehe)
        if m.any():
            farben = np.array([colors.LABEL_BGR.get(l, (255, 255, 255)) for l in labels],
                              dtype=np.uint8)
            _scheiben(streifen, x_sp[m], y_sp[m], farben[m], 1)

        # Die gefundene Bandenunterkante -- im entzerrten Streifen laeuft sie als
        # Kurve, an der man sofort sieht, ob die Abgriffszone darauf sitzt.
        if band_kante is not None and band_winkel is not None:
            gut = np.isfinite(band_kante)
            if gut.any():
                xb = (np.rint((np.asarray(band_winkel)[gut] % (2 * math.pi))
                              / (2 * math.pi) * breite).astype(np.int32) % breite)
                yb = np.rint((np.asarray(band_kante)[gut] - r_lo) * skala).astype(np.int32)
                mb = (yb >= 0) & (yb < hoehe)
                if mb.any():
                    _scheiben(streifen, xb[mb], yb[mb],
                              np.tile(np.uint8([255, 0, 255]), (int(mb.sum()), 1)), 1)

        # Radiusskala und die Marke fuer den Horizontring
        schritt = 10 if (r_hi - r_lo) < 120 else 20
        for r in range(int(r_lo) - int(r_lo) % schritt + schritt, int(r_hi), schritt):
            y = int(round((r - r_lo) * skala))
            if 0 <= y < hoehe:
                cv2.line(streifen, (0, y), (10, y), (200, 200, 200), 1)
                cv2.putText(streifen, str(r), (13, y + 4), cv2.FONT_HERSHEY_SIMPLEX,
                            0.35, (200, 200, 200), 1)
        y_ring = int(round((ring - r_lo) * skala))
        if 0 <= y_ring < hoehe:
            cv2.line(streifen, (breite - 60, y_ring), (breite - 1, y_ring), (0, 90, 160), 1)
            cv2.putText(streifen, 'Horizont', (breite - 130, y_ring + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 90, 160), 1)
        cv2.putText(streifen, 'entzerrt: Azimut ->, Radius v', (breite // 2 - 110, 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        return streifen

    # ---------------------------------------------------------------- #
    def _rows(self, stamp, idx, angles, dists, pts, u, v, theta, phi, bgr, hsv, labels):
        only_labeled = self.get_parameter('csv_only_labeled').value
        for i in range(len(idx)):
            if only_labeled and labels[i] in ('unbekannt', 'schwarz'):
                continue
            yield [
                f'{stamp:.6f}', int(idx[i]), f'{np.degrees(angles[i]):.3f}',
                f'{dists[i]:.4f}', f'{pts[i, 0]:.4f}', f'{pts[i, 1]:.4f}', f'{pts[i, 2]:.4f}',
                f'{u[i]:.2f}', f'{v[i]:.2f}', f'{np.degrees(theta[i]):.3f}',
                f'{np.degrees(phi[i]):.3f}',
                int(bgr[i, 0]), int(bgr[i, 1]), int(bgr[i, 2]),
                int(hsv[i, 0]), int(hsv[i, 1]), int(hsv[i, 2]), labels[i],
            ]

    def _write_csv(self, *args):
        mode = self.get_parameter('csv_mode').value
        if mode == 'continuous' and self.continuous_writer:
            handle, writer = self.continuous_writer
            writer.writerows(self._rows(*args))
            handle.flush()
            return
        if mode != 'trigger' or not self.capture_pending:
            return

        self.capture_pending = False
        os.makedirs(self.csv_dir, exist_ok=True)
        tag = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        # Das ROHBILD mit ablegen -- unveraendert, ohne Overlay. Ohne das laesst
        # sich die radiale Abtastung nicht nachrechnen: die CSV enthaelt nur die
        # bereits abgetastete Medianfarbe je Punkt, nicht das Profil dahinter.
        if self._capture_image is not None:
            bild = os.path.join(self.csv_dir, f'rohbild_{tag}.png')
            try:
                cv2.imwrite(bild, self._capture_image)
                self.get_logger().info(f'Rohbild abgelegt -> {bild}')
            except Exception as exc:  # noqa: BLE001
                self.get_logger().warn(f'Rohbild nicht schreibbar: {exc}')
            self._capture_image = None
        path = os.path.join(self.csv_dir, f'lidar_pixels_{tag}.csv')
        rows = list(self._rows(*args))
        with open(path, 'w', newline='') as handle:
            writer = csv.writer(handle)
            writer.writerow(CSV_HEADER)
            writer.writerows(rows)
        # Kalibrierung mitschreiben, damit die CSV spaeter nachvollziehbar bleibt.
        self.calib.to_yaml(os.path.join(self.csv_dir, f'lidar_pixels_{tag}_calib.yaml'))
        self.get_logger().info(f'{len(rows)} Punkte geschrieben -> {path}')

    def _open_continuous_csv(self):
        os.makedirs(self.csv_dir, exist_ok=True)
        tag = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        path = os.path.join(self.csv_dir, f'lidar_pixels_{tag}_continuous.csv')
        handle = open(path, 'w', newline='')
        writer = csv.writer(handle)
        writer.writerow(CSV_HEADER)
        self.continuous_writer = (handle, writer)
        self.get_logger().info(f'Schreibe fortlaufend nach {path}')

    def destroy_node(self):
        if self.continuous_writer:
            self.continuous_writer[0].close()
            self.continuous_writer = None
        return super().destroy_node()


def _stamp_sec(stamp) -> float:
    return stamp.sec + stamp.nanosec * 1e-9


def auf_bildzeit(pts, pose_scan, pose_bild, off_x=0.110, off_y=0.0,
                 lidar_yaw=math.pi):
    """Punkte aus dem Lidar-Frame der Scanzeit in den der Bildzeit drehen.

    ``pose_*`` sind ``(x, y, yaw)`` von base_link in der Welt. ``off_*`` ist der
    Lidar-Ursprung in base_link, ``lidar_yaw`` seine Verdrehung (Sensor haengt
    um 180 Grad gedreht, daher der Default).

    Herleitung: der Punkt steht in der Welt fest.
        W        = o_s + R(a_s) * P_scan
        P_bild   = R(a_i)^T * (W - o_i)
                 = R(a_s - a_i) * P_scan + R(a_i)^T * (o_s - o_i)
    mit ``o`` dem Lidar-Ursprung in der Welt und ``a = yaw + lidar_yaw``. Der
    Versatz base_link -> Lidar dreht beim Gieren mit, deshalb steckt er in ``o``
    und nicht einfach in der base_link-Verschiebung.

    Rueckgabe ``(pts_bild, dyaw, dtrans)``; ``dyaw`` ist die Drehung des
    Roboters zwischen Bild und Scan, ``dtrans`` der Betrag der Verschiebung im
    Lidar-Frame.
    """
    xs, ys, th_s = pose_scan
    xi, yi, th_i = pose_bild
    dth = math.atan2(math.sin(th_s - th_i), math.cos(th_s - th_i))

    ox_s = xs + math.cos(th_s) * off_x - math.sin(th_s) * off_y
    oy_s = ys + math.sin(th_s) * off_x + math.cos(th_s) * off_y
    ox_i = xi + math.cos(th_i) * off_x - math.sin(th_i) * off_y
    oy_i = yi + math.sin(th_i) * off_x + math.cos(th_i) * off_y

    # R(a_i)^T * R(a_s) = R(a_s - a_i), und a_s - a_i ist genau dyaw: dreht sich
    # der Roboter zwischen Bild und Scan um +dyaw, dann lag derselbe Weltpunkt
    # im Bild um +dyaw weiter herum.
    a_i = th_i + lidar_yaw
    ca, sa = math.cos(dth), math.sin(dth)

    wx, wy = ox_s - ox_i, oy_s - oy_i
    ci, si = math.cos(a_i), math.sin(a_i)
    tx = ci * wx + si * wy
    ty = -si * wx + ci * wy

    pts = np.asarray(pts, dtype=float)
    out = np.empty_like(pts)
    out[:, 0] = ca * pts[:, 0] - sa * pts[:, 1] + tx
    out[:, 1] = sa * pts[:, 0] + ca * pts[:, 1] + ty
    if out.shape[1] > 2:
        out[:, 2] = pts[:, 2]
    return out, dth, math.hypot(tx, ty)


def _packaged_default() -> str:
    try:
        from ament_index_python.packages import get_package_share_directory
        return os.path.join(get_package_share_directory('camera_lidar_fusion'),
                            'config', 'fisheye_calib.yaml')
    except Exception:  # noqa: BLE001
        return ''


def main(args=None):
    rclpy.init(args=args)
    node = LidarPixelMapper()
    # Vier Threads: Scan-Rechnung, Bildannahme, Odometrie und die kleinen
    # Dienst-Topics laufen nebeneinander. Mit rclpy.spin() (ein Thread)
    # blockierte die Scan-Rechnung die Bildannahme, wodurch das Bild zum Scan
    # alterte -- dasselbe wuerde sonst dem Posenpuffer passieren.
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
