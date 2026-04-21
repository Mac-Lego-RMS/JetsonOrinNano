#!/usr/bin/env python3

from platform import node

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan, Imu, Image
from geometry_msgs.msg import Twist, Point
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import String
from rclpy.qos import qos_profile_sensor_data
import math

# YOLO Imports 
from cv_bridge import CvBridge
import cv2
from ultralytics import YOLO
import numpy as np

from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped

import time
import json
import os


'''
=============================================================
      📍 DEIN HARDWARE-KOORDINATENSYSTEM (Lidar & Foxglove)
=============================================================

                          +Y (VORNE)
                              ^
                              |  Lidar: 180°
                              |  (Fahrtrichtung)
                              |
                              |
    (LINKS)                   |                   (RECHTS)
      -X  <--------------- [ 🤖 ] --------------->  +X
    Lidar: 270°               |                   Lidar: 90°
                              |
                              |
                              |
                              v
                          -Y (HINTEN)
                       Lidar: 360° / 0°

-------------------------------------------------------------
📐 MATHE-REGELN FÜR DIESEN NODE:
- Willst du die Karotte weiter nach VORNE schieben  -> Y wird größer (+Y)
- Willst du die Karotte weiter nach RECHTS schieben -> X wird größer (+X)
- Willst du die Karotte weiter nach LINKS schieben  -> X wird kleiner (-X)

🦊 FOXGLOVE ANSICHT (Top-Down):
- Oben auf dem Bildschirm  = +Y
- Rechts auf dem Bildschirm = +X
============================================================='''






class WallFollower(Node):
    def __init__(self):
        super().__init__('wall_follower')
        
        # Subscriber für LiDAR-Daten 
        self.sub_scan = self.create_subscription(LaserScan, '/ldlidar_node/scan', self.scan_callback, qos_profile_sensor_data)
        self.last_point_data = []  # Hier speichern wir die rohen LiDAR-Punkte für die Kamera-Fusion
        self.sub_imu = self.create_subscription(Imu, '/bno055/imu', self.imu_callback, 10)
        
        # Publisher für Bewegung und RViz [cite: 1, 19]
        self.pub_cmd_vel = self.create_publisher(Twist, '/cmd_vel', 10)
        self.pub_markers = self.create_publisher(MarkerArray, '/wall_follower_markers', 10)

        # Publisher für den Bézier-Pfad in Foxglove
        self.pub_path = self.create_publisher(Path, '/planned_trajectory', 10)
        

        self.yaw_offset = 0.0
        self.current_yaw = 0.0
        self.imu_ready = False  # <--- NEU: Ist der Gyro schon wach?
        self.target_yaw = 0.0      # Zielwinkel für die Kurve
        self.last_raw_yaw = None
        self.start_turn_yaw = None
        self.start_straight_yaw = 0.0
        
        
        # Konfiguration
        self.rviz_frame = 'ldlidar_link'  # Muss in RViz als "Fixed Frame" stehen
        self.get_logger().info('>>> WallFollower Template gestartet. Warte auf LiDAR... <<<')

        # --- STATE MACHINE & PFADPLANUNG ---
        self.state = 'STARTING'    # Startzustand
        self.fahrtrichtung = None     # Wird automatisch erkannt

        self.target_turns = 4
        self.turn_count = 0
        self.locked_turn_count = 0  # Merkt sich, in welcher "Runde" das Hindernis stand

        self.front_wall = None
        self.left_wall = None
        self.right_wall = None
        
        # Karotten-Parameter Geradeausfahrt
        self.lookahead_dist_straight = 0.60    # Wie weit schaut der Roboter voraus? (60 cm)
        self.min_wall_dist = 0.10       

        self.lane_ratio = 0.85       # Verhältnis des Bandenabstands innen zu außen Außen Bande: 0.85, Innen Bande: 0.20
        self.assumed_lane_width = 1.0 # Wenn eine Wand fehlt, gehen wir von 60cm Spurbreite aus
        self.turn_exit_angle = 25
        self.max_wall_lenght_for_turn = 0.25

        # Object Detection Parameter
        self.current_obstacle_cmd = "CLEAR"

        # Standard-Werte für die Ideallinie auf der Geraden(z.B. Außenbahn)
        self.default_lane_ratio = 0.85 
        self.default_max_turn = 0.8
        

        # --- PID-REGLER PARAMETER ---
        self.kp = 3.5   # Lenkt hart zur Karotte
        self.kd = 0   # Verhindert das Schlingern (Dämpfung)
        self.ki = 0.0   # Integral (oft bei WRO auf 0 gelassen, da schnelle Spurwechsel)
        
        self.prev_error = 0.0
        self.integral_error = 0.0

        # Karotten-Parameter Kurve
        self.lookahead_dist_turn = 0.20
        self.start_turn_dist = 0
        self.target_point = (0, 0)

        # --- TURN PID-REGLER PARAMETER ---
        self.turn_kp = 0.6
        self.turn_kd = 0.0
        self.turn_ki = 0.0

        # --- MOTOR PARAMETER (ESP PWM 0 - 1023) ---
        self.base_speed = 250.0  # Normale Geschwindigkeit auf der Geraden
        self.turn_speed = 250.0  # Leicht reduzierter Speed in der Kurve

        self.max_turn_angle = 0.635  # Maximaler Lenkwinkel in Grad (für Sicherheit)    Min Außen 0.435, Max Innen 0.800

        # --------------------------------
        # --- YOLO - Global Parameters ---
        # --------------------------------

        self.bridge = CvBridge()
        
        # 1. Das YOLO-Modell laden (TensorRT .engine)
        self.get_logger().info('Lade YOLO TensorRT Engine...')
        #self.model = YOLO('/workspace/best.engine', task='detect')
        self.get_logger().info('Modell erfolgreich geladen!')
        # In der __init__ Klasse hinzufügen:
        self.angle_calibration = 0.0  # In Grad: Korrigiert, wenn Lidar/Kamera verdreht sind
        self.lidar_height_offset = 0.05 # In Metern: Hebt/Senkt den Marker in RViz
        self.camera_to_lidar_dist = 0.03 # Falls die Kamera 3cm vor dem Lidar sitzt

        # 2. Subscriptions
        # Kamera-Bild   
        self.last_image_msg = None  # Hier speichern wir das letzte Bild für die Fusion
        self.image_width = None  # Breite des Kamerabildes (wird beim ersten Bild gesetzt)
        self.img_sub = self.create_subscription(
            Image, 
            '/camera/image_raw', 
            self.camera_sub_callback, 
            qos_profile_sensor_data
        )
        # Lidar-Scan hinzufügen
        self.scan_sub = self.create_subscription(
            LaserScan,
            '/ldlidar_node/scan',
            self.scan_callback,
            qos_profile_sensor_data
        )
        
        # 3. Publisher
        self.marker_pub = self.create_publisher(Marker, '/detected_obstacles', 10)
        self.pub_debug_img = self.create_publisher(Image, '/camera/yolo_debug', 10)
        self.avoid_trigger_dist = 0.85 # Schwellenwert: Ab 85cm vor dem Hindernis ausweichen
        
        # Variablen für die Fusion
        self.camera_fov = 115.0  # Dein Sichtfeld
        self.get_logger().info('YOLO Lidar Fusion Node gestartet.')

        self.turn_start_time = 0.0

        ############ debug ##############
        self.begin = True
        self.counter = 0

        # Mess Variablen
        self.messwerte_x = np.array([
            -1.0, -0.75, -0.50, -0.30, -0.15, -0.05, 
             0.0, 
             0.05, 0.15, 0.30, 0.50, 0.75, 1.0
        ])
        self.messwerte_y = np.zeros(13)
        self.turn_polynom = None
        self.current_test_index = 0

        self.test_state = "INIT"
        self.state_start_time = 0.0
        self.measure_start_yaw = None
        self.start_wall_distance = None
        self.single_test_mode = False

        self.lidar_to_turnpoint_distance = 0.12

        self.load_calibration()

    def send_line(self, marker_array, m_id, p1, p2, color=(1.0, 1.0, 1.0)):
        """Hilfsfunktion zum Erstellen einer Linie für das MarkerArray."""
        marker = Marker()
        marker.header.frame_id = self.rviz_frame
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "walls"
        marker.id = m_id
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        marker.scale.x = 0.03  # Dicke der Linie
        marker.color.r, marker.color.g, marker.color.b = color
        marker.color.a = 1.0
        
        point1 = Point()
        point1.x, point1.y = float(p1[0]), float(p1[1])
        
        point2 = Point()
        point2.x, point2.y = float(p2[0]), float(p2[1])
        
        marker.points = [point1, point2]
        marker_array.markers.append(marker)

    def publish_marker(self, x, y, name, class_id):
        marker = Marker()
        marker.header.frame_id = self.rviz_frame
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "yolo_obstacles"
        marker.id = class_id
        marker.type = Marker.CYLINDER
        marker.action = Marker.ADD
        
        # --- DER MAGISCHE FIX ---
        # Keine Winkelrechnungen mehr! Wir nehmen 1:1 die Lidar-Koordinaten.
        marker.pose.position.x = float(x)
        marker.pose.position.y = float(y)
        
        marker.pose.position.z = self.lidar_height_offset
        
        marker.scale.x, marker.scale.y, marker.scale.z = 0.15, 0.15, 0.3
        
        marker.color.a = 1.0
        if "red" in name:
            marker.color.r, marker.color.g, marker.color.b = 1.0, 0.0, 0.0
        else:
            marker.color.r, marker.color.g, marker.color.b = 0.0, 1.0, 0.0
            
        marker.lifetime = rclpy.duration.Duration(seconds=0.5).to_msg()
        self.marker_pub.publish(marker)

    def imu_callback(self, msg):
        """
        Wandelt Quaternionen in einen UNENDLICHEN, sprungfreien Winkel um.
        (Kein Zurückspringen bei 180° oder 360°!)
        """
        self.imu_ready = True
        q = msg.orientation
        
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        yaw_rad = math.atan2(siny_cosp, cosy_cosp)
        
        # Das ist der rohe Wert mit dem blöden Sprung bei 180 / -180
        raw_yaw = math.degrees(yaw_rad)
        
        # Beim allerersten Datenpaket initialisieren wir einfach
        if self.last_raw_yaw is None:
            self.last_raw_yaw = raw_yaw
            self.current_yaw = raw_yaw
            return

        # Wie weit haben wir uns seit der letzten Millisekunde gedreht?
        delta = raw_yaw - self.last_raw_yaw
        
        # ==========================================
        # DIE MAGIE: Den 360°-Sprung abfangen!
        # ==========================================
        if delta > 180.0:
            delta -= 360.0
        elif delta < -180.0:
            delta += 360.0
            
        # Wir addieren nur das saubere Delta zu unserem unendlichen Winkel
        self.current_yaw += delta
        self.last_raw_yaw = raw_yaw
    
    def camera_sub_callback(self, msg):
        self.last_image_msg = msg

    def send_text(self, marker_array, m_id, text, x, y, color=(1.0, 1.0, 1.0)):
        """Hilfsfunktion zum Erstellen von schwebendem Text in RViz."""
        marker = Marker()
        marker.header.frame_id = self.rviz_frame
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "labels"  # Eigener Namespace, damit es nicht mit den Linien crasht
        marker.id = m_id
        marker.type = Marker.TEXT_VIEW_FACING
        marker.action = Marker.ADD
        
        # Position des Textes (Z leicht erhöht, damit es über dem Lidar schwebt)
        marker.pose.position.x = float(x)
        marker.pose.position.y = float(y)
        marker.pose.position.z = 0.3 
        
        marker.scale.z = 0.15  # Textgröße
        
        marker.color.r, marker.color.g, marker.color.b = color
        marker.color.a = 1.0
        marker.text = text
        
        marker_array.markers.append(marker)

    def send_sphere(self, marker_array, m_id, x, y, color=(0.0, 1.0, 1.0)):
        """Zeichnet eine leuchtende Kugel (die 'Karotte') in RViz."""
        marker = Marker()
        marker.header.frame_id = self.rviz_frame
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "target"
        marker.id = m_id
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.position.x = float(x)
        marker.pose.position.y = float(y)
        marker.pose.position.z = 0.05 # Leicht über dem Boden
        marker.scale.x = 0.15 # 15 cm Durchmesser
        marker.scale.y = 0.15
        marker.scale.z = 0.15
        marker.color.r, marker.color.g, marker.color.b = color
        marker.color.a = 1.0
        marker_array.markers.append(marker)

    def get_all_clusters_sorted(self, point_data): # Angepasst auf Normalvektor und Skalarprodukt
        """
        Sucht ALLE zusammenhängenden Cluster und durchtrennt sie an 90°-Ecken.
        Nutzt die Manhattan-Norm für Lücken und das Skalarprodukt für Ecken.
        point_data: Liste aus (Winkel_deg, x, y, dist)
        """
        if len(point_data) < 2:
            return []

        # Nach Winkel sortieren (von rechts nach hinten nach links)
        sorted_points = sorted(point_data, key=lambda p: p[0])

        max_gap = 0.15      # Maximaler Abstand zwischen zwei Punkten (15cm)
        outlier_limit = 2   # Wie viele Punkte dürfen fehlen?
        
        clusters = []
        current_cluster = []
        
        i = 0
        while i < len(sorted_points):
            p_curr = sorted_points[i]
            
            if not current_cluster:
                current_cluster.append(p_curr)
                i += 1
                continue
            
            p_last = current_cluster[-1]
            
            # 1. MANHATTAN-DISTANZ (Lückenerkennung)
            dx = abs(p_curr[1] - p_last[1])
            dy = abs(p_curr[2] - p_last[2])
            dist_manhattan = dx + dy
            
            if dist_manhattan < max_gap:
                
                # =========================================================
                # 2. NEUE ECKEN-ERKENNUNG (Skalarprodukt / Dot-Product)
                # =========================================================
                is_corner = False
                
                # Wir brauchen mindestens 5 Punkte Historie, um einen Trend zu sehen
                if len(current_cluster) >= 5:
                    # Vektor A: Die Wand der Vergangenheit (Punkt vor 5 Steps bis letzter Punkt)
                    p_past = current_cluster[-5]
                    vec_a_x = p_last[1] - p_past[1]
                    vec_a_y = p_last[2] - p_past[2]
                    len_a = math.hypot(vec_a_x, vec_a_y)
                    
                    # Vektor B: Der neue Schritt (Letzter Punkt zum aktuellen Punkt)
                    vec_b_x = p_curr[1] - p_last[1]
                    vec_b_y = p_curr[2] - p_last[2]
                    len_b = math.hypot(vec_b_x, vec_b_y)
                    
                    if len_a > 0.01 and len_b > 0.01:
                        # Vektoren normalisieren (Länge 1)
                        vec_a_x /= len_a
                        vec_a_y /= len_a
                        vec_b_x /= len_b
                        vec_b_y /= len_b
                        
                        # Skalarprodukt: Gibt den Kosinus des eingeschlossenen Winkels
                        # 1.0 = Gleiche Richtung (Gerade Wand)
                        # 0.0 = 90 Grad (Perfekte Ecke)
                        # -1.0 = 180 Grad (Umdrehen)
                        dot_product = (vec_a_x * vec_b_x) + (vec_a_y * vec_b_y)
                        
                        # Wenn das Skalarprodukt unter 0.7 fällt (entspricht ca. > 45 Grad Knick), 
                        # haben wir eine Ecke erreicht!
                        if dot_product < 0.70:
                            is_corner = True

                # =========================================================
                # 3. ENTSCHEIDUNG TREFFEN
                # =========================================================
                if is_corner:
                    # Ecke erkannt! Wir beenden die alte Wand und starten sofort eine neue.
                    # Der aktuelle Punkt (p_curr) ist der allererste Punkt der neuen Wand!
                    clusters.append(current_cluster)
                    current_cluster = [p_curr]
                else:
                    # Keine Ecke, Punkt gehört zur aktuellen Wand
                    current_cluster.append(p_curr)
                
                i += 1
                    
            else:
                # 4. AUSREISSER-LOGIK (Lücken überbrücken)
                found_connection = False
                for look_ahead in range(1, outlier_limit + 1):
                    if i + look_ahead < len(sorted_points):
                        p_future = sorted_points[i + look_ahead]
                        dist_future = abs(p_future[1] - p_last[1]) + abs(p_future[2] - p_last[2])
                        
                        if dist_future < max_gap:
                            i += look_ahead
                            found_connection = True
                            break
                
                if not found_connection:
                    clusters.append(current_cluster)
                    current_cluster = [p_curr]
                    i += 1
        
        if current_cluster:
            clusters.append(current_cluster)

        # 5. WRAP-AROUND-FIX (Wenn die Wand bei 180 / -180 Grad durchschnitten wurde)
        if len(clusters) > 1:
            first_p = clusters[0][0]
            last_p = clusters[-1][-1]
            dist_wrap = abs(first_p[1] - last_p[1]) + abs(first_p[2] - last_p[2])
            
            if dist_wrap < max_gap:
                clusters[0] = clusters[-1] + clusters[0]
                clusters.pop()

        # Rückgabe: Die größten Wände zuerst
        clusters.sort(key=len, reverse=True)
        return clusters
    
    def get_cluster_angle(self, cluster): # Angepasst auf neues Koordinatensystem
        if cluster is None or len(cluster) < 2:
            return None

        n = len(cluster)
        mean_x = sum(p[1] for p in cluster) / n
        mean_y = sum(p[2] for p in cluster) / n

        s_xx = 0.0
        s_yy = 0.0
        s_xy = 0.0
        
        for p in cluster:
            dx = p[1] - mean_x
            dy = p[2] - mean_y
            s_xx += dx * dx
            s_yy += dy * dy
            s_xy += dx * dy

        # Wieder dein Original: s_yy - s_xx
        # Dadurch bleiben Seitenwände bei 0° und Frontwände bei 90°
        angle_rad = 0.5 * math.atan2(2.0 * s_xy, s_yy - s_xx)
        
        return math.degrees(angle_rad)

    def delete_marker(self, marker_array, m_id, ns="walls"):
        """Löscht einen Marker in einem bestimmten Namespace."""
        marker = Marker()
        marker.header.frame_id = self.rviz_frame
        marker.ns = ns  # Jetzt flexibel! Standardmäßig aber "walls"
        marker.id = m_id
        marker.action = Marker.DELETE
        marker_array.markers.append(marker)

    def scan_callback(self, msg):
        point_data = []

        for i, dist in enumerate(msg.ranges):
            # Filtere ungültige Werte (inf, nan oder außerhalb der Reichweite)
            if math.isinf(dist) or math.isnan(dist) or dist < 0.075 or dist > 3.0:
                continue

            angle_lidar_rad = msg.angle_min + i * msg.angle_increment
            
            # Foxglove & Mathe Basis: X = Rechts, Y = Vorne
            x_ros = dist * math.cos(angle_lidar_rad)    
            y_ros = dist * math.sin(angle_lidar_rad)   
            
            # ========================================================
            # 🚀 NEUES WINKEL-SYSTEM: 0° bis 360°, Startpunkt ist HINTEN
            # ========================================================
            # Roher Lidar-Winkel: 0°=Rechts, 90°=Vorne, 180°=Links, 270°/-90°=Hinten
            # Wir addieren 90°, damit Hinten zu 0° wird und begrenzen auf 0-360.
            # -> Hinten: 0° | Rechts: 90° | Vorne: 180° | Links: 270°
            angle_lidar_deg = math.degrees(angle_lidar_rad)
            angle_user_deg = (angle_lidar_deg + 90.0) % 360.0

            # Speichern: (Normierter 0-360° Winkel, X_Rechts, Y_Vorne, Distanz)
            point_data.append((angle_user_deg, x_ros, y_ros, dist))
        
        self.last_point_data = point_data
        self.main_logic(point_data)
    
    def get_closest_measure(self, point_data, target_angle):
        """
        Sucht den Punkt, der dem target_angle am nächsten ist,
        unter Berücksichtigung des Kreisumschlags und negativer Winkel.
        """
        if not point_data:
            return None

        def get_angular_diff(a, b):
            # Berechnet die kleinste Differenz auf einem 360-Grad-Kreis
            # (a - b + 180) % 360 - 180 normiert das Ergebnis auf den Bereich [-180, 180]
            diff = (a - b + 180) % 360 - 180
            return abs(diff)

        # Wir suchen das Tupel, bei dem die Kreis-Differenz am kleinsten ist
        closest_point = min(point_data, key=lambda p: get_angular_diff(p[0], target_angle))
        
        return closest_point
    
    def get_closest_point_in_cluster(self, cluster):
        """
        Gibt den Punkt eines Clusters zurück, der den kürzesten Abstand zum LiDAR hat.
        cluster: Liste von Punkten im Format (angle, x, y, dist)
        """
        if not cluster:
            return None

        # Sucht das Element im Cluster, bei dem der Wert an Index 3 (dist) am kleinsten ist
        closest_point = min(cluster, key=lambda p: p[3])
        
        return closest_point

    def middle_of_cluster(self, cluster):
        """Berechnet den Winkel des Mittelpunktes des Clusters zum Lidar."""
        sum = 0
        for c in cluster:
            sum += c[0]  # Winkel des Punktes
        return sum / len(cluster)

    def angle_diff(self, a, b):
        """Berechnet den kleinsten Unterschied zwischen zwei Winkeln (Rad)."""
        # Sorgt dafür, dass der Unterschied auch über die 0/360° Grenze korrekt bleibt
        return math.atan2(math.sin(a - b), math.cos(a - b))
        
    def visualize_clusters(self, kandidaten_RVIZ):
        if self.fahrtrichtung == 'links':
            innenbande = self.left_wall
            aussenbande = self.right_wall
        else:
            innenbande = self.right_wall
            aussenbande = self.left_wall
            
        
        marker_array = MarkerArray()

        if self.state == 'FOLLOW_LANE':
            # ALTE LOGIK FÜR DIE GERADE
            if self.fahrtrichtung == 'links':
                innenbande = self.left_wall
                aussenbande = self.right_wall
            else:
                innenbande = self.right_wall
                aussenbande = self.left_wall
            
            if target_x is not None and target_y is not None:
                # Cyan für Geradeaus-Fahrt
                self.send_sphere(marker_array, m_id=99, x=target_x, y=target_y, color=(0.0, 1.0, 1.0))
                
        elif self.state in ['TURN_LINKS', 'TURN_RECHTS']:
            # NEUE LOGIK FÜR DIE KURVE
            # get_target_point_turn erwartet das u_profile Array!
            u_profile = [self.right_wall, self.front_wall, self.left_wall]
            
            # Zielpunkt-Berechnung aufrufen (z.B. 40cm Soll-Abstand)
            target_pt = None
            
            if target_pt is not None:
                target_x, target_y = target_pt
                # Magenta für Kurven-Fahrt (damit du sofort siehst, dass die neue Logik greift!)
                self.send_sphere(marker_array, m_id=99, x=target_x, y=target_y, color=(1.0, 0.0, 1.0))
        else:
            self.delete_marker(marker_array, 99, ns="target")

        # Feste Farben: Rechts=Rot, Front=Grün, Links=Blau
        colors = [
            (1.0, 0.0, 0.0),    # Rot (Original)
            (0.0, 1.0, 0.0),    # Grün (Original)
            (0.0, 0.5, 1.0),    # Azurblau (Original)
            (1.0, 0.5, 0.0),    # Orange
            (0.5, 0.0, 1.0),    # Violett
            (0.0, 1.0, 1.0),    # Cyan
            (1.0, 0.0, 1.0),    # Magenta
            (1.0, 1.0, 0.0),    # Gelb
            (0.5, 0.5, 0.5),    # Grau
            (0.6, 0.3, 0.0)     # Braun
        ]


        # --- RVIZ TEXT-MARKER FÜR DIE FAHRTRICHTUNG UND BANDEN ---
        if self.fahrtrichtung is not None:
            # 1. Zeige die gelockte Fahrtrichtung direkt über dem Roboter an (X=0, Y=0)
            richtung_text = f"LOCKED: {self.fahrtrichtung.upper()}"
            self.send_text(marker_array, m_id=10, text=richtung_text, x=0.0, y=0.0, color=(1.0, 1.0, 0.0)) # Gelb

            # 2. Beschrifte die Innenbande
            if innenbande and len(innenbande) > 0:
                # Wir platzieren den Text in der Mitte der Wand
                mitte_x = sum(p[1] for p in innenbande) / len(innenbande)
                mitte_y = sum(p[2] for p in innenbande) / len(innenbande)
                self.send_text(marker_array, m_id=11, text="INNEN", x=mitte_x, y=mitte_y, color=(1.0, 0.5, 0.0)) # Orange

            # 3. Beschrifte die Außenbande
            if aussenbande and len(aussenbande) > 0:
                mitte_x = sum(p[1] for p in aussenbande) / len(aussenbande)
                mitte_y = sum(p[2] for p in aussenbande) / len(aussenbande)
                self.send_text(marker_array, m_id=12, text="AUSSEN", x=mitte_x, y=mitte_y, color=(1.0, 0.0, 1.0)) # Magenta
        else:
            # Solange er noch scannt, zeige das an
            self.send_text(marker_array, m_id=10, text="SCANNING DIRECTION...", x=0.0, y=0.0, color=(1.0, 1.0, 1.0)) # Weiß
        for i, cluster in enumerate(kandidaten_RVIZ):
            # Info in der Konsole ausgeben
            angle = self.get_cluster_angle(cluster)

            if angle is None:
                #self.get_logger().warn(f"Wand {i+1}: Winkel konnte nicht berechnet werden.")
                continue # Überspringt diesen Cluster und macht mit dem nächsten weiter
            #self.get_logger().info(f"Wand {i+1} (ID {i}): {len(cluster)} Punkte, Winkel: {angle:.2f}°")
            #self.get_logger().info(f"Erster Punkt: (X={cluster[0][1]:.2f}, Y={cluster[0][2]:.2f}), Letzter Punkt: (X={cluster[-1][1]:.2f}, Y={cluster[-1][2]:.2f})")
            if cluster is None or len(cluster) < 2:
                self.delete_marker(marker_array, m_id=i)
                continue # Gehe sofort zum nächsten Element im Array über

            # Ein Cluster braucht mindestens 2 Punkte für eine Linie
            if len(cluster) >= 2:
                # Start- und Endpunkt (Index 1 = X, Index 2 = Y)
                start_p = (cluster[0][1], cluster[0][2])
                ende_p = (cluster[-1][1], cluster[-1][2])
                
                # Linie zeichnen: ID entspricht dem Index (0, 1, oder 2)
                self.send_line(marker_array, m_id=i, p1=start_p, p2=ende_p, color=colors[i])

        # Alles veröffentlichen, damit es in RViz2 auftaucht!
        self.pub_markers.publish(marker_array)

    def cluster_wahl(self, point_data):
        # 1. Alle Cluster finden
        all_clusters = self.get_all_clusters_sorted(point_data)
        for c in all_clusters:
            self.visualize_clusters([c])
            skip = input("Drücke Enter, um zum nächsten Cluster zu gehen (oder 'q' für Auswahl)...")
            if skip.lower() == 'q':
                self.get_logger().info(">>> Starte Test Bench mit der gewählten Frontwand! <<<")
                return c
            else:
                continue
        return None
        
    def load_calibration(self):
        file_path = '/workspace/src/robot_vision/robot_vision/wro_calibration.json'
        
        if os.path.exists(file_path):
            try:
                with open(file_path, 'r') as f:
                    calib_data = json.load(f)
                
                # Werte in Numpy-Arrays zurückwandeln
                self.messwerte_x = np.array(calib_data["messwerte_x"])
                self.messwerte_y = np.array(calib_data["messwerte_y"])
                self.poly_coeffs = np.array(calib_data["poly_coeffs"])
                
                self.get_logger().info("Kalibrierungsdaten erfolgreich geladen.")
            except Exception as e:
                self.get_logger().error(f"JSON fehlerhaft, nutze Standardwerte. Fehler: {e}")
                self._set_default_calibration()
        else:
            self.get_logger().warn("Keine Kalibrierungsdatei gefunden! Nutze Standardwerte.")
            self._set_default_calibration()

    def _set_default_calibration(self):
        # Fallback, falls keine JSON existiert
        self.poly_coeffs = np.array([0.0, 1.5, 0.0]) # Lineare Standard-Annahme
        # self.messwerte_x und _y wie bisher definiert lassen

    def save_calibration(self):
        # 1. Radien in Krümmung (Kappa) umwandeln
        kappas = np.zeros_like(self.messwerte_y)
        for i, r in enumerate(self.messwerte_y):
            # Vorzeichen beibehalten! (Rechtskurven haben negatives u, also auch negatives kappa)
            # Wenn der Radius über den Lidar-Offset positiv reinkommt, müssen wir
            # das Vorzeichen von u übernehmen, damit die Mathe stimmt.
            vorzeichen = np.sign(self.messwerte_x[i])
            if vorzeichen == 0:
                vorzeichen = 1.0

            if abs(r) > 0.001:  # Schutz vor Division durch 0
                kappas[i] = vorzeichen * (1.0 / abs(r))
            else:
                kappas[i] = 0.0

        # 2. Polynom 3. Grades fitten: u = c3*kappa^3 + c2*kappa^2 + c1*kappa + c0
        # WICHTIG: Wir fitten u (x) über kappa, nicht umgekehrt! Grad 3 wegen Symmetrie.
        coeffs = np.polyfit(kappas, self.messwerte_x, 3)

        # 3. Daten in ein Dictionary packen
        calib_data = {
            "messwerte_x": self.messwerte_x.tolist(),
            "messwerte_y": self.messwerte_y.tolist(),
            "poly_coeffs": coeffs.tolist()
        }

        # 4. Als JSON speichern (Absoluter Pfad im Home-Verzeichnis)
        file_path = '/workspace/src/robot_vision/robot_vision/wro_calibration.json'
        try:
            with open(file_path, 'w') as f:
                json.dump(calib_data, f, indent=4)
            self.get_logger().info(f"Kalibrierung erfolgreich gespeichert unter {file_path}")
            self.get_logger().info(f"Gefundene Koeffizienten: {coeffs}")
        except Exception as e:
            self.get_logger().error(f"Fehler beim Speichern der Kalibrierung: {e}")

    def main_logic(self, point_data):
        """
        Zustandsautomat für die Kalibrierung.
        """
        if self.measure_start_yaw is None:
            self.measure_start_yaw = self.current_yaw
        # --- FIX FÜR INDEX 8 (Geradeausfahrt) ---
        # Da eine 180°-Drehung bei u=0 unmöglich ist, setzen wir R=0 und überspringen.
        if self.current_test_index == 6 and self.test_state != "DONE":
            self.messwerte_y[6] = 0.0
            self.get_logger().info("Index 8 (u=0.0) wird übersprungen (Geradeausfahrt).")
            
            if self.single_test_mode:
                self.save_calibration()
                self.test_state = "DONE"
            else:
                self.current_test_index += 1
            return

        # ---------------------------------------------------------
        if self.test_state == "INIT":
            self.measure_start_yaw = None
            if self.imu_ready:
                # 1. Benutzereingabe abfragen
                self.get_logger().info("Sensoren bereit. Warte auf Benutzereingabe im Terminal...")
                print("\n=== KALIBRIERUNGS-MENÜ ===")
                for i, x in enumerate(self.messwerte_x):
                    aktueller_radius = self.messwerte_y[i]
                    # Ausgabe formatiert: u auf 2 Nachkommastellen, Radius auf 3 Nachkommastellen (Meter)
                    print(f"[{i:2d}]: u = {x:5.2f}  |  Gespeicherter Radius: {aktueller_radius:6.3f} m")
                    
                eingabe = input("\nWelcher Index soll gemessen werden? ('all' für alle): ")
                
                if eingabe.lower() == 'all':
                    self.current_test_index = 0
                    self.single_test_mode = False
                else:
                    try:
                        idx = int(eingabe)
                        if 0 <= idx < len(self.messwerte_x):
                            self.current_test_index = idx
                            self.single_test_mode = True
                        else:
                            self.get_logger().error("Index out of bounds! Skript neustarten.")
                            self.test_state = "DONE"
                            return
                    except ValueError:
                        self.get_logger().error("Ungültige Eingabe! Skript neustarten.")
                        self.test_state = "DONE"
                        return

                self.get_logger().info(f"Starte Test für u={self.messwerte_x[self.current_test_index]}")
                
                # 2. Clusterwahl und Start der Messung
                cluster = self.cluster_wahl(point_data)
                if cluster is not None and len(cluster) > 0:
                    self.start_wall_distance = (self.get_closest_point_in_cluster(cluster)[3])
                    self.test_state = "APPLY_STEERING"

        # ---------------------------------------------------------
        elif self.test_state == "APPLY_STEERING":
            u = self.messwerte_x[self.current_test_index]
            test_cmd = Twist()
            test_cmd.linear.x = self.turn_speed
            test_cmd.angular.z = float(u)
            self.pub_cmd_vel.publish(test_cmd)
            self.test_state = "STOP_FOR_MEASUREMENT"
            self.get_logger().info(f"Starte Drehung für u={u}")

        # ---------------------------------------------------------
        elif self.test_state == "STOP_FOR_MEASUREMENT":
            u = self.messwerte_x[self.current_test_index]
            test_cmd = Twist()
            test_cmd.linear.x = self.turn_speed
            test_cmd.angular.z = float(u)
            self.pub_cmd_vel.publish(test_cmd)
            if abs(self.current_yaw - self.measure_start_yaw) >= 85.0:
                test_cmd = Twist()
                test_cmd.linear.x = 0.0
                test_cmd.angular.z = 0.0
                self.pub_cmd_vel.publish(test_cmd)
                self.test_state = "MEASURE"
                self.get_logger().info("Drehung abgeschlossen. Stoppe Roboter und starte Messung...")
            self.get_logger().info(f"Derzeitiger Yaw: {self.current_yaw:.1f}°, Ziel: {self.measure_start_yaw}°")

        # ---------------------------------------------------------
        elif self.test_state == "MEASURE":
            cluster = self.cluster_wahl(point_data)
            if cluster is not None and len(cluster) > 0:
                yaw_diff = abs(self.current_yaw - self.measure_start_yaw)
                new_distance = self.get_closest_point_in_cluster(cluster)[3]
                distance_diff = new_distance - self.start_wall_distance - self.lidar_to_turnpoint_distance
                #radius = distance_diff / 2.0
                radius = distance_diff

                # HIER wird der alte Wert im Array mit dem neuen überschrieben
                self.messwerte_y[self.current_test_index] = abs(radius)
                self.get_logger().info(f"Ergebnis u={self.messwerte_x[self.current_test_index]}: Radius {radius*1000:.0f} mm, Winkel {yaw_diff:.1f}°")
                self.test_state = "NEXT_TEST"

        # ---------------------------------------------------------
        elif self.test_state == "NEXT_TEST":
            # --- ABBRUCHBEDINGUNG FÜR EINZELMESSUNG ---
            if self.single_test_mode:
                self.get_logger().info("Einzelmessung abgeschlossen. Speichere neue JSON...")
                stop_cmd = Twist()
                stop_cmd.linear.x = 0.0
                stop_cmd.angular.z = 0.0
                self.pub_cmd_vel.publish(stop_cmd)
                
                # Speichert die 15 alten und den 1 neuen Wert + fittet das Polynom neu
                self.save_calibration()
                self.test_state = "DONE"
                
            else:
                self.current_test_index += 1
                if self.current_test_index < len(self.messwerte_x):
                    self.get_logger().info(f"Wechsle zu u={self.messwerte_x[self.current_test_index]}")
                    self.test_state = "APPLY_STEERING"
                else:
                    self.get_logger().info("Alle Messungen abgeschlossen. Stoppe Roboter.")
                    stop_cmd = Twist()
                    stop_cmd.linear.x = 0.0
                    stop_cmd.angular.z = 0.0
                    self.pub_cmd_vel.publish(stop_cmd)
                    
                    self.save_calibration()
                    self.test_state = "DONE"

        # ---------------------------------------------------------
        elif self.test_state == "DONE":
            self.test_state = "INIT"

        
def main(args=None):
    rclpy.init(args=args)
    node = WallFollower()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()