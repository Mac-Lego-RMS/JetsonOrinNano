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
from robot_vision.steering_lib import SteeringController



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
        self.saved_target_angle = None
        self.saved_curve_radius_m = None

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
        self.steering_ctrl = SteeringController(logger=self.get_logger())
        self.lookahead_dist_turn = 0.20
        self.start_turn_dist = 0
        self.target_point = (0, 0)

        self.TRACK_WIDTH_M = 1.0
        self.ROBOT_WIDTH_M = 0.15
        self.SAFETY_MARGIN_M = 0.05
        self.MAX_KINEMATIC_RADIUS_M = 1.2
        self.IDEAL_RADIUS_M = 0.45
        self.MIN_TURN_RADIUS_M = 0.2

        # --- TURN PID-REGLER PARAMETER ---
        self.turn_kp = 0.6
        self.turn_kd = 0.0
        self.turn_ki = 0.0

        # --- MOTOR PARAMETER (ESP PWM 0 - 1023) ---
        self.base_speed = 450.0  # Normale Geschwindigkeit auf der Geraden
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


        ############ debug ##############
        self.begin = True
        self.counter = 0
        self.test_is_turning = False
        self.curve_radius_m = None

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

    def send_text(self, marker_array, m_id, text, x, y, color=(1.0, 1.0, 1.0), scale=0.15):
        """
        Fügt einen sichtbaren Text-Marker zum MarkerArray hinzu.
        scale: Schriftgröße in Metern (Standard 15cm)
        """
        marker = Marker()
        marker.header.frame_id = "base_link"  # Passe dies an euren Frame an (z.B. "laser")
        marker.header.stamp = self.get_clock().now().to_msg()
        
        marker.ns = "text_labels"
        marker.id = m_id
        marker.type = Marker.TEXT_VIEW_FACING  # Wichtig: Typ 9 für Text
        marker.action = Marker.ADD
        
        # Position
        marker.pose.position.x = float(x)
        marker.pose.position.y = float(y)
        # Leichtes Anheben (z = 0.1) verhindert oft, dass der Text im Grid/Boden verschwindet
        marker.pose.position.z = 0.1 
        
        # Ausrichtung (bei TEXT_VIEW_FACING rotiert der Text automatisch zur Kamera,
        # die Quaternion muss aber trotzdem valide sein)
        marker.pose.orientation.w = 1.0
        
        # SKALIERUNG: Bei Text ist NUR scale.z relevant (Schriftgröße in m)
        marker.scale.z = float(scale) 
        
        # Farbe und Sichtbarkeit
        marker.color.r = float(color[0])
        marker.color.g = float(color[1])
        marker.color.b = float(color[2])
        marker.color.a = 1.0  # Wichtig: 1.0 bedeutet 100% sichtbar (Deckkraft)
        
        # Der eigentliche String
        marker.text = str(text)
        
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
    
    def validate_clusters_straight(self, clusters):
        # Wir nutzen while, da sich die Liste verkleinern kann
        while len(clusters) >= 3:
            
            # 1. Die aktuell größten 3 nehmen und von rechts nach links sortieren
            ordered = self.sort_clusters_right_to_left(clusters[:3])
            
            # 2. Winkel berechnen
            angles = [self.get_cluster_angle(c) for c in ordered]

            if any(angle is None for angle in angles):
                self.get_logger().warn("Fehler bei der Winkelberechnung. Überspringe...")
                return [None, None, None]

            # --- HILFSFUNKTION FÜR WINKEL-DIFFERENZ ---
            # Berechnet den kleinsten Schnittwinkel zwischen zwei Geraden (0° bis 90°)
            def get_angle_diff(a1, a2):
                diff = abs(a1 - a2) % 180
                if diff > 90:
                    diff = 180 - diff
                return diff

            # Differenzen berechnen (0=Rechts, 1=Front, 2=Links)
            diff_0_1 = get_angle_diff(angles[0], angles[1]) # Sollte ~90° sein (orthogonal)
            diff_1_2 = get_angle_diff(angles[1], angles[2]) # Sollte ~90° sein (orthogonal)
            diff_0_2 = get_angle_diff(angles[0], angles[2]) # Sollte ~0° sein (parallel)

            # --- ÜBERPRÜFUNG ---
            # Toleranz: Wir erlauben bis zu 20° Abweichung von der perfekten Geometrie
            
            # Check A: Sind Rechts und Front orthogonal? (Differenz sollte > 70° sein)
            if diff_0_1 < 70:
                self.get_logger().warn(f"Rechts und Front nicht orthogonal! Diff: {diff_0_1:.1f}°")
                # Finde das kleinere der beiden Cluster in 'ordered' und lösche es aus 'clusters'
                if len(ordered[0]) < len(ordered[1]):
                    clusters.remove(ordered[0])
                else:
                    clusters.remove(ordered[1])
                continue # Schleife sofort mit der bereinigten Liste neu starten

            # Check B: Sind Front und Links orthogonal?
            elif diff_1_2 < 70:
                self.get_logger().warn(f"Front und Links nicht orthogonal! Diff: {diff_1_2:.1f}°")
                if len(ordered[1]) < len(ordered[2]):
                    clusters.remove(ordered[1])
                else:
                    clusters.remove(ordered[2])
                continue

            # Check C: Sind Rechts und Links parallel? (Differenz sollte < 20° sein)
            elif diff_0_2 > 10:
                self.get_logger().warn(f"Rechts und Links nicht parallel! Diff: {diff_0_2:.1f}°")
                if len(ordered[0]) < len(ordered[2]):
                    clusters.remove(ordered[0])
                else:
                    clusters.remove(ordered[2])
                continue

            # --- ERFOLG ---
            
            # Gibt exakt zugeordnet zurück: (Rechte Bande, Frontwand, Linke Bande)
            return ordered
        # Wenn die Schleife abbricht (Liste hat weniger als 3 Cluster)
        self.get_logger().warn(f"Kein gültiges U-Profil gefunden. Nur noch {len(clusters)} Cluster übrig.")
        return [None, None, None]

    def validate_clusters_turn(self, front_wall, point_data):
        '''Überprüft die Cluster in der Kurve. Es muss eine Frontwand geben, aber die Banden können auch fehlen (z.B. bei der ersten Kurve).
        Gibt die Cluster von rechts nach links zurück mit der Reihenfolge: [Rechte Bande, Frontwand, Linke Bande]. Fehlende Banden werden mit None ersetzt.'''

        if front_wall is None:
            self.get_logger().warn("Keine Frontwand gefunden. Kann Kurvenprofil nicht validieren.")
            return [None, None, None]

        clusters = self.get_all_clusters_sorted(point_data)

        def kill_all_clusters_between(front_wall, all_clusters):
            # Finde die Grenzen des Winkelbereichs
            left_angle = min(front_wall, key=lambda p: p[0])[0]
            right_angle = max(front_wall, key=lambda p: p[0])[0]
            
            kept_clusters = []
            
            for cluster in all_clusters:
                # any() prüft, ob IRGENDEIN Punkt im Cluster im kritischen Bereich liegt
                is_in_killzone = any(left_angle < point[0] < right_angle for point in cluster)
                
                # Nur wenn kein Punkt in der Zone liegt, behalten wir den Cluster
                if not is_in_killzone:
                    kept_clusters.append(cluster)
                    
            return kept_clusters


        front_wall_cluster, combined_clusters = self.merge_clusters(clusters, [front_wall])
        front_wall_cluster = front_wall_cluster[0]
        
        clusters = kill_all_clusters_between(front_wall_cluster, clusters)
        clusters.append(front_wall_cluster)


         # Wir fügen die Frontwand wieder hinzu, damit sie in der Sortierung berücksichtigt wird

        if len(clusters) >= 2:
            minimal_cluster_size = 25
            ordered = self.sort_clusters_right_to_left(clusters)
            u_profile = [None, None, None] # 0=Rechts, 1=Front, 2=Links
            if front_wall_cluster in ordered:
                u_profile[1] = front_wall_cluster
                fw_index = ordered.index(front_wall_cluster)
            else: 
                self.get_logger().warn("Frontwand nicht in den Clustern gefunden. Kann Kurvenprofil nicht validieren.")
                return [None, None, None]
                
            # Der Cluster LINKS von der Frontwand hat einen KLEINEREN Index
            while u_profile[2] is None and fw_index > 0:
                if len(ordered[fw_index - 1]) > minimal_cluster_size:
                    u_profile[2] = ordered[fw_index - 1]
                    self.get_logger().info(f"Cluster links von der Frontwand gefunden. Größe: {len(ordered[fw_index - 1])} Punkte.")
                else: 
                    ordered.pop(fw_index - 1)
                    fw_index -= 1

            # Der Cluster RECHTS von der Frontwand hat einen GRÖSSEREN Index
            while u_profile[0] is None and fw_index < len(ordered) - 1:
                if len(ordered[fw_index + 1]) > minimal_cluster_size:
                   
                    u_profile[0] = ordered[fw_index + 1]
                    self.get_logger().info(f"Cluster rechts von der Frontwand gefunden. Größe: {len(ordered[fw_index + 1])} Punkte.")
                    
                else: 
                    ordered.pop(fw_index + 1)

            angles = [self.get_cluster_angle(c) for c in u_profile]
            if angles[1] is None:
                self.get_logger().warn("Fehler bei der Winkelberechnung der Frontwand. Überspringe...")
                return [None, None, None]

            # --- HILFSFUNKTION FÜR WINKEL-DIFFERENZ ---
            # Berechnet den kleinsten Schnittwinkel zwischen zwei Geraden (0° bis 90°)
            def get_angle_diff(a1, a2):
                    diff = abs(a1 - a2) % 180
                    if diff > 90:
                        diff = 180 - diff
                    return diff

                # Differenzen berechnen (0=Rechts, 1=Front, 2=Links)
            if angles[0] is not None:
                diff_0_1 = get_angle_diff(angles[0], angles[1]) # Sollte ~90° sein (orthogonal)
            else: 
                diff_0_1 = None
            if angles[2] is not None:
                diff_1_2 = get_angle_diff(angles[1], angles[2]) # Sollte ~90° sein (orthogonal)
            else:
                diff_1_2 = None
            if angles[0] is not None and angles[2] is not None:
                diff_0_2 = get_angle_diff(angles[0], angles[2]) # Sollte ~0° sein (parallel)
            else:
                diff_0_2 = None


                # --- ÜBERPRÜFUNG ---
                # Toleranz: Wir erlauben bis zu 20° Abweichung von der perfekten Geometrie
                
                # Check A: Sind Rechts und Front orthogonal? (Differenz sollte > 70° sein)
            if diff_0_1 is not None and diff_0_1 < 70:
                self.get_logger().warn(f"Rechts und Front nicht orthogonal! Diff: {diff_0_1:.1f}°")
                u_profile[0] = None


                # Check B: Sind Front und Links orthogonal?
            if diff_1_2 is not None and diff_1_2 < 70:
                self.get_logger().warn(f"Front und Links nicht orthogonal! Diff: {diff_1_2:.1f}°")
                u_profile[2] = None
            
            
            try: 
                wall_dist_left = self.get_closest_point_in_cluster(u_profile[2])[3]
            except: 
                wall_dist_left = None
            try:
                wall_dist_right = self.get_closest_point_in_cluster(u_profile[0])[3]
            except:
                wall_dist_right = None


            if self.fahrtrichtung == "links":
                if wall_dist_right is not None:
                    if wall_dist_right > 1.0:
                        self.get_logger().warn(f"Abstand der rechten Wand {wall_dist_right:.2f}")
                        u_profile[0] = None

                if wall_dist_left is not None:
                    if wall_dist_left < 1.5:
                        self.get_logger().warn(f"Abstand der linken Wand {wall_dist_left:.2f}")
                        u_profile[2] = None

            else: 
                if wall_dist_left is not None:
                    if wall_dist_left > 1.0:
                        self.get_logger().warn(f"Abstand der linken Wand {wall_dist_left:.2f}")
                        u_profile[2] = None

                if wall_dist_right is not None:
                    if wall_dist_right < 1.5:
                        self.get_logger().warn(f"Abstand der rechten Wand {wall_dist_right:.2f}")
                        u_profile[0] = None
            
            u_profile, _ = self.merge_clusters(clusters, u_profile)
            return u_profile

        self.get_logger().warn(f"Kein Cluster außer der Frontwand gefunden.")
        return [None, front_wall_cluster, None]
    
    def sort_clusters_right_to_left(self, clusters): # angepasst auf rechts 90° und links 270°
        """
        Nimmt eine Liste von Clustern und sortiert sie räumlich von rechts nach links.
        System: +X = Rechts, +Y = Vorne
        """
        if not clusters:
            return []

        def get_cluster_bearing(cluster):
            # 1. Schwerpunkt des Clusters berechnen
            # Index 1 = X, Index 2 = Y
            mean_x = sum(p[1] for p in cluster) / len(cluster)
            mean_y = sum(p[2] for p in cluster) / len(cluster)
            
            # 2. Winkel berechnen (0 = Vorne, Negativ = Rechts, Positiv = Links)
            # Das ist exakt die gleiche Logik wie bei unserem Lenk-Servo!
            return math.atan2(-mean_x, mean_y)

        # 3. Aufsteigend sortieren (kleinster/negativster Wert zuerst -> Rechts nach Links)
        sorted_clusters = sorted(clusters, key=get_cluster_bearing, reverse=True)
        
        return sorted_clusters

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
        #self.test_turn_main_logic(point_data)
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

    def merge_clusters(self, all_clusters, validated_clusters):# Angepasst auf neues Koordinatensystem
        """
        Versucht, benachbarte Cluster zu einem einzigen Cluster zu verschmelzen.
        Nutzt den Normalenvektor, um nur den senkrechten Abstand (Offset) zu prüfen.
        """
        # HIER KORRIGIERT: 0.05 Meter sind 5 cm! (Vorher stand hier 0.25)
        max_distance_gap = 0.05  
        max_angle_gap = 15.0      # 5 Grad maximale Winkelabweichung

        remaining_clusters = [c for c in all_clusters if c not in validated_clusters]
        if not remaining_clusters:
            self.get_logger().info("Keine Cluster zum Mergen gefunden")
            return validated_clusters, []

        def get_angle_diff(a1, a2):
            diff = abs(a1 - a2) % 180
            if diff > 90:
                diff = 180 - diff
            return diff

        combined_clusters = [[] for _ in range(len(validated_clusters))]

        for i, valid_cluster in enumerate(validated_clusters):
            angle = self.get_cluster_angle(valid_cluster)
            if valid_cluster is None:
                self.get_logger().warn("Leeres Cluster in Merge-Logik. Überspringe...")
                continue

            if angle is None: 
                continue

            bx = sum(p[1] for p in valid_cluster) / len(valid_cluster)
            by = sum(p[2] for p in valid_cluster) / len(valid_cluster)

            angle_rad = math.radians(angle)
            
            # =========================================================
            # DER KOORDINATEN-FIX FÜR DIE VEKTOREN
            # =========================================================
            # WICHTIG: Deine Funktion get_cluster_angle liefert 0° für Seitenwände
            # und 90° für Frontwände. Sie misst den Winkel also relativ zur Y-Achse!
            # Deshalb müssen wir Sinus und Kosinus hier vertauschen:
            
            # 1. RICHTUNGSVEKTOR (Parallel zur Wand)
            dir_x = math.sin(angle_rad)
            dir_y = math.cos(angle_rad)

            # 2. NORMALENVEKTOR (Senkrecht zur Wand)
            # Wir drehen den Vektor mathematisch um 90 Grad
            nx = math.cos(angle_rad)
            ny = -math.sin(angle_rad)

            clusters_to_remove = []

            for other in remaining_clusters:
                other_angle = self.get_cluster_angle(other)
                if other_angle is None: 
                    continue

                if get_angle_diff(angle, other_angle) < max_angle_gap:
                    ox = sum(p[1] for p in other) / len(other)
                    oy = sum(p[2] for p in other) / len(other)

                    # Der Offset wird nun mit dem echten Normalenvektor geprüft
                    offset = abs((ox - bx) * nx + (oy - by) * ny)

                    if offset < max_distance_gap:
                        self.get_logger().info(f"Wand gemergt! (Offset: {offset:.2f}m)")
                        valid_cluster.extend(other)
                        clusters_to_remove.append(other)
                        combined_clusters[i].append(other)
                    else:
                        pass # Optional: self.get_logger().info(f"Offset zu groß: {offset:.2f}m")

            for c in clusters_to_remove:
                remaining_clusters.remove(c)

            # Wir sortieren die Punkte nach ihrer geometrischen Position ENTLANG der Wand,
            # damit Rviz/Foxglove die Linie von Anfang bis Ende sauber durchzeichnet.
            valid_cluster.sort(key=lambda p: p[1] * dir_x + p[2] * dir_y)

        return validated_clusters, combined_clusters

    def get_target_point_straight(self, innenbande, aussenbande):
        """
        Berechnet den Zielpunkt (Karotte) im perfekten Verhältnis zur Innenbande.
        Mit absolutem Mindestabstand (Kraftfeld-Logik).
        """
        target_y = self.lookahead_dist_straight
        target_x = 0.0

        def get_x_at_y(wall, target_y):
            angle = self.get_cluster_angle(wall)
            mean_x = sum(p[1] for p in wall) / len(wall)
            mean_y = sum(p[2] for p in wall) / len(wall)
            if angle is None: return mean_x
            angle_rad = math.radians(angle)
            dy = target_y - mean_y
            return mean_x + (dy * math.tan(angle_rad))

        # ==========================================
        # NEU: DER "WAND-KUSCHLER" FIX (Single-Wall Tracking)
        # ==========================================
        # Wenn wir einem Hindernis ausweichen, verlassen wir uns NUR noch auf 
        # die Bande, an der wir gerade entlangfahren. Das verhindert, dass 
        # das passierte Hindernis die Spurbreiten-Rechnung zerschießt!
        if self.current_obstacle_cmd != "CLEAR":
            if self.lane_ratio < 0.5:
                # Wir wollen ganz nah an die Innenbande (Ratio z.B. 0.20)
                # -> Wir ignorieren die Außenbande (und das Hindernis dort) komplett!
                aussenbande = None
            else:
                # Wir wollen ganz nah an die Außenbande (Ratio z.B. 0.85)
                # -> Wir ignorieren die Innenbande komplett!
                innenbande = None

        # --- FALL 1: WIR SEHEN BEIDE BANDEN (Normalfall auf freier Strecke) ---
        if innenbande and aussenbande:
            x_innen = get_x_at_y(innenbande, target_y)
            x_aussen = get_x_at_y(aussenbande, target_y)
            
            if x_innen < 0: # Innenbande links
                target_x = x_innen + self.lane_ratio
            else:           # Innenbande rechts
                target_x = x_innen - self.lane_ratio
                
        # --- FALL 2: WIR SEHEN NUR DIE INNENBANDE (Oder haben Außen ignoriert) ---
        elif innenbande:
            x_innen = get_x_at_y(innenbande, target_y)
            if x_innen < 0:
                target_x = x_innen + self.lane_ratio
            else:
                target_x = x_innen - self.lane_ratio
                
        # --- FALL 3: WIR SEHEN NUR DIE AUSSENBANDE (Oder haben Innen ignoriert) ---
        elif aussenbande:
            x_aussen = get_x_at_y(aussenbande, target_y)
            inv_ratio = 1.0 - self.lane_ratio 
            if x_aussen < 0: # Außenbande ist links
                target_x = x_aussen + inv_ratio
            else:            # Außenbande ist rechts
                target_x = x_aussen - inv_ratio
                
        # Notfall
        else:
            target_x = 0.0  

        # ==========================================
        # KRAFTFELD (MINDESTABSTAND ERZWINGEN)
        # ==========================================
        if innenbande:
            # Wir prüfen, wo die Innenbande ist
            x_innen_check = get_x_at_y(innenbande, target_y)
            
            if x_innen_check < 0:  
                # Innenbande ist LINKS. Ziel MUSS mindestens +min_wall_dist entfernt sein
                if target_x < x_innen_check + self.min_wall_dist:
                    target_x = x_innen_check + self.min_wall_dist
                    
            else:                  
                # Innenbande ist RECHTS. Ziel MUSS mindestens -min_wall_dist entfernt sein
                if target_x > x_innen_check - self.min_wall_dist:
                    target_x = x_innen_check - self.min_wall_dist

        return (target_x, target_y)
    
    def get_target_point_turn(self, u_profile, x_soll, y_soll):
        if u_profile[1] is None:
            return None
        else:
            dist_front = self.get_closest_point_in_cluster(u_profile[1])[3]

        if self.fahrtrichtung == "links":
            if u_profile[0] is not None:
                dist_side = self.get_closest_point_in_cluster(u_profile[0])[3]
            elif u_profile[2] is not None:
                dist_side = self.get_closest_point_in_cluster(u_profile[2])[3]
            else: 
                dist_side = None
        else:
            if u_profile[2] is not None:
                dist_side = self.get_closest_point_in_cluster(u_profile[2])[3]
            elif u_profile[0] is not None:
                dist_side = self.get_closest_point_in_cluster(u_profile[0])[3]
            else:
                dist_side = None

        delta_x =  dist_side - x_soll if dist_side is not None else None
        delta_y = dist_front - y_soll

        return (delta_x, delta_y)
    
    # Chatler Idee
    def get_target_point_turn(self, u_profile, x_soll, y_soll):
        if u_profile[1] is None:
            return None
            
        # 1. Abstand zur Frontwand
        dist_front = self.get_closest_point_in_cluster(u_profile[1])[3]
        
        # 2. WIE SCHIEF STEHT DER ROBOTER?
        # Die Frontwand sollte idealerweise bei 90° sein.
        angle_front_deg = self.get_cluster_angle(u_profile[1])
        if angle_front_deg is None:
            return None
            
        # Winkel-Differenz in Bogenmaß umrechnen (z.B. Frontwand bei 60° -> alpha = -30°)
        alpha = math.radians(angle_front_deg - 90.0)

        # 3. Seitenabstand bestimmen (Eure bestehende Logik)
        dist_side = None
        is_left_wall = False
        
        if self.fahrtrichtung == "links":
            if u_profile[0] is not None:
                dist_side = self.get_closest_point_in_cluster(u_profile[0])[3]
                is_left_wall = False
            elif u_profile[2] is not None:
                dist_side = self.get_closest_point_in_cluster(u_profile[2])[3]
                is_left_wall = True
        else:
            if u_profile[2] is not None:
                dist_side = self.get_closest_point_in_cluster(u_profile[2])[3]
                is_left_wall = True
            elif u_profile[0] is not None:
                dist_side = self.get_closest_point_in_cluster(u_profile[0])[3]
                is_left_wall = False

        # 4. "Ideale" Deltas berechnen (Skalare Distanzen)
        delta_y_ideal = dist_front - y_soll
        
        if dist_side is not None:
            # Vorzeichen-Logik: Wenn wir uns an der linken Wand orientieren, 
            # muss das Ziel weiter nach links (-X) geschoben werden.
            if is_left_wall:
                delta_x_ideal = -(dist_side - x_soll)
            else:
                delta_x_ideal = (dist_side - x_soll)
        else:
            delta_x_ideal = 0.0

        # 5. DIE LÖSUNG: Rotationsmatrix anwenden!
        # Dreht die Koordinaten passend zur Schräglage des Roboters
        delta_x = delta_x_ideal * math.cos(alpha) - delta_y_ideal * math.sin(alpha)
        delta_y = delta_x_ideal * math.sin(alpha) + delta_y_ideal * math.cos(alpha)

        return (delta_x, delta_y)

    def track_front_wall(self, point_data, last_front_wall):
        """
        Trackt die Frontwand während der Kurve durch ein mitwanderndes Suchfenster (ROI).
        Gibt die aktualisierte Wand und einen Boolean (turn_finished) zurück.
        """
        if not last_front_wall or not point_data:
            return None

        # 1. Wo war die Wand im letzten Frame? (Min/Max Winkel finden)
        angles = [p[0] for p in last_front_wall]
        min_angle = min(angles)
        max_angle = max(angles)
        #self.get_logger().info(f"Tracking-ROI: Min Winkel {min_angle:.1f}°, Max Winkel {max_angle:.1f}°")

        # 2. Das dynamische Suchfenster (ROI) definieren
        # Wir geben in Bewegungsrichtung mehr Toleranz (z.B. +30 Grad), 
        # weil sich die Wand dorthin bewegt. Gegen die Bewegungsrichtung weniger (-10 Grad).
        
        if self.fahrtrichtung == 'links':
            # Wand wandert nach RECHTS (Winkel werden positiver)
            roi_min = min_angle - 5.0
            roi_max = max_angle + 30.0
        else:
            # Wand wandert nach LINKS (Winkel werden negativer)
            roi_min = min_angle - 30.0
            roi_max = max_angle + 5.0

        # 3. Scheuklappen aufsetzen: Punktewolke filtern!
        roi_points = []
        for p in point_data:
            angle = p[0]
            if roi_min <= angle <= roi_max:
                roi_points.append(p)

        # 4. Nur diese gefilterten Punkte in Cluster aufteilen
        roi_clusters = self.get_all_clusters_sorted(roi_points)

        # 5. Tracking überprüfen
        if not roi_clusters:
            self.get_logger().warn("ACHTUNG: Getrackte Wand im ROI verloren!")
            return last_front_wall
            
        # Da wir alle anderen Wände weggefiltert haben, ist das größte Cluster 
        # (Index 0) in diesem Bereich zu 99,9% unsere gesuchte Wand!
        tracked_wall = roi_clusters[0]
        

        return tracked_wall

    def update_avoidance_settings(self):
        """Passt Spur, Kurvenradius und Karotten-Distanz dynamisch an."""
        
        # --- STANDARD-WERTE (Kein Hindernis) ---
        if self.current_obstacle_cmd == "CLEAR" or self.fahrtrichtung is None:
            self.lane_ratio = 0.85 
            self.max_turn_angle = 0.635
            self.lookahead_dist_straight = 0.60  # Entspannt vorausschauen
            
            # WICHTIG: Das hier lieber auskommentieren, sonst spammt es dein Terminal voll!
            # self.get_logger().info("Keine Hindernisse erkannt. Fahre mit Standardparametern.")
            return

        # --- AUSWEICH-WERTE (Adrenalin-Modus) ---
        # Karotte näher ranholen, um viel direkter und schärfer zu lenken!
        self.lookahead_dist_straight = 0.35 

        # Logik-Matrix
        if self.fahrtrichtung == 'links': # Innenbande links
            if self.current_obstacle_cmd == "RED":       # (Vorher AVOID_RIGHT) Rechts vorbei
                self.lane_ratio = 0.85
                self.max_turn_angle = 0.635 # Weite Kurve (Außen)
            elif self.current_obstacle_cmd == "GREEN":   # (Vorher AVOID_LEFT) Links vorbei
                self.lane_ratio = 0.20
                self.max_turn_angle = 0.800 # Enge Kurve (Innen)

        elif self.fahrtrichtung == 'rechts': # Innenbande rechts
            if self.current_obstacle_cmd == "RED":       # (Vorher AVOID_RIGHT) Rechts vorbei
                self.lane_ratio = 0.20
                self.max_turn_angle = 0.800 # Enge Kurve (Innen)
            elif self.current_obstacle_cmd == "GREEN":   # (Vorher AVOID_LEFT) Links vorbei
                self.lane_ratio = 0.85  
                self.max_turn_angle = 0.635 # Weite Kurve (Außen)

    # -----------------------
    # --- YOLO - Function ---
    # -----------------------

    def image_callback(self, msg):
        if self.last_point_data is None: return

        cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        self.image_width = cv_image.shape[1]     
        results = self.model(cv_image, verbose=False)

        # --- NEU: HIER WIRD DAS DEBUG-BILD ERSTELLT UND PUBLIZIERT ---
        # 1. Bounding Boxes und Labels auf das Bild zeichnen
        annotated_frame = results[0].plot()

        # 2. Das OpenCV-Bild zurück in eine ROS-Message konvertieren
        debug_msg = self.bridge.cv2_to_imgmsg(annotated_frame, encoding="bgr8")
        # 3. Das Bild auf dem Topic /camera/yolo_debug veröffentlichen
        self.pub_debug_img.publish(debug_msg)
        # -------------------------------------------------------------
        
        return results
        
    def get_lidar_distance(self, camera_angle_rad, clusters):
        walls = [self.front_wall, self.left_wall, self.right_wall]
        clusters_without_walls = [c for c in clusters if c not in walls and c is not None]
        
        if not clusters_without_walls: 
            return None
            
        best_closest_point = None
        min_dist = 4.0  # WICHTIG: Wir suchen zwingend das NÄCHSTE Objekt!
        best_angle_deg = 0.0
        
        for cluster in clusters_without_walls:
            # FILTER 1: Punkte-Anzahl (Mindestens 2, max 25 für nahe Hindernisse)
            if len(cluster) < 2 or len(cluster) > 25:
                continue
                
            # FILTER 2: Breiten-Filter (WRO-Blöcke sind klein, max 35 cm)
            c_start = cluster[0]   
            c_end = cluster[-1]    
            width = math.hypot(c_start[1] - c_end[1], c_start[2] - c_end[2])
            if width > 0.35:
                continue

            angle_deg = self.middle_of_cluster(cluster)
            if angle_deg is None: continue
            angle_rad = math.radians(angle_deg)
            
            # FILTER 3: Ist das Cluster im 15-Grad-Sichtfeld der Kamera?
            diff = abs(self.angle_diff(angle_rad, camera_angle_rad))
            if diff < math.radians(15.0):
                
                closest_point = self.get_closest_point_in_cluster(cluster)
                if closest_point is not None:
                    dist = closest_point[3]
                    
                    # FILTER 4: VORDERGRUND-PRINZIP! (Verhindert das Springen)
                    # Wir nehmen von allen Clustern im Sichtfeld immer das, das am nächsten ist.
                    if dist < min_dist:
                        min_dist = dist
                        best_closest_point = closest_point
                        best_angle_deg = angle_deg
                        
        # Reparierter Log-Output (Druckt jetzt die echten Werte des Sieger-Clusters!)
        if best_closest_point is not None:
            self.get_logger().info(f"MATCH: Kamera {math.degrees(camera_angle_rad):.1f}° -> Lidar {best_angle_deg:.1f}° (Distanz: {min_dist:.2f}m)")
            return best_closest_point
            
        return None

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
        
## SPielerei###
    def calculate_intersection(self, c1, c2):
        """
        Berechnet den Schnittpunkt absolut robust über die Liniengleichung 
        der Start- und Endpunkte (unabhängig von Winkel-Ungenauigkeiten!).
        """
        if not c1 or not c2 or len(c1) < 2 or len(c2) < 2:
            return None

        # Punkte von Wand 1
        x1, y1 = c1[0][1], c1[0][2]
        x2, y2 = c1[-1][1], c1[-1][2]
        
        # Punkte von Wand 2
        x3, y3 = c2[0][1], c2[0][2]
        x4, y4 = c2[-1][1], c2[-1][2]

        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if abs(denom) < 1e-6:
            return None # Wände sind exakt parallel

        # Schnittpunkt (Px, Py)
        px = ((x1*y2 - y1*x2)*(x3 - x4) - (x1 - x2)*(x3*y4 - y3*x4)) / denom
        py = ((x1*y2 - y1*x2)*(y3 - y4) - (y1 - y2)*(x3*y4 - y3*x4)) / denom

        return (px, py)

    def generate_bezier_path_and_target(self, side_wall, front_wall, fahrtrichtung, is_right_wall, lookahead):
        """
        Generiert den Foxglove-Pfad UND extrahiert direkt den passenden 
        Zielpunkt (Karotte) für die Lenkung.
        """
        intersection = self.calculate_intersection(side_wall, front_wall)
        if not intersection:
            return None

        int_x, int_y = intersection
        
        # Abbiege-Abstand: Wie weit vor der Frontwand soll der neue Flur liegen?
        lane_offset = 0.40 

        # --- KONTROLLPUNKTE (System: X=Rechts, Y=Vorne) ---
        p0 = (0.0, 0.0) # Start am Roboter
        p1 = (0.0, 0.3) # Trägheit: Wir fahren erst einmal geradeaus (+Y)
        
        # Der Scheitelpunkt bleibt auf unserer Spur (X=0), exakt vor der Wand!
        p2 = (0.0, int_y - lane_offset)
        
        if fahrtrichtung == "links":
            # Ziel-Flur geht tief nach LINKS (-X)
            p3 = (-1.5, int_y - lane_offset) 
        else:
            # Ziel-Flur geht tief nach RECHTS (+X)
            p3 = (1.5, int_y - lane_offset) 

        # --- PFAD GENERIEREN & KAROTTE SUCHEN ---
        path_msg = Path()
        path_msg.header.stamp = self.get_clock().now().to_msg()
        path_msg.header.frame_id = self.rviz_frame
        
        target_pt = None

        num_points = 25 
        for i in range(num_points + 1):
            t = i / float(num_points)
            
            term0 = (1 - t)**3
            term1 = 3 * (1 - t)**2 * t
            term2 = 3 * (1 - t) * t**2
            term3 = t**3

            pt_x = term0 * p0[0] + term1 * p1[0] + term2 * p2[0] + term3 * p3[0]
            pt_y = term0 * p0[1] + term1 * p1[1] + term2 * p2[1] + term3 * p3[1]

            pose = PoseStamped()
            pose.header = path_msg.header
            pose.pose.position.x = float(pt_x)
            pose.pose.position.y = float(pt_y)
            pose.pose.position.z = 0.0
            pose.pose.orientation.w = 1.0 
            path_msg.poses.append(pose)

            # Karotte suchen (erster Punkt, der weit genug weg ist)
            if target_pt is None:
                dist_to_robot = math.hypot(pt_x, pt_y)
                if dist_to_robot >= lookahead:
                    target_pt = (pt_x, pt_y)

        if target_pt is None:
            target_pt = (p3[0], p3[1])

        self.pub_path.publish(path_msg)
        return target_pt   
    
    def visualize_clusters(self, kandidaten_RVIZ):
        if self.fahrtrichtung == 'links':
            innenbande = self.left_wall
            aussenbande = self.right_wall
        else:
            innenbande = self.right_wall
            aussenbande = self.left_wall
            
        target_x, target_y = self.get_target_point_straight(innenbande, aussenbande)
        
        marker_array = MarkerArray()

        if self.state == 'FOLLOW_LANE':
            # ALTE LOGIK FÜR DIE GERADE
            if self.fahrtrichtung == 'links':
                innenbande = self.left_wall
                aussenbande = self.right_wall
            else:
                innenbande = self.right_wall
                aussenbande = self.left_wall
                
            target_x, target_y = self.get_target_point_straight(innenbande, aussenbande)
            
            if target_x is not None and target_y is not None:
                # Cyan für Geradeaus-Fahrt
                self.send_sphere(marker_array, m_id=99, x=target_x, y=target_y, color=(0.0, 1.0, 1.0))
                
        elif self.state in ['TURN_LINKS', 'TURN_RECHTS']:
            # NEUE LOGIK FÜR DIE KURVE
            # get_target_point_turn erwartet das u_profile Array!
            u_profile = [self.right_wall, self.front_wall, self.left_wall]
            
            # Zielpunkt-Berechnung aufrufen (z.B. 40cm Soll-Abstand)
            target_pt = self.target_point
            
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
            '''richtung_text = f"LOCKED: {self.fahrtrichtung.upper()}"
            self.send_text(marker_array, m_id=10, text=richtung_text, x=0.0, y=0.0, color=(1.0, 1.0, 0.0)) # Gelb'''

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
                self.send_line(marker_array, m_id=i, p1=start_p, p2=ende_p, color=colors[0])

        # Alles veröffentlichen, damit es in RViz2 auftaucht!
        self.pub_markers.publish(marker_array)

    def calculate_steering_pid(self, target_x, target_y):
        """
        Berechnet den Lenkeinschlag für den Servo basierend auf der Karotte.
        """
        # 1. Fehler berechnen (Winkel zur Karotte)
        # X=Rechts, Y=Vorne. atan2(-X, Y) liefert den Winkel in Radiant.
        # Links lenken = Positiver Winkel, Rechts lenken = Negativer Winkel
        error = math.atan2(-target_x, target_y)

        # 2. PID-Anteile berechnen
        # Wir nutzen deine Variablen aus der __init__ (self.turn_kp etc.)
        p_out = self.turn_kp * error
        
        self.integral_error += error
        i_out = self.turn_ki * self.integral_error
        
        d_out = self.turn_kd * (error - self.prev_error)
        self.prev_error = error

        # 3. Summe ziehen
        steering_cmd = p_out + i_out + d_out
        
        # 4. Clamping (Sicherheitsbegrenzung für den Servo)
        # Verhindert, dass der Servo übersteuert und kaputt geht.
        if steering_cmd > self.max_turn_angle:
            steering_cmd = self.max_turn_angle
        elif steering_cmd < -self.max_turn_angle:
            steering_cmd = -self.max_turn_angle

        return steering_cmd

    def transform_world_to_robot(point_world, robot_pose):
        """
        Transformiert einen Punkt vom Weltkoordinatensystem (Banden-Profil) 
        in das lokale Roboterkoordinatensystem (Lidar).
        
        Parameter:
        point_world : tuple (x_w, y_w) - Der Zielpunkt im Weltsystem in Metern.
        robot_pose  : tuple (x_r, y_r, theta_r) - Die Pose des Roboters im Weltsystem.
                    (x_r, y_r) in Metern, theta_r in Radiant.
                    
        Rückgabe:
        tuple (x_b, y_b) - Die Koordinaten des Punktes relativ zum Roboter.
        """
        x_w, y_w = point_world
        x_r, y_r, theta_r = robot_pose
        
        # 1. Translation berechnen (Verschiebung)
        dx = x_w - x_r
        dy = y_w - y_r
        
        # Trigonometrische Funktionen für die Rotation 
        cos_theta = math.cos(theta_r)
        sin_theta = math.sin(theta_r)
        
        # 2. Rotation anwenden (Rotationsmatrix)
        x_b = dx * cos_theta + dy * sin_theta
        y_b = -dx * sin_theta + dy * cos_theta
        
        return (x_b, y_b)
    
    # -----------------------------------------------
    # Neue Koordinatensystem Idee für Kurvenfahrten
    # -----------------------------------------------

    def cluster_to_hnf(self, cluster):
        """
        Berechnet die Hessesche Normalform aus einem Cluster von Messpunkten.
        Erwartetes Cluster-Format: Liste aus Tupeln (winkel, x_coord, y_coord, abstand)
        """
        if cluster is None:
            return None
        
        # 1. Extrahieren der x- und y-Koordinaten in ein NumPy-Array
        # Index 1 ist x_coord, Index 2 ist y_coord
        points = np.array([[p[1], p[2]] for p in cluster])
        
        # Sicherstellen, dass das Cluster nicht leer ist
        if len(points) == 0:
            raise ValueError("Das Cluster ist leer.")
            
        # 2. Schwerpunkt berechnen
        centroid = np.mean(points, axis=0)
        
        # 3. Daten zentrieren
        centered_points = points - centroid
        
        # 4. Singulärwertzerlegung (SVD) für orthogonale Regression
        # Vh enthält die Eigenvektoren der Kovarianzmatrix
        _, _, Vh = np.linalg.svd(centered_points)
        
        # Der Normalenvektor entspricht dem Eigenvektor der geringsten Varianz
        # Bei der SVD in NumPy ist dies die letzte Zeile von Vh
        normal_vector = Vh[-1]
        
        # 5. Abstand d berechnen (Skalarprodukt aus Schwerpunkt und Normalenvektor)
        d = np.dot(centroid, normal_vector)
        
        # 6. Normierung: d muss größer oder gleich 0 sein
        if d < 0:
            normal_vector = -normal_vector
            d = -d
            
        n_x, n_y = normal_vector
        
        return n_x, n_y, d
    
    def extract_wall_lines(self, u_profile):
        if self.fahrtrichtung == "links":
            side_cluster, front_cluster, opposite_cluster = u_profile
        else:
            opposite_cluster, front_cluster, side_cluster = u_profile

        # 1. Frontwand extrahieren
        # Reihenfolge ist wichtig: 'is None' schützt 'len()' vor Fehlern
        if front_cluster is None or len(front_cluster) < 2:
            front_straight = None
        else:
            front_straight = self.cluster_to_hnf(front_cluster)

        # 2. Seitenwand extrahieren
        if side_cluster is not None and len(side_cluster) >= 2:
            # Primäres Ziel: Die innere Kurvenwand
            side_straight = self.cluster_to_hnf(side_cluster)
            
        elif opposite_cluster is not None and len(opposite_cluster) >= 2:
            # Fallback: Gegenüberliegende Wand über die 3m-Distanz spiegeln
            oppo_x, oppo_y, oppo_d = self.cluster_to_hnf(opposite_cluster)
            
            new_nx = -oppo_x
            new_ny = -oppo_y
            new_d = 3.0 - oppo_d
            
            # HNF-Bedingung: d muss immer >= 0 sein
            if new_d < 0:
                self.get_logger().warn("WARNUNG: HNF-Distanz negativ. Spiegelung korrigiert.")
                new_nx = -new_nx
                new_ny = -new_ny
                new_d = abs(new_d)
                
            side_straight = (new_nx, new_ny, new_d)
            
        else:
            # Keine verwertbaren Seitenwände vorhanden
            side_straight = None

        return front_straight, side_straight

    def calculate_target_line(self, side_line_params, front_line_params, obstacle, desired_wall_distance_m):
        if front_line_params is None:
            return None, None

        n_xf, n_yf, d_f = front_line_params
        
        # 1. Basis-Zielgerade
        d_ziel = d_f - desired_wall_distance_m
        
        # 2. Hindernis auswerten
        if obstacle is not None and len(obstacle.cluster) > 0:
            # Hier folgt die detaillierte Ausweichlogik basierend auf obstacle.farbe
            # ...
            # Beispielhaftes Überschreiben von d_ziel, falls Offset nötig ist:
            # d_ziel = neues_sicheres_d
            pass
            
        target_line_params = (n_xf, n_yf, d_ziel)
        
        if side_line_params is None:
            return target_line_params, None

        # 3. Radius Limitierung
        delta_d_neu = d_f - d_ziel
        n_xs, n_ys, d_s = side_line_params
        
        r_max_ziel = self.TRACK_WIDTH_M - delta_d_neu - (self.ROBOT_WIDTH_M / 2.0)
        
        max_allowed_radius_m = min(r_max_ziel, self.MAX_KINEMATIC_RADIUS_M)
        
        # Sicherstellen, dass der Radius physikalisch fahrbar bleibt (> 0)
        max_allowed_radius_m = max(max_allowed_radius_m, 0.0)
        
        return target_line_params, max_allowed_radius_m
    
    def get_intersection_point(self, target_line_params):
        """
        Berechnet den Schnittpunkt der y-Achse (Roboter-Trajektorie) mit der Zielgeraden.
        """
        if target_line_params is None:
            return None, None, None
            
        n_x, n_y, d = target_line_params
        
        # Sicherheitsprüfung auf Parallelität (Vermeidung von Division durch Null)
        epsilon = 1e-6
        if abs(n_y) < epsilon:
            # Zielgerade ist parallel zur Fahrtrichtung, kein Schnittpunkt
            return None, None, None
            
        # Schnittpunkt berechnen
        intersection_x_m = 0.0  # Roboter fährt per Definition auf x=0
        intersection_y_m = d / n_y
        
        # Plausibilitätsprüfung: Schnittpunkt muss vor dem Roboter (in Fahrtrichtung) liegen
        if intersection_y_m <= 0.0:
            # Mathematischer Schnittpunkt liegt hinter dem Roboter. 
            # Indiziert Fehler im Lidar-Clustering oder Odometrie-Sprung.
            return None, None, None
        
        # 1. Vorzeichen des Skalarprodukts basierend auf der Kurvenrichtung setzen
        if self.fahrtrichtung == 'links':
            nx_directional = n_x
        else:
            nx_directional = -n_x
            
        # 2. Clipping auf [-1.0, 1.0], aber OHNE den Betrag (abs)
        nx_clipped = max(-1.0, min(1.0, nx_directional))
        
        # 3. Winkel berechnen (liefert jetzt Werte zwischen 0° und 180°)
        turn_angle_deg = math.degrees(math.acos(nx_clipped))

        return intersection_x_m, intersection_y_m, turn_angle_deg

    def calculate_curve_geometry(self, intersection_y_m, turn_angle_deg, max_allowed_radius_m):
        """
        Berechnet den optimalen Kurvenradius und die Distanz zum Einlenkpunkt auf der y-Achse.
        """
        
        if intersection_y_m is None:
            # Kurve ist geometrisch oder mechanisch unmöglich
            self.get_logger().error("Ungültiger Schnittpunkt: Kein Einlenken möglich.")
            return None, None

        if max_allowed_radius_m is None:
            max_allowed_radius_m = self.MAX_KINEMATIC_RADIUS_M

        if max_allowed_radius_m < self.MIN_TURN_RADIUS_M:
            curve_radius_m = self.MIN_TURN_RADIUS_M # Fehler: Mögliche Kollison mit der Wand, da der Platz nicht ausreicht
            # return None, None 
        else:
            curve_radius_m = max(self.MIN_TURN_RADIUS_M, min(self.IDEAL_RADIUS_M, max_allowed_radius_m))
        # Radius wird zwischen dem mechanischen Minimum und dem Platz-Maximum eingeklemmt
        
        
        # 2. Tangentenlänge (Distanz vom Schnittpunkt zum Einlenkpunkt) berechnen
        alpha_rad = math.radians(abs(turn_angle_deg))
        tangent_length_m = curve_radius_m * math.tan(alpha_rad / 2.0)
        
        # 3. Finalen Einlenkpunkt auf der y-Achse festlegen
        entry_point_distance_m = intersection_y_m - tangent_length_m
        
        return curve_radius_m, entry_point_distance_m

    def check_turn_trigger(self, entry_point_distance_m):
        """
        Prüft anhand der aktuell berechneten Distanz, ob das Einlenkmanöver starten muss.
        """
        if entry_point_distance_m is None:
            return False

        # Toleranzwert (z.B. 7 cm), um Verzögerungen in der Hauptschleife abzufangen
        trigger_tolerance_m = 0.07 

        if entry_point_distance_m <= trigger_tolerance_m:
            return True
            
        return False

    def execute_turn(self, curve_radius_m):
        """
        Übersetzt den Radius über den SteeringController und publisht die Twist-Message.
        """

        if self.fahrtrichtung == "links":
            is_left_turn = True
        else:
            is_left_turn = False

        cmd = Twist()
        
        # 1. Sicherheitscheck auf ungültige Geometrie
        if curve_radius_m is None or curve_radius_m <= 0.0:
            self.get_logger().error("Ungültiger Kurvenradius.")
            return False
        
        # 3. Abruf des kalibrierten PWM-Signals (-1.0 bis 1.0)
        steering_signal = self.steering_ctrl.get_steering_for_radius(target_radius=curve_radius_m,fahrtrichtung_ist_links=is_left_turn)
        self.get_logger().info(f"Steering-Signal: {steering_signal:.3f}")

        # 4. Twist-Message konstruieren und senden
        cmd.linear.x = float(self.turn_speed)
        #cmd.linear.x = 0.0  # MOTOR AUS FÜR DEN TEST!
        cmd.angular.z = float(steering_signal)
        
        self.pub_cmd_vel.publish(cmd)
        
        return True

    def check_turn_completion_fused(self, turn_angle, front_line_params):
        """
        Kombiniert Gyro-Daten mit dem Wandwinkel für maximale Präzision am Kurvenausgang.
        """
        # 1. Grobe Prüfung via Gyro (Delta-Berechnung wie zuvor)
        error_gyro = abs(turn_angle) - abs(self.current_yaw - self.start_turn_yaw)
        self.get_logger().info(f"Gyro Error: {error_gyro:.1f}°, target: {turn_angle:.1f}°")
        error_gyro = abs(error_gyro)
        # Wenn wir noch nicht einmal 75 Grad gedreht haben, sicher noch nicht fertig
        if error_gyro > 180:
            return False
        
        # 2. Präzise Prüfung via LiDAR-Winkel (falls Wand sichtbar)
        if front_line_params is not None:
            n_x, n_y, d = front_line_params
            # Das entspricht einem Orientierungsfehler zur Wand von:
            wall_error_deg = math.degrees(math.atan2(abs(n_y), abs(n_x)))
            self.get_logger().info(f"Wall-Error {wall_error_deg:.1f}°")
            # Abbruchbedingung: Gyro ist nah dran UND Wand-Parallelität ist hoch
            if wall_error_deg < 6.0:
                self.get_logger().info(f"Fused Match: Gyro {error_gyro:.1f}°, Wall-Error {wall_error_deg:.1f}°")
                return True

        # 3. Fallback: Nur Gyro (falls LiDAR die Wand kurz verliert)
        if error_gyro <= 7.0:
            self.get_logger().warn("Nur Gyro-Abschluss (Wand nicht erkannt)")
            return True

        return False

    # -----------------------------------------------
    # Testfunktionen
    # -----------------------------------------------

    def visualize_hnf_line(self, hnf_params, m_id, farbe_name="rot", label="HNF_LINE"):
        """
        Erstellt einen Marker für eine Gerade aus der Hesseschen Normalform (n_x, n_y, d).
        
        Args:
            hnf_params: Tuple (n_x, n_y, d)
            m_id: Eindeutige ID für den Marker
            farbe_name: "rot", "gruen" oder "blau"
            label: Textbeschriftung an der Linie
        """
        if hnf_params is None:
            return

        # Farb-Mapping (RGB-Werte normiert auf 0.0 - 1.0)
        farben = {
            "rot": (1.0, 0.0, 0.0),
            "gruen": (0.0, 1.0, 0.0),
            "blau": (0.0, 0.5, 1.0)
        }
        
        # Fallback auf Weiß, falls die Farbe nicht im Dictionary ist
        rgb = farben.get(farbe_name.lower(), (1.0, 1.0, 1.0))
        
        n_x, n_y, d = hnf_params
        
        # 1. Lotpunkt berechnen (Punkt auf der Linie am nächsten zum Ursprung)
        p_lot_x = n_x * d
        p_lot_y = n_y * d
        
        # 2. Richtungsvektor der Geraden (senkrecht zum Normalenvektor)
        # Da y vorne ist: n=(nx, ny) -> v=(-ny, nx)
        v_x = -n_y
        v_y = n_x
        
        # 3. Zwei Punkte für eine 4 Meter lange Linie (2m in jede Richtung vom Lotpunkt)
        p1 = (p_lot_x + v_x * 3.0, p_lot_y + v_y * 3.0)
        p2 = (p_lot_x - v_x * 3.0, p_lot_y - v_y * 3.0)
        
        # 4. Marker-Array vorbereiten
        marker_array = MarkerArray()
        
        # 5. Linie zeichnen (Nutzt deine interne send_line Methode)
        self.send_line(marker_array, m_id=m_id, p1=p1, p2=p2, color=rgb)
        
        # 6. Text-Label am Lotpunkt (ID versetzt, damit Text und Linie koexistieren)
        self.send_text(marker_array, m_id=m_id + 1000, text=label, x=p_lot_x, y=p_lot_y, color=rgb)
        
        # 7. Veröffentlichen
        self.pub_markers.publish(marker_array)

    def test_extract_wall_lines(self, u_profile):
        '''Testet die Funktion extract_wall_lines() und visualisiert die Ergebnisse in RViz.'''
        front_straight, side_straight = self.extract_wall_lines(u_profile)
        self.get_logger().info(f"Front HNF: {front_straight}")
        self.get_logger().info(f"Side HNF: {side_straight}")
        self.visualize_hnf_line(front_straight, m_id=200, farbe_name="blau", label="")
        self.visualize_hnf_line(side_straight, m_id=300, farbe_name="blau", label="")
        return front_straight, side_straight


    def test_calculate_target_line(self, front_wall_cluster, side_wall_cluster, desired_wall_distance_m):
        '''Testet die Funktion calculate_target_line() und visualisiert die Ergebnisse in RViz.'''
        front_line_params, side_line_params = self.extract_wall_lines(front_wall_cluster, side_wall_cluster)
        target_line_params, max_radius = self.calculate_target_line(side_line_params, front_line_params, None, desired_wall_distance_m)
        radius = f"{max_radius:.2f}" if max_radius is not None else "None"
        self.get_logger().info(f"Target Line HNF: {target_line_params}, Max Radius: {radius} m")
        self.visualize_hnf_line(target_line_params, m_id=400, farbe_name="gruen", label="")
        return target_line_params, max_radius

    def test_get_intersection_point(self, target_line_params):
        '''Testet die Funktion get_intersection_point() und visualisiert den Schnittpunkt in RViz.'''
        intersection_x, intersection_y, angle = self.get_intersection_point(target_line_params)
        if intersection_x is not None and intersection_y is not None:
            self.get_logger().info(f"Schnittpunkt: (X={intersection_x:.2f}, Y={intersection_y:.2f}), Schnittwinkel: {angle:.2f}°")
            marker_array = MarkerArray()
            self.send_sphere(marker_array, m_id=500, x=intersection_x, y=intersection_y, color=(1.0, 1.0, 0.0)) # Gelb
            self.pub_markers.publish(marker_array)
            return intersection_x, intersection_y, angle
        
        else:
            self.get_logger().warn("Kein gültiger Schnittpunkt gefunden.")
            return None, None, None

    def test_calculate_curve_geometry(self, intersection_y_m, turn_angle_deg, max_allowed_radius_m):
        """Testet die Funktion calculate_curve_geometry() und loggt die Ergebnisse."""
        
        curve_radius_m, entry_point_distance_m = self.calculate_curve_geometry(intersection_y_m, turn_angle_deg, max_allowed_radius_m)
        
        radius_str = f"{curve_radius_m:.2f} m" if curve_radius_m is not None else "None"
        entry_str = f"{entry_point_distance_m:.2f} m" if entry_point_distance_m is not None else "None"
        self.get_logger().info(f"Berechnete Kurvengeometrie: Radius = {radius_str}, Einlenkpunkt-Distanz = {entry_str}")
        
        marker_array = MarkerArray()
    
        if entry_point_distance_m is not None and intersection_y_m is not None:
            # Einlenkpunkt (Start)
            self.send_sphere(marker_array, m_id=900, x=0.0, y=entry_point_distance_m, color=(0.0, 1.0, 0.0))
            
            # Logische Richtung ermitteln
            is_left = (self.fahrtrichtung == 'links')
            
            # GeoGebra Winkel-Sektor zeichnen (Radius 0.4 m)
            self.visualize_geogebra_angle(marker_array, intersection_y_m, turn_angle_deg, is_left_turn=is_left, m_id=960, radius=0.4)
            
            # Text-Label für den Winkel am Rand des Bogens platzieren
            text_x = -0.45 if is_left else 0.45
            self.send_text(marker_array, m_id=961, text=f"{turn_angle_deg:.1f}°", 
                        x=text_x, y=intersection_y_m + 0.2, color=(1.0, 1.0, 0.0))
        else:
            self.delete_marker(marker_array, m_id=900)
            self.delete_marker(marker_array, m_id=960)
            self.delete_marker(marker_array, m_id=961)
            
        if hasattr(self, 'pub_markers'):
            self.pub_markers.publish(marker_array)

        return curve_radius_m, entry_point_distance_m

    def test_check_turn_trigger(self, entry_point_distance_m):
        """Testet die Funktion check_turn_trigger() und loggt, ob der Trigger ausgelöst wird."""
        trigger = self.check_turn_trigger(entry_point_distance_m)
        status = "TRIGGERED" if trigger else "NOT TRIGGERED"
        entry_point_distance_m = f"{entry_point_distance_m:.2f} m" if entry_point_distance_m is not None else "None"
        self.get_logger().warn(f"Turn Trigger Check: {status} (Entry Point Distance: {entry_point_distance_m})")
        return trigger

    def test_execute_turn(self, curve_radius_m):
        """Testet die Funktion execute_turn() und loggt das Ergebnis."""
        success = self.execute_turn(curve_radius_m)
        if success:
            self.get_logger().info(f"Turn executed with radius {curve_radius_m:.2f} m")
        else:
            self.get_logger().error("Failed to execute turn.")
        return success

    def test_check_turn_completion_fused(self, target_angle, front_line_params):
        """Testet die Funktion check_turn_completion_fused() und loggt, ob die Kurve als abgeschlossen gilt."""
        completed = self.check_turn_completion_fused(target_angle, front_line_params)
        status = "COMPLETED" if completed else "IN PROGRESS"
        self.get_logger().warn(f"Turn Completion Check: {status} (Current Gyro: {self.current_yaw:.2f}°, Target Angle: {target_angle:.2f}°)")
        return completed

    def are_clusters_near(self, cluster_a, cluster_b, threshold=0.02):
        # Mittelpunkt A
        avg_a = np.mean(cluster_a, axis=0) # [x_avg, y_avg]
        # Mittelpunkt B
        avg_b = np.mean(cluster_b, axis=0)
        
        # Euklidische Distanz
        dist = np.linalg.norm(avg_a - avg_b)
        return dist < threshold
    
    def visualize_geogebra_angle(self, marker_array, intersection_y_m, turn_angle_deg, is_left_turn, m_id=960, radius=0.4):
        """
        Zeichnet den Schnittwinkel als Kreissektor und platziert die Gradzahl mittig darin.
        """
        # 1. Abbruchbedingung und Löschen der Marker
        if intersection_y_m is None or turn_angle_deg is None:
            self.delete_marker(marker_array, m_id)      # Bogen löschen
            self.delete_marker(marker_array, m_id + 1)  # Text löschen
            return

        # 2. Bogen-Marker einrichten
        marker = Marker()
        marker.header.frame_id = "base_link" 
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "angle_arc"
        marker.id = m_id
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        marker.scale.x = 0.015 
        marker.color.r, marker.color.g, marker.color.b, marker.color.a = 0.59, 0.0, 1.0, 1.0

        # 3. Winkel berechnen
        start_angle_rad = math.pi / 2.0  # y-Achse = 90 Grad
        turn_rad = math.radians(turn_angle_deg)
        
        if is_left_turn:
            end_angle_rad = start_angle_rad + turn_rad
        else:
            end_angle_rad = start_angle_rad - turn_rad

        # 4. Bogen zeichnen
        center_p = Point(x=0.0, y=intersection_y_m, z=0.0)
        marker.points.append(center_p)

        num_steps = 15
        angle_diff = end_angle_rad - start_angle_rad
        for i in range(num_steps + 1):
            current_angle = start_angle_rad + (i / num_steps) * angle_diff
            p = Point()
            p.x = radius * math.cos(current_angle)
            p.y = intersection_y_m + radius * math.sin(current_angle)
            p.z = 0.0
            marker.points.append(p)
            
        marker.points.append(center_p)
        marker_array.markers.append(marker)

        # 5. NEU: Text in der Winkelhalbierenden platzieren
        mid_angle_rad = (start_angle_rad + end_angle_rad) / 2.0
        
        # Text-Radius auf 60% des Bogen-Radius setzen, damit er im Sektor liegt
        text_radius = radius * 0.6 
        text_x = text_radius * math.cos(mid_angle_rad)
        text_y = intersection_y_m + text_radius * math.sin(mid_angle_rad)
        
        # Nutzt m_id + 1, um nicht mit dem Bogen-Marker zu kollidieren
        self.send_text(marker_array, m_id=m_id + 1, text=f"{turn_angle_deg:.1f}°", x=text_x, y=text_y, color=(0.59, 0.0, 1.0))

    def test_turn_main_logic(self, point_data):
        self.fahrtrichtung = "links"
        self.state = 'TURN_LINKS'  # Wir testen die Linkskurve
        
        # Initialisiere die Test-Variable dynamisch, falls nicht vorhanden
        if not hasattr(self, 'test_is_turning'):
            self.test_is_turning = False

        '''if self.counter  < 1:
            self.counter += 1
            return
        '''
        # 1. Alle Cluster finden
        all_clusters = self.get_all_clusters_sorted(point_data)
        
        if self.begin == True:
            for c in all_clusters:
                # Cluster in Foxglove anzeigen, BEVOR wir auf Input warten!
                self.visualize_clusters([c])
                
                skip = input("Drücke Enter, um zum nächsten Cluster zu gehen (oder 'q' für Auswahl)...")
                if skip.lower() in ['q', '#']:
                    self.begin = False
                    self.get_logger().info(">>> Starte Test Bench mit der gewählten Frontwand! <<<")
                    self.front_wall = c
                    
                    # Gyro Nullen für saubere Kurven-Mathematik
                    #self.start_turn_yaw = self.current_yaw
                    self.start_turn_dist = self.get_closest_point_in_cluster(c)[3]
                    self.get_logger().warn(f"Start Distanz ist {self.start_turn_dist:.2f}m")
                    
                    # PID-Regler nullen
                    self.integral_error = 0.0
                    self.prev_error = 0.0
                    break
                else:
                    continue
        
        # ==========================================
        # KONTINUIERLICHE BERECHNUNG (läuft immer)
        # ==========================================
        self.front_wall = self.track_front_wall(point_data, self.front_wall)
        validated_clusters = self.validate_clusters_turn(self.front_wall, point_data)
        
        front_wall_params, side_wall_params = self.test_extract_wall_lines(validated_clusters)
    
        # ==========================================
        # DIE NEUE TEST-LOGIK FÜR DEN SERVO/GYRO
        # ==========================================
        if not self.test_is_turning:
            # WARTE-MODUS: Zeigt Live-Daten, bis 't' gedrückt wird
            target_line_params, max_allowed_radius_m = self.test_calculate_target_line(self.front_wall, außenbande, desired_wall_distance_m=0.8)
            self.visualize_hnf_line((1.0, 0, 0), m_id=550, farbe_name="rot", label="")

            intersection_x, intersection_y, schnitt_winkel = self.test_get_intersection_point(target_line_params)
            self.curve_radius_m, entry_point_distance_m = self.test_calculate_curve_geometry(intersection_y, schnitt_winkel, max_allowed_radius_m)
        
            self.test_check_turn_trigger(entry_point_distance_m)

            start = input("Drücke 'q' + Enter für Kurven-Start (Servo), oder nur Enter für nächsten Frame: ")
            
            if start.lower() == 'q':
                self.test_is_turning = True
                self.saved_target_angle = schnitt_winkel
                self.start_turn_yaw = self.current_yaw
                self.get_logger().warn(">>> KURVE GESTARTET - DREHE DEN ROBOTER NUN PER HAND <<<")
                
        else:
            # AUSFÜHRUNGS-MODUS: Servo schlägt ein, System wartet auf Gyro-Vollendung
            if self.curve_radius_m is not None:
                # WICHTIG: test_execute_turn MUSS cmd.linear.x = 0.0 setzen!
                self.test_execute_turn(self.curve_radius_m)
            
            
            # Prüfung: Hast du den Roboter weit genug per Hand gedreht?
            self.get_logger().info(f"Alte Winkelberchnung: {self.get_cluster_angle(self.front_wall)}")
            turn_completed = self.test_check_turn_completion_fused(self.saved_target_angle, front_wall_params)
            
            if turn_completed:
                self.get_logger().info(">>> KURVE BEENDET! SETZE SERVO ZURÜCK <<<")
                self.test_is_turning = False
                
                # Servo sofort auf Neutral stellen
                from geometry_msgs.msg import Twist
                stop_cmd = Twist()
                stop_cmd.linear.x = 0.0
                stop_cmd.angular.z = 0.0 
                self.pub_cmd_vel.publish(stop_cmd)
                
                # Testzyklus beenden und neu starten
                input("Test abgeschlossen. Drücke Enter für neue Wand-Auswahl...")
                self.begin = True  # Startet die Cluster-Auswahl neu

        self.counter = 0
        return
        # ==========================================

    # -----------------------------------------
    # State Maschine Obstacle Parcour
    # -----------------------------------------

    def check_for_obstacles(self, point_data):
        return "CLEAR"


    def handle_lane_following(self, point_data, all_clusters, current_obstacle_status):
        cmd = Twist()

        self.check_undetected_turn()
        validated_clusters = self.validate_clusters_straight(all_clusters)
        merged_validated_clusters = self.merge_clusters(all_clusters, validated_clusters)
        self.right_wall = merged_validated_clusters[2]
        self.front_wall = merged_validated_clusters[1]
        self.left_wall  = merged_validated_clusters[0]
        right_wall_hnf = self.cluster_to_hnf(self.right_wall)
        front_wall_hnf = self.cluster_to_hnf(self.front_wall)
        left_wall_hnf = self.cluster_to_hnf(self.left_wall)
        if self.fahrtrichtung == 'links':
            innenbande = self.left_wall
            aussenbande = self.right_wall
        elif self.fahrtrichtung == 'rechts':
            innenbande = self.right_wall
            aussenbande = self.left_wall

        if current_obstacle_status != "CLEAR":
            pass
        else:
            if self.front_wall_hnf is not None and self.fahrtrichtung is not None:
                _, _, front_dist = front_wall_hnf
                
                if front_dist < 1.20:
                    self.state = f"TURN_{self.fahrtrichtung.upper()}"
                    self.start_turn_yaw = self.current_yaw
                    self.get_logger().warn(f">>> {self.state} EINGELEITET bei {(self.start_turn_yaw - self.yaw_offset):.1f}° <<<")
                    # WICHTIG: PID-Gedächtnis für die nächste Gerade löschen!
                    self.prev_error = 0.0
                    self.integral_error = 0.0
                else:
                    if front_dist < 1.40:
                        self.get_logger().info(f"Warte auf Ecke... (Frontwand ist noch {front_dist:.2f}m entfernt)")
            
            target_x, target_y = self.get_target_point_straight(innenbande, aussenbande)

            # 2. PID-REGLER BERECHNEN
            # Fehler: X-Abweichung der Karotte. Negatives X = Karotte links = Positiv lenken!
            error = -target_x 
            
            # Integral berechnen (mit Anti-Windup, damit der Wert nicht explodiert)
            self.integral_error += error
            self.integral_error = max(-1.0, min(1.0, self.integral_error))
            
            # Derivative berechnen (Veränderung zum letzten Frame)
            derivative = error - self.prev_error
            self.prev_error = error
            
            # Stellgröße (Lenkbefehl) berechnen
            steering_cmd = (self.kp * error) + (self.ki * self.integral_error) + (self.kd * derivative)
            
            # Auf ROS-Grenzen (-1.0 bis 1.0) kappen
            steering_cmd = max(-1.0, min(1.0, steering_cmd))
            
            # 3. BEFEHLE AN ESP SETZEN
            cmd.linear.x = self.base_speed
            cmd.angular.z = float(steering_cmd)

    def handle_turn_maneuver(self, point_data):
        """
        Kapselt die gesamte Logik für Kurven: Berechnung, Trigger, Ausführung und Abschluss.
        """
        # ---------------------------------------------------------
        # PHASE 1: ANNÄHERUNG (Berechnen & auf Trigger warten)
        # ---------------------------------------------------------
        if self.turn_phase == 'APPROACH':
            # 1. Wände tracken und Geometrie berechnen
            self.front_wall = self.track_front_wall(point_data, self.front_wall)
            validated_clusters = self.validate_clusters_turn(self.front_wall, point_data)
            
            front_wall_params, side_wall_params = self.extract_wall_lines(validated_clusters)
            target_line_params, max_allowed_radius = self.calculate_target_line(front_wall_params, side_wall_params, desired_wall_distance_m=0.50)
            
            intersection_x, intersection_y, turn_angle = self.get_intersection_point(target_line_params)
            
            curve_radius_m, entry_distance_m = self.calculate_curve_geometry(intersection_y, turn_angle, max_allowed_radius)
            
            # 2. Trigger prüfen
            # check_turn_trigger muss True zurückgeben, wenn die Distanz erreicht ist
            if self.check_turn_trigger(entry_distance_m):
                self.get_logger().info("Trigger erreicht. Wechsle in EXECUTE-Phase.")
                # Daten für die Abschlussprüfung einfrieren
                self.saved_target_angle = turn_angle
                self.saved_curve_radius_m = curve_radius_m
                self.turn_phase = 'EXECUTE'
                
        # ---------------------------------------------------------
        # PHASE 2: AUSFÜHRUNG (Servo lenkt, Gyro prüft)
        # ---------------------------------------------------------
        elif self.turn_phase == 'EXECUTE':
            # 1. Servo-Befehl kontinuierlich senden
            self.execute_turn(self.saved_curve_radius_m)
            self.front_wall = self.track_front_wall(point_data, self.front_wall)
            front_wall_params = self.cluster_to_hnf(self.front_wall)


            # 3. Abschluss prüfen
            if self.check_turn_completion_fused(self.saved_target_angle, front_wall_params):
                self.get_logger().info("Kurve physikalisch beendet.")
            
                # Aufräumen und State zurücksetzen
                self.turn_phase = 'APPROACH' # Reset für die nächste Kurve
                self.state = 'FOLLOW_LANE'   # Rückgabe der Kontrolle an die main_logic
    
    def execute_stop(self):
            cmd = Twist()
            cmd.linear.x = 0.0
            cmd.angular.z = 0.0
            self.pub_cmd_vel.publish(cmd)
            self.get_logger().warn(f">>> ZIEL ERREICHT: {self.turn_count} Kurven geschafft! Stoppe den Roboter. <<<")
            self.destroy_node()
            rclpy.shutdown()
            return

    def main_logic(self, point_data):
        # ... (Grundlegende LiDAR-Datenvorbereitung, falls für alle States nötig) ...

        current_obstacle_status = self.check_for_obstacles(point_data)

        if self.state == 'STARTING':
            self.execute_start(point_data)

        elif self.state == 'FOLLOW_LANE':
            self.handle_lane_following(point_data)
            
        elif self.state in ['TURN_LINKS', 'TURN_RECHTS']:
            self.handle_turn_maneuver(point_data, self.state)
            
        elif self.state == 'STOPPED':
            self.execute_stop()




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