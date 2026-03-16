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
        self.default_max_turn = 0.635
        

        # --- PID-REGLER PARAMETER ---
        self.kp = 3.5   # Lenkt hart zur Karotte
        self.kd = 0   # Verhindert das Schlingern (Dämpfung)
        self.ki = 0.0   # Integral (oft bei WRO auf 0 gelassen, da schnelle Spurwechsel)
        
        self.prev_error = 0.0
        self.integral_error = 0.0

        # Karotten-Parameter Kurve
        self.lookahead_dist_turn = 0.25

        # --- TURN PID-REGLER PARAMETER ---
        self.turn_kp = 1.0
        self.turn_kd = 0.0
        self.turn_ki = 0.0

        # --- MOTOR PARAMETER (ESP PWM 0 - 1023) ---
        self.base_speed = 450.0  # Normale Geschwindigkeit auf der Geraden
        self.turn_speed = 450.0  # Leicht reduzierter Speed in der Kurve

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
            self.get_logger().warn(f"?")
            return u_profile

        self.get_logger().warn(f"Kein Cluster außer der Frontwand gefunden.")
        return [None, front_wall_cluster, None]
    
    def sort_clusters_right_to_left(self, clusters):
        """
        Nimmt eine Liste von Clustern und sortiert sie räumlich von rechts nach links.
        Voraussetzung: Tupel-Format (angle, x, y, dist)
        """
        if not clusters:
            return []

        def get_cluster_bearing(cluster):
            # 1. Schwerpunkt des Clusters berechnen
            # Index 1 = X, Index 2 = Y
            mean_x = sum(p[1] for p in cluster) / len(cluster)
            mean_y = sum(p[2] for p in cluster) / len(cluster)
            
            return math.atan2(mean_y, mean_x)

        sorted_clusters = sorted(clusters, key=get_cluster_bearing, reverse=True)
        
        return sorted_clusters

    def get_all_clusters_sorted(self, point_data):
        """
        Sucht ALLE zusammenhängenden Cluster und durchtrennt sie an 90°-Ecken.
        Nutzt die Manhattan-Norm und X/Y-Gradienten-Überwachung (Vorausschauend).
        point_data: Liste aus (Winkel_deg, x, y, dist)
        """
        if len(point_data) < 2:
            return []

        sorted_points = sorted(point_data, key=lambda p: p[0])

        # --- PARAMETER ---
        max_gap = 0.10      
        outlier_limit = 2
        corner_sensitivity = 0.05 
        lookahead_steps = 5  # NEU: Wie viele Punkte schauen wir in die Zukunft?
        
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
            
            # 1. MANHATTAN-DISTANZ (|dx| + |dy|) zum aktuellen Punkt
            dx = abs(p_curr[1] - p_last[1])
            dy = abs(p_curr[2] - p_last[2])
            dist_manhattan = dx + dy
            
            if dist_manhattan < max_gap:
                
                # =========================================================
                # 2. NEUE ECKEN-ERKENNUNG (Blick in die Zukunft!)
                # =========================================================
                is_corner = False
                
                # Wir prüfen nur auf Ecken, wenn wir schon eine saubere Basis haben (10 Punkte)
                # UND wenn wir noch genug Punkte in der Zukunft haben (5 Punkte).
                if len(current_cluster) >= 10 and (i + lookahead_steps) <= len(sorted_points):
                    
                    # --- A. VERGANGENHEIT (Der Trend der bisherigen Wand) ---
                    p_base = current_cluster[-10] 
                    dx_past = p_last[1] - p_base[1]
                    dy_past = p_last[2] - p_base[2]
                    angle_past = math.atan2(dy_past, dx_past)
                    
                    # --- B. ZUKUNFT (Die nächsten 5 Punkte inkl. dem aktuellen) ---
                    future_points = sorted_points[i : i + lookahead_steps]
                    
                    # --- C. DISTANZ-PRÜFUNG DER ZUKUNFT ---
                    # Bevor wir Winkel berechnen, MÜSSEN wir sichergehen, dass die Zukunfts-Punkte 
                    # nicht einfach eine Lücke/ein Loch in der Bande sind!
                    valid_future = True
                    for f_idx in range(1, len(future_points)):
                        f_dx = abs(future_points[f_idx][1] - future_points[f_idx-1][1])
                        f_dy = abs(future_points[f_idx][2] - future_points[f_idx-1][2])
                        if (f_dx + f_dy) >= max_gap:
                            valid_future = False  # Lücke erkannt! Das ist keine verbundene Ecke.
                            break
                            
                    if valid_future:
                        # --- D. WINKEL DER ZUKUNFT BERECHNEN ---
                        p_future_end = future_points[-1]
                        dx_future = p_future_end[1] - p_curr[1]
                        dy_future = p_future_end[2] - p_curr[2]
                        angle_future = math.atan2(dy_future, dx_future)
                        
                        # --- E. WINKEL VERGLEICH ---
                        diff_rad = abs(angle_past - angle_future)
                        diff_deg = math.degrees(diff_rad)
                        if diff_deg > 180:
                            diff_deg = 360 - diff_deg
                            
                        # Hat sich der Punkt physikalisch auch weit genug bewegt?
                        dist_moved = math.hypot(dx_future, dy_future)
                        
                        # Wenn die Zukunfts-Punkte stark abbiegen (z.B. > 25 Grad)
                        if diff_deg > 25.0 and dist_moved > corner_sensitivity:
                            is_corner = True

                # =========================================================
                # 3. ENTSCHEIDUNG TREFFEN
                # =========================================================
                if is_corner:
                    # ZUKUNFTS-ECKE ERKANNT! 
                    # Wir packen die SAUBERE Wand weg, p_curr gehört schon zur neuen Wand.
                    clusters.append(current_cluster)
                    current_cluster = []
                    i += lookahead_steps
                    continue
                else:
                    # Keine Ecke in Sicht, Punkt normal zur aktuellen Wand hinzufügen
                    current_cluster.append(p_curr)
                    i += 1
                    
            else:
                # 4. AUSREISSER-LOGIK (Lücken überbrücken) bleibt exakt wie vorher!
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

        # 5. WRAP-AROUND-FIX (Mit Manhattan)
        if len(clusters) > 1:
            first_p = clusters[0][0]
            last_p = clusters[-1][-1]
            dist_wrap = abs(first_p[1] - last_p[1]) + abs(first_p[2] - last_p[2])
            
            if dist_wrap < max_gap:
                clusters[0] = clusters[-1] + clusters[0]
                clusters.pop()

        clusters.sort(key=len, reverse=True)
        return clusters
 
    def get_cluster_angle(self, cluster):
        """
        Berechnet den Durchschnittswinkel aller Punkte im Cluster (Ausgleichsgerade).
        Vorne/Hinten (Seitenbanden) = 0°
        Links/Rechts (Frontwand) = 90° oder -90°
        """

        if cluster is None:
            return None
        
        n = len(cluster)
        if n < 2:
            return None

        

        # 1. Schwerpunkt berechnen
        mean_x = sum(p[1] for p in cluster) / n
        mean_y = sum(p[2] for p in cluster) / n

        # 2. Abweichungen vom Schwerpunkt
        s_xx = 0.0
        s_yy = 0.0
        s_xy = 0.0
        
        for p in cluster:
            dx = p[1] - mean_x
            dy = p[2] - mean_y
            s_xx += dx * dx
            s_yy += dy * dy
            s_xy += dx * dy

        # 3. FIX: Wir tauschen s_xx und s_yy im Nenner! 
        # Dadurch berechnen wir den Winkel relativ zur Y-Achse (Fahrtrichtung) statt zur X-Achse.
        angle_rad = 0.5 * math.atan2(2.0 * s_xy, s_yy - s_xx)
        
        # In Grad umrechnen
        angle_deg = math.degrees(angle_rad)

        return angle_deg

    def delete_marker(self, marker_array, m_id, ns="walls"):
        """Löscht einen Marker in einem bestimmten Namespace."""
        marker = Marker()
        marker.header.frame_id = self.rviz_frame
        marker.ns = ns  # Jetzt flexibel! Standardmäßig aber "walls"
        marker.id = m_id
        marker.action = Marker.DELETE
        marker_array.markers.append(marker)

    def scan_callback(self, msg):
        # Das ist dein Array an Tupeln: (Grad, Distanz, X, Y)
        point_data = []

        for i, dist in enumerate(msg.ranges):
            # Filtere ungültige Werte (inf, nan oder außerhalb der Reichweite) [cite: 6]
            if math.isinf(dist) or math.isnan(dist) or dist < 0.075 or dist > 3.0:
                continue

            # 1. Originalen Lidar-Winkel in Radiant berechnen [cite: 5]
            angle_lidar_rad = msg.angle_min + i * msg.angle_increment
            angle_lidar_deg = math.degrees(angle_lidar_rad)

            # 2. Umrechnung: Vorne = 0, Rechts = positiv (+), Links = negativ (-)
            # Da 90° Lidar = vorne ist: 90 - 90 = 0 (vorne)
            # 80° Lidar (rechts) wird zu 90 - 80 = +10°
            # 100° Lidar (links) wird zu 90 - 100 = -10°
            angle_user_deg = 90.0 - angle_lidar_deg

            # 3. Kartesische Koordinaten für die Logik (X vorne, Y links)
            # WICHTIG: Für die Berechnung von X und Y nutzen wir den originalen Radiant-Wert,
            # damit die Geometrie für RViz und Standard-ROS korrekt bleibt (X = vorne).
            x = dist * math.cos(angle_lidar_rad)
            y = dist * math.sin(angle_lidar_rad)

            # Als Tupel speichern: (Angepasster Winkel, Roh-Distanz, X, Y) [cite: 10]
            point_data.append((angle_user_deg, x, y, dist))
        
        self.last_point_data = point_data  # Für die Kamera-Fusion speichern
        self.main_logic(point_data)
        #self.get_logger().info(f"Winkel 0: {self.get_closest_measure(point_data, target_angle=0)} Winkel 105: {self.get_closest_measure(point_data, target_angle=105)} Winkel -105: {self.get_closest_measure(point_data, target_angle=-105)}")  # Beispiel: Finde den Punkt direkt vor dem Roboter (0°)

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

    def merge_clusters(self, all_clusters, validated_clusters):
        """
        Versucht, benachbarte Cluster zu einem einzigen Cluster zu verschmelzen.
        Nutzt den Normalenvektor, um nur den senkrechten Abstand (Offset) zu prüfen.
        """
        max_distance_gap = 0.25  # 5 cm maximaler seitlicher Versatz
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
                continue # Überspringe diese leere Wand sofort!

            if angle is None: 
                #self.get_logger().warn("Fehler bei der Winkelberechnung. Überspringe...")
                continue

            bx = sum(p[1] for p in valid_cluster) / len(valid_cluster)
            by = sum(p[2] for p in valid_cluster) / len(valid_cluster)

            angle_rad = math.radians(angle)
            # Richtig: Normalenvektor (steht senkrecht auf der Wand)
            nx = math.cos(angle_rad)
            ny = math.sin(angle_rad)

            clusters_to_remove = []

            for other in remaining_clusters:
                other_angle = self.get_cluster_angle(other)
                if other_angle is None: 
                    continue

                if get_angle_diff(angle, other_angle) < max_angle_gap:
                    ox = sum(p[1] for p in other) / len(other)
                    oy = sum(p[2] for p in other) / len(other)

                    offset = abs((ox - bx) * nx + (oy - by) * ny)

                    self.get_logger().info(f"Offset = {offset:.2f}")

                    if offset < max_distance_gap:
                        self.get_logger().info(f"merged")
                        valid_cluster.extend(other)
                        clusters_to_remove.append(other)
                        combined_clusters[i].append(other)
                else:
                    self.get_logger().info(f"Winkeldifferenz zu groß: {get_angle_diff(angle, other_angle):.1f}")
                        

            for c in clusters_to_remove:
                remaining_clusters.remove(c)

            # --- DER MAGISCHE FIX ---
            # Wir berechnen den Richtungsvektor der Wand (parallel zur Wand)
            dir_x = -math.sin(angle_rad)
            dir_y = math.cos(angle_rad)
            
            # Wir sortieren die Punkte nach ihrer geometrischen Position ENTLANG der Wand!
            # Dadurch liegen der physikalisch erste und letzte Punkt immer an Index 0 und -1,
            # völlig unabhängig vom LiDAR-Wrap-Around.
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
            
            lane_width = abs(x_aussen - x_innen)
            
            if x_innen < 0: # Innenbande links
                target_x = x_innen + (lane_width * self.lane_ratio)
            else:           # Innenbande rechts
                target_x = x_innen - (lane_width * self.lane_ratio)
                
        # --- FALL 2: WIR SEHEN NUR DIE INNENBANDE (Oder haben Außen ignoriert) ---
        elif innenbande:
            x_innen = get_x_at_y(innenbande, target_y)
            if x_innen < 0:
                target_x = x_innen + (self.assumed_lane_width * self.lane_ratio)
            else:
                target_x = x_innen - (self.assumed_lane_width * self.lane_ratio)
                
        # --- FALL 3: WIR SEHEN NUR DIE AUSSENBANDE (Oder haben Innen ignoriert) ---
        elif aussenbande:
            x_aussen = get_x_at_y(aussenbande, target_y)
            inv_ratio = 1.0 - self.lane_ratio 
            if x_aussen < 0: # Außenbande ist links
                target_x = x_aussen + (self.assumed_lane_width * inv_ratio)
            else:            # Außenbande ist rechts
                target_x = x_aussen - (self.assumed_lane_width * inv_ratio)
                
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
        """
        Berechnet den Zielpunkt (Karotte) in der Kurve basierend auf der Frontwand.
        """
        if u_profile is None:
            return None
        
        # Berechenung der Ist-Position 
        ist_front_dist = self.get_closest_point_in_cluster(u_profile[1])[3]
        if self.fahrtrichtung == "links":
            if u_profile[0] is not None:
                ist_side_dist = self.get_closest_point_in_cluster(u_profile[0])[3]
            else:
                if u_profile[2] is not None:
                    ist_side_dist = 3.0 - (self.get_closest_point_in_cluster(u_profile[2])[3])
                else:
                    self.get_logger.warn("Kein Cluster außer der Frontwand gefunden.")
                    ist_side_dist = None
        else:
            if u_profile[2] is not None:
                ist_side_dist = self.get_closest_point_in_cluster(u_profile[2])[3]
            else:
                if u_profile[0] is not None:
                    ist_side_dist = 3.0 - (self.get_closest_point_in_cluster(u_profile[0])[3])
                else:
                    self.get_logger.warn("Kein Cluster außer der Frontwand gefunden.")
                    ist_side_dist = None

                    

        return (target_x, target_y)

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

    def get_unshadowed_leftovers(self, all_clusters, validated_clusters):
        """
        Gibt alle Cluster zurück, die weder Teil der validierten Wände sind,
        noch winkeltechnisch "hinter" den validierten Wänden versteckt liegen.
        """
        if not all_clusters:
            return []

        # 1. Sammle alle Punkte, die bereits erfolgreich zu Wänden gemergt wurden
        # (Wir nutzen ein Set für extrem schnelles Suchen)
        valid_points = set()
        for vc in validated_clusters:
            if vc is not None:
                valid_points.update(vc)
                
        # Finde die reinen Überbleibsel (Wenn der erste Punkt nicht im Set ist, 
        # wurde der Cluster nicht gemergt)
        leftovers = [c for c in all_clusters if c and c[0] not in valid_points]
        
        # 2. Berechne die abgedeckten Winkelbereiche ("Schatten") der echten Wände
        blocked_angle_ranges = []
        for vc in validated_clusters:
            if vc is not None and len(vc) > 0:
                angles = [p[0] for p in vc]
                min_a = min(angles)
                max_a = max(angles)
                blocked_angle_ranges.append((min_a, max_a))
                
        # 3. Filtere die Überbleibsel: Liegen sie in einem Schatten?
        final_free_clusters = []
        padding = 5.0  # 5 Grad Toleranzbereich an den Rändern der Wände
        
        for cluster in leftovers:
            # Wo befindet sich dieser Cluster im Raum? (Schwerpunkt-Winkel)
            c_mean_angle = sum(p[0] for p in cluster) / len(cluster)
            
            is_shadowed = False
            for (min_a, max_a) in blocked_angle_ranges:
                # Liegt der Schwerpunkt des Clusters winkeltechnisch exakt in der Wand?
                if (min_a - padding) <= c_mean_angle <= (max_a + padding):
                    is_shadowed = True
                    break
                    
            # Wenn er NICHT verdeckt wird, ist es ein eigenständiges Objekt (z.B. Hindernis)
            if not is_shadowed:
                final_free_clusters.append(cluster)
                
        return final_free_clusters

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
        

    def visualize_clusters(self, kandidaten_RVIZ):
        innenbande = self.right_wall
        aussenbande = self.left_wall
        target_x, target_y = self.get_target_point_straight(innenbande, aussenbande)
        marker_array = MarkerArray()

        if self.state == 'FOLLOW_LANE':
            self.send_sphere(marker_array, m_id=99, x=target_x, y=target_y, color=(0.0, 1.0, 1.0))
        else:
            self.delete_marker(marker_array, 99, ns="target")
            
        self.pub_markers.publish(marker_array)

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


    def main_logic(self, point_data):
        self.fahrtrichtung = "links"
        if self.counter  < 2:
            self.counter += 1
            return
        

        # 1. Alle Cluster finden (mit Kurven- und Ausreißer-Logik)
        all_clusters = self.get_all_clusters_sorted(point_data)
        front_wall = all_clusters[0] if len(all_clusters) > 0 else None
        if self.begin == True:
            for c in all_clusters:
                self.visualize_clusters([c])
                
                skip = input("Drücke Enter, um zum nächsten Cluster zu gehen...")
                if skip.lower() == 'q':
                    self.begin = False
                    self.get_logger().info("starting Test Bench")
                    self.front_wall = c
                    break
                else:
                    continue
        
        self.front_wall = self.track_front_wall(point_data, self.front_wall)
        validated_clusters = self.validate_clusters_turn(self.front_wall, point_data)
        #validated_clusters = self.merge_clusters(all_clusters, [self.front_wall])[0]
        #validated_clusters = all_clusters
        
        visualize = []
        if validated_clusters is None:
            self.get_logger().info("Validated clusters leer")
            
        else:
            for c in validated_clusters:
                if c is not None:
                    visualize.append(c)
                
            self.get_logger().warn(f"Länge des Visualize: {len(visualize)}")
            self.visualize_clusters(visualize)
        
        self.counter = 0

        
        









        


        
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