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
        
        # Karotten-Parameter
        self.lookahead_dist = 0.60    # Wie weit schaut der Roboter voraus? (60 cm)
        self.min_wall_dist = 0.10       

        self.lane_ratio = 0.85       # Verhältnis des Bandenabstands innen zu außen Außen Bande: 0.85, Innen Bande: 0.20
        self.assumed_lane_width = 1.0 # Wenn eine Wand fehlt, gehen wir von 60cm Spurbreite aus
        self.turn_exit_angle = 25
        self.max_wall_lenght_for_turn = 0.25

        # Object Detection Parameter
        self.sub_cmd = self.create_subscription(String, '/obstacle_cmd', self.cmd_callback, 10)
        self.current_obstacle_cmd = "CLEAR"

        # Standard-Werte für die Ideallinie (z.B. Außenbahn)
        self.default_lane_ratio = 0.85 
        self.default_max_turn = 0.635
        

        # --- PID-REGLER PARAMETER ---
        self.kp = 3.5   # Lenkt hart zur Karotte
        self.kd = 0   # Verhindert das Schlingern (Dämpfung)
        self.ki = 0.0   # Integral (oft bei WRO auf 0 gelassen, da schnelle Spurwechsel)
        
        self.prev_error = 0.0
        self.integral_error = 0.0
        
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
        self.model = YOLO('/workspace/best.engine', task='detect')
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
    
    def cmd_callback(self, msg):
        self.current_obstacle_cmd = msg.data

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
    
    def validate_clusters(self, clusters):
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
        Nutzt die Manhattan-Norm und X/Y-Gradienten-Überwachung.
        point_data: Liste aus (Winkel_deg, x, y, dist)
        """
        if len(point_data) < 2:
            return []

        sorted_points = sorted(point_data, key=lambda p: p[0])

        # --- PARAMETER ---
        # Max Gap etwas höher setzen, da Manhattan-Werte (dx+dy) größer sind als euklidische
        max_gap = 0.10      
        outlier_limit = 2
        # Ab welcher Streckenlänge trauen wir der Ecke? (Filtert Sensor-Rauschen)
        corner_sensitivity = 0.05 
        
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
            
            # 1. MANHATTAN-DISTANZ (|dx| + |dy|)
            dx = abs(p_curr[1] - p_last[1])
            dy = abs(p_curr[2] - p_last[2])
            dist_manhattan = dx + dy
            
            if dist_manhattan < max_gap:
                # 2. NEUE ECKEN-ERKENNUNG (Über echte Vektor-Winkel)
                if len(current_cluster) >= 20:
                    p_start = current_cluster[5] # Punkt vor der Kurve
                    p_mid = current_cluster[-10]   # Punkt am Scheitel
                    
                    # Vektor 1 (Trend vor der Kurve)
                    dx1 = p_mid[1] - p_start[1]
                    dy1 = p_mid[2] - p_start[2]
                    angle1 = math.atan2(dy1, dx1)
                    
                    # Vektor 2 (Aktuelle Bewegung)
                    dx2 = p_curr[1] - p_mid[1]
                    dy2 = p_curr[2] - p_mid[2]
                    angle2 = math.atan2(dy2, dx2)
                    
                    # Wie stark knickt die Wand physikalisch ab?
                    diff_rad = abs(angle1 - angle2)
                    diff_deg = math.degrees(diff_rad)
                    if diff_deg > 180:
                        diff_deg = 360 - diff_deg
                        
                    # Filter: Hat sich der Punkt auch ausreichend bewegt? (Rauschen ignorieren)
                    dist_moved = math.hypot(dx2, dy2)
                    
                    # Eine echte Parcours-Ecke knickt stark ab (z.B. > 70 Grad)
                    if diff_deg > 25.0 and dist_moved > corner_sensitivity:
                        # ECKE ERKANNT! Wir zerschneiden das Cluster genau hier.
                        clusters.append(current_cluster)
                        current_cluster = [p_curr]
                        i += 1
                        continue

                # Wenn keine Ecke, Punkt normal zum Cluster hinzufügen
                current_cluster.append(p_curr)
                i += 1
            else:
                # 3. AUSREISSER-LOGIK (Jetzt auch mit Manhattan-Norm)
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
                    # Echte Lücke gefunden -> Cluster abspeichern und neu beginnen
                    clusters.append(current_cluster)
                    current_cluster = [p_curr]
                    i += 1
        
        if current_cluster:
            clusters.append(current_cluster)

        # 4. WRAP-AROUND-FIX (Mit Manhattan)
        if len(clusters) > 1:
            first_p = clusters[0][0]
            last_p = clusters[-1][-1]
            dist_wrap = abs(first_p[1] - last_p[1]) + abs(first_p[2] - last_p[2])
            
            # Auch hier prüfen, ob sie über den 180°-Rand hinaus eigentlich die gleiche Wandachse sind
            if dist_wrap < max_gap:
                clusters[0] = clusters[-1] + clusters[0]
                clusters.pop()

        # 5. Sortieren nach Größe, größtes zuerst
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
        self.process_my_logic(point_data)
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
        max_distance_gap = 0.30  # 5 cm maximaler seitlicher Versatz
        max_angle_gap = 5.0      # 5 Grad maximale Winkelabweichung

        remaining_clusters = [c for c in all_clusters if c not in validated_clusters]
        if not remaining_clusters:
            return validated_clusters

        def get_angle_diff(a1, a2):
            diff = abs(a1 - a2) % 180
            if diff > 90:
                diff = 180 - diff
            return diff

        for valid_cluster in validated_clusters:
            angle = self.get_cluster_angle(valid_cluster)
            if valid_cluster is None:
                continue # Überspringe diese leere Wand sofort!

            if angle is None: 
                continue

            bx = sum(p[1] for p in valid_cluster) / len(valid_cluster)
            by = sum(p[2] for p in valid_cluster) / len(valid_cluster)

            angle_rad = math.radians(angle)
            # Normalenvektor für den Abstand
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

                    if offset < max_distance_gap:
                        valid_cluster.extend(other)
                        clusters_to_remove.append(other)

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

        return validated_clusters

    def get_target_point(self, innenbande, aussenbande):
        """
        Berechnet den Zielpunkt (Karotte) im perfekten Verhältnis zur Innenbande.
        Mit absolutem Mindestabstand (Kraftfeld-Logik).
        """
        target_y = self.lookahead_dist  
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
            self.lookahead_dist = 0.60  # Entspannt vorausschauen
            
            # WICHTIG: Das hier lieber auskommentieren, sonst spammt es dein Terminal voll!
            # self.get_logger().info("Keine Hindernisse erkannt. Fahre mit Standardparametern.")
            return

        # --- AUSWEICH-WERTE (Adrenalin-Modus) ---
        # Karotte näher ranholen, um viel direkter und schärfer zu lenken!
        self.lookahead_dist = 0.35 

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

    def process_my_logic(self, point_data):
        # 1. IMMER GANZ OBEN: Den Sammelkorb erstellen, damit er überall in der Funktion existiert!
        marker_array = MarkerArray()
        cmd = Twist()

        #self.get_logger().info(f"IMU Winkel: {(self.current_yaw - self.yaw_offset):.1f}°")
            
                            
        innenbande = None
        aussenbande = None
        kandidaten = [None, None, None]

        if self.last_image_msg is None:
            self.get_logger().info("Warte auf Kamera-Bild...", throttle_duration_sec=1.0)
            return
        results = self.image_callback(self.last_image_msg)
        if results is None: return

        detected_obstacles = []

        for r in results:
            for box in r.boxes:
                if float(box.conf[0]) > 0.8:
                    self.get_logger().info(f'Objekt erkannt: {self.model.names[int(box.cls[0])]} mit Konfidenz {box.conf[0]:.2f}')
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    center_x = (x1 + x2) / 2.0

                    # 1. Nullpunkt in die Mitte des Bildes schieben
                    # center_x ist 0 (ganz links) bis image_width (ganz rechts)
                    pixel_from_center = center_x - (self.image_width / 2.0)
                    
                    # 2. Pixel in Grad umrechnen (Kamera FOV auf die Breite verteilen)
                    cam_angle_deg = pixel_from_center * (self.camera_fov / self.image_width)
                    
                    # 3. Kalibrierung dazurechnen und in Bogenmaß (Radian) umwandeln
                    cam_angle_rad = math.radians(cam_angle_deg + self.angle_calibration)
                    
                    self.get_logger().info(f'Kamerawinkel: {cam_angle_deg:.1f}° (nach Kalibrierung: {cam_angle_deg + self.angle_calibration:.1f}°)')

                    # Lidar nach diesem Winkel durchsuchen
                    result = self.get_lidar_distance(cam_angle_rad, self.get_all_clusters_sorted(self.last_point_data))

                    if result is not None:
                        # result enthält jetzt: (angle_deg, x, y, dist)
                        _, obj_x, obj_y, obj_dist = result 
                        class_name = self.model.names[int(box.cls[0])].lower()
                        
                        # publish_marker bekommt jetzt direkt die rohen X- und Y-Werte!
                        self.publish_marker(obj_x, obj_y, class_name, int(box.cls[0]))
                        
                        detected_obstacles.append((obj_dist, class_name))

        self.current_obstacle_cmd = "CLEAR"  # Standardmäßig kein Hindernis

        current_min_obs_dist = 999.0
        if detected_obstacles:
            # Sortieren nach Distanz (das nächste Objekt kommt auf Index 0)
            detected_obstacles.sort(key=lambda x: x[0])
            closest_dist, closest_name = detected_obstacles[0]

            current_min_obs_dist = closest_dist

            # Timing: Sind wir nah genug dran für das Manöver?
            if closest_dist < self.avoid_trigger_dist:
                
                # Nur überschreiben, wenn wir aktuell auf CLEAR (mittig) stehen
                if self.current_obstacle_cmd == "CLEAR":
                    if "red" in closest_name:
                        self.current_obstacle_cmd = "RED"
                    elif "green" in closest_name:
                        self.current_obstacle_cmd = "GREEN"
                        
                    # HIER: Wir merken uns, in welchem Streckenabschnitt wir das gelockt haben!
                    self.locked_turn_count = self.turn_count
                    
                    self.get_logger().warn(f'+++ SPUR EINGELOGGT: {self.current_obstacle_cmd} ({closest_dist:.2f}m) +++')

       

        # 2. Punktwolke in Cluster (Wände) aufteilen
        if self.state == 'FOLLOW_LANE':
            total_gedreht = abs(self.current_yaw - self.yaw_offset)

            min_total_rotation = self.target_turns * 85.0
            if self.turn_count >= self.target_turns and total_gedreht >= min_total_rotation:
                if self.front_wall is not None:
                    closest_f = self.get_closest_point_in_cluster(self.front_wall)
                    if closest_f is not None and closest_f[3] < 1.80:
                        self.state = 'STOPPED'
                        return

            if abs(self.start_straight_yaw - self.current_yaw) > 75.0:
                self.get_logger().warn(f">>> GYRO KURVE ERKANNT! Zu weit auf der geraden gedreht (Gedreht: {abs(self.start_straight_yaw - self.current_yaw):.1f}°) <<<")
                self.start_straight_yaw = self.current_yaw
                self.turn_count += 1
                
            all_clusters = self.get_all_clusters_sorted(point_data)


            validated_clusters = self.validate_clusters(all_clusters)
            # 5. Wir haben ein perfektes U-Profil!
            #self.get_logger().info(f"Anzahl cluster {len(all_clusters)}")
            kandidaten = self.merge_clusters(all_clusters, validated_clusters)  # Die drei größten Cluster, die wir validiert haben

            self.right_wall = kandidaten[2]
            self.front_wall = kandidaten[1]
            self.left_wall  = kandidaten[0]

            # Hier war vorher die Wahl der Fahrtrichtung, aber wir haben sie jetzt schon in der Startphase festgelegt.
        
            if self.fahrtrichtung == 'links':
                innenbande = self.left_wall
                aussenbande = self.right_wall
            elif self.fahrtrichtung == 'rechts':
                innenbande = self.right_wall
                aussenbande = self.left_wall

            self.update_avoidance_settings()
            

        if self.state in ['TURN_LINKS', 'TURN_RECHTS']:
            turned_so_far = abs(self.current_yaw - self.start_turn_yaw)
            # Während der Kurve tracken wir die Frontwand mit einem dynamischen Suchfenster (ROI)
            self.front_wall = self.track_front_wall(point_data, self.front_wall)
            kandidaten = [None, self.front_wall, None]  # Wir haben nur die Frontwand, die wir tracken
            
            # --- 1. DER NEUE NOTFALL-ABBRUCH ---
            # Steht ein Objekt am Kurvenausgang extrem nah (< 50cm)?

            current_time = self.get_clock().now().nanoseconds / 1e9
            time_in_turn = current_time - self.turn_start_time

            if current_min_obs_dist < 0.50 and time_in_turn > 1: # Kurven Cooldown für Notfall Abbruch
                self.get_logger().warn(f">>> NOTFALL-ABBRUCH DER KURVE! Hindernis bei {current_min_obs_dist:.2f}m. <<<")
                self.state = 'FOLLOW_LANE'
                self.front_wall = None
                self.turn_count += 1
                self.start_straight_yaw = self.current_yaw
                
                # EXTREM WICHTIG: Damit der Speicher im nächsten Frame nicht gelöscht wird,
                # aktualisieren wir den Lock auf die "neue" Gerade!
                self.locked_turn_count = self.turn_count
            
            else:
                angle_to_front_wall = self.get_cluster_angle(self.front_wall)
                self.get_logger().info(f"Tracke Frontwand... Winkel zur Fahrtrichtung: {angle_to_front_wall:.1f}°")

                if abs(angle_to_front_wall) < self.turn_exit_angle or turned_so_far > 95.0:
                    self.get_logger().warn(">>> KURVE FAST FERTIG! Wechsel zurück in FOLLOW_LANE <<<")
                    self.state = 'FOLLOW_LANE'
                    self.front_wall = None
                    if turned_so_far > 45.0:
                        self.get_logger().warn(">>> KURVE NORMAL FERTIG! Wechsel zurück in FOLLOW_LANE <<<")
                        self.turn_count += 1
                    else:
                        self.get_logger().warn(f">>> FAKE-KURVE ERKANNT ({turned_so_far:.1f}°). Zähler ignoriert! <<<")
                        
                    self.start_straight_yaw = self.current_yaw

                else:
                    if self.state == 'TURN_LINKS':
                        cmd.linear.x = self.turn_speed
                        cmd.angular.z = self.max_turn_angle
                    else:
                        cmd.linear.x = self.turn_speed
                        cmd.angular.z = -self.max_turn_angle

        elif self.state == 'STOPPED':
            cmd.linear.x = 0.0
            cmd.angular.z = 0.0
            self.pub_cmd_vel.publish(cmd)
            self.get_logger().warn(f">>> ZIEL ERREICHT: {self.turn_count} Kurven geschafft! Stoppe den Roboter. <<<")
            self.destroy_node()
            rclpy.shutdown()
            return

        elif self.state == 'STARTING':
            self.get_logger().info("Starte den Roboter... Evaluiere Fahrtrichtung und Kalibriere Gyro.")

            all_clusters = self.get_all_clusters_sorted(point_data)


            validated_clusters = self.validate_clusters(all_clusters)
            # 5. Wir haben ein perfektes U-Profil!
            #self.get_logger().info(f"Anzahl cluster {len(all_clusters)}")
            kandidaten = self.merge_clusters(all_clusters, validated_clusters)  # Die drei größten Cluster, die wir validiert haben

            self.right_wall = kandidaten[2]
            self.front_wall = kandidaten[1]
            self.left_wall  = kandidaten[0]
            if self.fahrtrichtung is None:
            # Wir brauchen zwingend beide Seitenwände für den Längenvergleich
                if self.left_wall and self.right_wall:
                    # Echte physikalische Länge in Metern berechnen (Satz des Pythagoras)
                    left_len = math.hypot(self.left_wall[0][1] - self.left_wall[-1][1], self.left_wall[0][2] - self.left_wall[-1][2])
                    right_len = math.hypot(self.right_wall[0][1] - self.right_wall[-1][1], self.right_wall[0][2] - self.right_wall[-1][2])
                    
                    self.get_logger().info(f"Scanne Strecke... Länge Links: {left_len:.2f}m, Länge Rechts: {right_len:.2f}m")
                    
                    # Wir brauchen einen deutlichen Unterschied (z.B. 40 cm), um sicher zu sein!
                    if left_len > right_len + 0.30:
                        self.fahrtrichtung = 'rechts' # Rechte Wand ist kürzer = Innenbande = Wir fahren rechts herum!
                        self.get_logger().info(">>> LOCK: FAHRTRICHTUNG RECHTS (Uhrzeigersinn) <<<")
                    elif right_len > left_len + 0.30:
                        self.fahrtrichtung = 'links'  # Linke Wand ist kürzer = Innenbande = Wir fahren links herum!
                        self.get_logger().info(">>> LOCK: FAHRTRICHTUNG LINKS (Gegen den Uhrzeigersinn) <<<")
                
                else:
                    self.get_logger().info("Fahrtrichtung noch nicht erkannt... Warte auf beide Seitenwände für die Analyse.")
                    return

            if not self.imu_ready:
                self.get_logger().info("Warte auf Gyroskop-Bootvorgang...")
                return  # Brich hier ab, mach noch nichts!   
            self.yaw_offset = self.current_yaw
            self.start_straight_yaw = self.current_yaw  # Gyro-Kalibrierung: Aktuellen Winkel als Referenz setzen

            self.state = 'FOLLOW_LANE'
        else:
        
            pass
        
    
        # ==========================================
        # PFADPLANUNG, PID & STATE MACHINE 
        # ==========================================
        if self.current_obstacle_cmd != "CLEAR" and self.turn_count > self.locked_turn_count:
            self.current_obstacle_cmd = "CLEAR"
            self.get_logger().warn(">>> KURVE BEENDET: Ausweich-Speicher gelöscht, fahre wieder mittig! <<<")

        self.update_avoidance_settings()

        
        target_x, target_y = self.get_target_point(innenbande, aussenbande)

        # Twist-Nachricht für den ESP initialisieren
        
        # --- ZUSTAND 1: GERADEAUS FAHREN ---
        if self.state == 'FOLLOW_LANE':
            
            # 1. WECHSEL-BEDINGUNG PRÜFEN
            if self.front_wall is not None and self.fahrtrichtung is not None:
                front_dist = self.get_closest_point_in_cluster(self.front_wall)[3]
                
                max_y_innen = 0.0
                if innenbande and len(innenbande) > 0:
                    max_y_innen = max(p[2] for p in innenbande)
                
                if front_dist < 1.20 and max_y_innen < self.max_wall_lenght_for_turn:
                    self.state = f"TURN_{self.fahrtrichtung.upper()}"
                    self.start_turn_yaw = self.current_yaw
                    self.get_logger().warn(f">>> {self.state} EINGELEITET bei {(self.start_turn_yaw - self.yaw_offset):.1f}° <<<")
                    # WICHTIG: PID-Gedächtnis für die nächste Gerade löschen!
                    self.prev_error = 0.0
                    self.integral_error = 0.0

                    self.turn_start_time = self.get_clock().now().nanoseconds / 1e9
                else:
                    if front_dist < 1.30:
                        self.get_logger().info(f"Warte auf Ecke... (Innenbande ragt noch {max_y_innen:.2f}m nach vorne)")
            
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

        # --- ZUSTAND 2: LINKSKURVE ---
        elif self.state == 'TURN_LINKS':
            pass

        # --- ZUSTAND 3: RECHTSKURVE ---
        elif self.state == 'TURN_RECHTS':
            pass


        # --- OUTPUT AN HARDWARE & RVIZ ---
        
        # Befehle an den ESP senden! (Erstmal aufgebockt testen!)
        self.pub_cmd_vel.publish(cmd)
        
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


        # 6. Die drei Wände durchgehen und an RViz senden
        for i, cluster in enumerate(kandidaten):
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