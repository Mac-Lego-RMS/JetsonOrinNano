#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, LaserScan
from cv_bridge import CvBridge
from rclpy.qos import qos_profile_sensor_data
from std_msgs.msg import String
from ultralytics import YOLO
import cv2
import numpy as np
import math
from visualization_msgs.msg import Marker

class YoloObstacleDetector(Node):
    def __init__(self):
        super().__init__('yolo_detector')
        
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
        self.img_sub = self.create_subscription(
            Image, 
            '/camera/image_raw', 
            self.image_callback, 
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
        self.cmd_pub = self.create_publisher(String, '/obstacle_cmd', 10)
        self.avoid_trigger_dist = 0.85 # Schwellenwert: Ab 85cm vor dem Hindernis ausweichen
        
        # Variablen für die Fusion
        self.last_scan = None
        self.camera_fov = 160.0  # Dein Sichtfeld
        self.get_logger().info('YOLO Lidar Fusion Node gestartet.')

    def scan_callback(self, msg):
        # Speichert den aktuellsten Scan für die Verrechnung mit dem Bild   
        #self.get_logger().info('Lidar-Daten empfangen!')
        self.last_scan = msg

    def image_callback(self, msg):
        if self.last_scan is None: return

        cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        image_width = cv_image.shape[1]     
        results = self.model(cv_image, verbose=False)

        # --- NEU: HIER WIRD DAS DEBUG-BILD ERSTELLT UND PUBLIZIERT ---
        # 1. Bounding Boxes und Labels auf das Bild zeichnen
        annotated_frame = results[0].plot()

        # 2. Das OpenCV-Bild zurück in eine ROS-Message konvertieren
        debug_msg = self.bridge.cv2_to_imgmsg(annotated_frame, encoding="bgr8")
        # 3. Das Bild auf dem Topic /camera/yolo_debug veröffentlichen
        self.pub_debug_img.publish(debug_msg)
        # -------------------------------------------------------------
        
        detected_obstacles = []

        for r in results:
            for box in r.boxes:
                if float(box.conf[0]) > 0.8:
                    self.get_logger().info(f'Objekt erkannt: {self.model.names[int(box.cls[0])]} mit Konfidenz {box.conf[0]:.2f}')
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    center_x = (x1 + x2) / 2.0

                    cam_angle_deg = 170.0 - (center_x / image_width) * self.camera_fov
                    cam_angle_rad = math.radians(cam_angle_deg + self.angle_calibration)

                    result = self.get_lidar_distance(cam_angle_rad)

                    if result is not None:
                        lidar_dist, lidar_angle = result
                        actual_dist = lidar_dist + self.camera_to_lidar_dist
                        class_name = self.model.names[int(box.cls[0])].lower()
                        
                        self.publish_marker(lidar_angle, actual_dist, class_name, int(box.cls[0]))
                        
                        # Hindernis für die Entscheidungsfindung speichern
                        detected_obstacles.append((actual_dist, class_name))

        # --- DIE KOMMANDO-LOGIK ---
        cmd_msg = String()
        cmd_msg.data = "CLEAR" # Standard: Wir bleiben auf der normalen Ideallinie

        if detected_obstacles:
            # Sortieren nach Distanz (das nächste Objekt kommt auf Index 0)
            detected_obstacles.sort(key=lambda x: x[0])
            closest_dist, closest_name = detected_obstacles[0]

            # Timing: Sind wir nah genug dran für das Manöver?
            if closest_dist < self.avoid_trigger_dist:
                if "red" in closest_name:
                    cmd_msg.data = "AVOID_RIGHT"
                elif "green" in closest_name:
                    cmd_msg.data = "AVOID_LEFT"
                
                self.get_logger().info(f'+++ KOMMANDO: {cmd_msg.data} ({closest_dist:.2f}m) +++')

        self.cmd_pub.publish(cmd_msg)

    def get_lidar_distance(self, camera_angle_rad):
        if self.last_scan is None: return None

        msg = self.last_scan
        num_points = len(msg.ranges)
        
        # 1. Suchbereich in Indizes umrechnen
        # Wir schauen ca. 25 Grad links und rechts vom Kamera-Winkel
        search_rad = math.radians(25)
        angle_min = (camera_angle_rad - search_rad)
        angle_max = (camera_angle_rad + search_rad)
        
        clusters = []
        current_cluster = []
        
        # 2. Wir laufen sequenziell durch das Array
        for i in range(num_points):
            dist = msg.ranges[i]
            if math.isnan(dist) or math.isinf(dist) or dist < 0.15 or dist > 3.5:
                continue
                
            angle = (msg.angle_min + i * msg.angle_increment)
            # Winkel normalisieren auf (-pi bis pi) oder (0 bis 2pi) je nach Lidar
            angle = math.atan2(math.sin(angle), math.cos(angle))
            cam_angle_norm = math.atan2(math.sin(camera_angle_rad), math.cos(camera_angle_rad))

            # Nur Punkte im Kamera-Sichtfeld betrachten
            if abs(self.angle_diff(angle, cam_angle_norm)) < search_rad:
                if not current_cluster:
                    current_cluster.append((angle, dist))
                else:
                    # DER ENTSCHEIDENDE SPRUNG:
                    # Vergleich mit dem direkten Nachbarn im Array
                    prev_dist = current_cluster[-1][1]
                    
                    # Wenn der Sprung zwischen zwei Nachbarn > 15cm ist, 
                    # fängt ein neues Objekt an (Säule endet oder Bande beginnt)
                    if abs(dist - prev_dist) < 0.15:
                        current_cluster.append((angle, dist))
                    else:
                        clusters.append(current_cluster)
                        current_cluster = [(angle, dist)]
        
        if current_cluster: clusters.append(current_cluster)

        # 3. Das schmalste/nächste Cluster finden, das zur Säule passt
        best_c = None
        min_dist = 4.0
        
        for c in clusters:
            # Eine Säule hat typischerweise 4-10 benachbarte Punkte
            if 3 <= len(c) <= 12:
                avg_dist = sum(p[1] for p in c) / len(c)
                avg_angle = sum(p[0] for p in c) / len(c)
                
                # Wir nehmen das Cluster, das am nächsten ist (Vordergrund-Prinzip)
                if avg_dist < min_dist:
                    min_dist = avg_dist
                    best_c = (avg_dist, avg_angle)

        return best_c # Liefert (Distanz, Winkel)

    def angle_diff(self, a, b):
        """Berechnet den kleinsten Unterschied zwischen zwei Winkeln (Rad)."""
        # Sorgt dafür, dass der Unterschied auch über die 0/360° Grenze korrekt bleibt
        return math.atan2(math.sin(a - b), math.cos(a - b))
    
    def publish_marker(self, angle_rad, dist, name, class_id):
        marker = Marker()
        marker.header.frame_id = "ldlidar_link"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "yolo_obstacles"
        marker.id = class_id
        marker.type = Marker.CYLINDER
        marker.action = Marker.ADD
        
        # Hier wird die Position final berechnet
        marker.pose.position.x = dist * math.cos(angle_rad)
        marker.pose.position.y = dist * math.sin(angle_rad)
        marker.pose.position.z = self.lidar_height_offset # Höhenkorrektur
        
        marker.scale.x, marker.scale.y, marker.scale.z = 0.15, 0.15, 0.3
        
        marker.color.a = 1.0
        if "red" in name:
            marker.color.r, marker.color.g, marker.color.b = 1.0, 0.0, 0.0
        else:
            marker.color.r, marker.color.g, marker.color.b = 0.0, 1.0, 0.0
            
        marker.lifetime = rclpy.duration.Duration(seconds=0.5).to_msg()
        self.marker_pub.publish(marker)

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

def main(args=None):
    rclpy.init(args=args)
    node = YoloObstacleDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            node.destroy_node()
            rclpy.shutdown()

if __name__ == '__main__':
    main()