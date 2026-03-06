#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, LaserScan
from cv_bridge import CvBridge
from rclpy.qos import qos_profile_sensor_data
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
        
        for r in results:
            for box in r.boxes:
                if float(box.conf[0]) > 0.6:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    center_x = (x1 + x2) / 2.0

                    # Nur eine grobe Richtung von der Kamera
                    cam_angle_deg = 170.0 - (center_x / image_width) * self.camera_fov
                    cam_angle_rad = math.radians(cam_angle_deg + self.angle_calibration)

                    # Die Suche im Lidar (liefert Distanz UND ECHTEN WINKEL)
                    result = self.get_lidar_distance(cam_angle_rad)

                    if result is not None:
                        lidar_dist, lidar_angle = result # <--- DAS IST DER FIX!
                        
                        actual_dist = lidar_dist + self.camera_to_lidar_dist
                        
                        # Wir nutzen lidar_angle für den Marker, NICHT cam_angle_rad!
                        self.publish_marker(lidar_angle, actual_dist, self.model.names[int(box.cls[0])].lower(), int(box.cls[0]))
                        
                        self.get_logger().info(f'Säule auf Lidar-Punkte eingerastet! Winkel: {math.degrees(lidar_angle):.1f}°')
                    else:
                        # Wenn Lidar nichts findet, zeichnen wir gar keinen Marker (verhindert den roten Kreis)
                        self.get_logger().warn('Kamera sieht was, aber Lidar findet kein passendes Cluster.')

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