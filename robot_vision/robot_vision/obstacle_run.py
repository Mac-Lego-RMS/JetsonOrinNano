#!/usr/bin/env python3

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_CORETYPE"] = "ARMV8"

from platform import node

import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
import threading
from sensor_msgs.msg import LaserScan, Imu, Image
from geometry_msgs.msg import Twist, Point
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import String, Float64, Bool
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
from robot_vision.camera_lib import TrackAnalyzer
from robot_vision.obstacle import Obstacle
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
import time
import logging

'''
=============================================================
     HARDWARE-KOORDINATENSYSTEM (Lidar & Foxglove)
=============================================================

                          +Y (VORNE)
                              ^
                              |  Lidar: 180°
                              |  (Fahrtrichtung)
                              |
                              |
    (LINKS)                   |                   (RECHTS)
      -X  <------------- [ ROBOTER ] ------------->  +X
    Lidar: 270°               |                   Lidar: 90°
                              |
                              |
                              |
                              v
                          -Y (HINTEN)
                       Lidar: 360° / 0°

-------------------------------------------------------------
Zone Ids:

20  |   21     |
|   |   |      |
10  |   11     | <-- outer_wall
|   |   |      |
00  |   01     |
    ^
    |
 ROBOTER
============================================================='''

class Obstacle_Run(Node):
    def __init__(self):
        super().__init__('obstacle_run')

        # MULTITHREADING SETUP
        self.sensor_cbg = MutuallyExclusiveCallbackGroup()
        self.yolo_cbg = MutuallyExclusiveCallbackGroup()
        
        # Thread-Lock für Datensicherheit beim Zugriff auf gemeinsame Variablen
        self.data_lock = threading.Lock()
        self.latest_yolo_results = []
        
        
        
        self.scan_sub = self.create_subscription(
            LaserScan,
            '/ldlidar_node/scan',
            self.scan_callback,
            qos_profile_sensor_data,
            callback_group=self.sensor_cbg  # Läuft in eigenem Thread
        )
        
        imu_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        self.sub_imu = self.create_subscription(
            Imu, 
            '/bno055/imu', 
            self.imu_callback, 
            imu_qos,
            callback_group=self.sensor_cbg
        )

        self.button_sub = self.create_subscription(
            Bool,
            '/button_state',
            self.button_callback,
            10
        )

        self.button_start = True
        self.mit_ausparken = False

        self.led_pub = self.create_publisher(Bool, '/led_cmd', 10)

        self.button_state = False
        
        # Publisher für RViz
        self.pub_cmd_vel = self.create_publisher(Twist, '/cmd_vel', 10)
        self.pub_markers = self.create_publisher(MarkerArray, '/wall_follower_markers', 10)

        # IMU Variablen
        self.yaw_offset = 0.0
        self.current_yaw = 0.0
        self.imu_ready = False
        self.last_raw_yaw = None
        self.start_turn_yaw = None
        self.start_straight_yaw = 0.0
        self.last_turn_aborted = False
        
        
        self.rviz_frame = 'ldlidar_link'
        self.get_logger().info('>>> WallFollower Template gestartet. Warte auf LiDAR... <<<')

        # STATE MACHINE Startzustände
        self.state = 'INITIALIZING'
        self.turn_phase = 'APPROACH'
        self.parking_phase = None
        self.park_direction = None
        self.park_turn_richtung = None

        self.fahrtrichtung = None
        self.saved_intersection_angle = None
        self.saved_curve_radius_m = None

        self.target_turns = 12
        self.turn_count = 0
        self.is_start_finish_straight = False
        self.last_turn_for_parking = False
        self.parking_straight = False

        self.front_wall = None
        self.left_wall = None
        self.right_wall = None
        
        self.standard_lane_ratio_approach = 0.60
        self.standard_lane_ratio_exit = 0.45
        self.lane_ratio = self.standard_lane_ratio_approach       # Verhältnis des Bandenabstands innen zu außen
        self.lane_ratio_approach = self.standard_lane_ratio_approach
        self.lane_ratio_exit = self.standard_lane_ratio_exit

        self.base_obst_cmd = None
        self.base_entry_distance = None
        self.assumed_lane_width = 1.0
        self.turn_exit_angle = 25
        self.max_wall_lenght_for_turn = 0.25

        # Object Detection Parameter
        self.current_obstacle_cmd = "CLEAR"
        self.obstacle_memory = [None, None, None, None]

        # PID-REGLER Parameter
        self.kp = 2.0
        self.ki = 0.07
        self.kd = 0.05

        self.lookahead_dist_straight = 0.20
        self.lookahead_dist_parking = 0.40
        self.min_wall_dist = 0.15
        
        self.prev_error = 0.0
        self.integral_error = 0.0

        self.auspark_step = 0
        self.auspark_timer = None
        self.auspark_sequenz_done = False

        self.steering_ctrl = SteeringController(logger=self.get_logger())

        self.waiting_timer = None

        self.TRACK_WIDTH_M = 1.0
        self.ROBOT_WIDTH_M = 0.15
        self.LIDAR_OFFSET_M = 0.08
        self.SAFETY_MARGIN_M = 0.05
        self.MAX_KINEMATIC_RADIUS_M = 1.2
        self.IDEAL_RADIUS_M = 0.28
        self.MIN_TURN_RADIUS_M = 0.20

        # MOTOR Parameter
        self.base_speed = 350.0
        self.turn_speed = 350.0
        self.parking_speed = 250.0

        # YOLO Parameter
        self.image_width = 1280
        self.bridge = CvBridge()

        self.target_yolo_fps = 8.0 
        self.last_yolo_time = self.get_clock().now().nanoseconds / 1e9

        self.analyzer = TrackAnalyzer(
            logger=self.get_logger(),
            visualizer_cb=self.visualize_cluster_line
        )

        self.K = np.array([
            [820.7558,   0.0000, 639.0000],
            [  0.0000, 817.4016, 359.0000],
            [  0.0000,   0.0000,   1.0000]
        ])
        
        self.D = np.array([
            [-0.05801], 
            [ 0.22847], 
            [-0.58155], 
            [ 0.41406]
        ])
        
        self.get_logger().info('Lade YOLO TensorRT Engine')
        self.model = YOLO('/workspace/best.engine', task='detect')
        self.get_logger().info('Modell erfolgreich geladen')

        self.get_logger().info('Starte TensorRT Warm-up')
        dummy_cv_image = np.zeros((360, 640, 3), dtype=np.uint8) 
        dummy_cv_image = np.ascontiguousarray(dummy_cv_image)
        self.model.predict(
            dummy_cv_image, 
            half=True, 
            imgsz=640, 
            device=0, 
            verbose=False
        )
        self.get_logger().info('Warm-up abgeschlossen. GPU ist bereit.')


        self.angle_calibration = 0.0
        self.lidar_height_offset = 0.05
        self.camera_to_lidar_dist = 0.03

        
        # Marker Publisher
        self.marker_pub = self.create_publisher(Marker, '/detected_obstacles', 10)
        self.pub_debug_img = self.create_publisher(Image, '/camera/yolo_debug', 10)
        
        self.avoid_trigger_dist = 0.85
        
        # Variablen für die Kamera, Lidar Fusion
        self.camera_fov = 115.0
        self.get_logger().info('YOLO Lidar Fusion Node gestartet.')

        self.img_sub = self.create_subscription(
            Image, 
            '/video_source/raw', 
            self.camera_sub_callback, 
            qos_profile_sensor_data,
            callback_group=self.yolo_cbg
        )

        self.pub_timer = self.create_publisher(Float64, '/robot_timer', 10)
        self.start_time_stamp = None
        self.elapsed_time = 0.0
        self.timer_active = False

        # Debug
        self.begin = True
        self.counter = 0
        self.test_is_turning = False
        self.curve_radius_m = None
        self.pub_obstacle_markers = self.create_publisher(MarkerArray, 'rviz_obstacles', 10)
        self.camera_calibration = False
        self.debug = True
        self.debug_start = self.debug

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

    def set_led(self, state: bool):
        msg = Bool()
        msg.data = state
        self.led_pub.publish(msg)

    def update_strategy_params(self):
        if self.turn_count > 4:
            self.base_speed = 500.0
            self.turn_speed = 350.0
            self.IDEAL_RADIUS_M = 0.28
            self.standard_lane_ratio_approach = 0.60
            self.kp = 2.0
            self.ki = 0.07
            self.kd = 0.09

        else:
            self.base_speed = 350.0
            self.turn_speed = 300.0
            self.IDEAL_RADIUS_M = 0.25
            self.standard_lane_ratio_approach = 0.70
            self.kp = 2.3
            self.ki = 0.07
            self.kd = 0.05
        
        if self.turn_count % 4 == 0 and self.state == 'FOLLOW_LANE' or self.turn_count % 4 == 3 and self.state in ['TURN_LINKS', 'TURN_RECHTS']:
            self.is_start_finish_straight = True
            self.standard_lane_ratio_approach = 0.60
        else:
            self.is_start_finish_straight = False
            self.standard_lane_ratio_approach = 0.70

        if (self.target_turns - self.turn_count) == 1 and self.state in ['TURN_LINKS', 'TURN_RECHTS']:
            self.last_turn_for_parking = True
            self.turn_speed = 250.0
            self.base_speed = 250.0
        else: 
            self.last_turn_for_parking = False

        if self.target_turns == self.turn_count:
            self.parking_straight = True
            self.IDEAL_RADIUS_M = 0.20
            self.base_speed = 250.0
            self.turn_speed = 250.0
        else:
            self.parking_straight = False


    def imu_callback(self, msg):
        """
        Gibt die Ergebnisse des IMU Sensors zurück.
        """
        self.imu_ready = True
        q = msg.orientation
        
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        yaw_rad = math.atan2(siny_cosp, cosy_cosp)
        
        raw_yaw = math.degrees(yaw_rad)
        
        if self.last_raw_yaw is None:
            self.last_raw_yaw = raw_yaw
            self.current_yaw = raw_yaw
            return

        delta = raw_yaw - self.last_raw_yaw
        
        if delta > 180.0:
            delta -= 360.0
        elif delta < -180.0:
            delta += 360.0
            
        self.current_yaw += delta
        self.last_raw_yaw = raw_yaw
    
    def camera_sub_callback(self, msg):
        cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        
        results = self.model.predict(
            cv_image, 
            half=True,    
            imgsz=640,    
            device=0,     
            verbose=False,
            conf=0.75 
        )

        if len(results[0].boxes) > 0:
            raw_boxes = results[0].boxes.xyxy.cpu().numpy().astype(float)
            
            if self.camera_calibration:
                self.test_log_bbox_y_values(raw_boxes)

        if self.pub_debug_img.get_subscription_count() > 0:
            annotated_frame = results[0].plot()
            debug_msg = self.bridge.cv2_to_imgmsg(annotated_frame, encoding="bgr8")
            self.pub_debug_img.publish(debug_msg)

        with self.data_lock:
            self.latest_yolo_results = results

    def button_callback(self, msg):
        if msg.data:
            self.get_logger().info("Hardware-Interrupt empfangen. Trigger ausgelöst.")
            self.button_state = True

    def test_log_bbox_y_values(self, bounding_boxes):
        """
        Testfunktion für die Zollstock-Kalibrierung der Kamera.
        Gibt die Y-Pixelwerte der erkannten Bounding Boxes im Terminal aus.
        """
        if bounding_boxes is None:
            return

        self.get_logger().info(f"--- Starte Y-Wert Messung ({len(bounding_boxes)} Objekt(e)) ---")
        
        for i, bbox in enumerate(bounding_boxes):
            try:
                y_min = int(bbox[1]) # Oberkante des Objekts
                y_max = int(bbox[3]) # Unterkante des Objekts
                
                self.get_logger().info(f"Objekt {i+1} -> Oberkante: {y_min} px | (Unterkante/Boden): y_max = {y_max} px")

            except IndexError:
                self.get_logger().error(f"Fehler: Bounding Box hat unerwartetes Format: {bbox}")

    def send_text(self, marker_array, m_id, text, x, y, color=(1.0, 1.0, 1.0), scale=0.15):
        """
        Fügt einen Text-Marker zum MarkerArray hinzu.
        Schriftgröße in Metern
        """
        marker = Marker()
        marker.header.frame_id = "base_link"
        marker.header.stamp = self.get_clock().now().to_msg()
        
        marker.ns = "text_labels"
        marker.id = m_id
        marker.type = Marker.TEXT_VIEW_FACING
        marker.action = Marker.ADD
        
        # Position
        marker.pose.position.x = float(x)
        marker.pose.position.y = float(y)
        marker.pose.position.z = 0.1 
        
        marker.pose.orientation.w = 1.0
        
        marker.scale.z = float(scale) 
        
        # Farbe und Sichtbarkeit
        marker.color.r = float(color[0])
        marker.color.g = float(color[1])
        marker.color.b = float(color[2])
        marker.color.a = 1.0
        
        marker.text = str(text)
        
        marker_array.markers.append(marker)

    def send_sphere(self, marker_array, m_id, x, y, color=(0.0, 1.0, 1.0)):
        """Zeichnet einen Punkt in RViz."""
        marker = Marker()
        marker.header.frame_id = self.rviz_frame
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "target"
        marker.id = m_id
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.position.x = float(x)
        marker.pose.position.y = float(y)
        marker.pose.position.z = 0.05
        marker.scale.x = 0.15
        marker.scale.y = 0.15
        marker.scale.z = 0.15
        marker.color.r, marker.color.g, marker.color.b = color
        marker.color.a = 1.0
        marker_array.markers.append(marker)
    
    def validate_clusters_straight(self, clusters):
        u_profile = [None, None, None]
        if not clusters:
            return u_profile

        delta_yaw = 0.0
        if not self.last_turn_aborted:
            delta_yaw = self.current_yaw - self.start_straight_yaw
            
            # Normierung auf -180 bis 180
            while delta_yaw > 180: delta_yaw -= 360
            while delta_yaw < -180: delta_yaw += 360

        right_candidates = []
        left_candidates = []
        front_candidates = []

        for c in clusters:
            if len(c) < 20: continue
            
            local_angle = self.get_cluster_angle(c)
            if local_angle is None: continue

            shifted_angle = local_angle - delta_yaw
            
            angle_norm = abs(shifted_angle) % 180
            if angle_norm > 90:
                angle_norm = 180 - angle_norm

            if angle_norm <= 45.0:
                mean_x_local = sum(p[1] for p in c) / len(c)
                
                if abs(mean_x_local) <= 1.0:
                    if mean_x_local < 0:
                        left_candidates.append(c)
                    else:
                        right_candidates.append(c)
                        
            # FRONTWÄNDE (> 45°)
            else:
                # p[2] ist Vorne/Hinten
                mean_y = sum(p[2] for p in c) / len(c)
                if mean_y >= 0.15: # Nur Wände vor dem Roboter
                    front_candidates.append(c)

        # SEITENWÄNDE ZUORDNEN
        best_right = right_candidates[0] if right_candidates else None
        best_left = left_candidates[0] if left_candidates else None
        
        best_left_hnf = self.cluster_to_hnf(best_left) if (best_left is not None) else None
        best_right_hnf = self.cluster_to_hnf(best_right) if (best_right is not None) else None

        if (best_right is not None) and (best_left is not None):
            self.visualize_cluster_line(best_right, 10, "cyan")
            self.visualize_cluster_line(best_left, 11, "magenta")
            
            if best_left_hnf is not None and best_right_hnf is not None:
                _, _, left_dist = best_left_hnf
                _, _, right_dist = best_right_hnf
                
                track_width = abs(left_dist) + abs(right_dist)
                self.get_logger().info(f"Echte Spurbreite: {track_width:.2f}m, L: {abs(left_dist):.2f}m, R: {abs(right_dist):.2f}m")
                
                if 0.75 <= track_width <= 1.25:
                    u_profile[0] = best_right
                    u_profile[2] = best_left
                else:
                    self.get_logger().warn(f"Spurbreite unplausibel ({track_width:.2f}m). Verwerfe kürzere Wand.")
                    len_r = math.hypot(best_right[-1][1] - best_right[0][1], best_right[-1][2] - best_right[0][2])
                    len_l = math.hypot(best_left[-1][1] - best_left[0][1], best_left[-1][2] - best_left[0][2])
                    
                    if len_r > len_l:
                        u_profile[0] = best_right
                    else:
                        u_profile[2] = best_left
            else:
                self.get_logger().warn("HNF Berechnung für eine der Wände fehlgeschlagen.")
                return [None, None, None]
                    
        elif best_right is not None:
            u_profile[0] = best_right
        elif best_left is not None:
            u_profile[2] = best_left

        # FRONTWAND ZUORDNEN
        if front_candidates:
            groups = []
            for c in front_candidates:
                mean_y = sum(p[2] for p in c) / len(c)
                
                # Alles was extrem nah ist, verwerfen
                if mean_y < 0.20:
                    continue 
                
                placed = False
                for g in groups:
                    if abs(g['base_y'] - mean_y) < 0.15:
                        g['clusters'].append(c)
                        placed = True
                        break
                if not placed:
                    groups.append({'base_y': mean_y, 'clusters': [c]})

            valid_wall_groups = []
            for g in groups:
                all_x = [p[1] for c in g['clusters'] for p in c]
                total_width = max(all_x) - min(all_x)
                
                min_x = min(all_x)
                max_x = max(all_x)
                mean_y = g['base_y']  # Distanz zur Wand

                is_far_wall = mean_y > 0.70
                min_allowed_width = 0.45 if self.is_start_finish_straight else 0.35
                
                is_blocking_path = (min_x < -0.15) and (max_x > 0.15)
                
                if total_width > min_allowed_width and (is_far_wall or is_blocking_path):
                    if not is_far_wall and is_blocking_path:
                        self.get_logger().warn(f"Nahe aber blockierende Wand gefunden. Abstand: {mean_y:.2f}m")
                    valid_wall_groups.append(g)

            if valid_wall_groups:
                valid_wall_groups.sort(key=lambda g: g['base_y'])
                winner_group = valid_wall_groups[0]['clusters']
                winner_group.sort(key=len, reverse=True)
                u_profile[1] = winner_group[0]
            else:
                u_profile[1] = None
                if front_candidates:
                    self.get_logger().debug("Front-Kandidaten vorhanden, aber als Phantomwände abgelehnt.")

        return u_profile

    def validate_clusters_turn(self, front_wall, point_data):
        '''Überprüft die Cluster in der Kurve. Es muss eine Frontwand geben, Außenbanden können fehlen (z.B. bei der ersten Kurve).
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
                is_in_killzone = any(left_angle < point[0] < right_angle for point in cluster)
                
                if not is_in_killzone:
                    kept_clusters.append(cluster)
                    
            return kept_clusters


        front_wall_cluster, combined_clusters = self.merge_clusters(clusters, [front_wall])
        front_wall_cluster = front_wall_cluster[0]
        
        clusters = kill_all_clusters_between(front_wall_cluster, clusters)
        clusters.append(front_wall_cluster)

        if len(clusters) >= 2:
            minimal_cluster_size = 25
            ordered = self.sort_clusters_right_to_left(clusters)
            u_profile = [None, None, None] # 0=Rechts, 1=Front, 2=Links
            if any(c is front_wall_cluster for c in ordered):
                u_profile[1] = front_wall_cluster
                fw_index = next(i for i, c in enumerate(ordered) if c is front_wall_cluster)
            else: 
                self.get_logger().warn("Frontwand nicht in den Clustern gefunden. Kann Kurvenprofil nicht validieren.")
                return [None, None, None]
                
            while u_profile[2] is None and fw_index > 0:
                if len(ordered[fw_index - 1]) > minimal_cluster_size:
                    u_profile[2] = ordered[fw_index - 1]
                    
                else: 
                    ordered.pop(fw_index - 1)
                    fw_index -= 1

            while u_profile[0] is None and fw_index < len(ordered) - 1:
                if len(ordered[fw_index + 1]) > minimal_cluster_size:
                   
                    u_profile[0] = ordered[fw_index + 1]
                    
                else: 
                    ordered.pop(fw_index + 1)

            angles = [self.get_cluster_angle(c) for c in u_profile]
            if angles[1] is None:
                self.get_logger().warn("Fehler bei der Winkelberechnung der Frontwand. Überspringe...")
                return [None, None, None]

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

            if diff_0_1 is not None and diff_0_1 < 70:
                self.get_logger().warn(f"Rechts und Front nicht orthogonal! Diff: {diff_0_1:.1f}°")
                u_profile[0] = None

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
                    if 0.90 < wall_dist_left < 1.80:
                        self.get_logger().warn(f"Abstand der linken Wand {wall_dist_left:.2f}")
                        u_profile[2] = None

            else: 
                if wall_dist_left is not None:
                    if wall_dist_left > 1.0:
                        self.get_logger().warn(f"Abstand der linken Wand {wall_dist_left:.2f}")
                        u_profile[2] = None

                if wall_dist_right is not None:
                    if 0.90 < wall_dist_right < 1.8:
                        self.get_logger().warn(f"Abstand der rechten Wand {wall_dist_right:.2f}")
                        u_profile[0] = None
            
            u_profile, _ = self.merge_clusters(clusters, u_profile)
            self.visualize_cluster_line(u_profile[0], 0, "cyan")
            self.visualize_cluster_line(u_profile[1], 1, "rot")
            self.visualize_cluster_line(u_profile[2], 2, "cyan")
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
            # Schwerpunkt des Clusters berechnen
            # Index 1 = X, Index 2 = Y
            mean_x = sum(p[1] for p in cluster) / len(cluster)
            mean_y = sum(p[2] for p in cluster) / len(cluster)
            
            # Winkel berechnen (0 = Vorne, Negativ = Rechts, Positiv = Links)
            return math.atan2(-mean_x, mean_y)

        # Aufsteigend sortieren
        sorted_clusters = sorted(clusters, key=get_cluster_bearing, reverse=True)
        
        return sorted_clusters

    def get_all_clusters_sorted(self, point_data):
        if len(point_data) < 2:
            return []

        # Sortieren
        points = point_data[np.argsort(point_data[:, 0])]
        x = points[:, 1]
        y = points[:, 2]

        split_mask = np.zeros(len(points), dtype=bool)

        # LÜCKENERKENNUNG (Manhattan-Distanz)
        dx = np.diff(x)
        dy = np.diff(y)
        dist_manhattan = np.abs(dx) + np.abs(dy)
        
        split_mask[1:] = dist_manhattan >= 0.15

        #AUSREISSER-LOGIK
        if len(points) > 2:
            dist_2 = np.abs(x[2:] - x[:-2]) + np.abs(y[2:] - y[:-2])
            fix_2 = (split_mask[2:]) & (dist_2 < 0.15)
            split_mask[2:][fix_2] = False 

        if len(points) > 3:
            dist_3 = np.abs(x[3:] - x[:-3]) + np.abs(y[3:] - y[:-3])
            fix_3 = (split_mask[3:]) & (dist_3 < 0.15)
            split_mask[3:][fix_3] = False 

        # ECKEN-ERKENNUNG
        if len(points) > 6:
            vec_a_x = x[3:-3] - x[:-6]
            vec_a_y = y[3:-3] - y[:-6]
            
            vec_b_x = x[6:] - x[3:-3]
            vec_b_y = y[6:] - y[3:-3]

            len_a = np.hypot(vec_a_x, vec_a_y)
            len_b = np.hypot(vec_b_x, vec_b_y)

            valid = (len_a > 0.01) & (len_b > 0.01) & (len_a < 0.30) & (len_b < 0.30)

            dot_product = np.ones(len(points) - 6)
            dot_product[valid] = (
                (vec_a_x[valid] / len_a[valid]) * (vec_b_x[valid] / len_b[valid]) +
                (vec_a_y[valid] / len_a[valid]) * (vec_b_y[valid] / len_b[valid])
            )

            sharp_angles = dot_product < 0.70
            padded_dot = np.pad(dot_product, (1, 1), mode='edge')
            is_local_min = (dot_product <= padded_dot[:-2]) & (dot_product <= padded_dot[2:])

            corner_splits = sharp_angles & is_local_min
            
            corner_indices = np.where(corner_splits)[0] + 3 
            if len(corner_indices) > 0:
                idx_prev = np.clip(corner_indices - 1, 0, len(split_mask) - 1)
                idx_next = np.clip(corner_indices + 1, 0, len(split_mask) - 1)
                
                # Alle 3 Punkte auf einen Schlag True setzen (C-Speed)
                split_mask[corner_indices] = True
                split_mask[idx_prev] = True
                split_mask[idx_next] = True

        split_indices = np.where(split_mask)[0]
        clusters_raw = np.split(points, split_indices)
        
        clusters = [c for c in clusters_raw if len(c) > 2]

        if len(clusters) > 1:
            c_last = clusters[-1]
            c_first = clusters[0]
            
            dist_wrap = abs(c_first[0][1] - c_last[-1][1]) + abs(c_first[0][2] - c_last[-1][2])
            
            if dist_wrap < 0.15 and len(c_last) > 3 and len(c_first) > 3:
                v1_x = c_last[-1][1] - c_last[-4][1]
                v1_y = c_last[-1][2] - c_last[-4][2]
                v2_x = c_first[3][1] - c_first[0][1]
                v2_y = c_first[3][2] - c_first[0][2]
                
                l1 = math.hypot(v1_x, v1_y)
                l2 = math.hypot(v2_x, v2_y)
                
                if l1 > 0 and l2 > 0:
                    dot_wrap = (v1_x * v2_x + v1_y * v2_y) / (l1 * l2)
                    if dot_wrap >= 0.85: 
                        clusters[0] = np.vstack((c_last, c_first))
                        clusters.pop()

        # NACH PHYSISCHER LÄNGE SORTIEREN (METER)
        def get_physical_length(c):
            if len(c) < 2: return 0.0
            # Satz des Pythagoras zwischen dem ersten und dem letzten Punkt
            return math.hypot(c[-1][1] - c[0][1], c[-1][2] - c[0][2])

        clusters.sort(key=get_physical_length, reverse=True)
        return clusters
    
    def get_cluster_angle(self, cluster):
        if cluster is None or len(cluster) < 2:
            return None
        
        # Wandelt cluster (Liste) in ein NumPy Array um
        pts = np.array(cluster) 
        x = pts[:, 1]
        y = pts[:, 2]
        
        # Schwerpunkt Berechnung
        x_mean = np.mean(x)
        y_mean = np.mean(y)
        dx = x - x_mean
        dy = y - y_mean
        
        s_xx = np.sum(dx * dx)
        s_yy = np.sum(dy * dy)
        s_xy = np.sum(dx * dy)
        
        angle_rad = 0.5 * np.arctan2(2.0 * s_xy, s_yy - s_xx)
        return np.degrees(angle_rad)

    def delete_marker(self, marker_array, m_id, ns="walls"):
        """Löscht einen Marker in einem bestimmten Namespace."""
        marker = Marker()
        marker.header.frame_id = self.rviz_frame
        marker.ns = ns
        marker.id = m_id
        marker.action = Marker.DELETE
        marker_array.markers.append(marker)

    def scan_callback(self, msg):
        ranges = np.array(msg.ranges)

        # Anforderungen für gültige Werte definieren
        valid_mask = np.isfinite(ranges) & (ranges >= 0.075) & (ranges <= 3.0)

        # Lidar-Winkel für alle Punkte generieren
        angles_rad = msg.angle_min + np.arange(len(ranges)) * msg.angle_increment

        # Nur gültige Werte übernehmen
        valid_ranges = ranges[valid_mask]
        valid_angles_rad = angles_rad[valid_mask]

        # Koordinaten berechnen
        x_ros = valid_ranges * np.cos(valid_angles_rad)
        y_ros = valid_ranges * np.sin(valid_angles_rad)

        angles_deg = np.degrees(valid_angles_rad)
        user_angles_deg = np.mod(angles_deg + 90.0, 360.0)

        # Als N x 4 Array zusammenfügen: (Winkel, X, Y, Distanz)
        point_data = np.column_stack((user_angles_deg, x_ros, y_ros, valid_ranges))
        
        self.last_point_data = point_data

        self.update_strategy_params()

        if self.camera_calibration or self.debug:
            self.debug_main_logic(point_data)
        else:
            self.main_logic(point_data)
    
    def get_closest_point_in_cluster(self, cluster):
        """
        Gibt den Punkt eines Clusters zurück, der den kürzesten Abstand zum LiDAR hat.
        """
        if cluster is None or len(cluster) == 0:
            return None

        # Sucht das Element im Cluster mit geringster Dist
        closest_point = min(cluster, key=lambda p: p[3])
        
        return closest_point

    def merge_clusters(self, all_clusters, validated_clusters):
        """
        Versucht, benachbarte Cluster zu einem einzigen Cluster zu verschmelzen.
        Prüft den orthogonalen (senkrechten) Abstand und den parallelen (Längs-) Abstand.
        """
        # KONFIGURATION DER TOLERANZEN
        max_angle_gap = 10.0      # 10 Grad maximale Winkelabweichung
        max_perp_gap = 0.08       # Max 8 cm Abstand weg von der Wand
        max_parallel_gap = 0.60   # Max 60 cm Lücke entlang der Wand

        valid_ids = [id(v) for v in validated_clusters]
        remaining_clusters = [c for c in all_clusters if id(c) not in valid_ids]
        
        if not remaining_clusters:
            return validated_clusters, []

        def get_angle_diff(a1, a2):
            diff = abs(a1 - a2) % 180
            if diff > 90:
                diff = 180 - diff
            return diff

        combined_clusters = [[] for _ in range(len(validated_clusters))]

        for i, valid_cluster in enumerate(validated_clusters):
            angle = self.get_cluster_angle(valid_cluster)
            if valid_cluster is None or len(valid_cluster) == 0:
                continue

            if angle is None: 
                continue

            # Schwerpunkt des Clusters berechnen
            bx = np.mean(valid_cluster[:, 1])
            by = np.mean(valid_cluster[:, 2])

            angle_rad = math.radians(angle)
            
            dir_x = math.sin(angle_rad)
            dir_y = math.cos(angle_rad)

            nx = math.cos(angle_rad)
            ny = -math.sin(angle_rad)

            clusters_to_remove = []

            for other in remaining_clusters:
                other_angle = self.get_cluster_angle(other)
                if other_angle is None: 
                    continue

                if get_angle_diff(angle, other_angle) < max_angle_gap:
                    ox = np.mean(other[:, 1])
                    oy = np.mean(other[:, 2])
                    offset_perp = abs((ox - bx) * nx + (oy - by) * ny)

                    proj_valid = valid_cluster[:, 1] * dir_x + valid_cluster[:, 2] * dir_y
                    proj_other = other[:, 1] * dir_x + other[:, 2] * dir_y
                    
                    min_v, max_v = np.min(proj_valid), np.max(proj_valid)
                    min_o, max_o = np.min(proj_other), np.max(proj_other)
                    
                    offset_parallel = max(0, min_o - max_v, min_v - max_o)

                    if offset_perp < max_perp_gap and offset_parallel < max_parallel_gap:
                        validated_clusters[i] = np.vstack((validated_clusters[i], other))
                        
                        clusters_to_remove.append(other)
                        combined_clusters[i].append(other)

            remove_ids = [id(c) for c in clusters_to_remove]
            remaining_clusters = [c for c in remaining_clusters if id(c) not in remove_ids]

            projections = validated_clusters[i][:, 1] * dir_x + validated_clusters[i][:, 2] * dir_y
            sort_indices = np.argsort(projections)
            validated_clusters[i] = validated_clusters[i][sort_indices]

        return validated_clusters, combined_clusters

    def visualize_target_point(self, x, y, m_id=24, farbe_name="gelb", label="TARGET"):
        """
        Visualisiert den Zielpunkt als Punkt und Text-Label im Foxglove.
        """

        farben = {
            "rot": (1.0, 0.0, 0.0),
            "gruen": (0.0, 1.0, 0.0),
            "blau": (0.0, 0.5, 1.0),
            "cyan": (0.0, 1.0, 1.0),
            "magenta": (1.0, 0.0, 1.0),
            "gelb": (1.0, 1.0, 0.0),
            "orange": (1.0, 0.5, 0.0)
        }
        rgb = farben.get(farbe_name.lower(), (1.0, 1.0, 0.0))

        marker_array = MarkerArray()

        sphere_marker = Marker()
        sphere_marker.header.frame_id = "base_link"
        sphere_marker.header.stamp = self.get_clock().now().to_msg()
        sphere_marker.ns = "target_point"
        sphere_marker.id = m_id
        sphere_marker.type = Marker.SPHERE
        sphere_marker.action = Marker.ADD
        
        # Position setzen
        sphere_marker.pose.position.x = float(x)
        sphere_marker.pose.position.y = float(y)
        sphere_marker.pose.position.z = 0.1  # 10cm über dem Boden
        
        # Größe
        sphere_marker.scale.x = 0.15
        sphere_marker.scale.y = 0.15
        sphere_marker.scale.z = 0.15
        
        # Farbe setzen
        sphere_marker.color.r = rgb[0]
        sphere_marker.color.g = rgb[1]
        sphere_marker.color.b = rgb[2]
        sphere_marker.color.a = 1.0
        
        marker_array.markers.append(sphere_marker)

        self.send_text(marker_array, m_id=m_id + 1, text=label, x=x, y=y, color=rgb)

        self.pub_markers.publish(marker_array)

    def get_target_point_straight(self, hnf_innen, hnf_aussen):
        """
        Berechnet den Zielpunkt mithilfe der HNF.
        Ist nur eine Wand vorhanden, wird der Offset direkt von dieser berechnet, ohne eine fehleranfällige virtuelle Wand zu projizieren!
        """
        target_y = self.lookahead_dist_straight
        target_x = 0.0
        
        # Wände, die zu weit weg sind, ignorieren
        if hnf_innen is not None and hnf_innen[2] > 1.0:
            hnf_innen = None
        if hnf_aussen is not None and hnf_aussen[2] > 1.0:
            hnf_aussen = None

        def get_x_at_y(hnf_params, y_val):
            nx, ny, d = hnf_params
            if abs(nx) < 1e-6: return 0.0 
            return (d - ny * y_val) / nx

        # Schnittpunkt mit Y-Sichtachse berechnen
        x_innen = get_x_at_y(hnf_innen, target_y) if hnf_innen else None
        x_aussen = get_x_at_y(hnf_aussen, target_y) if hnf_aussen else None

        if x_innen is not None and x_aussen is not None:
            if x_innen < 0: # Innenbande ist links
                t_innen = x_innen + self.lane_ratio
                t_aussen = x_aussen - (1.0 - self.lane_ratio)
            else:           # Innenbande ist rechts
                t_innen = x_innen - self.lane_ratio
                t_aussen = x_aussen + (1.0 - self.lane_ratio)
                
            target_x = (t_innen + t_aussen) / 2.0

        elif x_innen is not None:
            if x_innen < 0: 
                target_x = x_innen + self.lane_ratio
            else:           
                target_x = x_innen - self.lane_ratio

        elif x_aussen is not None:
            if x_aussen > 0: # Außenbande ist rechts
                target_x = x_aussen - (1.0 - self.lane_ratio)
            else:            # Außenbande ist links
                target_x = x_aussen + (1.0 - self.lane_ratio)
                
        else:
            # Notfall, beide Wände fehlen komplett
            target_x = 0.0

        if x_innen is not None:
            if x_innen < 0 and target_x < x_innen + self.min_wall_dist:
                target_x = x_innen + self.min_wall_dist
            elif x_innen > 0 and target_x > x_innen - self.min_wall_dist:
                target_x = x_innen - self.min_wall_dist
                
        if x_aussen is not None:
            if x_aussen < 0 and target_x < x_aussen + self.min_wall_dist:
                target_x = x_aussen + self.min_wall_dist
            elif x_aussen > 0 and target_x > x_aussen - self.min_wall_dist:
                target_x = x_aussen - self.min_wall_dist

        return (target_x, target_y)

    def track_front_wall(self, point_data, last_front_wall):
        if self.state == 'PARKING':
            richtung = self.park_turn_richtung
        else:
            richtung = self.fahrtrichtung

        if last_front_wall is None or len(point_data) == 0:
            return None

        last_fw_array = np.array(last_front_wall)
        min_angle = np.min(last_fw_array[:, 0])
        max_angle = np.max(last_fw_array[:, 0])

        # Dynamisches Suchfenster
        if richtung == 'links':
            roi_min = min_angle - 30.0
            roi_max = max_angle + 5.0
        else:
            roi_min = min_angle - 5.0
            roi_max = max_angle + 30.0

        mask = (point_data[:, 0] >= roi_min) & (point_data[:, 0] <= roi_max)
        roi_points = point_data[mask]

        roi_clusters = self.get_all_clusters_sorted(roi_points)

        if not roi_clusters:
            self.get_logger().warn("ACHTUNG: Getrackte Wand im ROI verloren!")
            return last_front_wall

        return roi_clusters[0]

    def get_closest_measure(self, point_data, target_angle):
        if len(point_data) == 0:
            self.get_logger().info(f"Point_Data ist leer!")
            return None

        diffs = (point_data[:, 0] - target_angle + 180.0) % 360.0 - 180.0
        abs_diffs = np.abs(diffs)

        closest_idx = np.argmin(abs_diffs)
        
        return point_data[closest_idx]

    # ------------------------
    # --- YOLO - Functions ---
    # ------------------------
        
    def get_obstacles_from_camera(self, point_data): 
        with self.data_lock:
            results = self.latest_yolo_results
            
        if not results:
            return None

        # KONFIGURATION FÜR 1280p
        H_FOV = 120.0  
        IMAGE_WIDTH = 1280.0
        CENTER_X = IMAGE_WIDTH / 2.0
        DEG_PER_PIXEL = H_FOV / IMAGE_WIDTH 

        class_mapping = {0: 'green', 1: 'red', 2: 'pink'}

        yolo_boxes = []

        for r in results:
            for box in r.boxes:
                if float(box.conf[0]) > 0.75:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    
                    # Winkelberechnung
                    center_x = (x1 + x2) / 2.0
                    pixel_offset = center_x - CENTER_X
                    angle_offset_deg = pixel_offset * DEG_PER_PIXEL
                    
                    # Berechnung des Winkels relativ zum Roboter
                    cam_angle_deg = 180.0 - angle_offset_deg + self.angle_calibration
                    cam_angle_rad = math.radians(cam_angle_deg)
                    
                    class_id = int(box.cls[0])
                    class_name = class_mapping.get(class_id, f"unknown_{class_id}")

                    y_max = float(y2)

                    yolo_boxes.append({
                        'y_max': y_max, 
                        'angle_rad': cam_angle_rad,
                        'class_name': class_name,
                        'class_id': class_id
                    })

        if not yolo_boxes:
            return []

        # Sortieren: Nah nach Fern
        yolo_boxes.sort(key=lambda b: b['y_max'], reverse=True)

        detected_obstacles = []
        available_clusters = self.get_all_clusters_sorted(point_data)

        for box_data in yolo_boxes:
            # Cluster finden
            obstacle_cluster = self.get_lidar_distance(box_data['angle_rad'], available_clusters)

            if obstacle_cluster is not None:
                obj_x, obj_y = self.get_weight_point_for_cluster(obstacle_cluster) 
                
                lidar_dist = abs(obj_y)
                
                # Kamera-Distanz
                camera_dist = self.analyzer.get_distance_from_bbox(box_data['y_max'])
                
                TOLERANZ = 0.40 
                
                if abs(camera_dist - lidar_dist) > TOLERANZ:
                    self.get_logger().warn(
                        f"Geister-Filter: {box_data['class_name']} abgelehnt! "
                        f"Kamera: {camera_dist:.2f}m, LiDAR: {lidar_dist:.2f}m."
                    )
                    continue 

                # Marker senden & Objekt speichern
                self.publish_marker(obj_x, obj_y, box_data['class_name'], box_data['class_id'])
                detected_obstacles.append((obj_x, obj_y, box_data['class_name']))
                
                available_clusters = [c for c in available_clusters if not np.array_equal(c, obstacle_cluster)]
                        
        return detected_obstacles
    
    def get_weight_point_for_cluster(self, cluster):
        if cluster is None or len(cluster) == 0:
            return None
            
        n = len(cluster)
        sum_x = sum(p[1] for p in cluster)
        sum_y = sum(p[2] for p in cluster)
        
        return sum_x / n, sum_y / n

    def get_lidar_distance(self, camera_angle_rad, clusters):
        walls = [self.front_wall, self.left_wall, self.right_wall]
        wall_ids = [id(w) for w in walls if w is not None]
        clusters_without_walls = [c for c in clusters if id(c) not in wall_ids and c is not None]
        
        if not clusters_without_walls: 
            return None
            
        best_cluster = None
        best_closest_point = None
        min_dist = 4.0
        best_angle_deg = 0.0
        
        for cluster in clusters_without_walls:
            # Filtern nach Punkte-Anzahl (Mindestens 2, max 60 für nahe Hindernisse)
            if len(cluster) < 2 or len(cluster) > 60:
                continue
                
            # Breiten-Filter
            c_start = cluster[0]   
            c_end = cluster[-1]    
            width = math.hypot(c_start[1] - c_end[1], c_start[2] - c_end[2])
            if width > 0.35:
                continue

            angle_deg = self.middle_of_cluster(cluster)
            if angle_deg is None: continue
            angle_rad = math.radians(angle_deg)
            
            diff = abs(self.angle_diff(angle_rad, camera_angle_rad))
            if diff < math.radians(12.0):
                
                closest_point = self.get_closest_point_in_cluster(cluster)
                if closest_point is not None:
                    dist = closest_point[3]
                    
                    if dist < min_dist:
                        min_dist = dist
                        best_closest_point = closest_point
                        best_angle_deg = angle_deg
                        best_cluster = cluster
                        
        if best_cluster is not None:
            self.get_logger().info(f"MATCH: Kamera {math.degrees(camera_angle_rad):.1f}° -> Lidar {best_angle_deg:.1f}° (Distanz: {min_dist:.2f}m), yAchsenAbstand: {closest_point[2]:.2f}m")
            return best_cluster
            
        return None

    def middle_of_cluster(self, cluster):
        """Berechnet den Winkel des Mittelpunktes des Clusters zum Lidar."""
        sum = 0
        for c in cluster:
            sum += c[0]  # Winkel des Punktes
        return sum / len(cluster)

    def angle_diff(self, a, b):
        """Berechnet den Unterschied zwischen zwei Winkeln (Rad)."""
        return math.atan2(math.sin(a - b), math.cos(a - b))


    # -----------------------------------------------
    # Neue Koordinatensystem Idee für Kurvenfahrten
    # -----------------------------------------------

    def cluster_to_hnf(self, cluster):
        """
        Berechnet die Hessesche Normalform aus einem Cluster von Messpunkten.
        """
        if cluster is None:
            return None
        
        # Extrahieren der x- und y-Koordinaten in ein NumPy-Array
        # Index 1 ist x_coord, Index 2 ist y_coord
        points = np.array([[p[1], p[2]] for p in cluster])
        
        if len(points) == 0:
            raise ValueError("Das Cluster ist leer.")
            
        # Schwerpunkt berechnen
        centroid = np.mean(points, axis=0)
        
        # Daten zentrieren
        centered_points = points - centroid
        
        # Singulärwertzerlegung (SVD) für orthogonale Regression
        # Vh enthält die Eigenvektoren der Kovarianzmatrix
        _, _, Vh = np.linalg.svd(centered_points)
        
        # Der Normalenvektor entspricht dem Eigenvektor der geringsten Varianz
        normal_vector = Vh[-1]
        
        # Abstand d berechnen (Skalarprodukt aus Schwerpunkt und Normalenvektor)
        d = np.dot(centroid, normal_vector)
        
        # Normierung: d muss größer oder gleich 0 sein
        if d < 0:
            normal_vector = -normal_vector
            d = -d
            
        n_x, n_y = normal_vector
        
        return n_x, n_y, d
    
    def extract_wall_lines(self, u_profile):
        if self.state == 'PARKING':
            richtung = self.park_turn_richtung
        else:
            richtung = self.fahrtrichtung
        if richtung == "links":
            # Bei Linkskurve ist Links (2) die Innenbande (side)
            opposite_cluster, front_cluster, side_cluster = u_profile
        else:
            # Bei Rechtskurve ist Rechts (0) die Innenbande (side)
            side_cluster, front_cluster, opposite_cluster = u_profile

        # Frontwand extrahieren
        if front_cluster is None or len(front_cluster) < 2:
            front_straight = None
        else:
            front_straight = self.cluster_to_hnf(front_cluster)

        # Seitenwand extrahieren
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

    def calculate_target_line(self, side_line_params, front_line_params, desired_lane_ratio):
        if front_line_params is None:
            return None, None

        n_xf, n_yf, d_f = front_line_params
        
        d_ziel = d_f - (1.0 - desired_lane_ratio)
            
        target_line_params = (n_xf, n_yf, d_ziel)
        
        if side_line_params is None:
            return target_line_params, None

        # Radius Limitierung
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
        if self.state == 'PARKING':
            richtung = self.park_turn_richtung
        else:
            richtung = self.fahrtrichtung

        if target_line_params is None:
            return None, None, None
            
        n_x, n_y, d = target_line_params
        
        # Prüfung auf Parallelität (Vermeidung von Division durch Null)
        epsilon = 1e-6
        if abs(n_y) < epsilon:
            # Zielgerade ist parallel zur Fahrtrichtung, kein Schnittpunkt
            return None, None, None
            
        # Schnittpunkt berechnen
        intersection_x_m = 0.0  # Roboter fährt per Definition auf x=0
        intersection_y_m = d / n_y
        
        # Schnittpunkt muss vor dem Roboter liegen
        if intersection_y_m <= 0.0:
            # Schnittpunkt liegt hinter dem Roboter.
            return None, None, None
        
        # Vorzeichen des Skalarprodukts
        if richtung == 'links':
            nx_directional = n_x
        else:
            nx_directional = -n_x
            
        nx_clipped = max(-1.0, min(1.0, nx_directional))
        
        # Winkel berechnen
        turn_angle_deg = math.degrees(math.acos(nx_clipped))
        self.get_logger().info(f"Turn Angle Fehler: {turn_angle_deg:.1f}°")

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
            curve_radius_m = self.MIN_TURN_RADIUS_M

        else:
            curve_radius_m = max(self.MIN_TURN_RADIUS_M, min(self.IDEAL_RADIUS_M, max_allowed_radius_m))

        alpha_rad = math.radians(abs(turn_angle_deg))
        tangent_length_m = curve_radius_m * math.tan(alpha_rad / 2.0)
        
        lidar_entry_dist_m = intersection_y_m - tangent_length_m
        
        real_axle_dist_m = lidar_entry_dist_m + self.LIDAR_OFFSET_M
        
        if real_axle_dist_m <= 0.0:
            self.get_logger().warn(
                f"NOT-EINLENKEN: Drehpunkt verpasst ({real_axle_dist_m:.2f} m). "
                "Forciere sofortige Kurve."
            )

            real_axle_dist_m = 0.0
            
        return curve_radius_m, real_axle_dist_m

    def check_turn_trigger(self, entry_point_distance_m):
        """
        Prüft anhand der aktuell berechneten Distanz, ob das Einlenkmanöver starten muss.
        """
        if entry_point_distance_m is None:
            return False

        trigger_tolerance_m = 0.07

        if entry_point_distance_m <= trigger_tolerance_m:
            return True
            
        return False

    def execute_turn(self, curve_radius_m):
        """
        Übersetzt den Radius über den SteeringController und publisht die Twist-Message.
        """

        if self.state == 'PARKING':
            richtung = self.park_turn_richtung
        else:
            richtung = self.fahrtrichtung

        if richtung == "links":
            is_left_turn = True
        else:
            is_left_turn = False

        cmd = Twist()
        
        if curve_radius_m is None or curve_radius_m <= 0.0:
            self.get_logger().error("Ungültiger Kurvenradius.")
            return False
        
        steering_signal = self.steering_ctrl.get_steering_for_radius(target_radius=curve_radius_m,fahrtrichtung_ist_links=is_left_turn)
        self.get_logger().info(f"Steering-Signal: {steering_signal:.3f}")

        if self.park_direction == 'PARKING_RIGHT_OBST' and self.parking_phase == 'EXECUTE_TURN':
            self.turn_speed = - abs(self.turn_speed)
            steering_signal = - abs(steering_signal)

        cmd.linear.x = float(self.turn_speed)
        cmd.angular.z = float(steering_signal)
        
        self.pub_cmd_vel.publish(cmd)

        self.turn_speed = abs(self.turn_speed)
        
        return True

    def check_turn_completion_fused(self, turn_angle, front_line_params):
        """
        Gibt True zurück wenn die Kurve nach Gyro oder Wandwinkel beendet ist.
        """

        yaw_diff = (self.current_yaw - self.start_turn_yaw + 180) % 360 - 180
        progressed_angle = abs(yaw_diff)
        
        target_angle_abs = abs(turn_angle)
        
        if progressed_angle >= (target_angle_abs - 10.0):
            self.get_logger().info(f"Kurve beendet (Gyro Hard-Exit): {progressed_angle:.1f}° erreicht.")
            return True
            
        if progressed_angle < (target_angle_abs - 25.0):
            return False
            
        if front_line_params is not None:
            n_x, n_y, d = front_line_params
            
            wall_angle_deg = math.degrees(math.atan2(n_y, n_x))
            
            wall_error_deg = min(abs(wall_angle_deg % 180), abs(180 - (wall_angle_deg % 180)))
            
            if wall_error_deg < 15.0:
                self.get_logger().info(f"Fused Match! Gyro bei {progressed_angle:.1f}°, Wand perfekt parallel (Error: {wall_error_deg:.1f}°).")
                return True

        return False

    # -----------------------------------------------
    # Testfunktionen
    # -----------------------------------------------

    def clear_all_lines(self):
        """
        Löscht alle Linien-Marker im Namespace 'walls'.
        """
        marker_array = MarkerArray()

        for i in range(25):
            self.delete_marker(marker_array, m_id=i, ns="walls")
        self.pub_markers.publish(marker_array)

    def visualize_cluster_line(self, cluster, m_id, farbe_name="rot", label="CLUSTER_LINE"):
        """
        Erstellt einen Marker für einen Cluster.
        """

        if cluster is None or len(cluster) < 2:
            return

        farben = {
            "rot": (1.0, 0.0, 0.0),
            "gruen": (0.0, 1.0, 0.0),
            "blau": (0.0, 0.5, 1.0),
            "cyan": (0.0, 1.0, 1.0),
            "magenta": (1.0, 0.0, 1.0),
            "gelb": (1.0, 1.0, 0.0)
        }
        
        rgb = farben.get(farbe_name.lower(), (1.0, 1.0, 1.0))
        
        start_p = (cluster[0][1], cluster[0][2])
        ende_p = (cluster[-1][1], cluster[-1][2])
        
        mitte_x = (start_p[0] + ende_p[0]) / 2.0
        mitte_y = (start_p[1] + ende_p[1]) / 2.0
        
        marker_array = MarkerArray()
        
        self.send_line(marker_array, m_id=m_id, p1=start_p, p2=ende_p, color=rgb)
        
        self.send_text(marker_array, m_id=m_id + 1000, text=label, x=mitte_x, y=mitte_y, color=rgb)
        
        self.pub_markers.publish(marker_array)

    def visualize_hnf_line(self, hnf_params, m_id, farbe_name="rot", label="HNF_LINE"):
        """
        Erstellt einen Marker für eine Gerade aus der Hesseschen Normalform (n_x, n_y, d).
        """
        if hnf_params is None:
            return

        farben = {
            "rot": (1.0, 0.0, 0.0),
            "gruen": (0.0, 1.0, 0.0),
            "blau": (0.0, 0.5, 1.0)
        }
        
        # Fallback auf Weiß, falls die Farbe nicht im Dictionary ist
        rgb = farben.get(farbe_name.lower(), (0.0, 0.0, 0.0))
        
        n_x, n_y, d = hnf_params
        
        # Lotpunkt berechnen
        p_lot_x = n_x * d
        p_lot_y = n_y * d
        
        # Richtungsvektor der Geraden
        v_x = -n_y
        v_y = n_x
        
        # Zwei Punkte für eine 6 Meter lange Linie
        p1 = (p_lot_x + v_x * 3.0, p_lot_y + v_y * 3.0)
        p2 = (p_lot_x - v_x * 3.0, p_lot_y - v_y * 3.0)
        
        marker_array = MarkerArray()
        
        # Linie zeichnen
        self.send_line(marker_array, m_id=m_id, p1=p1, p2=p2, color=rgb)
        self.send_text(marker_array, m_id=m_id + 1000, text=label, x=p_lot_x, y=p_lot_y, color=rgb)
        
        self.pub_markers.publish(marker_array)

    def test_extract_wall_lines(self, u_profile):
        '''Testet die Funktion extract_wall_lines() und visualisiert die Ergebnisse in RViz.'''
        front_straight, side_straight = self.extract_wall_lines(u_profile)
        self.get_logger().info(f"Front HNF: {front_straight}")
        self.get_logger().info(f"Side HNF: {side_straight}")
        self.visualize_hnf_line(front_straight, m_id=1, farbe_name="blau", label="")
        self.visualize_hnf_line(side_straight, m_id=0, farbe_name="blau", label="")
        return front_straight, side_straight

    def test_calculate_target_line(self, u_profile, desired_lane_ratio):
        '''Testet die Funktion calculate_target_line() und visualisiert die Ergebnisse in RViz.'''
        front_line_params, side_line_params = self.extract_wall_lines(u_profile)
        target_line_params, max_radius = self.calculate_target_line(side_line_params, front_line_params, desired_lane_ratio)
        radius = f"{max_radius:.2f}" if max_radius is not None else "None"
        self.visualize_hnf_line(target_line_params, m_id=3, farbe_name="gruen", label="")
        return target_line_params, max_radius

    def test_get_intersection_point(self, target_line_params):
        '''Testet die Funktion get_intersection_point() und visualisiert den Schnittpunkt in RViz.'''
        intersection_x, intersection_y, angle = self.get_intersection_point(target_line_params)
        if intersection_x is not None and intersection_y is not None:
            self.get_logger().info(f"Schnittpunkt: (X={intersection_x:.2f}, Y={intersection_y:.2f}), Schnittwinkel: {angle:.2f}°")
            marker_array = MarkerArray()
            self.send_sphere(marker_array, m_id=20, x=intersection_x, y=intersection_y, color=(1.0, 1.0, 0.0)) # Gelb
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
            self.send_sphere(marker_array, m_id=21, x=0.0, y=entry_point_distance_m, color=(0.0, 1.0, 0.0))
            
            if self.state == 'PARKING':
                richtung = self.park_turn_richtung
            else:
                richtung = self.fahrtrichtung
            is_left = (richtung == 'links')
            
            self.visualize_geogebra_angle(marker_array, intersection_y_m, turn_angle_deg, is_left_turn=is_left, m_id=20, radius=0.4)
            
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
        self.get_logger().info(f"Turn Trigger Check: {status} (Entry Point Distance: {entry_point_distance_m})")
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
        """Testet die Funktion check_turn_completion_fused() und loggt, ob die Kurve abgeschlossen ist."""
        self.visualize_hnf_line(front_line_params, m_id=1, farbe_name="rot", label="front_wall")
        completed = self.check_turn_completion_fused(target_angle, front_line_params)
        status = "COMPLETED" if completed else "IN PROGRESS"
        self.get_logger().info(f"Turn Completion Check: {status} (Current Gyro: {self.current_yaw:.2f}°, Target Angle: {target_angle:.2f}°)")
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
        Zeichnet den Schnittwinkel als Kreissektor mit Gradzahl.
        """

        if intersection_y_m is None or turn_angle_deg is None:
            self.delete_marker(marker_array, m_id)
            self.delete_marker(marker_array, m_id + 1)
            return

        # Bogen-Marker einrichten
        marker = Marker()
        marker.header.frame_id = "base_link" 
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "angle_arc"
        marker.id = m_id
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        marker.scale.x = 0.015 
        marker.color.r, marker.color.g, marker.color.b, marker.color.a = 0.59, 0.0, 1.0, 1.0

        # Winkel berechnen
        start_angle_rad = math.pi / 2.0  # y-Achse = 90 Grad
        turn_rad = math.radians(turn_angle_deg)
        
        if is_left_turn:
            end_angle_rad = start_angle_rad + turn_rad
        else:
            end_angle_rad = start_angle_rad - turn_rad

        # Bogen zeichnen
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

        mid_angle_rad = (start_angle_rad + end_angle_rad) / 2.0
        
        text_radius = radius * 0.6 
        text_x = text_radius * math.cos(mid_angle_rad)
        text_y = intersection_y_m + text_radius * math.sin(mid_angle_rad)
        
        self.send_text(marker_array, m_id=m_id + 1, text=f"{turn_angle_deg:.1f}°", x=text_x, y=text_y, color=(0.59, 0.0, 1.0))

    def handle_ausparken(self, point_data):
        if not self.auspark_sequenz_done:
            self.execute_blind_steps()
            self.auspark_sequenz_done = True
            return

        self.handle_last_parking_step(point_data)

    def execute_blind_steps(self):
        self.get_logger().info("Starte blinde Auspark-Sequenz (Schritte 0-9)...")

        steer_mult = -1.0 if self.fahrtrichtung == "links" else 1.0
        if self.fahrtrichtung == 'links':
            steps = [
                (0.0,   0.0, 1.0),  # Schritt 0: Warten
                (0.0,   0.8, 2.0),   # Schritt 1: Im Stand lenken
                (-225.0, 0.8, 0.65),  # Schritt 2: Rückwärts
                (0.0,  -0.8, 1.5),   # Schritt 3: Im Stand gegenlenken
                (195.0, -0.8, 0.60),   # Schritt 4: Vorwärts
                (0.0,   0.8, 1.5),   # Schritt 5: Im Stand lenkengi
                (-225.0, 1.4, 0.4),  # Schritt 6: Rückwärts
                (0.0,  -0.8, 1.5),   # Schritt 7: Im Stand gegenlenken
                (195.0, -0.8, 2.3),   # Schritt 8: Vorwärts
                (0.0,   0.8, 0.5),   # Schritt 9: Im Stand lenken
            ]

        else:
            steps = [
                (0.0,   0.0, 1.0),  # Schritt 0: Warten
                (0.0,   0.8, 2.0),   # Schritt 1: Im Stand lenken
                (-220.0, 0.8, 0.55),  # Schritt 2: Rückwärts
                (0.0,  -0.8, 1.5),   # Schritt 3: Im Stand gegenlenken
                (195.0, -0.8, 0.60),   # Schritt 4: Vorwärts
                (0.0,   0.8, 1.5),   # Schritt 5: Im Stand lenkengi
                (-220.0, 1.4, 0.35),  # Schritt 6: Rückwärts
                (0.0,  -0.8, 1.5),   # Schritt 7: Im Stand gegenlenken
                (195.0, -0.8, 2.3),   # Schritt 8: Vorwärts
                (0.0,   0.8, 0.5),   # Schritt 9: Im Stand lenken
            ]


        for i in range(len(steps)):
            drive_x, steer_z, duration = steps[i]
            
            cmd = Twist()
            cmd.linear.x = float(drive_x)
            cmd.angular.z = float(steer_z) * steer_mult
            
            start_time = time.monotonic()
            end_time = start_time + duration
            counter = 0
            
            while time.monotonic() < end_time:
                if counter % 20 == 0:
                    self.pub_cmd_vel.publish(cmd)
                
                counter += 1
                time.sleep(0.01)

            stop_cmd = Twist()
            stop_cmd.linear.x = 0.0
            stop_cmd.angular.z = 0.0
            self.pub_cmd_vel.publish(stop_cmd)

        self.get_logger().info("Blinde Sequenz beendet.")

    def handle_last_parking_step(self, point_data):
        cmd = Twist()
        if self.fahrtrichtung == "links":
            self.check_for_obstacle_color(point_data, (self.turn_count + 1), 0.25, 2.0)
            cmd.linear.x = 230.0
        else:
            cmd.linear.x = 230.0

        
        cmd.angular.z = 0.8 * (-1.0 if self.fahrtrichtung == "links" else 1.0)
        self.pub_cmd_vel.publish(cmd)
        
        duration = 1.6 if self.fahrtrichtung == "links" else 1.1

        self.get_logger().info("Letzten Ausparkschritt erreicht")

        if self.auspark_timer is None:
            self.auspark_timer = time.monotonic() + duration
        
        if time.monotonic() > self.auspark_timer:
            self.get_logger().info("Ausparken abgeschlossen.")
            self.state = 'FOLLOW_LANE'

    def debug_main_logic(self, point_data):
        if self.camera_calibration:
            return

        if self.debug_start:
            self.yaw_offset = self.current_yaw
            self.start_straight_yaw = self.current_yaw
            self.debug_start = False
            self.parking_phase = 'ADDRESSING_PARKING_SPACE'
            self.state = 'PARKING'

        self.parking_pid_steering(point_data)


    # -----------------------------------------
    # State Maschine Obstacle Run
    # -----------------------------------------

    def check_for_obstacle_color(self, point_data, requested_turn_count, min_distance_to_obstacle=0.0, max_distance_to_obstacle=2.0, front_wall_dist=None):
        current_straight = requested_turn_count % 4
        current_obstacle = self.obstacle_memory[current_straight]
        prediction = False
        
        if current_obstacle is None:
            detected_obstacles = self.get_obstacles_from_camera(point_data)
            
            if detected_obstacles:
                # Y-FILTER: Nur Hindernisse ab der Mindestdistanz
                detected_obstacles = list(filter(lambda x: x[1] > min_distance_to_obstacle, detected_obstacles))
                
                if requested_turn_count == self.turn_count:
                    detected_obstacles = list(filter(lambda x: abs(x[0]) <= 0.45, detected_obstacles))
                else:
                    prediction = True
                    if self.fahrtrichtung == 'links':
                        detected_obstacles = list(filter(lambda x: x[0] <= 0.15, detected_obstacles))
                    elif self.fahrtrichtung == 'rechts':
                        detected_obstacles = list(filter(lambda x: x[0] >= -0.15, detected_obstacles))

                # Nach euklidischer Distanz sortiert
                detected_obstacles.sort(key=lambda x: math.hypot(x[0], x[1]))
                
                if detected_obstacles:
                    closest_x, closest_y, closest_color = detected_obstacles[0]
                    closest_dist = math.hypot(closest_x, closest_y)
                    
                    if min_distance_to_obstacle < closest_dist < max_distance_to_obstacle:
                        new_obstacle = Obstacle(closest_color, None, prediction)
                        self.obstacle_memory[current_straight] = new_obstacle
                        self.get_logger().warn(f"+++ HINDERNIS GELOCKT: {closest_color.upper()}, {closest_dist:.2f} m auf Gerade: {current_straight}, Prediction: {prediction}, front_wall_dist: {front_wall_dist} m+++")
                        return closest_color, new_obstacle
                    else:
                        return None, None
                else:
                    return None, None
            else: 
                return None, None
        else:
            return current_obstacle.color, current_obstacle

    def calculate_zone_id(self, obst_to_front_wall_dist, obst_to_outer_wall_dist):
        zone_id = 0
        if obst_to_front_wall_dist >= 0.80 and obst_to_front_wall_dist <= 1.20:
            zone_id = 20
        elif obst_to_front_wall_dist >= 1.30 and obst_to_front_wall_dist <= 1.70:
            zone_id = 10
        elif obst_to_front_wall_dist >= 1.80 and obst_to_front_wall_dist <= 2.20:
            zone_id = 0
        else:
            self.get_logger().warn("Abstand zur Front_wall ist außerhalb des gültigen Bereichs.")
            return None
            

        if obst_to_outer_wall_dist >= 0.20 and obst_to_outer_wall_dist < 0.50:
            zone_id += 1
        elif obst_to_outer_wall_dist >= 0.50 and obst_to_outer_wall_dist <= 0.80:
            zone_id += 0
        else:
            self.get_logger().warn("Abstand zur Side_wall ist außerhalb des gültigen Bereichs.")
            zone_id = None

        return zone_id

    def set_lane_ratio_for_obstacle_cmd(self, obstacle_cmd, obstacle, front_wall_dist, is_turn_exit=False, apply_state=True):
        is_left = (self.fahrtrichtung == "links")
        max_shift = 0.005 
    
        actual_dist = front_wall_dist if front_wall_dist is not None else 2.0

        if obstacle_cmd is None and 1.40 < actual_dist < 1.60:
            if obstacle is not None:
                obstacle.prediction = True
                self.get_logger().info("Ghost-Trigger: Kein Obstacle bei 1.5m -> Prediction auf True gesetzt!")

        if is_turn_exit:
            target_ratio = self.standard_lane_ratio_exit
        else:
            target_ratio = self.standard_lane_ratio_approach
            
        is_evading = False  # True, wenn im Notfall ohne Glättung ausgewichen werden muss

        # HINDERNIS-LOGIK
        if obstacle_cmd is not None and obstacle is not None:
            if self.last_turn_for_parking or self.parking_straight:
                obst_is_green = obstacle.color == 'green'
                if (obst_is_green and is_left) or ((not obst_is_green) and (not is_left)):
                    target_ratio = 0.25
                else:
                    obst_is_relevant = False
                    if obstacle.is_localized:
                        obst_is_relevant = (obstacle.zone_id < 2) # Hindernis ist nur relevant wenn es direkt zu beginn der Geraden steht
                    else: 
                        if obstacle.prediction:
                            obst_is_relevant = True

                    if obst_is_relevant:
                        target_ratio = 0.65
                    else:
                        target_ratio = 0.25
                return target_ratio
            

            if not obstacle.is_localized:
                if obstacle.prediction and actual_dist < 1.65:
                    # Wir haben das vorhergesagte Hindernis passiert
                    target_ratio = self.standard_lane_ratio_approach
                else:
                    # Wir müssen ausweichen
                    is_evading = True
                    
                    if self.is_start_finish_straight:
                        if obstacle_cmd == "green":
                            target_ratio = 0.25 if is_left else 0.60
                        else:
                            target_ratio = 0.60 if is_left else 0.25
                    else:
                        if obstacle_cmd == "green":
                            target_ratio = 0.25 if is_left else 0.75
                        else:
                            target_ratio = 0.75 if is_left else 0.25
            else:
                obst_zone_id = obstacle.zone_id
                obst_passed = False
                
                if actual_dist is not None:
                    # Prüfen, ob wir am Hindernis vorbei sind
                    if obst_zone_id <= 1:
                        obst_passed = actual_dist < 1.75
                        obst_y = 2.0
                    elif obst_zone_id <= 11:
                        obst_passed = actual_dist < 1.35
                        obst_y = 1.5
                    else:
                        obst_passed = actual_dist < 0.85
                        obst_y = 1.0
                        
                    dist_to_obst = actual_dist - obst_y
                
                if not obst_passed:
                    # Wenn das Hindernis kritisch nah ist, Glättung abschalten
                    if dist_to_obst < 0.80:
                        is_evading = True

                    if is_turn_exit and obst_zone_id >= 20:
                        target_ratio = 0.50
                    elif actual_dist is None or dist_to_obst < 1.20:
                        obst_is_green = obstacle_cmd == "green"
                        obst_is_outer = obst_zone_id % 10 == 1
                        
                        if (obst_is_green and is_left) or ((not obst_is_green) and (not is_left)):
                            target_ratio = 0.35 if obst_is_outer else 0.22
                        else:
                            target_ratio = 0.80 if obst_is_outer else 0.65
                    else:
                        pass
        
        if not apply_state:
            return target_ratio
            
        if is_evading:
            # Sofortiges Umstellen ohne Glättung
            dist_to_inner_wall = target_ratio
        else:
            if actual_dist > 1.40:
                diff = target_ratio - self.lane_ratio
                if diff > max_shift:
                    dist_to_inner_wall = self.lane_ratio + max_shift
                elif diff < -max_shift:
                    dist_to_inner_wall = self.lane_ratio - max_shift
                else:
                    dist_to_inner_wall = target_ratio
            else:
                dist_to_inner_wall = self.lane_ratio

        return dist_to_inner_wall

    def set_obstacle_position(self, point_data, u_profile_hnf):
        """
        Verarbeitet Sensordaten, berechnet die exakte physische Position eines Hindernisses 
        und verortet es im topologischen Speicher des Roboters.
        """
        front_hnf, side_hnf = u_profile_hnf
        if front_hnf is None:
            return None
        
        nx_f, ny_f, dist_to_front = front_hnf
        
        color, current_obstacle = self.check_for_obstacle_color(point_data, self.turn_count, 0.0, dist_to_front - 0.85)
        
        if color is None and current_obstacle is None:
            self.get_logger().warn("Kein Hindernis erkannt.")
            return None
            
        elif current_obstacle is not None and current_obstacle.is_localized:
            return current_obstacle
            
        else:
            detected_obstacles = self.get_obstacles_from_camera(point_data)

            if detected_obstacles:
                # Sortieren nach euklidische Distanz zum LiDAR
                detected_obstacles.sort(key=lambda x: math.hypot(x[0], x[1]))
                closest_x, closest_y, closest_color = detected_obstacles[0]
                
                if side_hnf is None:
                    return current_obstacle
                    
                obst_to_front_wall_dist = abs(closest_x * nx_f + closest_y * ny_f - dist_to_front)
                
                if obst_to_front_wall_dist < 0.85:
                    self.get_logger().info(f"Obst zu nah an der front wall")
                    return current_obstacle
                
                obst_to_outer_wall_dist = self.get_obstacle_to_wall_distance(closest_x, closest_y, side_hnf)
                
                current_segment = self.turn_count % 4
                zone_id = self.calculate_zone_id(obst_to_front_wall_dist, obst_to_outer_wall_dist)
                
                if self.is_start_finish_straight and zone_id is not None:
                    if zone_id % 10 == 1:
                        zone_id -= 1  # Zwinge ID auf Innenbahn
                        self.get_logger().info(f"set_obst_pos: Zone auf Innenbahn korrigiert (Neue ID: {zone_id})")

                if zone_id is not None:
                    # Speichert Zone im Objekt
                    current_obstacle = Obstacle(closest_color, zone_id, False)
                    # Im Gedächtnis des Roboters speichern
                    self.obstacle_memory[current_segment] = current_obstacle
                    side_dist_str = f"{obst_to_outer_wall_dist:.2f}m" if obst_to_outer_wall_dist else "N/A"
                    self.get_logger().warn(
                        f"+++ HINDERNIS GELOCKT: {closest_color.upper()} +++\n"
                        f" -> Segment: {current_segment} | Zone: {zone_id}\n"
                        f" -> Zur Frontwand: {obst_to_front_wall_dist:.2f}m\n"
                        f" -> Zur Seitenwand: {side_dist_str}"
                    )
                return current_obstacle

    def get_obstacle_to_wall_distance(self, obstacle_x, obstacle_y, hnf_wall):
        """
        Berechnet den orthogonalen Abstand eines Punktes (Hindernis) zu einer Wand (HNF).
        """
        if hnf_wall is None:
            return None
            
        nx, ny, d = hnf_wall
        
        # HNF-Abstandsformel: D = |x*nx + y*ny - d|
        distance = abs(obstacle_x * nx + obstacle_y * ny - d)
        
        return distance
                
    def check_undetected_turn(self, front_wall_hnf):
        total_gedreht = abs(self.current_yaw - self.yaw_offset)
        min_total_rotation = self.target_turns * 89.0
        if self.turn_count >= self.target_turns and total_gedreht >= min_total_rotation:
            if front_wall_hnf is not None:
                closest_f = front_wall_hnf[2]
                if closest_f < 1.70:
                        self.state = 'PARKING'
                
        if abs(self.start_straight_yaw - self.current_yaw) > 75.0:
            self.get_logger().warn(f">>> GYRO KURVE ERKANNT! Zu weit auf der geraden gedreht (Gedreht: {abs(self.start_straight_yaw - self.current_yaw):.1f}°) <<<")
            self.start_straight_yaw = self.current_yaw
            self.turn_count += 1

    def evaluate_steering_straight(self, innenbande_hnf, aussenbande_hnf):
        if self.state == 'PARKING' and self.parking_phase == 'ADDRESSING_PARKING_SPACE':
            if self.park_direction == "PARKING_LEFT":
                target_x, target_y = self.parking_target_point(aussenbande_hnf, innenbande_hnf)
            else:
                target_x, target_y = self.parking_target_point(innenbande_hnf, aussenbande_hnf)
        else:
            target_x, target_y = self.get_target_point_straight(innenbande_hnf, aussenbande_hnf)
        self.visualize_target_point(target_x, target_y, farbe_name="orange", label="Target_Point")

        error = -target_x
        self.get_logger().info(f"SteeringError: {error:.2f}")
        
        # Integral berechnen
        self.integral_error += error
        self.integral_error = max(-1.0, min(1.0, self.integral_error))
        
        # Derivative berechnen
        derivative = error - self.prev_error
        self.prev_error = error
        
        # Stellgröße berechnen
        steering_cmd = (self.kp * error) + (self.ki * self.integral_error) + (self.kd * derivative)
        
        steering_cmd = max(-1.0, min(1.0, steering_cmd))

        return steering_cmd
    
    def execute_init(self):
        self.get_logger().info("Starte den Roboter...")
        with self.data_lock:
            if len(self.latest_yolo_results) == 0:
                self.get_logger().info("Warte auf ersten Kamera-Frame und YOLO-Inferenz...")
                return
        
        if not self.imu_ready:
            self.get_logger().info("Warte auf Gyroskop-Bootvorgang...")
            return

        self.set_led(True)
        self.state = 'STARTING'

    def execute_start(self, point_data):
        if self.button_start:
            if not self.button_state:
                return
            self.button_state = False
        else:
            pass

        self.button_state = False  # Flag sofort zurücksetzen
        self.set_led(False)        # LED ausschalten als Bestätigung
        
        self.yaw_offset = self.current_yaw
        self.start_straight_yaw = self.current_yaw

        self.get_logger().info("Evaluiere Fahrtrichtung und Parke aus")
            
        if self.mit_ausparken:
            if self.fahrtrichtung is None:
                dist_left_point = self.get_closest_measure(point_data, 270.0)
                dist_right_point = self.get_closest_measure(point_data, 90.0)

                if dist_right_point is not None and dist_left_point is not None:
                    dist_right = dist_right_point[3]
                    dist_left = dist_left_point[3]
                else: 
                    self.get_logger().info("Fehler: Mindestens eine Seite gibt keine Werte zurück")
                    return

                self.get_logger().info(f"Scanne Strecke... Abstand Links: {dist_left:.2f}m, Länge Rechts: {dist_right:.2f}m")
                if abs(dist_left - dist_right) > 0.50:
                    if dist_left < dist_right:        
                        self.fahrtrichtung = 'rechts' # Rechte Wand ist kürzer = Innenbande = Fahrtrichtung ist rechts!
                        self.get_logger().info(">>> LOCK: FAHRTRICHTUNG RECHTS (Uhrzeigersinn) <<<")
                    else:
                        self.fahrtrichtung = 'links'  # Linke Wand ist kürzer = Innenbande = Fahrtrichtung ist links!
                        self.get_logger().info(">>> LOCK: FAHRTRICHTUNG LINKS (Gegen den Uhrzeigersinn) <<<")
                else:
                    self.get_logger().info("Fehler! Keine Wand ist länger als die Andere!")
                    return
            self.state = 'PARKING_OUT'

        else:
            self.get_logger().info("Starte den Roboter... Evaluiere Fahrtrichtung und Kalibriere Gyro.")
            all_clusters = self.get_all_clusters_sorted(point_data)
            validated_clusters = self.validate_clusters_straight(all_clusters)
            merged_validated_clusters, _ = self.merge_clusters(all_clusters, validated_clusters)
            self.right_wall = merged_validated_clusters[0]
            self.front_wall = merged_validated_clusters[1]
            self.left_wall  = merged_validated_clusters[2]
            right_wall_hnf = self.cluster_to_hnf(self.right_wall)
            front_wall_hnf = self.cluster_to_hnf(self.front_wall)
            left_wall_hnf = self.cluster_to_hnf(self.left_wall)

            self.current_obstacle_cmd = self.check_for_obstacle_color(point_data, self.turn_count, 0.0, 0.8)

            if self.fahrtrichtung is None:
                if (self.left_wall is not None and len(self.left_wall) > 0) and (self.right_wall is not None and len(self.right_wall) > 0):
                        # Euklidische Länge berechnen
                        left_len = math.hypot(self.left_wall[0][1] - self.left_wall[-1][1], self.left_wall[0][2] - self.left_wall[-1][2])
                        right_len = math.hypot(self.right_wall[0][1] - self.right_wall[-1][1], self.right_wall[0][2] - self.right_wall[-1][2])
                        
                        self.get_logger().info(f"Scanne Strecke... Länge Links: {left_len:.2f}m, Länge Rechts: {right_len:.2f}m")
                        
                        # Braucht einen deutlichen Unterschied, um sicher zu sein
                        if left_len > right_len + 0.30:
                            self.fahrtrichtung = 'rechts' # Rechte Wand ist kürzer = Innenbande = Fahrtrichtung ist rechts!
                            self.get_logger().info(">>> LOCK: FAHRTRICHTUNG RECHTS (Uhrzeigersinn) <<<")
                        elif right_len > left_len + 0.30:
                            self.fahrtrichtung = 'links'  # Linke Wand ist kürzer = Innenbande = Fahrtrichtung ist links!
                            self.get_logger().info(">>> LOCK: FAHRTRICHTUNG LINKS (Gegen den Uhrzeigersinn) <<<")
                        else:    
                            self.get_logger().info("Fehler! Keine Wand ist länger als die")
                            return
                else:
                    self.get_logger().info("Fahrtrichtung noch nicht erkannt... Warte auf beide Seitenwände für die Analyse.")
                    return
            self.state = 'FOLLOW_LANE'

    def handle_lane_following(self, point_data):
        cmd = Twist()
        all_clusters = self.get_all_clusters_sorted(point_data)

        self.get_logger().info(f"Verfolge die Spur... Aktuelle Yaw: {self.current_yaw:.1f}°, Start-Yaw: {self.start_straight_yaw:.1f}°, Gedreht seit Start: {abs(self.current_yaw - self.start_straight_yaw):.1f}°")

        validated_clusters = self.validate_clusters_straight(all_clusters)
        merged_validated_clusters, _ = self.merge_clusters(all_clusters, validated_clusters)
        self.right_wall = merged_validated_clusters[0]
        self.front_wall = merged_validated_clusters[1]
        self.left_wall  = merged_validated_clusters[2]
        right_wall_hnf = self.cluster_to_hnf(self.right_wall)
        front_wall_hnf = self.cluster_to_hnf(self.front_wall)
        left_wall_hnf = self.cluster_to_hnf(self.left_wall)

        self.check_undetected_turn(front_wall_hnf)

        if self.park_direction == 'PARKING_LEFT' and front_wall_hnf is not None and front_wall_hnf[2] < 1.80:
            self.parking_phase = 'STOP_AFTER_OBST_RUN'
            return
        elif self.park_direction == 'PARKING_RIGHT_OBST' and front_wall_hnf is not None and front_wall_hnf[2] < 1.67:
            self.parking_phase = 'STOP_AFTER_OBST_RUN'

        self.visualize_hnf_line(front_wall_hnf, m_id=1, farbe_name="rot", label="Front HNF")
        self.visualize_hnf_line(left_wall_hnf, m_id=0, farbe_name="blau", label="Links HNF")
        self.visualize_hnf_line(right_wall_hnf, m_id=2, farbe_name="gruen", label="Rechts HNF")

        if self.fahrtrichtung == 'links':
            innenbande = self.left_wall
            innenbande_hnf = left_wall_hnf
            aussenbande = self.right_wall
            aussenbande_hnf = right_wall_hnf
        else:
            innenbande = self.right_wall
            innenbande_hnf = right_wall_hnf
            aussenbande = self.left_wall 
            aussenbande_hnf = left_wall_hnf

        current_straight = self.turn_count % 4
        current_obstacle = self.obstacle_memory[current_straight]

        if self.turn_count != 0 or self.fahrtrichtung == 'rechts':      # Damit wir keine falschen Hindernisse direkt nachdem Ausparken erkennen
            if current_obstacle is None or not current_obstacle.is_localized:
                current_obstacle = self.set_obstacle_position(point_data, (front_wall_hnf, aussenbande_hnf))

            if current_obstacle is None:
                if front_wall_hnf is not None:
                    self.current_obstacle_cmd, current_obstacle = self.check_for_obstacle_color(point_data, self.turn_count, 0.0, front_wall_hnf[2] - 0.75)
            else:
                self.current_obstacle_cmd = current_obstacle.color

        front_dist = front_wall_hnf[2] if front_wall_hnf is not None else None
        self.lane_ratio = self.set_lane_ratio_for_obstacle_cmd(self.current_obstacle_cmd, current_obstacle, front_dist)
        self.get_logger().info(f"Current_Obst_Cmd: {self.current_obstacle_cmd}, Current_Obst: {current_obstacle} , Lane_Ratio: {self.lane_ratio}, front_dist: {front_dist}m")

        if front_wall_hnf is not None and aussenbande_hnf is not None:
            self.check_for_obstacle_color(point_data, (self.turn_count + 1), front_wall_hnf[2] - 0.75, 2.0)

        if front_wall_hnf is not None and self.fahrtrichtung is not None:
            _, _, front_dist = front_wall_hnf
            
            if front_dist < 1.20:
                self.state = f"TURN_{self.fahrtrichtung.upper()}"
                self.get_logger().warn(f">>> {self.state} EINGELEITET<<<")
                self.get_logger().info(f"Abstand zur Frontwall: {front_dist:.2f}m")
                self.last_turn_aborted = False
                return
            else:
                if front_dist < 1.40:
                    self.get_logger().info(f"Warte auf Ecke... (Frontwand ist noch {front_dist:.2f}m entfernt)")
        
        steering_cmd = self.evaluate_steering_straight(innenbande_hnf, aussenbande_hnf)
        
        cmd.linear.x = self.base_speed
        cmd.angular.z = float(steering_cmd)
        self.pub_cmd_vel.publish(cmd)
        self.get_logger().info(f"Lenkung: Speed={self.base_speed:.3f}, Steering={steering_cmd:.3f}")
        self.pub_cmd_vel.publish(cmd)

    def handle_turn_maneuver(self, point_data):
        cmd = Twist()
        
        if self.turn_phase == 'APPROACH':
            self.front_wall = self.track_front_wall(point_data, self.front_wall)
            front_wall_hnf = self.cluster_to_hnf(self.front_wall)
            self.visualize_hnf_line(front_wall_hnf, m_id=1, farbe_name="rot", label="Front HNF")
            if front_wall_hnf is not None:
                allowed_obst_dist = max(0.25, front_wall_hnf[2] - 0.25)
            else:
                allowed_obst_dist = 0.25
            exit_obstacle_cmd, exit_obstacle = self.check_for_obstacle_color(point_data, (self.turn_count + 1), allowed_obst_dist, 2.0)
            self.lane_ratio_exit = self.set_lane_ratio_for_obstacle_cmd(exit_obstacle_cmd, exit_obstacle, 2.0, is_turn_exit=True, apply_state=False)
            self.get_logger().info(f"Geplanter Exit: Obst={exit_obstacle_cmd}, Ratio={self.lane_ratio_exit:.2f}")
            
            validated_clusters = self.validate_clusters_turn(self.front_wall, point_data)
            
            right_wall_hnf = self.cluster_to_hnf(validated_clusters[0])
            left_wall_hnf = self.cluster_to_hnf(validated_clusters[2])
            
            if self.fahrtrichtung == 'links':
                innenbande_hnf = left_wall_hnf
                aussenbande_hnf = right_wall_hnf
            else:
                innenbande_hnf = right_wall_hnf
                aussenbande_hnf = left_wall_hnf

            # Lenkung für Approach berechnen
            self.get_logger().info(f"InnenbandeHNF: {innenbande_hnf}, AussenbandeHNF: {aussenbande_hnf}")
            steering_cmd = self.evaluate_steering_straight(innenbande_hnf, aussenbande_hnf)
            
            # Kurvengeometrie berechnen
            front_wall_params, side_wall_params = self.extract_wall_lines(validated_clusters)
            target_line_params, max_allowed_radius = self.test_calculate_target_line(validated_clusters, self.lane_ratio_exit)
            intersection_x, intersection_y, intersection_angle = self.test_get_intersection_point(target_line_params)
            curve_radius_m, entry_distance_m = self.test_calculate_curve_geometry(intersection_y, intersection_angle, max_allowed_radius)
            
            # Trigger prüfen
            if self.test_check_turn_trigger(entry_distance_m):
                self.get_logger().info(f"Trigger erreicht. Wechsle in EXECUTE-Phase. Abstand zur Wand: {front_wall_hnf[2]:.2f}m, Radius: {curve_radius_m:.2f}m")
                self.saved_intersection_angle = intersection_angle
                self.saved_curve_radius_m = curve_radius_m
                self.start_turn_yaw = self.current_yaw
                self.base_obst_cmd = self.current_obstacle_cmd
                self.base_entry_distance = entry_distance_m

                cmd.linear.x = self.turn_speed
                cmd.angular.z = 0.0
                self.pub_cmd_vel.publish(cmd)
                
                self.turn_phase = 'EXECUTE'
                return
            
            # Lenken in der Annäherung
            cmd.linear.x = self.turn_speed
            cmd.angular.z = float(steering_cmd)
            self.pub_cmd_vel.publish(cmd)
                
        elif self.turn_phase == 'EXECUTE':          
            if self.current_obstacle_cmd != self.base_obst_cmd:
                if self.current_obstacle_cmd is not None:
                    is_left = (self.fahrtrichtung == "links")
                    needs_inner = (self.current_obstacle_cmd == "green" and is_left) or (self.current_obstacle_cmd == "red" and not is_left)
                    min_r = self.MIN_TURN_RADIUS_M
                    max_r = self.base_entry_distance - 0.15
                    
                    if needs_inner:
                        self.saved_curve_radius_m = min_r
                    else:
                        self.saved_curve_radius_m = min(max_r, self.MAX_KINEMATIC_RADIUS_M)
                    
                    self.get_logger().warn(f"MID-TURN AUSWEICHEN!. Radius: {self.saved_curve_radius_m:.2f}m")

            # Ausführen und Tracken
            self.execute_turn(self.saved_curve_radius_m)
            self.front_wall = self.track_front_wall(point_data, self.front_wall)
            
            if self.front_wall is not None:
                front_wall_params = self.cluster_to_hnf(self.front_wall)
            else:
                front_wall_params = None
            
            # Gyro oder Wand-Parallelität
            turn_completed = self.test_check_turn_completion_fused(self.saved_intersection_angle, front_wall_params)
            
            # Panic-Exit
            panic_exit = False
            
            yaw_diff = (self.current_yaw - self.start_turn_yaw + 180) % 360 - 180
            progressed_angle = abs(yaw_diff)
            
            if progressed_angle > 55.0:
                
                # Nur Hindernisse, die unmittelbar auf der neuen Gerade stehen (0.0 bis 0.7m)
                last_exit_ratio = self.lane_ratio_exit
                exit_obstacle_cmd, exit_obstacle = self.check_for_obstacle_color(point_data, (self.turn_count + 1), 0.0, 0.7)
                self.lane_ratio_exit = self.set_lane_ratio_for_obstacle_cmd(exit_obstacle_cmd, exit_obstacle, 2.0, is_turn_exit=True, apply_state=False)
                
                if exit_obstacle is not None and exit_obstacle_cmd != "CLEAR":
                    if abs(last_exit_ratio - self.lane_ratio_exit) > 0.15:
                        self.get_logger().warn(f"PANIC EXIT. Kritisches Hindernis ({exit_obstacle_cmd}) erzwingt Abbruch bei {progressed_angle:.1f}°!")
                        panic_exit = True
                        self.last_turn_aborted = True

            # Kurve beenden
            if turn_completed or panic_exit:
                if panic_exit:
                    self.get_logger().info("Notabbruch! Übergebe an Lane-Follower zur Kollisionsvermeidung.")
                else:
                    self.get_logger().info("Kurve regulär beendet. Übergebe an Lane-Follower.")
                
                cmd.linear.x = self.base_speed
                cmd.angular.z = 0.0
                self.pub_cmd_vel.publish(cmd)

                self.lane_ratio = self.lane_ratio_exit
                    
                self.turn_phase = 'APPROACH' # Reset für die nächste Kurve
                
                self.turn_count += 1

                if self.turn_count == self.target_turns:
                    self.state = 'PARKING'
                else:
                    self.state = 'FOLLOW_LANE'
                
                # PID - Werte für nächste Gerade nullen
                self.prev_error = 0.0
                self.integral_error = 0.0
                
                self.start_straight_yaw = self.current_yaw


    def update_timer(self):
        now = self.get_clock().now()

        if self.state != 'STARTING' and not self.timer_active:
            self.start_time_stamp = now
            self.timer_active = True
            self.get_logger().info("Timer gestartet!")

        elif self.state == 'STOPPED' and self.timer_active:
            self.timer_active = False
            self.get_logger().info(f"Timer gestoppt! Endzeit: {self.elapsed_time:.2f}s")

        if self.timer_active and self.start_time_stamp is not None:
            diff = now - self.start_time_stamp
            self.elapsed_time = diff.nanoseconds / 1e9
            
            timer_msg = Float64()
            timer_msg.data = self.elapsed_time
            self.pub_timer.publish(timer_msg)

    # =========================================================
    # Park Funktionen
    # =========================================================

    def observe_sorroundings_while_waiting(self, point_data):
        all_clusters = self.get_all_clusters_sorted(point_data)

        self.get_logger().info("Beobachte die Umgebung, während 3 Sekunden gewartet werden.")

        validated_clusters = self.validate_clusters_straight(all_clusters)
        merged_validated_clusters, _ = self.merge_clusters(all_clusters, validated_clusters)
        self.right_wall = merged_validated_clusters[0]
        self.front_wall = merged_validated_clusters[1]
        self.left_wall  = merged_validated_clusters[2]
        right_wall_hnf = self.cluster_to_hnf(self.right_wall)
        front_wall_hnf = self.cluster_to_hnf(self.front_wall)
        left_wall_hnf = self.cluster_to_hnf(self.left_wall)

        self.visualize_hnf_line(front_wall_hnf, m_id=1, farbe_name="rot", label="Front HNF")
        self.visualize_hnf_line(left_wall_hnf, m_id=0, farbe_name="blau", label="Links HNF")
        self.visualize_hnf_line(right_wall_hnf, m_id=2, farbe_name="gruen", label="Rechts HNF")

    def positioning_before_parking(self, point_data):
        cmd = Twist()
        all_clusters = self.get_all_clusters_sorted(point_data)

        self.get_logger().info(f"Positioniere mich fürs Einparken. Aktuelle Yaw: {self.current_yaw:.1f}°, Start-Yaw: {self.start_straight_yaw:.1f}°, Gedreht seit Start: {abs(self.current_yaw - self.start_straight_yaw):.1f}°")

        validated_clusters = self.validate_clusters_straight(all_clusters)
        merged_validated_clusters, _ = self.merge_clusters(all_clusters, validated_clusters)
        self.right_wall = merged_validated_clusters[0]
        self.front_wall = merged_validated_clusters[1]
        self.left_wall  = merged_validated_clusters[2]
        right_wall_hnf = self.cluster_to_hnf(self.right_wall)
        front_wall_hnf = self.cluster_to_hnf(self.front_wall)
        left_wall_hnf = self.cluster_to_hnf(self.left_wall)

        self.visualize_hnf_line(front_wall_hnf, m_id=1, farbe_name="rot", label="Front HNF")
        self.visualize_hnf_line(left_wall_hnf, m_id=0, farbe_name="blau", label="Links HNF")
        self.visualize_hnf_line(right_wall_hnf, m_id=2, farbe_name="gruen", label="Rechts HNF")

        if self.fahrtrichtung == 'links':
            innenbande = self.left_wall
            innenbande_hnf = left_wall_hnf
            aussenbande = self.right_wall
            aussenbande_hnf = right_wall_hnf
        else:
            innenbande = self.right_wall
            innenbande_hnf = right_wall_hnf
            aussenbande = self.left_wall 
            aussenbande_hnf = left_wall_hnf

        if front_wall_hnf is not None and self.fahrtrichtung is not None:
            _, _, front_dist = front_wall_hnf
            
            if front_dist < 1.50:
                self.parking_phase = 'TURN_PREPARATION'
                self.get_logger().warn(">>> PARK TURN_PREPARATION EINGELEITET<<<")
                self.get_logger().info(f"Abstand zur Frontwall: {front_dist:.2f}m")
            else:
                if front_dist < 1.60:
                    self.get_logger().info(f"Warte auf Ecke... (Frontwand ist noch {front_dist:.2f}m entfernt)")
        
        self.lane_ratio = 0.25
        steering_cmd = self.evaluate_steering_straight(innenbande_hnf, aussenbande_hnf)
        
        cmd.linear.x = self.parking_speed
        cmd.angular.z = float(steering_cmd)
        self.pub_cmd_vel.publish(cmd)
        self.get_logger().info(f"Lenkung: Speed={self.base_speed:.3f}, Steering={steering_cmd:.3f}")

    def handle_turn_preparation(self, point_data):
        if self.park_direction == 'PARKING_RIGHT_NORMAL':
            self.lane_ratio = 0.22
        else:
            self.lane_ratio = 0.25


        cmd = Twist()
        all_clusters = self.get_all_clusters_sorted(point_data)

        validated_clusters = self.validate_clusters_straight(all_clusters)
        merged_validated_clusters, _ = self.merge_clusters(all_clusters, validated_clusters)
        self.right_wall = merged_validated_clusters[0]
        self.front_wall = merged_validated_clusters[1]
        self.left_wall  = merged_validated_clusters[2]
        right_wall_hnf = self.cluster_to_hnf(self.right_wall)
        front_wall_hnf = self.cluster_to_hnf(self.front_wall)
        left_wall_hnf = self.cluster_to_hnf(self.left_wall)
        
        if self.fahrtrichtung == 'links':
            innenbande_hnf = left_wall_hnf
            aussenbande_hnf = right_wall_hnf
        else:
            innenbande_hnf = right_wall_hnf
            aussenbande_hnf = left_wall_hnf

        # Lenkung für Approach berechnen
        self.get_logger().info(f"InnenbandeHNF: {innenbande_hnf}, AussenbandeHNF: {aussenbande_hnf}")
        steering_cmd = self.evaluate_steering_straight(innenbande_hnf, aussenbande_hnf)
        
        # Kurvengeometrie berechnen
        front_wall_params, side_wall_params = self.extract_wall_lines(validated_clusters)
        if self.park_direction == 'PARKING_LEFT':
            target_line_params, max_allowed_radius = self.test_calculate_target_line(validated_clusters, -0.10)
        else:
            target_line_params, max_allowed_radius = self.test_calculate_target_line(validated_clusters, -0.875)
        intersection_x, intersection_y, intersection_angle = self.test_get_intersection_point(target_line_params)
        curve_radius_m, entry_distance_m = self.test_calculate_curve_geometry(intersection_y, intersection_angle, max_allowed_radius)
        
        # Trigger prüfen
        if self.test_check_turn_trigger(entry_distance_m):
            self.get_logger().info(f"Trigger erreicht. Wechsle in EXECUTE-Phase. Abstand zur Wand: {front_wall_hnf[2]:.2f}m, Radius: {curve_radius_m:.2f}m")
            self.saved_intersection_angle = intersection_angle
            self.saved_curve_radius_m = curve_radius_m
            self.start_turn_yaw = self.current_yaw
            self.base_obst_cmd = self.current_obstacle_cmd
            self.base_entry_distance = entry_distance_m

            cmd.linear.x = self.parking_speed
            cmd.angular.z = 0.0
            self.pub_cmd_vel.publish(cmd)
            
            self.parking_phase = 'EXECUTE_TURN'
            return
        
        # Lenken in der Annäherung
        cmd.linear.x = self.turn_speed
        cmd.angular.z = float(steering_cmd)
        self.pub_cmd_vel.publish(cmd)

    def handle_parking_turn(self, point_data):
        cmd = Twist()

        self.execute_turn(self.saved_curve_radius_m)
        self.front_wall = self.track_front_wall(point_data, self.front_wall)
        
        if self.front_wall is not None:
            front_wall_params = self.cluster_to_hnf(self.front_wall)
        else:
            front_wall_params = None
        turn_completed = self.test_check_turn_completion_fused(self.saved_intersection_angle, front_wall_params)


        if turn_completed:
            cmd.linear.x = self.parking_speed
            cmd.angular.z = 0.0
            self.pub_cmd_vel.publish(cmd)

            self.lane_ratio = self.lane_ratio_exit

            if self.park_direction == 'PARKING_RIGHT_NORMAL':
                self.parking_phase = 'STOP_AFTER_OBST_RUN'
            else:
                self.parking_phase = 'ADDRESSING_PARKING_SPACE'

            self.prev_error = 0.0
            self.integral_error = 0.0
            
            self.start_straight_yaw = self.current_yaw
            self.pub_cmd_vel.publish(cmd)

    def parking_pid_steering(self, point_data):
        self.lane_ratio = 1.13
        #self.parking_speed = 0.0
        self.kp = 1.4
        self.ki = 0.07
        self.kd = 0.08
        cmd = Twist()
        if not hasattr(self, 'einpark_timer'):
            self.einpark_timer = time.time() + 60.0
        if time.time() < self.einpark_timer:
            all_clusters = self.get_all_clusters_sorted(point_data)

            self.get_logger().info(f"Einparken in die Parklücke. Aktuelle Yaw: {self.current_yaw:.1f}°, Start-Yaw: {self.start_straight_yaw:.1f}°, Gedreht seit Start: {abs(self.current_yaw - self.start_straight_yaw):.1f}°")

            validated_clusters = self.validate_clusters_parking(all_clusters)
            merged_validated_clusters, _ = self.merge_clusters(all_clusters, validated_clusters)
            self.right_wall = merged_validated_clusters[0]
            self.front_wall = merged_validated_clusters[1]
            self.left_wall  = merged_validated_clusters[2]
            right_wall_hnf = self.cluster_to_hnf(self.right_wall)
            front_wall_hnf = self.cluster_to_hnf(self.front_wall)
            left_wall_hnf = self.cluster_to_hnf(self.left_wall)

            self.visualize_hnf_line(front_wall_hnf, m_id=1, farbe_name="rot", label="Front HNF")
            self.visualize_hnf_line(left_wall_hnf, m_id=0, farbe_name="blau", label="Links HNF")
            self.visualize_hnf_line(right_wall_hnf, m_id=2, farbe_name="gruen", label="Rechts HNF")

            if self.fahrtrichtung == 'links':
                innenbande = self.left_wall
                innenbande_hnf = left_wall_hnf
                aussenbande = self.right_wall
                aussenbande_hnf = right_wall_hnf
            else:
                innenbande = self.right_wall
                innenbande_hnf = right_wall_hnf
                aussenbande = self.left_wall 
                aussenbande_hnf = left_wall_hnf

            if front_wall_hnf is not None and self.fahrtrichtung is not None:
                _, _, front_dist = front_wall_hnf
            
            steering_cmd = self.evaluate_steering_straight(innenbande_hnf, aussenbande_hnf)
            
            # BEFEHLE AN ESP SETZEN
            cmd.linear.x = self.parking_speed
            cmd.angular.z = float(steering_cmd)
            self.pub_cmd_vel.publish(cmd)
            self.get_logger().info(f"Lenkung: Speed={self.base_speed:.3f}, Steering={steering_cmd:.3f}")
        else:
            self.execute_stop()
            self.get_logger().info("PARKEN BEENDET!")
            self.state = 'STOPPED'

    def parking_pid_steering_reverse(self, point_data):
        self.lane_ratio = 1.13
        cmd = Twist()

        self.get_logger().info(f"Einparken in die Parklücke. Aktuelle Yaw: {self.current_yaw:.1f}°, Start-Yaw: {self.start_straight_yaw:.1f}°, Gedreht seit Start: {abs(self.current_yaw - self.start_straight_yaw):.1f}°")
        
        all_clusters = self.get_all_clusters_sorted(point_data)
        validated_clusters = self.validate_clusters_parking(all_clusters)
        merged_validated_clusters, _ = self.merge_clusters(all_clusters, validated_clusters)
        self.right_wall = merged_validated_clusters[0]
        self.front_wall = merged_validated_clusters[1]
        self.left_wall  = merged_validated_clusters[2]
        right_wall_hnf = self.cluster_to_hnf(self.right_wall)
        front_wall_hnf = self.cluster_to_hnf(self.front_wall)
        left_wall_hnf = self.cluster_to_hnf(self.left_wall)

        _, _, front_dist = front_wall_hnf

        self.visualize_hnf_line(front_wall_hnf, m_id=1, farbe_name="rot", label="Front HNF")
        self.visualize_hnf_line(left_wall_hnf, m_id=0, farbe_name="blau", label="Links HNF")
        self.visualize_hnf_line(right_wall_hnf, m_id=2, farbe_name="gruen", label="Rechts HNF")

        if front_dist < 0.80:
            if self.fahrtrichtung == 'links':
                innenbande = self.left_wall
                innenbande_hnf = left_wall_hnf
                aussenbande = self.right_wall
                aussenbande_hnf = right_wall_hnf
            else:
                innenbande = self.right_wall
                innenbande_hnf = right_wall_hnf
                aussenbande = self.left_wall 
                aussenbande_hnf = left_wall_hnf

            if front_wall_hnf is not None and self.fahrtrichtung is not None:
                _, _, front_dist = front_wall_hnf
            
            steering_cmd = self.evaluate_steering_straight(innenbande_hnf, aussenbande_hnf)
            
            # BEFEHLE AN ESP SETZEN
            cmd.linear.x = - self.parking_speed
            cmd.angular.z = - float(steering_cmd)
            self.pub_cmd_vel.publish(cmd)
            self.get_logger().info(f"Lenkung: Speed={self.base_speed:.3f}, Steering={steering_cmd:.3f}")
        else:
            self.parking_phase = 'ADDRESSING_PARKING_SPACE'

    def parking_target_point(self, hnf_innen, hnf_aussen):
        self.get_logger().info(f"Lane_Ratio: {self.lane_ratio} m")
        target_y = self.lookahead_dist_parking
        
        # Plausibilitäts-Check: Toleranzen auf 0.30m senken, um Abbrüche bei leichtem Drift zu vermeiden
        if hnf_innen is not None and (0.30 > hnf_innen[2] or hnf_innen[2] > 3.5):
            hnf_innen = None
        if hnf_aussen is not None and (0.30 > hnf_aussen[2] or hnf_aussen[2] > 3.5):
            hnf_aussen = None

        def get_x_at_y(hnf_params, y_val):
            nx, ny, d = hnf_params
            if abs(nx) < 1e-6: return 0.0
            return (d - ny * y_val) / nx

        # 1. Wo schneiden die erkannten Wände unsere Y-Sichtachse?
        x_innen = get_x_at_y(hnf_innen, target_y) if hnf_innen else None
        x_aussen = get_x_at_y(hnf_aussen, target_y) if hnf_aussen else None

        # ========================================================
        # NEUE LOGIK: Orientierung zwingend an der durchgehenden Außenbande
        # ========================================================
        if x_aussen is not None:
            if x_aussen > 0: 
                # Außenbande ist rechts (Roboter parkt in die linke Lücke)
                target_x = x_aussen - self.lane_ratio
            else:            
                # Außenbande ist links (Roboter parkt in die rechte Lücke)
                target_x = x_aussen + self.lane_ratio
                
        elif x_innen is not None:
            # Nur Innenbande sichtbar: Stur geradeaus fahren
            target_x = 0.0
            
        else:
            # Keine Wand zu sehen -> stur geradeaus nach Gyro
            target_x = 0.0

        return (target_x, target_y)
    
    def validate_clusters_parking(self, clusters):
        u_profile = [None, None, None]
        if not clusters:
            return u_profile

        delta_yaw = 0.0
        if not self.last_turn_aborted:
            delta_yaw = self.current_yaw - self.start_straight_yaw
            
            # Normierung auf -180 bis 180
            while delta_yaw > 180: delta_yaw -= 360
            while delta_yaw < -180: delta_yaw += 360

        right_candidates = []
        left_candidates = []
        front_candidates = []

        for c in clusters:
            if len(c) < 20: continue
            
            local_angle = self.get_cluster_angle(c)
            if local_angle is None: continue

            shifted_angle = local_angle - delta_yaw
            
            angle_norm = abs(shifted_angle) % 180
            if angle_norm > 90:
                angle_norm = 180 - angle_norm

            if angle_norm <= 45.0:
                mean_x_local = sum(p[1] for p in c) / len(c)
                
                if abs(mean_x_local) <= 1.0:
                    if mean_x_local < 0:
                        left_candidates.append(c)
                    else:
                        right_candidates.append(c)
                        
            # FRONTWÄNDE (> 45°)
            else:
                # p[2] ist Vorne/Hinten
                mean_y = sum(p[2] for p in c) / len(c)
                if mean_y >= 0.15: # Nur Wände vor dem Roboter
                    front_candidates.append(c)

        # SEITENWÄNDE ZUORDNEN
        best_right = right_candidates[0] if right_candidates else None
        best_left = left_candidates[0] if left_candidates else None
        
        best_left_hnf = self.cluster_to_hnf(best_left) if (best_left is not None) else None
        best_right_hnf = self.cluster_to_hnf(best_right) if (best_right is not None) else None

        if (best_right is not None) and (best_left is not None):
            self.visualize_cluster_line(best_right, 10, "cyan")
            self.visualize_cluster_line(best_left, 11, "magenta")
            
            if best_left_hnf is not None and best_right_hnf is not None:
                _, _, left_dist = best_left_hnf
                _, _, right_dist = best_right_hnf
                
                track_width = abs(left_dist) + abs(right_dist)
                self.get_logger().info(f"Echte Spurbreite: {track_width:.2f}m, L: {abs(left_dist):.2f}m, R: {abs(right_dist):.2f}m")
                
                if 0.75 <= track_width <= 1.25:
                    u_profile[0] = best_right
                    u_profile[2] = best_left
                else:
                    self.get_logger().warn(f"Spurbreite unplausibel ({track_width:.2f}m). Verwerfe kürzere Wand.")
                    len_r = math.hypot(best_right[-1][1] - best_right[0][1], best_right[-1][2] - best_right[0][2])
                    len_l = math.hypot(best_left[-1][1] - best_left[0][1], best_left[-1][2] - best_left[0][2])
                    
                    if len_r > len_l:
                        u_profile[0] = best_right
                    else:
                        u_profile[2] = best_left
            else:
                self.get_logger().warn("HNF Berechnung für eine der Wände fehlgeschlagen.")
                return [None, None, None]
                    
        elif best_right is not None:
            u_profile[0] = best_right
        elif best_left is not None:
            u_profile[2] = best_left

        # FRONTWAND ZUORDNEN
        if front_candidates:
            groups = []
            for c in front_candidates:
                mean_y = sum(p[2] for p in c) / len(c)
                
                # Alles was extrem nah ist, verwerfen
                if mean_y < 0.20:
                    continue 
                
                placed = False
                for g in groups:
                    if abs(g['base_y'] - mean_y) < 0.15:
                        g['clusters'].append(c)
                        placed = True
                        break
                if not placed:
                    groups.append({'base_y': mean_y, 'clusters': [c]})

            valid_wall_groups = []
            for g in groups:
                all_x = [p[1] for c in g['clusters'] for p in c]
                total_width = max(all_x) - min(all_x)
                
                min_x = min(all_x)
                max_x = max(all_x)
                mean_y = g['base_y']  # Distanz zur Wand

                is_far_wall = mean_y > 0.70
                min_allowed_width = 0.45 if self.is_start_finish_straight else 0.35
                
                is_blocking_path = (min_x < -0.15) and (max_x > 0.15)
                
                if total_width > min_allowed_width and (is_far_wall or is_blocking_path):
                    if not is_far_wall and is_blocking_path:
                        self.get_logger().warn(f"Nahe aber blockierende Wand gefunden. Abstand: {mean_y:.2f}m")
                    valid_wall_groups.append(g)

            if valid_wall_groups:
                valid_wall_groups.sort(key=lambda g: g['base_y'])
                winner_group = valid_wall_groups[0]['clusters']
                winner_group.sort(key=len, reverse=True)
                u_profile[1] = winner_group[0]
            else:
                u_profile[1] = None
                if front_candidates:
                    self.get_logger().debug("Front-Kandidaten vorhanden, aber als Phantomwände abgelehnt.")

        return u_profile
    
    def einparken(self):
        if not hasattr(self, 'einpark_timer'):
            self.einpark_timer = time.time() + 3.0
            
        cmd = Twist()
        if time.time() < self.einpark_timer:
            cmd.linear.x = float(self.parking_speed)
            cmd.angular.z = 0.0
            self.pub_cmd_vel.publish(cmd)
        else:
            self.execute_stop()
            self.get_logger().info("PARKEN BEENDET!")
            self.state = 'STOPPED'

    def evaluate_reverse_turn(self, point_data):
        cmd = Twist()

        all_clusters = self.get_all_clusters_sorted(point_data)
        validated_clusters = self.validate_clusters_straight(all_clusters)
        merged_validated_clusters, _ = self.merge_clusters(all_clusters, validated_clusters)
        self.right_wall = merged_validated_clusters[0]
        self.front_wall = merged_validated_clusters[1]
        self.left_wall  = merged_validated_clusters[2]
        right_wall_hnf = self.cluster_to_hnf(self.right_wall)
        front_wall_hnf = self.cluster_to_hnf(self.front_wall)
        left_wall_hnf = self.cluster_to_hnf(self.left_wall)
        front_wall_hnf = self.cluster_to_hnf(self.front_wall)
        
        if self.fahrtrichtung == 'links':
            innenbande_hnf = left_wall_hnf
            aussenbande_hnf = right_wall_hnf
        else:
            innenbande_hnf = right_wall_hnf
            aussenbande_hnf = left_wall_hnf

        self.get_logger().info(f"InnenbandeHNF: {innenbande_hnf}, AussenbandeHNF: {aussenbande_hnf}")

        n_xf, n_yf, d_f = front_wall_hnf
        d_ziel = 0.20
        target_line_params = (n_xf, n_yf, d_ziel)

        _, _, intersection_angle = self.test_get_intersection_point(target_line_params)

        curve_radius_m = 0.20
        entry_distance_m = 0.0

        self.get_logger().info(f"Warte auf das Ablaufen der 3s um dann rückwärts abzubiegen")
        self.saved_intersection_angle = intersection_angle
        self.saved_curve_radius_m = curve_radius_m
        self.start_turn_yaw = self.current_yaw
        self.base_entry_distance = entry_distance_m
        
        cmd.linear.x = 0.0
        cmd.angular.z = 0.0
        self.pub_cmd_vel.publish(cmd)
    
    def decide_park_direction(self, point_data):
        if self.park_direction is None:
            if self.fahrtrichtung == 'links':
                self.park_direction = 'PARKING_LEFT'
                self.parking_phase = 'POSITIONING_FOR_STOP'
                self.park_turn_richtung = 'rechts'

            elif self.obstacle_memory[0] is not None and (self.obstacle_memory[0].prediction or self.obstacle_memory[0].zone_id < 2) and self.obstacle_memory[0].color == 'green':
                self.park_direction = 'PARKING_RIGHT_OBST'
                self.parking_phase = 'POSITIONING_FOR_STOP'
                self.park_turn_richtung = 'links'

            else:
                self.park_direction = 'PARKING_RIGHT_NORMAL'
                self.parking_phase = 'APPROACH_TURN'
                self.park_turn_richtung = 'links'

        if self.park_direction == 'PARKING_LEFT':
            self.handle_park_maneuver_left(point_data)

        elif self.park_direction == 'PARKING_RIGHT_NORMAL':
            self.handle_park_maneuver_right_normal(point_data)
            
        elif self.park_direction == 'PARKING_RIGHT_OBST':
            self.handle_park_maneuver_right_obst(point_data)

    def handle_park_maneuver_left(self, point_data):
        if self.parking_phase == 'POSITIONING_FOR_STOP':
            self.handle_lane_following(point_data)

        elif self.parking_phase == 'STOP_AFTER_OBST_RUN':
            if self.waiting_timer is None:
                self.waiting_timer = time.time() + 3.5
            if time.time() < self.waiting_timer:
                cmd = Twist()
                cmd.linear.x = 0.0
                cmd.angular.z = 0.0
                self.pub_cmd_vel.publish(cmd)
                self.observe_sorroundings_while_waiting(point_data)
                pass
            else:
                self.parking_phase = 'POSITIONING_BEFORE_PARKING'
                self.waiting_timer = None

        elif self.parking_phase == 'POSITIONING_BEFORE_PARKING':
            self.positioning_before_parking(point_data)

        elif self.parking_phase == 'TURN_PREPARATION':
            self.handle_turn_preparation(point_data)

        elif self.parking_phase == 'EXECUTE_TURN':
            self.handle_parking_turn(point_data)
            
        elif self.parking_phase == 'ADDRESSING_PARKING_SPACE':
            self.parking_pid_steering(point_data)

    def handle_park_maneuver_right_normal(self, point_data):
        if self.parking_phase == 'APPROACH_TURN':
            self.handle_turn_preparation(point_data)

        elif self.parking_phase == 'EXECUTE_TURN':
            self.handle_parking_turn(point_data)

        elif self.parking_phase == 'STOP_AFTER_OBST_RUN':
            if self.waiting_timer is None:
                self.waiting_timer = time.time() + 3.5
            if time.time() < self.waiting_timer:
                cmd = Twist()
                cmd.linear.x = 0.0
                cmd.angular.z = 0.0
                self.pub_cmd_vel.publish(cmd)
                self.observe_sorroundings_while_waiting(point_data)
            else: 
                self.waiting_timer = None
                self.parking_phase = 'ADDRESSING_PARKING_SPACE'

        elif self.parking_phase == 'BACKING_UP':
            self.parking_pid_steering_reverse(point_data)
        
        elif self.parking_phase == 'ADDRESSING_PARKING_SPACE':
            self.parking_pid_steering(point_data)

    def handle_park_maneuver_right_obst(self, point_data):
        if self.parking_phase == 'POSITIONING_FOR_STOP':
            self.handle_lane_following(point_data)

        elif self.parking_phase == 'STOP_AFTER_OBST_RUN':
            if self.waiting_timer is None:
                self.waiting_timer = time.time() + 3.5
            if time.time() < self.waiting_timer:
                cmd = Twist()
                cmd.linear.x = 0.0
                cmd.angular.z = 0.0
                self.pub_cmd_vel.publish(cmd)
                self.evaluate_reverse_turn(point_data)
            else:
                self.waiting_timer = None
                self.parking_phase = 'EXECUTE_TURN'
                cmd = Twist()
                cmd.linear.x = 0.0
                cmd.angular.z = - 0.8
                self.pub_cmd_vel.publish(cmd)
        
        elif self.parking_phase == 'EXECUTE_TURN':
            self.handle_parking_turn(point_data)

        elif self.parking_phase == 'ADDRESSING_PARKING_SPACE':
            self.parking_pid_steering(point_data)

    def execute_stop(self):
            cmd = Twist()
            cmd.linear.x = 0.0
            cmd.angular.z = 0.0
            self.pub_cmd_vel.publish(cmd)
            self.get_logger().warn(f">>> ZIEL ERREICHT: {self.turn_count} Kurven geschafft! Stoppe den Roboter. <<<")
            self.get_logger().warn(f">>> Erkannte Obst {self.obstacle_memory}")
            self.destroy_node()
            rclpy.shutdown()
            return

    def main_logic(self, point_data):
        if self.counter == 15:
            self.counter = 0
            self.clear_all_lines()
        self.get_logger().info(f"Aktueller Status: {self.state}, Aktuelle Parking_Phase: {self.parking_phase}, TurnCount: {self.turn_count}, Lane_Ratio: {self.lane_ratio}, EXIT_Ratio: {self.lane_ratio_exit}")

        if self.state == 'INITIALIZING':
            self.execute_init()
        
        elif self.state == 'STARTING':
            self.execute_start(point_data)

        elif self.state == 'PARKING_OUT':
            self.handle_ausparken(point_data)

        elif self.state == 'FOLLOW_LANE':
            self.handle_lane_following(point_data)
            
        elif self.state in ['TURN_LINKS', 'TURN_RECHTS']:
            self.handle_turn_maneuver(point_data)
            
        elif self.state == 'STOPPED':
            self.execute_stop()

        elif self.state == 'PARKING':
            self.decide_park_direction(point_data)

        self.counter += 1
        self.update_timer()
        
def main(args=None):
    rclpy.init(args=args)
    node = Obstacle_Run()
    
    # Verwaltet die parallelen Callback-Gruppen
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)
    
    try:
        # Startet alle Threads parallel
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()