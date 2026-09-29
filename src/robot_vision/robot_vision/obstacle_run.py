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
     HARDWARE COORDINATE SYSTEM (Lidar & Foxglove)
=============================================================

                          +Y (FRONT)
                              ^
                              |  Lidar: 180 deg
                              |  (driving direction)
                              |
                              |
    (LEFT)                    |                   (RIGHT)
      -X  <-------------- [ ROBOT ] -------------->  +X
    Lidar: 270 deg            |                   Lidar: 90 deg
                              |
                              |
                              |
                              v
                          -Y (REAR)
                       Lidar: 360 deg / 0 deg

-------------------------------------------------------------
Zone Ids:

20  |   21     |
|   |   |      |
10  |   11     | <-- outer_wall
|   |   |      |
00  |   01     |
    ^
    |
 ROBOT
============================================================='''

class Obstacle_Run(Node):
    def __init__(self):
        super().__init__('obstacle_run')

        # MULTITHREADING SETUP
        self.sensor_cbg = MutuallyExclusiveCallbackGroup()
        self.yolo_cbg = MutuallyExclusiveCallbackGroup()
        
        # Thread lock for data safety when accessing shared variables
        self.data_lock = threading.Lock()
        self.latest_yolo_results = []
        
        
        
        self.scan_sub = self.create_subscription(
            LaserScan,
            '/ldlidar_node/scan',
            self.scan_callback,
            qos_profile_sensor_data,
            callback_group=self.sensor_cbg  # runs in its own thread
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
        self.with_unpark = True

        self.led_pub = self.create_publisher(Bool, '/led_cmd', 10)

        self.button_state = False
        
        # Publisher for RViz
        self.pub_cmd_vel = self.create_publisher(Twist, '/cmd_vel', 10)
        self.pub_markers = self.create_publisher(MarkerArray, '/wall_follower_markers', 10)

        # IMU variables
        self.yaw_offset = 0.0
        self.current_yaw = 0.0
        self.imu_ready = False
        self.last_raw_yaw = None
        self.start_turn_yaw = None
        self.start_straight_yaw = 0.0
        self.panic_phase = None
        self.panic_timer = None
        self.panic_straight_yaw_est = None
        self.panic_increment_turn = False
        self.panic_close_obstacle = False
        
        
        self.rviz_frame = 'ldlidar_link'
        self.get_logger().info('>>> WallFollower template started. Waiting for LiDAR... <<<')

        # STATE MACHINE initial states
        self.state = 'INITIALIZING'
        self.turn_phase = 'APPROACH'
        self.parking_phase = None
        self.park_direction = None
        self.park_turn_direction = None

        self.direction = None
        self.saved_intersection_angle = None
        self.saved_curve_radius_m = None

        self.target_turns = 12
        self.turn_count = 0
        self.is_start_finish_straight = False
        self.last_turn_for_parking = False
        self.parking_straight = False
        self.parking_left_with_obstacle = False

        self.front_wall = None
        self.left_wall = None
        self.right_wall = None
        
        self.standard_lane_ratio_approach = 0.60
        self.standard_lane_ratio_exit = 0.50
        self.lane_ratio = self.standard_lane_ratio_approach       # ratio of the wall distances inner to outer
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

        # PID CONTROLLER parameters
        self.kp = 2.0
        self.ki = 0.0
        self.kd = 0.05

        self.kp_gyro = 0.07
        self.ki_gyro = 0.0
        self.kd_gyro = 0.0
        self.prev_gyro_error = 0.0
        self.integral_gyro_error = 0.0

        self.lookahead_dist_straight = 0.20
        self.lookahead_dist_parking = 0.40
        self.min_wall_dist = 0.15
        
        self.prev_error = 0.0
        self.integral_error = 0.0

        self.unpark_step = 0
        self.unpark_timer = None
        self.unpark_sequence_done = False

        self.steering_ctrl = SteeringController(logger=self.get_logger())

        self.waiting_timer = None

        self.TRACK_WIDTH_M = 1.0
        self.ROBOT_WIDTH_M = 0.15
        self.LIDAR_OFFSET_M = 0.08
        self.SAFETY_MARGIN_M = 0.05
        self.MAX_KINEMATIC_RADIUS_M = 1.2
        self.IDEAL_RADIUS_M = 0.28
        self.MIN_TURN_RADIUS_M = 0.20

        self.turn_exit_tolerance = 10.0

        # MOTOR Parameter
        self.base_speed = 350.0
        self.turn_speed = 350.0
        self.parking_speed = 250.0
        self.panic_speed = 250.0
        self.panic_stop_duration = 1.0
        self.panic_reverse_duration = 1.5
        self.panic_obstacle_dist = 0.25

        # Global speeds
        self.SPEED_STRAIGHT_SLOW = 430.0
        self.SPEED_STRAIGHT_MED = 750.0
        self.SPEED_STRAIGHT_FAST = 900.0
        self.SPEED_TURN_SLOW = 350.0
        self.SPEED_TURN_MED = 450.0
        self.SPEED_TURN_FAST = 750.0
        self.SPEED_STRAIGHT_SLOW_saved = self.SPEED_STRAIGHT_SLOW
        
        self.is_obstacle_passed = False

        # YOLO Parameter
        self.image_width = 1280
        self.bridge = CvBridge()

        self.target_yolo_fps = 8.0 
        self.last_yolo_time = self.get_clock().now().nanoseconds / 1e9

        self.analyzer = TrackAnalyzer(
            logger=self.get_logger(),
            visualizer_cb=self.visualize_cluster_line
        )
        
        self.get_logger().info('Loading YOLO TensorRT engine')
        self.model = YOLO('/workspace/best.engine', task='detect')
        self.get_logger().info('Model loaded successfully')

        self.get_logger().info('Starting TensorRT warm-up')
        dummy_cv_image = np.zeros((360, 640, 3), dtype=np.uint8) 
        dummy_cv_image = np.ascontiguousarray(dummy_cv_image)
        self.model.predict(
            dummy_cv_image, 
            half=True, 
            imgsz=640, 
            device=0, 
            verbose=False
        )
        self.get_logger().info('Warm-up finished. GPU is ready.')


        self.angle_calibration = 0.0
        self.lidar_height_offset = 0.05
        self.camera_to_lidar_dist = 0.03

        
        # Marker Publisher
        self.marker_pub = self.create_publisher(Marker, '/detected_obstacles', 10)
        self.pub_debug_img = self.create_publisher(Image, '/camera/yolo_debug', 10)
        
        self.avoid_trigger_dist = 0.85
        
        # Variables for the camera/lidar fusion
        self.camera_fov = 115.0
        self.get_logger().info('YOLO lidar fusion node started.')

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
        self.debug = False
        self.debug_start = self.debug

    def send_line(self, marker_array, m_id, p1, p2, color=(1.0, 1.0, 1.0)):
        """Helper to create a line for the MarkerArray."""
        marker = Marker()
        marker.header.frame_id = self.rviz_frame
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "walls"
        marker.id = m_id
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        marker.scale.x = 0.03  # line thickness
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

    def is_narrow_lane(self, color, id):
        is_left = (self.direction == 'left')
        
        # inner lane
        if is_left and color == 'green' or not is_left and color == 'red':
            if id % 10 == 0:
                return True
            else:
                return False            

        # outer lane
        if is_left and color == 'red' or not is_left and color == 'green':
            if id % 10 == 1:
                return True
            else:
                return False

    def takes_inside_lane(self, color):
        is_left_turn = (self.direction == 'left')
        
        # Left turn: green = pass on the left = inner lane
        if is_left_turn and color == 'green': 
            return True
        # Right turn: red = pass on the right = inner lane
        if not is_left_turn and color == 'red': 
            return True
            
        return False

    def update_strategy_params(self):
        if self.turn_count > 4:
            self.IDEAL_RADIUS_M = 0.28
        else:
            if self.turn_count == 0 and self.direction == 'right':
                self.SPEED_STRAIGHT_SLOW = 280.0
            else:
                self.SPEED_STRAIGHT_SLOW = self.SPEED_STRAIGHT_SLOW_saved
            self.IDEAL_RADIUS_M = 0.20
        
        if self.turn_count % 4 == 0 and self.state == 'FOLLOW_LANE' or self.turn_count % 4 == 3 and self.state in ['TURN_LEFT', 'TURN_RIGHT']:
            self.is_start_finish_straight = True
            self.standard_lane_ratio_approach = 0.60
        else:
            self.is_start_finish_straight = False
            self.standard_lane_ratio_approach = 0.70

        if self.turn_count % 4 == 3 and self.state in ['TURN_LEFT', 'TURN_RIGHT']:
            self.standard_lane_ratio_exit = 0.35
        else:
            self.standard_lane_ratio_exit = 0.45

        # variable speed adjustments
        current_straight = self.turn_count % 4
        next_straight = (self.turn_count + 1) % 4
        
        obst_current = self.obstacle_memory[current_straight]
        obst_next = self.obstacle_memory[next_straight]

        if self.turn_count < 4 or (self.turn_count == 4 and self.state == 'FOLLOW_LANE'):
            # Straights
            if obst_current is not None and obst_next is not None and self.is_obstacle_passed:
                self.base_speed = self.SPEED_STRAIGHT_MED
            else:
                self.base_speed = self.SPEED_STRAIGHT_SLOW
                
            # Turns
            self.turn_speed = self.SPEED_TURN_SLOW # placeholder for lap 1

        else:
            # STRAIGHTS:
            if self.is_obstacle_passed:
                # we are past the obstacle
                self.base_speed = self.SPEED_STRAIGHT_FAST

            elif obst_current is not None and obst_current.is_localized:
                # obstacle detected
                if self.is_narrow_lane(obst_current.color, obst_current.zone_id):
                    self.base_speed = self.SPEED_STRAIGHT_MED
                else:
                    if self.is_start_finish_straight:
                        self.base_speed = self.SPEED_STRAIGHT_MED
                    else:
                        self.base_speed = self.SPEED_STRAIGHT_FAST

            elif obst_current is not None and not obst_current.is_localized and obst_current.prediction:
                # obstacle in predict state
                self.base_speed = self.SPEED_STRAIGHT_MED

            else:
                # no obstacles
                self.base_speed = self.SPEED_STRAIGHT_SLOW
                
            # TURNS:
            must_drive_slow = False
            
            if obst_next is not None and obst_next.is_localized:
                if obst_next.zone_id < 2 and self.takes_inside_lane(obst_next.color):
                    must_drive_slow = True

            elif obst_next is not None and not obst_next.is_localized:
                must_drive_slow = True

            if obst_current is not None and obst_current.is_localized:
                if obst_current.zone_id >= 20 and self.takes_inside_lane(obst_current.color):
                    must_drive_slow = True
                    
            if must_drive_slow:
                self.turn_speed = self.SPEED_TURN_MED
            else:
                self.turn_speed = self.SPEED_TURN_FAST

        if self.turn_count == self.target_turns - 1 and self.state == 'FOLLOW_LANE':
            if self.base_speed > self.SPEED_STRAIGHT_MED:
                self.base_speed = self.SPEED_STRAIGHT_MED

        if (self.target_turns - self.turn_count) == 1 and self.state in ['TURN_LEFT', 'TURN_RIGHT']:
            self.last_turn_for_parking = True
            self.turn_speed = self.parking_speed
            self.base_speed = self.parking_speed
        else: 
            self.last_turn_for_parking = False

        if self.target_turns == self.turn_count:
            self.parking_straight = True
            self.IDEAL_RADIUS_M = 0.20
            self.base_speed = self.parking_speed
            self.turn_speed = self.parking_speed
        else:
            self.parking_straight = False

        if self.base_speed == self.SPEED_STRAIGHT_SLOW:
            self.kp = 2.5
            self.ki = 0.0
            self.kd = 0.08

        elif self.base_speed == self.SPEED_STRAIGHT_MED:
            self.kp = 2.5
            self.ki = 0.0
            self.kd = 0.11

        elif self.base_speed == self.SPEED_STRAIGHT_FAST:
            self.kp = 2.3
            self.ki = 0.0
            self.kd = 0.15
            
        elif self.base_speed == self.parking_speed:
            self.kp = 2.5
            self.ki = 0.0
            self.kd = 0.08

        if self.turn_speed == self.SPEED_TURN_SLOW:
            self.turn_exit_tolerance = 10.0

        elif self.turn_speed == self.SPEED_TURN_MED:
            self.turn_exit_tolerance = 13.0

        elif self.turn_speed == self.SPEED_TURN_FAST:
            self.turn_exit_tolerance = 16.0

        elif self.turn_speed == self.parking_speed:
            self.turn_exit_tolerance = 8.0

        self.get_logger().warn(f"BaseSpeed: {self.base_speed} , TurnSpeed: {self.turn_speed}")


    def imu_callback(self, msg):
        """
        Returns the results of the IMU sensor.
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
            self.get_logger().info("Hardware interrupt received. Trigger fired.")
            self.button_state = True

    def test_log_bbox_y_values(self, bounding_boxes):
        """
        Test function for the folding-rule calibration of the camera.
        Prints the Y pixel values of the detected bounding boxes in the terminal.
        """
        if bounding_boxes is None:
            return

        self.get_logger().info(f"--- Starting Y value measurement ({len(bounding_boxes)} object(s)) ---")
        
        for i, bbox in enumerate(bounding_boxes):
            try:
                y_min = int(bbox[1]) # top edge of the object
                y_max = int(bbox[3]) # bottom edge of the object
                
                self.get_logger().info(f"Object {i+1} -> top edge: {y_min} px | (bottom edge/floor): y_max = {y_max} px")

            except IndexError:
                self.get_logger().error(f"Error: bounding box has an unexpected format: {bbox}")

    def send_text(self, marker_array, m_id, text, x, y, color=(1.0, 1.0, 1.0), scale=0.15):
        """
        Adds a text marker to the MarkerArray.
        Font size in metres
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
        
        # colour and visibility
        marker.color.r = float(color[0])
        marker.color.g = float(color[1])
        marker.color.b = float(color[2])
        marker.color.a = 1.0
        
        marker.text = str(text)
        
        marker_array.markers.append(marker)

    def send_sphere(self, marker_array, m_id, x, y, color=(0.0, 1.0, 1.0)):
        """Draws a point in RViz."""
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

    def validate_clusters_far_front_wall(self, clusters):
        u_profile = [None, None, None]
        if not clusters:
            return u_profile

        delta_yaw = self.current_yaw - self.start_straight_yaw

        # normalise to -180 to 180
        while delta_yaw > 180: delta_yaw -= 360
        while delta_yaw < -180: delta_yaw += 360

        right_candidates = []
        left_candidates = []
        front_candidates = []

        for c in clusters:
            if len(c) < 12:
                #self.get_logger().info(f"Cluster has too few points")
                continue
            
            local_angle = self.get_cluster_angle(c)
            if local_angle is None: 
                continue

            shifted_angle = local_angle - delta_yaw
            
            angle_norm = abs(shifted_angle) % 180
            if angle_norm > 90:
                angle_norm = 180 - angle_norm

            mean_x_local = sum(p[1] for p in c) / len(c)
            mean_y_local = sum(p[2] for p in c) / len(c)
            
            rad = math.radians(delta_yaw)
            mean_x_straight = mean_x_local * math.cos(rad) - mean_y_local * math.sin(rad)
            mean_y_straight = mean_x_local * math.sin(rad) + mean_y_local * math.cos(rad)

            if angle_norm <= 45.0:
                if abs(mean_x_straight) <= 1.0:
                    if mean_x_straight > 0:
                        right_candidates.append(c)
                    else:
                        left_candidates.append(c)
                else:
                    self.get_logger().info("Cluster has too large an x value")

            else:  
                if mean_y_straight >= 0.15:
                    front_candidates.append(c)

        # ASSIGN SIDE WALLS
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
                self.get_logger().info(f"Real track width: {track_width:.2f}m, L: {abs(left_dist):.2f}m, R: {abs(right_dist):.2f}m")
                
                if 0.75 <= track_width <= 1.25:
                    u_profile[0] = best_right
                    u_profile[2] = best_left
                else:
                    self.get_logger().warn(f"Track width implausible ({track_width:.2f}m). Dropping the shorter wall.")
                    len_r = math.hypot(best_right[-1][1] - best_right[0][1], best_right[-1][2] - best_right[0][2])
                    len_l = math.hypot(best_left[-1][1] - best_left[0][1], best_left[-1][2] - best_left[0][2])
                    
                    if len_r > len_l:
                        u_profile[0] = best_right
                    else:
                        u_profile[2] = best_left
            else:
                self.get_logger().warn("HNF calculation failed for one of the walls.")
                return [None, None, None]
                    
        elif best_right is not None:
            u_profile[0] = best_right
        elif best_left is not None:
            u_profile[2] = best_left

        # ASSIGN FRONT WALL
        if front_candidates:
            self.get_logger().warn(f"Number of FrontCandidates: {len(front_candidates)}")
            groups = []
            for c in front_candidates:
                mean_y = sum(p[2] for p in c) / len(c)
                self.visualize_cluster_line(c, 20,'blue')
                # drop everything that is extremely close
                if mean_y < 1.40:
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
                mean_y = g['base_y']  # distance to the wall

                is_far_wall = mean_y > 1.0
                min_allowed_width = 0.40 if self.is_start_finish_straight else 0.35
                
                is_blocking_path = (min_x < -0.15) and (max_x > 0.15)
                
                if total_width > min_allowed_width and (is_far_wall or is_blocking_path):
                    if not is_far_wall and is_blocking_path:
                        self.get_logger().warn(f"Found a close but blocking wall. Distance: {mean_y:.2f}m")
                    valid_wall_groups.append(g)

            if valid_wall_groups:
                valid_wall_groups.sort(key=lambda g: g['base_y'])
                winner_group = valid_wall_groups[0]['clusters']
                winner_group.sort(key=len, reverse=True)
                u_profile[1] = winner_group[0]
            else:
                u_profile[1] = None
                if front_candidates:
                    self.get_logger().debug("Front candidates present, but rejected as phantom walls.")

        return u_profile
    
    def validate_clusters_straight(self, clusters):
        u_profile = [None, None, None]
        if not clusters:
            return u_profile

        delta_yaw = self.current_yaw - self.start_straight_yaw

        # normalise to -180 to 180
        while delta_yaw > 180: delta_yaw -= 360
        while delta_yaw < -180: delta_yaw += 360

        right_candidates = []
        left_candidates = []
        front_candidates = []

        for c in clusters:
            if len(c) < 20:
                #self.get_logger().info(f"Cluster has too few points")
                continue
            
            local_angle = self.get_cluster_angle(c)
            if local_angle is None: 
                continue

            shifted_angle = local_angle - delta_yaw
            
            angle_norm = abs(shifted_angle) % 180
            if angle_norm > 90:
                angle_norm = 180 - angle_norm

            mean_x_local = sum(p[1] for p in c) / len(c)
            mean_y_local = sum(p[2] for p in c) / len(c)
            
            rad = math.radians(delta_yaw)
            mean_x_straight = mean_x_local * math.cos(rad) - mean_y_local * math.sin(rad)
            mean_y_straight = mean_x_local * math.sin(rad) + mean_y_local * math.cos(rad)

            if angle_norm <= 45.0:
                if abs(mean_x_straight) <= 1.0:
                    if mean_x_straight > 0:
                        right_candidates.append(c)
                    else:
                        left_candidates.append(c)
                else:
                    self.get_logger().info("Cluster has too large an x value")

            else:  
                if mean_y_straight >= 0.15:
                    front_candidates.append(c)

        # ASSIGN SIDE WALLS
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
                self.get_logger().info(f"Real track width: {track_width:.2f}m, L: {abs(left_dist):.2f}m, R: {abs(right_dist):.2f}m")
                
                if 0.75 <= track_width <= 1.25:
                    u_profile[0] = best_right
                    u_profile[2] = best_left
                else:
                    self.get_logger().warn(f"Track width implausible ({track_width:.2f}m). Dropping the shorter wall.")
                    len_r = math.hypot(best_right[-1][1] - best_right[0][1], best_right[-1][2] - best_right[0][2])
                    len_l = math.hypot(best_left[-1][1] - best_left[0][1], best_left[-1][2] - best_left[0][2])
                    
                    if len_r > len_l:
                        u_profile[0] = best_right
                    else:
                        u_profile[2] = best_left
            else:
                self.get_logger().warn("HNF calculation failed for one of the walls.")
                return [None, None, None]
                    
        elif best_right is not None:
            u_profile[0] = best_right
        elif best_left is not None:
            u_profile[2] = best_left

        # ASSIGN FRONT WALL
        if front_candidates:
            self.get_logger().warn(f"Number of FrontCandidates: {len(front_candidates)}")
            groups = []
            for c in front_candidates:
                mean_y = sum(p[2] for p in c) / len(c)
                self.visualize_cluster_line(c, 20,'blue')
                # drop everything that is extremely close
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
                mean_y = g['base_y']  # distance to the wall

                is_far_wall = mean_y > 0.80
                min_allowed_width = 0.40 if self.is_start_finish_straight else 0.35
                
                is_blocking_path = (min_x < -0.15) and (max_x > 0.15)
                
                if total_width > min_allowed_width and (is_far_wall or is_blocking_path):
                    if not is_far_wall and is_blocking_path:
                        self.get_logger().warn(f"Found a close but blocking wall. Distance: {mean_y:.2f}m")
                    valid_wall_groups.append(g)

            if valid_wall_groups:
                valid_wall_groups.sort(key=lambda g: g['base_y'])
                winner_group = valid_wall_groups[0]['clusters']
                winner_group.sort(key=len, reverse=True)
                u_profile[1] = winner_group[0]
            else:
                u_profile[1] = None
                if front_candidates:
                    self.get_logger().debug("Front candidates present, but rejected as phantom walls.")

        return u_profile

    def validate_clusters_turn(self, front_wall, point_data):
        '''Checks the clusters in the turn. There must be a front wall, outer walls may be missing (e.g. in the first turn).
        Returns the clusters from right to left in the order: [right wall, front wall, left wall]. Missing walls are replaced by None.'''

        if front_wall is None:
            self.get_logger().warn("No front wall found. Cannot validate the turn profile.")
            return [None, None, None]

        clusters = self.get_all_clusters_sorted(point_data)

        def kill_all_clusters_between(front_wall, all_clusters):
            # find the limits of the angle range
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
            u_profile = [None, None, None] # 0=right, 1=front, 2=left
            if any(c is front_wall_cluster for c in ordered):
                u_profile[1] = front_wall_cluster
                fw_index = next(i for i, c in enumerate(ordered) if c is front_wall_cluster)
            else: 
                self.get_logger().warn("Front wall not found in the clusters. Cannot validate the turn profile.")
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
                self.get_logger().warn("Error in the angle calculation of the front wall. Skipping...")
                return [None, None, None]

            # computes the smallest intersection angle between two lines (0 to 90 deg)
            def get_angle_diff(a1, a2):
                    diff = abs(a1 - a2) % 180
                    if diff > 90:
                        diff = 180 - diff
                    return diff

            # compute differences (0=right, 1=front, 2=left)
            if angles[0] is not None:
                diff_0_1 = get_angle_diff(angles[0], angles[1]) # should be ~90 deg (orthogonal)
            else: 
                diff_0_1 = None
            if angles[2] is not None:
                diff_1_2 = get_angle_diff(angles[1], angles[2]) # should be ~90 deg (orthogonal)
            else:
                diff_1_2 = None
            if angles[0] is not None and angles[2] is not None:
                diff_0_2 = get_angle_diff(angles[0], angles[2]) # should be ~0 deg (parallel)
            else:
                diff_0_2 = None

            if diff_0_1 is not None and diff_0_1 < 70:
                self.get_logger().warn(f"Right and front not orthogonal! Diff: {diff_0_1:.1f} deg")
                u_profile[0] = None

            if diff_1_2 is not None and diff_1_2 < 70:
                self.get_logger().warn(f"Front and left not orthogonal! Diff: {diff_1_2:.1f} deg")
                u_profile[2] = None
            
            try: 
                wall_dist_left = self.get_closest_point_in_cluster(u_profile[2])[3]
            except: 
                wall_dist_left = None
            try:
                wall_dist_right = self.get_closest_point_in_cluster(u_profile[0])[3]
            except:
                wall_dist_right = None


            if self.direction == "left":
                if wall_dist_right is not None:
                    if wall_dist_right > 1.0:
                        self.get_logger().warn(f"Distance of the right wall {wall_dist_right:.2f}")
                        u_profile[0] = None

                if wall_dist_left is not None:
                    if 0.90 < wall_dist_left < 1.80:
                        self.get_logger().warn(f"Distance of the left wall {wall_dist_left:.2f}")
                        u_profile[2] = None

            else: 
                if wall_dist_left is not None:
                    if wall_dist_left > 1.0:
                        self.get_logger().warn(f"Distance of the left wall {wall_dist_left:.2f}")
                        u_profile[2] = None

                if wall_dist_right is not None:
                    if 0.90 < wall_dist_right < 1.8:
                        self.get_logger().warn(f"Distance of the right wall {wall_dist_right:.2f}")
                        u_profile[0] = None
            
            u_profile, _ = self.merge_clusters(clusters, u_profile)
            self.visualize_cluster_line(u_profile[0], 0, "cyan")
            self.visualize_cluster_line(u_profile[1], 1, "rgb_red")
            self.visualize_cluster_line(u_profile[2], 2, "cyan")
            return u_profile

        self.get_logger().warn(f"No cluster found apart from the front wall.")
        return [None, front_wall_cluster, None]
    
    def sort_clusters_right_to_left(self, clusters): # adapted to right 90 deg and left 270 deg
        """
        Takes a list of clusters and sorts them spatially from right to left.
        System: +X = right, +Y = front
        """
        if not clusters:
            return []

        def get_cluster_bearing(cluster):
            # compute the centroid of the cluster
            # Index 1 = X, Index 2 = Y
            mean_x = sum(p[1] for p in cluster) / len(cluster)
            mean_y = sum(p[2] for p in cluster) / len(cluster)
            
            # compute the angle (0 = front, negative = right, positive = left)
            return math.atan2(-mean_x, mean_y)

        # sort ascending
        sorted_clusters = sorted(clusters, key=get_cluster_bearing, reverse=True)
        
        return sorted_clusters

    def get_all_clusters_sorted(self, point_data):
        if len(point_data) < 2:
            return []

        # sort
        points = point_data[np.argsort(point_data[:, 0])]
        x = points[:, 1]
        y = points[:, 2]

        split_mask = np.zeros(len(points), dtype=bool)

        # GAP DETECTION (Manhattan distance)
        dx = np.diff(x)
        dy = np.diff(y)
        dist_manhattan = np.abs(dx) + np.abs(dy)
        
        split_mask[1:] = dist_manhattan >= 0.15

        #OUTLIER LOGIC
        if len(points) > 2:
            dist_2 = np.abs(x[2:] - x[:-2]) + np.abs(y[2:] - y[:-2])
            fix_2 = (split_mask[2:]) & (dist_2 < 0.15)
            split_mask[2:][fix_2] = False 

        if len(points) > 3:
            dist_3 = np.abs(x[3:] - x[:-3]) + np.abs(y[3:] - y[:-3])
            fix_3 = (split_mask[3:]) & (dist_3 < 0.15)
            split_mask[3:][fix_3] = False 

        # CORNER DETECTION
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
                
                # set all 3 points True in one go (C speed)
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

        # SORT BY PHYSICAL LENGTH (METRES)
        def get_physical_length(c):
            if len(c) < 2: return 0.0
            # Pythagoras between the first and the last point
            return math.hypot(c[-1][1] - c[0][1], c[-1][2] - c[0][2])

        clusters.sort(key=get_physical_length, reverse=True)
        return clusters
    
    def get_cluster_angle(self, cluster):
        if cluster is None or len(cluster) < 2:
            return None
        
        # converts cluster (list) into a NumPy array
        pts = np.array(cluster) 
        x = pts[:, 1]
        y = pts[:, 2]
        
        # centroid calculation
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
        """Deletes a marker in a given namespace."""
        marker = Marker()
        marker.header.frame_id = self.rviz_frame
        marker.ns = ns
        marker.id = m_id
        marker.action = Marker.DELETE
        marker_array.markers.append(marker)

    def scan_callback(self, msg):
        ranges = np.array(msg.ranges)

        # define the requirements for valid values
        valid_mask = np.isfinite(ranges) & (ranges >= 0.075) & (ranges <= 3.0)

        # generate lidar angles for all points
        angles_rad = msg.angle_min + np.arange(len(ranges)) * msg.angle_increment

        # keep only valid values
        valid_ranges = ranges[valid_mask]
        valid_angles_rad = angles_rad[valid_mask]

        # compute coordinates
        x_ros = valid_ranges * np.cos(valid_angles_rad)
        y_ros = valid_ranges * np.sin(valid_angles_rad)

        angles_deg = np.degrees(valid_angles_rad)
        user_angles_deg = np.mod(angles_deg + 90.0, 360.0)

        # join into an N x 4 array: (angle, X, Y, distance)
        point_data = np.column_stack((user_angles_deg, x_ros, y_ros, valid_ranges))
        
        self.last_point_data = point_data

        self.update_strategy_params()

        if self.camera_calibration or self.debug:
            self.debug_main_logic(point_data)
        else:
            self.main_logic(point_data)
    
    def get_closest_point_in_cluster(self, cluster):
        """
        Returns the point of a cluster that has the shortest distance to the LiDAR.
        """
        if cluster is None or len(cluster) == 0:
            return None

        # finds the element in the cluster with the smallest dist
        closest_point = min(cluster, key=lambda p: p[3])
        
        return closest_point

    def merge_clusters(self, all_clusters, validated_clusters):
        """
        Tries to merge neighbouring clusters into a single cluster.
        Checks the orthogonal (perpendicular) distance and the parallel (longitudinal) distance.
        """
        # CONFIGURATION OF THE TOLERANCES
        max_angle_gap = 10.0      # 10 degrees maximum angle deviation
        max_perp_gap = 0.08       # max 8 cm distance away from the wall
        max_parallel_gap = 0.60   # max 60 cm gap along the wall

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

            # compute the centroid of the cluster
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

    def visualize_target_point(self, x, y, m_id=24, colour_name="yellow", label="TARGET"):
        """
        Visualises the target point as a point and text label in Foxglove.
        """

        colours = {
            "rgb_red": (1.0, 0.0, 0.0),
            "rgb_green": (0.0, 1.0, 0.0),
            "blue": (0.0, 0.5, 1.0),
            "cyan": (0.0, 1.0, 1.0),
            "magenta": (1.0, 0.0, 1.0),
            "yellow": (1.0, 1.0, 0.0),
            "orange": (1.0, 0.5, 0.0)
        }
        rgb = colours.get(colour_name.lower(), (1.0, 1.0, 0.0))

        marker_array = MarkerArray()

        sphere_marker = Marker()
        sphere_marker.header.frame_id = "base_link"
        sphere_marker.header.stamp = self.get_clock().now().to_msg()
        sphere_marker.ns = "target_point"
        sphere_marker.id = m_id
        sphere_marker.type = Marker.SPHERE
        sphere_marker.action = Marker.ADD
        
        # set position
        sphere_marker.pose.position.x = float(x)
        sphere_marker.pose.position.y = float(y)
        sphere_marker.pose.position.z = 0.1  # 10cm above the floor
        
        # size
        sphere_marker.scale.x = 0.15
        sphere_marker.scale.y = 0.15
        sphere_marker.scale.z = 0.15
        
        # set colour
        sphere_marker.color.r = rgb[0]
        sphere_marker.color.g = rgb[1]
        sphere_marker.color.b = rgb[2]
        sphere_marker.color.a = 1.0
        
        marker_array.markers.append(sphere_marker)

        self.send_text(marker_array, m_id=m_id + 1, text=label, x=x, y=y, color=rgb)

        self.pub_markers.publish(marker_array)

    def get_target_point_straight(self, hnf_inner, hnf_outer):
        """
        Computes the target point using the HNF.
        If only one wall is present, the offset is computed directly from it, without projecting an error-prone virtual wall!
        """
        target_y = self.lookahead_dist_straight
        target_x = 0.0
        
        # ignore walls that are too far away
        if hnf_inner is not None and hnf_inner[2] > 1.0:
            hnf_inner = None
        if hnf_outer is not None and hnf_outer[2] > 1.0:
            hnf_outer = None

        def get_x_at_y(hnf_params, y_val):
            nx, ny, d = hnf_params
            if abs(nx) < 1e-6: return 0.0 
            return (d - ny * y_val) / nx

        # compute the intersection with the Y view axis
        x_inner = get_x_at_y(hnf_inner, target_y) if hnf_inner else None
        x_outer = get_x_at_y(hnf_outer, target_y) if hnf_outer else None

        if x_inner is not None and x_outer is not None:
            if x_inner < 0: # inner wall is on the left
                t_inner = x_inner + self.lane_ratio
                t_outer = x_outer - (1.0 - self.lane_ratio)
            else:           # inner wall is on the right
                t_inner = x_inner - self.lane_ratio
                t_outer = x_outer + (1.0 - self.lane_ratio)
                
            target_x = (t_inner + t_outer) / 2.0

        elif x_inner is not None:
            if x_inner < 0: 
                target_x = x_inner + self.lane_ratio
            else:           
                target_x = x_inner - self.lane_ratio

        elif x_outer is not None:
            if x_outer > 0: # outer wall is on the right
                target_x = x_outer - (1.0 - self.lane_ratio)
            else:            # outer wall is on the left
                target_x = x_outer + (1.0 - self.lane_ratio)
                
        else:
            # emergency, both walls are missing completely
            target_x = 0.0

        if x_inner is not None:
            if x_inner < 0 and target_x < x_inner + self.min_wall_dist:
                target_x = x_inner + self.min_wall_dist
            elif x_inner > 0 and target_x > x_inner - self.min_wall_dist:
                target_x = x_inner - self.min_wall_dist
                
        if x_outer is not None:
            if x_outer < 0 and target_x < x_outer + self.min_wall_dist:
                target_x = x_outer + self.min_wall_dist
            elif x_outer > 0 and target_x > x_outer - self.min_wall_dist:
                target_x = x_outer - self.min_wall_dist

        return (target_x, target_y)

    def track_front_wall(self, point_data, last_front_wall):
        if self.state == 'PARKING':
            active_direction = self.park_turn_direction
        else:
            active_direction = self.direction

        if last_front_wall is None or len(point_data) == 0:
            return None

        last_fw_array = np.array(last_front_wall)
        min_angle = np.min(last_fw_array[:, 0])
        max_angle = np.max(last_fw_array[:, 0])

        # narrower search window, so the parking wall is not picked up
        if active_direction == 'left':
            roi_min = min_angle - 18.0
            roi_max = max_angle + 8.0
        else:
            roi_min = min_angle - 8.0
            roi_max = max_angle + 18.0

        mask = (point_data[:, 0] >= roi_min) & (point_data[:, 0] <= roi_max)
        roi_points = point_data[mask]

        roi_clusters = self.get_all_clusters_sorted(roi_points)

        if not roi_clusters:
            self.get_logger().warn("WARNING: tracked wall lost in the ROI!")
            return last_front_wall

        # limit the wall orientation (wall angle relative to the robot) per iteration
        MAX_WALL_ANGLE_SHIFT_DEG = 15.0
        candidate = roi_clusters[0]
        last_wall_angle = self.get_cluster_angle(last_front_wall)
        new_wall_angle = self.get_cluster_angle(candidate)

        if last_wall_angle is not None and new_wall_angle is not None:
            diff = abs(new_wall_angle - last_wall_angle) % 180.0
            if diff > 90.0:
                diff = 180.0 - diff
            if diff > MAX_WALL_ANGLE_SHIFT_DEG:
                self.get_logger().warn(
                    f"FW tracker: wall angle change too large ({diff:.1f} deg). Keeping the last front wall."
                )
                return last_front_wall

        return candidate

    def get_closest_measure(self, point_data, target_angle):
        if len(point_data) == 0:
            self.get_logger().info(f"Point_Data is empty!")
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

        # CONFIGURATION FOR 1280p
        H_FOV = 120.0  
        IMAGE_WIDTH = 1280.0
        CENTER_X = IMAGE_WIDTH / 2.0
        DEG_PER_PIXEL = H_FOV / IMAGE_WIDTH 

        class_mapping = {0: 'green', 1: 'red', 2: 'pink'}

        yolo_boxes = []

        for r in results:
            for box in r.boxes:
                if float(box.conf[0]) > 0.85:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    
                    # angle calculation
                    center_x = (x1 + x2) / 2.0
                    pixel_offset = center_x - CENTER_X
                    angle_offset_deg = pixel_offset * DEG_PER_PIXEL
                    
                    # calculation of the angle relative to the robot
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

        # sort: near to far
        yolo_boxes.sort(key=lambda b: b['y_max'], reverse=True)

        detected_obstacles = []
        available_clusters = self.get_all_clusters_sorted(point_data)

        for box_data in yolo_boxes:
            # find cluster
            obstacle_cluster = self.get_lidar_distance(box_data['angle_rad'], available_clusters)

            if obstacle_cluster is not None:
                obj_x, obj_y = self.get_weight_point_for_cluster(obstacle_cluster) 
                
                lidar_dist = abs(obj_y)
                
                # camera distance
                camera_dist = self.analyzer.get_distance_from_bbox(box_data['y_max'])
                
                TOLERANCE = 0.40 
                
                if abs(camera_dist - lidar_dist) > TOLERANCE:
                    self.get_logger().warn(
                        f"Ghost filter: {box_data['class_name']} rejected! "
                        f"Camera: {camera_dist:.2f}m, LiDAR: {lidar_dist:.2f}m."
                    )
                    continue 

                # send marker & store object
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
            # filter by number of points (at least 2, max 60 for close obstacles)
            if len(cluster) < 2 or len(cluster) > 60:
                continue

            # width filter
            c_start = cluster[0]
            c_end = cluster[-1]
            width = math.hypot(c_start[1] - c_end[1], c_start[2] - c_end[2])
            if width > 0.25:
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
            self.get_logger().info(f"MATCH: camera {math.degrees(camera_angle_rad):.1f} deg -> lidar {best_angle_deg:.1f} deg (distance: {min_dist:.2f}m), yAxisDist: {closest_point[2]:.2f}m")
            return best_cluster

        return None

    def middle_of_cluster(self, cluster):
        """Computes the angle of the cluster's midpoint to the lidar."""
        sum = 0
        for c in cluster:
            sum += c[0]  # angle of the point
        return sum / len(cluster)

    def angle_diff(self, a, b):
        """Computes the difference between two angles (rad)."""
        return math.atan2(math.sin(a - b), math.cos(a - b))


    # -----------------------------------------------
    # New coordinate system idea for driving turns
    # -----------------------------------------------

    def cluster_to_hnf(self, cluster):
        """
        Computes the Hesse normal form from a cluster of measurement points.
        """
        if cluster is None:
            return None
        
        # extract the x and y coordinates into a NumPy array
        # index 1 is x_coord, index 2 is y_coord
        points = np.array([[p[1], p[2]] for p in cluster])
        
        if len(points) == 0:
            raise ValueError("The cluster is empty.")
            
        # compute the centroid
        centroid = np.mean(points, axis=0)
        
        # centre the data
        centered_points = points - centroid
        
        # singular value decomposition (SVD) for orthogonal regression
        # Vh holds the eigenvectors of the covariance matrix
        _, _, Vh = np.linalg.svd(centered_points)
        
        # the normal vector is the eigenvector with the smallest variance
        normal_vector = Vh[-1]
        
        # compute distance d (dot product of centroid and normal vector)
        d = np.dot(centroid, normal_vector)
        
        # normalisation: d must be greater than or equal to 0
        if d < 0:
            normal_vector = -normal_vector
            d = -d
            
        n_x, n_y = normal_vector
        
        return n_x, n_y, d
    
    def extract_wall_lines(self, u_profile):
        if self.state == 'PARKING':
            active_direction = self.park_turn_direction
        else:
            active_direction = self.direction
        if active_direction == "left":
            # in a left turn, left (2) is the inner wall (side)
            opposite_cluster, front_cluster, side_cluster = u_profile
        else:
            # in a right turn, right (0) is the inner wall (side)
            side_cluster, front_cluster, opposite_cluster = u_profile

        # extract the front wall
        if front_cluster is None or len(front_cluster) < 2:
            front_straight = None
        else:
            front_straight = self.cluster_to_hnf(front_cluster)

        # extract the side wall
        if side_cluster is not None and len(side_cluster) >= 2:
            # primary target: the inner wall of the turn
            side_straight = self.cluster_to_hnf(side_cluster)
            
        elif opposite_cluster is not None and len(opposite_cluster) >= 2:
            # fallback: mirror the opposite wall across the 3m distance
            oppo_x, oppo_y, oppo_d = self.cluster_to_hnf(opposite_cluster)
            
            new_nx = -oppo_x
            new_ny = -oppo_y
            new_d = 3.0 - oppo_d
            
            # HNF condition: d must always be >= 0
            if new_d < 0:
                self.get_logger().warn("WARNING: HNF distance negative. Mirroring corrected.")
                new_nx = -new_nx
                new_ny = -new_ny
                new_d = abs(new_d)
                
            side_straight = (new_nx, new_ny, new_d)
            
        else:
            # no usable side walls present
            side_straight = None

        return front_straight, side_straight

    def calculate_target_line(self, side_line_params, front_line_params, desired_lane_ratio):
        if front_line_params is None:
            return None, None

        n_xf, n_yf, d_f = front_line_params
        
        d_target = d_f - (1.0 - desired_lane_ratio)
            
        target_line_params = (n_xf, n_yf, d_target)
        
        if side_line_params is None:
            return target_line_params, None

        # radius limit
        delta_d_new = d_f - d_target
        n_xs, n_ys, d_s = side_line_params
        
        r_max_target = self.TRACK_WIDTH_M - delta_d_new - (self.ROBOT_WIDTH_M / 2.0)
        
        max_allowed_radius_m = min(r_max_target, self.MAX_KINEMATIC_RADIUS_M)
        
        # make sure the radius stays physically drivable (> 0)
        max_allowed_radius_m = max(max_allowed_radius_m, 0.0)
        
        return target_line_params, max_allowed_radius_m
    
    def get_intersection_point(self, target_line_params):
        """
        Computes the intersection of the y axis (robot trajectory) with the target line.
        """
        if self.state == 'PARKING':
            active_direction = self.park_turn_direction
        else:
            active_direction = self.direction

        if target_line_params is None:
            return None, None, None
            
        n_x, n_y, d = target_line_params
        
        # check for parallelism (avoid division by zero)
        epsilon = 1e-6
        if abs(n_y) < epsilon:
            # target line is parallel to the driving direction, no intersection
            return None, None, None
            
        # compute the intersection
        intersection_x_m = 0.0  # by definition the robot drives on x=0
        intersection_y_m = d / n_y
        
        # the intersection must lie in front of the robot
        if intersection_y_m <= 0.0:
            # the intersection lies behind the robot.
            return None, None, None
        
        # sign of the dot product
        if active_direction == 'left':
            nx_directional = n_x
        else:
            nx_directional = -n_x
            
        nx_clipped = max(-1.0, min(1.0, nx_directional))
        
        # compute the angle
        turn_angle_deg = math.degrees(math.acos(nx_clipped))
        self.get_logger().info(f"Turn angle error: {turn_angle_deg:.1f} deg")

        return intersection_x_m, intersection_y_m, turn_angle_deg

    def calculate_curve_geometry(self, intersection_y_m, turn_angle_deg, max_allowed_radius_m):
        """
        Computes the optimal turn radius and the distance to the turn-in point on the y axis.
        """
        
        if intersection_y_m is None:
            # turn is geometrically or mechanically impossible
            self.get_logger().error("Invalid intersection: no turn-in possible.")
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
                f"EMERGENCY TURN-IN: pivot point missed ({real_axle_dist_m:.2f} m). "
                "Forcing an immediate turn."
            )

            real_axle_dist_m = 0.0
            
        return curve_radius_m, real_axle_dist_m

    def check_turn_trigger(self, entry_point_distance_m):
        """
        Checks from the currently computed distance whether the turn-in manoeuvre has to start.
        """
        if entry_point_distance_m is None:
            return False

        trigger_tolerance_m = 0.07

        if entry_point_distance_m <= trigger_tolerance_m:
            return True
            
        return False

    def execute_turn(self, curve_radius_m):
        """
        Translates the radius via the SteeringController and publishes the Twist message.
        """

        if self.state == 'PARKING':
            active_direction = self.park_turn_direction
        else:
            active_direction = self.direction

        if active_direction == "left":
            is_left_turn = True
        else:
            is_left_turn = False

        cmd = Twist()
        
        if curve_radius_m is None or curve_radius_m <= 0.0:
            self.get_logger().error("Invalid turn radius.")
            return False
        
        steering_signal = self.steering_ctrl.get_steering_for_radius(target_radius=curve_radius_m,turning_left=is_left_turn)
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
        Returns True if the turn is finished according to the gyro or the wall angle.
        """
        
        exit_tolerance_imu = self.turn_exit_tolerance
        exit_tolerance_lidar = self.turn_exit_tolerance * 1.5

        yaw_diff = (self.current_yaw - self.start_turn_yaw + 180) % 360 - 180
        progressed_angle = abs(yaw_diff)
        
        target_angle_abs = abs(turn_angle)
        
        if progressed_angle >= (target_angle_abs - exit_tolerance_imu):
            self.get_logger().info(f"Turn finished (gyro hard exit): {progressed_angle:.1f} deg reached.")
            return True
            
        if progressed_angle < (target_angle_abs - 30.0):
            return False
            
        if front_line_params is not None:
            n_x, n_y, d = front_line_params
            
            wall_angle_deg = math.degrees(math.atan2(n_y, n_x))
            
            wall_error_deg = min(abs(wall_angle_deg % 180), abs(180 - (wall_angle_deg % 180)))
            
            if wall_error_deg < exit_tolerance_lidar:
                self.get_logger().info(f"Fused match! Gyro at {progressed_angle:.1f} deg, wall perfectly parallel (error: {wall_error_deg:.1f} deg).")
                return True

        return False

    # -----------------------------------------------
    # Test functions
    # -----------------------------------------------

    def clear_all_lines(self):
        """
        Deletes all line markers in the namespace 'walls'.
        """
        marker_array = MarkerArray()

        for i in range(25):
            self.delete_marker(marker_array, m_id=i, ns="walls")
        self.pub_markers.publish(marker_array)

    def visualize_cluster_line(self, cluster, m_id, colour_name="rgb_red", label="CLUSTER_LINE"):
        """
        Creates a marker for a cluster.
        """

        if cluster is None or len(cluster) < 2:
            return

        colours = {
            "rgb_red": (1.0, 0.0, 0.0),
            "rgb_green": (0.0, 1.0, 0.0),
            "blue": (0.0, 0.5, 1.0),
            "cyan": (0.0, 1.0, 1.0),
            "magenta": (1.0, 0.0, 1.0),
            "yellow": (1.0, 1.0, 0.0)
        }
        
        rgb = colours.get(colour_name.lower(), (1.0, 1.0, 1.0))
        
        start_p = (cluster[0][1], cluster[0][2])
        end_p = (cluster[-1][1], cluster[-1][2])
        
        middle_x = (start_p[0] + end_p[0]) / 2.0
        middle_y = (start_p[1] + end_p[1]) / 2.0
        
        marker_array = MarkerArray()
        
        self.send_line(marker_array, m_id=m_id, p1=start_p, p2=end_p, color=rgb)
        
        self.send_text(marker_array, m_id=m_id + 1000, text=label, x=middle_x, y=middle_y, color=rgb)
        
        self.pub_markers.publish(marker_array)

    def visualize_hnf_line(self, hnf_params, m_id, colour_name="rgb_red", label="HNF_LINE"):
        """
        Creates a marker for a line from the Hesse normal form (n_x, n_y, d).
        """
        if hnf_params is None:
            return

        colours = {
            "rgb_red": (1.0, 0.0, 0.0),
            "rgb_green": (0.0, 1.0, 0.0),
            "blue": (0.0, 0.5, 1.0)
        }
        
        # fallback to white if the colour is not in the dictionary
        rgb = colours.get(colour_name.lower(), (0.0, 0.0, 0.0))
        
        n_x, n_y, d = hnf_params
        
        # compute the foot of the perpendicular
        p_lot_x = n_x * d
        p_lot_y = n_y * d
        
        # direction vector of the line
        v_x = -n_y
        v_y = n_x
        
        # two points for a 6 metre long line
        p1 = (p_lot_x + v_x * 3.0, p_lot_y + v_y * 3.0)
        p2 = (p_lot_x - v_x * 3.0, p_lot_y - v_y * 3.0)
        
        marker_array = MarkerArray()
        
        # draw the line
        self.send_line(marker_array, m_id=m_id, p1=p1, p2=p2, color=rgb)
        self.send_text(marker_array, m_id=m_id + 1000, text=label, x=p_lot_x, y=p_lot_y, color=rgb)
        
        self.pub_markers.publish(marker_array)

    def test_extract_wall_lines(self, u_profile):
        '''Tests the function extract_wall_lines() and visualises the results in RViz.'''
        front_straight, side_straight = self.extract_wall_lines(u_profile)
        self.get_logger().info(f"Front HNF: {front_straight}")
        self.get_logger().info(f"Side HNF: {side_straight}")
        self.visualize_hnf_line(front_straight, m_id=1, colour_name="blue", label="")
        self.visualize_hnf_line(side_straight, m_id=0, colour_name="blue", label="")
        return front_straight, side_straight

    def test_calculate_target_line(self, u_profile, desired_lane_ratio):
        '''Tests the function calculate_target_line() and visualises the results in RViz.'''
        front_line_params, side_line_params = self.extract_wall_lines(u_profile)
        target_line_params, max_radius = self.calculate_target_line(side_line_params, front_line_params, desired_lane_ratio)
        radius = f"{max_radius:.2f}" if max_radius is not None else "None"
        self.visualize_hnf_line(target_line_params, m_id=3, colour_name="rgb_green", label="")
        return target_line_params, max_radius

    def test_get_intersection_point(self, target_line_params):
        '''Tests the function get_intersection_point() and visualises the intersection in RViz.'''
        intersection_x, intersection_y, angle = self.get_intersection_point(target_line_params)
        if intersection_x is not None and intersection_y is not None:
            self.get_logger().info(f"Intersection: (X={intersection_x:.2f}, Y={intersection_y:.2f}), intersection angle: {angle:.2f} deg")
            marker_array = MarkerArray()
            self.send_sphere(marker_array, m_id=20, x=intersection_x, y=intersection_y, color=(1.0, 1.0, 0.0)) # yellow
            self.pub_markers.publish(marker_array)
            return intersection_x, intersection_y, angle
        
        else:
            self.get_logger().warn("No valid intersection found.")
            return None, None, None

    def test_calculate_curve_geometry(self, intersection_y_m, turn_angle_deg, max_allowed_radius_m):
        """Tests the function calculate_curve_geometry() and logs the results."""
        
        curve_radius_m, entry_point_distance_m = self.calculate_curve_geometry(intersection_y_m, turn_angle_deg, max_allowed_radius_m)
        
        radius_str = f"{curve_radius_m:.2f} m" if curve_radius_m is not None else "None"
        entry_str = f"{entry_point_distance_m:.2f} m" if entry_point_distance_m is not None else "None"
        self.get_logger().info(f"Computed turn geometry: radius = {radius_str}, turn-in point distance = {entry_str}")
        
        marker_array = MarkerArray()
    
        if entry_point_distance_m is not None and intersection_y_m is not None:
            self.send_sphere(marker_array, m_id=21, x=0.0, y=entry_point_distance_m, color=(0.0, 1.0, 0.0))
            
            if self.state == 'PARKING':
                active_direction = self.park_turn_direction
            else:
                active_direction = self.direction
            is_left = (active_direction == 'left')
            
            self.visualize_geogebra_angle(marker_array, intersection_y_m, turn_angle_deg, is_left_turn=is_left, m_id=20, radius=0.4)
            
            text_x = -0.45 if is_left else 0.45
            self.send_text(marker_array, m_id=961, text=f"{turn_angle_deg:.1f} deg", 
                        x=text_x, y=intersection_y_m + 0.2, color=(1.0, 1.0, 0.0))
        else:
            self.delete_marker(marker_array, m_id=900)
            self.delete_marker(marker_array, m_id=960)
            self.delete_marker(marker_array, m_id=961)
            
        if hasattr(self, 'pub_markers'):
            self.pub_markers.publish(marker_array)

        return curve_radius_m, entry_point_distance_m

    def test_check_turn_trigger(self, entry_point_distance_m):
        """Tests the function check_turn_trigger() and logs whether the trigger fires."""
        trigger = self.check_turn_trigger(entry_point_distance_m)
        status = "TRIGGERED" if trigger else "NOT TRIGGERED"
        entry_point_distance_m = f"{entry_point_distance_m:.2f} m" if entry_point_distance_m is not None else "None"
        self.get_logger().info(f"Turn Trigger Check: {status} (Entry Point Distance: {entry_point_distance_m})")
        return trigger

    def test_execute_turn(self, curve_radius_m):
        """Tests the function execute_turn() and logs the result."""
        success = self.execute_turn(curve_radius_m)
        if success:
            self.get_logger().info(f"Turn executed with radius {curve_radius_m:.2f} m")
        else:
            self.get_logger().error("Failed to execute turn.")
        return success

    def test_check_turn_completion_fused(self, target_angle, front_line_params):
        """Tests the function check_turn_completion_fused() and logs whether the turn is completed."""
        self.visualize_hnf_line(front_line_params, m_id=1, colour_name="rgb_red", label="front_wall")
        completed = self.check_turn_completion_fused(target_angle, front_line_params)
        status = "COMPLETED" if completed else "IN PROGRESS"
        self.get_logger().info(f"Turn Completion Check: {status} (Current Gyro: {self.current_yaw:.2f} deg, Target Angle: {target_angle:.2f} deg)")
        return completed

    def are_clusters_near(self, cluster_a, cluster_b, threshold=0.02):
        # midpoint A
        avg_a = np.mean(cluster_a, axis=0) # [x_avg, y_avg]
        # midpoint B
        avg_b = np.mean(cluster_b, axis=0)
        
        # Euclidean distance
        dist = np.linalg.norm(avg_a - avg_b)
        return dist < threshold
    
    def visualize_geogebra_angle(self, marker_array, intersection_y_m, turn_angle_deg, is_left_turn, m_id=960, radius=0.4):
        """
        Draws the intersection angle as a circular sector with the degree value.
        """

        if intersection_y_m is None or turn_angle_deg is None:
            self.delete_marker(marker_array, m_id)
            self.delete_marker(marker_array, m_id + 1)
            return

        # set up the arc marker
        marker = Marker()
        marker.header.frame_id = "base_link" 
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "angle_arc"
        marker.id = m_id
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        marker.scale.x = 0.015 
        marker.color.r, marker.color.g, marker.color.b, marker.color.a = 0.59, 0.0, 1.0, 1.0

        # compute the angle
        start_angle_rad = math.pi / 2.0  # y axis = 90 degrees
        turn_rad = math.radians(turn_angle_deg)
        
        if is_left_turn:
            end_angle_rad = start_angle_rad + turn_rad
        else:
            end_angle_rad = start_angle_rad - turn_rad

        # draw the arc
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
        
        self.send_text(marker_array, m_id=m_id + 1, text=f"{turn_angle_deg:.1f} deg", x=text_x, y=text_y, color=(0.59, 0.0, 1.0))

    def handle_unpark(self, point_data):
        if not self.unpark_sequence_done:
            self.execute_blind_steps()
            self.unpark_sequence_done = True
            return

        self.handle_last_parking_step(point_data)

    def execute_blind_steps(self):
        self.get_logger().info("Starting blind unpark sequence (steps 0-9)...")

        steer_mult = -1.0 if self.direction == "left" else 1.0
        if self.direction == 'left':
            steps = [
                (0.0,   0.0, 1.0),  # step 0: wait
                (0.0,   0.8, 2.0),   # step 1: steer at standstill
                (-257.0, 0.8, 0.63),  # step 2: reverse
                (0.0,  -0.8, 1.5),   # step 3: counter-steer at standstill
                (205.0, -0.8, 0.48),   # step 4: forward
                (0.0,   0.8, 1.5),   # step 5: steer at standstill
                (-257.0, 1.4, 0.45),  # step 6: reverse
                (0.0,  -0.8, 1.5),   # step 7: counter-steer at standstill
                (205.0, -0.8, 2.0),   # step 8: forward
                (0.0,   0.8, 0.5),   # step 9: steer at standstill
            ]

        else:
            steps = [
                (0.0,   0.0, 1.0),  # step 0: wait
                (0.0,   0.8, 2.0),   # step 1: steer at standstill
                (-225.0, 0.8, 0.45),  # step 2: reverse
                (0.0,  -0.8, 1.5),   # step 3: counter-steer at standstill
                (200.0, -0.8, 0.55),   # step 4: forward
                (0.0,   0.8, 1.5),   # step 5: steer at standstill
                (-225.0, 1.4, 0.35),  # step 6: reverse
                (0.0,  -0.8, 1.5),   # step 7: counter-steer at standstill
                (230.0, -0.8, 1.4),   # step 8: forward
                (0.0,   0.0, 0.5),   # step 9: steer at standstill
                (250.0, 0.0, 0.7),   # step 10: forward
                (0.0,   0.8, 0.5),   # step 11: steer at standstill
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

        self.get_logger().info("Blind sequence finished.")

    def handle_last_parking_step(self, point_data):
        cmd = Twist()
        if self.direction == "left":
            self.check_for_obstacle_color(point_data, (self.turn_count + 1), 0.25, 2.0)
            cmd.linear.x = 270.0
        else:
            cmd.linear.x = 270.0

        
        cmd.angular.z = 0.8 * (-1.0 if self.direction == "left" else 1.0)
        self.pub_cmd_vel.publish(cmd)
        
        duration = 1.2 if self.direction == "left" else 1.6

        self.get_logger().info("Last unpark step reached")

        if self.unpark_timer is None:
            self.unpark_timer = time.monotonic() + duration
        
        if time.monotonic() > self.unpark_timer:
            self.get_logger().info("Unparking finished.")
            self.state = 'FOLLOW_LANE'

    def debug_main_logic(self, point_data):
        if self.camera_calibration:
            return

        if not self.imu_ready:
            self.get_logger().info("Waiting for IMU for debug mode...")
            return

        if self.debug_start:
            self.yaw_offset = self.current_yaw
            self.start_straight_yaw = self.current_yaw
            self.debug_start = False
            self.parking_phase = 'ADDRESSING_PARKING_SPACE'
            self.state = 'PARKING'

        all_clusters = self.get_all_clusters_sorted(point_data)

        counter = 12
        for c in all_clusters:
            self.visualize_cluster_line(c, counter, 'rgb_green', "Cluster")
            counter += 1

        validated_clusters = self.validate_clusters_far_front_wall(all_clusters)
        merged_validated_clusters, _ = self.merge_clusters(all_clusters, validated_clusters)
        self.right_wall = merged_validated_clusters[0]
        self.front_wall = merged_validated_clusters[1]
        self.left_wall  = merged_validated_clusters[2]
        right_wall_hnf = self.cluster_to_hnf(self.right_wall)
        front_wall_hnf = self.cluster_to_hnf(self.front_wall)
        left_wall_hnf = self.cluster_to_hnf(self.left_wall)

        self.visualize_hnf_line(front_wall_hnf, m_id=1, colour_name="rgb_red", label="Front HNF")
        self.visualize_hnf_line(left_wall_hnf, m_id=0, colour_name="blue", label="Left HNF")
        self.visualize_hnf_line(right_wall_hnf, m_id=2, colour_name="rgb_green", label="Right HNF")

        

    def check_for_obstacle_color(self, point_data, requested_turn_count, min_distance_to_obstacle=0.0, max_distance_to_obstacle=2.0, front_wall_dist=None):
        current_straight = requested_turn_count % 4
        current_obstacle = self.obstacle_memory[current_straight]
        prediction = False
        
        if current_obstacle is None:
            detected_obstacles = self.get_obstacles_from_camera(point_data)
            
            if detected_obstacles:
                # Y FILTER: only obstacles from the minimum distance on
                detected_obstacles = list(filter(lambda x: x[1] > min_distance_to_obstacle, detected_obstacles))
                
                if requested_turn_count == self.turn_count:
                    # relative to the lane direction.
                    yaw_drift = math.radians(self.current_yaw - self.start_straight_yaw)
                    def lane_lateral(x, y):
                        rng = math.hypot(x, y)
                        return rng * math.sin(math.atan2(x, y) - yaw_drift)
                    detected_obstacles = list(filter(lambda x: abs(lane_lateral(x[0], x[1])) <= 0.45, detected_obstacles))
                else:
                    prediction = True
                    if self.direction == 'left':
                        detected_obstacles = list(filter(lambda x: x[0] <= 0.15, detected_obstacles))
                    elif self.direction == 'right':
                        detected_obstacles = list(filter(lambda x: x[0] >= -0.15, detected_obstacles))

                # sorted by Euclidean distance
                detected_obstacles.sort(key=lambda x: math.hypot(x[0], x[1]))
                
                if detected_obstacles:
                    closest_x, closest_y, closest_color = detected_obstacles[0]
                    closest_dist = math.hypot(closest_x, closest_y)

                    if self.state == 'FOLLOW_LANE' and closest_dist < self.panic_obstacle_dist:
                        self.panic_close_obstacle = True
                        self.get_logger().error(
                            f"!!! CLOSE OBSTACLE {closest_dist:.2f}m < {self.panic_obstacle_dist:.2f}m -> PANIC"
                        )
                    
                    if min_distance_to_obstacle < closest_dist < max_distance_to_obstacle:
                        new_obstacle = Obstacle(closest_color, None, prediction)
                        self.obstacle_memory[current_straight] = new_obstacle
                        self.get_logger().warn(f"+++ OBSTACLE LOCKED: {closest_color.upper()}, {closest_dist:.2f} m on straight: {current_straight}, Prediction: {prediction}, front_wall_dist: {front_wall_dist} m+++")
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
            self.get_logger().warn("Distance to the front_wall is outside the valid range.")
            return None
            

        if obst_to_outer_wall_dist >= 0.20 and obst_to_outer_wall_dist < 0.50:
            zone_id += 1
        elif obst_to_outer_wall_dist >= 0.50 and obst_to_outer_wall_dist <= 0.80:
            zone_id += 0
        else:
            self.get_logger().warn("Distance to the side_wall is outside the valid range.")
            zone_id = None

        return zone_id

    def set_lane_ratio_for_obstacle_cmd(self, obstacle_cmd, obstacle, front_wall_dist, is_turn_exit=False, apply_state=True):
        is_left = (self.direction == "left")
        if self.turn_count < 4:
            max_shift = 0.007
        else:
            max_shift = 0.008
    
        actual_dist = front_wall_dist if front_wall_dist is not None else 2.0

        is_evading = False  # True if we have to dodge without smoothing in an emergency

        if is_turn_exit:
            target_ratio = self.standard_lane_ratio_exit
        else:
            if self.turn_count == 0 and not is_left:
                target_ratio = 0.45
                is_evading = True
            else:
                target_ratio = self.standard_lane_ratio_approach

        # PARKING OBSTACLE LOGIC
        if obstacle_cmd is not None and obstacle is not None:
            obst_is_green = obstacle.color == 'green'
            if self.last_turn_for_parking or self.parking_straight:
                if (obst_is_green and is_left) or ((not obst_is_green) and (not is_left)):
                    target_ratio = 0.22
                else:
                    obst_is_relevant = False
                    if obstacle.is_localized:
                        obst_is_relevant = (obstacle.zone_id < 2) # obstacle is only relevant if it stands right at the start of the straight
                    else: 
                        obst_is_relevant = True

                    if obst_is_relevant:
                        target_ratio = 0.65
                    else:
                        target_ratio = 0.22
                return target_ratio
            

            if not obstacle.is_localized:
                if obstacle.prediction and actual_dist < 1.70:
                    # we have passed the predicted obstacle
                    target_ratio = self.standard_lane_ratio_approach
                    self.is_obstacle_passed = True
                else:
                    # we have to dodge
                    is_evading = True
                    
                    if self.is_start_finish_straight:
                        if obstacle_cmd == "green":
                            target_ratio = 0.25 if is_left else 0.60
                        else:
                            target_ratio = 0.60 if is_left else 0.25
                    else:
                        if obstacle_cmd == "green":
                            target_ratio = 0.25 if is_left else 0.80
                        else:
                            target_ratio = 0.80 if is_left else 0.25
            else:
                obst_zone_id = obstacle.zone_id
                obst_passed = False
                
                if actual_dist is not None:
                    # check whether we are past the obstacle
                    if obst_zone_id <= 1:
                        obst_passed = actual_dist < 1.75
                        obst_y = 2.0
                    elif obst_zone_id <= 11:
                        obst_passed = actual_dist < 1.35
                        obst_y = 1.5
                    else:
                        obst_passed = actual_dist < 0.85
                        obst_y = 1.0
                    
                    self.is_obstacle_passed = obst_passed

                    dist_to_obst = actual_dist - obst_y
                
                if not obst_passed:
                    # if the obstacle is critically close, switch off smoothing
                    if dist_to_obst < 0.65:
                        is_evading = True

                    if is_turn_exit and obst_zone_id >= 20:
                        if (obst_is_green and is_left) or ((not obst_is_green) and (not is_left)):
                            target_ratio = 0.40
                        else:
                            target_ratio = 0.60

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
            # switch immediately without smoothing
            dist_to_inner_wall = target_ratio
        else:
            shift_dist = 1.12 if self.turn_count < 4 else 1.30
            if actual_dist > shift_dist:
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
        Processes sensor data, computes the exact physical position of an obstacle 
        and places it in the robot's topological memory.
        """
        front_hnf, side_hnf = u_profile_hnf
        if front_hnf is None:
            return None
        
        nx_f, ny_f, dist_to_front = front_hnf
        
        color, current_obstacle = self.check_for_obstacle_color(point_data, self.turn_count, 0.0, dist_to_front - 0.85)

        if color is None and current_obstacle is None:
            self.get_logger().warn("No obstacle detected.")
            return None

        elif current_obstacle is not None and current_obstacle.is_localized:
            return current_obstacle
            
        else:
            detected_obstacles = self.get_obstacles_from_camera(point_data)

            if detected_obstacles:
                # sort by Euclidean distance to the LiDAR
                detected_obstacles.sort(key=lambda x: math.hypot(x[0], x[1]))
                closest_x, closest_y, closest_color = detected_obstacles[0]

                if side_hnf is None:
                    return current_obstacle
                    
                obst_to_front_wall_dist = abs(closest_x * nx_f + closest_y * ny_f - dist_to_front)
                
                if obst_to_front_wall_dist < 0.85:
                    self.get_logger().info(f"Obst too close to the front wall")
                    return current_obstacle
                
                obst_to_outer_wall_dist = self.get_obstacle_to_wall_distance(closest_x, closest_y, side_hnf)
                
                current_segment = self.turn_count % 4
                zone_id = self.calculate_zone_id(obst_to_front_wall_dist, obst_to_outer_wall_dist)

                if self.is_start_finish_straight and zone_id is not None:
                    if zone_id % 10 == 1:
                        zone_id -= 1  # force the ID onto the inner lane
                        self.get_logger().info(f"set_obst_pos: zone corrected to the inner lane (new ID: {zone_id})")

                    if self.direction == 'left' and zone_id == 20:
                        self.get_logger().warn(f"Obstacle should be set to 20, but that is not possible")
                        return current_obstacle
                    elif self.direction == 'right' and zone_id == 10:
                        self.get_logger().warn(f"Obstacle should be set to 10, but that is not possible")
                        return current_obstacle

                if zone_id is not None:
                    if current_obstacle is not None and current_obstacle.prediction and closest_color != current_obstacle.color:
                        self.get_logger().error(f"!!! COLOUR MISMATCH: predicted {current_obstacle.color.upper()} != detected {closest_color.upper()} -> PANIC")
                        self.panic_close_obstacle = True

                    # stores the zone in the object
                    current_obstacle = Obstacle(closest_color, zone_id, False)
                    # store in the robot's memory
                    self.obstacle_memory[current_segment] = current_obstacle
                    side_dist_str = f"{obst_to_outer_wall_dist:.2f}m" if obst_to_outer_wall_dist else "N/A"
                    self.get_logger().warn(
                        f"+++ OBSTACLE LOCKED: {closest_color.upper()} +++\n"
                        f" -> Segment: {current_segment} | Zone: {zone_id}\n"
                        f" -> to front wall: {obst_to_front_wall_dist:.2f}m\n"
                        f" -> to side wall: {side_dist_str}"
                    )
                return current_obstacle

    def get_obstacle_to_wall_distance(self, obstacle_x, obstacle_y, hnf_wall):
        """
        Computes the orthogonal distance of a point (obstacle) to a wall (HNF).
        """
        if hnf_wall is None:
            return None
            
        nx, ny, d = hnf_wall
        
        # HNF distance formula: D = |x*nx + y*ny - d|
        distance = abs(obstacle_x * nx + obstacle_y * ny - d)
        
        return distance
                
    def check_undetected_turn(self, front_wall_hnf):
        total_turned = abs(self.current_yaw - self.yaw_offset)
        min_total_rotation = self.target_turns * 89.0
        if self.turn_count >= self.target_turns and total_turned >= min_total_rotation:
            if front_wall_hnf is not None:
                closest_f = front_wall_hnf[2]
                if closest_f < 1.70:
                        self.state = 'PARKING'
                
        if abs(self.start_straight_yaw - self.current_yaw) > 75.0:
            self.get_logger().warn(f">>> GYRO TURN DETECTED! Turned too far on the straight (turned: {abs(self.start_straight_yaw - self.current_yaw):.1f} deg) <<<")
            self.start_straight_yaw = self.current_yaw
            self.turn_count += 1

    def evaluate_steering_straight(self, inner_wall_hnf, outer_wall_hnf):
        target_x, target_y = self.get_target_point_straight(inner_wall_hnf, outer_wall_hnf)

        error = -target_x
        self.get_logger().info(f"SteeringError: {error:.2f}")
        
        # compute integral
        self.integral_error += error
        self.integral_error = max(-1.0, min(1.0, self.integral_error))
        
        # compute derivative
        derivative = error - self.prev_error
        self.prev_error = error
        
        # compute the control output
        steering_cmd = (self.kp * error) + (self.ki * self.integral_error) + (self.kd * derivative)
        
        steering_cmd = max(-1.0, min(1.0, steering_cmd))

        return steering_cmd

    def evaluate_steering_straight_parking(self, left_wall_hnf, right_wall_hnf):
        target_x, target_y = self.parking_target_point(left_wall_hnf, right_wall_hnf)

        error = -target_x
        self.get_logger().warn(f"SteeringError: {error:.2f}")
        
        # compute integral
        self.integral_error += error
        self.integral_error = max(-1.0, min(1.0, self.integral_error))
        
        # compute derivative
        derivative = error - self.prev_error
        self.prev_error = error
        
        # compute the control output
        steering_cmd = (self.kp * error) + (self.ki * self.integral_error) + (self.kd * derivative)
        
        steering_cmd = max(-1.0, min(1.0, steering_cmd))

        return steering_cmd
    
    def evaluate_steering_gyro(self):
        error = self.start_straight_yaw - self.current_yaw
        
        self.integral_gyro_error += error
        self.integral_gyro_error = max(-20.0, min(20.0, self.integral_gyro_error))
        
        derivative = error - self.prev_gyro_error
        self.prev_gyro_error = error
        
        steering_cmd = (self.kp_gyro * error) + (self.ki_gyro * self.integral_gyro_error) + (self.kd_gyro * derivative)

        self.get_logger().info(f"Error: {error}")

        return max(-1.0, min(1.0, steering_cmd))
    
    def execute_init(self):
        self.get_logger().info("Starting the robot...")
        with self.data_lock:
            if len(self.latest_yolo_results) == 0:
                self.get_logger().info("Waiting for the first camera frame and YOLO inference...")
                return
        
        if not self.imu_ready:
            self.get_logger().info("Waiting for the gyroscope to boot...")
            return

        self.set_led(True)
        self.state = 'STARTING'

    def execute_start(self, point_data):
        if self.button_start:
            if not self.button_state:
                return

            
        else:
            pass

        self.set_led(False)        # switch off the LED as confirmation
        
        self.yaw_offset = self.current_yaw
        self.start_straight_yaw = self.current_yaw

        self.get_logger().info("Evaluating the driving direction and unparking")
            
        if self.with_unpark:
            if self.direction is None:
                dist_left_point = self.get_closest_measure(point_data, 270.0)
                dist_right_point = self.get_closest_measure(point_data, 90.0)

                if dist_right_point is not None and dist_left_point is not None:
                    dist_right = dist_right_point[3]
                    dist_left = dist_left_point[3]
                else: 
                    self.get_logger().info("Error: at least one side returns no values")
                    return

                self.get_logger().info(f"Scanning track... distance left: {dist_left:.2f}m, length right: {dist_right:.2f}m")
                if abs(dist_left - dist_right) > 0.50:
                    if dist_left < dist_right:        
                        self.direction = 'right' # right wall is shorter = inner wall = driving direction is right!
                        self.get_logger().info(">>> LOCK: DIRECTION RIGHT (clockwise) <<<")
                    else:
                        self.direction = 'left'  # left wall is shorter = inner wall = driving direction is left!
                        self.get_logger().info(">>> LOCK: DIRECTION LEFT (counter-clockwise) <<<")
                else:
                    self.get_logger().info("Error! No wall is longer than the other!")
                    return
            self.state = 'PARKING_OUT'
            self.button_state = False

        else:
            self.get_logger().info("Starting the robot... Evaluating the driving direction and calibrating the gyro.")
            all_clusters = self.get_all_clusters_sorted(point_data)
            validated_clusters = self.validate_clusters_straight(all_clusters)
            merged_validated_clusters, _ = self.merge_clusters(all_clusters, validated_clusters)
            self.right_wall = merged_validated_clusters[0]
            self.front_wall = merged_validated_clusters[1]
            self.left_wall  = merged_validated_clusters[2]
            right_wall_hnf = self.cluster_to_hnf(self.right_wall)
            front_wall_hnf = self.cluster_to_hnf(self.front_wall)
            left_wall_hnf = self.cluster_to_hnf(self.left_wall)

            self.current_obstacle_cmd, _ = self.check_for_obstacle_color(point_data, self.turn_count, 0.0, 0.8)

            if self.direction is None:
                if (self.left_wall is not None and len(self.left_wall) > 0) and (self.right_wall is not None and len(self.right_wall) > 0):
                        # compute the Euclidean length
                        left_len = math.hypot(self.left_wall[0][1] - self.left_wall[-1][1], self.left_wall[0][2] - self.left_wall[-1][2])
                        right_len = math.hypot(self.right_wall[0][1] - self.right_wall[-1][1], self.right_wall[0][2] - self.right_wall[-1][2])
                        
                        self.get_logger().info(f"Scanning track... length left: {left_len:.2f}m, length right: {right_len:.2f}m")
                        
                        # needs a clear difference to be sure
                        if left_len > right_len + 0.30:
                            self.direction = 'right' # right wall is shorter = inner wall = driving direction is right!
                            self.get_logger().info(">>> LOCK: DIRECTION RIGHT (clockwise) <<<")
                        elif right_len > left_len + 0.30:
                            self.direction = 'left'  # left wall is shorter = inner wall = driving direction is left!
                            self.get_logger().info(">>> LOCK: DIRECTION LEFT (counter-clockwise) <<<")
                        else:    
                            self.get_logger().info("Error! No wall is longer than the")
                            return
                else:
                    self.get_logger().info("Driving direction not detected yet... Waiting for both side walls for the analysis.")
                    return
            self.state = 'FOLLOW_LANE'
            self.button_state = False

    def handle_lane_following(self, point_data):
        cmd = Twist()

        if self.panic_close_obstacle:
            self.get_logger().error("Close obstacle -> starting panic recovery (straight).")
            self.trigger_panic_recovery(increment_turn=False)
            return

        all_clusters = self.get_all_clusters_sorted(point_data)

        self.get_logger().info(f"Following the lane... current yaw: {self.current_yaw:.1f} deg, start yaw: {self.start_straight_yaw:.1f} deg, turned since start: {abs(self.current_yaw - self.start_straight_yaw):.1f} deg")

        if self.parking_straight:
            validated_clusters = self.validate_clusters_far_front_wall(all_clusters)
        else:
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

        self.visualize_hnf_line(front_wall_hnf, m_id=1, colour_name="rgb_red", label="Front HNF")
        self.visualize_hnf_line(left_wall_hnf, m_id=0, colour_name="blue", label="Left HNF")
        self.visualize_hnf_line(right_wall_hnf, m_id=2, colour_name="rgb_green", label="Right HNF")

        if self.direction == 'left':
            inner_wall = self.left_wall
            inner_wall_hnf = left_wall_hnf
            outer_wall = self.right_wall
            outer_wall_hnf = right_wall_hnf
        else:
            inner_wall = self.right_wall
            inner_wall_hnf = right_wall_hnf
            outer_wall = self.left_wall 
            outer_wall_hnf = left_wall_hnf

        current_straight = self.turn_count % 4
        current_obstacle = self.obstacle_memory[current_straight]

        if self.turn_count != 0 or self.direction == 'right':      # so that we do not detect false obstacles right after unparking
            if current_obstacle is None or not current_obstacle.is_localized:
                result = self.set_obstacle_position(point_data, (front_wall_hnf, outer_wall_hnf))
                if result is not None:
                    current_obstacle = result
                elif current_obstacle is not None:
                    self.get_logger().info(
                        f"No front_wall: obstacle kept from memory ({current_obstacle.color}, "
                        f"localized={current_obstacle.is_localized})"
                    )

            if current_obstacle is None:
                if front_wall_hnf is not None:
                    self.current_obstacle_cmd, current_obstacle = self.check_for_obstacle_color(point_data, self.turn_count, 0.0, front_wall_hnf[2] - 0.75)
                else:
                    self.current_obstacle_cmd = None  # Fix D: no context -> reset
            else:
                self.current_obstacle_cmd = current_obstacle.color

        front_dist = front_wall_hnf[2] if front_wall_hnf is not None else None

        # Fix C: ghost trigger -- obstacle lost from the camera but still in memory
        if self.current_obstacle_cmd is None and current_obstacle is None and front_dist is not None:
            stored = self.obstacle_memory[current_straight]
            if stored is not None and 1.40 < front_dist < 1.60:
                stored.prediction = True
                self.current_obstacle_cmd = stored.color
                current_obstacle = stored
                self.get_logger().info("Ghost trigger: obstacle lost at 1.5m -> prediction set!")
        self.lane_ratio = self.set_lane_ratio_for_obstacle_cmd(self.current_obstacle_cmd, current_obstacle, front_dist)
        self.get_logger().info(f"Current_Obst_Cmd: {self.current_obstacle_cmd}, Current_Obst: {current_obstacle} , Lane_Ratio: {self.lane_ratio}, front_dist: {front_dist}m")

        if front_wall_hnf is not None and outer_wall_hnf is not None:
            self.check_for_obstacle_color(point_data, (self.turn_count + 1), front_wall_hnf[2] - 0.75, 2.0)

        if front_wall_hnf is not None and self.direction is not None:
            _, _, front_dist = front_wall_hnf
            
            if front_dist < 1.20:
                self.state = f"TURN_{self.direction.upper()}"
                self.get_logger().warn(f">>> {self.state} INITIATED<<<")
                self.get_logger().info(f"Distance to the front wall: {front_dist:.2f}m")
                return
            else:
                if front_dist < 1.40:
                    self.get_logger().info(f"Waiting for the corner... (front wall is still {front_dist:.2f}m away)")
        
        steering_cmd = self.evaluate_steering_straight(inner_wall_hnf, outer_wall_hnf)
        
        cmd.linear.x = self.base_speed
        cmd.angular.z = float(steering_cmd)
        self.pub_cmd_vel.publish(cmd)
        self.get_logger().info(f"Steering: Speed={self.base_speed:.3f}, Steering={steering_cmd:.3f}")

    def handle_turn_maneuver(self, point_data):
        cmd = Twist()

        # The close-obstacle panic only applies on straights (FOLLOW_LANE). In the turn
        # the original panic (55-80 deg in EXECUTE) stays solely in charge.
        self.panic_close_obstacle = False
        
        if self.turn_phase == 'APPROACH':
            self.front_wall = self.track_front_wall(point_data, self.front_wall)
            front_wall_hnf = self.cluster_to_hnf(self.front_wall)
            self.visualize_hnf_line(front_wall_hnf, m_id=1, colour_name="rgb_red", label="Front HNF")
            if front_wall_hnf is not None:
                allowed_obst_dist = max(0.25, front_wall_hnf[2] - 0.25)
            else:
                allowed_obst_dist = 0.25
            exit_obstacle_cmd, exit_obstacle = self.check_for_obstacle_color(point_data, (self.turn_count + 1), allowed_obst_dist, 2.0)
            self.lane_ratio_exit = self.set_lane_ratio_for_obstacle_cmd(exit_obstacle_cmd, exit_obstacle, 2.0, is_turn_exit=True, apply_state=False)
            self.get_logger().info(f"Planned exit: Obst={exit_obstacle_cmd}, Ratio={self.lane_ratio_exit:.2f}")
            
            validated_clusters = self.validate_clusters_turn(self.front_wall, point_data)
            
            right_wall_hnf = self.cluster_to_hnf(validated_clusters[0])
            left_wall_hnf = self.cluster_to_hnf(validated_clusters[2])
            
            if self.direction == 'left':
                inner_wall_hnf = left_wall_hnf
                outer_wall_hnf = right_wall_hnf
            else:
                inner_wall_hnf = right_wall_hnf
                outer_wall_hnf = left_wall_hnf

            # compute steering for the approach
            self.get_logger().info(f"InnerWallHNF: {inner_wall_hnf}, OuterWallHNF: {outer_wall_hnf}")
            steering_cmd = self.evaluate_steering_straight(inner_wall_hnf, outer_wall_hnf)
            
            # compute the turn geometry
            front_wall_params, side_wall_params = self.extract_wall_lines(validated_clusters)
            target_line_params, max_allowed_radius = self.test_calculate_target_line(validated_clusters, self.lane_ratio_exit)
            intersection_x, intersection_y, intersection_angle = self.test_get_intersection_point(target_line_params)
            curve_radius_m, entry_distance_m = self.test_calculate_curve_geometry(intersection_y, intersection_angle, max_allowed_radius)
            
            # check the trigger
            if self.test_check_turn_trigger(entry_distance_m):
                self.get_logger().info(f"Trigger reached. Switching to EXECUTE phase. Distance to the wall: {front_wall_hnf[2]:.2f}m, radius: {curve_radius_m:.2f}m")
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
            
            # steer during the approach
            cmd.linear.x = self.turn_speed
            cmd.angular.z = float(steering_cmd)
            self.pub_cmd_vel.publish(cmd)
                
        elif self.turn_phase == 'EXECUTE':          
            if self.current_obstacle_cmd != self.base_obst_cmd:
                if self.current_obstacle_cmd is not None:
                    is_left = (self.direction == "left")
                    needs_inner = (self.current_obstacle_cmd == "green" and is_left) or (self.current_obstacle_cmd == "red" and not is_left)
                    min_r = self.MIN_TURN_RADIUS_M
                    max_r = self.base_entry_distance - 0.15
                    
                    if needs_inner:
                        self.saved_curve_radius_m = min_r
                    else:
                        self.saved_curve_radius_m = min(max_r, self.MAX_KINEMATIC_RADIUS_M)
                    
                    self.get_logger().warn(f"MID-TURN DODGE!. Radius: {self.saved_curve_radius_m:.2f}m")

            # execute and track
            self.execute_turn(self.saved_curve_radius_m)
            self.front_wall = self.track_front_wall(point_data, self.front_wall)
            
            if self.front_wall is not None:
                front_wall_params = self.cluster_to_hnf(self.front_wall)
            else:
                front_wall_params = None
            
            # gyro or wall parallelism
            turn_completed = self.test_check_turn_completion_fused(self.saved_intersection_angle, front_wall_params)
            
            # Panic-Exit
            yaw_diff = (self.current_yaw - self.start_turn_yaw + 180) % 360 - 180
            progressed_angle = abs(yaw_diff)

            if 80.0 > progressed_angle > 55.0:

                # only obstacles standing right at the start of the new straight (0.0 to 0.7m)
                last_exit_ratio = self.lane_ratio_exit
                exit_obstacle_cmd, exit_obstacle = self.check_for_obstacle_color(point_data, (self.turn_count + 1), 0.0, 0.7)
                self.lane_ratio_exit = self.set_lane_ratio_for_obstacle_cmd(exit_obstacle_cmd, exit_obstacle, 2.0, is_turn_exit=True, apply_state=False)

                if exit_obstacle is not None and exit_obstacle_cmd != "CLEAR":
                    if abs(last_exit_ratio - self.lane_ratio_exit) > 0.15:
                        self.get_logger().warn(f"PANIC EXIT. Critical obstacle ({exit_obstacle_cmd}) forces an abort at {progressed_angle:.1f} deg!")
                        self.trigger_panic_recovery(increment_turn=True)
                        return

            # finish the turn normally
            if turn_completed:
                self.get_logger().info("Turn finished normally. Handing over to the lane follower.")

                cmd.linear.x = self.base_speed
                cmd.angular.z = 0.0
                self.pub_cmd_vel.publish(cmd)

                self.lane_ratio = self.lane_ratio_exit

                self.turn_phase = 'APPROACH' # reset for the next turn

                self.turn_count += 1

                if self.turn_count == self.target_turns:
                    self.state = 'PARKING'
                else:
                    self.state = 'FOLLOW_LANE'

                # zero the PID values for the next straight
                self.prev_error = 0.0
                self.integral_error = 0.0

                self.start_straight_yaw = self.current_yaw
                self.is_obstacle_passed = False


    def estimate_new_straight_yaw(self, point_data):
        """
        Looks for two roughly parallel walls with different x signs
        (left + right wall of the NEW straight) and derives the target yaw from them.

        Geometry: validate_clusters_straight assigns side walls via
        local_angle ~ delta_yaw (= current_yaw - start_straight_yaw).
        So the measured side wall angle IS the desired delta_yaw
        -> start_straight_yaw = current_yaw - wall_angle.

        Returns the reference yaw, or None if no two walls were found.
        """
        clusters = self.get_all_clusters_sorted(point_data)
        if not clusters:
            return None

        left = []   # mean_x < 0  (left)
        right = []  # mean_x > 0  (right)
        for c in clusters:
            if len(c) < 15:
                continue
            angle = self.get_cluster_angle(c)
            if angle is None:
                continue
            # side walls run along the driving direction -> |angle| small
            if abs(angle) > 45.0:
                continue
            mean_x = sum(p[1] for p in c) / len(c)
            if abs(mean_x) > 1.0:
                continue
            if mean_x < 0:
                left.append((angle, mean_x))
            else:
                right.append((angle, mean_x))

        if not left or not right:
            return None

        # clusters are sorted by physical length -> [0] = longest wall
        left_angle = left[0][0]
        right_angle = right[0][0]

        # check parallelism
        if abs(left_angle - right_angle) > 15.0:
            self.get_logger().warn(
                f"PANIC-RECOVERY: walls not parallel (L={left_angle:.1f} deg, R={right_angle:.1f} deg)."
            )
            return None

        wall_angle = (left_angle + right_angle) / 2.0
        return self.current_yaw - wall_angle

    def trigger_panic_recovery(self, increment_turn):
        cmd = Twist()
        cmd.linear.x = 0.0
        cmd.angular.z = 0.0
        self.pub_cmd_vel.publish(cmd)
        self.state = 'PANIC_RECOVERY'
        self.panic_phase = 'STOP'
        self.panic_timer = None
        self.panic_increment_turn = increment_turn
        self.panic_close_obstacle = False

    def handle_panic_recovery(self, point_data):
        cmd = Twist()
        now = time.monotonic()

        if self.panic_phase == 'STOP':
            if self.panic_timer is None:
                self.panic_timer = now + self.panic_stop_duration

            cmd.linear.x = 0.0
            cmd.angular.z = 0.0
            self.pub_cmd_vel.publish(cmd)

            # keep looking for the walls during the stop, keep the best one
            est = self.estimate_new_straight_yaw(point_data)
            if est is not None:
                self.panic_straight_yaw_est = est

            if now >= self.panic_timer:
                if self.panic_straight_yaw_est is not None:
                    self.start_straight_yaw = self.panic_straight_yaw_est
                    self.get_logger().warn(
                        f"PANIC-RECOVERY: new straight from walls -> yaw_ref={self.start_straight_yaw:.1f} deg (current={self.current_yaw:.1f} deg)"
                    )
                elif self.panic_increment_turn:
                    # turn aborted without walls: planned turn angle + turn direction
                    yaw_diff = (self.current_yaw - self.start_turn_yaw + 180) % 360 - 180
                    turn_sign = 1.0 if yaw_diff >= 0 else -1.0
                    target = abs(self.saved_intersection_angle) if self.saved_intersection_angle is not None else 90.0
                    self.start_straight_yaw = self.start_turn_yaw + turn_sign * target
                    self.get_logger().warn(
                        f"PANIC-RECOVERY: no walls found -> fallback turn angle yaw_ref={self.start_straight_yaw:.1f} deg"
                    )
                else:
                    # close obstacle on a straight: keep the already aligned start_straight_yaw
                    self.get_logger().warn(
                        f"PANIC-RECOVERY: no walls -> keeping start_straight_yaw={self.start_straight_yaw:.1f} deg"
                    )

                # zero the PID for the new straight
                self.prev_error = 0.0
                self.integral_error = 0.0

                self.panic_phase = 'REVERSE'
                self.panic_timer = None
                self.panic_straight_yaw_est = None
            return

        if self.panic_phase == 'REVERSE':
            if self.panic_timer is None:
                self.panic_timer = now + self.panic_reverse_duration
                self.integral_gyro_error = 0.0
                self.prev_gyro_error = 0.0

            # reverse onto start_straight_yaw under gyro control.
            # reversing inverts the steering effect -> flip the sign.
            yaw_error = (self.start_straight_yaw - self.current_yaw + 180) % 360 - 180
            steering_cmd = -self.evaluate_steering_gyro()

            cmd.linear.x = -abs(self.panic_speed)
            cmd.angular.z = float(steering_cmd)
            self.pub_cmd_vel.publish(cmd)

            # DEBUG: check the computed straight yaw against the live wall measurement.
            # live_wall_yaw should be ~= target if the calculation is right.
            live_est = self.estimate_new_straight_yaw(point_data)
            live_str = f"{live_est:.1f}" if live_est is not None else "n/a"
            self.get_logger().warn(
                f"PANIC-REVERSE: target={self.start_straight_yaw:.1f} deg actual={self.current_yaw:.1f} deg "
                f"yaw_err={yaw_error:.1f} deg steer={steering_cmd:.2f} | live_wall_yaw={live_str}"
            )

            if now >= self.panic_timer:
                cmd.linear.x = 0.0
                cmd.angular.z = 0.0
                self.pub_cmd_vel.publish(cmd)

                self.lane_ratio = self.lane_ratio_exit
                self.turn_phase = 'APPROACH'
                if self.panic_increment_turn:
                    self.turn_count += 1
                self.is_obstacle_passed = False
                self.panic_phase = None
                self.panic_timer = None

                if self.turn_count == self.target_turns:
                    self.state = 'PARKING'
                else:
                    self.state = 'FOLLOW_LANE'

                self.get_logger().warn(
                    f"PANIC-RECOVERY finished. State={self.state}, TurnCount={self.turn_count}"
                )
            return

    def update_timer(self):
        now = self.get_clock().now()

        if self.state == 'PARKING' and self.parking_phase == 'STOP_AFTER_OBST_RUN' and self.timer_active:
            self.timer_active = False
            self.get_logger().info(f"Timer stopped! Final time: {self.elapsed_time:.2f}s")

        if self.timer_active and self.start_time_stamp is not None:
            diff = now - self.start_time_stamp
            self.elapsed_time = diff.nanoseconds / 1e9
            
            timer_msg = Float64()
            timer_msg.data = self.elapsed_time
            self.pub_timer.publish(timer_msg)

    # =========================================================
    # Parking functions
    # =========================================================

    def observe_sorroundings_while_waiting(self, point_data):
        all_clusters = self.get_all_clusters_sorted(point_data)

        self.get_logger().info("Observing the surroundings while waiting 3 seconds.")

        validated_clusters = self.validate_clusters_straight(all_clusters)
        merged_validated_clusters, _ = self.merge_clusters(all_clusters, validated_clusters)
        self.right_wall = merged_validated_clusters[0]
        self.front_wall = merged_validated_clusters[1]
        self.left_wall  = merged_validated_clusters[2]
        right_wall_hnf = self.cluster_to_hnf(self.right_wall)
        front_wall_hnf = self.cluster_to_hnf(self.front_wall)
        left_wall_hnf = self.cluster_to_hnf(self.left_wall)

        self.visualize_hnf_line(front_wall_hnf, m_id=1, colour_name="rgb_red", label="Front HNF")
        self.visualize_hnf_line(left_wall_hnf, m_id=0, colour_name="blue", label="Left HNF")
        self.visualize_hnf_line(right_wall_hnf, m_id=2, colour_name="rgb_green", label="Right HNF")

    def positioning_before_parking(self, point_data):
        cmd = Twist()
        all_clusters = self.get_all_clusters_sorted(point_data)

        self.get_logger().info(f"Positioning myself for parking. Current yaw: {self.current_yaw:.1f} deg, start yaw: {self.start_straight_yaw:.1f} deg, turned since start: {abs(self.current_yaw - self.start_straight_yaw):.1f} deg")

        validated_clusters = self.validate_clusters_straight(all_clusters)
        merged_validated_clusters, _ = self.merge_clusters(all_clusters, validated_clusters)
        self.right_wall = merged_validated_clusters[0]
        self.front_wall = merged_validated_clusters[1]
        self.left_wall  = merged_validated_clusters[2]
        right_wall_hnf = self.cluster_to_hnf(self.right_wall)
        front_wall_hnf = self.cluster_to_hnf(self.front_wall)
        left_wall_hnf = self.cluster_to_hnf(self.left_wall)

        self.visualize_hnf_line(front_wall_hnf, m_id=1, colour_name="rgb_red", label="Front HNF")
        self.visualize_hnf_line(left_wall_hnf, m_id=0, colour_name="blue", label="Left HNF")
        self.visualize_hnf_line(right_wall_hnf, m_id=2, colour_name="rgb_green", label="Right HNF")

        if self.direction == 'left':
            inner_wall = self.left_wall
            inner_wall_hnf = left_wall_hnf
            outer_wall = self.right_wall
            outer_wall_hnf = right_wall_hnf
        else:
            inner_wall = self.right_wall
            inner_wall_hnf = right_wall_hnf
            outer_wall = self.left_wall 
            outer_wall_hnf = left_wall_hnf

        if front_wall_hnf is not None and self.direction is not None:
            _, _, front_dist = front_wall_hnf
            
            if front_dist < 1.50:
                self.parking_phase = 'TURN_PREPARATION'
                self.get_logger().warn(">>> PARK TURN_PREPARATION INITIATED<<<")
                self.get_logger().info(f"Distance to the front wall: {front_dist:.2f}m")
            else:
                if front_dist < 1.60:
                    self.get_logger().info(f"Waiting for the corner... (front wall is still {front_dist:.2f}m away)")
        
        if self.park_direction == 'PARKING_LEFT' and self.parking_left_with_obstacle:
            self.lane_ratio = 0.35
        else:
            self.lane_ratio = 0.25

        steering_cmd = self.evaluate_steering_straight(inner_wall_hnf, outer_wall_hnf)
        
        cmd.linear.x = self.parking_speed
        cmd.angular.z = float(steering_cmd)
        self.pub_cmd_vel.publish(cmd)
        self.get_logger().info(f"Steering: Speed={self.base_speed:.3f}, Steering={steering_cmd:.3f}")

    def handle_turn_preparation(self, point_data):
        if self.park_direction == 'PARKING_RIGHT_NORMAL':
            self.lane_ratio = 0.20
        elif self.park_direction == 'PARKING_LEFT' and self.parking_left_with_obstacle:
            self.lane_ratio = 0.35
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
        
        if self.direction == 'left':
            inner_wall_hnf = left_wall_hnf
            outer_wall_hnf = right_wall_hnf
        else:
            inner_wall_hnf = right_wall_hnf
            outer_wall_hnf = left_wall_hnf

        # compute steering for the approach
        self.get_logger().info(f"InnerWallHNF: {inner_wall_hnf}, OuterWallHNF: {outer_wall_hnf}")
        steering_cmd = self.evaluate_steering_straight(inner_wall_hnf, outer_wall_hnf)
        
        # compute the turn geometry
        front_wall_params, side_wall_params = self.extract_wall_lines(validated_clusters)
        if self.park_direction == 'PARKING_LEFT':
            target_line_params, max_allowed_radius = self.test_calculate_target_line(validated_clusters, -0.10)
        else:
            target_line_params, max_allowed_radius = self.test_calculate_target_line(validated_clusters, -0.87)
        intersection_x, intersection_y, intersection_angle = self.test_get_intersection_point(target_line_params)
        curve_radius_m, entry_distance_m = self.test_calculate_curve_geometry(intersection_y, intersection_angle, max_allowed_radius)
        
        # check the trigger
        if self.test_check_turn_trigger(entry_distance_m):
            self.get_logger().info(f"Trigger reached. Switching to EXECUTE phase. Distance to the wall: {front_wall_hnf[2]:.2f}m, radius: {curve_radius_m:.2f}m")
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
        
        # steer during the approach
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

    def parking_lidar_pid_steering(self, point_data):
        if self.park_direction == 'PARKING_RIGHT_NORMAL':
            self.lane_ratio = 1.13
        else:
            self.lane_ratio = 1.11

        self.kp = 2.6
        self.ki = 0.0
        self.kd = 0.10
        cmd = Twist()

        all_clusters = self.get_all_clusters_sorted(point_data)

        self.get_logger().info(f"Parking into the parking space. Current yaw: {self.current_yaw:.1f} deg, start yaw: {self.start_straight_yaw:.1f} deg, turned since start: {abs(self.current_yaw - self.start_straight_yaw):.1f} deg")

        validated_clusters = self.validate_clusters_parking(all_clusters)
        merged_validated_clusters, _ = self.merge_clusters(all_clusters, validated_clusters)
        self.right_wall = merged_validated_clusters[0]
        self.front_wall = merged_validated_clusters[1]
        self.left_wall  = merged_validated_clusters[2]
        right_wall_hnf = self.cluster_to_hnf(self.right_wall)
        front_wall_hnf = self.cluster_to_hnf(self.front_wall)
        left_wall_hnf = self.cluster_to_hnf(self.left_wall)

        self.visualize_hnf_line(front_wall_hnf, m_id=1, colour_name="rgb_red", label="Front HNF")
        self.visualize_hnf_line(left_wall_hnf, m_id=0, colour_name="blue", label="Left HNF")
        self.visualize_hnf_line(right_wall_hnf, m_id=2, colour_name="rgb_green", label="Right HNF")

        if front_wall_hnf is not None:
            _, _, front_dist = front_wall_hnf
        else:
            front_dist = None
        
        steering_cmd = self.evaluate_steering_straight_parking(left_wall_hnf, right_wall_hnf)
            
        """        if front_dist is not None and front_dist <= 0.35:
            self.parking_phase = 'RETIRE_THE_CAR'
            self.get_logger().warn("Handing over to the IMU PID!")"""

        if not hasattr(self, 'park_timer'):
            self.park_timer = time.time() + 6.0
        if time.time() >= self.park_timer:
            self.state = 'STOPPED'
            return
        else:
            cmd.linear.x = self.parking_speed
            cmd.angular.z = float(steering_cmd)
            self.pub_cmd_vel.publish(cmd)
            self.get_logger().info(f"Steering: Speed={self.base_speed:.3f}, Steering={steering_cmd:.3f}")

    def parking_imu_pid_steering(self):
        if not hasattr(self, 'park_timer'):
            self.park_timer = time.time() + 5.0
        if time.time() < self.park_timer:
            cmd = Twist()
            cmd.linear.x = self.parking_speed
            cmd.angular.z = self.evaluate_steering_gyro()
            self.pub_cmd_vel.publish(cmd)
        else:
            self.state = 'STOPPED'
            self.get_logger().info("PARKING FINISHED!")
            self.execute_stop()
            return

    def parking_pid_steering_reverse(self, point_data):
        self.lane_ratio = 1.13
        cmd = Twist()

        self.get_logger().info(f"Parking into the parking space. Current yaw: {self.current_yaw:.1f} deg, start yaw: {self.start_straight_yaw:.1f} deg, turned since start: {abs(self.current_yaw - self.start_straight_yaw):.1f} deg")
        
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

        self.visualize_hnf_line(front_wall_hnf, m_id=1, colour_name="rgb_red", label="Front HNF")
        self.visualize_hnf_line(left_wall_hnf, m_id=0, colour_name="blue", label="Left HNF")
        self.visualize_hnf_line(right_wall_hnf, m_id=2, colour_name="rgb_green", label="Right HNF")

        if front_dist < 0.80:
            if self.direction == 'left':
                inner_wall = self.left_wall
                inner_wall_hnf = left_wall_hnf
                outer_wall = self.right_wall
                outer_wall_hnf = right_wall_hnf
            else:
                inner_wall = self.right_wall
                inner_wall_hnf = right_wall_hnf
                outer_wall = self.left_wall 
                outer_wall_hnf = left_wall_hnf

            if front_wall_hnf is not None and self.direction is not None:
                _, _, front_dist = front_wall_hnf
            
            steering_cmd = self.evaluate_steering_straight(inner_wall_hnf, outer_wall_hnf)
            
            # SET COMMANDS FOR THE ESP
            cmd.linear.x = - self.parking_speed
            cmd.angular.z = - float(steering_cmd)
            self.pub_cmd_vel.publish(cmd)
            self.get_logger().info(f"Steering: Speed={self.base_speed:.3f}, Steering={steering_cmd:.3f}")
        else:
            self.parking_phase = 'ADDRESSING_PARKING_SPACE'

    def parking_target_point(self, hnf_inner, hnf_outer):
        target_y = self.lookahead_dist_parking
        target_x = 0.0
        
        if hnf_inner is not None and (0.30 > hnf_inner[2] or hnf_inner[2] > 3.5):
            hnf_inner = None
        if hnf_outer is not None and (0.30 > hnf_outer[2] or hnf_outer[2] > 3.5):
            hnf_outer = None

        def get_x_at_y(hnf_params, y_val):
            nx, ny, d = hnf_params
            if abs(nx) < 1e-6: return 0.0
            return (d - ny * y_val) / nx

        x_inner = get_x_at_y(hnf_inner, target_y) if hnf_inner else None
        x_outer = get_x_at_y(hnf_outer, target_y) if hnf_outer else None

        if x_outer is not None:
            if x_outer > 0: 
                target_x = x_outer - (3.0 - self.lane_ratio)
            else:            
                target_x = x_outer + self.lane_ratio
                
        elif x_inner is not None:
            target_x = 0.0

        else:
            target_x = 0.0

        if target_x >= 0.20:    # capped against impossible values
            target_x = 0

        self.get_logger().info(f"Target_Point: {target_x} , {target_y}")

        return (target_x, target_y)
    
    def validate_clusters_parking(self, clusters):
        u_profile = [None, None, None]
        if not clusters:
            return u_profile

        delta_yaw = self.current_yaw - self.start_straight_yaw

        while delta_yaw > 180: delta_yaw -= 360
        while delta_yaw < -180: delta_yaw += 360

        right_candidates = []
        left_candidates = []
        front_candidates = []

        for c in clusters:
            if len(c) < 20:
                #self.get_logger().info(f"Cluster has too few points")
                continue
            
            local_angle = self.get_cluster_angle(c)
            if local_angle is None: 
                continue

            shifted_angle = local_angle - delta_yaw
            
            angle_norm = abs(shifted_angle) % 180
            if angle_norm > 90:
                angle_norm = 180 - angle_norm

            mean_x_local = sum(p[1] for p in c) / len(c)
            mean_y_local = sum(p[2] for p in c) / len(c)
            
            rad = math.radians(delta_yaw)
            mean_x_straight = mean_x_local * math.cos(rad) - mean_y_local * math.sin(rad)
            mean_y_straight = mean_x_local * math.sin(rad) + mean_y_local * math.cos(rad)

            if angle_norm <= 45.0:
                if 0.30 <= abs(mean_x_straight) <= 3.5:
                    if mean_x_straight > 0:
                        right_candidates.append(c)
                    else:
                        left_candidates.append(c)
                else:
                    self.get_logger().info("Cluster has too large an x value")
                        
            else:
                if mean_y_straight >= 0.10:
                    front_candidates.append(c)

        # ASSIGN SIDE WALLS
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
                self.get_logger().info(f"Real track width: {track_width:.2f}m, L: {abs(left_dist):.2f}m, R: {abs(right_dist):.2f}m")
                
                if 2.0 <= track_width <= 4.0:
                    u_profile[0] = best_right
                    u_profile[2] = best_left
                else:
                    self.get_logger().warn(f"Track width implausible ({track_width:.2f}m). Dropping the shorter wall.")
                    len_r = math.hypot(best_right[-1][1] - best_right[0][1], best_right[-1][2] - best_right[0][2])
                    len_l = math.hypot(best_left[-1][1] - best_left[0][1], best_left[-1][2] - best_left[0][2])
                    
                    if len_r > len_l:
                        u_profile[0] = best_right
                    else:
                        u_profile[2] = best_left
            else:
                self.get_logger().warn("HNF calculation failed for one of the walls.")
                return [None, None, None]
                    
        elif best_right is not None:
            u_profile[0] = best_right
        elif best_left is not None:
            u_profile[2] = best_left

        # ASSIGN FRONT WALL
        if front_candidates:
            groups = []
            for c in front_candidates:
                mean_y = sum(p[2] for p in c) / len(c)
                
                # drop everything that is extremely close
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
                mean_y = g['base_y']  # distance to the wall

                is_far_wall = mean_y > 0.70
                min_allowed_width = 0.50 if self.is_start_finish_straight else 0.35
                
                is_blocking_path = (min_x < -0.15) and (max_x > 0.15)
                
                if total_width > min_allowed_width and (is_far_wall or is_blocking_path):
                    if not is_far_wall and is_blocking_path:
                        self.get_logger().warn(f"Found a close but blocking wall. Distance: {mean_y:.2f}m")
                    valid_wall_groups.append(g)

            if valid_wall_groups:
                valid_wall_groups.sort(key=lambda g: g['base_y'])
                winner_group = valid_wall_groups[0]['clusters']
                winner_group.sort(key=len, reverse=True)
                u_profile[1] = winner_group[0]
            else:
                u_profile[1] = None
                if front_candidates:
                    self.get_logger().debug("Front candidates present, but rejected as phantom walls.")

        return u_profile
    
    def park(self):
        if not hasattr(self, 'park_timer'):
            self.park_timer = time.time() + 3.0
            
        cmd = Twist()
        if time.time() < self.park_timer:
            cmd.linear.x = float(self.parking_speed)
            cmd.angular.z = 0.0
            self.pub_cmd_vel.publish(cmd)
        else:
            self.execute_stop()
            self.get_logger().info("PARKING FINISHED!")
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
        
        if self.direction == 'left':
            inner_wall_hnf = left_wall_hnf
            outer_wall_hnf = right_wall_hnf
        else:
            inner_wall_hnf = right_wall_hnf
            outer_wall_hnf = left_wall_hnf

        self.get_logger().info(f"InnerWallHNF: {inner_wall_hnf}, OuterWallHNF: {outer_wall_hnf}")

        n_xf, n_yf, d_f = front_wall_hnf
        d_target = 0.20
        target_line_params = (n_xf, n_yf, d_target)

        _, _, intersection_angle = self.test_get_intersection_point(target_line_params)

        curve_radius_m = 0.20
        entry_distance_m = 0.0

        self.get_logger().info(f"Waiting for the 3s to run out, then turning off in reverse")
        self.saved_intersection_angle = intersection_angle
        self.saved_curve_radius_m = curve_radius_m
        self.start_turn_yaw = self.current_yaw
        self.base_entry_distance = entry_distance_m
        
        cmd.linear.x = 0.0
        cmd.angular.z = 0.0
        self.pub_cmd_vel.publish(cmd)
    
    def decide_park_direction(self, point_data):
        if self.park_direction is None:
            if self.direction == 'left':
                self.park_direction = 'PARKING_LEFT'
                self.parking_phase = 'POSITIONING_FOR_STOP'
                self.park_turn_direction = 'right'
                if self.obstacle_memory[0] is not None and (not self.obstacle_memory[0].is_localized or self.obstacle_memory[0].prediction or self.obstacle_memory[0].zone_id < 2) and self.obstacle_memory[0].color == 'red':
                    self.parking_left_with_obstacle = True
                else:
                    self.parking_left_with_obstacle = False

            elif self.obstacle_memory[0] is not None and (not self.obstacle_memory[0].is_localized or self.obstacle_memory[0].prediction or self.obstacle_memory[0].zone_id < 2) and self.obstacle_memory[0].color == 'green':
                self.park_direction = 'PARKING_RIGHT_OBST'
                self.parking_phase = 'POSITIONING_FOR_STOP'
                self.park_turn_direction = 'left'

            else:
                self.park_direction = 'PARKING_RIGHT_NORMAL'
                self.parking_phase = 'APPROACH_TURN'
                self.park_turn_direction = 'left'

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
            self.parking_lidar_pid_steering(point_data)

        elif self.parking_phase == 'RETIRE_THE_CAR':
            self.parking_imu_pid_steering()

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
            self.parking_lidar_pid_steering(point_data)

        elif self.parking_phase == 'RETIRE_THE_CAR':
            self.parking_imu_pid_steering()

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
            self.parking_lidar_pid_steering(point_data)

        elif self.parking_phase == 'RETIRE_THE_CAR':
            self.parking_imu_pid_steering()

    def execute_stop(self):
            # stop the robot (in every cycle, as long as we are in the STOPPED state)
            cmd = Twist()
            cmd.linear.x = 0.0
            cmd.angular.z = 0.0
            self.pub_cmd_vel.publish(cmd)

            # print the final banner only once, then shut down cleanly
            if not getattr(self, '_goal_reached_logged', False):
                self._goal_reached_logged = True
                self.get_logger().info("")
                self.get_logger().info("  +==============================================+")
                self.get_logger().info("  |              [OK] GOAL REACHED               |")
                self.get_logger().info(f"  |   {self.turn_count:>3} turns mastered cleanly.                |")
                self.get_logger().info("  |   Stopping the robot.                        |")
                self.get_logger().info("  +----------------------------------------------+")
                self.get_logger().info(f"  Detected obstacles: {self.obstacle_memory}")
                self.get_logger().info("")

                # trigger the shutdown only once; main() cleans up the node.
                if rclpy.ok():
                    rclpy.shutdown()
            return

    def main_logic(self, point_data):
        if self.counter == 15:
            self.counter = 0
            self.clear_all_lines()
        self.get_logger().info(f"Current state: {self.state}, current Parking_Phase: {self.parking_phase}, TurnCount: {self.turn_count}, Lane_Ratio: {self.lane_ratio}, EXIT_Ratio: {self.lane_ratio_exit}")

        if self.state == 'INITIALIZING':
            self.execute_init()
        
        elif self.state == 'STARTING':
            self.execute_start(point_data)

        elif self.state == 'PARKING_OUT':
            self.handle_unpark(point_data)

        elif self.state == 'FOLLOW_LANE':
            self.handle_lane_following(point_data)
            
        elif self.state in ['TURN_LEFT', 'TURN_RIGHT']:
            self.handle_turn_maneuver(point_data)

        elif self.state == 'PANIC_RECOVERY':
            self.handle_panic_recovery(point_data)

        elif self.state == 'STOPPED':
            self.execute_stop()

        elif self.state == 'PARKING':
            self.decide_park_direction(point_data)

        self.counter += 1
        self.update_timer()
        
def main(args=None):
    rclpy.init(args=args)
    node = Obstacle_Run()
    
    # manages the parallel callback groups
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)
    
    try:
        # starts all threads in parallel
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        # node cleanup; rclpy.shutdown() may already have happened in execute_stop()
        if node.context.ok():
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()