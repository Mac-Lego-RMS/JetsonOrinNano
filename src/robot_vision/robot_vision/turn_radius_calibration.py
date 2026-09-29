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
      YOUR HARDWARE COORDINATE SYSTEM (Lidar & Foxglove)
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
MATHS RULES FOR THIS NODE:
- To move the carrot further to the FRONT  -> Y gets larger (+Y)
- To move the carrot further to the RIGHT  -> X gets larger (+X)
- To move the carrot further to the LEFT   -> X gets smaller (-X)

FOXGLOVE VIEW (top-down):
- Top of the screen   = +Y
- Right of the screen = +X
============================================================='''






class WallFollower(Node):
    def __init__(self):
        super().__init__('wall_follower')
        
        # Subscriber for LiDAR data 
        self.sub_scan = self.create_subscription(LaserScan, '/ldlidar_node/scan', self.scan_callback, qos_profile_sensor_data)
        self.last_point_data = []  # here we store the raw LiDAR points for the camera fusion
        self.sub_imu = self.create_subscription(Imu, '/bno055/imu', self.imu_callback, 10)
        
        # Publisher for motion and RViz [cite: 1, 19]
        self.pub_cmd_vel = self.create_publisher(Twist, '/cmd_vel', 10)
        self.pub_markers = self.create_publisher(MarkerArray, '/wall_follower_markers', 10)

        # Publisher for the Bezier path in Foxglove
        self.pub_path = self.create_publisher(Path, '/planned_trajectory', 10)
        

        self.yaw_offset = 0.0
        self.current_yaw = 0.0
        self.imu_ready = False  # <--- NEW: is the gyro awake yet?
        self.target_yaw = 0.0      # target angle for the turn
        self.last_raw_yaw = None
        self.start_turn_yaw = None
        self.start_straight_yaw = 0.0
        
        
        # Configuration
        self.rviz_frame = 'ldlidar_link'  # must be set as "Fixed Frame" in RViz
        self.get_logger().info('>>> WallFollower template started. Waiting for LiDAR... <<<')

        # --- STATE MACHINE & PATH PLANNING ---
        self.state = 'STARTING'    # initial state
        self.direction = None     # detected automatically

        self.target_turns = 4
        self.turn_count = 0
        self.locked_turn_count = 0  # remembers in which "lap" the obstacle stood

        self.front_wall = None
        self.left_wall = None
        self.right_wall = None
        
        # carrot parameters for driving straight
        self.lookahead_dist_straight = 0.60    # how far ahead does the robot look? (60 cm)
        self.min_wall_dist = 0.10       

        self.lane_ratio = 0.85       # ratio of the wall distances inner to outer. Outer wall: 0.85, inner wall: 0.20
        self.assumed_lane_width = 1.0 # if a wall is missing, we assume a 60cm lane width
        self.turn_exit_angle = 25
        self.max_wall_lenght_for_turn = 0.25

        # Object Detection Parameter
        self.current_obstacle_cmd = "CLEAR"

        # default values for the racing line on the straight (e.g. outer lane)
        self.default_lane_ratio = 0.85 
        self.default_max_turn = 0.8
        

        # --- PID CONTROLLER PARAMETERS ---
        self.kp = 3.5   # steers hard towards the carrot
        self.kd = 0   # prevents weaving (damping)
        self.ki = 0.0   # integral (often left at 0 in WRO because of fast lane changes)
        
        self.prev_error = 0.0
        self.integral_error = 0.0

        # carrot parameters for the turn
        self.lookahead_dist_turn = 0.20
        self.start_turn_dist = 0
        self.target_point = (0, 0)

        # --- TURN PID CONTROLLER PARAMETERS ---
        self.turn_kp = 0.6
        self.turn_kd = 0.0
        self.turn_ki = 0.0

        # --- MOTOR PARAMETER (ESP PWM 0 - 1023) ---
        self.base_speed = 350.0  # normal speed on the straight
        self.turn_speed = 350.0  # slightly reduced speed in the turn

        self.max_turn_angle = 0.635  # maximum steering angle in degrees (for safety)    min outer 0.435, max inner 0.800

        # --------------------------------
        # --- YOLO - Global Parameters ---
        # --------------------------------

        self.bridge = CvBridge()
        
        # 1. Load the YOLO model (TensorRT .engine)
        self.get_logger().info('Loading YOLO TensorRT engine...')
        #self.model = YOLO('/workspace/best.engine', task='detect')
        self.get_logger().info('Model loaded successfully!')
        # Add to the __init__ class:
        self.angle_calibration = 0.0  # in degrees: corrects if lidar/camera are rotated
        self.lidar_height_offset = 0.05 # in metres: raises/lowers the marker in RViz
        self.camera_to_lidar_dist = 0.03 # in case the camera sits 3cm in front of the lidar

        # 2. Subscriptions
        # Camera image   
        self.last_image_msg = None  # here we store the last image for the fusion
        self.image_width = None  # width of the camera image (set with the first image)
        self.img_sub = self.create_subscription(
            Image, 
            '/camera/image_raw', 
            self.camera_sub_callback, 
            qos_profile_sensor_data
        )
        # Add lidar scan
        self.scan_sub = self.create_subscription(
            LaserScan,
            '/ldlidar_node/scan',
            self.scan_callback,
            qos_profile_sensor_data
        )
        
        # 3. Publisher
        self.marker_pub = self.create_publisher(Marker, '/detected_obstacles', 10)
        self.pub_debug_img = self.create_publisher(Image, '/camera/yolo_debug', 10)
        self.avoid_trigger_dist = 0.85 # Threshold: dodge from 85cm in front of the obstacle
        
        # Variables for the fusion
        self.camera_fov = 115.0  # Your field of view
        self.get_logger().info('YOLO lidar fusion node started.')

        self.turn_start_time = 0.0

        ############ debug ##############
        self.begin = True
        self.counter = 0

        # measurement variables
        self.samples_x = np.array([
            -1.0, -0.75, -0.50, -0.30, -0.15, -0.05, 
             0.0, 
             0.05, 0.15, 0.30, 0.50, 0.75, 1.0
        ])
        self.samples_y = np.zeros(13)
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
        
        # --- THE MAGIC FIX ---
        # No more angle calculations! We take the lidar coordinates 1:1.
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
        Converts quaternions into an UNBOUNDED angle without jumps.
        (No jumping back at 180 or 360 deg!)
        """
        self.imu_ready = True
        q = msg.orientation
        
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        yaw_rad = math.atan2(siny_cosp, cosy_cosp)
        
        # this is the raw value with the stupid jump at 180 / -180
        raw_yaw = math.degrees(yaw_rad)
        
        # with the very first data packet we simply initialise
        if self.last_raw_yaw is None:
            self.last_raw_yaw = raw_yaw
            self.current_yaw = raw_yaw
            return

        # how far have we turned since the last millisecond?
        delta = raw_yaw - self.last_raw_yaw
        
        # ==========================================
        # THE MAGIC: catch the 360 deg jump!
        # ==========================================
        if delta > 180.0:
            delta -= 360.0
        elif delta < -180.0:
            delta += 360.0
            
        # we only add the clean delta to our unbounded angle
        self.current_yaw += delta
        self.last_raw_yaw = raw_yaw
    
    def camera_sub_callback(self, msg):
        self.last_image_msg = msg

    def send_text(self, marker_array, m_id, text, x, y, color=(1.0, 1.0, 1.0)):
        """Helper to create floating text in RViz."""
        marker = Marker()
        marker.header.frame_id = self.rviz_frame
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "labels"  # own namespace so it does not clash with the lines
        marker.id = m_id
        marker.type = Marker.TEXT_VIEW_FACING
        marker.action = Marker.ADD
        
        # position of the text (Z slightly raised so it floats above the lidar)
        marker.pose.position.x = float(x)
        marker.pose.position.y = float(y)
        marker.pose.position.z = 0.3 
        
        marker.scale.z = 0.15  # text size
        
        marker.color.r, marker.color.g, marker.color.b = color
        marker.color.a = 1.0
        marker.text = text
        
        marker_array.markers.append(marker)

    def send_sphere(self, marker_array, m_id, x, y, color=(0.0, 1.0, 1.0)):
        """Draws a glowing sphere (the 'carrot') in RViz."""
        marker = Marker()
        marker.header.frame_id = self.rviz_frame
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "target"
        marker.id = m_id
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.position.x = float(x)
        marker.pose.position.y = float(y)
        marker.pose.position.z = 0.05 # slightly above the floor
        marker.scale.x = 0.15 # 15 cm diameter
        marker.scale.y = 0.15
        marker.scale.z = 0.15
        marker.color.r, marker.color.g, marker.color.b = color
        marker.color.a = 1.0
        marker_array.markers.append(marker)

    def get_all_clusters_sorted(self, point_data): # adapted to normal vector and dot product
        """
        Finds ALL connected clusters and cuts them at 90 deg corners.
        Uses the Manhattan norm for gaps and the dot product for corners.
        point_data: list of (angle_deg, x, y, dist)
        """
        if len(point_data) < 2:
            return []

        # sort by angle (from right via rear to left)
        sorted_points = sorted(point_data, key=lambda p: p[0])

        max_gap = 0.15      # maximum distance between two points (15cm)
        outlier_limit = 2   # how many points may be missing?
        
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
            
            # 1. MANHATTAN DISTANCE (gap detection)
            dx = abs(p_curr[1] - p_last[1])
            dy = abs(p_curr[2] - p_last[2])
            dist_manhattan = dx + dy
            
            if dist_manhattan < max_gap:
                
                # =========================================================
                # 2. NEW CORNER DETECTION (dot product)
                # =========================================================
                is_corner = False
                
                # we need at least 5 points of history to see a trend
                if len(current_cluster) >= 5:
                    # vector A: the wall of the past (point 5 steps ago to the last point)
                    p_past = current_cluster[-5]
                    vec_a_x = p_last[1] - p_past[1]
                    vec_a_y = p_last[2] - p_past[2]
                    len_a = math.hypot(vec_a_x, vec_a_y)
                    
                    # vector B: the new step (last point to the current point)
                    vec_b_x = p_curr[1] - p_last[1]
                    vec_b_y = p_curr[2] - p_last[2]
                    len_b = math.hypot(vec_b_x, vec_b_y)
                    
                    if len_a > 0.01 and len_b > 0.01:
                        # normalise the vectors (length 1)
                        vec_a_x /= len_a
                        vec_a_y /= len_a
                        vec_b_x /= len_b
                        vec_b_y /= len_b
                        
                        # dot product: gives the cosine of the enclosed angle
                        # 1.0 = same direction (straight wall)
                        # 0.0 = 90 degrees (perfect corner)
                        # -1.0 = 180 degrees (turning back)
                        dot_product = (vec_a_x * vec_b_x) + (vec_a_y * vec_b_y)
                        
                        # if the dot product drops below 0.7 (about > 45 degrees bend), 
                        # we have reached a corner!
                        if dot_product < 0.70:
                            is_corner = True

                # =========================================================
                # 3. MAKE THE DECISION
                # =========================================================
                if is_corner:
                    # corner detected! We end the old wall and start a new one right away.
                    # the current point (p_curr) is the very first point of the new wall!
                    clusters.append(current_cluster)
                    current_cluster = [p_curr]
                else:
                    # no corner, the point belongs to the current wall
                    current_cluster.append(p_curr)
                
                i += 1
                    
            else:
                # 4. OUTLIER LOGIC (bridge gaps)
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

        # 5. WRAP-AROUND FIX (if the wall was cut at 180 / -180 degrees)
        if len(clusters) > 1:
            first_p = clusters[0][0]
            last_p = clusters[-1][-1]
            dist_wrap = abs(first_p[1] - last_p[1]) + abs(first_p[2] - last_p[2])
            
            if dist_wrap < max_gap:
                clusters[0] = clusters[-1] + clusters[0]
                clusters.pop()

        # return: the largest walls first
        clusters.sort(key=len, reverse=True)
        return clusters
    
    def get_cluster_angle(self, cluster): # adapted to the new coordinate system
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

        # your original again: s_yy - s_xx
        # this keeps side walls at 0 deg and front walls at 90 deg
        angle_rad = 0.5 * math.atan2(2.0 * s_xy, s_yy - s_xx)
        
        return math.degrees(angle_rad)

    def delete_marker(self, marker_array, m_id, ns="walls"):
        """Deletes a marker in a given namespace."""
        marker = Marker()
        marker.header.frame_id = self.rviz_frame
        marker.ns = ns  # flexible now! "walls" by default though
        marker.id = m_id
        marker.action = Marker.DELETE
        marker_array.markers.append(marker)

    def scan_callback(self, msg):
        point_data = []

        for i, dist in enumerate(msg.ranges):
            # filter invalid values (inf, nan or out of range)
            if math.isinf(dist) or math.isnan(dist) or dist < 0.075 or dist > 3.0:
                continue

            angle_lidar_rad = msg.angle_min + i * msg.angle_increment
            
            # Foxglove & maths basis: X = right, Y = front
            x_ros = dist * math.cos(angle_lidar_rad)    
            y_ros = dist * math.sin(angle_lidar_rad)   
            
            # ========================================================
            # NEW ANGLE SYSTEM: 0 to 360 deg, start point is at the REAR
            # ========================================================
            # raw lidar angle: 0 deg=right, 90 deg=front, 180 deg=left, 270/-90 deg=rear
            # we add 90 deg so that the rear becomes 0 deg, and limit to 0-360.
            # -> rear: 0 deg | right: 90 deg | front: 180 deg | left: 270 deg
            angle_lidar_deg = math.degrees(angle_lidar_rad)
            angle_user_deg = (angle_lidar_deg + 90.0) % 360.0

            # store: (normalised 0-360 deg angle, X_right, Y_front, distance)
            point_data.append((angle_user_deg, x_ros, y_ros, dist))
        
        self.last_point_data = point_data
        self.main_logic(point_data)
    
    def get_closest_measure(self, point_data, target_angle):
        """
        Finds the point that is closest to target_angle,
        taking the circle wrap-around and negative angles into account.
        """
        if not point_data:
            return None

        def get_angular_diff(a, b):
            # computes the smallest difference on a 360 degree circle
            # (a - b + 180) % 360 - 180 normalises the result to the range [-180, 180]
            diff = (a - b + 180) % 360 - 180
            return abs(diff)

        # we look for the tuple with the smallest circular difference
        closest_point = min(point_data, key=lambda p: get_angular_diff(p[0], target_angle))
        
        return closest_point
    
    def get_closest_point_in_cluster(self, cluster):
        """
        Returns the point of a cluster that has the shortest distance to the LiDAR.
        cluster: list of points in the format (angle, x, y, dist)
        """
        if not cluster:
            return None

        # finds the element in the cluster whose value at index 3 (dist) is the smallest
        closest_point = min(cluster, key=lambda p: p[3])
        
        return closest_point

    def middle_of_cluster(self, cluster):
        """Computes the angle of the cluster's midpoint to the lidar."""
        sum = 0
        for c in cluster:
            sum += c[0]  # angle of the point
        return sum / len(cluster)

    def angle_diff(self, a, b):
        """Computes the smallest difference between two angles (rad)."""
        # Makes sure the difference also stays correct across the 0/360 deg boundary
        return math.atan2(math.sin(a - b), math.cos(a - b))
        
    def visualize_clusters(self, candidates_RVIZ):
        if self.direction == 'left':
            inner_wall = self.left_wall
            outer_wall = self.right_wall
        else:
            inner_wall = self.right_wall
            outer_wall = self.left_wall
            
        
        marker_array = MarkerArray()

        if self.state == 'FOLLOW_LANE':
            # OLD LOGIC FOR THE STRAIGHT
            if self.direction == 'left':
                inner_wall = self.left_wall
                outer_wall = self.right_wall
            else:
                inner_wall = self.right_wall
                outer_wall = self.left_wall
            
            if target_x is not None and target_y is not None:
                # cyan for driving straight
                self.send_sphere(marker_array, m_id=99, x=target_x, y=target_y, color=(0.0, 1.0, 1.0))
                
        elif self.state in ['TURN_LEFT', 'TURN_RIGHT']:
            # NEW LOGIC FOR THE TURN
            # get_target_point_turn expects the u_profile array!
            u_profile = [self.right_wall, self.front_wall, self.left_wall]
            
            # call the target point calculation (e.g. 40cm target distance)
            target_pt = None
            
            if target_pt is not None:
                target_x, target_y = target_pt
                # magenta for driving the turn (so you see right away that the new logic kicks in!)
                self.send_sphere(marker_array, m_id=99, x=target_x, y=target_y, color=(1.0, 0.0, 1.0))
        else:
            self.delete_marker(marker_array, 99, ns="target")

        # fixed colours: right=red, front=green, left=blue
        colors = [
            (1.0, 0.0, 0.0),    # red (original)
            (0.0, 1.0, 0.0),    # green (original)
            (0.0, 0.5, 1.0),    # azure (original)
            (1.0, 0.5, 0.0),    # Orange
            (0.5, 0.0, 1.0),    # violet
            (0.0, 1.0, 1.0),    # Cyan
            (1.0, 0.0, 1.0),    # Magenta
            (1.0, 1.0, 0.0),    # yellow
            (0.5, 0.5, 0.5),    # grey
            (0.6, 0.3, 0.0)     # brown
        ]


        # --- RVIZ TEXT MARKERS FOR THE DRIVING DIRECTION AND WALLS ---
        if self.direction is not None:
            # 1. show the locked driving direction directly above the robot (X=0, Y=0)
            direction_text = f"LOCKED: {self.direction.upper()}"
            self.send_text(marker_array, m_id=10, text=direction_text, x=0.0, y=0.0, color=(1.0, 1.0, 0.0)) # yellow

            # 2. label the inner wall
            if inner_wall and len(inner_wall) > 0:
                # we place the text in the middle of the wall
                middle_x = sum(p[1] for p in inner_wall) / len(inner_wall)
                middle_y = sum(p[2] for p in inner_wall) / len(inner_wall)
                self.send_text(marker_array, m_id=11, text="INNER", x=middle_x, y=middle_y, color=(1.0, 0.5, 0.0)) # Orange

            # 3. label the outer wall
            if outer_wall and len(outer_wall) > 0:
                middle_x = sum(p[1] for p in outer_wall) / len(outer_wall)
                middle_y = sum(p[2] for p in outer_wall) / len(outer_wall)
                self.send_text(marker_array, m_id=12, text="OUTER", x=middle_x, y=middle_y, color=(1.0, 0.0, 1.0)) # Magenta
        else:
            # as long as it is still scanning, show that
            self.send_text(marker_array, m_id=10, text="SCANNING DIRECTION...", x=0.0, y=0.0, color=(1.0, 1.0, 1.0)) # white
        for i, cluster in enumerate(candidates_RVIZ):
            # print info in the console
            angle = self.get_cluster_angle(cluster)

            if angle is None:
                #self.get_logger().warn(f"Wall {i+1}: angle could not be computed.")
                continue # skips this cluster and carries on with the next one
            #self.get_logger().info(f"Wall {i+1} (ID {i}): {len(cluster)} points, angle: {angle:.2f} deg")
            #self.get_logger().info(f"First point: (X={cluster[0][1]:.2f}, Y={cluster[0][2]:.2f}), last point: (X={cluster[-1][1]:.2f}, Y={cluster[-1][2]:.2f})")
            if cluster is None or len(cluster) < 2:
                self.delete_marker(marker_array, m_id=i)
                continue # go straight to the next element in the array

            # a cluster needs at least 2 points for a line
            if len(cluster) >= 2:
                # start and end point (index 1 = X, index 2 = Y)
                start_p = (cluster[0][1], cluster[0][2])
                end_p = (cluster[-1][1], cluster[-1][2])
                
                # draw the line: ID corresponds to the index (0, 1 or 2)
                self.send_line(marker_array, m_id=i, p1=start_p, p2=end_p, color=colors[i])

        # publish everything so it shows up in RViz2!
        self.pub_markers.publish(marker_array)

    def cluster_choice(self, point_data):
        # 1. find all clusters
        all_clusters = self.get_all_clusters_sorted(point_data)
        for c in all_clusters:
            self.visualize_clusters([c])
            skip = input("Press Enter to go to the next cluster (or 'q' to select)...")
            if skip.lower() == 'q':
                self.get_logger().info(">>> Starting test bench with the selected front wall! <<<")
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
                
                # convert the values back into NumPy arrays
                self.samples_x = np.array(calib_data["samples_x"])
                self.samples_y = np.array(calib_data["samples_y"])
                self.poly_coeffs = np.array(calib_data["poly_coeffs"])
                
                self.get_logger().info("Calibration data loaded successfully.")
            except Exception as e:
                self.get_logger().error(f"JSON broken, using default values. Error: {e}")
                self._set_default_calibration()
        else:
            self.get_logger().warn("No calibration file found! Using default values.")
            self._set_default_calibration()

    def _set_default_calibration(self):
        # fallback in case no JSON exists
        self.poly_coeffs = np.array([0.0, 1.5, 0.0]) # linear default assumption
        # leave self.samples_x and _y defined as before

    def save_calibration(self):
        # 1. convert radii into curvature (kappa)
        kappas = np.zeros_like(self.samples_y)
        for i, r in enumerate(self.samples_y):
            # keep the sign! (right turns have negative u, so also negative kappa)
            # if the radius comes in positive via the lidar offset, we have to
            # take over the sign of u so that the maths is right.
            sign_x = np.sign(self.samples_x[i])
            if sign_x == 0:
                sign_x = 1.0

            if abs(r) > 0.001:  # protection against division by 0
                kappas[i] = sign_x * (1.0 / abs(r))
            else:
                kappas[i] = 0.0

        # 2. fit a 3rd degree polynomial: u = c3*kappa^3 + c2*kappa^2 + c1*kappa + c0
        # IMPORTANT: we fit u (x) over kappa, not the other way round! Degree 3 because of symmetry.
        coeffs = np.polyfit(kappas, self.samples_x, 3)

        # 3. put the data into a dictionary
        calib_data = {
            "samples_x": self.samples_x.tolist(),
            "samples_y": self.samples_y.tolist(),
            "poly_coeffs": coeffs.tolist()
        }

        # 4. save as JSON (absolute path in the home directory)
        file_path = '/workspace/src/robot_vision/robot_vision/wro_calibration.json'
        try:
            with open(file_path, 'w') as f:
                json.dump(calib_data, f, indent=4)
            self.get_logger().info(f"Calibration saved successfully at {file_path}")
            self.get_logger().info(f"Coefficients found: {coeffs}")
        except Exception as e:
            self.get_logger().error(f"Error saving the calibration: {e}")

    def main_logic(self, point_data):
        """
        State machine for the calibration.
        """
        if self.measure_start_yaw is None:
            self.measure_start_yaw = self.current_yaw
        # --- FIX FOR INDEX 8 (driving straight) ---
        # since a 180 deg turn is impossible at u=0, we set R=0 and skip it.
        if self.current_test_index == 5 and self.test_state != "DONE":
            self.samples_y[5] = 0.0
            self.get_logger().info("Index 8 (u=0.0) is skipped (driving straight).")
            
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
                # 1. ask for user input
                self.get_logger().info("Sensors ready. Waiting for user input in the terminal...")
                print("\n=== CALIBRATION MENU ===")
                for i, x in enumerate(self.samples_x):
                    current_radius = self.samples_y[i]
                    # formatted output: u to 2 decimal places, radius to 3 decimal places (metres)
                    print(f"[{i:2d}]: u = {x:5.2f}  |  stored radius: {current_radius:6.3f} m")
                    
                user_input = input("\nWhich index should be measured? ('all' for all): ")
                
                if user_input.lower() == 'all':
                    self.current_test_index = 0
                    self.single_test_mode = False
                else:
                    try:
                        idx = int(user_input)
                        if 0 <= idx < len(self.samples_x):
                            self.current_test_index = idx
                            self.single_test_mode = True
                            u = self.samples_x[self.current_test_index]
                            test_cmd = Twist()
                            test_cmd.linear.x = 0.0
                            test_cmd.angular.z = float(u)
                            self.pub_cmd_vel.publish(test_cmd)
                        else:
                            self.get_logger().error("Index out of bounds! Restart the script.")
                            self.test_state = "DONE"
                            return
                    except ValueError:
                        self.get_logger().error("Invalid input! Restart the script.")
                        self.test_state = "DONE"
                        return

                self.get_logger().info(f"Starting test for u={self.samples_x[self.current_test_index]}")
                
                # 2. cluster choice and start of the measurement
                cluster = self.cluster_choice(point_data)
                if cluster is not None and len(cluster) > 0:
                    self.start_wall_distance = (self.get_closest_point_in_cluster(cluster)[3])
                    self.test_state = "APPLY_STEERING"

        # ---------------------------------------------------------
        elif self.test_state == "APPLY_STEERING":
            u = self.samples_x[self.current_test_index]
            test_cmd = Twist()
            test_cmd.linear.x = self.turn_speed
            test_cmd.angular.z = float(u)
            self.pub_cmd_vel.publish(test_cmd)
            self.test_state = "STOP_FOR_MEASUREMENT"
            self.get_logger().info(f"Starting turn for u={u}")

        # ---------------------------------------------------------
        elif self.test_state == "STOP_FOR_MEASUREMENT":
            u = self.samples_x[self.current_test_index]
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
                self.get_logger().info("Turn finished. Stopping the robot and starting the measurement...")
            self.get_logger().info(f"Current yaw: {self.current_yaw:.1f} deg, target: {self.measure_start_yaw} deg")

        # ---------------------------------------------------------
        elif self.test_state == "MEASURE":
            cluster = self.cluster_choice(point_data)
            if cluster is not None and len(cluster) > 0:
                yaw_diff = abs(self.current_yaw - self.measure_start_yaw)
                new_distance = self.get_closest_point_in_cluster(cluster)[3]
                distance_diff = new_distance - self.start_wall_distance - self.lidar_to_turnpoint_distance
                #radius = distance_diff / 2.0
                radius = distance_diff

                # HERE the old value in the array is overwritten with the new one
                self.samples_y[self.current_test_index] = abs(radius)
                self.get_logger().info(f"Result u={self.samples_x[self.current_test_index]}: radius {radius*1000:.0f} mm, angle {yaw_diff:.1f} deg")
                self.test_state = "NEXT_TEST"

        # ---------------------------------------------------------
        elif self.test_state == "NEXT_TEST":
            # --- STOP CONDITION FOR A SINGLE MEASUREMENT ---
            if self.single_test_mode:
                self.get_logger().info("Single measurement finished. Saving new JSON...")
                stop_cmd = Twist()
                stop_cmd.linear.x = 0.0
                stop_cmd.angular.z = 0.0
                self.pub_cmd_vel.publish(stop_cmd)
                
                # stores the 15 old values and the 1 new one + refits the polynomial
                self.save_calibration()
                self.test_state = "DONE"
                
            else:
                self.current_test_index += 1
                if self.current_test_index < len(self.samples_x):
                    self.get_logger().info(f"Switching to u={self.samples_x[self.current_test_index]}")
                    self.test_state = "APPLY_STEERING"
                else:
                    self.get_logger().info("All measurements finished. Stopping the robot.")
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
        # node cleanup, guarded against a double shutdown
        if node.context.ok():
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()