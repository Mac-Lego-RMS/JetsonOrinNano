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
        
        # Subscriber for LiDAR data 
        self.sub_scan = self.create_subscription(LaserScan, '/ldlidar_node/scan', self.scan_callback, qos_profile_sensor_data)
        self.last_point_data = []  # here we store the raw LiDAR points for the camera fusion
        self.sub_imu = self.create_subscription(Imu, '/bno055/imu', self.imu_callback, 10)
        
        # Publisher for motion and RViz [cite: 1, 19]
        self.pub_cmd_vel = self.create_publisher(Twist, '/cmd_vel', 10)
        self.pub_markers = self.create_publisher(MarkerArray, '/wall_follower_markers', 10)
        

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
        self.default_max_turn = 0.635
        

        # --- PID CONTROLLER PARAMETERS ---
        self.kp = 3.5   # steers hard towards the carrot
        self.kd = 0   # prevents weaving (damping)
        self.ki = 0.0   # integral (often left at 0 in WRO because of fast lane changes)
        
        self.prev_error = 0.0
        self.integral_error = 0.0

        # carrot parameters for the turn
        self.lookahead_dist_turn = 0.25

        # --- TURN PID CONTROLLER PARAMETERS ---
        self.turn_kp = 1.0
        self.turn_kd = 0.0
        self.turn_ki = 0.0

        # --- MOTOR PARAMETER (ESP PWM 0 - 1023) ---
        self.base_speed = 450.0  # normal speed on the straight
        self.turn_speed = 450.0  # slightly reduced speed in the turn

        self.max_turn_angle = 0.635  # maximum steering angle in degrees (for safety)    min outer 0.435, max inner 0.800

        # --------------------------------
        # --- YOLO - Global Parameters ---
        # --------------------------------

        self.bridge = CvBridge()
        
        # 1. Load the YOLO model (TensorRT .engine)
        self.get_logger().info('Loading YOLO TensorRT engine...')
        self.model = YOLO('/workspace/best.engine', task='detect')
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
    
    def validate_clusters_straight(self, clusters):
        # we use while because the list can shrink
        while len(clusters) >= 3:
            
            # 1. take the currently largest 3 and sort them from right to left
            ordered = self.sort_clusters_right_to_left(clusters[:3])
            
            # 2. compute the angles
            angles = [self.get_cluster_angle(c) for c in ordered]

            if any(angle is None for angle in angles):
                self.get_logger().warn("Error in the angle calculation. Skipping...")
                return [None, None, None]

            # --- HELPER FOR THE ANGLE DIFFERENCE ---
            # computes the smallest intersection angle between two lines (0 to 90 deg)
            def get_angle_diff(a1, a2):
                diff = abs(a1 - a2) % 180
                if diff > 90:
                    diff = 180 - diff
                return diff

            # compute differences (0=right, 1=front, 2=left)
            diff_0_1 = get_angle_diff(angles[0], angles[1]) # should be ~90 deg (orthogonal)
            diff_1_2 = get_angle_diff(angles[1], angles[2]) # should be ~90 deg (orthogonal)
            diff_0_2 = get_angle_diff(angles[0], angles[2]) # should be ~0 deg (parallel)

            # --- CHECK ---
            # tolerance: we allow up to 20 deg deviation from the perfect geometry
            
            # Check A: are right and front orthogonal? (difference should be > 70 deg)
            if diff_0_1 < 70:
                self.get_logger().warn(f"Right and front not orthogonal! Diff: {diff_0_1:.1f} deg")
                # find the smaller of the two clusters in 'ordered' and delete it from 'clusters'
                if len(ordered[0]) < len(ordered[1]):
                    clusters.remove(ordered[0])
                else:
                    clusters.remove(ordered[1])
                continue # restart the loop right away with the cleaned-up list

            # Check B: are front and left orthogonal?
            elif diff_1_2 < 70:
                self.get_logger().warn(f"Front and left not orthogonal! Diff: {diff_1_2:.1f} deg")
                if len(ordered[1]) < len(ordered[2]):
                    clusters.remove(ordered[1])
                else:
                    clusters.remove(ordered[2])
                continue

            # Check C: are right and left parallel? (difference should be < 20 deg)
            elif diff_0_2 > 10:
                self.get_logger().warn(f"Right and left not parallel! Diff: {diff_0_2:.1f} deg")
                if len(ordered[0]) < len(ordered[2]):
                    clusters.remove(ordered[0])
                else:
                    clusters.remove(ordered[2])
                continue

            # --- SUCCESS ---
            
            # returns exactly assigned: (right wall, front wall, left wall)
            return ordered
        # if the loop ends (list has fewer than 3 clusters)
        self.get_logger().warn(f"No valid U profile found. Only {len(clusters)} clusters left.")
        return [None, None, None]
    

    def validate_clusters_turn(self, front_wall, point_data):
        '''Checks the clusters in the turn. There must be a front wall, but the walls may also be missing (e.g. in the first turn).
        Returns the clusters from right to left in the order: [right wall, front wall, left wall]. Missing walls are replaced by None.'''

        if front_wall is None:
            self.get_logger().warn("No front wall found. Cannot validate the turn profile.")
            return [None, None, None]

        all_clusters = self.get_all_clusters_sorted(point_data)
        clusters = self.get_unshadowed_leftovers(all_clusters, [front_wall])
        clusters.append(front_wall) # we add the front wall back so it is taken into account in the sorting

        if len(clusters) >= 2:
            minimal_cluster_size = 15
            ordered = self.sort_clusters_right_to_left(clusters)

            u_profile = [None, None, None] # 0=right, 1=front, 2=left
            if front_wall in ordered:
                u_profile[1] = front_wall
                fw_index = ordered.index(front_wall)
            else: 
                self.get_logger().warn("Front wall not found in the clusters. Cannot validate the turn profile.")
                return [None, None, None]
                
            # the cluster LEFT of the front wall has a SMALLER index
            while u_profile[0] is None and fw_index > 0:
                if len(ordered[fw_index - 1]) > minimal_cluster_size:
                    u_profile[0] = ordered[fw_index - 1]
                    self.get_logger().info(f"Cluster found left of the front wall. Size: {len(ordered[fw_index - 1])} points.")
                else: 
                    ordered.pop(fw_index - 1)
                    fw_index -= 1

            # the cluster RIGHT of the front wall has a LARGER index
            while u_profile[2] is None and fw_index < len(ordered) - 1:
                if len(ordered[fw_index + 1]) > minimal_cluster_size:
                    u_profile[2] = ordered[fw_index + 1]
                    self.get_logger().info(f"Cluster found right of the front wall. Size: {len(ordered[fw_index + 1])} points.")
                else: 
                    ordered.pop(fw_index + 1)

            angles = [self.get_cluster_angle(c) for c in u_profile]
            if angles[1] is None:
                self.get_logger().warn("Error in the angle calculation of the front wall. Skipping...")
                return [None, None, None]

            # --- HELPER FOR THE ANGLE DIFFERENCE ---
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


                # --- CHECK ---
                # tolerance: we allow up to 20 deg deviation from the perfect geometry
                
                # Check A: are right and front orthogonal? (difference should be > 70 deg)
            if diff_0_1 is not None and diff_0_1 < 70:
                self.get_logger().warn(f"Right and front not orthogonal! Diff: {diff_0_1:.1f} deg")
                u_profile[0] = None


                # Check B: are front and left orthogonal?
            elif diff_1_2 is not None and diff_1_2 < 70:
                self.get_logger().warn(f"Front and left not orthogonal! Diff: {diff_1_2:.1f} deg")
                u_profile[2] = None

                # Check C: are right and left parallel? (difference should be < 20 deg)
            elif diff_0_2 is not None and diff_0_2 > 15:
                self.get_logger().warn(f"Right and left not parallel! Diff: {diff_0_2:.1f} deg")
                return [None, front_wall, None]
            
            else:
                wall_dist = self.get_closest_point_in_cluster(u_profile[0])[3] + self.get_closest_point_in_cluster(u_profile[2])[3] if u_profile[0] and u_profile[2] else None
                if wall_dist is None:
                    return
                if wall_dist < 2.0:
                    self.get_logger().warn("Both side walls too close together! Probably a false detection. Ignoring the inner walls.")
                    if self.direction == "left":
                        u_profile[2] = None
                        return self.merge_clusters(clusters, u_profile)
                    else:
                        u_profile[0] = None
                        return self.merge_clusters(clusters, u_profile)
                else:
                    return 

            return self.merge_clusters(clusters, u_profile)
        return [None, front_wall, None]
    
    def sort_clusters_right_to_left(self, clusters):
        """
        Takes a list of clusters and sorts them spatially from right to left.
        Requirement: tuple format (angle, x, y, dist)
        """
        if not clusters:
            return []

        def get_cluster_bearing(cluster):
            # 1. compute the centroid of the cluster
            # Index 1 = X, Index 2 = Y
            mean_x = sum(p[1] for p in cluster) / len(cluster)
            mean_y = sum(p[2] for p in cluster) / len(cluster)
            
            return math.atan2(mean_y, mean_x)

        sorted_clusters = sorted(clusters, key=get_cluster_bearing, reverse=True)
        
        return sorted_clusters

    def get_all_clusters_sorted(self, point_data):
        """
        Finds ALL connected clusters and cuts them at 90 deg corners.
        Uses the Manhattan norm and X/Y gradient monitoring.
        point_data: list of (angle_deg, x, y, dist)
        """
        if len(point_data) < 2:
            return []

        sorted_points = sorted(point_data, key=lambda p: p[0])

        # --- PARAMETER ---
        # Set max gap a bit higher, since Manhattan values (dx+dy) are larger than Euclidean ones
        max_gap = 0.10      
        outlier_limit = 2
        # From which segment length do we trust the corner? (filters sensor noise)
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
            
            # 1. MANHATTAN DISTANCE (|dx| + |dy|)
            dx = abs(p_curr[1] - p_last[1])
            dy = abs(p_curr[2] - p_last[2])
            dist_manhattan = dx + dy
            
            if dist_manhattan < max_gap:
                # 2. NEW CORNER DETECTION (via real vector angles)
                if len(current_cluster) >= 20:
                    p_start = current_cluster[5] # point before the curve
                    p_mid = current_cluster[-10]   # point at the apex
                    
                    # Vector 1 (trend before the curve)
                    dx1 = p_mid[1] - p_start[1]
                    dy1 = p_mid[2] - p_start[2]
                    angle1 = math.atan2(dy1, dx1)
                    
                    # Vector 2 (current movement)
                    dx2 = p_curr[1] - p_mid[1]
                    dy2 = p_curr[2] - p_mid[2]
                    angle2 = math.atan2(dy2, dx2)
                    
                    # How sharply does the wall bend physically?
                    diff_rad = abs(angle1 - angle2)
                    diff_deg = math.degrees(diff_rad)
                    if diff_deg > 180:
                        diff_deg = 360 - diff_deg
                        
                    # Filter: has the point also moved far enough? (ignore noise)
                    dist_moved = math.hypot(dx2, dy2)
                    
                    # A real course corner bends sharply (e.g. > 70 degrees)
                    if diff_deg > 25.0 and dist_moved > corner_sensitivity:
                        # CORNER DETECTED! We cut the cluster exactly here.
                        clusters.append(current_cluster)
                        current_cluster = [p_curr]
                        i += 1
                        continue

                # If no corner, add the point to the cluster normally
                current_cluster.append(p_curr)
                i += 1
            else:
                # 3. OUTLIER LOGIC (now also with Manhattan norm)
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
                    # Real gap found -> store the cluster and start again
                    clusters.append(current_cluster)
                    current_cluster = [p_curr]
                    i += 1
        
        if current_cluster:
            clusters.append(current_cluster)

        # 4. WRAP-AROUND FIX (with Manhattan)
        if len(clusters) > 1:
            first_p = clusters[0][0]
            last_p = clusters[-1][-1]
            dist_wrap = abs(first_p[1] - last_p[1]) + abs(first_p[2] - last_p[2])
            
            # Here too, check whether across the 180 deg edge they are actually the same wall axis
            if dist_wrap < max_gap:
                clusters[0] = clusters[-1] + clusters[0]
                clusters.pop()

        # 5. Sort by size, largest first
        clusters.sort(key=len, reverse=True)
        
        return clusters
 
    def get_cluster_angle(self, cluster):
        """
        Computes the average angle of all points in the cluster (regression line).
        Front/back (side walls) = 0 deg
        Left/right (front wall) = 90 or -90 deg
        """

        if cluster is None:
            return None
        
        n = len(cluster)
        if n < 2:
            return None

        

        # 1. compute the centroid
        mean_x = sum(p[1] for p in cluster) / n
        mean_y = sum(p[2] for p in cluster) / n

        # 2. deviations from the centroid
        s_xx = 0.0
        s_yy = 0.0
        s_xy = 0.0
        
        for p in cluster:
            dx = p[1] - mean_x
            dy = p[2] - mean_y
            s_xx += dx * dx
            s_yy += dy * dy
            s_xy += dx * dy

        # 3. FIX: we swap s_xx and s_yy in the denominator! 
        # This way we compute the angle relative to the Y axis (driving direction) instead of the X axis.
        angle_rad = 0.5 * math.atan2(2.0 * s_xy, s_yy - s_xx)
        
        # convert to degrees
        angle_deg = math.degrees(angle_rad)

        return angle_deg

    def delete_marker(self, marker_array, m_id, ns="walls"):
        """Deletes a marker in a given namespace."""
        marker = Marker()
        marker.header.frame_id = self.rviz_frame
        marker.ns = ns  # flexible now! "walls" by default though
        marker.id = m_id
        marker.action = Marker.DELETE
        marker_array.markers.append(marker)

    def scan_callback(self, msg):
        # this is your array of tuples: (degrees, distance, X, Y)
        point_data = []

        for i, dist in enumerate(msg.ranges):
            # filter invalid values (inf, nan or out of range) [cite: 6]
            if math.isinf(dist) or math.isnan(dist) or dist < 0.075 or dist > 3.0:
                continue

            # 1. compute the original lidar angle in radians [cite: 5]
            angle_lidar_rad = msg.angle_min + i * msg.angle_increment
            angle_lidar_deg = math.degrees(angle_lidar_rad)

            # 2. conversion: front = 0, right = positive (+), left = negative (-)
            # since 90 deg lidar = front: 90 - 90 = 0 (front)
            # 80 deg lidar (right) becomes 90 - 80 = +10 deg
            # 100 deg lidar (left) becomes 90 - 100 = -10 deg
            angle_user_deg = 90.0 - angle_lidar_deg

            # 3. Cartesian coordinates for the logic (X front, Y left)
            # IMPORTANT: for computing X and Y we use the original radian value,
            # so that the geometry stays correct for RViz and standard ROS (X = front).
            x = dist * math.cos(angle_lidar_rad)
            y = dist * math.sin(angle_lidar_rad)

            # store as a tuple: (adjusted angle, raw distance, X, Y) [cite: 10]
            point_data.append((angle_user_deg, x, y, dist))
        
        self.last_point_data = point_data  # store for the camera fusion
        self.process_my_logic(point_data)
        #self.get_logger().info(f"Angle 0: {self.get_closest_measure(point_data, target_angle=0)} Angle 105: {self.get_closest_measure(point_data, target_angle=105)} Angle -105: {self.get_closest_measure(point_data, target_angle=-105)}")  # example: find the point directly in front of the robot (0 deg)

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

    def merge_clusters(self, all_clusters, validated_clusters):
        """
        Tries to merge neighbouring clusters into a single cluster.
        Uses the normal vector to check only the perpendicular distance (offset).
        """
        max_distance_gap = 0.10  # 5 cm maximum lateral offset
        max_angle_gap = 5.0      # 5 degrees maximum angle deviation

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
                continue # skip this empty wall right away!

            if angle is None: 
                continue

            bx = sum(p[1] for p in valid_cluster) / len(valid_cluster)
            by = sum(p[2] for p in valid_cluster) / len(valid_cluster)

            angle_rad = math.radians(angle)
            # normal vector for the distance
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

            # --- THE MAGIC FIX ---
            # we compute the direction vector of the wall (parallel to the wall)
            dir_x = -math.sin(angle_rad)
            dir_y = math.cos(angle_rad)
            
            # we sort the points by their geometric position ALONG the wall!
            # This way the physically first and last point are always at index 0 and -1,
            # completely independent of the LiDAR wrap-around.
            valid_cluster.sort(key=lambda p: p[1] * dir_x + p[2] * dir_y)

        return validated_clusters

    def get_target_point_straight(self, inner_wall, outer_wall):
        """
        Computes the target point (carrot) in the perfect ratio to the inner wall.
        With an absolute minimum distance (force field logic).
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
        # NEW: THE "WALL HUGGER" FIX (single-wall tracking)
        # ==========================================
        # When we dodge an obstacle, we rely ONLY on 
        # the wall we are driving along. This prevents 
        # the passed obstacle from wrecking the lane width calculation!
        if self.current_obstacle_cmd != "CLEAR":
            if self.lane_ratio < 0.5:
                # we want to get very close to the inner wall (ratio e.g. 0.20)
                # -> we ignore the outer wall (and the obstacle there) completely!
                outer_wall = None
            else:
                # we want to get very close to the outer wall (ratio e.g. 0.85)
                # -> we ignore the inner wall completely!
                inner_wall = None

        # --- CASE 1: WE SEE BOTH WALLS (normal case on a clear track) ---
        if inner_wall and outer_wall:
            x_inner = get_x_at_y(inner_wall, target_y)
            x_outer = get_x_at_y(outer_wall, target_y)
            
            lane_width = abs(x_outer - x_inner)
            
            if x_inner < 0: # inner wall on the left
                target_x = x_inner + (lane_width * self.lane_ratio)
            else:           # inner wall on the right
                target_x = x_inner - (lane_width * self.lane_ratio)
                
        # --- CASE 2: WE ONLY SEE THE INNER WALL (or have ignored the outer one) ---
        elif inner_wall:
            x_inner = get_x_at_y(inner_wall, target_y)
            if x_inner < 0:
                target_x = x_inner + (self.assumed_lane_width * self.lane_ratio)
            else:
                target_x = x_inner - (self.assumed_lane_width * self.lane_ratio)
                
        # --- CASE 3: WE ONLY SEE THE OUTER WALL (or have ignored the inner one) ---
        elif outer_wall:
            x_outer = get_x_at_y(outer_wall, target_y)
            inv_ratio = 1.0 - self.lane_ratio 
            if x_outer < 0: # outer wall is on the left
                target_x = x_outer + (self.assumed_lane_width * inv_ratio)
            else:            # outer wall is on the right
                target_x = x_outer - (self.assumed_lane_width * inv_ratio)
                
        # emergency
        else:
            target_x = 0.0  

        # ==========================================
        # FORCE FIELD (ENFORCE MINIMUM DISTANCE)
        # ==========================================
        if inner_wall:
            # we check where the inner wall is
            x_inner_check = get_x_at_y(inner_wall, target_y)
            
            if x_inner_check < 0:  
                # inner wall is LEFT. Target MUST be at least +min_wall_dist away
                if target_x < x_inner_check + self.min_wall_dist:
                    target_x = x_inner_check + self.min_wall_dist
                    
            else:                  
                # inner wall is RIGHT. Target MUST be at least -min_wall_dist away
                if target_x > x_inner_check - self.min_wall_dist:
                    target_x = x_inner_check - self.min_wall_dist

        return (target_x, target_y)
    
    def get_target_point_turn(self, front_wall):
        """
        Computes the target point (carrot) in the turn based on the front wall.
        """
        
        x = sum(p[1] for p in front_wall) / len(front_wall)
        y = sum(p[2] for p in front_wall) / len(front_wall)

        # we shift the target point along the direction of the front wall
        angle = self.get_cluster_angle(front_wall)
        if angle is None:
            return (x, y)  # if the angle calculation fails, aim straight at the wall

        angle_rad = math.radians(angle)
        
        # shift along the wall direction (direction of the turn)
        offset_x = self.lookahead_dist_straight * math.cos(angle_rad)
        offset_y = self.lookahead_dist_straight * math.sin(angle_rad)

        target_x = x + offset_x
        target_y = y + offset_y

        return (target_x, target_y)

    def  track_front_wall(self, point_data, last_front_wall):
        """
        Tracks the front wall during the turn with a search window (ROI) that moves along.
        Returns the updated wall and a boolean (turn_finished).
        """
        if not last_front_wall or not point_data:
            return None

        # 1. where was the wall in the last frame? (find min/max angle)
        angles = [p[0] for p in last_front_wall]
        min_angle = min(angles)
        max_angle = max(angles)
        #self.get_logger().info(f"Tracking ROI: min angle {min_angle:.1f} deg, max angle {max_angle:.1f} deg")

        # 2. define the dynamic search window (ROI)
        # we give more tolerance in the direction of motion (e.g. +30 degrees), 
        # because the wall moves there. Less against the direction of motion (-10 degrees).
        
        if self.direction == 'left':
            # wall moves to the RIGHT (angles become more positive)
            roi_min = min_angle - 5.0
            roi_max = max_angle + 30.0
        else:
            # wall moves to the LEFT (angles become more negative)
            roi_min = min_angle - 30.0
            roi_max = max_angle + 5.0

        # 3. put on blinkers: filter the point cloud!
        roi_points = []
        for p in point_data:
            angle = p[0]
            if roi_min <= angle <= roi_max:
                roi_points.append(p)

        # 4. split only these filtered points into clusters
        roi_clusters = self.get_all_clusters_sorted(roi_points)

        # 5. check the tracking
        if not roi_clusters:
            self.get_logger().warn("WARNING: tracked wall lost in the ROI!")
            return last_front_wall
            
        # since we filtered out all other walls, the largest cluster 
        # (index 0) in this range is 99.9% the wall we are looking for!
        tracked_wall = roi_clusters[0]
        clusters = self.get_all_clusters_sorted(point_data)
        cluster = self.merge_clusters(clusters, [tracked_wall])
        

        return tracked_wall

    def get_unshadowed_leftovers(self, all_clusters, validated_clusters):
        """
        Returns all clusters that are neither part of the validated walls
        nor hidden "behind" the validated walls in terms of angle.
        """
        if not all_clusters:
            return []

        # 1. collect all points that have already been merged into walls successfully
        # (we use a set for extremely fast lookup)
        valid_points = set()
        for vc in validated_clusters:
            if vc is not None:
                valid_points.update(vc)
                
        # find the pure leftovers (if the first point is not in the set, 
        # the cluster was not merged)
        leftovers = [c for c in all_clusters if c and c[0] not in valid_points]
        
        # 2. compute the covered angle ranges ("shadows") of the real walls
        blocked_angle_ranges = []
        for vc in validated_clusters:
            if vc is not None and len(vc) > 0:
                angles = [p[0] for p in vc]
                min_a = min(angles)
                max_a = max(angles)
                blocked_angle_ranges.append((min_a, max_a))
                
        # 3. filter the leftovers: do they lie in a shadow?
        final_free_clusters = []
        padding = 5.0  # 5 degrees tolerance at the edges of the walls
        
        for cluster in leftovers:
            # where is this cluster in space? (centroid angle)
            c_mean_angle = sum(p[0] for p in cluster) / len(cluster)
            
            is_shadowed = False
            for (min_a, max_a) in blocked_angle_ranges:
                # does the centroid of the cluster lie exactly inside the wall in terms of angle?
                if (min_a - padding) <= c_mean_angle <= (max_a + padding):
                    is_shadowed = True
                    break
                    
            # if it is NOT hidden, it is a separate object (e.g. obstacle)
            if not is_shadowed:
                final_free_clusters.append(cluster)
                
        return final_free_clusters

    def update_avoidance_settings(self):
        """Adjusts lane, turn radius and carrot distance dynamically."""
        
        # --- DEFAULT VALUES (no obstacle) ---
        if self.current_obstacle_cmd == "CLEAR" or self.direction is None:
            self.lane_ratio = 0.85 
            self.max_turn_angle = 0.635
            self.lookahead_dist_straight = 0.60  # look ahead relaxed
            
            # IMPORTANT: better keep this commented out, otherwise it spams your terminal!
            # self.get_logger().info("No obstacles detected. Driving with default parameters.")
            return

        # --- DODGE VALUES (adrenaline mode) ---
        # pull the carrot closer to steer much more directly and sharply!
        self.lookahead_dist_straight = 0.35 

        # logic matrix
        if self.direction == 'left': # inner wall on the left
            if self.current_obstacle_cmd == "RED":       # (formerly AVOID_RIGHT) pass on the right
                self.lane_ratio = 0.85
                self.max_turn_angle = 0.635 # wide turn (outer)
            elif self.current_obstacle_cmd == "GREEN":   # (formerly AVOID_LEFT) pass on the left
                self.lane_ratio = 0.20
                self.max_turn_angle = 0.800 # tight turn (inner)

        elif self.direction == 'right': # inner wall on the right
            if self.current_obstacle_cmd == "RED":       # (formerly AVOID_RIGHT) pass on the right
                self.lane_ratio = 0.20
                self.max_turn_angle = 0.800 # tight turn (inner)
            elif self.current_obstacle_cmd == "GREEN":   # (formerly AVOID_LEFT) pass on the left
                self.lane_ratio = 0.85  
                self.max_turn_angle = 0.635 # wide turn (outer)

    # -----------------------
    # --- YOLO - Function ---
    # -----------------------

    def image_callback(self, msg):
        if self.last_point_data is None: return

        cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        self.image_width = cv_image.shape[1]     
        results = self.model(cv_image, verbose=False)

        # --- NEW: THE DEBUG IMAGE IS CREATED AND PUBLISHED HERE ---
        # 1. Draw bounding boxes and labels onto the image
        annotated_frame = results[0].plot()

        # 2. Convert the OpenCV image back into a ROS message
        debug_msg = self.bridge.cv2_to_imgmsg(annotated_frame, encoding="bgr8")
        # 3. Publish the image on the topic /camera/yolo_debug
        self.pub_debug_img.publish(debug_msg)
        # -------------------------------------------------------------
        
        return results
        
    def get_lidar_distance(self, camera_angle_rad, clusters):
        walls = [self.front_wall, self.left_wall, self.right_wall]
        clusters_without_walls = [c for c in clusters if c not in walls and c is not None]
        
        if not clusters_without_walls: 
            return None
            
        best_closest_point = None
        min_dist = 4.0  # IMPORTANT: we strictly look for the CLOSEST object!
        best_angle_deg = 0.0
        
        for cluster in clusters_without_walls:
            # FILTER 1: number of points (at least 2, max 25 for close obstacles)
            if len(cluster) < 2 or len(cluster) > 25:
                continue
                
            # FILTER 2: width filter (WRO blocks are small, max 35 cm)
            c_start = cluster[0]   
            c_end = cluster[-1]    
            width = math.hypot(c_start[1] - c_end[1], c_start[2] - c_end[2])
            if width > 0.35:
                continue

            angle_deg = self.middle_of_cluster(cluster)
            if angle_deg is None: continue
            angle_rad = math.radians(angle_deg)
            
            # FILTER 3: is the cluster within the 15 degree field of view of the camera?
            diff = abs(self.angle_diff(angle_rad, camera_angle_rad))
            if diff < math.radians(15.0):
                
                closest_point = self.get_closest_point_in_cluster(cluster)
                if closest_point is not None:
                    dist = closest_point[3]
                    
                    # FILTER 4: FOREGROUND PRINCIPLE! (prevents jumping)
                    # of all clusters in the field of view we always take the closest one.
                    if dist < min_dist:
                        min_dist = dist
                        best_closest_point = closest_point
                        best_angle_deg = angle_deg
                        
        # repaired log output (now prints the real values of the winning cluster!)
        if best_closest_point is not None:
            self.get_logger().info(f"MATCH: camera {math.degrees(camera_angle_rad):.1f} deg -> lidar {best_angle_deg:.1f} deg (distance: {min_dist:.2f}m)")
            return best_closest_point
            
        return None

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

    def process_my_logic(self, point_data):
        # 1. ALWAYS AT THE VERY TOP: create the collecting basket so it exists everywhere in the function!
        marker_array = MarkerArray()
        cmd = Twist()

        #self.get_logger().info(f"IMU angle: {(self.current_yaw - self.yaw_offset):.1f} deg")
            
                            
        inner_wall = None
        outer_wall = None
        candidates = [None, None, None]

        if self.last_image_msg is None:
            self.get_logger().info("Waiting for camera image...", throttle_duration_sec=1.0)
            return
        results = self.image_callback(self.last_image_msg)
        if results is None: return

        detected_obstacles = []

        for r in results:
            for box in r.boxes:
                if float(box.conf[0]) > 0.8:
                    self.get_logger().info(f'Object detected: {self.model.names[int(box.cls[0])]} with confidence {box.conf[0]:.2f}')
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    center_x = (x1 + x2) / 2.0

                    # 1. move the zero point to the centre of the image
                    # center_x goes from 0 (far left) to image_width (far right)
                    pixel_from_center = center_x - (self.image_width / 2.0)
                    
                    # 2. convert pixels to degrees (spread the camera FOV over the width)
                    cam_angle_deg = pixel_from_center * (self.camera_fov / self.image_width)
                    
                    # 3. add the calibration and convert to radians
                    cam_angle_rad = math.radians(cam_angle_deg + self.angle_calibration)
                    
                    self.get_logger().info(f'Camera angle: {cam_angle_deg:.1f} deg (after calibration: {cam_angle_deg + self.angle_calibration:.1f} deg)')

                    # search the lidar for this angle
                    result = self.get_lidar_distance(cam_angle_rad, self.get_all_clusters_sorted(self.last_point_data))

                    if result is not None:
                        # result now contains: (angle_deg, x, y, dist)
                        _, obj_x, obj_y, obj_dist = result 
                        class_name = self.model.names[int(box.cls[0])].lower()
                        
                        # publish_marker now gets the raw X and Y values directly!
                        self.publish_marker(obj_x, obj_y, class_name, int(box.cls[0]))
                        
                        detected_obstacles.append((obj_dist, class_name))

        self.current_obstacle_cmd = "CLEAR"  # no obstacle by default

        current_min_obs_dist = 999.0
        if detected_obstacles:
            # Sort by distance (the closest object ends up at index 0)
            detected_obstacles.sort(key=lambda x: x[0])
            closest_dist, closest_name = detected_obstacles[0]

            current_min_obs_dist = closest_dist

            # Timing: are we close enough for the manoeuvre?
            if closest_dist < self.avoid_trigger_dist:
                
                # only overwrite if we are currently on CLEAR (centre)
                if self.current_obstacle_cmd == "CLEAR":
                    if "red" in closest_name:
                        self.current_obstacle_cmd = "RED"
                    elif "green" in closest_name:
                        self.current_obstacle_cmd = "GREEN"
                        
                    # HERE: we remember in which track section we locked it!
                    self.locked_turn_count = self.turn_count
                    
                    self.get_logger().warn(f'+++ LANE LOCKED IN: {self.current_obstacle_cmd} ({closest_dist:.2f}m) +++')

       

        # 2. split the point cloud into clusters (walls)
        if self.state == 'FOLLOW_LANE':
            total_turned = abs(self.current_yaw - self.yaw_offset)

            min_total_rotation = self.target_turns * 85.0
            if self.turn_count >= self.target_turns and total_turned >= min_total_rotation:
                if self.front_wall is not None:
                    closest_f = self.get_closest_point_in_cluster(self.front_wall)
                    if closest_f is not None and closest_f[3] < 1.80:
                        self.state = 'STOPPED'
                        return

            if abs(self.start_straight_yaw - self.current_yaw) > 75.0:
                self.get_logger().warn(f">>> GYRO TURN DETECTED! Turned too far on the straight (turned: {abs(self.start_straight_yaw - self.current_yaw):.1f} deg) <<<")
                self.start_straight_yaw = self.current_yaw
                self.turn_count += 1
                
            all_clusters = self.get_all_clusters_sorted(point_data)


            validated_clusters = self.validate_clusters_straight(all_clusters)
            # 5. we have a perfect U profile!
            #self.get_logger().info(f"Number of clusters {len(all_clusters)}")
            candidates = self.merge_clusters(all_clusters, validated_clusters)  # the three largest clusters that we have validated

            self.right_wall = candidates[2]
            self.front_wall = candidates[1]
            self.left_wall  = candidates[0]
            
            # the choice of the driving direction used to be here, but we now fix it already in the start phase.
            if self.direction == 'left':
                inner_wall = self.left_wall
                outer_wall = self.right_wall
            elif self.direction == 'right':
                inner_wall = self.right_wall
                outer_wall = self.left_wall

            self.update_avoidance_settings()
            

        if self.state in ['TURN_LEFT', 'TURN_RIGHT']:
            turned_so_far = abs(self.current_yaw - self.start_turn_yaw)
            # during the turn we track the front wall with a dynamic search window (ROI)
            self.front_wall = self.track_front_wall(point_data, self.front_wall)
            candidates = self.validate_clusters_turn(self.front_wall, point_data)  # we only have the front wall that we track
            self.get_logger().info(f"Turn validation: {len([c for c in candidates if c is not None])} candidates found for the front wall.")
            '''cluster = self.get_all_clusters_sorted(point_data)
            front = self.merge_clusters(cluster, [self.front_wall])
            done = self.get_unshadowed_leftovers(cluster, [self.front_wall])
            done.append(self.front_wall)
            candidates = done'''
            # --- 1. THE NEW EMERGENCY ABORT ---
            # is there an object extremely close at the turn exit (< 50cm)?

            current_time = self.get_clock().now().nanoseconds / 1e9
            time_in_turn = current_time - self.turn_start_time

            if current_min_obs_dist < 0.50 and time_in_turn > 1: # turn cooldown for the emergency abort
                self.get_logger().warn(f">>> EMERGENCY ABORT OF THE TURN! Obstacle at {current_min_obs_dist:.2f}m. <<<")
                self.state = 'FOLLOW_LANE'
                self.front_wall = None
                self.turn_count += 1
                self.start_straight_yaw = self.current_yaw
                
                # EXTREMELY IMPORTANT: so that the memory is not cleared in the next frame,
                # we update the lock to the "new" straight!
                self.locked_turn_count = self.turn_count
            
            else:
                angle_to_front_wall = self.get_cluster_angle(self.front_wall)
                if angle_to_front_wall is None:
                    return

                self.get_logger().info(f"Tracking front wall... angle to the driving direction: {angle_to_front_wall:.1f} deg")

                if abs(angle_to_front_wall) < self.turn_exit_angle or turned_so_far > 95.0:
                    self.get_logger().warn(">>> TURN ALMOST DONE! Switching back to FOLLOW_LANE <<<")
                    self.state = 'FOLLOW_LANE'
                    self.front_wall = None
                    if turned_so_far > 45.0:
                        self.get_logger().warn(">>> TURN DONE NORMALLY! Switching back to FOLLOW_LANE <<<")
                        self.turn_count += 1
                    else:
                        self.get_logger().warn(f">>> FAKE TURN DETECTED ({turned_so_far:.1f} deg). Counter ignored! <<<")
                        
                    self.start_straight_yaw = self.current_yaw

                else:
                    if self.state == 'TURN_LEFT':
                        cmd.linear.x = self.turn_speed
                        cmd.angular.z = self.max_turn_angle
                    else:
                        cmd.linear.x = self.turn_speed
                        cmd.angular.z = -self.max_turn_angle

        elif self.state == 'STOPPED':
            cmd.linear.x = 0.0
            cmd.angular.z = 0.0
            self.pub_cmd_vel.publish(cmd)

            # print the final banner only once, then shut down cleanly
            if not getattr(self, '_goal_reached_logged', False):
                self._goal_reached_logged = True
                self.get_logger().info("")
                self.get_logger().info("  +==============================================+")
                self.get_logger().info("  |              [OK] GOAL REACHED               |")
                self.get_logger().info(f"  |   {self.turn_count:>3} turns mastered cleanly.              |")
                self.get_logger().info("  |   Stopping the robot.                        |")
                self.get_logger().info("  +----------------------------------------------+")
                self.get_logger().info("")

                # trigger the shutdown only once; main() cleans up the node.
                if rclpy.ok():
                    rclpy.shutdown()
            return

        elif self.state == 'STARTING':

            self.get_logger().info("Starting the robot... Evaluating the driving direction and calibrating the gyro.")

            all_clusters = self.get_all_clusters_sorted(point_data)


            validated_clusters = self.validate_clusters_straight(all_clusters)
            # 5. we have a perfect U profile!
            #self.get_logger().info(f"Number of clusters {len(all_clusters)}")
            candidates = self.merge_clusters(all_clusters, validated_clusters)  # the three largest clusters that we have validated

            self.right_wall = candidates[2]
            self.front_wall = candidates[1]
            self.left_wall  = candidates[0]
            if self.direction is None:
            # we strictly need both side walls for the length comparison
                if self.left_wall and self.right_wall:
                    # compute the real physical length in metres (Pythagoras)
                    left_len = math.hypot(self.left_wall[0][1] - self.left_wall[-1][1], self.left_wall[0][2] - self.left_wall[-1][2])
                    right_len = math.hypot(self.right_wall[0][1] - self.right_wall[-1][1], self.right_wall[0][2] - self.right_wall[-1][2])
                    
                    self.get_logger().info(f"Scanning track... length left: {left_len:.2f}m, length right: {right_len:.2f}m")
                    
                    # we need a clear difference (e.g. 40 cm) to be sure!
                    if left_len > right_len + 0.30:
                        self.direction = 'right' # right wall is shorter = inner wall = we drive around to the right!
                        self.get_logger().info(">>> LOCK: DIRECTION RIGHT (clockwise) <<<")
                    elif right_len > left_len + 0.30:
                        self.direction = 'left'  # left wall is shorter = inner wall = we drive around to the left!
                        self.get_logger().info(">>> LOCK: DIRECTION LEFT (counter-clockwise) <<<")
                
                else:
                    self.get_logger().info("Driving direction not detected yet... Waiting for both side walls for the analysis.")
                    return

            if not self.imu_ready:
                self.get_logger().info("Waiting for the gyroscope to boot...")
                return  # stop here, do nothing yet!   
            self.yaw_offset = self.current_yaw
            self.start_straight_yaw = self.current_yaw  # gyro calibration: set the current angle as reference

            self.state = 'FOLLOW_LANE'

        else:
        
            pass
        
    
        # ==========================================
        # PATH PLANNING, PID & STATE MACHINE 
        # ==========================================
        if self.current_obstacle_cmd != "CLEAR" and self.turn_count > self.locked_turn_count:
            self.current_obstacle_cmd = "CLEAR"
            self.get_logger().warn(">>> TURN FINISHED: dodge memory cleared, driving in the centre again! <<<")

        self.update_avoidance_settings()

        
        target_x, target_y = self.get_target_point_straight(inner_wall, outer_wall)

        # initialise the Twist message for the ESP
        
        # --- STATE 1: DRIVE STRAIGHT ---
        if self.state == 'FOLLOW_LANE':
            
            # 1. CHECK THE SWITCH CONDITION
            if self.front_wall is not None and self.direction is not None:
                front_dist = self.get_closest_point_in_cluster(self.front_wall)[3]
                
                max_y_inner = 0.0
                if inner_wall and len(inner_wall) > 0:
                    max_y_inner = max(p[2] for p in inner_wall)
                
                if front_dist < 1.20 and max_y_inner < self.max_wall_lenght_for_turn:
                    self.state = f"TURN_{self.direction.upper()}"
                    self.start_turn_yaw = self.current_yaw
                    self.get_logger().warn(f">>> {self.state} INITIATED at {(self.start_turn_yaw - self.yaw_offset):.1f} deg <<<")
                    # IMPORTANT: clear the PID memory for the next straight!
                    self.prev_error = 0.0
                    self.integral_error = 0.0

                    self.turn_start_time = self.get_clock().now().nanoseconds / 1e9
                else:
                    if front_dist < 1.30:
                        self.get_logger().info(f"Waiting for the corner... (inner wall still reaches {max_y_inner:.2f}m ahead)")
            
            # 2. COMPUTE THE PID CONTROLLER
            # error: X deviation of the carrot. Negative X = carrot on the left = steer positive!
            error = -target_x 
            
            # compute integral (with anti-windup so the value does not explode)
            self.integral_error += error
            self.integral_error = max(-1.0, min(1.0, self.integral_error))
            
            # compute derivative (change since the last frame)
            derivative = error - self.prev_error
            self.prev_error = error
            
            # compute the control output (steering command)
            steering_cmd = (self.kp * error) + (self.ki * self.integral_error) + (self.kd * derivative)
            
            # clamp to the ROS limits (-1.0 to 1.0)
            steering_cmd = max(-1.0, min(1.0, steering_cmd))
            
            # 3. SET COMMANDS FOR THE ESP
            cmd.linear.x = self.base_speed
            cmd.angular.z = float(steering_cmd)

        # --- STATE 2: LEFT TURN ---
        elif self.state == 'TURN_LEFT':
            pass

        # --- STATE 3: RIGHT TURN ---
        elif self.state == 'TURN_RIGHT':
            pass


        # --- OUTPUT TO HARDWARE & RVIZ ---
        
        # send commands to the ESP! (test jacked up first!)
        self.pub_cmd_vel.publish(cmd)
        
        if self.state == 'FOLLOW_LANE':
            self.send_sphere(marker_array, m_id=99, x=target_x, y=target_y, color=(0.0, 1.0, 1.0))
        else:
            self.delete_marker(marker_array, 99, ns="target")
            
        self.pub_markers.publish(marker_array)

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


        # 6. go through the three walls and send them to RViz
        if candidates is not None:
            candidates_RVIZ = [c for c in candidates if c is not None]
        else: 
            return
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

def main(args=None):
    rclpy.init(args=args)
    node = WallFollower()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # node cleanup; rclpy.shutdown() may already have happened in the STOPPED state
        if node.context.ok():
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()