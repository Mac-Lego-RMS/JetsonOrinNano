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
        self.img_sub = self.create_subscription(
            Image, 
            '/camera/image_raw', 
            self.image_callback, 
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
        self.cmd_pub = self.create_publisher(String, '/obstacle_cmd', 10)
        self.avoid_trigger_dist = 0.85 # Threshold: dodge from 85cm in front of the obstacle
        
        # Variables for the fusion
        self.last_scan = None
        self.camera_fov = 160.0  # Your field of view
        self.get_logger().info('YOLO lidar fusion node started.')

    def scan_callback(self, msg):
        # Stores the most recent scan for combining with the image   
        #self.get_logger().info('Lidar data received!')
        self.last_scan = msg

    def image_callback(self, msg):
        if self.last_scan is None: return

        cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        image_width = cv_image.shape[1]     
        results = self.model(cv_image, verbose=False)

        # --- NEW: THE DEBUG IMAGE IS CREATED AND PUBLISHED HERE ---
        # 1. Draw bounding boxes and labels onto the image
        annotated_frame = results[0].plot()

        # 2. Convert the OpenCV image back into a ROS message
        debug_msg = self.bridge.cv2_to_imgmsg(annotated_frame, encoding="bgr8")
        # 3. Publish the image on the topic /camera/yolo_debug
        self.pub_debug_img.publish(debug_msg)
        # -------------------------------------------------------------
        
        detected_obstacles = []

        for r in results:
            for box in r.boxes:
                if float(box.conf[0]) > 0.8:
                    self.get_logger().info(f'Object detected: {self.model.names[int(box.cls[0])]} with confidence {box.conf[0]:.2f}')
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
                        
                        # Store the obstacle for the decision making
                        detected_obstacles.append((actual_dist, class_name))

        # --- THE COMMAND LOGIC ---
        cmd_msg = String()
        cmd_msg.data = "CLEAR" # Default: we stay on the normal racing line

        if detected_obstacles:
            # Sort by distance (the closest object ends up at index 0)
            detected_obstacles.sort(key=lambda x: x[0])
            closest_dist, closest_name = detected_obstacles[0]

            # Timing: are we close enough for the manoeuvre?
            if closest_dist < self.avoid_trigger_dist:
                if "red" in closest_name:
                    cmd_msg.data = "AVOID_RIGHT"
                elif "green" in closest_name:
                    cmd_msg.data = "AVOID_LEFT"
                
                self.get_logger().info(f'+++ COMMAND: {cmd_msg.data} ({closest_dist:.2f}m) +++')

        self.cmd_pub.publish(cmd_msg)

    def get_lidar_distance(self, camera_angle_rad):
        if self.last_scan is None: return None

        msg = self.last_scan
        num_points = len(msg.ranges)
        
        # 1. Convert the search range into indices
        # We look about 25 degrees left and right of the camera angle
        search_rad = math.radians(25)
        angle_min = (camera_angle_rad - search_rad)
        angle_max = (camera_angle_rad + search_rad)
        
        clusters = []
        current_cluster = []
        
        # 2. We walk through the array sequentially
        for i in range(num_points):
            dist = msg.ranges[i]
            if math.isnan(dist) or math.isinf(dist) or dist < 0.15 or dist > 3.5:
                continue
                
            angle = (msg.angle_min + i * msg.angle_increment)
            # Normalise angle to (-pi to pi) or (0 to 2pi) depending on the lidar
            angle = math.atan2(math.sin(angle), math.cos(angle))
            cam_angle_norm = math.atan2(math.sin(camera_angle_rad), math.cos(camera_angle_rad))

            # Only look at points in the camera field of view
            if abs(self.angle_diff(angle, cam_angle_norm)) < search_rad:
                if not current_cluster:
                    current_cluster.append((angle, dist))
                else:
                    # THE DECISIVE JUMP:
                    # Compare with the direct neighbour in the array
                    prev_dist = current_cluster[-1][1]
                    
                    # If the jump between two neighbours is > 15cm, 
                    # a new object starts (pillar ends or wall begins)
                    if abs(dist - prev_dist) < 0.15:
                        current_cluster.append((angle, dist))
                    else:
                        clusters.append(current_cluster)
                        current_cluster = [(angle, dist)]
        
        if current_cluster: clusters.append(current_cluster)

        # 3. Find the narrowest/closest cluster that fits the pillar
        best_c = None
        min_dist = 4.0
        
        for c in clusters:
            # A pillar typically has 4-10 neighbouring points
            if 3 <= len(c) <= 12:
                avg_dist = sum(p[1] for p in c) / len(c)
                avg_angle = sum(p[0] for p in c) / len(c)
                
                # We take the cluster that is closest (foreground principle)
                if avg_dist < min_dist:
                    min_dist = avg_dist
                    best_c = (avg_dist, avg_angle)

        return best_c # Returns (distance, angle)

    def angle_diff(self, a, b):
        """Computes the smallest difference between two angles (rad)."""
        # Makes sure the difference also stays correct across the 0/360 deg boundary
        return math.atan2(math.sin(a - b), math.cos(a - b))
    
    def publish_marker(self, angle_rad, dist, name, class_id):
        marker = Marker()
        marker.header.frame_id = "ldlidar_link"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "yolo_obstacles"
        marker.id = class_id
        marker.type = Marker.CYLINDER
        marker.action = Marker.ADD
        
        # The position is finally computed here
        marker.pose.position.x = dist * math.cos(angle_rad)
        marker.pose.position.y = dist * math.sin(angle_rad)
        marker.pose.position.z = self.lidar_height_offset # height correction
        
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