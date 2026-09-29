#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from rclpy.qos import qos_profile_sensor_data
import cv2
import sys
import gc

class CsiCameraPublisher(Node):
    def __init__(self):
        super().__init__('camera_publisher')
        
        self.publisher_ = self.create_publisher(Image, '/camera/image_raw', qos_profile_sensor_data)
        self.bridge = CvBridge()
        
        # Change the call in __init__:
        pipeline = self.gstreamer_pipeline(
            capture_width=640, capture_height=480, # smallest possible native mode
            display_width=640, display_height=360, 
            framerate=20, # 20 FPS is plenty for WRO and saves a lot of RAM
            flip_method=0
        )
        
        self.get_logger().info('Starting CSI camera...')
        self.cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
        
        if not self.cap.isOpened():
            self.get_logger().error('ERROR: could not open the camera! Check the CSI cable.')
            sys.exit(1) # Ends the script hard so it does not keep running
            
        self.error_counter = 0 # counter for failed attempts
        self.timer = self.create_timer(1.0 / 30.0, self.timer_callback)
        self.get_logger().info('Camera node running. Publishing on topic: /camera/image_raw')

    def gstreamer_pipeline(self, capture_width, capture_height, display_width, display_height, framerate, flip_method):
        return (
            "nvarguscamerasrc sensor-id=0 sensor-mode=4 ! "
            "video/x-raw(memory:NVMM), width=(int)1280, height=(int)720, format=(string)NV12, framerate=(fraction)60/1 ! "
            f"nvvidconv flip-method={flip_method} ! "
            f"video/x-raw, width=(int){display_width}, height=(int){display_height}, format=(string)BGRx ! "
            "videoconvert ! "
            "video/x-raw, format=(string)BGR ! "
            # --- THE FIX: HARDWARE THROTTLING ---
            "videorate ! "
            "video/x-raw, framerate=8/1 ! "
            "appsink drop=true max-buffers=1 sync=false"
        )

    def timer_callback(self):
        ret, frame = self.cap.read()
        
        if ret:
            self.error_counter = 0
            msg = self.bridge.cv2_to_imgmsg(frame, encoding="bgr8")
            self.publisher_.publish(msg)
            
            # --- MANUAL MEMORY CLEANUP ---
            # We delete the references explicitly
            del frame
            del msg
            
            # Force the garbage collector every 300 frames (about every 10 s)
            if self.get_clock().now().nanoseconds % 300 == 0:
                gc.collect()
        else:
            self.error_counter += 1
            self.get_logger().warning(f'Error reading the image frame. Attempt {self.error_counter}/10')
            
            # If 10 frames in a row fail, kill the node
            if self.error_counter >= 10:
                self.get_logger().error('Camera blocked permanently. Stopping node for safety reasons!')
                sys.exit(1)

    def destroy_node(self):
        self.get_logger().info('Releasing camera resources...')
        if hasattr(self, 'cap') and self.cap.isOpened():
            self.cap.release()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = CsiCameraPublisher()
    
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except SystemExit:
        pass # Catches our sys.exit(1) from the error counter
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()