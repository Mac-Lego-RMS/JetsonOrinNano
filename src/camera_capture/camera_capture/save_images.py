#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from rclpy.qos import qos_profile_sensor_data  # <-- ADD THIS LINE
import cv2
import threading
import os
import time

class ImageSaver(Node):
    def __init__(self):
        super().__init__('image_saver_node')
        # Adjust the topic name if yours is called differently
        self.subscription = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.image_callback,
            qos_profile_sensor_data)  
        self.bridge = CvBridge()
        self.latest_cv_image = None
        
        # Target folder for the images
        self.save_dir = 'dataset_wro'
        os.makedirs(self.save_dir, exist_ok=True)
        self.get_logger().info(f"Node started. Images are saved in ./{self.save_dir}/.")

    def image_callback(self, msg):
        # Converts the ROS message into an OpenCV image and overwrites the previous one
        try:
            self.latest_cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f"Error during image conversion: {e}")

def input_loop(node):
    image_count = 0
    while rclpy.ok():
        # Blocks the thread until Enter is pressed
        input("\n>>> Press [ENTER] in the terminal to save the current image...\n")
        
        if node.latest_cv_image is not None:
            # Uses the Unix timestamp to prevent overwriting
            filename = os.path.join(node.save_dir, f"wro_obstacle_{int(time.time())}.jpg")
            cv2.imwrite(filename, node.latest_cv_image)
            image_count += 1
            print(f"[SUCCESS] Image {image_count} saved: {filename}")
        else:
            print("[WARNING] No image received yet. Check the topic /camera/image_raw.")

def main(args=None):
    rclpy.init(args=args)
    node = ImageSaver()

    # The input call blocks. So that ROS 2 can keep receiving messages in the
    # background (rclpy.spin), we move the input into a thread.
    thread = threading.Thread(target=input_loop, args=(node,), daemon=True)
    thread.start()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down node...")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()