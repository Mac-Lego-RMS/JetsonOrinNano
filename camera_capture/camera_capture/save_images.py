#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from rclpy.qos import qos_profile_sensor_data  # <-- DIESE ZEILE HINZUFÜGEN
import cv2
import threading
import os
import time

class ImageSaver(Node):
    def __init__(self):
        super().__init__('image_saver_node')
        # Passe den Topic-Namen an, falls er bei dir anders heißt
        self.subscription = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.image_callback,
            qos_profile_sensor_data)  
        self.bridge = CvBridge()
        self.latest_cv_image = None
        
        # Zielordner für die Bilder
        self.save_dir = 'dataset_wro'
        os.makedirs(self.save_dir, exist_ok=True)
        self.get_logger().info(f"Node gestartet. Bilder werden in ./{self.save_dir}/ gespeichert.")

    def image_callback(self, msg):
        # Konvertiert die ROS-Message in ein OpenCV-Bild und überschreibt das vorherige
        try:
            self.latest_cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f"Fehler bei der Bildkonvertierung: {e}")

def input_loop(node):
    image_count = 0
    while rclpy.ok():
        # Blockiert den Thread, bis Enter gedrückt wird
        input("\n>>> Drücke [ENTER] im Terminal, um das aktuelle Bild zu speichern...\n")
        
        if node.latest_cv_image is not None:
            # Nutzt den Unix-Timestamp, um Überschreiben zu verhindern
            filename = os.path.join(node.save_dir, f"wro_obstacle_{int(time.time())}.jpg")
            cv2.imwrite(filename, node.latest_cv_image)
            image_count += 1
            print(f"[ERFOLG] Bild {image_count} gespeichert: {filename}")
        else:
            print("[WARNUNG] Noch kein Bild empfangen. Prüfe den Topic /camera/image_raw.")

def main(args=None):
    rclpy.init(args=args)
    node = ImageSaver()

    # Der Input-Befehl blockiert. Damit ROS 2 weiterhin im Hintergrund 
    # Messages empfangen kann (rclpy.spin), lagern wir die Eingabe in einen Thread aus.
    thread = threading.Thread(target=input_loop, args=(node,), daemon=True)
    thread.start()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Beende Node...")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()