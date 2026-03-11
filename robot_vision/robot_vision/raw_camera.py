#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from rclpy.qos import qos_profile_sensor_data
import cv2
import sys

class CsiCameraPublisher(Node):
    def __init__(self):
        super().__init__('camera_publisher')
        
        self.publisher_ = self.create_publisher(Image, '/camera/image_raw', qos_profile_sensor_data)
        self.bridge = CvBridge()
        
        pipeline = self.gstreamer_pipeline(
            capture_width=1280, capture_height=720, 
            display_width=640, display_height=360, 
            framerate=30, flip_method=0
        )
        
        self.get_logger().info('Starte CSI-Kamera...')
        self.cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
        
        if not self.cap.isOpened():
            self.get_logger().error('FEHLER: Konnte die Kamera nicht öffnen! CSI-Kabel prüfen.')
            sys.exit(1) # Beendet das Skript hart, damit es nicht weiterläuft
            
        self.error_counter = 0 # Zähler für Fehlversuche
        self.timer = self.create_timer(1.0 / 30.0, self.timer_callback)
        self.get_logger().info('Kamera Node läuft. Publiziert auf Topic: /camera/image_raw')

    def gstreamer_pipeline(self, capture_width, capture_height, display_width, display_height, framerate, flip_method):
        """Generiert den GStreamer-String für IMX219 Kameras"""
        return (
            "nvarguscamerasrc ! "
            "video/x-raw(memory:NVMM), "
            f"width=(int){capture_width}, height=(int){capture_height}, "
            f"format=(string)NV12, framerate=(fraction){framerate}/1 ! "
            f"nvvidconv flip-method={flip_method} ! "
            f"video/x-raw, width=(int){display_width}, height=(int){display_height}, format=(string)BGRx ! "
            "videoconvert ! "
            # WICHTIG: max-buffers=1 hinzugefügt, um Speicherlecks zu verhindern!
            "video/x-raw, format=(string)BGR ! appsink drop=true max-buffers=1 sync=false"
        )

    def timer_callback(self):
        ret, frame = self.cap.read()
        
        if ret:
            self.error_counter = 0 # Reset bei Erfolg
            msg = self.bridge.cv2_to_imgmsg(frame, encoding="bgr8")
            self.publisher_.publish(msg)
        else:
            self.error_counter += 1
            self.get_logger().warning(f'Fehler beim Lesen des Bild-Frames. Versuch {self.error_counter}/10')
            
            # Wenn 10 Frames nacheinander scheitern, Node abschießen
            if self.error_counter >= 10:
                self.get_logger().error('Kamera blockiert dauerhaft. Beende Node aus Sicherheitsgründen!')
                sys.exit(1)

    def destroy_node(self):
        self.get_logger().info('Gebe Kamera-Ressourcen frei...')
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
        pass # Fängt unseren sys.exit(1) aus dem Error-Counter auf
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()