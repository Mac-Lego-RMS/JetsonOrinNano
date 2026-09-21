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
        
        # In der __init__ den Aufruf ändern:
        pipeline = self.gstreamer_pipeline(
            capture_width=640, capture_height=480, # Kleinstmöglicher nativer Modus
            display_width=640, display_height=360, 
            framerate=20, # 20 FPS reicht für WRO völlig und spart massiv RAM
            flip_method=0
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
        return (
            "nvarguscamerasrc sensor-id=0 sensor-mode=4 ! "
            "video/x-raw(memory:NVMM), width=(int)1280, height=(int)720, format=(string)NV12, framerate=(fraction)60/1 ! "
            f"nvvidconv flip-method={flip_method} ! "
            f"video/x-raw, width=(int){display_width}, height=(int){display_height}, format=(string)BGRx ! "
            "videoconvert ! "
            "video/x-raw, format=(string)BGR ! "
            # --- DER FIX: HARDWARE-DROSSELUNG ---
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
            
            # --- MANUELLE SPEICHERREINIGUNG ---
            # Wir löschen die Referenzen explizit
            del frame
            del msg
            
            # Alle 300 Frames (ca. alle 10 Sek.) den Müllsammler zwingen
            if self.get_clock().now().nanoseconds % 300 == 0:
                gc.collect()
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