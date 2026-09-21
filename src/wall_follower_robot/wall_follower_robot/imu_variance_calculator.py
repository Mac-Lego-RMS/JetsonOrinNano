import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu
import numpy as np
import time
import sys

class ImuDataCollector(Node):
    def __init__(self):
        super().__init__('imu_data_collector')
        
        # Topic-Abonnement
        self.subscription = self.create_subscription(
            Imu,
            '/bno055/imu',
            self.imu_callback,
            10
        )
        
        self.data = []
        self.duration = 20.0
        self.start_time = None
        
        self.get_logger().info('Sammle angular_velocity Daten für 20 Sekunden...')

    def imu_callback(self, msg):
        # Startzeit beim ersten empfangenen Paket setzen
        if self.start_time is None:
            self.start_time = time.time()
            
        current_time = time.time()
        
        # Daten sammeln, solange die Zeit nicht abgelaufen ist
        if (current_time - self.start_time) <= self.duration:
            self.data.append([
                msg.angular_velocity.x,
                msg.angular_velocity.y,
                msg.angular_velocity.z
            ])
        else:
            self.evaluate_and_exit()

    def evaluate_and_exit(self):
        if not self.data:
            self.get_logger().error('Keine Daten empfangen. Überprüfe das Topic.')
            sys.exit(1)
            
        # Konvertierung in NumPy Array (Nx3 Matrix)
        np_data = np.array(self.data)
        
        # Berechnung (axis=0 berechnet die Werte spaltenweise für x, y, z)
        mean_vals = np.mean(np_data, axis=0)
        var_vals = np.var(np_data, axis=0)
        
        print("\n--- Auswertung nach 20 Sekunden ---")
        print(f"Anzahl der Messpunkte: {len(self.data)}")
        print(f"Mittelwert [x, y, z] in rad/s:  {mean_vals}")
        print(f"Varianz    [x, y, z] in rad²/s²: {var_vals}")
        print("-----------------------------------\n")
        
        # Node sauber beenden
        rclpy.shutdown()

def main(args=None):
    rclpy.init(args=args)
    node = ImuDataCollector()
    
    try:
        rclpy.spin(node)
    except Exception as e:
        # Fängt den Shutdown ab, um Fehler im Terminal zu vermeiden
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()