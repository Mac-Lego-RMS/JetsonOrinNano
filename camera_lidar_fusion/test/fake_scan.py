"""Synthetischer 360-Grad-Scan auf /scan, nur zum Testen der Fusion-Node."""
import math, rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from rclpy.qos import qos_profile_sensor_data

rclpy.init()
n = Node('fake_scan')
pub = n.create_publisher(LaserScan, '/scan', qos_profile_sensor_data)
N = 720
def tick():
    m = LaserScan()
    m.header.stamp = n.get_clock().now().to_msg()
    m.header.frame_id = 'laser'
    m.angle_min = -math.pi; m.angle_max = math.pi
    m.angle_increment = 2*math.pi/N
    m.range_min = 0.05; m.range_max = 12.0
    m.ranges = [0.6 + 0.3*math.sin(4*(-math.pi + i*2*math.pi/N)) for i in range(N)]
    pub.publish(m)
n.create_timer(0.1, tick)
rclpy.spin(n)
