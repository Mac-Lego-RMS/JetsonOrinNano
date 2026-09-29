#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point

class CoordinateTester(Node):
    def __init__(self):
        super().__init__('coordinate_tester')
        
        # Publisher for our test markers
        self.pub_markers = self.create_publisher(MarkerArray, '/coord_test_markers', 10)
        
        # Timer that fires the markers twice per second
        self.timer = self.create_timer(0.5, self.publish_test_markers)
        
        # IMPORTANT: enter the frame here that you have as "Global Frame" in Foxglove
        self.rviz_frame = 'ldlidar_link' 
        self.get_logger().info('>>> Coordinate tester started! Look at /coord_test_markers in Foxglove <<<')

    def create_arrow(self, m_id, dx, dy, color, text):
        marker = Marker()
        marker.header.frame_id = self.rviz_frame
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "test_axes"
        marker.id = m_id
        marker.type = Marker.ARROW
        marker.action = Marker.ADD
        
        # Start point (0,0) and end point (dx, dy)
        p_start = Point(x=0.0, y=0.0, z=0.0)
        p_end = Point(x=float(dx), y=float(dy), z=0.0)
        marker.points = [p_start, p_end]
        
        # Arrow thickness
        marker.scale.x = 0.05 # shaft
        marker.scale.y = 0.1  # head width
        marker.scale.z = 0.1  # head length
        
        marker.color.r, marker.color.g, marker.color.b = color
        marker.color.a = 1.0
        
        return marker

    def create_text(self, m_id, x, y, text, color):
        marker = Marker()
        marker.header.frame_id = self.rviz_frame
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "test_labels"
        marker.id = m_id
        marker.type = Marker.TEXT_VIEW_FACING
        marker.action = Marker.ADD
        
        marker.pose.position.x = float(x)
        marker.pose.position.y = float(y)
        marker.pose.position.z = 0.2
        marker.scale.z = 0.2 # text size
        
        marker.color.r, marker.color.g, marker.color.b = color
        marker.color.a = 1.0
        marker.text = text
        
        return marker

    def publish_test_markers(self):
        ma = MarkerArray()
        
        # 1. Red arrow for +X (should be front)
        ma.markers.append(self.create_arrow(1, dx=1.0, dy=0.0, color=(1.0, 0.0, 0.0), text="+X"))
        ma.markers.append(self.create_text(11, x=1.1, y=0.0, text="PLUS X (X=1, Y=0)", color=(1.0, 0.0, 0.0)))
        
        # 2. Green arrow for +Y (should be left, but for you it is probably right)
        ma.markers.append(self.create_arrow(2, dx=0.0, dy=1.0, color=(0.0, 1.0, 0.0), text="+Y"))
        ma.markers.append(self.create_text(12, x=0.0, y=1.1, text="PLUS Y (X=0, Y=1)", color=(0.0, 1.0, 0.0)))

        # 3. Blue point for X=1, Y=1 (the quadrant test)
        ma.markers.append(self.create_arrow(3, dx=1.0, dy=1.0, color=(0.0, 0.0, 1.0), text="X=1, Y=1"))
        ma.markers.append(self.create_text(13, x=1.1, y=1.1, text="X=1, Y=1", color=(0.0, 0.0, 1.0)))

        self.pub_markers.publish(ma)

def main(args=None):
    rclpy.init(args=args)
    node = CoordinateTester()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()