#!/usr/bin/env python3
"""
scan_processor node.

Subscribes:
  /scan       sensor_msgs/LaserScan   raw LiDAR
  /ekf/odom   nav_msgs/Odometry       current pose estimate from the EKF

Publishes:
  /wall_matches   robot_msgs/WallMatchArray   matched walls for the EKF update

Runs the wall-extraction pipeline (stages 1-4 from wall_extraction.py) on each
scan and matches the found walls against the fixed map. Approach B: matching
uses the LATEST EKF pose (no interpolation to scan time yet). The scan
timestamp is carried in the output header so the EKF can place the update
correctly in its own timeline.
"""
import numpy as np

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry

from robot_msgs.msg import WallMatch, WallMatchArray

from ekf.wall_extraction import (
    scan_to_points, cluster_points, merge_wraparound, split_at_corners,
    fit_wall_hnf, lidar_to_base_link, match_walls,
)

# Fixed map for start position 5, CCW. Walls in the map frame (alpha, d),
# normal pointing into the field interior, d negative. RELATIVE TO REAR AXLE.
# TODO: replace with map generated from field geometry for full-track running.
MAP_WALLS = [
    (np.radians(-90.0), -0.50),   # inner band, left
    (np.radians(+90.0), -0.50),   # outer band, right
    (np.radians(180.0), -1.84),   # front band ahead (base_link frame)
]


def yaw_from_quaternion(q):
    """Extract yaw (z-rotation) from a geometry_msgs quaternion."""
    siny = 2.0 * (q.w * q.z + q.x * q.y)
    cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return np.arctan2(siny, cosy)


class ScanProcessor(Node):
    def __init__(self):
        super().__init__('scan_processor')
        self.pose = (0.0, 0.0, 0.0)          # latest EKF pose (x, y, theta)

        self.create_subscription(LaserScan, '/scan', self.scan_cb, 10)
        self.create_subscription(Odometry, '/ekf/odom', self.pose_cb, 10)
        self.pub = self.create_publisher(WallMatchArray, '/wall_matches', 10)

    def pose_cb(self, msg):
        # keep only the latest pose (approach B)
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        theta = yaw_from_quaternion(msg.pose.pose.orientation)
        self.pose = (x, y, theta)

    def scan_cb(self, msg):
        pts = scan_to_points(msg)
        clusters = merge_wraparound(cluster_points(pts))
        # split any cluster that contains a corner (L-shape) into straight walls
        split = []
        for c in clusters:
            split.extend(split_at_corners(c))
        clusters = split

        measured = []
        for c in clusters:
            hnf = fit_wall_hnf(c)
            if hnf is not None:
                measured.append(lidar_to_base_link(*hnf))

        matches = match_walls(measured, MAP_WALLS, self.pose, d_tol=0.12)

        out = WallMatchArray()
        out.header = msg.header          # carry the scan stamp + frame_id
        for m in matches:
            wm = WallMatch()
            wm.header = msg.header
            wm.alpha_meas = float(m['measured'][0])
            wm.d_meas = float(m['measured'][1])
            wm.alpha_map = float(m['map'][0])
            wm.d_map = float(m['map'][1])
            out.matches.append(wm)

        self.pub.publish(out)


def main():
    rclpy.init()
    rclpy.spin(ScanProcessor())


if __name__ == '__main__':
    main()