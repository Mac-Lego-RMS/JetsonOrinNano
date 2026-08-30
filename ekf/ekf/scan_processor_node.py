#!/usr/bin/env python3
"""
scan_processor node with automatic start-position detection.

On start the node is in DETECTING: it runs start detection on each scan and
votes over START_VOTES scans. Once a confident majority is reached it generates
the map for that position (direction defaults to CW, resolved later at the
first corner) and switches to RUNNING, matching normally from then on.

Subscribes: /scan, /ekf/odom
Publishes:  /wall_matches
"""
import numpy as np
from collections import Counter

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry

from robot_msgs.msg import WallMatch, WallMatchArray

from ekf.wall_extraction import (
    scan_to_points, cluster_points, merge_wraparound, split_at_corners,
    fit_wall_hnf, lidar_to_base_link, match_walls,
)
from ekf.field_map import generate_map, START_POSES_CW, START_POSES_CCW
from ekf.start_detection import detect_start_obstacle

RACE_MODE = 'obstacle'      # TODO: make this a ROS parameter / launch arg
START_VOTES = 5             # scans to vote over before committing the map


def yaw_from_quaternion(q):
    siny = 2.0 * (q.w * q.z + q.x * q.y)
    cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return np.arctan2(siny, cosy)


class ScanProcessor(Node):
    def __init__(self):
        super().__init__('scan_processor')
        self.pose = (0.0, 0.0, 0.0)
        self.map_walls = None            # set once detection commits
        self.votes = []                  # detected positions during DETECTING

        self.create_subscription(LaserScan, '/scan', self.scan_cb, 10)
        self.create_subscription(Odometry, '/ekf/odom', self.pose_cb, 10)
        self.pub = self.create_publisher(WallMatchArray, '/wall_matches', 10)
        self.get_logger().info(f'start detection running (mode={RACE_MODE})...')

    def pose_cb(self, msg):
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        theta = yaw_from_quaternion(msg.pose.pose.orientation)
        self.pose = (x, y, theta)

    def _extract(self, msg):
        pts = scan_to_points(msg)
        clusters = merge_wraparound(cluster_points(pts))
        split = []
        for c in clusters:
            split.extend(split_at_corners(c))
        measured = []
        for c in split:
            hnf = fit_wall_hnf(c)
            if hnf is not None:
                measured.append(lidar_to_base_link(*hnf))
        return measured

    def _commit_map(self, position):
        # direction defaults to CW; resolved at the first corner later
        key = f'pos{position}'
        start_pose = START_POSES_CW[key]
        self.map_walls = generate_map(start_pose)
        self.get_logger().info(
            f'start position {position} detected -> map committed '
            f'({len(self.map_walls)} walls, CW default)')

    def scan_cb(self, msg):
        measured = self._extract(msg)

        # --- DETECTING: vote on the start position ---
        if self.map_walls is None:
            res = detect_start_obstacle(measured)
            if res['valid']:
                self.votes.append(res['position'])
            if len(self.votes) >= START_VOTES:
                winner, _ = Counter(self.votes).most_common(1)[0]
                self._commit_map(winner)
            return

        # --- RUNNING: normal matching ---
        matches = match_walls(measured, self.map_walls, self.pose, d_tol=0.12)

        out = WallMatchArray()
        out.header = msg.header
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