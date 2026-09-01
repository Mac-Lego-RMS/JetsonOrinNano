#!/usr/bin/env python3
"""
scan_processor node with automatic start-position detection.

DETECTING: runs start detection on each scan and votes over START_VOTES scans.
On a confident majority it commits the map and switches to RUNNING.

  race_mode 'obstacle': commits the full generated field map for the detected
                        position (direction defaults to CW, resolved at the
                        first corner).
  race_mode 'open':     inner-band geometry is unknown, so it commits a reduced
                        3-wall start map (left/right/front) built from the
                        distances averaged over the winning-position votes.

race_mode is a ROS parameter (default 'obstacle').

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

from std_msgs.msg import Float64, String
from rclpy.qos import QoSProfile, DurabilityPolicy

from ekf.direction_detection import detect_direction

from ekf.wall_extraction import scan_to_points, cluster_points, merge_wraparound, split_at_corners, fit_wall_hnf, lidar_to_base_link, match_walls
from ekf.field_map import generate_map, start_map_3wall, outer_box_map, START_POSES_CW, START_POSES_CCW
from ekf.start_detection import detect_start_obstacle, detect_start_open

from geometry_msgs.msg import Point
from robot_msgs.msg import CornerGeometry, WallHNF

START_VOTES = 5             # scans to vote over before committing the map
DIRECTION_VOTES = 5        # confident, agreeing scans before latching direction

def yaw_from_quaternion(q):
    siny = 2.0 * (q.w * q.z + q.x * q.y)
    cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return np.arctan2(siny, cosy)


class ScanProcessor(Node):
    def __init__(self):
        super().__init__('scan_processor')
        self.race_mode = self.declare_parameter(
            'race_mode', 'obstacle').get_parameter_value().string_value

        self.pose = (0.0, 0.0, 0.0)
        self.map_walls = None            # set once detection commits
        self.front_wall_x = None         # front wall x in the map frame (for corner stop)
        self.votes = []                  # (position, front_d, left_d, right_d) per valid scan

        self.direction = None            # latched CW/CCW once DIRECTION_VOTES confident scans
        self.dir_votes = []            # recent direction votes (for majority vote)
        self.lane_width = None                  # set from the start-detection result (obstacle challenge only)
        self.position = None

        self.create_subscription(LaserScan, '/scan', self.scan_cb, 10)
        self.create_subscription(Odometry, '/ekf/odom', self.pose_cb, 10)
        self.pub = self.create_publisher(WallMatchArray, '/wall_matches', 10)
        latched = QoSProfile(depth=1)
        latched.durability = DurabilityPolicy.TRANSIENT_LOCAL
        self.front_wall_pub = self.create_publisher(Float64, '/front_wall_x', latched)
        self.direction_pub = self.create_publisher(String, '/race_direction', latched)
        self.corner_pub = self.create_publisher(CornerGeometry, '/corner_geometry', latched)
        self.get_logger().info(
            f'start detection running (mode={self.race_mode})...')

    @staticmethod
    def _front_wall_x_from_map(map_walls):
        """Map-frame x of the front wall (alpha ~ +-180). Works for dict walls
        and legacy (alpha, d) tuples. Returns |d| of that wall, or None."""
        for w in map_walls:
            alpha = w['alpha'] if isinstance(w, dict) else w[0]
            d = w['d'] if isinstance(w, dict) else w[1]
            if abs(abs(alpha) - np.pi) < np.radians(30.0):
                return abs(d)
        return None

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

    def _detect(self, measured):
        if self.race_mode == 'open':
            return detect_start_open(measured)
        return detect_start_obstacle(measured)

    def _commit_obstacle(self, position):
        # direction defaults to CW; resolved at the first corner later
        self.position = position
        start_pose = START_POSES_CW[f'pos{position}']
        self.map_walls = generate_map(start_pose)
        # front wall = the alpha ~ +-180 wall; its endpoints give map-frame x
        self.lane_width = 1.0
        self.front_wall_x = self._front_wall_x_from_map(self.map_walls)
        self.get_logger().info(
            f'[obstacle] start position {position} -> map committed '
            f'({len(self.map_walls)} walls, CW default)')
        self._publish_front_wall_x()

    def _commit_open(self, position, front_d, left_d, right_d):
        self.position = position
        self.map_walls = start_map_3wall(front_d, left_d, right_d)
        self.lane_width = left_d + right_d
        self.front_wall_x = front_d          # robot starts at x=0, front ahead at +front_d
        self.get_logger().info(
            f'[open] start position {position} -> 3-wall map committed '
            f'(front={front_d:.2f}, left={left_d:.2f}, right={right_d:.2f})')
        self._publish_front_wall_x()

    def _commit(self):
        """Pick the winning position and commit the mode-specific map."""
        positions = [v[0] for v in self.votes]
        winner, _ = Counter(positions).most_common(1)[0]
        win = [v for v in self.votes if v[0] == winner]

        if self.race_mode == 'open':
            # side distances only exist / are needed in open mode
            front_d = float(np.mean([v[1] for v in win]))
            left_d = float(np.mean([v[2] for v in win]))
            right_d = float(np.mean([v[3] for v in win]))
            self._commit_open(winner, front_d, left_d, right_d)
        else:
            self._commit_obstacle(winner)

    def scan_cb(self, msg):
        measured = self._extract(msg)

        # --- DETECTING: vote on the start position ---
        if self.map_walls is None:
            res = self._detect(measured)
            if res['valid']:
                # obstacle detect has no left_d/right_d fields -> fall back to None-safe
                left_d = res.get('left_d')
                right_d = res.get('right_d')
                self.votes.append((res['position'], res['front_dist'],
                                   left_d, right_d))
            if len(self.votes) >= START_VOTES:
                self._commit()
            return

        # --- RUNNING: normal matching ---
        matches = match_walls(measured, self.map_walls, self.pose, d_tol=0.12)
        self._update_direction(measured)
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

    def _publish_front_wall_x(self):
        if self.front_wall_x is not None:
            self.front_wall_pub.publish(Float64(data=float(self.front_wall_x)))
            self.get_logger().info(f'published front_wall_x = {self.front_wall_x:.3f}')

    def _update_direction(self, measured):
        """Run direction detection each scan; latch once after DIRECTION_VOTES
        confident, agreeing scans. Frozen after latch (direction is fixed for
        the whole run) -- no reset."""
        if self.direction is not None:
            return                                    # already latched -> frozen
 
        res = detect_direction(measured, lane_width=self.lane_width)
        if not res['confident']:
            return
        self.dir_votes.append(res['direction'])
        # keep only the most recent DIRECTION_VOTES votes
        if len(self.dir_votes) > DIRECTION_VOTES:
            self.dir_votes.pop(0)
        # latch only if the last DIRECTION_VOTES all agree
        if len(self.dir_votes) == DIRECTION_VOTES and len(set(self.dir_votes)) == 1:
            self.direction = self.dir_votes[0]
            self.direction_pub.publish(String(data=self.direction))
            self.direction_pub.publish(String(data=self.direction))
            self.get_logger().info(f'race direction latched: {self.direction}')
            if self.race_mode == 'open':
                self._publish_corner_geometry(self._open_start_pose())
            else:
                poses = START_POSES_CW if self.direction == 'CW' else START_POSES_CCW
                self._publish_corner_geometry(poses[f'pos{self.position}'])

    def _publish_corner_geometry(self, start_pose):
        corners, walls, edge = outer_box_map(start_pose)
        msg = CornerGeometry()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'map'
        for i in range(4):
            p = Point()
            p.x = float(corners[i][0])
            p.y = float(corners[i][1])
            p.z = 0.0
            msg.corners[i] = p
            w = WallHNF()
            w.nx, w.ny, w.d = walls[i]
            msg.walls[i] = w
        msg.edge_length = float(edge)
        self.corner_pub.publish(msg)
        self.get_logger().info('published corner_geometry (outer box)')

    def _open_start_pose(self):
        """Centred field start pose for the open challenge, given the detected
        position, measured lane width, and latched direction.
        x from position, y = lane centre, theta from direction."""
        # x along the lane: pos1 front 1.45 -> |x|=0.05 ; pos2 front 1.95 -> 0.45
        x_mag = 0.05 if self.position == 1 else 0.45
        # outer wall fixed at y=1.5; lane centre sits lane_width/2 inside it
        y_centre = 1.5 - self.lane_width / 2.0
        if self.direction == 'CW':
            return (x_mag, y_centre, 0.0)
        else:  # CCW: mirrored (faces -x), x flips sign
            return (-x_mag, y_centre, np.pi)


def main():
    rclpy.init()
    rclpy.spin(ScanProcessor())


if __name__ == '__main__':
    main()