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
                        During round 1 it measures the lane width of each
                        straight and reconstructs the inner band at the end of
                        the round, extending the matching map.

race_mode is a ROS parameter (default 'obstacle').

Subscribes: /scan, /ekf/odom, /round1_controller/lap_state (latched)
Publishes:  /wall_matches
            /front_wall_x      (latched) front wall x in the map frame
            /race_direction    (latched) CW / CCW, latched once, then frozen
            /corner_geometry   (latched) outer box, published at direction latch
            /inner_geometry    (latched) inner band, published after round 1
                                         (open mode only; edge_length unused)
"""
import numpy as np
from collections import Counter

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry

from robot_msgs.msg import WallMatch, WallMatchArray

from std_msgs.msg import Float64, String, Int32MultiArray
from rclpy.qos import QoSProfile, DurabilityPolicy

from geometry_msgs.msg import Point
from robot_msgs.msg import CornerGeometry, WallHNF

from ekf.ekf import wrap
from ekf.direction_detection import detect_direction
from ekf.wall_extraction import (
    scan_to_points, cluster_points, merge_wraparound, split_at_corners,
    fit_wall_hnf, lidar_to_base_link, match_walls,
)
from ekf.field_map import (
    generate_map, start_map_3wall, outer_box_map, outer_walls_map,
    inner_band_from_widths, START_POSES_CW, START_POSES_CCW,
)
from ekf.start_detection import detect_start_obstacle, detect_start_open

START_VOTES = 5                # scans to vote over before committing the map
DIRECTION_VOTES = 5            # confident, agreeing scans before latching direction
MAP_SWITCH_MAX_X = 0.30        # only switch the map while still near the start
LANE_NOMINALS = (0.60, 1.00)   # plausible lane widths (open challenge)
LANE_PLAUS_TOL = 0.15          # measurement must be within this of a nominal
MIN_WIDTH_SAMPLES = 10         # samples per straight before it counts as learned


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
        self.map_walls = None        # set once detection commits
        self.front_wall_x = None     # front wall x in the map frame (corner stop)
        self.votes = []              # (position, front_d, left_d, right_d) per scan
        self.position = None         # detected start position (1 or 2)
        self.lane_width = None       # start-straight lane width

        self.direction = None        # latched CW/CCW, then frozen
        self.dir_votes = []          # recent confident direction votes

        # --- round-1 lane-width learning (open mode) ---
        self.lap_state = None        # [corner_idx, corner_count, lap]
        self.width_samples = {}      # outer wall index -> [measured widths]
        self.inner_walls = None      # set once the inner band is learned

        latched = QoSProfile(depth=1)
        latched.durability = DurabilityPolicy.TRANSIENT_LOCAL

        self.create_subscription(LaserScan, '/scan', self.scan_cb, 10)
        self.create_subscription(Odometry, '/ekf/odom', self.pose_cb, 10)
        self.create_subscription(Int32MultiArray,
                                 '/round1_controller/lap_state',
                                 self.lap_state_cb, latched)

        self.pub = self.create_publisher(WallMatchArray, '/wall_matches', 10)
        self.front_wall_pub = self.create_publisher(Float64, '/front_wall_x', latched)
        self.direction_pub = self.create_publisher(String, '/race_direction', latched)
        self.corner_pub = self.create_publisher(CornerGeometry, '/corner_geometry', latched)
        self.inner_pub = self.create_publisher(CornerGeometry, '/inner_geometry', latched)

        self.get_logger().info(
            f'start detection running (mode={self.race_mode})...')

    # ------------------------------------------------------------------ #
    # callbacks
    # ------------------------------------------------------------------ #

    def pose_cb(self, msg):
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        theta = yaw_from_quaternion(msg.pose.pose.orientation)
        self.pose = (x, y, theta)

    def lap_state_cb(self, msg):
        """Track [corner_idx, corner_count, lap]. When lap flips to 1, round 1
        is done -> try to reconstruct the inner band from the learned widths."""
        prev = self.lap_state
        self.lap_state = list(msg.data)
        if prev is not None and prev[2] == 0 and self.lap_state[2] >= 1:
            self._commit_inner_band()

    def scan_cb(self, msg):
        measured = self._extract(msg)

        # --- DETECTING: vote on the start position ---
        if self.map_walls is None:
            res = self._detect(measured)
            if res['valid']:
                # obstacle detect has no left_d/right_d fields -> None-safe
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
        self._learn_lane_width(measured)

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

    # ------------------------------------------------------------------ #
    # extraction / start detection
    # ------------------------------------------------------------------ #

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

    def _commit_obstacle(self, position):
        # direction defaults to CW; resolved at the first corner later
        self.position = position
        self.lane_width = 1.0
        start_pose = START_POSES_CW[f'pos{position}']
        self.map_walls = generate_map(start_pose)
        self.front_wall_x = self._front_wall_x_from_map(self.map_walls)
        self.get_logger().info(
            f'[obstacle] start position {position} -> map committed '
            f'({len(self.map_walls)} walls, CW default)')
        self._publish_front_wall_x()

    def _commit_open(self, position, front_d, left_d, right_d):
        self.position = position
        self.lane_width = left_d + right_d
        self.map_walls = start_map_3wall(front_d, left_d, right_d)
        self.front_wall_x = front_d   # robot starts at x=0, front at +front_d
        self.get_logger().info(
            f'[open] start position {position} -> 3-wall map committed '
            f'(front={front_d:.2f}, left={left_d:.2f}, right={right_d:.2f})')
        self._publish_front_wall_x()

    # ------------------------------------------------------------------ #
    # direction latch + map switch
    # ------------------------------------------------------------------ #

    def _update_direction(self, measured):
        """Run direction detection each scan; latch once after DIRECTION_VOTES
        confident, agreeing scans. Frozen after latch (direction is fixed for
        the whole run) -- no reset."""
        if self.direction is not None:
            return                                # already latched -> frozen

        res = detect_direction(measured, lane_width=self.lane_width)
        if not res['confident']:
            return
        self.dir_votes.append(res['direction'])
        if len(self.dir_votes) > DIRECTION_VOTES:
            self.dir_votes.pop(0)
        # latch only if the last DIRECTION_VOTES all agree
        if len(self.dir_votes) == DIRECTION_VOTES and len(set(self.dir_votes)) == 1:
            self.direction = self.dir_votes[0]
            self.direction_pub.publish(String(data=self.direction))
            self.get_logger().info(f'race direction latched: {self.direction}')

            # switch the EKF matching map to the latched direction (jump-guarded)
            self._switch_map_to_direction()

            # publish the outer box for the controller (both modes)
            if self.race_mode == 'open':
                self._publish_corner_geometry(self._open_start_pose())
            else:
                poses = START_POSES_CW if self.direction == 'CW' else START_POSES_CCW
                self._publish_corner_geometry(poses[f'pos{self.position}'])

    def _switch_map_to_direction(self):
        """Switch the EKF matching map to the latched direction. Safe only near
        the start (straight); guarded against a late latch that would jump."""
        if abs(self.pose[0]) > MAP_SWITCH_MAX_X:
            self.get_logger().warn(
                f'direction latched late (x={self.pose[0]:.2f} m) -- NOT '
                f'switching map to avoid a pose jump; check latch timing')
            return

        poses = START_POSES_CW if self.direction == 'CW' else START_POSES_CCW
        start_pose = (poses[f'pos{self.position}'] if self.race_mode != 'open'
                      else self._open_start_pose())

        if self.race_mode == 'open':
            self.map_walls = outer_walls_map(start_pose)   # outer rim only
        else:
            self.map_walls = generate_map(start_pose)      # full field map
        self.front_wall_x = self._front_wall_x_from_map(self.map_walls)
        self.get_logger().info(
            f'matching map switched to {self.direction} '
            f'({len(self.map_walls)} walls)')

    def _open_start_pose(self):
        """Centred field start pose for the open challenge, from the detected
        position, measured lane width, and latched direction."""
        # x along the lane: pos1 front 1.45 -> |x|=0.05 ; pos2 front 1.95 -> 0.45
        x_mag = 0.05 if self.position == 1 else 0.45
        # outer wall fixed at y=1.5; lane centre sits lane_width/2 inside it
        y_centre = 1.5 - self.lane_width / 2.0
        if self.direction == 'CW':
            return (x_mag, y_centre, 0.0)
        return (-x_mag, y_centre, np.pi)      # CCW: faces -x, x flips sign

    # ------------------------------------------------------------------ #
    # round-1 lane-width learning (open mode)
    # ------------------------------------------------------------------ #

    def _current_outer_wall_index(self):
        """Which outer wall the robot is driving along right now.

        corner_idx names the corner AHEAD (end of the current straight).
        CCW (indices ascending): came from corner k-1 -> wall k-1.
        CW  (indices descending): came from corner k+1 -> wall k.
        """
        if self.lap_state is None or self.direction is None:
            return None
        k = self.lap_state[0]
        return (k - 1) % 4 if self.direction == 'CCW' else k % 4

    def _learn_lane_width(self, measured):
        """Collect lane-width samples per straight during round 1 (open mode)."""
        if self.race_mode != 'open' or self.inner_walls is not None:
            return
        if self.lap_state is None or self.lap_state[2] != 0:
            return                                # only during round 1
        wall_idx = self._current_outer_wall_index()
        if wall_idx is None:
            return

        # need one clean wall on each side (alpha ~ -90 left, +90 right)
        left_d = right_d = None
        for w in measured:
            a, d = w[0], w[1]
            if abs(wrap(a + np.radians(90.0))) < np.radians(25.0):
                if left_d is None or abs(d) < abs(left_d):
                    left_d = d
            elif abs(wrap(a - np.radians(90.0))) < np.radians(25.0):
                if right_d is None or abs(d) < abs(right_d):
                    right_d = d
        if left_d is None or right_d is None:
            return

        width = abs(left_d) + abs(right_d)
        # plausibility: must be near a legal width (rejects corner geometry)
        if min(abs(width - n) for n in LANE_NOMINALS) > LANE_PLAUS_TOL:
            return

        self.width_samples.setdefault(wall_idx, []).append(width)

    def _commit_inner_band(self):
        """Reconstruct the inner band from the learned widths, extend the
        matching map, and publish it. Publishes nothing if any straight is
        missing -- better no inner geometry than a wrong one."""
        if self.race_mode != 'open' or self.inner_walls is not None:
            return
        widths = {}
        for i in range(4):
            s = self.width_samples.get(i, [])
            if len(s) < MIN_WIDTH_SAMPLES:
                self.get_logger().warn(
                    f'lane width for straight {i}: only {len(s)} samples '
                    f'-> inner band NOT committed')
                return
            widths[i] = float(np.median(s))       # median: robust to outliers

        result = inner_band_from_widths(self._open_start_pose(), widths)
        if result is None:
            self.get_logger().warn('inner band reconstruction failed (degenerate)')
            return
        inner_walls, inner_corners = result

        self.inner_walls = inner_walls
        self.map_walls = list(self.map_walls) + inner_walls
        wtxt = ', '.join(f'{i}:{widths[i]:.3f}' for i in range(4))
        self.get_logger().info(
            f'inner band learned ({wtxt}) -> map extended to '
            f'{len(self.map_walls)} walls')
        self._publish_inner_geometry(inner_walls, inner_corners)

    # ------------------------------------------------------------------ #
    # publishing
    # ------------------------------------------------------------------ #

    def _publish_front_wall_x(self):
        if self.front_wall_x is not None:
            self.front_wall_pub.publish(Float64(data=float(self.front_wall_x)))
            self.get_logger().info(
                f'published front_wall_x = {self.front_wall_x:.3f}')

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

    def _publish_inner_geometry(self, walls, corners):
        msg = CornerGeometry()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'map'
        for i in range(4):
            p = Point()
            p.x, p.y, p.z = float(corners[i][0]), float(corners[i][1]), 0.0
            msg.corners[i] = p
            w = WallHNF()
            w.nx = float(np.cos(walls[i]['alpha']))
            w.ny = float(np.sin(walls[i]['alpha']))
            w.d = float(walls[i]['d'])
            msg.walls[i] = w
        msg.edge_length = 0.0     # inner band is a rectangle: no single edge
        self.inner_pub.publish(msg)
        self.get_logger().info('published inner_geometry')

    # ------------------------------------------------------------------ #
    # helpers
    # ------------------------------------------------------------------ #

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


def main():
    rclpy.init()
    rclpy.spin(ScanProcessor())


if __name__ == '__main__':
    main()