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
                        At the direction latch the exact field start pose is
                        computed FROM THOSE MEASUREMENTS (not from nominal
                        values), so the absolute map matches the 3-wall map and
                        the pose does not jump.

                        Lane-width learning: the START straight's width is taken
                        straight from the stationary start detection (the best
                        measurement available -- no motion smear, and there may
                        be too little driving distance left on it, especially
                        from position 1). The other straights are measured while
                        driving. The inner band is committed as soon as all four
                        widths are known; learning continues past round 1 if a
                        straight is still missing, because a late commit beats
                        no commit.

race_mode is a ROS parameter (default 'obstacle').

Subscribes: /scan, /ekf/odom, /round1_controller/lap_state (latched)
Publishes:  /wall_matches
            /wall_distances    live [left, right] wall distance in metres,
                               raw from the scan in the base_link frame, NaN
                               where no wall is seen on that side. Published
                               every scan, from the very first one.
            /front_wall_x      (latched) front wall x in the map frame
            /race_direction    (latched) CW / CCW, latched once, then frozen
            /corner_geometry   (latched) outer box, published at direction latch
            /inner_geometry    (latched) inner band, published as soon as all
                                         four lane widths are known
                                         (open mode only; edge_length unused)
"""
import numpy as np
from collections import Counter

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry

from robot_msgs.msg import WallMatch, WallMatchArray

from std_msgs.msg import Float64, String, Int32MultiArray, Float64MultiArray
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
MIN_WIDTH_SAMPLES = 10         # driving samples per straight before it counts
SIDE_ALPHA_TOL = np.radians(25.0)   # how far off +-90 deg a wall may be to
                                    # still count as a side wall

OUTER_HALF = 1.5               # outer wall position in the field frame


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
        self.lane_width = None       # start-straight lane width (stationary)

        # measured start distances, kept for the exact start-pose computation.
        # front_d_meas is separate from front_wall_x because the latter gets
        # overwritten when the matching map is switched.
        self.left_d = None
        self.right_d = None
        self.front_d_meas = None

        self.direction = None        # latched CW/CCW, then frozen
        self.dir_votes = []          # recent confident direction votes

        # --- lane-width learning (open mode) ---
        self.lap_state = None        # [corner_idx, corner_count, lap]
        self.width_samples = {}      # outer wall index -> [driving samples]
        self.width_fixed = {}        # outer wall index -> width known directly
                                     # (start straight, from start detection)
        self.inner_walls = None      # set once the inner band is learned

        latched = QoSProfile(depth=1)
        latched.durability = DurabilityPolicy.TRANSIENT_LOCAL

        self.create_subscription(LaserScan, '/scan', self.scan_cb, 10)
        self.create_subscription(Odometry, '/ekf/odom', self.pose_cb, 10)
        self.create_subscription(Int32MultiArray,
                                 '/round1_controller/lap_state',
                                 self.lap_state_cb, latched)

        self.pub = self.create_publisher(WallMatchArray, '/wall_matches', 10)
        self.wall_dist_pub = self.create_publisher(
            Float64MultiArray, '/wall_distances', 10)
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
        """Track [corner_idx, corner_count, lap]. The inner band is normally
        committed as soon as all four widths are known; a lap change is only a
        fallback check that also reports what is still missing."""
        prev = self.lap_state
        self.lap_state = list(msg.data)
        if prev is not None and self.lap_state[2] > prev[2]:
            self._maybe_commit_inner_band(verbose=True)

    def scan_cb(self, msg):
        measured = self._extract(msg)

        # live side distances -- published from the very first scan, before the
        # map is committed, so the controller always has them
        self._publish_wall_distances(measured)

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

    @staticmethod
    def _side_distances(measured):
        """Nearest wall distance on each side, in metres (positive).

        Left is alpha ~ -90 deg (+y side), right is alpha ~ +90 deg (-y side),
        per the verified convention. Returns (left, right); either may be None
        if no wall was seen on that side -- which is normal in a corner.
        """
        left = right = None
        for w in measured:
            a, d = w[0], w[1]
            if abs(wrap(a + np.radians(90.0))) < SIDE_ALPHA_TOL:
                if left is None or abs(d) < left:
                    left = abs(d)
            elif abs(wrap(a - np.radians(90.0))) < SIDE_ALPHA_TOL:
                if right is None or abs(d) < right:
                    right = abs(d)
        return left, right

    def _publish_wall_distances(self, measured):
        """Live [left, right] side-wall distances, raw from the scan. NaN on a
        side with no wall, so the consumer can tell 'not seen' from a value."""
        left, right = self._side_distances(measured)
        msg = Float64MultiArray()
        msg.data = [float(left) if left is not None else float('nan'),
                    float(right) if right is not None else float('nan')]
        self.wall_dist_pub.publish(msg)

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
        # keep the raw measurements for the exact start-pose computation
        self.left_d = left_d
        self.right_d = right_d
        self.front_d_meas = front_d
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

        if self.race_mode == 'open':
            start_pose = self._open_start_pose()
            self.get_logger().info(
                f'open start pose (from measurements): '
                f'({start_pose[0]:+.3f}, {start_pose[1]:+.3f}, '
                f'{np.degrees(start_pose[2]):+.1f} deg)')
            self.map_walls = outer_walls_map(start_pose)   # outer rim only
        else:
            poses = START_POSES_CW if self.direction == 'CW' else START_POSES_CCW
            start_pose = poses[f'pos{self.position}']
            self.map_walls = generate_map(start_pose)      # full field map

        self.front_wall_x = self._front_wall_x_from_map(self.map_walls)
        self.get_logger().info(
            f'matching map switched to {self.direction} '
            f'({len(self.map_walls)} walls)')

    def _open_start_pose(self):
        """Exact field start pose for the open challenge, from the measured
        distances -- no nominal values.

        The OUTER wall is the only fixed reference (always at +-1.5 in the field
        frame); the lane centre is not, because the inner band varies. So the
        lateral position comes from the distance to the OUTER wall, which is on
        the left for CW and on the right for CCW.

            CW  (faces +x): x = 1.5 - front_d,  y = 1.5 - left_d,   theta = 0
            CCW (faces -x): x = front_d - 1.5,  y = 1.5 - right_d,  theta = pi

        Sanity check against the nominal poses (front 1.95, outer wall 0.5):
            CW  -> (-0.45, 1.0, 0)   == START_POSES_CW['pos2']
            CCW -> (+0.45, 1.0, pi)  == START_POSES_CCW['pos2']

        theta is taken as exactly 0 / pi (robot is placed aligned by hand). The
        measured side-wall alpha could refine this later if needed.
        """
        f = self.front_d_meas
        if self.direction == 'CW':
            return (OUTER_HALF - f, OUTER_HALF - self.left_d, 0.0)
        return (f - OUTER_HALF, OUTER_HALF - self.right_d, np.pi)

    # ------------------------------------------------------------------ #
    # lane-width learning (open mode)
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
        """Learn lane widths per straight (open mode).

        The START straight is taken from the stationary start detection, which
        is both more accurate and available immediately -- important from
        position 1, where there is little driving distance left on it. All other
        straights are sampled while driving. Learning continues past round 1
        until the inner band is committed.
        """
        if self.race_mode != 'open' or self.inner_walls is not None:
            return
        wall_idx = self._current_outer_wall_index()
        if wall_idx is None:
            return

        # start straight: take the stationary measurement directly.
        # corner_count == 0 guarantees no corner has been driven yet, so the
        # current wall index really is the start straight.
        if self.lap_state[1] == 0 and wall_idx not in self.width_fixed:
            self.width_fixed[wall_idx] = float(self.lane_width)
            self.get_logger().info(
                f'start straight {wall_idx}: lane width '
                f'{self.lane_width:.3f} taken from start detection')
            self._maybe_commit_inner_band()
            return
        if wall_idx in self.width_fixed:
            return                                # already known, no sampling

        left, right = self._side_distances(measured)
        if left is None or right is None:
            return

        width = left + right
        # plausibility: must be near a legal width (rejects corner geometry)
        if min(abs(width - n) for n in LANE_NOMINALS) > LANE_PLAUS_TOL:
            return

        self.width_samples.setdefault(wall_idx, []).append(width)
        # publish as early as possible: the moment the last straight is covered
        self._maybe_commit_inner_band()

    def _maybe_commit_inner_band(self, verbose=False):
        """Commit + publish the inner band once every straight's width is known
        -- either fixed (start straight) or with enough driving samples.

        Called after each new measurement (silent) and at a lap change
        (verbose, reports what is still missing). Publishes nothing while a
        straight is missing: better no inner geometry than a wrong one.
        """
        if self.race_mode != 'open' or self.inner_walls is not None:
            return
        widths = {}
        for i in range(4):
            if i in self.width_fixed:
                widths[i] = self.width_fixed[i]
                continue
            s = self.width_samples.get(i, [])
            if len(s) < MIN_WIDTH_SAMPLES:
                if verbose:
                    self.get_logger().warn(
                        f'lane width for straight {i}: only {len(s)} samples '
                        f'-> inner band not committed yet, will keep measuring')
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
        where = (f'lap {self.lap_state[2]}, corner {self.lap_state[0]}'
                 if self.lap_state is not None else 'lap unknown')
        self.get_logger().info(
            f'inner band learned ({wtxt}) at {where} -> map extended to '
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