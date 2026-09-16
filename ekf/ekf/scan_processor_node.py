#!/usr/bin/env python3
"""
scan_processor node: start detection, map management, and perception outputs
for both challenges. race_mode ('obstacle' | 'open') is a ROS parameter.

  obstacle: commits the full generated field map for the detected position
            (direction defaults to CW, resolved at the first corner). Detects
            traffic signs from the colour-classified cloud.
  open:     inner-band geometry is unknown, so it commits a reduced 3-wall
            start map, computes the exact field start pose from the measured
            distances at the direction latch, learns each straight's lane width
            and reconstructs the inner band. No obstacle detection.

Two layers of obstacle output, mirroring the wall outputs:
  /obstacles_live  raw, per scan, base_link frame -- for REACTING. Needs no map,
                   so it works from the first scan.
  /obstacles       snapped to the seat grid, accumulated, map frame -- for
                   PLANNING. The grid needs the start pose, so it only exists
                   after the direction latch; detections from before that are
                   BUFFERED with their pose and replayed when the grid is
                   built, so start-straight obstacles are not lost.

Subscribes: /scan, /ekf/odom, /round1_controller/lap_state (latched),
            /camera_lidar/colored_scan (obstacle mode only)
Publishes:  /wall_matches
            /wall_distances    live [left, right] side distances, NaN if unseen
            /obstacles_live    raw obstacles, base_link frame, every scan
            /front_wall_x      (latched) front wall x in the map frame
            /race_direction    (latched) CW / CCW, latched once, then frozen
            /corner_geometry   (latched) outer box, at the direction latch
            /inner_geometry    (latched) inner band
            /obstacles         (latched) accumulated obstacle set, map frame
"""
import numpy as np
from collections import Counter, deque

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan, PointCloud2
from nav_msgs.msg import Odometry

from robot_msgs.msg import (WallMatch, WallMatchArray, CornerGeometry, WallHNF,
                            Obstacle, ObstacleArray)

from std_msgs.msg import Float64, String, Int32MultiArray, Float64MultiArray
from rclpy.qos import QoSProfile, DurabilityPolicy

from geometry_msgs.msg import Point

from ekf.ekf import wrap
from ekf.direction_detection import detect_direction
from ekf.obstacle_detection import (detect_obstacles, mask_sectors,
                                    sector_from_robot_point)
from ekf.obstacle_map import ObstacleMap, MIN_SEAT_VOTES
from ekf.wall_extraction import (
    scan_to_points, cluster_points, merge_wraparound, split_at_corners,
    fit_wall_hnf, lidar_to_base_link, match_walls,
)
from ekf.field_map import (
    generate_map, start_map_3wall, outer_box_map, outer_walls_map,
    inner_walls_map, inner_band_from_widths, obstacle_seats_map,
    seat_group_to_wall_index, START_POSES_CW, START_POSES_CCW,
)
from ekf.start_detection import detect_start_obstacle, detect_start_open

START_VOTES = 5                # scans to vote over before committing the map
DIRECTION_VOTES = 5            # confident, agreeing scans before latching
LANE_NOMINALS = (0.60, 1.00)   # plausible lane widths (open challenge)
LANE_PLAUS_TOL = 0.15          # measurement must be within this of a nominal
MIN_WIDTH_SAMPLES = 10         # driving samples per straight before it counts
SIDE_ALPHA_TOL = np.radians(25.0)

# The map switch is jump-free for the whole start straight: the CW and CCW maps
# describe the SAME three walls there and differ only in which side is the inner
# band, which first matters at the corner. So the guard runs up to just short of
# the corner -- driving far, or swerving around an obstacle, must not block it.
MAP_SWITCH_CORNER_MARGIN = 0.40
MAP_SWITCH_FALLBACK_X = 0.30

# Detections taken before the seat grid exists are kept with their pose and
# replayed once it does. Bounded so a long pre-latch phase cannot grow without
# limit (15 Hz -> 60 s of scans).
PENDING_MAX = 900

# A seat is masked out of the wall extraction far earlier than it is reported
# as an obstacle: one vote is already reason enough to keep those directions
# out of a wall fit, while reporting still needs the full threshold.
MASK_MIN_VOTES = max(1, MIN_SEAT_VOTES // 4)

OUTER_HALF = 1.5               # outer wall position in the field frame

COLOR_CODE = {'red': Obstacle.COLOR_RED, 'green': Obstacle.COLOR_GREEN}


def yaw_from_quaternion(q):
    siny = 2.0 * (q.w * q.z + q.x * q.y)
    cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return np.arctan2(siny, cosy)


class ScanProcessor(Node):
    def __init__(self):
        super().__init__('scan_processor')
        self.race_mode = self.declare_parameter(
            'race_mode', 'obstacle').get_parameter_value().string_value
        self.parking_lot_present = self.declare_parameter(
            'parking_lot_present', False).get_parameter_value().bool_value

        self.pose = (0.0, 0.0, 0.0)
        self.map_walls = None
        self.front_wall_x = None
        self.votes = []
        self.position = None
        self.lane_width = None

        self.left_d = None
        self.right_d = None
        self.front_d_meas = None

        self.direction = None
        self.dir_votes = []

        # --- lane-width learning (open mode) ---
        self.lap_state = None        # [corner_idx, corner_count, lap]
        self.width_samples = {}
        self.width_fixed = {}
        self.inner_walls = None

        # --- obstacles (obstacle mode) ---
        self.obstacle_map = None     # built at the direction latch
        self.seat_wall_idx = None
        self.start_wall_idx = None
        self.obstacle_state = None
        self.pending_dets = deque(maxlen=PENDING_MAX)   # (detections, pose)
        # angular sectors of the pillars seen in the most recent colour scan.
        # The wall extraction masks these out: a pillar merged into a wall
        # corrupts its fit, and near a corner that delays the direction latch
        # by seconds -- which in turn delays the seat grid and the obstacle map.
        self.obstacle_sectors = []

        latched = QoSProfile(depth=1)
        latched.durability = DurabilityPolicy.TRANSIENT_LOCAL

        self.create_subscription(LaserScan, '/scan', self.scan_cb, 10)
        self.create_subscription(Odometry, '/ekf/odom', self.pose_cb, 10)
        self.create_subscription(Int32MultiArray,
                                 '/round1_controller/lap_state',
                                 self.lap_state_cb, latched)
        self.create_subscription(PointCloud2, '/camera_lidar/colored_scan',
                                 self.colored_scan_cb, 10)

        self.pub = self.create_publisher(WallMatchArray, '/wall_matches', 10)
        self.wall_dist_pub = self.create_publisher(
            Float64MultiArray, '/wall_distances', 10)
        self.obstacle_live_pub = self.create_publisher(
            ObstacleArray, '/obstacles_live', 10)
        self.front_wall_pub = self.create_publisher(Float64, '/front_wall_x', latched)
        self.direction_pub = self.create_publisher(String, '/race_direction', latched)
        self.corner_pub = self.create_publisher(CornerGeometry, '/corner_geometry', latched)
        self.inner_pub = self.create_publisher(CornerGeometry, '/inner_geometry', latched)
        self.obstacle_pub = self.create_publisher(ObstacleArray, '/obstacles', latched)

        self.get_logger().info(
            f'start detection running (mode={self.race_mode}, '
            f'parking_lot={self.parking_lot_present})...')

    # ------------------------------------------------------------------ #
    # callbacks
    # ------------------------------------------------------------------ #

    def pose_cb(self, msg):
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        theta = yaw_from_quaternion(msg.pose.pose.orientation)
        self.pose = (x, y, theta)

    def lap_state_cb(self, msg):
        prev = self.lap_state
        self.lap_state = list(msg.data)
        if self.lap_state[1] == 0 and self.start_wall_idx is None:
            self.start_wall_idx = self._current_outer_wall_index()
        if prev is not None and self.lap_state[2] > prev[2]:
            self._maybe_commit_inner_band(verbose=True)

    def scan_cb(self, msg):
        measured = self._extract(msg)
        self._publish_wall_distances(measured)

        if self.map_walls is None:
            res = self._detect(measured)
            if res['valid']:
                self.votes.append((res['position'], res['front_dist'],
                                   res.get('left_d'), res.get('right_d')))
            if len(self.votes) >= START_VOTES:
                self._commit()
            return

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

    def colored_scan_cb(self, msg):
        """Obstacle detection, obstacle mode only.

        Raw detections go out every scan on /obstacles_live -- no map needed.
        For the seat grid: if it does not exist yet (before the direction
        latch), the detections are buffered WITH the pose they were taken at
        and replayed when the grid is built. The pose is start-anchored and
        valid from the first scan, so a replayed detection snaps exactly as it
        would have live.
        """
        if self.race_mode != 'obstacle':
            return
        dets = detect_obstacles(msg)

        # hand the pillar directions to the wall extraction. Set every scan,
        # including the empty case, so the mask clears once a pillar is passed.
        # Colour scan and /scan come from the same LiDAR at the same rate, so
        # the sectors are at most one scan interval old -- well inside the
        # margin they are widened by.
        self.obstacle_sectors = [d['sector'] for d in dets]

        self._publish_obstacles_live(dets, msg.header.stamp)
        if not dets:
            return

        if self.obstacle_map is None:
            self.pending_dets.append((dets, self.pose))
            return

        self.obstacle_map.add_detections(dets, self.pose,
                                         allowed=self._seat_allowed)
        self._publish_obstacles_if_changed()

    # ------------------------------------------------------------------ #
    # extraction / start detection
    # ------------------------------------------------------------------ #

    def _map_obstacle_sectors(self):
        """Sectors for pillars whose position is already known, recomputed from
        the current pose.

        This is the layer that carries a close pass. /obstacles_live cannot:
        it arrives at ~6 Hz, and at 0.1-0.2 m the bearing sweeps ~25 deg
        between messages while the pillar is ~20 deg wide, so the previous
        sector no longer overlaps. The fusion also drops everything below its
        0.15 m range floor, so the pillar vanishes from the live topic exactly
        in the window where it breaks the wall fit. The map position does not
        vanish, and the pose is available at scan rate.
        """
        if self.obstacle_map is None:
            return []
        px, py, th = self.pose
        c, s = np.cos(th), np.sin(th)
        out = []
        for p in self.obstacle_map.seats_for_mask(MASK_MIN_VOTES):
            dx, dy = p[0] - px, p[1] - py
            xr = c * dx + s * dy          # map -> robot frame
            yr = -s * dx + c * dy
            sec = sector_from_robot_point(xr, yr)
            if sec is not None:
                out.append(sec)
        return out

    def _extract(self, msg):
        pts = scan_to_points(msg)
        # drop the directions occupied by pillars, so they cannot end up inside
        # a wall cluster. A missing slice of wall is harmless (gap clustering
        # splits it, both parts still match the same map wall); a pillar inside
        # a wall is not. Two sources: known positions from the map (fast, works
        # at any range) and the live topic (for pillars not yet mapped).
        pts = mask_sectors(pts, self.obstacle_sectors + self._map_obstacle_sectors())
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
        Left is alpha ~ -90 (+y), right is alpha ~ +90 (-y)."""
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
        positions = [v[0] for v in self.votes]
        winner, _ = Counter(positions).most_common(1)[0]
        win = [v for v in self.votes if v[0] == winner]

        if self.race_mode == 'open':
            front_d = float(np.mean([v[1] for v in win]))
            left_d = float(np.mean([v[2] for v in win]))
            right_d = float(np.mean([v[3] for v in win]))
            self._commit_open(winner, front_d, left_d, right_d)
        else:
            self._commit_obstacle(winner)

    def _commit_obstacle(self, position):
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
        self.left_d = left_d
        self.right_d = right_d
        self.front_d_meas = front_d
        self.map_walls = start_map_3wall(front_d, left_d, right_d)
        self.front_wall_x = front_d
        self.get_logger().info(
            f'[open] start position {position} -> 3-wall map committed '
            f'(front={front_d:.2f}, left={left_d:.2f}, right={right_d:.2f})')
        self._publish_front_wall_x()

    # ------------------------------------------------------------------ #
    # direction latch + map switch
    # ------------------------------------------------------------------ #

    def _update_direction(self, measured):
        if self.direction is not None:
            return                                # already latched -> frozen

        res = detect_direction(measured, lane_width=self.lane_width)
        if not res['confident']:
            return
        self.dir_votes.append(res['direction'])
        if len(self.dir_votes) > DIRECTION_VOTES:
            self.dir_votes.pop(0)
        if len(self.dir_votes) == DIRECTION_VOTES and len(set(self.dir_votes)) == 1:
            self.direction = self.dir_votes[0]
            self.direction_pub.publish(String(data=self.direction))
            self.get_logger().info(f'race direction latched: {self.direction}')

            self._switch_map_to_direction()

            start_pose = self._start_pose_for_direction()
            self._publish_corner_geometry(start_pose)
            if self.race_mode == 'obstacle':
                inner, corners = inner_walls_map(start_pose)
                self._publish_inner_geometry(inner, corners)
                self._init_obstacle_map(start_pose)

    def _start_pose_for_direction(self):
        if self.race_mode == 'open':
            return self._open_start_pose()
        poses = START_POSES_CW if self.direction == 'CW' else START_POSES_CCW
        return poses[f'pos{self.position}']

    def _map_switch_limit(self):
        """How far along the straight the map may still be switched: up to just
        short of the corner, since both maps hold the same three walls until
        the inner band ends."""
        if self.front_wall_x is None:
            return MAP_SWITCH_FALLBACK_X
        return max(MAP_SWITCH_FALLBACK_X,
                   self.front_wall_x - MAP_SWITCH_CORNER_MARGIN)

    def _switch_map_to_direction(self):
        limit = self._map_switch_limit()
        if abs(self.pose[0]) > limit:
            self.get_logger().warn(
                f'direction latched at x={self.pose[0]:.2f} m, past the '
                f'{limit:.2f} m limit (corner) -- NOT switching map to avoid '
                f'a pose jump')
            return

        start_pose = self._start_pose_for_direction()
        if self.race_mode == 'open':
            self.get_logger().info(
                f'open start pose (from measurements): '
                f'({start_pose[0]:+.3f}, {start_pose[1]:+.3f}, '
                f'{np.degrees(start_pose[2]):+.1f} deg)')
            self.map_walls = outer_walls_map(start_pose)
        else:
            self.map_walls = generate_map(start_pose)

        self.front_wall_x = self._front_wall_x_from_map(self.map_walls)
        self.get_logger().info(
            f'matching map switched to {self.direction} '
            f'({len(self.map_walls)} walls, at x={self.pose[0]:.2f} m)')

    def _open_start_pose(self):
        """Exact field start pose for the open challenge, from the measured
        distances. The OUTER wall is the only fixed reference (always +-1.5);
        it is on the left for CW and on the right for CCW.

            CW  (faces +x): x = 1.5 - front_d,  y = 1.5 - left_d,   theta = 0
            CCW (faces -x): x = front_d - 1.5,  y = 1.5 - right_d,  theta = pi
        """
        f = self.front_d_meas
        if self.direction == 'CW':
            return (OUTER_HALF - f, OUTER_HALF - self.left_d, 0.0)
        return (f - OUTER_HALF, OUTER_HALF - self.right_d, np.pi)

    # ------------------------------------------------------------------ #
    # obstacles
    # ------------------------------------------------------------------ #

    def _publish_obstacles_live(self, dets, stamp):
        """Raw detections, base_link frame, every scan. No map needed, so this
        works from the first scan on. id and wall_idx are -1: without a map
        there is no seat to assign."""
        msg = ObstacleArray()
        msg.header.stamp = stamp
        msg.header.frame_id = 'base_link'
        for d in dets:
            o = Obstacle()
            o.id = -1
            o.position = Point(x=float(d['x']), y=float(d['y']), z=0.0)
            o.color = COLOR_CODE.get(d['color'], Obstacle.COLOR_UNKNOWN)
            o.wall_idx = -1
            msg.obstacles.append(o)
        self.obstacle_live_pub.publish(msg)

    def _init_obstacle_map(self, start_pose):
        """Build the seat grid, then replay everything seen before it existed."""
        seats = obstacle_seats_map(start_pose)
        self.seat_wall_idx = seat_group_to_wall_index(start_pose)
        self.obstacle_map = ObstacleMap(seats)
        self.get_logger().info(
            f'obstacle seat grid ready (24 seats, groups -> walls '
            f'{self.seat_wall_idx})')

        if self.pending_dets:
            n = sum(len(d) for d, _ in self.pending_dets)
            for dets, pose in self.pending_dets:
                self.obstacle_map.add_detections(dets, pose,
                                                 allowed=self._seat_allowed)
            self.get_logger().info(
                f'replayed {n} buffered detections from '
                f'{len(self.pending_dets)} scans taken before the latch')
            self.pending_dets.clear()
            self._publish_obstacles_if_changed()

    def _seat_allowed(self, seat_group, column):
        """Parking-lot rule: on the start straight only the inner column is
        legal, because the signs are moved inward when a lot is placed."""
        if not self.parking_lot_present:
            return True
        if self.start_wall_idx is None or self.seat_wall_idx is None:
            return True
        if self.seat_wall_idx[seat_group] != self.start_wall_idx:
            return True
        return column == 'inner'

    @staticmethod
    def _seat_id(seat):
        return seat['straight'] * 6 + seat['row'] * 2 + \
            (0 if seat['column'] == 'outer' else 1)

    def _publish_obstacles_if_changed(self):
        occupied = self.obstacle_map.occupied_seats()
        state = tuple(sorted((self._seat_id(s), s['color']) for s in occupied))
        if state == self.obstacle_state:
            return
        self.obstacle_state = state

        msg = ObstacleArray()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'map'
        for s in occupied:
            o = Obstacle()
            o.id = self._seat_id(s)
            o.position = Point(x=float(s['p'][0]), y=float(s['p'][1]), z=0.0)
            o.color = COLOR_CODE.get(s['color'], Obstacle.COLOR_UNKNOWN)
            o.wall_idx = int(self.seat_wall_idx[s['straight']])
            msg.obstacles.append(o)
        self.obstacle_pub.publish(msg)

        txt = ', '.join(
            f"#{self._seat_id(s)}({s['color'][0]},w{self.seat_wall_idx[s['straight']]})"
            for s in sorted(occupied, key=self._seat_id))
        self.get_logger().info(f'obstacles: {len(occupied)} [{txt}]')
        self.get_logger().info('votes: ' + self.obstacle_map.vote_summary())

    # ------------------------------------------------------------------ #
    # lane-width learning (open mode)
    # ------------------------------------------------------------------ #

    def _current_outer_wall_index(self):
        """Outer wall the robot is driving along. corner_idx names the corner
        AHEAD; CCW came from corner k-1 (wall k-1), CW from k+1 (wall k)."""
        if self.lap_state is None or self.direction is None:
            return None
        k = self.lap_state[0]
        return (k - 1) % 4 if self.direction == 'CCW' else k % 4

    def _learn_lane_width(self, measured):
        """The START straight's width comes from the stationary start detection.
        The others are sampled while driving. Learning continues past round 1
        until the inner band is committed."""
        if self.race_mode != 'open' or self.inner_walls is not None:
            return
        wall_idx = self._current_outer_wall_index()
        if wall_idx is None:
            return

        if self.lap_state[1] == 0 and wall_idx not in self.width_fixed:
            self.width_fixed[wall_idx] = float(self.lane_width)
            self.get_logger().info(
                f'start straight {wall_idx}: lane width '
                f'{self.lane_width:.3f} taken from start detection')
            self._maybe_commit_inner_band()
            return
        if wall_idx in self.width_fixed:
            return

        left, right = self._side_distances(measured)
        if left is None or right is None:
            return
        width = left + right
        if min(abs(width - n) for n in LANE_NOMINALS) > LANE_PLAUS_TOL:
            return

        self.width_samples.setdefault(wall_idx, []).append(width)
        self._maybe_commit_inner_band()

    def _maybe_commit_inner_band(self, verbose=False):
        """Commit + publish the inner band once every straight's width is known.
        Publishes nothing while one is missing: better no inner geometry than a
        wrong one."""
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
            widths[i] = float(np.median(s))

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
            msg.corners[i] = Point(x=float(corners[i][0]),
                                   y=float(corners[i][1]), z=0.0)
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
            msg.corners[i] = Point(x=float(corners[i][0]),
                                   y=float(corners[i][1]), z=0.0)
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
        """Map-frame x of the front wall (alpha ~ +-180), as |d|."""
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