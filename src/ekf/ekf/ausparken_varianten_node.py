#!/usr/bin/env python3
"""
ONLY unpark -- one of the six variants, for calibrating and tuning.

    python3 -m ekf.unpark_variants_node --ros-args -p placement:=middle
    python3 -m ekf.unpark_variants_node --ros-args -p placement:=inner -p direction:=CW
    python3 -m ekf.unpark_variants_node --ros-args -p placement:=outer \\
        -p steps:="[0.0,0.0, 100.0,6.0, -100.0,-4.5, 100.0,9.6, 0.0,5.0, -100.0,24.0, 0.0,0.0]"

The variants are in unpark.py: STEPS_{CW,CCW}_{INNER,MIDDLE,OUTER}.
Sequence:
  1. Measure the driving direction from the scan, the way the controller does it (or
     preset it with direction:=CW/CCW). That fixes the variant.
  2. Dry run against the bay dimensions: collision? Final pose according to the model?
  3. Countdown, then drive the moves -- through the same move sequencer as the
     controller, i.e. exactly the way it happens later in the run.
  4. Stop at the end and measure with the lidar where it stands in the lane:
     distance to the outer wall and heading to the wall.
  5. Print the driven list for pasting into unpark.py.

With steps:=[...] it drives this list instead of the table -- for trying things
without touching unpark.py. If it fits, it goes into the table by hand.

You put it back into the bay by hand; there is no way back.

At start-up the EKF and scan_processor are restarted fresh (windows 8/9, via
the restart watchdog in window 11, see ekf/estimation_restart.py) -- as
with the controller. To switch off: -p estimation_restart:=false.

esp_serial_bridge, IMU and lidar must be running (i.e. start_robot.sh). The
controller must NOT be running: during a move /cmd_vel has no effect, and the
controller would interfere. The emergency stop goes through
/esp_serial_bridge/emergency.
"""
import math

import numpy as np
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float32, Float32MultiArray, Int32, Int32MultiArray

from ekf.unpark import (PLACEMENTS, STEER_SOURCE, direction_from_scan,
                        steps_from_flat, steps_for_variant, simulate,
                        mirror_steps, bay_start_pose, turn_radius_of)
from ekf.estimation_restart import restart_estimation
from ekf.wall_extraction import LIDAR_OFFSET_X, scan_to_points
from ekf.move_sequencer import MoveSequencer, pose_text, wrap

LANE_WIDTH = 1.00        # outer wall to inner wall on the start straight [m]


def yaw_from_quaternion(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                      1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def in_start_frame(start, pose):
    """(long, lat, yaw) of ``pose`` in the frame of ``start``."""
    dx, dy = pose[0] - start[0], pose[1] - start[1]
    c, s = math.cos(start[2]), math.sin(start[2])
    return c * dx + s * dy, -s * dx + c * dy, wrap(pose[2] - start[2])


def measure_wall(points, side_left, max_dist=0.9, half_angle_deg=55.0,
                 front_only=True):
    """Line through the wall points on one side (principal axis).

    ``points`` in the base_link frame. Returns (distance from base_link [m],
    angle of the wall against the car's long axis [rad], point count) or None.
    The angle is positive when the wall gets closer towards the front -- then
    the nose points towards the wall.
    """
    middle = math.pi / 2.0 if side_left else -math.pi / 2.0
    w = np.arctan2(points[:, 1], points[:, 0])
    r = np.hypot(points[:, 0], points[:, 1])
    d = np.abs(np.arctan2(np.sin(w - middle), np.cos(w - middle)))
    # Outer wall only in front of the rear axle (front_only): diagonally behind
    # it, after unparking, stands the front bay wall, perpendicular to the wall -- it would
    # twist the line. The inner wall on the other hand often ends just behind
    # the robot (CCW: island corner at x=0.25, it stands at 0.29) and can only be
    # seen diagonally behind.
    k = (d <= math.radians(half_angle_deg)) & (r <= max_dist)
    if front_only:
        k &= points[:, 0] >= 0.0
    p = points[k]
    if len(p) < 8:
        return None
    for _ in range(2):
        # twice: fit, drop outliers (pylons, bay remains) > 3 cm
        m = p.mean(axis=0)
        _u, _s, vt = np.linalg.svd(p - m)
        n = np.array([-vt[0][1], vt[0][0]])
        rest = np.abs((p - m) @ n)
        if (rest > 0.03).sum() == 0 or (rest <= 0.03).sum() < 8:
            break
        p = p[rest <= 0.03]
    m = p.mean(axis=0)
    _u, _s, vt = np.linalg.svd(p - m)
    direction = vt[0]
    if direction[0] < 0:
        direction = -direction
    normal = np.array([-direction[1], direction[0]])
    dist = abs(float(normal @ m))
    angle = math.atan2(direction[1], direction[0])     # wall against +x
    # Right (y<0): if the wall gets closer towards the front, y rises -> angle > 0.
    # Left mirrored.
    to_wall = angle if not side_left else -angle
    return dist, to_wall, len(p)


class UnparkVariants(Node):

    def __init__(self):
        super().__init__('unpark_variants')
        from rcl_interfaces.msg import ParameterDescriptor, ParameterType
        arr = ParameterDescriptor(type=ParameterType.PARAMETER_DOUBLE_ARRAY)
        self.declare_parameter('placement', 'middle')
        self.declare_parameter('direction', '')
        self.declare_parameter('steps', [0.0], arr)   # [0.0] = from the table
        self.declare_parameter('scans', 5)
        self.declare_parameter('sector_deg', 20.0)
        self.declare_parameter('direction_timeout', 8.0)
        self.declare_parameter('countdown_s', 3.0)
        self.declare_parameter('steer_wait_s', 0.6)
        self.declare_parameter('move_timeout', 15.0)
        self.declare_parameter('travel_tol_cm', 1.0)
        self.declare_parameter('measure_scans', 10)
        self.declare_parameter('pid', [4.0, 140.0, 8.0, 90.0], arr)
        self.declare_parameter('pid_after', [4.0, 1023.0], arr)

        self.placement = str(self.get_parameter('placement').value).strip().lower()
        if self.placement not in PLACEMENTS:
            raise ValueError('placement must be %s, not "%s"'
                             % (' / '.join(PLACEMENTS), self.placement))
        self.preset_direction = str(self.get_parameter('direction').value).strip().upper()
        if self.preset_direction not in ('', 'CW', 'CCW'):
            raise ValueError('direction must be empty, CW or CCW')
        custom = [float(v) for v in self.get_parameter('steps').value]
        self.custom = custom if len(custom) >= 2 else None

        self.scans = max(1, int(self.get_parameter('scans').value))
        self.sector_deg = float(self.get_parameter('sector_deg').value)
        self.direction_timeout = float(self.get_parameter('direction_timeout').value)
        self.countdown_s = float(self.get_parameter('countdown_s').value)
        self.steer_wait_s = float(self.get_parameter('steer_wait_s').value)
        self.move_timeout = float(self.get_parameter('move_timeout').value)
        self.travel_tol_cm = float(self.get_parameter('travel_tol_cm').value)
        self.measure_scans = max(3, int(self.get_parameter('measure_scans').value))

        self.pub_steer = self.create_publisher(Float32, '/esp_serial_bridge/steer', 10)
        self.pub_move = self.create_publisher(Float32, '/esp_serial_bridge/move', 10)
        self.pub_pid = self.create_publisher(
            Float32MultiArray, '/esp_serial_bridge/pid_set', 10)
        self.pub_motor = self.create_publisher(Int32, '/esp_serial_bridge/motor', 10)
        self.create_subscription(Int32MultiArray, '/esp_serial_bridge/move_done',
                                 self.move_done_cb, 10)
        self.create_subscription(Odometry, '/ekf/odom', self.odom_cb, 10)
        self.create_subscription(LaserScan, '/scan', self.scan_cb, 10)

        self.pose = None
        self.state = 'WAITING'
        self.t0 = self.now_s()
        self.votes = []
        self.last_reason = None
        self.direction = None
        self.raw = None
        self.name = None
        self.sequence = None
        self.sequencer = None
        self.start_pose = None
        self.measurements = []
        self.model_end = None

        self.get_logger().info(
            '>>> Unparking, variant "%s". Driving direction: %s <<<'
            % (self.placement, self.preset_direction + ' (preset)' if self.preset_direction
               else 'is measured from the scan'))
        self.get_logger().info(
            'Steering: %s, full lock R = %.3f m.'
            % (STEER_SOURCE or 'FALLBACK (steer_calib.json not found!)',
               turn_radius_of(100.0)))
        self.create_timer(1.0 / 30.0, self.control_loop)

    # --------------------------------------------------------------- Inputs
    def now_s(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def odom_cb(self, msg):
        p = msg.pose.pose
        self.pose = (p.position.x, p.position.y, yaw_from_quaternion(p.orientation))

    def move_done_cb(self, msg):
        if len(msg.data) >= 3 and self.sequencer is not None:
            self.sequencer.ack = (self.now_s(), int(msg.data[1]), msg.data[2] / 10.0)

    def scan_cb(self, msg):
        if self.state == 'DIRECTION':
            e = direction_from_scan(scan_to_points(msg), half_angle_deg=self.sector_deg)
            self.last_reason = e['reason']
            if not e['confident']:
                self.votes = []
                return
            if self.votes and self.votes[-1] != e['direction']:
                self.votes = []
            self.votes.append(e['direction'])
        elif self.state == 'MEASURING':
            p = scan_to_points(msg)
            p = np.column_stack((p[:, 0] + LIDAR_OFFSET_X, p[:, 1]))   # -> base_link
            self.measurements.append(p)

    def _pid(self, vals):
        vals = list(vals)
        for i in range(0, len(vals) - 1, 2):
            self.pub_pid.publish(Float32MultiArray(data=[float(vals[i]), float(vals[i + 1])]))

    def _abort(self, reason):
        self.pub_motor.publish(Int32(data=0))
        self._pid(self.get_parameter('pid_after').value)
        self.get_logger().error('Aborted: %s' % reason)
        if self.pose is not None:
            self.get_logger().error('           stands at  %s' % pose_text(self.pose))
        raise SystemExit(1)

    # ----------------------------------------------------------- Preparation
    def _choose_sequence(self, direction, source):
        self.direction = direction
        if self.custom:
            self.raw, self.name = list(self.custom), 'steps:= (command line)'
        else:
            self.raw, self.name = steps_for_variant(direction, self.placement)
        raw_pairs = steps_from_flat(self.raw)
        self.sequence = mirror_steps(raw_pairs, direction == 'CCW')
        log = self.get_logger()
        log.info('Driving direction %s (%s) -> %s, %d moves, %.0f cm travel.'
                 % (direction, source, self.name, len(self.sequence),
                    sum(abs(cm) for _l, cm in self.sequence)))

        # Dry run in the bay frame: outer wall y=0, open side +y.
        e = simulate(mirror_steps(raw_pairs, True))
        x0 = bay_start_pose()[0]
        end = e['end_pose']
        self.model_end = end
        log.info('Dry run: %s, closest distance %.0f mm to magenta, %.0f mm '
                 'to the outer wall, %s.'
                 % ('COLLISION in move %d' % e['at_step'] if e['collision']
                    else 'no collision',
                    e['magenta_dist_m'] * 1000, e['wall_dist_m'] * 1000,
                    'clear at the end' if e['clear'] else 'STILL IN THE BAY at the end'))
        log.info('Model final pose: base_link %.1f cm from the outer wall (lane centre '
                 '= %.0f cm), %.1f cm ahead, heading %+.1f deg.'
                 % (end[1] * 100, LANE_WIDTH * 50, (end[0] - x0) * 100,
                    math.degrees(end[2])))
        if e['collision']:
            log.warn('The sequence does not work out on paper -- it is driven '
                     'anyway, the model can differ from the bay. '
                     'Hand on the emergency stop.')

    # ------------------------------------------------------------------ Tick
    def control_loop(self):
        if self.pose is None:
            self.get_logger().warn('waiting for /ekf/odom -- is ekf_node running?',
                                   throttle_duration_sec=2.0)
            return
        t_now = self.now_s()

        if self.state == 'WAITING':
            missing = [n for n, pub in (('steer', self.pub_steer), ('move', self.pub_move),
                                        ('pid_set', self.pub_pid), ('motor', self.pub_motor))
                       if pub.get_subscription_count() == 0]
            if missing:
                self.get_logger().info('waiting for the bridge (%s)' % ', '.join(missing),
                                       throttle_duration_sec=1.0)
                if t_now - self.t0 > 15.0:
                    self._abort('bridge is not listening (%s)' % ', '.join(missing))
                return
            self.t0 = t_now
            if self.preset_direction:
                self._choose_sequence(self.preset_direction, 'preset')
                self.state = 'COUNTDOWN'
            else:
                self.state = 'DIRECTION'
                self.get_logger().info('looking for the open side, %d agreeing scans needed.'
                                       % self.scans)
            return

        if self.state == 'DIRECTION':
            if len(self.votes) < self.scans:
                if t_now - self.t0 > self.direction_timeout:
                    self._abort('no clear driving direction in %.0f s -- last: %s. '
                                'Is it standing in the bay? Otherwise direction:=CW or CCW.'
                                % (self.direction_timeout,
                                   self.last_reason or 'no /scan received'))
                return
            self._choose_sequence(self.votes[-1], 'measured from the scan')
            self.state = 'COUNTDOWN'
            self.t0 = t_now
            return

        if self.state == 'COUNTDOWN':
            rest = self.countdown_s - (t_now - self.t0)
            if rest > 0.0:
                self.get_logger().warn('Start in %.0f s -- it is about to DRIVE.' % math.ceil(rest),
                                       throttle_duration_sec=0.9)
                return
            self._pid(self.get_parameter('pid').value)
            self.start_pose = self.pose
            self.get_logger().info('Start at  %s' % pose_text(self.pose))
            self.sequencer = MoveSequencer(self, self.sequence, 'UNPARK')
            self.state = 'DRIVING'
            return

        if self.state == 'DRIVING':
            if self.sequencer.tick(t_now, self.pose):
                if self.sequencer.error:
                    self._abort(self.sequencer.error)
                self._pid(self.get_parameter('pid_after').value)
                self.measurements = []
                self.state = 'MEASURING'
                self.t0 = t_now
            return

        if self.state == 'MEASURING':
            # first let it settle briefly, then collect a few scans
            if t_now - self.t0 < 0.5:
                self.measurements = []
                return
            if len(self.measurements) < self.measure_scans and t_now - self.t0 < 4.0:
                return
            self._report()
            raise SystemExit(0)

    # ---------------------------------------------------------------- Report
    def _report(self):
        log = self.get_logger()
        long, lat, yaw = in_start_frame(self.start_pose, self.pose)
        end = self.model_end
        log.info('=== %s, driving direction %s ===' % (self.name, self.direction))
        log.info('Odometry from the start: %.1f cm ahead, %.1f cm to the open side, '
                 'heading %+.1f deg to the start pose'
                 % (long * 100, (lat if self.direction == 'CCW' else -lat) * 100,
                    math.degrees(yaw if self.direction == 'CCW' else -yaw)))
        log.info('           stands at  %s' % pose_text(self.pose))

        open_left = self.direction == 'CCW'
        if self.measurements:
            p = np.vstack(self.measurements)
            outer = measure_wall(p, side_left=not open_left)
            # The inner wall cannot be measured reliably here: after unparking the robot
            # stands right next to the island corner (run 16:
            # "lane 1.28 m"). In the obstacle race the lane is always
            # LANE_WIDTH wide though -- the position follows from the outer wall alone.
            if outer:
                a, w, n = outer
                log.info('Lidar: outer wall %.1f cm from base_link (model %.1f), '
                         'heading %+.1f deg to the wall (%s)  [%d points]'
                         % (a * 100, end[1] * 100, math.degrees(w),
                            'nose towards the wall' if w > 0 else 'nose towards the lane centre', n))
                log.info('Position in the lane: %.0f %% from the outside (0 = outer wall, '
                         '50 = centre, 100 = inner wall), at %.2f m lane width.'
                         % (100 * a / LANE_WIDTH, LANE_WIDTH))
            else:
                log.warn('Lidar: outer wall not found (too close or hidden).')
        else:
            log.warn('No scans received for the final measurement.')

        # for pasting in
        pairs = ',\n'.join('    %6.1f, %5.1f' % (self.raw[i], self.raw[i + 1])
                           for i in range(0, len(self.raw), 2))
        table = ('STEPS_%s_%s' % (self.direction, self.placement.upper())
                 if not self.name.startswith('STEPS') else self.name)
        log.info('Driven list for unpark.py:\n%s = [\n%s,\n]' % (table, pairs))


def main(args=None):
    rclpy.init(args=args)
    # Before our own node: otherwise the latched topics would still come from the
    # old scan_processor.
    restart_estimation('unpark_variants')
    node = UnparkVariants()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        try:
            node.pub_motor.publish(Int32(data=0))
        except Exception:
            pass
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
