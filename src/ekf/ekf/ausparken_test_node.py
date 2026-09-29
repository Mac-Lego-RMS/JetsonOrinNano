#!/usr/bin/env python3
"""
Drive the unpark sequence out and back again, to MEASURE its error.

    ros2 run ekf unpark_test

First the sequence is driven forwards, then the same sequence backwards:
steps in reverse order, each with the same steering and
negative travel. Kinematically that cancels the outbound path exactly -- the robot
should end up exactly where it started.

What is left over is the RETURN ERROR, and that is the measurement result.
It needs no bay dimensions and no map, only the odometry, and it
separates two things that otherwise get mixed up:

  * An error that CANCELS on the way back (outbound and return deviate
    the same way) is in the model -- turning circle, wheelbase, trim.
  * An error that STAYS is in the mechanics -- slip, play in the
    steering, overrun of the position controller.

The comparison between the driven and the computed end pose of the outbound
path also shows how well the curve from steer_calib.json fits right now.

CAUTION: this node DRIVES. It counts down before the start, and during
a position move /cmd_vel has no effect -- the emergency stop goes through
/esp_serial_bridge/emergency.
"""
import math
import sys
import threading

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from std_msgs.msg import Float32, Float32MultiArray, Int32, Int32MultiArray

from sensor_msgs.msg import LaserScan

from ekf.unpark import (trajectory, cm_to_deg, direction_from_scan,
                        steps_from_flat, steps_for, mirror_steps,
                        STEER_SOURCE, STEPS_DEFAULT, turn_radius_of)
from ekf.wall_extraction import scan_to_points
from ekf.move_sequencer import MoveSequencer, pose_text, reverse_steps, wrap


def yaw_from_quaternion(q):
    siny = 2.0 * (q.w * q.z + q.x * q.y)
    cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny, cosy)


def in_start_frame(start, pose):
    """Deviation from ``start`` in its own frame: (long, lat, yaw).

    Long and lat say more than x/y in the odom frame -- "3 cm too
    far" and "3 cm off to the side" are different errors.
    """
    xs, ys, ths = start
    dx, dy = pose[0] - xs, pose[1] - ys
    c, s = math.cos(ths), math.sin(ths)
    return (c * dx + s * dy, -s * dx + c * dy, wrap(pose[2] - ths))


class UnparkTest(Node):

    def __init__(self):
        super().__init__('unpark_test')
        arr = self._array_type()
        self.declare_parameter('steps', list(STEPS_DEFAULT), arr)
        # Empty = MEASURE from the scan, the way the controller does it. The robot
        # stands in the same bay during the test; a guessed direction
        # mirrors the sequence the wrong way round and then measures something other
        # than what is driven later. CW or CCW forces a direction, for
        # trials outside the bay.
        self.declare_parameter('direction', '')
        self.declare_parameter('scans', 5)
        self.declare_parameter('sector_deg', 20.0)
        self.declare_parameter('direction_timeout', 8.0)
        self.declare_parameter('repetitions', 1)
        # At the turnaround wait for Enter instead of the clock: that is where
        # you want to measure, and a fixed time is always either too short
        # or too long for that. Without a terminal (stdin not a TTY) it falls back to
        # pause_s, otherwise the node hangs there forever.
        self.declare_parameter('pause_on_key', True)
        self.declare_parameter('pause_s', 2.0)
        self.declare_parameter('countdown_s', 3.0)
        self.declare_parameter('steer_wait_s', 0.6)
        self.declare_parameter('move_timeout', 15.0)
        self.declare_parameter('travel_tol_cm', 1.0)
        self.declare_parameter('pid', [4.0, 140.0, 8.0, 90.0], arr)
        self.declare_parameter('pid_after', [4.0, 1023.0], arr)

        self.steer_wait_s = float(self.get_parameter('steer_wait_s').value)
        self.move_timeout = float(self.get_parameter('move_timeout').value)
        self.travel_tol_cm = float(self.get_parameter('travel_tol_cm').value)
        self.pause_s = float(self.get_parameter('pause_s').value)
        self.pause_on_key = bool(self.get_parameter('pause_on_key').value)
        self.proceed = False
        self.key_wait_active = False
        self.countdown_s = float(self.get_parameter('countdown_s').value)
        self.rounds = max(1, int(self.get_parameter('repetitions').value))

        self.preset_direction = str(self.get_parameter('direction').value).strip().upper()
        if self.preset_direction not in ('', 'CW', 'CCW'):
            raise ValueError('direction must be empty, CW or CCW, not "%s"'
                             % self.preset_direction)
        self.scans = max(1, int(self.get_parameter('scans').value))
        self.sector_deg = float(self.get_parameter('sector_deg').value)
        self.direction_timeout = float(
            self.get_parameter('direction_timeout').value)
        self.votes = []
        self.last_reason = None
        self.direction = None
        self.steps_out = None
        self.steps_back = None

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
        self.sequencer = None
        self.round_no = 0
        self.start_pose = None
        self.turnaround_pose = None
        self.log = []

        self.get_logger().info(
            ">>> Unpark test: out and the same path back, %dx. "
            "Driving direction: %s <<<"
            % (self.rounds,
               self.preset_direction + ' (preset)' if self.preset_direction
               else 'is measured from the scan'))
        self.get_logger().info(
            "Steering: %s, full lock R = %.3f m."
            % (STEER_SOURCE or 'FALLBACK (steer_calib.json not found!)',
               turn_radius_of(100.0)))
        self.create_timer(1.0 / 30.0, self.control_loop)

    @staticmethod
    def _array_type():
        from rcl_interfaces.msg import ParameterDescriptor, ParameterType
        return ParameterDescriptor(type=ParameterType.PARAMETER_DOUBLE_ARRAY)

    def now_s(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def odom_cb(self, msg):
        p = msg.pose.pose
        self.pose = (p.position.x, p.position.y,
                     yaw_from_quaternion(p.orientation))

    def scan_cb(self, msg):
        """One vote for the driving direction, while searching."""
        if self.state != 'DIRECTION':
            return
        e = direction_from_scan(scan_to_points(msg),
                                half_angle_deg=self.sector_deg)
        self.last_reason = e['reason']
        if not e['confident']:
            self.votes = []
            return
        # Only AGREEING votes count: a contradiction resets. Whoever touches the
        # robot during the search gets no decision instead of
        # a narrow one -- the same rule as in the controller.
        if self.votes and self.votes[-1] != e['direction']:
            self.votes = []
        self.votes.append(e['direction'])

    def _choose_sequence(self, direction, source):
        """Pick the step sequence for this direction and mirror it."""
        self.direction = direction
        raw, origin = steps_for(
            direction, shared=list(self.get_parameter('steps').value))
        self.steps_out = mirror_steps(steps_from_flat(raw), direction == 'CCW')
        self.steps_back = reverse_steps(self.steps_out)
        end = trajectory((0.0, 0.0, 0.0), self.steps_out)[-1][0]
        self.get_logger().info(
            "Driving direction %s (%s). %s: %d moves, %.0f cm per direction."
            % (direction, source, origin, len(self.steps_out),
               sum(abs(cm) for _l, cm in self.steps_out)))
        self.get_logger().info(
            "Model predicts for the outbound path: %.1f cm ahead, %.1f cm to the side, "
            "%+.1f deg." % (end[0] * 100, end[1] * 100,
                            math.degrees(end[2])))

    def move_done_cb(self, msg):
        if len(msg.data) >= 3 and self.sequencer is not None:
            self.sequencer.ack = (self.now_s(), int(msg.data[1]),
                                  msg.data[2] / 10.0)

    def _pid(self, vals):
        vals = list(vals)
        for i in range(0, len(vals) - 1, 2):
            self.pub_pid.publish(
                Float32MultiArray(data=[float(vals[i]), float(vals[i + 1])]))

    def _abort(self, reason):
        self.pub_motor.publish(Int32(data=0))
        self._pid(self.get_parameter('pid_after').value)
        self.get_logger().error("Aborted: %s" % reason)
        if self.pose is not None:
            self.get_logger().error("           stands at  %s"
                                    % pose_text(self.pose))
        self.state = 'END'

    def control_loop(self):
        if self.pose is None:
            self.get_logger().warn("waiting for /ekf/odom -- is ekf_node running?",
                                   throttle_duration_sec=2.0)
            return
        t_now = self.now_s()

        if self.state == 'WAITING':
            missing = [n for n, pub in (('steer', self.pub_steer),
                                        ('move', self.pub_move),
                                        ('pid_set', self.pub_pid),
                                        ('motor', self.pub_motor))
                       if pub.get_subscription_count() == 0]
            if missing:
                self.get_logger().info("waiting for the bridge (%s)"
                                       % ', '.join(missing),
                                       throttle_duration_sec=1.0)
                if t_now - self.t0 > 15.0:
                    self._abort("bridge is not listening (%s)"
                                % ', '.join(missing))
                return
            self.state = 'DIRECTION'
            self.t0 = t_now
            if self.preset_direction:
                self._choose_sequence(self.preset_direction, 'preset')
                self.state = 'COUNTDOWN'
            else:
                self.get_logger().info(
                    "looking for the open side, %d agreeing scans needed."
                    % self.scans)
            return

        if self.state == 'DIRECTION':
            if len(self.votes) < self.scans:
                if t_now - self.t0 > self.direction_timeout:
                    self._abort(
                        "no clear driving direction in %.0f s -- last: "
                        "%s. Is it standing in the bay? Otherwise preset direction:=CW or "
                        "CCW."
                        % (self.direction_timeout,
                           self.last_reason or 'no /scan received'))
                else:
                    self.get_logger().info(
                        "%d/%d votes -- %s"
                        % (len(self.votes), self.scans,
                           self.last_reason or 'waiting for /scan'),
                        throttle_duration_sec=1.0)
                return
            self._choose_sequence(self.votes[-1], 'measured from the scan')
            self.state = 'COUNTDOWN'
            self.t0 = t_now
            return

        if self.state == 'COUNTDOWN':
            remaining = self.countdown_s - (t_now - self.t0)
            if remaining > 0.0:
                self.get_logger().warn("Start in %.0f s -- it is about to DRIVE."
                                       % math.ceil(remaining),
                                       throttle_duration_sec=0.9)
                return
            self._pid(self.get_parameter('pid').value)
            self._new_round()
            return

        if self.state in ('OUTBOUND', 'RETURN'):
            if self.sequencer.tick(t_now, self.pose):
                if self.sequencer.error:
                    self._abort(self.sequencer.error)
                    return
                self._leg_done(t_now)
            return

        if self.state == 'PAUSE':
            if not self._pause_over(t_now):
                return
            self.get_logger().info("Way back: the same sequence, in reverse.")
            self.sequencer = MoveSequencer(self, self.steps_back, 'WAY-BACK')
            self.state = 'RETURN'
            return

        if self.state == 'END':
            self._report()
            raise SystemExit(0)

    def _wait_for_key(self):
        """Wait for Enter, in a thread of its own -- rclpy.spin blocks."""
        try:
            sys.stdin.readline()
        except Exception:
            pass
        self.proceed = True

    def _pause_over(self, t_now):
        if not self.pause_on_key or not sys.stdin.isatty():
            if not self.key_wait_active:
                self.key_wait_active = True
                if self.pause_on_key:
                    self.get_logger().warn(
                        "no terminal on stdin -- waiting %.1f s instead of for "
                        "Enter." % self.pause_s)
            return t_now - self.t0 >= self.pause_s
        if not self.key_wait_active:
            self.key_wait_active = True
            threading.Thread(target=self._wait_for_key,
                             daemon=True).start()
            self.get_logger().info(
                ">>> Press ENTER, then it drives the way back. <<<")
        return self.proceed

    def _new_round(self):
        self.round_no += 1
        self.start_pose = self.pose
        self.sequencer = MoveSequencer(self, self.steps_out, 'WAY-OUT')
        self.state = 'OUTBOUND'
        self.get_logger().info("--- Round %d/%d, start at  %s ---"
                               % (self.round_no, self.rounds,
                                  pose_text(self.pose)))

    def _leg_done(self, t_now):
        if self.state == 'OUTBOUND':
            self.turnaround_pose = self.pose
            long, lat, yaw = in_start_frame(self.start_pose, self.pose)
            model = trajectory((0.0, 0.0, 0.0), self.steps_out)[-1][0]
            self.get_logger().info(
                "WAY-OUT done: %.1f cm ahead, %.1f cm to the side, %+.1f deg "
                "| model: %.1f / %.1f / %+.1f"
                % (long * 100, lat * 100, math.degrees(yaw),
                   model[0] * 100, model[1] * 100, math.degrees(model[2])))
            self.get_logger().info("           stands at  %s"
                                   % pose_text(self.pose))
            self.state = 'PAUSE'
            self.proceed = False
            self.key_wait_active = False
            self.t0 = t_now
            return

        long, lat, yaw = in_start_frame(self.start_pose, self.pose)
        self.log.append((long, lat, yaw))
        self.get_logger().info(
            "RETURN ERROR round %d: %+.1f cm long, %+.1f cm lat, "
            "%+.1f deg  (distance %.1f cm)"
            % (self.round_no, long * 100, lat * 100, math.degrees(yaw),
               math.hypot(long, lat) * 100))
        self.get_logger().info("           stands at  %s  (start was %s)"
                               % (pose_text(self.pose),
                                  pose_text(self.start_pose)))
        if self.round_no < self.rounds:
            self._new_round()
        else:
            self._pid(self.get_parameter('pid_after').value)
            self.state = 'END'

    def _report(self):
        if not self.log:
            self.get_logger().warn("No complete round driven.")
            return
        self.get_logger().info("=== Result over %d round(s) ==="
                               % len(self.log))
        for i, (l, q, g) in enumerate(self.log, 1):
            self.get_logger().info(
                "  Round %d: %+6.1f cm long  %+6.1f cm lat  %+6.1f deg"
                % (i, l * 100, q * 100, math.degrees(g)))
        n = len(self.log)
        m_long = sum(p[0] for p in self.log) / n
        m_lat = sum(p[1] for p in self.log) / n
        m_yaw = sum(p[2] for p in self.log) / n
        self.get_logger().info(
            "  Mean:    %+6.1f cm long  %+6.1f cm lat  %+6.1f deg"
            % (m_long * 100, m_lat * 100, math.degrees(m_yaw)))
        self.get_logger().info(
            "An error that cancels here is in the model "
            "(turning circle, wheelbase); what is left over is mechanics "
            "(slip, steering play, overrun).")


def main(args=None):
    rclpy.init(args=args)
    node = UnparkTest()
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
