#!/usr/bin/env python3
"""
Drive a manoeuvre sequence move by move through the bridge.

Shared by unpark_test_node and round1_controller_node: both
drive moves the SAME way. The measurement series from the unpark test apply to
exactly this implementation -- a second implementation in the controller would
sooner or later drift away from it, and then the measurements no longer apply.

Sequence per move:
  1. Repeat the steering value for steer_wait_s (steer at standstill; the first
     command on a fresh connection gets lost in DDS discovery).
  2. Send the travel as shaft rotation in degrees on /esp_serial_bridge/move.
     The ESP controls the travel via the encoders, not the EKF.
  3. Wait for the ack on /esp_serial_bridge/move_done.

During a move nobody may send /cmd_vel: the bridge would set the
steering again, and a motor command replaces the running move.

The owner (``node``) must provide:
    pub_steer, pub_move            publishers (Float32)
    steer_wait_s, move_timeout     seconds
    travel_tol_cm                  tolerance for the plausibility check
    get_logger()
and set ``ack`` from outside when move_done arrives:
    sequencer.ack = (timestamp, status, value)
"""
import math

from std_msgs.msg import Float32

from ekf.unpark import trajectory, cm_to_deg


def wrap(a):
    return math.atan2(math.sin(a), math.cos(a))


def pose_text(pose):
    return ('x=%+.3f m  y=%+.3f m  heading=%+.1f deg'
            % (pose[0], pose[1], math.degrees(pose[2])))


def reverse_steps(steps):
    """The sequence that undoes the outbound path: backwards through the list,
    every travel negated, every steering unchanged. That turns unparking into
    parking -- measured on the robot to about 2 cm."""
    return [(steer, -cm) for steer, cm in reversed(steps)]


class MoveSequencer:
    """Drives a step sequence through the bridge.

    First steer, then trigger the position move, then wait for the
    ack. No /cmd_vel in between.
    """

    def __init__(self, node, steps, name):
        self.node = node
        self.steps = steps
        self.name = name
        self.i = 0
        self.phase = 'steer'
        self.sent = False
        self.t0 = 0.0
        self.ack = None
        self.go_t = None
        self.pose0 = None
        self.error = None

    @property
    def done(self):
        return self.i >= len(self.steps)

    def tick(self, now, pose):
        """Call once per control tick. True when done or aborted
        (then the reason is in ``error``)."""
        if self.done:
            return True
        steer, cm = self.steps[self.i]

        if self.phase == 'steer':
            if not self.sent:
                self.sent = True
                self.t0 = now
            self.node.pub_steer.publish(Float32(data=float(steer)))
            if now - self.t0 < self.node.steer_wait_s:
                return False
            self.ack = None
            self.go_t = now
            self.pose0 = pose
            self.node.pub_move.publish(Float32(data=float(cm_to_deg(cm))))
            self.phase = 'drive'
            self.node.get_logger().info(
                "%s move %d/%d: steering %+.0f %%, %+.1f cm"
                % (self.name, self.i + 1, len(self.steps), steer, cm))
            return False

        q = self.ack
        if q is not None and q[0] >= self.go_t:
            status = q[1]
            plan = trajectory((0.0, 0.0, 0.0), [(steer, cm)])[-1][0]
            target_rot = math.degrees(plan[2])
            target_travel = math.hypot(plan[0], plan[1])
            actual_rot = math.degrees(wrap(pose[2] - self.pose0[2]))
            actual_travel = math.hypot(pose[0] - self.pose0[0], pose[1] - self.pose0[1])
            self.node.get_logger().info(
                "%s move %d done: rotation %+.1f deg (model %+.1f), "
                "travel %.1f cm (model %.1f)%s"
                % (self.name, self.i + 1, actual_rot, target_rot,
                   actual_travel * 100, target_travel * 100,
                   '' if status == 0 else '  [status %d]' % status))
            self.node.get_logger().info("           stands at  %s"
                                        % pose_text(pose))
            if status == 2:
                self.error = ("move %d was replaced by a motor command"
                               % (self.i + 1))
                return True
            # CAUTION: actual_travel comes from the EKF pose. If the localisation jumps,
            # this check fails although the move was driven correctly
            # (happened in the CCW test, run 5). As soon as it is clear what
            # move_done reports in data[2], q[2] should be used here.
            if status == 1 and abs(actual_travel - target_travel) > \
                    self.node.travel_tol_cm / 100.0:
                self.error = ("move %d: timeout AND %.1f cm too "
                               "short" % (self.i + 1, (target_travel - actual_travel) * 100))
                return True
            self.i += 1
            self.phase = 'steer'
            self.sent = False
            return self.done

        if now - self.go_t > self.node.move_timeout:
            self.error = ("move %d without ack after %.0f s -- is "
                           "esp_serial_bridge running?" % (self.i + 1, self.node.move_timeout))
            return True
        return False