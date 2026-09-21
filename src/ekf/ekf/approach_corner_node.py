#!/usr/bin/env python3
"""
APPROACH_CORNER controller (Round-1 milestone).

Evolution of the straight-line bring-up. Drives the start straight centred,
pose-locked against the 3-wall provisional map (start_map_3wall), and brakes on
the existing trapezoidal profile as it nears the first corner.

Key differences from the bring-up node:
  - The corner x-coordinate comes from the /front_wall_x topic (latched,
    transient-local) instead of a parameter. The node will not drive until it
    has received it -- no corner threshold means no safe stop.
  - The brake profile ramps down to v_approach (= turn-entry speed), NOT to
    zero. The only full stop is CORNER_REACHED, a pure safety backstop. In the
    final flow the turn-in trigger fires before that, TURN takes over carrying
    v_approach into the curve, and no stop ever happens -- the "no-stop"
    behaviour you specified.
  - When the robot crosses decision_window_dist (1.2 m from the wall) it logs a
    one-shot marker. That is where scan_processor's latched direction will be
    read once its publication exists. The controller does NOT run detection
    itself; perception votes every scan and latches. This marker only shows the
    timing/margin relative to braking during this isolated test.

Command convention: REP 103.
  linear.x  = forward velocity [m/s], positive = forward
  angular.z = yaw rate [rad/s],  positive = CCW = left
The esp_bridge does the Ackermann inverse kinematics and throttle mapping.
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from std_msgs.msg import Bool, Float64
import numpy as np


def yaw_from_quaternion(q):
    """Extract yaw (theta) from a geometry_msgs Quaternion. Same formula as scan_processor."""
    siny = 2.0 * (q.w * q.z + q.x * q.y)
    cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return float(np.arctan2(siny, cosy))


class ApproachCorner(Node):
    def __init__(self):
        super().__init__('approach_corner')

        # --- Field geometry (map frame; origin = start pose, +x = driving direction) ---
        # Corner / front-wall x-coordinate arrives on /front_wall_x. pose.x is the
        # rear axle, so nose_offset gives the nose position. MEASURE nose_offset.
        self.declare_parameter('nose_offset', 0.14)
        # Safety-backstop gap: nose-to-corner distance at the hard stop. This is a
        # placeholder terminal; TURN replaces it and normally fires earlier.
        self.declare_parameter('stop_gap', 0.35)
        # Where the direction decision window opens (distance from the wall, rear
        # axle). Logged as a one-shot marker; perception latches around here.
        self.declare_parameter('decision_window_dist', 1.2)
        self.declare_parameter('target_y', 0.0)   # 0 = centre line of the start straight

        # --- Lateral controller (identical to bring-up) ---
        self.declare_parameter('k_y', 1.5)
        self.declare_parameter('k_theta', 1.2)
        self.declare_parameter('max_yaw_rate', 1.5)

        # --- Longitudinal profile (m/s, m/s^2). v_approach = turn-entry speed. ---
        self.declare_parameter('v_cruise', 0.3)
        self.declare_parameter('v_approach', 0.2)
        # Braking begins at this nose-to-corner distance. Raise it (e.g. >=1.2) to
        # already be slow at the decision window if motion smear hurts detection.
        self.declare_parameter('brake_start', 1.0)
        self.declare_parameter('accel', 0.6)
        self.declare_parameter('decel', 1.0)

        # --- Sequencing / safety ---
        self.declare_parameter('require_button', True)
        self.declare_parameter('control_rate', 30.0)
        self.declare_parameter('odom_timeout', 0.3)

        g = lambda n: self.get_parameter(n).value
        self.nose_offset          = float(g('nose_offset'))
        self.stop_gap             = float(g('stop_gap'))
        self.decision_window_dist = float(g('decision_window_dist'))
        self.target_y             = float(g('target_y'))
        self.k_y                  = float(g('k_y'))
        self.k_theta              = float(g('k_theta'))
        self.max_yaw_rate         = float(g('max_yaw_rate'))
        self.v_cruise             = float(g('v_cruise'))
        self.v_approach           = float(g('v_approach'))
        self.brake_start          = float(g('brake_start'))
        self.accel                = float(g('accel'))
        self.decel                = float(g('decel'))
        self.require_button       = bool(g('require_button'))
        self.control_rate         = float(g('control_rate'))
        self.odom_timeout         = float(g('odom_timeout'))

        # --- State ---
        # WAIT_INPUTS -> WAIT_BUTTON -> DRIVE -> CORNER_REACHED
        self.state = 'WAIT_INPUTS'
        self.pose = None                 # (x, y, theta)
        self.front_wall_x = None         # from /front_wall_x (latched)
        self.last_odom_time = None
        self.button_pressed = False
        self.v_cmd = 0.0
        self._decision_window_logged = False

        # --- IO ---
        self.create_subscription(Odometry, '/ekf/odom', self.odom_cb, 10)

        # /front_wall_x is latched (transient-local). The subscriber MUST mirror
        # that durability or the connection never matches and the value never
        # arrives -- even though the publisher is up.
        latched = QoSProfile(depth=1)
        latched.durability = DurabilityPolicy.TRANSIENT_LOCAL
        self.create_subscription(Float64, '/front_wall_x', self.front_wall_cb, latched)

        if self.require_button:
            self.create_subscription(Bool, '/button_state', self.button_cb, 10)
        self.pub_cmd = self.create_publisher(Twist, '/cmd_vel', 10)

        self.dt = 1.0 / self.control_rate
        self.create_timer(self.dt, self.control_loop)

        self.get_logger().info(
            ">>> ApproachCorner bereit. Warte auf /ekf/odom und /front_wall_x... <<<"
        )

    # ------------------------------------------------------------------ callbacks
    def odom_cb(self, msg):
        p = msg.pose.pose
        self.pose = (p.position.x, p.position.y, yaw_from_quaternion(p.orientation))
        self.last_odom_time = self.get_clock().now()

    def front_wall_cb(self, msg):
        if self.front_wall_x is None:
            self.get_logger().info(f"/front_wall_x empfangen: {msg.data:.3f} m.")
        self.front_wall_x = float(msg.data)

    def button_cb(self, msg):
        if msg.data:
            self.button_pressed = True

    # ------------------------------------------------------------------ helpers
    def publish_stop(self):
        self.pub_cmd.publish(Twist())
        self.v_cmd = 0.0

    def compute_speed(self, front_dist):
        """Target speed from nose-to-corner distance; brake down to v_approach, then slew-limit."""
        if front_dist <= self.stop_gap:
            target = 0.0
        elif front_dist >= self.brake_start:
            target = self.v_cruise
        else:
            r = (front_dist - self.stop_gap) / (self.brake_start - self.stop_gap)
            target = self.v_approach + r * (self.v_cruise - self.v_approach)

        if target > self.v_cmd:
            self.v_cmd = min(target, self.v_cmd + self.accel * self.dt)
        else:
            self.v_cmd = max(target, self.v_cmd - self.decel * self.dt)
        return self.v_cmd

    def odom_is_stale(self):
        if self.last_odom_time is None:
            return True
        age = (self.get_clock().now() - self.last_odom_time).nanoseconds / 1e9
        return age > self.odom_timeout

    # ------------------------------------------------------------------ main loop
    def control_loop(self):
        if self.pose is None:
            return
        x, y, theta = self.pose

        if self.state == 'WAIT_INPUTS':
            # Do not drive without the corner threshold -- otherwise we have no
            # safe stop and would run into the wall.
            if self.front_wall_x is None:
                return
            self.state = 'WAIT_BUTTON' if self.require_button else 'DRIVE'
            if self.state == 'WAIT_BUTTON':
                self.get_logger().info("Eingaben da. Warte auf Button-Start...")
            else:
                self.get_logger().info("Eingaben da. Fahre los.")
            return

        if self.state == 'WAIT_BUTTON':
            if self.button_pressed:
                self.state = 'DRIVE'
                self.get_logger().info("Start.")
            else:
                self.publish_stop()
                return

        if self.state == 'CORNER_REACHED':
            # Safety backstop. In the final flow TURN takes over before this.
            self.publish_stop()
            return

        # --- DRIVE ---
        if self.odom_is_stale():
            self.get_logger().warn("Pose veraltet (kein /ekf/odom). Stoppe.")
            self.publish_stop()
            return

        wall_dist  = self.front_wall_x - x                    # rear axle -> corner
        front_dist = wall_dist - self.nose_offset             # nose -> corner

        # One-shot decision-window marker (where the latched direction gets read later).
        if not self._decision_window_logged and wall_dist <= self.decision_window_dist:
            self._decision_window_logged = True
            self.get_logger().info(
                f"--- ENTSCHEIDUNGSFENSTER bei {wall_dist:.2f} m (v={self.v_cmd:.2f} m/s). "
                f"Hier wird spaeter die gelatchte Richtung gelesen. ---"
            )

        if front_dist <= self.stop_gap:
            self.state = 'CORNER_REACHED'
            self.publish_stop()
            self.get_logger().info(
                f"CORNER_REACHED (Sicherheits-Halt). front_dist={front_dist:.2f} m, x={x:.2f} m. "
                f"Hier uebernimmt spaeter TURN."
            )
            return

        # lateral PD -> yaw rate (REP 103: +y left, +omega CCW/left)
        e_y = self.target_y - y
        e_theta = 0.0 - theta
        omega = self.k_y * e_y + self.k_theta * e_theta
        omega = max(-self.max_yaw_rate, min(self.max_yaw_rate, omega))

        v = self.compute_speed(front_dist)

        cmd = Twist()
        cmd.linear.x = float(v)
        cmd.angular.z = float(omega)
        self.pub_cmd.publish(cmd)


def main(args=None):
    rclpy.init(args=args)
    node = ApproachCorner()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.publish_stop()
        except Exception:
            pass
        if node.context.ok():
            node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()