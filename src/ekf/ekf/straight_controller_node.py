#!/usr/bin/env python3
"""
Straight-line bring-up controller.

Drives the FIRST straight of the run using only the fused pose from /ekf/odom:
  - lateral PD keeps the robot on the centre line (map x-axis, since the map
    origin sits on the start pose and +x points down the first straight),
  - forward speed ramps down and stops at a fixed distance from the front wall,
    whose x-coordinate in the map frame is known from the start position.

No wall subscription. Perception stays in scan_processor / the EKF stack.

Command convention: REP 103.
  linear.x  = forward velocity [m/s], positive = forward
  angular.z = yaw rate [rad/s],  positive = CCW = left
The esp_bridge is responsible for the Ackermann inverse kinematics
(delta = atan(L * omega / v)) and the throttle mapping.
"""

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from std_msgs.msg import Bool
import numpy as np


def yaw_from_quaternion(q):
    """Extract yaw (theta) from a geometry_msgs Quaternion. Same formula as scan_processor."""
    siny = 2.0 * (q.w * q.z + q.x * q.y)
    cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return float(np.arctan2(siny, cosy))


class StraightController(Node):
    def __init__(self):
        super().__init__('straight_controller')

        # --- Field geometry (map frame; origin = start pose, +x = driving direction) ---
        # front_wall_x is simply the start distance to the front wall, because the
        # rear axle starts at the origin. Set 1.95 or 1.45 per start position.
        self.declare_parameter('front_wall_x', 1.95)
        # Desired NOSE-to-wall gap at stop.
        self.declare_parameter('stop_distance', 0.40)
        # Rear-axle -> front-bumper offset. pose.x is the rear axle, so we subtract
        # this to get the nose position. MEASURE THIS on the real car.
        # Safety direction: too large only stops earlier, too small risks a crash.
        self.declare_parameter('nose_offset', 0.14)
        # Target lateral position. 0.0 = centre line. Later, when processed_walls
        # exists, this gets driven from the measured left/right wall distances so the
        # robot can centre itself even from an off-centre start. For now: constant.
        self.declare_parameter('target_y', 0.0)

        # --- Lateral controller ---
        self.declare_parameter('k_y', 2.0)          # cross-track gain [rad/s per m]
        self.declare_parameter('k_theta', 1.2)      # heading gain     [rad/s per rad]
        self.declare_parameter('max_yaw_rate', 1.5) # clamp [rad/s]

        # --- Longitudinal profile (all in m/s, m/s^2) ---
        self.declare_parameter('v_cruise', 0.3)
        self.declare_parameter('v_approach', 0.20)   # crawl speed near the wall
        self.declare_parameter('brake_start', 1.0)  # front_dist at which braking begins
        self.declare_parameter('accel', 0.6)
        self.declare_parameter('decel', 1.0)

        # --- Sequencing / safety ---
        self.declare_parameter('require_button', True)
        self.declare_parameter('control_rate', 30.0)  # Hz
        self.declare_parameter('odom_timeout', 0.3)   # stop if pose goes stale [s]

        g = lambda n: self.get_parameter(n).value
        self.front_wall_x  = float(g('front_wall_x'))
        self.stop_distance = float(g('stop_distance'))
        self.nose_offset   = float(g('nose_offset'))
        self.target_y      = float(g('target_y'))
        self.k_y           = float(g('k_y'))
        self.k_theta       = float(g('k_theta'))
        self.max_yaw_rate  = float(g('max_yaw_rate'))
        self.v_cruise      = float(g('v_cruise'))
        self.v_approach    = float(g('v_approach'))
        self.brake_start   = float(g('brake_start'))
        self.accel         = float(g('accel'))
        self.decel         = float(g('decel'))
        self.require_button = bool(g('require_button'))
        self.control_rate  = float(g('control_rate'))
        self.odom_timeout  = float(g('odom_timeout'))

        # --- State ---
        # WAIT_ODOM -> WAIT_BUTTON -> DRIVE -> STOPPED
        self.state = 'WAIT_ODOM'
        self.pose = None             # (x, y, theta)
        self.last_odom_time = None   # rclpy Time
        self.button_pressed = False
        self.v_cmd = 0.0             # slew-limited commanded speed

        # --- IO ---
        self.create_subscription(Odometry, '/ekf/odom', self.odom_cb, 10)
        if self.require_button:
            self.create_subscription(Bool, '/button_state', self.button_cb, 10)
        self.pub_cmd = self.create_publisher(Twist, '/cmd_vel', 10)

        self.dt = 1.0 / self.control_rate
        self.create_timer(self.dt, self.control_loop)

        self.get_logger().info(
            f">>> StraightController bereit. front_wall_x={self.front_wall_x:.2f} m, "
            f"Stopp bei {self.stop_distance:.2f} m (nose_offset={self.nose_offset:.2f} m). "
            f"Warte auf /ekf/odom... <<<"
        )

    # ------------------------------------------------------------------ callbacks
    def odom_cb(self, msg):
        p = msg.pose.pose
        self.pose = (p.position.x, p.position.y, yaw_from_quaternion(p.orientation))
        self.last_odom_time = self.get_clock().now()

    def button_cb(self, msg):
        if msg.data:
            self.button_pressed = True

    # ------------------------------------------------------------------ helpers
    def publish_stop(self):
        self.pub_cmd.publish(Twist())  # all-zero: v=0, omega=0
        self.v_cmd = 0.0

    def compute_speed(self, front_dist):
        """Target speed from front_dist, then slew-limit toward it."""
        if front_dist <= self.stop_distance:
            target = 0.0
        elif front_dist >= self.brake_start:
            target = self.v_cruise
        else:
            r = (front_dist - self.stop_distance) / (self.brake_start - self.stop_distance)
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

        if self.state == 'WAIT_ODOM':
            self.state = 'WAIT_BUTTON' if self.require_button else 'DRIVE'
            if self.state == 'WAIT_BUTTON':
                self.get_logger().info("Pose empfangen. Warte auf Button-Start...")
            else:
                self.get_logger().info("Pose empfangen. Fahre los.")
            return

        if self.state == 'WAIT_BUTTON':
            if self.button_pressed:
                self.state = 'DRIVE'
                self.get_logger().info("Start.")
            else:
                self.publish_stop()
                return

        if self.state == 'STOPPED':
            self.publish_stop()
            return

        # --- DRIVE ---
        # Safety: if the pose feed dies, do not keep driving blind toward the wall.
        if self.odom_is_stale():
            self.get_logger().warn("Pose veraltet (kein /ekf/odom). Stoppe.")
            self.publish_stop()
            return

        # nose-to-wall distance in the map frame
        front_dist = self.front_wall_x - x - self.nose_offset

        if front_dist <= self.stop_distance:
            self.state = 'STOPPED'
            self.publish_stop()
            self.get_logger().info(f"Ziel erreicht. front_dist={front_dist:.2f} m, x={x:.2f} m.")
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
    node = StraightController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # Make sure the car is commanded to stop before we go down.
        try:
            node.publish_stop()
        except Exception:
            pass
        if node.context.ok():
            node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()