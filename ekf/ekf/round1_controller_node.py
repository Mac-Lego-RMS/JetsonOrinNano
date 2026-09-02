#!/usr/bin/env python3
"""
Round-1 controller (first corner).

Evolution of approach_corner_node. Adds the turn:

  APPROACH  -- centered drive against the 3-wall provisional map, brake profile
               (unchanged from approach_corner).
  DECIDE    -- once /corner_geometry and /race_direction are latched (they
               publish near the corner), plan the arc for the first corner.
  TURN      -- pose-native arc tracking: cross-track to the planned arc +
               heading to the arc tangent + feedforward omega = s*v/R. No wall
               anchor -- the wall enters only implicitly via the EKF pose.
  EXIT      -- hold the exit line straight for a short distance, then stop.
               (Isolated-validation terminal; later this hands to the next
               APPROACH.)

Geometry, all in the start-anchored map frame:
  - entry line  = outer side wall A, offset inward by o_in  (0.5 m default)
  - exit line   = outer front wall B, offset inward by o_out (0.5 m default)
  - the two offset lines meet at 90 deg at P; the inscribed arc of radius R is
    tangent to both. Center C = P + R*nA + R*nB (nA,nB inward unit normals).
    Tangent points T_A = C - R*nA (entry), T_B = C - R*nB (exit).
  - turn sign s: CCW=+1 (left turns), CW=-1 (right turns).

ASSUMPTION for this first fixed version: 1.0 m start lane, so centered driving
(target_y=0) already sits on the 0.5 m entry line -> no entry transient. In a
0.6 m lane use o_in=0.3 instead.

Command convention: REP 103 (linear.x m/s fwd, angular.z rad/s CCW=left).
The esp_bridge does the Ackermann inverse kinematics.

NOTE: the arc-tracking signs (s, cross-track, heading) are geometry-derived but
should be confirmed on a jacked-up bench run before driving -- the usual sign
check.
"""

import math
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from std_msgs.msg import Bool, Float64, String
import numpy as np


def yaw_from_quaternion(q):
    siny = 2.0 * (q.w * q.z + q.x * q.y)
    cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return float(np.arctan2(siny, cosy))


def wrap(a):
    """Wrap angle to (-pi, pi]."""
    return math.atan2(math.sin(a), math.cos(a))


def line_intersect(l1, l2):
    """Intersect two lines (nx,ny,d): nx*x+ny*y=d. Returns (x,y) or None if parallel."""
    n1x, n1y, d1 = l1
    n2x, n2y, d2 = l2
    det = n1x * n2y - n1y * n2x
    if abs(det) < 1e-9:
        return None
    x = (d1 * n2y - d2 * n1y) / det
    y = (n1x * d2 - n2x * d1) / det
    return (x, y)


class Round1Controller(Node):
    def __init__(self):
        super().__init__('round1_controller')

        # --- Corner-stop safety (APPROACH only) ---
        self.declare_parameter('nose_offset', 0.14)
        self.declare_parameter('stop_gap', 0.35)
        self.declare_parameter('target_y', 0.0)         # centered approach

        # --- Turn geometry ---
        self.declare_parameter('o_in', 0.50)            # entry offset from outer wall
        self.declare_parameter('o_out', 0.50)           # exit offset from outer wall
        self.declare_parameter('turn_radius', 0.40)     # R, field-tunable
        self.declare_parameter('sweep_tol_deg', 6.0)    # completion tolerance on the 90 deg sweep
        self.declare_parameter('exit_hold_dist', 1.0)  # how far to hold the exit line before stopping

        # --- Lateral PD (straight: approach + exit) ---
        self.declare_parameter('k_y', 1.5)
        self.declare_parameter('k_theta', 1.2)

        # --- Arc tracking (turn) ---
        self.declare_parameter('k_ct', 3.0)             # cross-track-to-arc gain
        self.declare_parameter('k_th', 1.0)             # heading-to-tangent gain

        self.declare_parameter('max_yaw_rate', 3.0)     # rad/s clamp (turn needs more than straight)

        # --- Longitudinal (m/s) ---
        self.declare_parameter('v_cruise', 0.5)
        self.declare_parameter('v_turn', 0.45)          # constant speed through the arc
        self.declare_parameter('v_approach', 0.25)      # brake target near corner (= turn entry speed)
        self.declare_parameter('brake_start', 1.0)
        self.declare_parameter('accel', 0.6)
        self.declare_parameter('decel', 1.0)

        # --- Sequencing / safety ---
        self.declare_parameter('require_button', False)
        self.declare_parameter('control_rate', 30.0)
        self.declare_parameter('odom_timeout', 0.3)

        g = lambda n: self.get_parameter(n).value
        self.nose_offset   = float(g('nose_offset'))
        self.stop_gap      = float(g('stop_gap'))
        self.target_y      = float(g('target_y'))
        self.o_in          = float(g('o_in'))
        self.o_out         = float(g('o_out'))
        self.R             = float(g('turn_radius'))
        self.sweep_tol     = math.radians(float(g('sweep_tol_deg')))
        self.exit_hold_dist = float(g('exit_hold_dist'))
        self.k_y           = float(g('k_y'))
        self.k_theta       = float(g('k_theta'))
        self.k_ct          = float(g('k_ct'))
        self.k_th          = float(g('k_th'))
        self.max_yaw_rate  = float(g('max_yaw_rate'))
        self.v_cruise      = float(g('v_cruise'))
        self.v_turn        = float(g('v_turn'))
        self.v_approach    = float(g('v_approach'))
        self.brake_start   = float(g('brake_start'))
        self.accel         = float(g('accel'))
        self.decel         = float(g('decel'))
        self.require_button = bool(g('require_button'))
        self.control_rate  = float(g('control_rate'))
        self.odom_timeout  = float(g('odom_timeout'))

        # --- State ---
        # WAIT_INPUTS -> WAIT_BUTTON -> APPROACH -> TURN -> EXIT -> DONE
        self.state = 'WAIT_INPUTS'
        self.pose = None
        self.front_wall_x = None
        self.race_direction = None        # 'CW' | 'CCW'
        self.corner_walls = None          # list of (nx,ny,d), inward normals
        self.last_odom_time = None
        self.button_pressed = False
        self.v_cmd = 0.0

        # arc plan (filled in DECIDE)
        self.arc = None                   # dict: C, s, T_A, T_B, a0, travel, LB, u_B
        self.exit_start_xy = None

        # --- IO ---
        latched = QoSProfile(depth=1)
        latched.durability = DurabilityPolicy.TRANSIENT_LOCAL

        self.create_subscription(Odometry, '/ekf/odom', self.odom_cb, 10)
        self.create_subscription(Float64, '/front_wall_x', self.front_wall_cb, latched)
        self.create_subscription(String, '/race_direction', self.direction_cb, latched)
        self.create_subscription(self._corner_msg_type(), '/corner_geometry',
                                 self.corner_cb, latched)
        if self.require_button:
            self.create_subscription(Bool, '/button_state', self.button_cb, 10)
        self.pub_cmd = self.create_publisher(Twist, '/cmd_vel', 10)

        self.dt = 1.0 / self.control_rate
        self.create_timer(self.dt, self.control_loop)

        self.get_logger().info(">>> Round1Controller bereit. Warte auf /ekf/odom und /front_wall_x... <<<")

    def _corner_msg_type(self):
        from robot_msgs.msg import CornerGeometry
        return CornerGeometry

    # ------------------------------------------------------------------ callbacks
    def odom_cb(self, msg):
        p = msg.pose.pose
        self.pose = (p.position.x, p.position.y, yaw_from_quaternion(p.orientation))
        self.last_odom_time = self.get_clock().now()

    def front_wall_cb(self, msg):
        if self.front_wall_x is None:
            self.get_logger().info(f"/front_wall_x empfangen: {msg.data:.3f} m.")
        self.front_wall_x = float(msg.data)

    def direction_cb(self, msg):
        if self.race_direction is None:
            self.get_logger().info(f"/race_direction empfangen: {msg.data}.")
        self.race_direction = msg.data

    def corner_cb(self, msg):
        walls = [(w.nx, w.ny, w.d) for w in msg.walls]
        if self.corner_walls is None:
            self.get_logger().info(f"/corner_geometry empfangen: {len(walls)} Waende.")
        self.corner_walls = walls

    def button_cb(self, msg):
        if msg.data:
            self.button_pressed = True

    # ------------------------------------------------------------------ helpers
    def publish_stop(self):
        self.pub_cmd.publish(Twist())
        self.v_cmd = 0.0

    def compute_speed(self, front_dist):
        if front_dist <= self.stop_gap:
            target = self.v_approach
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

    def geometry_ready(self):
        return self.corner_walls is not None and self.race_direction in ('CW', 'CCW')

    def plan_arc(self):
        """Plan the inscribed arc for the first corner from the box walls + direction."""
        _, _, th = self.pose
        tx, ty = math.cos(th), math.sin(th)          # travel direction (snapshot on the straight)
        s = 1.0 if self.race_direction == 'CCW' else -1.0

        # entry-wall inward normal points to rotate(travel, s*90deg)
        # +90: (x,y)->(-y,x); scaled by s handles CW as -90
        ax_dir = (-s * ty, s * tx)

        walls = self.corner_walls
        # exit/front wall B: inward normal most opposite to travel
        B = min(walls, key=lambda w: w[0] * tx + w[1] * ty)
        # entry/outer side wall A: inward normal aligned with ax_dir
        A = max(walls, key=lambda w: w[0] * ax_dir[0] + w[1] * ax_dir[1])

        LA = (A[0], A[1], A[2] + self.o_in)          # entry line (offset inward)
        LB = (B[0], B[1], B[2] + self.o_out)         # exit line

        P = line_intersect(LA, LB)
        if P is None:
            self.get_logger().error("Eintritts-/Austrittslinie parallel -- kann Bogen nicht planen.")
            return False

        R = self.R
        C = (P[0] + R * (A[0] + B[0]), P[1] + R * (A[1] + B[1]))
        T_A = (C[0] - R * A[0], C[1] - R * A[1])
        T_B = (C[0] - R * B[0], C[1] - R * B[1])
        a0 = math.atan2(T_A[1] - C[1], T_A[0] - C[0])

        # exit travel direction (tangent at T_B)
        rbx, rby = (T_B[0] - C[0]) / R, (T_B[1] - C[1]) / R
        u_B = (-s * rby, s * rbx)

        self.arc = dict(C=C, s=s, T_A=T_A, T_B=T_B, a0=a0, travel=(tx, ty), LB=LB, u_B=u_B)
        self.get_logger().info(
            f"DECIDE: Richtung={self.race_direction}, R={R:.2f} m. "
            f"T_A=({T_A[0]:.2f},{T_A[1]:.2f}) T_B=({T_B[0]:.2f},{T_B[1]:.2f})."
        )
        return True

    def straight_steer(self, x, y, theta, target_line=None):
        """Lateral PD -> yaw rate. Against target_y (approach) or a target line (exit)."""
        if target_line is None:
            e_y = self.target_y - y
        else:
            nx, ny, d = target_line
            # signed distance to line; inward normal, so positive = interior side
            e_y = d - (nx * x + ny * y)
        e_theta = -theta if target_line is None else wrap(math.atan2(self.arc['u_B'][1],
                                                                     self.arc['u_B'][0]) - theta)
        omega = self.k_y * e_y + self.k_theta * e_theta
        return max(-self.max_yaw_rate, min(self.max_yaw_rate, omega))

    # ------------------------------------------------------------------ main loop
    def control_loop(self):
        if self.pose is None:
            return
        x, y, theta = self.pose

        if self.state == 'WAIT_INPUTS':
            if self.front_wall_x is None:
                return
            self.state = 'WAIT_BUTTON' if self.require_button else 'APPROACH'
            self.get_logger().info("Eingaben da. " +
                                   ("Warte auf Button-Start..." if self.require_button else "Fahre los."))
            return

        if self.state == 'WAIT_BUTTON':
            if self.button_pressed:
                self.state = 'APPROACH'
                self.get_logger().info("Start.")
            else:
                self.publish_stop()
            return

        if self.state == 'DONE':
            self.publish_stop()
            return

        if self.odom_is_stale():
            self.get_logger().warn("Pose veraltet (kein /ekf/odom). Stoppe.")
            self.publish_stop()
            return

        # ---------------- APPROACH ----------------
        if self.state == 'APPROACH':
            wall_dist = self.front_wall_x - x
            front_dist = wall_dist - self.nose_offset

            # Once geometry is latched, plan the arc and watch for the turn-in point.
            if self.arc is None and self.geometry_ready():
                self.plan_arc()

            if self.arc is not None:
                # turn-in when pose crosses T_A along travel
                tA = self.arc['T_A']; tr = self.arc['travel']
                if (x - tA[0]) * tr[0] + (y - tA[1]) * tr[1] >= 0.0:
                    self.state = 'TURN'
                    self.get_logger().info("TURN: Einlenken.")
                    return

            # safety backstop: reached corner without turning in -> stop
            if front_dist <= self.stop_gap:
                self.state = 'DONE'
                self.publish_stop()
                self.get_logger().warn(
                    f"CORNER_REACHED ohne Einlenken (front_dist={front_dist:.2f}). "
                    f"Geometrie zu spaet gelatcht? Stoppe."
                )
                return

            omega = self.straight_steer(x, y, theta)
            v = self.compute_speed(front_dist)
            cmd = Twist(); cmd.linear.x = float(v); cmd.angular.z = float(omega)
            self.pub_cmd.publish(cmd)
            return

        # ---------------- TURN ----------------
        if self.state == 'TURN':
            C = self.arc['C']; s = self.arc['s']; R = self.R
            rx, ry = x - C[0], y - C[1]
            dist = math.hypot(rx, ry)
            if dist < 1e-6:
                dist = 1e-6
            r_hat = (rx / dist, ry / dist)

            # cross-track to arc (positive = outside the circle)
            e_ct = dist - R
            # desired tangent heading on the arc
            t_hat = (-s * r_hat[1], s * r_hat[0])
            psi_d = math.atan2(t_hat[1], t_hat[0])
            e_th = wrap(psi_d - theta)

            omega = s * (self.v_turn / R) + s * self.k_ct * e_ct + self.k_th * e_th
            omega = max(-self.max_yaw_rate, min(self.max_yaw_rate, omega))

            # completion: swept 90 deg around C
            a = math.atan2(ry, rx)
            swept = s * wrap(a - self.arc['a0'])
            if swept < -0.1:
                swept += 2.0 * math.pi
            if swept >= (math.pi / 2.0 - self.sweep_tol):
                self.state = 'EXIT'
                self.exit_start_xy = (x, y)
                self.get_logger().info(
                    f"TURN fertig (swept={math.degrees(swept):.1f} deg, theta={math.degrees(theta):.1f} deg). "
                    f"Halte Austrittslinie."
                )
                return

            cmd = Twist(); cmd.linear.x = float(self.v_turn); cmd.angular.z = float(omega)
            self.pub_cmd.publish(cmd)
            return

        # ---------------- EXIT ----------------
        if self.state == 'EXIT':
            ex, ey = self.exit_start_xy
            travelled = math.hypot(x - ex, y - ey)
            if travelled >= self.exit_hold_dist:
                self.state = 'DONE'
                self.publish_stop()
                self.get_logger().info(f"EXIT fertig ({travelled:.2f} m gehalten). STOP.")
                return

            omega = self.straight_steer(x, y, theta, target_line=self.arc['LB'])
            cmd = Twist(); cmd.linear.x = float(self.v_turn); cmd.angular.z = float(omega)
            self.pub_cmd.publish(cmd)
            return


def main(args=None):
    rclpy.init(args=args)
    node = Round1Controller()
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