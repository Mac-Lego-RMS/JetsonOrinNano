#!/usr/bin/env python3
"""
Round-1 controller (first corner) -- consolidated.

State machine:
  WAIT_INPUTS -> [WAIT_BUTTON] -> APPROACH -> TURN -> EXIT -> DONE

  APPROACH  Centred drive against the 3-wall provisional map, trapezoidal brake
            profile toward the corner. Once /corner_geometry and /race_direction
            are latched (they publish near the corner), the arc is planned and
            the turn-in point T_A is watched.
  TURN      Pose-native arc tracking: cross-track to the planned circle +
            heading to the arc tangent + speed-honest feedforward omega = s*v/R
            (uses the MEASURED speed so the effective radius stays R even if v
            dips). The feedforward is blended out over the last ff_blend_deg
            before theta_target so the hand-off to EXIT is ruck-free (no
            over-rotation). Completion is theta-based: end when the real vehicle
            heading reaches theta_target = theta_start + s*90deg.
  EXIT      Hold the exit line (outer wall B offset inward by o_out) straight
            for exit_hold_dist, then stop. Later this hands to the next APPROACH.

Geometry (start-anchored map frame):
  entry line = outer side wall A offset inward by o_in
  exit line  = outer front wall B offset inward by o_out
  The offset lines meet at 90 deg at P; the inscribed arc of radius R is tangent
  to both. Center C = P + R*(nA + nB), tangent points T_A = C - R*nA,
  T_B = C - R*nB. Turn sign s: CCW = +1 (left), CW = -1 (right).

Assumption (first fixed version): 1.0 m lanes, so o_in = o_out = 0.5 = lane
centre. For 0.6 m lanes use 0.3.

Command convention: REP 103 (linear.x m/s fwd, angular.z rad/s CCW=left). The
esp_bridge does the calibrated Ackermann inverse (angular.z -> steering angle
via delta = atan(L*omega/v)) and the speed control.

Validated: CCW first corner (angle + position). Open: CW gegentest, corner
selection for corners 2-4, occasional /ekf/odom stale gaps (sensor side).
"""

import math

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy
from rcl_interfaces.msg import SetParametersResult
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from std_msgs.msg import Bool, Float64, String


def yaw_from_quaternion(q):
    siny = 2.0 * (q.w * q.z + q.x * q.y)
    cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny, cosy)


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

    # param_name -> (attribute_name, converter). Single source of truth so that
    # init-load and live re-load on `ros2 param set` stay consistent.
    _PARAMS = {
        'nose_offset':   ('nose_offset',   float),
        'stop_gap':      ('stop_gap',      float),
        'target_y':      ('target_y',      float),
        'o_in':          ('o_in',          float),
        'o_out':         ('o_out',         float),
        'turn_radius':   ('R',             float),
        'sweep_tol_deg': ('sweep_tol',     lambda v: math.radians(float(v))),
        'ff_blend_deg':  ('ff_blend',      lambda v: math.radians(float(v))),
        'exit_hold_dist':('exit_hold_dist',float),
        'k_y':           ('k_y',           float),
        'k_theta':       ('k_theta',       float),
        'k_ct':          ('k_ct',          float),
        'k_th':          ('k_th',          float),
        'max_yaw_rate':  ('max_yaw_rate',  float),
        'v_cruise':      ('v_cruise',      float),
        'v_turn':        ('v_turn',        float),
        'v_approach':    ('v_approach',    float),
        'brake_start':   ('brake_start',   float),
        'accel':         ('accel',         float),
        'decel':         ('decel',         float),
    }

    def __init__(self):
        super().__init__('round1_controller')

        # --- Corner-stop safety (APPROACH only) ---
        self.declare_parameter('nose_offset', 0.14)
        self.declare_parameter('stop_gap', 0.35)
        self.declare_parameter('target_y', 0.0)          # centred approach

        # --- Turn geometry ---
        self.declare_parameter('o_in', 0.50)             # entry offset from outer wall
        self.declare_parameter('o_out', 0.50)            # exit offset from outer wall
        self.declare_parameter('turn_radius', 0.40)      # R, field-tunable
        self.declare_parameter('sweep_tol_deg', 3.0)     # theta-completion tolerance
        self.declare_parameter('ff_blend_deg', 20.0)     # feedforward blend-out window before target
        self.declare_parameter('exit_hold_dist', 1.0)    # exit-line hold distance

        # --- Lateral PD (straight: approach + exit) ---
        self.declare_parameter('k_y', 3.4)
        self.declare_parameter('k_theta', 2.5)

        # --- Arc tracking (turn) ---
        self.declare_parameter('k_ct', 3.0)              # cross-track-to-arc gain
        self.declare_parameter('k_th', 1.0)              # heading-to-tangent gain
        self.declare_parameter('max_yaw_rate', 3.0)      # rad/s clamp

        # --- Longitudinal (m/s) ---
        self.declare_parameter('v_cruise', 0.5)
        self.declare_parameter('v_turn', 0.45)           # speed through the arc
        self.declare_parameter('v_approach', 0.25)       # brake target near corner
        self.declare_parameter('brake_start', 1.0)
        self.declare_parameter('accel', 0.6)
        self.declare_parameter('decel', 1.0)

        # --- Sequencing / safety (structural: read once) ---
        self.declare_parameter('require_button', False)
        self.declare_parameter('control_rate', 30.0)
        self.declare_parameter('odom_timeout', 0.3)

        # load tunables + structural once
        self._load_params()
        self.require_button = bool(self.get_parameter('require_button').value)
        self.control_rate = float(self.get_parameter('control_rate').value)
        self.odom_timeout = float(self.get_parameter('odom_timeout').value)
        # live re-load on `ros2 param set` (fixes silent "param didn't apply")
        self.add_on_set_parameters_callback(self._on_params)

        # --- State ---
        self.state = 'WAIT_INPUTS'
        self.pose = None                  # (x, y, theta)
        self.v_ist = 0.0                  # measured forward speed (EKF)
        self.front_wall_x = None
        self.race_direction = None        # 'CW' | 'CCW'
        self.corner_walls = None          # list of (nx,ny,d), inward normals
        self.last_odom_time = None
        self.button_pressed = False
        self.v_cmd = 0.0
        self.arc = None                   # planned in DECIDE
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

    # ------------------------------------------------------------------ params
    def _load_params(self):
        for name, (attr, conv) in self._PARAMS.items():
            setattr(self, attr, conv(self.get_parameter(name).value))

    def _on_params(self, params):
        for p in params:
            if p.name in self._PARAMS:
                attr, conv = self._PARAMS[p.name]
                setattr(self, attr, conv(p.value))
        return SetParametersResult(successful=True)

    # ------------------------------------------------------------------ callbacks
    def odom_cb(self, msg):
        p = msg.pose.pose
        self.pose = (p.position.x, p.position.y, yaw_from_quaternion(p.orientation))
        self.v_ist = float(msg.twist.twist.linear.x)
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

    def publish_cmd(self, v, omega):
        cmd = Twist()
        cmd.linear.x = float(v)
        cmd.angular.z = float(max(-self.max_yaw_rate, min(self.max_yaw_rate, omega)))
        self.pub_cmd.publish(cmd)

    def compute_speed(self, front_dist):
        """Trapezoidal brake profile toward the corner, slew-limited."""
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

    def straight_steer(self, x, y, theta, target_line=None):
        """Lateral PD -> yaw rate. Against target_y (approach) or a target line (exit)."""
        if target_line is None:
            e_y = self.target_y - y
            e_theta = -theta
        else:
            nx, ny, d = target_line
            e_y = d - (nx * x + ny * y)          # signed dist; inward normal -> +interior
            heading_B = math.atan2(self.arc['u_B'][1], self.arc['u_B'][0])
            e_theta = wrap(heading_B - theta)
        return self.k_y * e_y + self.k_theta * e_theta

    def plan_arc(self):
        """Plan the inscribed arc for the first corner from the box walls + direction."""
        _, _, th = self.pose
        tx, ty = math.cos(th), math.sin(th)          # travel direction (snapshot on the straight)
        s = 1.0 if self.race_direction == 'CCW' else -1.0

        # entry-wall inward normal points to rotate(travel, s*90deg)
        ax_dir = (-s * ty, s * tx)

        walls = self.corner_walls
        B = min(walls, key=lambda w: w[0] * tx + w[1] * ty)          # front wall (exit)
        A = max(walls, key=lambda w: w[0] * ax_dir[0] + w[1] * ax_dir[1])  # outer side wall (entry)

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

        rbx, rby = (T_B[0] - C[0]) / R, (T_B[1] - C[1]) / R
        u_B = (-s * rby, s * rbx)                    # exit travel direction (tangent at T_B)

        theta_target = wrap(th + s * math.pi / 2.0)  # theta-based completion target

        self.arc = dict(C=C, s=s, T_A=T_A, T_B=T_B, a0=a0, travel=(tx, ty),
                        LB=LB, u_B=u_B, theta_target=theta_target)

        self.get_logger().info(
            f"DECIDE: Richtung={self.race_direction}, R={R:.2f} m. "
            f"T_A=({T_A[0]:.2f},{T_A[1]:.2f}) T_B=({T_B[0]:.2f},{T_B[1]:.2f}) "
            f"theta_target={math.degrees(theta_target):.1f} deg."
        )
        return True

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

        if self.state == 'APPROACH':
            self._approach(x, y, theta)
        elif self.state == 'TURN':
            self._turn(x, y, theta)
        elif self.state == 'EXIT':
            self._exit(x, y, theta)

    # ------------------------------------------------------------------ states
    def _approach(self, x, y, theta):
        front_dist = self.front_wall_x - x - self.nose_offset

        # once geometry is latched, plan the arc and watch for the turn-in point
        if self.arc is None and self.geometry_ready():
            self.plan_arc()

        if self.arc is not None:
            tA = self.arc['T_A']; tr = self.arc['travel']
            if (x - tA[0]) * tr[0] + (y - tA[1]) * tr[1] >= 0.0:   # crossed T_A along travel
                self.state = 'TURN'
                self.get_logger().info(
                    f"TURN: Einlenken bei ({x:.2f},{y:.2f}, {math.degrees(theta):.1f} deg).")
                return

        # safety backstop: reached corner without turning in
        if front_dist <= self.stop_gap:
            self.state = 'DONE'
            self.publish_stop()
            self.get_logger().warn(
                f"CORNER_REACHED ohne Einlenken (front_dist={front_dist:.2f}). "
                f"Geometrie zu spaet gelatcht? Stoppe.")
            return

        omega = self.straight_steer(x, y, theta)
        v = self.compute_speed(front_dist)
        self.publish_cmd(v, omega)

    def _turn(self, x, y, theta):
        C = self.arc['C']; s = self.arc['s']; R = self.R
        rx, ry = x - C[0], y - C[1]
        dist = math.hypot(rx, ry) or 1e-6
        r_hat = (rx / dist, ry / dist)

        e_ct = dist - R                              # cross-track to circle (+ = outside)
        t_hat = (-s * r_hat[1], s * r_hat[0])        # desired tangent heading
        e_th = wrap(math.atan2(t_hat[1], t_hat[0]) - theta)

        theta_err = wrap(self.arc['theta_target'] - theta)

        # speed-honest feedforward, blended out over the last ff_blend rad so the
        # hand-off to EXIT carries no residual steering (no over-rotation).
        v_meas = abs(self.v_ist) if abs(self.v_ist) > 0.05 else self.v_turn
        blend = max(0.0, min(1.0, abs(theta_err) / self.ff_blend)) if self.ff_blend > 1e-6 else 1.0
        omega = s * (v_meas / R) * blend + s * self.k_ct * e_ct + self.k_th * e_th

        # theta-based completion: end when the real heading reaches theta_target
        if s * theta_err <= self.sweep_tol:
            a = math.atan2(ry, rx)
            swept = s * wrap(a - self.arc['a0'])
            if swept < -0.1:
                swept += 2.0 * math.pi
            self.state = 'EXIT'
            self.exit_start_xy = (x, y)
            self.get_logger().info(
                f"TURN fertig (theta={math.degrees(theta):.1f} deg, "
                f"ziel={math.degrees(self.arc['theta_target']):.1f} deg, "
                f"swept={math.degrees(swept):.1f} deg).")
            return

        self.publish_cmd(self.v_turn, omega)

    def _exit(self, x, y, theta):
        ex, ey = self.exit_start_xy
        travelled = math.hypot(x - ex, y - ey)
        if travelled >= self.exit_hold_dist:
            self.state = 'DONE'
            self.publish_stop()
            self.get_logger().info(f"EXIT fertig ({travelled:.2f} m gehalten). STOP.")
            return

        omega = self.straight_steer(x, y, theta, target_line=self.arc['LB'])
        self.publish_cmd(self.v_turn, omega)


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