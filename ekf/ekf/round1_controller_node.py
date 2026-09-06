#!/usr/bin/env python3
"""
Round-1 controller -- multi-corner (full lap).

State machine:
  WAIT_INPUTS -> [WAIT_BUTTON] -> APPROACH -> TURN -> EXIT -> (loop) -> FINISHING -> DONE

  APPROACH   Drive the current straight, centred against the target line (outer
             wall of the current edge, offset inward by o_out). Watch for the
             turn-in point of the current corner.
  TURN       Pose-native arc tracking through the current corner (cross-track to
             the planned circle + heading to the tangent + speed-honest
             feedforward, blended out near the target). theta-based completion.
  EXIT       Stanley path-following onto the exit line for a short settle
             distance, then advance to the next corner (APPROACH) -- or, after a
             full lap, to FINISHING.
  FINISHING  Ramp speed down to a smooth stop on the finish straight.

Corners come from /corner_geometry: 4 outer-box corners + 4 outer walls,
edge-synchronous (walls[i] = edge corners[i]->corners[i+1]), CCW-indexed,
index 0 = largest x. The two walls at corner idx are walls[idx] and
walls[(idx-1)%4]. Direction step through the index: CCW -> +1, CW -> -1.
/corner_geometry is ALWAYS the 4 outer walls (both modes); the EKF's internal
8-wall matching map is separate and not used here.

Command convention: REP 103 (linear.x m/s fwd, angular.z rad/s CCW=left). The
esp_bridge does the calibrated Ackermann inverse and speed control.
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
    return math.atan2(math.sin(a), math.cos(a))


def line_from_points(p0, p1):
    """HNF (nx,ny,d) of the line through p0,p1, unit normal. Sign arbitrary."""
    dx, dy = p1[0] - p0[0], p1[1] - p0[1]
    n = math.hypot(dx, dy)
    if n < 1e-9:
        return None
    nx, ny = -dy / n, dx / n
    d = nx * p0[0] + ny * p0[1]
    return (nx, ny, d)


def line_intersect(l1, l2):
    n1x, n1y, d1 = l1
    n2x, n2y, d2 = l2
    det = n1x * n2y - n1y * n2x
    if abs(det) < 1e-9:
        return None
    x = (d1 * n2y - d2 * n1y) / det
    y = (n1x * d2 - n2x * d1) / det
    return (x, y)


class Round1Controller(Node):

    # param_name -> (attribute_name, converter). Table declares AND loads, so an
    # entry can never be half-present (declared but not read, or vice versa).
    _PARAMS = {
        'nose_offset':   ('nose_offset',   0.14,  float),
        'stop_gap':      ('stop_gap',      0.35,  float),
        'o_in':          ('o_in',          0.50,  float),
        'o_out':         ('o_out',         0.50,  float),
        'turn_radius':   ('R',             0.50,  float),
        'sweep_tol_deg': ('sweep_tol',     2.0,   lambda v: math.radians(float(v))),
        'ff_blend_deg':  ('ff_blend',      35.0,  lambda v: math.radians(float(v))),
        'k_ct':          ('k_ct',          3.0,   float),
        'k_th':          ('k_th',          1.0,   float),
        'k_stanley':     ('k_stanley',     1.5,   float),
        'k_stanley_i':   ('k_stanley_i',   0.4,   float),   # cross-track integral gain
        'k_heading':     ('k_heading',     1.0,   float),   # Stanley heading-term weight (damping)
        'i_ct_limit':    ('i_ct_limit',    math.radians(15.0), lambda v: math.radians(float(v))),  # anti-windup [deg->rad]
        'max_steer_deg': ('max_steer',     25.0,  lambda v: math.radians(float(v))),
        'wheelbase':     ('wheelbase',     0.10,  float),
        'max_yaw_rate':  ('max_yaw_rate',  3.0,   float),
        # speed profile (distance-based)
        'v_drive':       ('v_drive',       0.55,  float),   # straight cruise
        'v_turn':        ('v_turn',        0.35,  float),   # through the arc
        'accel_dist':    ('accel_dist',    0.2,   float),   # ramp v_turn->v_drive after a corner
        'brake_dist':    ('brake_dist',    0.2,   float),   # ramp v_drive->v_turn before T_A
        # lap / finish
        'n_corners':     ('n_corners',     4,     int),
        'finish_front_dist': ('finish_front_dist', 1.5, float),
        'finish_decel':  ('finish_decel',  0.8,   float),   # look-ahead brake decel [m/s^2]
        'finish_lead_time': ('finish_lead_time', 0.15, float),  # reaction lead [s] -> stops on point
        'v_finish_min':  ('v_finish_min',  0.15,  float),   # DRIVABLE crawl, just above deadband
        'finish_tol':    ('finish_tol',    0.04,  float),   # stop tolerance on front_dist
        'debug':         ('debug',         1.0,   lambda v: bool(float(v))),
    }

    def __init__(self):
        super().__init__('round1_controller')

        for name, (attr, default, conv) in self._PARAMS.items():
            self.declare_parameter(name, default)
        # structural (read once)
        self.declare_parameter('require_button', False)
        self.declare_parameter('control_rate', 30.0)
        self.declare_parameter('odom_timeout', 0.5)   # bridge past short EKF gaps

        # per-corner overrides (index = corner_idx). Empty -> use the global scalar
        # (o_in / o_out / turn_radius). Set a 4-element list to override per corner,
        # e.g. o_in_list:=[0.5,0.3,0.5,0.3]. o_out[N] and o_in[N+1] need NOT match
        # (asymmetric racing line is allowed; Stanley drives the transition smoothly).
        from rcl_interfaces.msg import ParameterDescriptor, ParameterType
        arr = ParameterDescriptor(type=ParameterType.PARAMETER_DOUBLE_ARRAY)
        self.declare_parameter('o_in_list', [0.5, 0.5, 0.5, 0.5], arr)
        self.declare_parameter('o_out_list', [0.5, 0.5, 0.5, 0.5], arr)
        self.declare_parameter('turn_radius_list', [0.5, 0.5, 0.5, 0.5], arr)        

        self._load_params()
        self.require_button = bool(self.get_parameter('require_button').value)
        self.control_rate = float(self.get_parameter('control_rate').value)
        self.odom_timeout = float(self.get_parameter('odom_timeout').value)
        self.add_on_set_parameters_callback(self._on_params)

        # --- state ---
        self.state = 'WAIT_INPUTS'
        self.pose = None
        self.v_ist = 0.0
        self.front_wall_x = None
        self.race_direction = None        # 'CW' | 'CCW'
        self.corners = None               # [(x,y)] * 4
        self.walls = None                 # [(nx,ny,d)] * 4
        self.last_odom_time = None
        self.button_pressed = False
        self.v_cmd = 0.0
        self.arc = None
        self.drive_start_xy = (0.0, 0.0)  # for the post-corner accel ramp
        self.ct_integral = 0.0            # Stanley cross-track integrator (reset per straight)
        self.corner_idx = None            # index of the corner currently targeted
        self.corner_count = 0             # corners completed
        self.last_cmd = (0.0, 0.0)        # (v, omega) held during short odom gaps

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
        self.get_logger().info(">>> Round1Controller (multi-corner) bereit. Warte auf Eingaben... <<<")

    def _corner_msg_type(self):
        from robot_msgs.msg import CornerGeometry
        return CornerGeometry

    # ------------------------------------------------------------- params
    def _load_params(self):
        for name, (attr, _default, conv) in self._PARAMS.items():
            setattr(self, attr, conv(self.get_parameter(name).value))
        self._load_lists()

    def _load_lists(self):
        self.o_in_list = [float(v) for v in self.get_parameter('o_in_list').value]
        self.o_out_list = [float(v) for v in self.get_parameter('o_out_list').value]
        self.R_list = [float(v) for v in self.get_parameter('turn_radius_list').value]

    def _on_params(self, params):
        for p in params:
            if p.name in self._PARAMS:
                attr, _default, conv = self._PARAMS[p.name]
                setattr(self, attr, conv(p.value))
            elif p.name == 'o_in_list':
                self.o_in_list = [float(v) for v in p.value]
            elif p.name == 'o_out_list':
                self.o_out_list = [float(v) for v in p.value]
            elif p.name == 'turn_radius_list':
                self.R_list = [float(v) for v in p.value]
        return SetParametersResult(successful=True)

    def corner_o_in(self, idx):
        return self.o_in_list[idx] if idx < len(self.o_in_list) else self.o_in

    def corner_o_out(self, idx):
        return self.o_out_list[idx] if idx < len(self.o_out_list) else self.o_out

    def corner_R(self, idx):
        return self.R_list[idx] if idx < len(self.R_list) else self.R

    # ------------------------------------------------------------- callbacks
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
        corners = [(p.x, p.y) for p in msg.corners]
        walls = [(w.nx, w.ny, w.d) for w in msg.walls]
        if self.corners is None:
            self.get_logger().info(f"/corner_geometry empfangen: {len(corners)} Ecken.")
            self._assert_edge_convention(corners, walls)
        self.corners = corners
        self.walls = walls

    def _assert_edge_convention(self, corners, walls):
        """Verify walls[i] lies on the line through corners[i]->corners[i+1]."""
        ok = True
        for i in range(4):
            p0, p1 = corners[i], corners[(i + 1) % 4]
            nx, ny, d = walls[i]
            e0 = abs(nx * p0[0] + ny * p0[1] - d)
            e1 = abs(nx * p1[0] + ny * p1[1] - d)
            if e0 > 0.02 or e1 > 0.02:
                ok = False
                self.get_logger().error(
                    f"ASSERT: wall[{i}] passt nicht zu corners[{i}]->[{i+1}] "
                    f"(Abw {e0:.3f}/{e1:.3f} m). Kanten-Ecken-Konvention verletzt!")
        if ok:
            self.get_logger().info("Kanten-Ecken-Konvention verifiziert (walls<->corners).")

    def button_cb(self, msg):
        if msg.data:
            self.button_pressed = True

    # ------------------------------------------------------------- helpers
    def publish_stop(self):
        self.pub_cmd.publish(Twist())
        self.v_cmd = 0.0
        self.last_cmd = (0.0, 0.0)

    def publish_cmd(self, v, omega):
        omega = max(-self.max_yaw_rate, min(self.max_yaw_rate, omega))
        cmd = Twist()
        cmd.linear.x = float(v)
        cmd.angular.z = float(omega)
        self.pub_cmd.publish(cmd)
        self.last_cmd = (float(v), float(omega))

    def republish_last(self):
        """Hold the last command during a short odom gap (don't stop mid-manoeuvre)."""
        v, omega = self.last_cmd
        cmd = Twist()
        cmd.linear.x = float(v)
        cmd.angular.z = float(omega)
        self.pub_cmd.publish(cmd)

    def odom_is_stale(self):
        if self.last_odom_time is None:
            return True
        age = (self.get_clock().now() - self.last_odom_time).nanoseconds / 1e9
        return age > self.odom_timeout

    def inputs_ready(self):
        return (self.corners is not None and self.walls is not None
                and self.race_direction in ('CW', 'CCW'))

    def dir_step(self):
        return 1 if self.race_direction == 'CCW' else -1

    def pick_first_corner(self, x, y, theta):
        """Which corner is the robot heading toward.

        A corner ahead has positive projection on the travel direction. But two
        corners can share the same forward projection while one is far to the
        side -- so among the corners ahead, pick the one with the SMALLEST
        lateral offset from the travel line (closest to straight ahead).
        """
        tx, ty = math.cos(theta), math.sin(theta)
        px, py = -ty, tx                                  # left-perpendicular
        best_i, best_lat = None, 1e9
        for i, c in enumerate(self.corners):
            fwd = (c[0] - x) * tx + (c[1] - y) * ty       # along travel (ahead > 0)
            if fwd <= 0.1:
                continue
            lat = abs((c[0] - x) * px + (c[1] - y) * py)  # sideways distance
            if lat < best_lat:
                best_lat = lat
                best_i = i
        return best_i

    # ------------------------------------------------------------- arc planning
    def plan_arc(self, theta):
        """Plan the inscribed arc for the current corner_idx from the box walls."""
        s = float(self.dir_step())
        idx = self.corner_idx

        # the two walls meeting at corner idx
        wall_a = self.walls[(idx - 1) % 4]   # edge ending at corner idx (entry side)
        wall_b = self.walls[idx]             # edge starting at corner idx (exit side)

        # Orient normals inward (toward box centre) so offsetting is consistent.
        cx = sum(c[0] for c in self.corners) / 4.0
        cy = sum(c[1] for c in self.corners) / 4.0
        A = self._inward(wall_a, cx, cy)
        B = self._inward(wall_b, cx, cy)

        # Decide which is the "entry" (roughly parallel to current travel) and
        # which is the "exit" (roughly perpendicular / ahead). Entry wall's normal
        # is perpendicular to travel; exit wall's normal opposes travel.
        tx, ty = math.cos(theta), math.sin(theta)
        if abs(A[0] * tx + A[1] * ty) > abs(B[0] * tx + B[1] * ty):
            A, B = B, A   # ensure A = entry (normal perp to travel), B = exit (normal along -travel)

        o_in = self.corner_o_in(idx)
        o_out = self.corner_o_out(idx)
        R = self.corner_R(idx)

        LA = (A[0], A[1], A[2] + o_in)
        LB = (B[0], B[1], B[2] + o_out)
        P = line_intersect(LA, LB)
        if P is None:
            self.get_logger().error("Eintritts-/Austrittslinie parallel -- kann Bogen nicht planen.")
            return False

        C = (P[0] + R * (A[0] + B[0]), P[1] + R * (A[1] + B[1]))
        T_A = (C[0] - R * A[0], C[1] - R * A[1])
        T_B = (C[0] - R * B[0], C[1] - R * B[1])
        a0 = math.atan2(T_A[1] - C[1], T_A[0] - C[0])

        # travel direction along THIS straight = parallel to entry wall A, sign
        # chosen to match the current heading. Derived from the box geometry, NOT
        # from the current theta -- otherwise a small heading error at plan time
        # accumulates from corner to corner (theta_target drifts over the lap).
        thx, thy = math.cos(theta), math.sin(theta)
        wa1 = (-A[1], A[0])
        travel = wa1 if (wa1[0] * thx + wa1[1] * thy) >= 0 else (A[1], -A[0])

        # exit travel direction = parallel to exit wall B, sign = the turn outcome
        u_B = (-s * (T_B[1] - C[1]) / R, s * (T_B[0] - C[0]) / R)
        # theta_target = heading of the exit straight, absolute from wall B
        wb1 = (-B[1], B[0])
        u_exit = wb1 if (wb1[0] * u_B[0] + wb1[1] * u_B[1]) >= 0 else (B[1], -B[0])
        theta_target = math.atan2(u_exit[1], u_exit[0])
        u_B = u_exit   # keep exit travel consistent with theta_target

        tx, ty = travel
        self.arc = dict(C=C, s=s, R=R, T_A=T_A, T_B=T_B, a0=a0, travel=travel,
                        LA=LA, LB=LB, u_B=u_B, theta_target=theta_target)
        corner = self.corners[idx]
        self.get_logger().info(
            f"DECIDE Ecke {self.corner_count+1}/{self.n_corners} (idx {idx}, {self.race_direction}): "
            f"Eckpunkt=({corner[0]:.2f},{corner[1]:.2f}) o_in={o_in:.2f} o_out={o_out:.2f} R={R:.2f} "
            f"T_A=({T_A[0]:.2f},{T_A[1]:.2f}) T_B=({T_B[0]:.2f},{T_B[1]:.2f}) "
            f"theta_target={math.degrees(theta_target):.1f}.")
        if self.debug:
            self.get_logger().info(
                f"  [GEO] travel=({tx:+.2f},{ty:+.2f}) "
                f"A(entry)=({A[0]:+.2f},{A[1]:+.2f},{A[2]:+.2f}) "
                f"B(exit)=({B[0]:+.2f},{B[1]:+.2f},{B[2]:+.2f}) "
                f"C=({C[0]:.2f},{C[1]:.2f}) "
                f"LA=({LA[0]:+.2f},{LA[1]:+.2f},{LA[2]:+.2f}) "
                f"LB=({LB[0]:+.2f},{LB[1]:+.2f},{LB[2]:+.2f})")
        return True

    @staticmethod
    def _inward(wall, cx, cy):
        """Return wall HNF with normal pointing toward (cx,cy)."""
        nx, ny, d = wall
        # signed distance of centre; if negative, flip so centre is on +normal side
        if nx * cx + ny * cy - d < 0:
            return (-nx, -ny, -d)
        return (nx, ny, d)

    # ------------------------------------------------------------- main loop
    def control_loop(self):
        if self.pose is None:
            return
        x, y, theta = self.pose

        if self.state == 'WAIT_INPUTS':
            if not self.inputs_ready():
                return
            self.state = 'WAIT_BUTTON' if self.require_button else 'DRIVE'
            if self.state == 'DRIVE':
                self._enter_drive(x, y, theta)
            self.get_logger().info("Eingaben da. " +
                                   ("Warte auf Button..." if self.require_button else "Fahre los."))
            return

        if self.state == 'WAIT_BUTTON':
            if self.button_pressed:
                self.state = 'DRIVE'
                self._enter_drive(x, y, theta)
                self.get_logger().info("Start.")
            else:
                self.publish_stop()
            return

        if self.state == 'DONE':
            self.publish_stop()
            return

        # --- odom-stale handling: hold last cmd through short gaps, stop on long ---
        if self.odom_is_stale():
            if self.state in ('TURN', 'DRIVE'):
                self.republish_last()   # bridge past the gap; bridge watchdog is the backstop
            else:
                self.publish_stop()
            return

        if self.state == 'DRIVE':
            self._drive(x, y, theta)
        elif self.state == 'TURN':
            self._turn(x, y, theta)

    # ------------------------------------------------------------- states
    def _enter_drive(self, x, y, theta):
        """Enter DRIVE: ensure a corner is targeted and its arc is planned."""
        self.drive_start_xy = (x, y)
        self.ct_integral = 0.0
        if self.corner_idx is None:
            self.corner_idx = self.pick_first_corner(x, y, theta)
            if self.corner_idx is None:
                self.get_logger().warn("Keine Ecke voraus gefunden -- nehme idx 0.")
                self.corner_idx = 0
        if self.arc is None:
            self.plan_arc(theta)

    def _speed_profile(self, dist_to_TA, dist_since_corner):
        """Distance-based speed: accelerate v_turn->v_drive over accel_dist after a
        corner, cruise v_drive, brake v_drive->v_turn over brake_dist before T_A.
        The lower of the two ramps wins (handles short straights)."""
        # acceleration ramp (grows from v_turn to v_drive over accel_dist)
        if self.accel_dist > 1e-3:
            ra = max(0.0, min(1.0, dist_since_corner / self.accel_dist))
        else:
            ra = 1.0
        v_acc = self.v_turn + ra * (self.v_drive - self.v_turn)
        # braking ramp (falls from v_drive to v_turn as dist_to_TA -> 0)
        if self.brake_dist > 1e-3:
            rb = max(0.0, min(1.0, dist_to_TA / self.brake_dist))
        else:
            rb = 1.0
        v_brk = self.v_turn + rb * (self.v_drive - self.v_turn)
        return min(v_acc, v_brk)

    def _drive(self, x, y, theta):
        """Lane-following on the current straight (Stanley holds the centre line).
        Watches the turn-in point T_A; at the last corner, stops mid-lane at
        finish_front_dist instead of turning in."""
        if self.arc is None:
            if not self.plan_arc(theta):
                return

        tr = self.arc['travel']
        tA = self.arc['T_A']
        # hold the entry line of THIS straight (LA); Stanley keeps us centred
        omega = self._stanley_steer(x, y, theta, self.arc['LA'], tr)

        # signed distance to T_A along travel (positive = T_A still ahead)
        to_TA = (tA[0] - x) * tr[0] + (tA[1] - y) * tr[1]
        px, py = -tr[1], tr[0]
        lateral = abs((x - tA[0]) * px + (y - tA[1]) * py)
        # distance travelled since the corner start (for the accel ramp)
        dsc = math.hypot(x - self.drive_start_xy[0], y - self.drive_start_xy[1])

        # --- final straight: stop mid-lane instead of turning in ---
        if self.corner_count >= self.n_corners:
            # corner_idx already points at the corner ahead on THIS straight
            # (advanced at the end of the last turn); its front wall is the goal.
            fc = self.corners[self.corner_idx]
            front_dist = (fc[0] - x) * tr[0] + (fc[1] - y) * tr[1]
            if self.debug:
                self.get_logger().info(
                    f"[FINISH] pos=({x:+.2f},{y:+.2f}) th={math.degrees(theta):+.1f} "
                    f"front_dist={front_dist:+.2f} (Ziel {self.finish_front_dist:.2f}) om={omega:+.2f}",
                    throttle_duration_sec=0.2)
            # remaining distance to the STOP point, compensated for the reaction
            # lead (a tick + motor/vehicle latency): stop when the robot will be
            # AT the target after it coasts through the lead, not when it first
            # crosses the line -- otherwise it overshoots, worse at higher speed.
            v_now = max(abs(self.v_ist), 0.0)
            lead = v_now * self.finish_lead_time
            remain = front_dist - self.finish_front_dist - lead

            if remain <= self.finish_tol:
                self.state = 'DONE'
                self.publish_stop()
                self.get_logger().info(
                    f"ZIEL ({self.corner_count} Ecken, {front_dist:.2f} m vor Frontwand, "
                    f"v={v_now:.2f}). STOP.")
                return

            # look-ahead braking: v = sqrt(2*a*remain) reaches 0 exactly at the
            # target under constant decel a. Clamp to v_drive above, and to a
            # drivable crawl below so it never starves short of the point.
            v_brake = math.sqrt(2.0 * self.finish_decel * max(remain, 0.0))
            v = min(self.v_drive, v_brake)
            v = max(v, self.v_finish_min)
            self.publish_cmd(v, omega)
            return

        if self.debug:
            self.get_logger().info(
                f"[DRIVE idx{self.corner_idx}] pos=({x:+.2f},{y:+.2f}) th={math.degrees(theta):+.1f} "
                f"to_TA={to_TA:+.2f} lat={lateral:+.2f} om={omega:+.2f}",
                throttle_duration_sec=0.25)

        # --- turn-in when pose crosses T_A ---
        if to_TA <= 0.0:
            if lateral > 0.6:
                self.state = 'DONE'
                self.publish_stop()
                self.get_logger().error(
                    f"NOTSTOP: Einlenkpunkt seitlich verfehlt (lat={lateral:.2f}). "
                    f"Falsche Ecke? idx {self.corner_idx}.")
                return
            self.state = 'TURN'
            self.get_logger().info(
                f"TURN: Einlenken bei ({x:.2f},{y:.2f}, {math.degrees(theta):.1f}).")
            return

        v = self._speed_profile(to_TA, dsc)
        self.publish_cmd(v, omega)

    def _turn(self, x, y, theta):
        C = self.arc['C']; s = self.arc['s']; R = self.arc['R']
        rx, ry = x - C[0], y - C[1]
        dist = math.hypot(rx, ry) or 1e-6
        r_hat = (rx / dist, ry / dist)
        e_ct = dist - R
        t_hat = (-s * r_hat[1], s * r_hat[0])
        e_th = wrap(math.atan2(t_hat[1], t_hat[0]) - theta)
        theta_err = wrap(self.arc['theta_target'] - theta)

        v_meas = abs(self.v_ist) if abs(self.v_ist) > 0.05 else self.v_turn
        blend = max(0.0, min(1.0, abs(theta_err) / self.ff_blend)) if self.ff_blend > 1e-6 else 1.0
        omega = s * (v_meas / R) * blend + s * self.k_ct * e_ct + self.k_th * e_th

        if s * theta_err <= self.sweep_tol:
            # corner done: advance index, plan next arc, back to DRIVE (no stop)
            self.corner_count += 1
            self.get_logger().info(
                f"TURN fertig Ecke {self.corner_count} (theta={math.degrees(theta):.1f}, "
                f"ziel={math.degrees(self.arc['theta_target']):.1f}).")
            self.corner_idx = (self.corner_idx + self.dir_step()) % 4
            self.arc = None
            self.drive_start_xy = (x, y)
            self.ct_integral = 0.0        # fresh cross-track integrator for the new straight
            self.plan_arc(theta)
            self.state = 'DRIVE'
            return
        if self.debug:
            self.get_logger().info(
                f"[TURN idx{self.corner_idx}] pos=({x:+.2f},{y:+.2f}) th={math.degrees(theta):+.1f} "
                f"distC-R={e_ct:+.3f} th_err={math.degrees(theta_err):+.1f} blend={blend:.2f} om={omega:+.2f}",
                throttle_duration_sec=0.2)
        self.publish_cmd(self.v_turn, omega)

    # ------------------------------------------------------------- Stanley
    def _stanley_steer(self, x, y, theta, target_line, u_dir):
        """Stanley path-following -> yaw rate.

        Cross-track is defined explicitly as the robot's offset to the LEFT of
        the travel line (positive = robot is left of the line), independent of
        the arbitrary sign of the HNF normal. A left offset needs a RIGHT
        (negative) steer to return, hence the minus on the cross-track term.

        Convention: positive angular.z / delta = LEFT (confirmed).
        """
        ux, uy = u_dir
        un = math.hypot(ux, uy) or 1e-9
        ux, uy = ux / un, uy / un
        # left-of-travel unit normal
        lx, ly = -uy, ux
        # foot of the line: any point on it. Use the HNF: closest point to origin
        # is (nx*d, ny*d); signed lateral offset of robot from line, measured
        # positive to the LEFT of travel.
        nx, ny, d = target_line
        # signed distance from robot to line along the HNF normal:
        dist_along_n = (nx * x + ny * y) - d
        # component of the HNF normal in the left-of-travel direction:
        n_dot_left = nx * lx + ny * ly
        # robot's left-offset from the line = -(signed distance) projected so that
        # +e_ct means "robot is left of the line"
        e_ct = -dist_along_n * (1.0 if n_dot_left >= 0 else -1.0)

        heading_line = math.atan2(uy, ux)
        e_theta = wrap(heading_line - theta)

        # integral of cross-track over this straight -> closes the residual that a
        # pure Stanley (P-like) leaves standing on short straights. Reset at each
        # corner exit (see _turn) so it never accumulates across the lap.
        self.ct_integral += self.k_stanley_i * e_ct * self.dt
        self.ct_integral = max(-self.i_ct_limit, min(self.i_ct_limit, self.ct_integral))

        v = max(abs(self.v_ist), 0.05)
        delta = self.k_heading * e_theta + math.atan2(self.k_stanley * e_ct, v) + self.ct_integral
        delta = max(-self.max_steer, min(self.max_steer, delta))
        omega = v * math.tan(delta) / self.wheelbase
        return omega


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