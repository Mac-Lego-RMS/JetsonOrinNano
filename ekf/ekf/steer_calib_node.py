#!/usr/bin/env python3
"""
Steering calibration tool (yaw-rate based, both directions, for Ackermann / Weg 2).

Measures  servo-percent -> real steering angle delta  for BOTH turn directions,
kept as two separate branches so a servo asymmetry (trim offset, linkage) is
visible instead of averaged away.

Method: drive a steady circle at fixed forward speed, measure REAL yaw rate
omega from the slope of unwrapped EKF heading, back out the steering angle from
the bicycle model:

    delta = atan( L * omega / v )        (L = wheelbase, v = forward speed)

Sign convention (REP 103): +angular.z = CCW = left. So positive servo steps turn
left (omega > 0), negative steps turn right (omega < 0).

Output: table per step + a separate fit for the left and the right branch, plus
the centre offset between them.

SAFETY: robot drives ~0.4-1 m radius circles at V_REF, alternating direction.
Clear a ~1.5 m circle. Battery, speed controller running.
Terminal: [Enter] run step | r = redo | q = quit + report.
"""

import math
import time
import threading
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry

V_REF = 0.35                 # m/s forward speed during calibration
L_WHEELBASE = 0.10           # m, axle-to-axle

# Both signs: negative = right (CW), positive = left (CCW).
SERVO_STEPS = [-1.00, -0.80, -0.65, -0.50, -0.35, -0.20,
                0.20,  0.35,  0.50,  0.65,  0.80, 1.00]

SETTLE_S = 2.0
WINDOW_S = 2.5
RATE_HZ  = 30.0


def yaw_from_quaternion(q):
    siny = 2.0 * (q.w * q.z + q.x * q.y)
    cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny, cosy)


class SteerCalib(Node):
    def __init__(self):
        super().__init__('steer_calib')
        self.pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.create_subscription(Odometry, '/ekf/odom', self.odom_cb, 10)
        self.theta = None
        self.t_odom = None
        self.results = []            # (servo_pct, omega, delta_deg)
        self.dt = 1.0 / RATE_HZ
        self.worker = threading.Thread(target=self.run_sequence, daemon=True)
        self.worker.start()

    def odom_cb(self, msg):
        self.theta = yaw_from_quaternion(msg.pose.pose.orientation)
        self.t_odom = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

    def publish(self, v, w):
        cmd = Twist()
        cmd.linear.x = float(v)
        cmd.angular.z = float(w)
        self.pub.publish(cmd)

    def stop(self):
        for _ in range(3):
            self.publish(0.0, 0.0)
            time.sleep(0.02)

    def drive_and_measure(self, servo_pct):
        t0 = time.monotonic()
        while time.monotonic() - t0 < SETTLE_S:
            self.publish(V_REF, servo_pct)
            time.sleep(self.dt)

        ts, ths = [], []
        theta_unwrap = None
        prev = None
        tw = time.monotonic()
        while time.monotonic() - tw < WINDOW_S:
            self.publish(V_REF, servo_pct)
            if self.theta is not None and self.t_odom is not None:
                th = self.theta
                if prev is None:
                    theta_unwrap = th
                else:
                    d = th - prev
                    if d > math.pi:
                        d -= 2 * math.pi
                    elif d < -math.pi:
                        d += 2 * math.pi
                    theta_unwrap += d
                prev = th
                ts.append(self.t_odom)
                ths.append(theta_unwrap)
            time.sleep(self.dt)

        self.stop()

        if len(ts) < 5:
            self.get_logger().warn("  Zu wenige Samples.")
            return None

        t0 = ts[0]
        xs = [t - t0 for t in ts]
        n = len(xs)
        mx = sum(xs) / n
        my = sum(ths) / n
        num = sum((x - mx) * (y - my) for x, y in zip(xs, ths))
        den = sum((x - mx) ** 2 for x in xs)
        omega = num / den if den > 1e-9 else 0.0

        delta = math.atan(L_WHEELBASE * omega / V_REF) if abs(V_REF) > 1e-6 else 0.0
        delta_deg = math.degrees(delta)
        radius = abs(V_REF / omega) if abs(omega) > 1e-6 else float('inf')

        self.get_logger().info(
            f"  servo {servo_pct:+.2f} -> omega {omega:+.3f} rad/s, "
            f"delta {delta_deg:+.2f} deg, |R| {radius:.2f} m"
        )
        return (servo_pct, omega, delta_deg)

    def run_sequence(self):
        time.sleep(0.5)
        print("\n=== Lenk-Kalibrierung BEIDE RICHTUNGEN (AKKU) ===")
        print(f"v_ref = {V_REF} m/s, L = {L_WHEELBASE} m")
        print("negativ = rechts (CW), positiv = links (CCW)")
        print("Enter = Stufe fahren | r = wiederholen | q = beenden\n")
        idx = 0
        while idx < len(SERVO_STEPS):
            s = SERVO_STEPS[idx]
            side = "rechts" if s < 0 else "links"
            try:
                c = input(f"[{idx+1}/{len(SERVO_STEPS)}] servo {s:+.2f} ({side}). "
                          f"Kreis frei? Enter/r/q: ").strip().lower()
            except EOFError:
                break
            if c == 'q':
                break
            if c == 'r' and self.results:
                self.results.pop(); idx = max(0, idx - 1); continue
            res = self.drive_and_measure(s)
            if res is not None:
                self.results.append(res)
            idx += 1
        self.report()
        rclpy.shutdown()

    @staticmethod
    def _fit(pts):
        """pts = [(servo, delta_rad)]. Returns (a, b) for delta = a*servo + b."""
        if len(pts) < 2:
            return None
        ss = [s for s, _ in pts]
        ds = [d for _, d in pts]
        n = len(ss)
        ms = sum(ss) / n
        md = sum(ds) / n
        num = sum((s - ms) * (d - md) for s, d in pts)
        den = sum((s - ms) ** 2 for s in ss)
        a = num / den if den > 1e-9 else 0.0
        b = md - a * ms
        return a, b

    def report(self):
        print("\n=== Ergebnis ===")
        print("servo   omega[rad/s]   delta[deg]")
        for s, w, d in sorted(self.results):
            print(f"{s:+.2f}    {w:+.3f}        {d:+.2f}")

        left  = [(s, math.radians(d)) for s, w, d in self.results if s > 0]
        right = [(s, math.radians(d)) for s, w, d in self.results if s < 0]

        fl = self._fit(left)
        fr = self._fit(right)

        print()
        if fl:
            a, b = fl
            print(f"LINKS  (CCW): delta = {a:.4f}*servo + {b:.5f} rad "
                  f"({math.degrees(a):.2f} deg/Einheit)")
        if fr:
            a, b = fr
            print(f"RECHTS (CW):  delta = {a:.4f}*servo + {b:.5f} rad "
                  f"({math.degrees(a):.2f} deg/Einheit)")

        if fl and fr:
            al, bl = fl
            ar, br = fr
            # steering angle at servo = 0 from each branch -> centre offset
            print(f"\nMittenversatz: links b={math.degrees(bl):+.2f} deg, "
                  f"rechts b={math.degrees(br):+.2f} deg")
            if abs(math.degrees(bl - br)) > 1.0:
                print("  -> merkliche Asymmetrie: Bridge braucht getrennte "
                      "links/rechts-Kennlinie oder Trim.")
            else:
                print("  -> Asymmetrie klein: eine Kennlinie reicht evtl.")
            avg_a = (al + ar) / 2.0
            avg_b = (bl + br) / 2.0
            print(f"\nBridge-Umkehrung (Ackermann):  servo = (delta - b)/a")
            print(f"  delta = atan(L*omega/v), L={L_WHEELBASE}")
            print(f"  gemittelt: a={avg_a:.4f}, b={avg_b:.5f} "
                  f"(nur falls Asymmetrie klein)")
        print()


def main(args=None):
    rclpy.init(args=args)
    node = SteerCalib()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.publish(0.0, 0.0)
        except Exception:
            pass
        if node.context.ok():
            node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()