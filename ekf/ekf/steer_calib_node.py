#!/usr/bin/env python3
"""
Steering calibration across MULTIPLE speeds.

Purpose: find out whether the servo->steering-angle relationship is actually
speed-dependent (tyre slip / dynamics at higher speed) or whether one static
curve suffices. For each target speed it runs the servo steps (both directions),
measures the REAL yaw rate (slope of unwrapped EKF heading) AND the REAL forward
speed (from /ekf/odom), and backs out delta with the MEASURED v -- not the
commanded one:

    delta = atan( L * omega / v_real )

If the per-speed curves come out (nearly) identical -> not speed-dependent, the
single static curve you already have is fine (any earlier mismatch was a
measurement error in v). If they differ -> real speed dependence, the bridge
needs a v-interpolated curve.

SAFETY: circles ~0.4-1 m radius at up to 0.75 m/s. Clear a ~1.5 m circle.
Battery, speed controller running. Terminal: [Enter] run | r = redo | q = quit.
"""

import math
import time
import threading
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry

L_WHEELBASE = 0.10
SPEEDS = [0.35, 0.50, 0.75]                 # sicher / medium / riskant
SERVO_STEPS = [-0.80, -0.65, -0.50, -0.35,
                0.35,  0.50,  0.65,  0.80]  # both directions, skip tiny/extreme

SETTLE_S = 2.0
WINDOW_S = 2.5
RATE_HZ  = 30.0
V_TOL    = 0.06     # warn if real speed deviates more than this from target


def yaw_from_quaternion(q):
    siny = 2.0 * (q.w * q.z + q.x * q.y)
    cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny, cosy)


class SteerCalibV(Node):
    def __init__(self):
        super().__init__('steer_calib_vspeed')
        self.pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.create_subscription(Odometry, '/ekf/odom', self.odom_cb, 10)
        self.theta = None
        self.t_odom = None
        self.v_fwd = 0.0
        self.results = {}          # speed -> list of (servo, omega, v_real, delta_deg)
        self.dt = 1.0 / RATE_HZ
        self.worker = threading.Thread(target=self.run_sequence, daemon=True)
        self.worker.start()

    def odom_cb(self, msg):
        self.theta = yaw_from_quaternion(msg.pose.pose.orientation)
        self.t_odom = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        self.v_fwd = float(msg.twist.twist.linear.x)

    def publish(self, v, w):
        cmd = Twist(); cmd.linear.x = float(v); cmd.angular.z = float(w)
        self.pub.publish(cmd)

    def stop(self):
        for _ in range(3):
            self.publish(0.0, 0.0); time.sleep(0.02)

    def drive_and_measure(self, v_target, servo_pct):
        t0 = time.monotonic()
        while time.monotonic() - t0 < SETTLE_S:
            self.publish(v_target, servo_pct); time.sleep(self.dt)

        ts, ths, vs = [], [], []
        theta_unwrap, prev = None, None
        tw = time.monotonic()
        while time.monotonic() - tw < WINDOW_S:
            self.publish(v_target, servo_pct)
            if self.theta is not None and self.t_odom is not None:
                th = self.theta
                if prev is None:
                    theta_unwrap = th
                else:
                    d = th - prev
                    if d > math.pi: d -= 2*math.pi
                    elif d < -math.pi: d += 2*math.pi
                    theta_unwrap += d
                prev = th
                ts.append(self.t_odom); ths.append(theta_unwrap); vs.append(self.v_fwd)
            time.sleep(self.dt)
        self.stop()

        if len(ts) < 5:
            self.get_logger().warn("  Zu wenige Samples."); return None

        # yaw rate = slope of theta over the window
        t0 = ts[0]; xs = [t - t0 for t in ts]
        n = len(xs); mx = sum(xs)/n; my = sum(ths)/n
        num = sum((x-mx)*(y-my) for x, y in zip(xs, ths))
        den = sum((x-mx)**2 for x in xs)
        omega = num/den if den > 1e-9 else 0.0

        v_real = sum(abs(v) for v in vs) / len(vs)     # MEASURED forward speed
        delta = math.atan(L_WHEELBASE * omega / v_real) if abs(v_real) > 1e-6 else 0.0
        delta_deg = math.degrees(delta)

        warn = ""
        if abs(v_real - v_target) > V_TOL:
            warn = f"  <<< v real {v_real:.2f} weicht von Soll {v_target:.2f} ab!"
        self.get_logger().info(
            f"  v_soll {v_target:.2f} servo {servo_pct:+.2f} -> "
            f"v_real {v_real:.2f}, omega {omega:+.3f}, delta {delta_deg:+.2f} deg{warn}")
        return (servo_pct, omega, v_real, delta_deg)

    def run_sequence(self):
        time.sleep(0.5)
        print("\n=== Lenk-Kalibrierung ueber GESCHWINDIGKEITEN (AKKU) ===")
        print(f"L={L_WHEELBASE} m, Speeds={SPEEDS}")
        print("negativ=rechts, positiv=links | Enter=fahren  r=wiederholen  q=beenden\n")
        for v_target in SPEEDS:
            self.results[v_target] = []
            print(f"\n--- Geschwindigkeit {v_target:.2f} m/s ---")
            idx = 0
            while idx < len(SERVO_STEPS):
                s = SERVO_STEPS[idx]
                side = "rechts" if s < 0 else "links"
                try:
                    c = input(f"[v{v_target:.2f} {idx+1}/{len(SERVO_STEPS)}] "
                              f"servo {s:+.2f} ({side}). Kreis frei? Enter/r/q: ").strip().lower()
                except EOFError:
                    return
                if c == 'q':
                    self.report(); rclpy.shutdown(); return
                if c == 'r' and self.results[v_target]:
                    self.results[v_target].pop(); idx = max(0, idx-1); continue
                res = self.drive_and_measure(v_target, s)
                if res is not None:
                    self.results[v_target].append(res)
                idx += 1
        self.report()
        rclpy.shutdown()

    @staticmethod
    def _fit(pts):
        if len(pts) < 2: return None
        ss = [s for s, _ in pts]; ds = [d for _, d in pts]
        n = len(ss); ms = sum(ss)/n; md = sum(ds)/n
        num = sum((s-ms)*(d-md) for s, d in pts); den = sum((s-ms)**2 for s in ss)
        a = num/den if den > 1e-9 else 0.0
        return a, md - a*ms

    def report(self):
        print("\n=== Ergebnis pro Geschwindigkeit ===")
        fits = {}
        for v_target, rows in self.results.items():
            print(f"\n-- v={v_target:.2f} --")
            print("servo  v_real  omega   delta")
            for s, w, vr, d in sorted(rows):
                print(f"{s:+.2f}  {vr:.2f}  {w:+.3f}  {d:+.2f}")
            left  = [(s, math.radians(d)) for s, w, vr, d in rows if s > 0]
            right = [(s, math.radians(d)) for s, w, vr, d in rows if s < 0]
            fl, fr = self._fit(left), self._fit(right)
            fits[v_target] = (fl, fr)
            if fl: print(f"  LINKS : a={fl[0]:.4f} b={fl[1]:+.5f} ({math.degrees(fl[0]):.1f} deg/E)")
            if fr: print(f"  RECHTS: a={fr[0]:.4f} b={fr[1]:+.5f} ({math.degrees(fr[0]):.1f} deg/E)")

        # the key comparison: do the curves change with speed?
        print("\n=== Geschwindigkeits-Abhaengigkeit ===")
        speeds = sorted(fits.keys())
        if len(speeds) >= 2:
            for side, i in (("LINKS", 0), ("RECHTS", 1)):
                aa = [fits[v][i][0] for v in speeds if fits[v][i]]
                if len(aa) >= 2:
                    spread = max(aa) - min(aa)
                    rel = spread / (sum(aa)/len(aa)) * 100 if aa else 0
                    print(f"{side}: Steigung a ueber Speeds = "
                          f"{[f'{x:.3f}' for x in aa]}  (Spanne {spread:.3f}, {rel:.0f}%)")
                    if rel < 8:
                        print(f"  -> nahezu konstant: EINE Kennlinie reicht, NICHT speed-abhaengig.")
                    else:
                        print(f"  -> variiert deutlich: Bridge braucht v-interpolierte Kennlinie.")
        print()


def main(args=None):
    rclpy.init(args=args)
    node = SteerCalibV()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try: node.publish(0.0, 0.0)
        except Exception: pass
        if node.context.ok(): node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()