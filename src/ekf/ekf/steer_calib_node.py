#!/usr/bin/env python3
"""
Steering calibration across MULTIPLE speeds -> writes steer_calib.json.

Measures the servo->steering-angle relationship per speed (side-split), using the
REAL yaw rate (slope of unwrapped EKF heading) AND the REAL forward speed
(from /ekf/odom), so delta is backed out with the MEASURED v:

    delta = atan( L * omega / v_real )

At the end it writes a JSON that the bridge's SteerLUT reads. The JSON stores the
raw (servo, delta_rad) points per side per speed -- the bridge builds the inverse
lookup from them. Set SPEEDS and SERVO_STEPS below to choose how many speeds and
how many interpolation points you want; the bridge adapts with no code change.

A centre point (servo = CENTER_TRIM, delta = 0) is added per side so the curve is
defined around straight-ahead (calibration steps usually skip the tiny angles).

SAFETY: circles at up to max(SPEEDS). Clear a big enough circle. Battery, speed
controller running. Terminal: [Enter] run | r = redo | q = quit + write JSON.
"""

import json
import math
import time
import threading
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry

# ---- what to measure ----
L_WHEELBASE = 0.10
SPEEDS = [0.35, 0.50, 0.75]                      # 1..N speeds
SERVO_STEPS = [-1.00, -0.80, -0.65, -0.50, -0.35,
                0.35,  0.50,  0.65,  0.80, 1.00]  # servo steps (both sides)
CENTER_TRIM = -0.02        # servo at straight-ahead (steer_center_servo)

OUT_PATH = "/workspace/src/wall_follower_robot/wall_follower_robot/steer_calib.json"

SETTLE_S = 2.0
WINDOW_S = 2.5
RATE_HZ  = 30.0
V_TOL    = 0.06


def yaw_from_quaternion(q):
    siny = 2.0 * (q.w * q.z + q.x * q.y)
    cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny, cosy)


class SteerCalib(Node):
    def __init__(self):
        super().__init__('steer_calib_vspeed')
        self.pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.create_subscription(Odometry, '/ekf/odom', self.odom_cb, 10)
        self.theta = None
        self.t_odom = None
        self.v_fwd = 0.0
        self.results = {}          # speed -> list of (servo, omega, v_real, delta_rad)
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

        t0 = ts[0]; xs = [t - t0 for t in ts]
        n = len(xs); mx = sum(xs)/n; my = sum(ths)/n
        num = sum((x-mx)*(y-my) for x, y in zip(xs, ths))
        den = sum((x-mx)**2 for x in xs)
        omega = num/den if den > 1e-9 else 0.0

        v_real = sum(abs(v) for v in vs) / len(vs)
        delta = math.atan(L_WHEELBASE * omega / v_real) if abs(v_real) > 1e-6 else 0.0

        warn = ""
        if abs(v_real - v_target) > V_TOL:
            warn = f"  <<< v real {v_real:.2f} weicht von Soll {v_target:.2f} ab!"
        self.get_logger().info(
            f"  v_soll {v_target:.2f} servo {servo_pct:+.2f} -> "
            f"v_real {v_real:.2f}, omega {omega:+.3f}, delta {math.degrees(delta):+.2f} deg{warn}")
        return (servo_pct, omega, v_real, delta)

    def run_sequence(self):
        time.sleep(0.5)
        print("\n=== Lenk-Kalibrierung ueber GESCHWINDIGKEITEN (AKKU) ===")
        print(f"L={L_WHEELBASE} m, Speeds={SPEEDS}, {len(SERVO_STEPS)} Servo-Stufen")
        print("negativ=rechts, positiv=links | Enter=fahren  r=wiederholen  q=beenden+schreiben\n")
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
                    self.finish(); return
                if c == 'q':
                    self.finish(); return
                if c == 'r' and self.results[v_target]:
                    self.results[v_target].pop(); idx = max(0, idx-1); continue
                res = self.drive_and_measure(v_target, s)
                if res is not None:
                    self.results[v_target].append(res)
                idx += 1
        self.finish()

    def finish(self):
        self.report()
        self.write_json()
        rclpy.shutdown()

    def report(self):
        print("\n=== Ergebnis pro Geschwindigkeit ===")
        for v_target, rows in self.results.items():
            print(f"\n-- v={v_target:.2f} --")
            print("servo  v_real  omega   delta")
            for s, w, vr, d in sorted(rows):
                print(f"{s:+.2f}  {vr:.2f}  {w:+.3f}  {math.degrees(d):+.2f}")

    def write_json(self):
        speeds_out = []
        for v_target in sorted(self.results.keys()):
            rows = self.results[v_target]
            if not rows:
                continue
            left  = sorted([[s, d] for s, w, vr, d in rows if s > 0])
            right = sorted([[s, d] for s, w, vr, d in rows if s < 0])
            # add the straight-ahead centre point per side (servo=trim, delta=0)
            left  = [[CENTER_TRIM, 0.0]] + left
            right = right + [[CENTER_TRIM, 0.0]]
            speeds_out.append({"v": v_target, "left": left, "right": right})

        if not speeds_out:
            self.get_logger().warn("Keine Daten -- JSON nicht geschrieben.")
            return

        data = {
            "wheelbase": L_WHEELBASE,
            "note": ("servo -> steering angle (delta, rad) per speed, side-split. "
                     "Right=servo<0, left=servo>0. Centre point "
                     f"(servo={CENTER_TRIM}, delta=0) applies the straight-ahead trim."),
            "speeds": speeds_out,
        }
        try:
            with open(OUT_PATH, "w") as f:
                json.dump(data, f, indent=2)
            self.get_logger().info(f"JSON geschrieben: {OUT_PATH} "
                                   f"({len(speeds_out)} Geschwindigkeiten)")
        except Exception as e:
            self.get_logger().error(f"JSON schreiben fehlgeschlagen: {e}")


def main(args=None):
    rclpy.init(args=args)
    node = SteerCalib()
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