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

Am Anfang wird gefragt, welche Geschwindigkeiten kalibriert werden sollen (alle
oder einzelne). Beim Schreiben wird die BESTEHENDE JSON eingelesen und nur die
neu gemessenen Geschwindigkeiten werden ersetzt -- die anderen bleiben erhalten.
Vorher wird eine Sicherung steer_calib.json.bak angelegt.
"""

import json
import math
import shutil
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

OUT_PATH = "/workspace/src/esp_bridge/esp_bridge/steer_calib.json"

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

    def choose_speeds(self):
        """Fragt, welche Geschwindigkeiten gemessen werden. Enter = alle."""
        print("Welche Geschwindigkeiten kalibrieren?")
        for i, v in enumerate(SPEEDS, 1):
            print(f"  {i} = {v:.2f} m/s")
        print("  Enter = alle (mehrere z.B. als 2,3)")
        while True:
            try:
                c = input("Auswahl: ").strip().lower()
            except EOFError:
                return list(SPEEDS)
            if c in ('', 'a', 'alle'):
                return list(SPEEDS)
            try:
                nummern = sorted({int(t) for t in c.replace(' ', '').split(',') if t})
                if nummern and all(1 <= n <= len(SPEEDS) for n in nummern):
                    return [SPEEDS[n - 1] for n in nummern]
            except ValueError:
                pass
            print(f"  Ungueltig -- Nummer(n) von 1 bis {len(SPEEDS)} oder Enter.")

    def run_sequence(self):
        time.sleep(0.5)
        print("\n=== Lenk-Kalibrierung ueber GESCHWINDIGKEITEN (AKKU) ===")
        print(f"L={L_WHEELBASE} m, Speeds={SPEEDS}, {len(SERVO_STEPS)} Servo-Stufen\n")
        auswahl = self.choose_speeds()
        andere = [v for v in SPEEDS if v not in auswahl]
        print(f"\nKalibriert: {', '.join(f'{v:.2f}' for v in auswahl)} m/s"
              + (f" -- {', '.join(f'{v:.2f}' for v in andere)} m/s bleiben in der JSON wie sie sind."
                 if andere else ""))
        print("negativ=rechts, positiv=links | Enter=fahren  r=wiederholen  q=beenden+schreiben\n")
        for v_target in auswahl:
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

    def load_existing(self):
        """Bestehende JSON lesen, um ungemessene Geschwindigkeiten zu behalten."""
        try:
            with open(OUT_PATH) as f:
                return json.load(f)
        except FileNotFoundError:
            return None
        except Exception as e:
            self.get_logger().warn(f"Bestehende JSON nicht lesbar ({e}) -- wird ersetzt.")
            return None

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

        # Mit der bestehenden Datei zusammenfuehren: nur die gemessenen
        # Geschwindigkeiten ersetzen, alle anderen unveraendert uebernehmen.
        alt = self.load_existing()
        nach_v = {}
        if alt:
            for e in alt.get("speeds", []):
                nach_v[round(float(e["v"]), 3)] = e
            if abs(float(alt.get("wheelbase", L_WHEELBASE)) - L_WHEELBASE) > 1e-6:
                self.get_logger().warn(
                    f"Radstand in der bestehenden JSON ({alt.get('wheelbase')}) weicht von "
                    f"L_WHEELBASE={L_WHEELBASE} ab -- die behaltenen Geschwindigkeiten "
                    f"wurden mit dem alten Radstand gerechnet. Besser alle neu kalibrieren.")
        ersetzt = []
        for e in speeds_out:
            k = round(float(e["v"]), 3)
            if k in nach_v:
                ersetzt.append(k)
            nach_v[k] = e
        behalten = [k for k in nach_v if k not in {round(float(e["v"]), 3) for e in speeds_out}]
        speeds_out = [nach_v[k] for k in sorted(nach_v)]

        if alt:
            try:
                shutil.copyfile(OUT_PATH, OUT_PATH + ".bak")
                self.get_logger().info(f"Sicherung: {OUT_PATH}.bak")
            except Exception as e:
                self.get_logger().warn(f"Sicherung fehlgeschlagen: {e}")
        self.get_logger().info(
            "Neu gemessen: %s | aus der alten Datei behalten: %s" % (
                ', '.join(f'{e["v"]:.2f}' for e in speeds_out
                          if round(float(e["v"]), 3) not in behalten) or '-',
                ', '.join(f'{k:.2f}' for k in sorted(behalten)) or '-'))

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