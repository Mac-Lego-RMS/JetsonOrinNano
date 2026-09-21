#!/usr/bin/env python3
"""
Speed verification tool (position-based, battery bring-up).

Measures REAL ground speed for a commanded linear.x by integrating `position`
(rad, drive axle) over a fixed window -- robust against the velocity-field
jitter and packet loss that make `ros2 topic echo --once` useless here.

Per step: publish linear.x = v_cmd, wait SETTLE_S, then record position at the
window start and end (with their header timestamps) and compute
  v_real = (pos_end - pos_start) * K_POS_M_PER_RAD / (t_end - t_start)

Terminal-triggered, distance-guarded for a ~3 m space. This is the tool to use
ON THE BATTERY -- the PWM->m/s curve depends on supply voltage, so the
competition calibration must be measured here, not on a bench supply.

Controls: [Enter] run shown step | r = redo | q = quit + report
"""

import time
import threading
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import JointState

K_POS_M_PER_RAD = 0.0148   # verify once via a hand-pushed 1.00 m if unsure

# Commanded speeds to verify (m/s, since the bridge now takes real m/s).
V_CMD_STEPS = [0.20, 0.30, 0.40, 0.50]

SETTLE_S  = 1.2     # let speed settle before the measurement window
WINDOW_S  = 2.0     # integration window
RATE_HZ   = 30.0
MAX_DIST  = 2.5     # abort if integrated distance exceeds this (3 m room)


class SpeedVerify(Node):
    def __init__(self):
        super().__init__('speed_verify')
        self.pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.create_subscription(JointState, '/esp_serial_bridge/joint_states',
                                 self.js_cb, 10)
        self.pos = None            # latest (position_rad, t_sec)
        self.results = []          # (v_cmd, v_real, ratio)
        self.dt = 1.0 / RATE_HZ
        self.worker = threading.Thread(target=self.run_sequence, daemon=True)
        self.worker.start()

    def js_cb(self, msg):
        if msg.position:
            t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
            self.pos = (float(msg.position[0]), t)

    def publish_v(self, v):
        cmd = Twist()
        cmd.linear.x = float(v)
        cmd.angular.z = 0.0
        self.pub.publish(cmd)

    def drive_and_measure(self, v_cmd):
        # settle
        t0 = time.monotonic()
        while time.monotonic() - t0 < SETTLE_S:
            self.publish_v(v_cmd)
            time.sleep(self.dt)

        # window start
        start = self.pos
        if start is None:
            self.get_logger().warn("  Keine position empfangen.")
            self._stop()
            return None
        pos0, t0s = start

        tw = time.monotonic()
        aborted = False
        while time.monotonic() - tw < WINDOW_S:
            self.publish_v(v_cmd)
            if self.pos is not None:
                dist = abs(self.pos[0] - pos0) * K_POS_M_PER_RAD
                if dist >= MAX_DIST:
                    aborted = True
                    break
            time.sleep(self.dt)

        pos1, t1s = self.pos
        self._stop()

        dt_meas = t1s - t0s
        if dt_meas <= 0.0:
            self.get_logger().warn("  Zeitfenster ungueltig (Stempel gleich?).")
            return None
        v_real = (pos1 - pos0) * K_POS_M_PER_RAD / dt_meas
        ratio = v_real / v_cmd if v_cmd > 1e-6 else 0.0
        if aborted:
            self.get_logger().warn(f"  Distanz-Stopp (~{MAX_DIST} m).")
        self.get_logger().info(
            f"  v_cmd {v_cmd:.2f} -> v_real {v_real:.3f} m/s "
            f"(Faktor {ratio:.3f}, dt {dt_meas:.2f} s)"
        )
        return v_real

    def _stop(self):
        for _ in range(3):
            self.publish_v(0.0)
            time.sleep(0.02)

    def run_sequence(self):
        time.sleep(0.5)
        print("\n=== Speed-Verifikation (position-basiert, AKKU) ===")
        print("Enter = Stufe fahren | r = wiederholen | q = beenden\n")
        idx = 0
        while idx < len(V_CMD_STEPS):
            v = V_CMD_STEPS[idx]
            try:
                c = input(f"[{idx+1}/{len(V_CMD_STEPS)}] v_cmd {v:.2f} m/s. "
                          f"Roboter frei? Enter/r/q: ").strip().lower()
            except EOFError:
                break
            if c == 'q':
                break
            if c == 'r' and self.results:
                self.results.pop(); idx = max(0, idx - 1); continue
            v_real = self.drive_and_measure(v)
            if v_real is not None:
                self.results.append((v, v_real, v_real / v if v > 1e-6 else 0.0))
            idx += 1
        self.report()
        rclpy.shutdown()

    def report(self):
        print("\n=== Ergebnis ===")
        print("v_cmd   v_real   Faktor")
        for vc, vr, r in self.results:
            print(f"{vc:.2f}    {vr:.3f}    {r:.3f}")
        if len(self.results) >= 2:
            # constant offset vs. constant factor:
            factors = [r for _, _, r in self.results]
            fmean = sum(factors) / len(factors)
            fspread = max(factors) - min(factors)
            print(f"\nFaktor v_real/v_cmd: Mittel {fmean:.3f}, Spanne {fspread:.3f}")
            if fspread < 0.06:
                print("  -> nahezu konstanter Faktor => v_max anpassen:")
                print(f"     v_max_neu = v_max_alt * {fmean:.3f}")
            else:
                print("  -> Faktor variiert => eher pwm_deadband (additiver Offset).")
                print("     Vergleiche (v_real - v_cmd) ueber die Stufen.")
        print()


def main(args=None):
    rclpy.init(args=args)
    node = SpeedVerify()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.publish_v(0.0)
        except Exception:
            pass
        if node.context.ok():
            node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()