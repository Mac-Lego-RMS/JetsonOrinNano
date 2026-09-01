#!/usr/bin/env python3
"""
Speed calibration tool (terminal-triggered, for a ~3 m space).

Each PWM step is started by hand from the terminal: press Enter, the robot
drives ONE constant-PWM step, measures steady-state ground speed, stops, and
waits. Between steps you reposition the robot by hand (direction doesn't matter,
speed is symmetric). A distance guard aborts a step before it runs out of room.

Bridge relationship (current): duty = clamp(linear.x / max_linear, -1, 1)*1023
with max_linear = 1.0, so publishing linear.x = p commands PWM fraction p.

Ground speed: velocity (rad/s, drive axle) * K_POS_M_PER_RAD.

Controls at each prompt:
  [Enter] run the shown step   |   r = redo last step   |   q = quit + report
"""

import time
import threading
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import JointState

# Ground metres per rad of `position` (verify with your own numbers). ~0.0148.
K_POS_M_PER_RAD = 0.0148

# Steps clustered around the working point (~0.25). Edit freely.
PWM_STEPS = [0.18, 0.24, 0.30, 0.40, 0.50, 0.60, 0.70]

STEP_S    = 2.0      # drive time per step
SETTLE_S  = 0.6      # discard this leading part (accel transient)
RATE_HZ   = 30.0
MAX_DIST  = 2.5      # abort a step after this many metres (3 m room, 0.5 m margin)


class SpeedCalib(Node):
    def __init__(self):
        super().__init__('speed_calib')
        self.pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.create_subscription(JointState, '/esp_serial_bridge/joint_states',
                                 self.js_cb, 10)
        self.vel_rad = 0.0
        self.results = []          # (pwm, duty, v_mps)
        self.dt = 1.0 / RATE_HZ

        self.worker = threading.Thread(target=self.run_sequence, daemon=True)
        self.worker.start()

    def js_cb(self, msg):
        if msg.velocity:
            self.vel_rad = float(msg.velocity[0])

    def publish_pwm(self, pwm):
        cmd = Twist()
        cmd.linear.x = float(pwm)
        cmd.angular.z = 0.0
        self.pub.publish(cmd)

    def drive_step(self, pwm):
        """Drive one constant-PWM step, distance-guarded. Returns v_mps."""
        samples = []
        dist = 0.0
        t0 = time.monotonic()
        aborted = False
        while True:
            t = time.monotonic() - t0
            if t >= STEP_S:
                break
            self.publish_pwm(pwm)
            v_mps_inst = self.vel_rad * K_POS_M_PER_RAD
            dist += abs(v_mps_inst) * self.dt
            if t >= SETTLE_S:
                samples.append(self.vel_rad)
            if dist >= MAX_DIST:
                aborted = True
                break
            time.sleep(self.dt)

        # stop
        for _ in range(3):
            self.publish_pwm(0.0)
            time.sleep(0.02)

        mean_rad = sum(samples) / len(samples) if samples else 0.0
        v_mps = mean_rad * K_POS_M_PER_RAD
        if aborted:
            self.get_logger().warn(
                f"  Distanz-Stopp bei ~{dist:.2f} m (Stufe evtl. zu kurz gemessen)."
            )
        self.get_logger().info(
            f"  PWM {pwm:.2f} (duty {round(pwm*1023):4d}) -> "
            f"{mean_rad:6.2f} rad/s -> {v_mps:5.3f} m/s  (gefahren ~{dist:.2f} m)"
        )
        return v_mps

    def run_sequence(self):
        time.sleep(0.5)
        print("\n=== Speed-Kalibrierung (Terminal) ===")
        print("Enter = Stufe fahren | r = wiederholen | q = beenden\n")
        idx = 0
        while idx < len(PWM_STEPS):
            pwm = PWM_STEPS[idx]
            try:
                cmd = input(f"[Stufe {idx+1}/{len(PWM_STEPS)}] PWM {pwm:.2f} bereit. "
                            f"Roboter frei? Enter/r/q: ").strip().lower()
            except EOFError:
                break
            if cmd == 'q':
                break
            if cmd == 'r' and self.results:
                self.results.pop()          # drop last, redo previous
                idx = max(0, idx - 1)
                continue
            v = self.drive_step(pwm)
            self.results.append((pwm, round(pwm * 1023), v))
            idx += 1

        self.report()
        rclpy.shutdown()

    def report(self):
        print("\n=== Ergebnis ===")
        print("pwm    duty    v_mps")
        for pwm, duty, v in self.results:
            print(f"{pwm:.2f}   {duty:4d}   {v:.3f}")

        moving = [(p, v) for (p, _, v) in self.results if v > 0.02]
        if len(moving) >= 2:
            ps = [p for p, _ in moving]
            vs = [v for _, v in moving]
            n = len(ps)
            mp = sum(ps) / n
            mv = sum(vs) / n
            num = sum((p - mp) * (v - mv) for p, v in moving)
            den = sum((p - mp) ** 2 for p in ps)
            slope = num / den if den > 1e-9 else 0.0
            intercept = mv - slope * mp
            v_full = slope * 1.0 + intercept
            deadband = -intercept / slope if slope > 1e-9 else 0.0
            print(f"\nFit: v = {slope:.3f}*pwm + {intercept:.3f}")
            print(f"  -> max_linear ~ {v_full:.3f} m/s")
            print(f"  -> Deadband ~ pwm {deadband:.3f} (duty {round(deadband*1023)})")
        print()


def main(args=None):
    rclpy.init(args=args)
    node = SpeedCalib()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.publish_pwm(0.0)
        except Exception:
            pass
        if node.context.ok():
            node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()