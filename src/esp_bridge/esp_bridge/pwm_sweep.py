#!/usr/bin/env python3
"""Reproducible speed sweep for the vibration and current measurements.

Steps through the setpoints deterministically, so that recordings made before
and after a mechanical change can be compared. Sweeps typed by hand cannot.

Every step has two phases:

  drive  /cmd_vel is published at 20 Hz and the bridge's velocity controller
         holds the setpoint.
  coast  /cmd_vel is NOT published. After cmd_vel_timeout (0.5 s)
         _velocity_control cuts the motor hard to zero. The drivetrain keeps
         turning while the PWM stage is idle.

The coast phase is the point of the measurement, not a by-product: only there
can mechanical vibration be told apart from electrical interference, because
the speed is high and the output stage is off. Comparing both at the same
speed separates the two cleanly.

Why not ~/motor: _velocity_control is the only writer of the motor command
and runs at 50 Hz. An open-loop command on ~/motor lives for at most one
control period.

The current step is published on /pwm_test/stage and so ends up in the bag,
which saves matching timestamps when the data is evaluated.

    ros2 run esp_bridge pwm_sweep
    ros2 run esp_bridge pwm_sweep --max 1.6 --step 0.2
    ros2 run esp_bridge pwm_sweep --steps 0.3,0.6,0.9,1.2,1.5
"""

from __future__ import annotations

import argparse
import csv
import sys
import time

import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node
from std_msgs.msg import Float32


class Sweep(Node):
    def __init__(self, args):
        super().__init__("pwm_sweep")
        self.a = args
        self.pub_cmd = self.create_publisher(Twist, "/cmd_vel", 10)
        self.pub_stage = self.create_publisher(Float32, "/pwm_test/stage", 10)
        self.rows: list[tuple] = []
        self.t0 = time.time()

    # --- building blocks ----------------------------------------------
    def mark(self, v: float, phase: str) -> None:
        now = time.time()
        self.rows.append((f"{now:.3f}", f"{now - self.t0:.3f}", f"{v:.3f}", phase))
        self.pub_stage.publish(Float32(data=float(v)))

    def drive(self, v: float, seconds: float) -> None:
        """Hold a step. /cmd_vel at a fixed rate, otherwise the timeout fires."""
        self.mark(v, "drive")
        msg = Twist()
        msg.linear.x = float(v)
        period = 1.0 / self.a.rate
        end = time.time() + seconds
        while time.time() < end:
            self.pub_cmd.publish(msg)
            rclpy.spin_once(self, timeout_sec=0.0)
            time.sleep(period)

    def coast(self, seconds: float) -> None:
        """Do NOT publish. The bridge's timeout cuts the motor hard."""
        self.mark(0.0, "coast")
        end = time.time() + seconds
        while time.time() < end:
            rclpy.spin_once(self, timeout_sec=0.05)
            time.sleep(0.02)

    def stop_hard(self) -> None:
        msg = Twist()
        for _ in range(10):
            self.pub_cmd.publish(msg)
            time.sleep(0.02)

    # --- sequence -----------------------------------------------------
    def run(self, steps: list[float]) -> None:
        a = self.a
        self.get_logger().info(f"steps [m/s]: {steps}")
        total = 2 * a.settle + len(steps) * (a.drive + a.coast_s)
        self.get_logger().info(f"duration about {total:.0f} s ({total/60:.1f} min)")

        self.get_logger().info("rest - noise floor")
        self.mark(0.0, "rest")
        time.sleep(a.settle)

        for i, v in enumerate(steps, 1):
            t = time.time() - self.t0
            self.get_logger().info(f"[{t:6.1f}s] {i}/{len(steps)}  {v:.2f} m/s - drive")
            self.drive(v, a.drive)
            t = time.time() - self.t0
            self.get_logger().info(f"[{t:6.1f}s] {i}/{len(steps)}  coast")
            self.coast(a.coast_s)

        self.get_logger().info("rest - noise floor")
        self.mark(0.0, "rest")
        time.sleep(a.settle)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--max", type=float, default=1.8, help="highest step in m/s")
    p.add_argument("--step", type=float, default=0.2, help="step size in m/s")
    p.add_argument("--steps", default="", help="explicit list, e.g. 0.3,0.9,1.5")
    p.add_argument("--drive", type=float, default=6.0, help="s per drive phase")
    p.add_argument("--coast-s", type=float, default=6.0, help="s per coast phase")
    p.add_argument("--settle", type=float, default=5.0, help="s of rest at start and end")
    p.add_argument("--rate", type=float, default=20.0, help="Hz for /cmd_vel")
    p.add_argument("--out", default="pwm_sweep.csv")
    a = p.parse_args(argv if argv is not None else sys.argv[1:])

    if a.steps:
        steps = [float(x) for x in a.steps.split(",") if x.strip()]
    else:
        n = int(round(a.max / a.step))
        steps = [round(a.step * k, 3) for k in range(1, n + 1)]

    rclpy.init()
    node = Sweep(a)
    try:
        node.run(steps)
    except KeyboardInterrupt:
        node.get_logger().warn("aborted")
    finally:
        node.stop_hard()
        with open(a.out, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["t_unix", "t_rel", "v_cmd_mps", "phase"])
            w.writerows(node.rows)
        node.get_logger().info(f"{len(node.rows)} marks -> {a.out}")
        node.destroy_node()
        rclpy.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
