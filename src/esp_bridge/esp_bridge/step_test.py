#!/usr/bin/env python3
"""Full-throttle step from standstill (test T07).

Drives the vehicle straight from standstill with 100 % PWM for a fixed time,
then cuts the motor (duty 0, the drivetrain coasts).

Why not ~/motor: _velocity_control is the only writer of the motor command
and runs at 50 Hz, so an open-loop command on ~/motor lives for at most one
control period. Instead the velocity controller is driven into saturation:

  - its acceleration ramp (parameter vel_accel) is lifted for the test, so
    the setpoint jumps in one control period instead of ramping;
  - the setpoint (--speed) lies far above v_max, so the feedforward is
    already full duty and the PI term only pushes further into the clamp.

The duty is then 1023 = 100 % from the first control period on, as long as
the measured speed stays below the setpoint. The controller is formally
active but cannot act; the duty in ~/motor_state shows whether that held.
vel_accel is restored afterwards, also on Ctrl+C.

With --race the normal controller is used (vel_accel untouched), for the
comparison with the competition setpoint.

/t07/stage marks the phases in the bag: 1.0 at the step (t = 0), 0.0 when
the motor is cut, -1.0 at rest before the step.

The velocity controller needs /ekf/odom; without it it stops the motor.

    ros2 run esp_bridge step_test                  # 100 %, 1.5 s
    ros2 run esp_bridge step_test --drive 1.0      # shorter track
    ros2 run esp_bridge step_test --race --speed 1.0
"""

from __future__ import annotations

import argparse
import sys
import time

import rclpy
from geometry_msgs.msg import Twist
from rcl_interfaces.msg import Parameter, ParameterType, ParameterValue
from rcl_interfaces.srv import GetParameters, SetParameters
from rclpy.node import Node
from std_msgs.msg import Float32

BRIDGE = "/esp_serial_bridge"


class StepTest(Node):
    def __init__(self, args):
        super().__init__("step_test")
        self.a = args
        self.pub_cmd = self.create_publisher(Twist, "/cmd_vel", 10)
        self.pub_stage = self.create_publisher(Float32, "/t07/stage", 10)
        self.cli_get = self.create_client(GetParameters, f"{BRIDGE}/get_parameters")
        self.cli_set = self.create_client(SetParameters, f"{BRIDGE}/set_parameters")
        self.saved_accel: float | None = None

    # --- bridge parameter ----------------------------------------------
    def _call(self, client, request):
        if not client.wait_for_service(timeout_sec=3.0):
            raise RuntimeError(f"{client.srv_name} not available - is the bridge running?")
        future = client.call_async(request)
        rclpy.spin_until_future_complete(self, future, timeout_sec=3.0)
        if future.result() is None:
            raise RuntimeError(f"{client.srv_name}: no answer")
        return future.result()

    def get_accel(self) -> float:
        res = self._call(self.cli_get, GetParameters.Request(names=["vel_accel"]))
        value = res.values[0]
        if value.type != ParameterType.PARAMETER_DOUBLE:
            raise RuntimeError("bridge has no parameter vel_accel (double)")
        return value.double_value

    def set_accel(self, accel: float) -> None:
        p = Parameter(name="vel_accel", value=ParameterValue(
            type=ParameterType.PARAMETER_DOUBLE, double_value=float(accel)))
        res = self._call(self.cli_set, SetParameters.Request(parameters=[p]))
        if not res.results[0].successful:
            raise RuntimeError(f"vel_accel not set: {res.results[0].reason}")

    # --- building blocks -----------------------------------------------
    def publish(self, v: float, seconds: float, stage: float) -> None:
        """/cmd_vel straight ahead at a fixed rate, otherwise the timeout fires."""
        self.pub_stage.publish(Float32(data=float(stage)))
        msg = Twist()
        msg.linear.x = float(v)
        period = 1.0 / self.a.rate
        end = time.time() + seconds
        while time.time() < end:
            self.pub_cmd.publish(msg)
            rclpy.spin_once(self, timeout_sec=0.0)
            time.sleep(period)

    # --- sequence ------------------------------------------------------
    def run(self) -> None:
        a = self.a
        if not a.race:
            self.saved_accel = self.get_accel()
            self.set_accel(a.accel)
            self.get_logger().info(
                f"vel_accel {self.saved_accel} -> {a.accel} (restored afterwards)")
        mode = "race controller" if a.race else "saturated controller, 100 % PWM"
        self.get_logger().info(f"{mode}: {a.speed} m/s for {a.drive} s, steering 0")

        for k in range(a.countdown, 0, -1):
            self.get_logger().info(f"start in {k}")
            self.publish(0.0, 1.0, -1.0)
        self.get_logger().info("STEP")
        self.publish(a.speed, a.drive, 1.0)
        self.get_logger().info("motor off - coasting")
        self.publish(0.0, a.after, 0.0)

    def finish(self) -> None:
        stop = Twist()
        for _ in range(10):
            self.pub_cmd.publish(stop)
            time.sleep(0.02)
        if self.saved_accel is not None:
            try:
                self.set_accel(self.saved_accel)
                self.get_logger().info(f"vel_accel restored to {self.saved_accel}")
            except RuntimeError as e:
                self.get_logger().error(
                    f"{e} - set it back by hand: ros2 param set {BRIDGE} "
                    f"vel_accel {self.saved_accel}")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--drive", type=float, default=1.5, help="s at full throttle")
    p.add_argument("--speed", type=float, default=5.0,
                   help="setpoint in m/s; far above v_max = saturated (default 5.0)")
    p.add_argument("--race", action="store_true",
                   help="normal controller and ramp, e.g. with --speed 1.0")
    p.add_argument("--accel", type=float, default=1000.0,
                   help="vel_accel during the test in m/s^2 (default 1000 = no ramp)")
    p.add_argument("--countdown", type=int, default=3, help="s of rest before the step")
    p.add_argument("--after", type=float, default=2.0, help="s of logging after the cut")
    p.add_argument("--rate", type=float, default=50.0, help="Hz for /cmd_vel")
    a = p.parse_args(argv if argv is not None else sys.argv[1:])

    # Ctrl+C must not shut the ROS context down before finish() has stopped
    # the motor and restored vel_accel.
    try:
        from rclpy.signals import SignalHandlerOptions
        rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    except (ImportError, TypeError):
        rclpy.init()
    node = StepTest(a)
    try:
        node.run()
    except KeyboardInterrupt:
        node.get_logger().warn("aborted")
    except RuntimeError as e:
        node.get_logger().error(str(e))
    finally:
        node.finish()
        node.destroy_node()
        rclpy.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
