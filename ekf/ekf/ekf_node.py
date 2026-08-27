#!/usr/bin/env python3
"""
ekf_node: 6-state dead-reckoning EKF with LiDAR wall correction.

Subscribes:
  /bno055/imu                    sensor_msgs/Imu         gyro (yaw rate)
  /esp_serial_bridge/joint_states sensor_msgs/JointState  wheel velocity
  /wall_matches                  robot_msgs/WallMatchArray  matched walls

Publishes:
  /ekf/odom                      nav_msgs/Odometry       pose estimate

Gyro and encoder run through a stamp-ordered queue (measurement-driven predict
+ update, negative-dt discard, zero-motion at standstill). Wall matches are a
correction applied to the current state on arrival (approach B: scan latency
ignored for now; they bypass the time queue since they carry no predict step).
"""
import heapq
import numpy as np

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu, JointState
from nav_msgs.msg import Odometry

from robot_msgs.msg import WallMatchArray

from ekf.ekf import DeadReckoningEKF, wrap

# Gyro sign + scale correction, applied at the source in gyro_cb.
# Negative: the BNO055 yaw axis reads CW as positive; REP-103 wants CCW positive.
# Magnitude 0.9674: scale factor from the 5x360deg calibration.
GYRO_SCALE = -0.9674

def stamp_to_sec(stamp):
    return stamp.sec + stamp.nanosec * 1e-9


class EKFNode(Node):
    def __init__(self):
        super().__init__('ekf_node')
        self.ekf = DeadReckoningEKF()
        self.r_eff = self.ekf.r_eff          # rad/s -> m/s for the encoder

        # zero-motion detection thresholds
        self.last_gyro_z = 0.0
        self.v_thresh = 0.03
        self.w_thresh = 3e-3

        # stamp-ordered queue for gyro + encoder
        self.queue = []
        self.counter = 0
        self.window = 0.015                  # 15 ms wait window
        self.last_processed = None

        self.create_subscription(Imu, '/bno055/imu', self.gyro_cb, 50)
        self.create_subscription(JointState, '/esp_serial_bridge/joint_states',
                                 self.enc_cb, 50)
        self.create_subscription(WallMatchArray, '/wall_matches', self.wall_cb, 10)
        self.pub = self.create_publisher(Odometry, '/ekf/odom', 10)

    # --- gyro / encoder: stamp-ordered queue ------------------------------
    def gyro_cb(self, msg):
        t = stamp_to_sec(msg.header.stamp)
        self._push(t, 'gyro', msg.angular_velocity.z * GYRO_SCALE)
        self._drain(t)

    def enc_cb(self, msg):
        t = stamp_to_sec(msg.header.stamp)
        v = msg.velocity[0] * self.r_eff     # rad/s -> m/s
        self._push(t, 'enc', v)
        self._drain(t)

    def _push(self, t, kind, z):
        heapq.heappush(self.queue, (t, self.counter, kind, z))
        self.counter += 1

    def _apply(self, kind, z):
        if kind == 'gyro':
            self.ekf.update_gyro(z)
            self.last_gyro_z = z
        elif kind == 'enc':
            self.ekf.update_encoder(z)
            if abs(z) < self.v_thresh and abs(self.last_gyro_z) < self.w_thresh:
                self.ekf.update_zero_motion()

    def _drain(self, now_t):
        while self.queue:
            t = self.queue[0][0]
            if t > now_t - self.window:
                break
            t, _, kind, z = heapq.heappop(self.queue)

            if self.last_processed is None:
                self.last_processed = t
                self._apply(kind, z)
                continue

            dt = t - self.last_processed
            if dt > 0:
                self.ekf.predict(dt)
                self._apply(kind, z)
                self.last_processed = t
                self._publish(t)
            else:
                self.get_logger().warn(
                    f'stale measurement dropped: dt={dt*1e3:.2f} ms, kind={kind}',
                    throttle_duration_sec=1.0)

    # --- wall correction: applied to current state (approach B) -----------
    def wall_cb(self, msg):
        for wm in msg.matches:
            self.ekf.update_wall(wm.alpha_meas, wm.d_meas, wm.alpha_map, wm.d_map)
        if msg.matches:
            self._publish(stamp_to_sec(msg.header.stamp))

    # --- output -----------------------------------------------------------
    def _publish(self, t):
        msg = Odometry()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'odom'
        msg.child_frame_id = 'base_link'
        x, y, th = self.ekf.x[0], self.ekf.x[1], self.ekf.x[2]
        msg.pose.pose.position.x = x
        msg.pose.pose.position.y = y
        msg.pose.pose.orientation.z = np.sin(th / 2)     # yaw -> quaternion
        msg.pose.pose.orientation.w = np.cos(th / 2)
        self.pub.publish(msg)


def main():
    rclpy.init()
    rclpy.spin(EKFNode())


if __name__ == '__main__':
    main()