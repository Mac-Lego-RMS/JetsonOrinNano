import heapq
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu, JointState
from nav_msgs.msg import Odometry
import numpy as np

def wrap(a):
    return (a + np.pi) % (2*np.pi) - np.pi

class DeadReckoningEKF:
    # State: [x, y, theta, v, omega, b_g]
    # Konvention: theta CCW von x-Achse, rechtshaendig (REP-103).
    def __init__(self):
        self.x = np.zeros(6)
        self.P = np.diag([1e-3, 1e-3, 1e-3, 1e-2, 1e-2, 1e-4])
        self.q_v, self.q_w, self.q_bg = 1.0, 1.0, 1e-6   # random-walk PSD, Tuning spaeter
        self.r_gyro = 2.83e-7     # Yaw-Varianz (rad/s)² — Stillstand-Test
        self.r_enc  = 9.3e-4      # (m/s)² — eingeschwungene Varianz · r_eff², × 3 Kurvenfaktor
        self.r_eff  = 0.0150      # m — Strecken-Kalibrierung (2,41 m / 10431 Ticks)
        # measurement noise, wall update (from static multi-frame fit, clean walls x3)
        self.r_wall_alpha = 1.0e-5    # rad^2  (~0.19 deg std)
        self.r_wall_d     = 3.6e-6    # m^2    (~1.9 mm std)
        self.last_stamp = None

    def predict(self, dt):
        x, y, th, v, w, bg = self.x
        th_mid = th + 0.5*w*dt
        self.x[0] = x + v*np.cos(th_mid)*dt
        self.x[1] = y + v*np.sin(th_mid)*dt
        self.x[2] = wrap(th + w*dt)
        # v, w, bg: random walk, bleiben

        # ---- DEIN TEIL: F = d(x_neu)/d(x_alt), 6x6 ----
        F = np.eye(6)
        F[0] = [1, 0, -v*np.sin(th_mid)*dt, np.cos(th_mid)*dt, -0.5*v*np.sin(th_mid)*dt**2, 0]
        F[1] = [0, 1, v*np.cos(th_mid)*dt, np.sin(th_mid)*dt, 0.5*v*np.cos(th_mid)*dt**2, 0]
        F[2] = [0, 0, 1, 0, dt, 0]
        # -----------------------------------------------

        Q = np.zeros((6,6))
        Q[3,3] = self.q_v  * dt
        Q[4,4] = self.q_w  * dt
        Q[5,5] = self.q_bg * dt
        self.P = F @ self.P @ F.T + Q

    def _update(self, z, h, H, r):
        S = H @ self.P @ H.T + r
        K = (self.P @ H.T) / S
        self.x = self.x + K * (z - h)
        self.x[2] = wrap(self.x[2])
        self.P = (np.eye(6) - np.outer(K, H)) @ self.P

    def update_gyro(self, omega_meas):        # z = omega + b_g
        H = np.array([0,0,0,0,1,1.0])
        self._update(omega_meas, self.x[4]+self.x[5], H, self.r_gyro)

    def update_encoder(self, v_meas):         # z = v
        H = np.array([0,0,0,1.0,0,0])
        self._update(v_meas, self.x[3], H, self.r_enc)

    def update_zero_motion(self):             # Stillstand: omega=0 -> zieht b_g
        H = np.array([0,0,0,0,1.0,0])
        self._update(0.0, self.x[4], H, 1e-4)

def stamp_to_sec(stamp):
    return stamp.sec + stamp.nanosec * 1e-9

class EKFNode(Node):
    def __init__(self):
        super().__init__('ekf_node')
        self.ekf = DeadReckoningEKF()
        self.ekf.r_gyro = 2.83e-7
        self.ekf.r_enc  = 9.3e-4
        self.r_eff      = 0.0150

        self.last_gyro_z = 0.0
        self.last_v = 0.0
        self.v_thresh = 0.03
        self.w_thresh = 3e-3

        self.queue = []          # Min-Heap, sortiert nach stamp
        self.counter = 0         # Tie-Breaker, gleich stamp -> stabile Ordnung
        self.window = 0.015      # 15 ms Warte-Fenster
        self.last_processed = None

        self.create_subscription(Imu, '/bno055/imu', self.gyro_cb, 50)
        self.create_subscription(JointState, '/esp_serial_bridge/joint_states', self.enc_cb, 50)
        self.pub = self.create_publisher(Odometry, '/ekf/odom', 10)

    def gyro_cb(self, msg):
        t = stamp_to_sec(msg.header.stamp)
        z = msg.angular_velocity.z            # Yaw, rad/s
        self._push(t, 'gyro', z)
        self._drain(t)

    def enc_cb(self, msg):
        t = stamp_to_sec(msg.header.stamp)
        v = msg.velocity[0] * self.r_eff      # rad/s -> m/s
        self._push(t, 'enc', v)
        self._drain(t)

    def _push(self, t, kind, z):
        heapq.heappush(self.queue, (t, self.counter, kind, z))
        self.counter += 1

    def _drain(self, now_t):
        while self.queue:                          # solange was da ist
            t = self.queue[0][0]                   # Stamp der aeltesten (Heap-Kopf, nur angucken)
            if t > now_t - self.window:
                break                              # noch zu jung -> warten, Schleife stoppt
            t, _, kind, z = heapq.heappop(self.queue)   # jetzt wirklich rausziehen

            if self.last_processed is None:
                self.last_processed = t
                if kind == 'gyro':
                    self.ekf.update_gyro(z)
                    self.last_gyro_z = z
                elif kind == 'enc':
                    self.ekf.update_encoder(z)
                    self.last_v = z                    # v merken (schon in m/s)
                    # Stillstand? -> Zero-Motion-Pseudo-Update
                    if abs(z) < self.v_thresh and abs(self.last_gyro_z) < self.w_thresh:
                        self.ekf.update_zero_motion()
                continue

            dt = t - self.last_processed
            if dt > 0:
                self.ekf.predict(dt)
                if kind == 'gyro':
                    self.ekf.update_gyro(z)
                elif kind == 'enc':
                    self.ekf.update_encoder(z)
                self.last_processed = t
                self._publish(t)
            else:
                self.get_logger().warn(
                    f'stale measurement verworfen: dt={dt*1e3:.2f} ms, kind={kind}', 
                    throttle_duration_sec=1.0)

    def _publish(self, t):
        msg = Odometry()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'odom'
        msg.child_frame_id = 'base_link'
        x, y, th = self.ekf.x[0], self.ekf.x[1], self.ekf.x[2]
        msg.pose.pose.position.x = x
        msg.pose.pose.position.y = y
        msg.pose.pose.orientation.z = np.sin(th/2)   # yaw -> quaternion
        msg.pose.pose.orientation.w = np.cos(th/2)
        self.pub.publish(msg)

def main():
    rclpy.init()
    rclpy.spin(EKFNode())

if __name__ == '__main__':
    main()