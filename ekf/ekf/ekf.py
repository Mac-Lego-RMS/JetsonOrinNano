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

    def _update_multi(self, z, h, H, R):
        """Generic vector measurement update (Kalman correction).

        z, h : (m,) measurement and predicted measurement
        H    : (m, 6) measurement Jacobian
        R    : (m, m) measurement noise covariance
        Handles m-dimensional measurements in one matrix update.
        """
        y = z - h                                    # innovation (m,)
        S = H @ self.P @ H.T + R                      # innovation covariance (m,m)
        K = self.P @ H.T @ np.linalg.inv(S)           # Kalman gain (6,m)
        self.x = self.x + K @ y
        self.x[2] = wrap(self.x[2])
        self.P = (np.eye(6) - K @ H) @ self.P

    def update_wall(self, alpha_meas, d_meas, alpha_map, d_map):
        """2D wall correction from one matched wall (HNF).

        Measurement model h(x) (map wall seen in robot frame):
            alpha_pred = alpha_map - theta
            d_pred     = d_map - (x*cos(alpha_map) + y*sin(alpha_map))
        """
        x, y, theta = self.x[0], self.x[1], self.x[2]

        alpha_pred = wrap(alpha_map - theta)
        d_pred     = d_map - (x * np.cos(alpha_map) + y * np.sin(alpha_map))

        # innovation; alpha component MUST be wrapped (+-pi jump)
        z = np.array([wrap(alpha_meas - alpha_pred),
                    d_meas - d_pred])
        h = np.zeros(2)                               # innovation already computed as z

        # Jacobian H (2x6): rows = [d alpha_pred / dstate], [d d_pred / dstate]
        H = np.zeros((2, 6))
        H[0, 2] = -1.0                                # d(alpha_pred)/d(theta)
        H[1, 0] = -np.cos(alpha_map)                  # d(d_pred)/dx
        H[1, 1] = -np.sin(alpha_map)                  # d(d_pred)/dy

        R = np.diag([self.r_wall_alpha, self.r_wall_d])
        self._update_multi(z, h, H, R)

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