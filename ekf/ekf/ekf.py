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