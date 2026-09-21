"""Fisheye-Projektionsmodell und Kalibrier-IO fuer die liegende 360-Grad-Kamera.

Konventionen
------------
Roboter-/Lidar-Frame (ROS REP-103): X vorne, Y links, Z oben.
LaserScan-Winkel zaehlen CCW ab +X.

Optischer Kamera-Frame: Z = optische Achse (zeigt aus der Linse heraus),
X = im Bild nach rechts, Y = im Bild nach unten.

Die Kamera haengt liegend, die optische Achse zeigt also nominell nach OBEN.
Fuer genau diesen Nominalfall faellt der optische Frame mit dem Roboter-Frame
zusammen (X vorne = rechts im Bild, Y links = unten im Bild, Z oben = optische
Achse) -- die Rotationsmatrix ist dann die Einheitsmatrix. Die drei Winkel
beschreiben die Abweichung davon:

    yaw   Drehung um die optische Achse. Das ist die "Verdrehung der Kamera",
          die rotation_calibration bestimmt.
    pitch/roll  Verkippung der optischen Achse aus der Senkrechten
          (Montage nicht exakt waagerecht).

    P_cam = R @ (P_robot - t),   R = Rx(roll) @ Ry(pitch) @ Rz(yaw)

Radialmodell: equidistant (r = f * theta), optional mit ungeraden
Polynomtermen r = theta * (k0 + k1*theta^2 + k2*theta^4 + ...), falls das
echte Objektiv abweicht.
"""

from dataclasses import dataclass, field, asdict
import math
import os

import numpy as np
import yaml


@dataclass
class FisheyeCalib:
    """Alle Groessen, die Lidar-Punkt -> Pixel beschreiben."""

    # --- Bildkreis (intrinsisch) ---
    cx: float = 678.5           # Mittelpunkt Bildkreis in px
    cy: float = 451.0
    radius_px: float = 452.0    # Radius des Bildkreises in px
    fov_deg: float = 270.0      # voller Oeffnungswinkel des Objektivs
    f_px: float = 0.0           # 0 => aus radius_px/theta_max abgeleitet
    poly_coeffs: list = field(default_factory=list)
    mirror: bool = False        # True, falls das Bild seitenverkehrt ist

    # --- Lage der Kamera im Roboter-Frame (extrinsisch) ---
    yaw_deg: float = 0.0        # Drehung um die optische Achse
    pitch_deg: float = 0.0
    roll_deg: float = 0.0
    cam_x: float = 0.0          # Kameraposition relativ zum Lidar-Ursprung [m]
    cam_y: float = 0.0
    cam_z: float = 0.05         # Kamera sitzt ueber der Lidar-Ebene

    # --- Lidar: verbaute Sektoren ---
    # Paare [von_grad, bis_grad] im Lidar-Frame, die dauerhaft blockiert sind
    # (Kabel, Elektronik, Aufbauten). Dort misst der Scanner nur sich selbst --
    # meist ein paar Zentimeter -- und genau diese Kurz-Returns waeren sonst
    # immer die naechsten Punkte. Laeuft ein Sektor ueber +-180, einfach
    # von > bis schreiben, das wird zyklisch verstanden.
    lidar_blind_sectors_deg: list = field(default_factory=list)

    # --- Metadaten ---
    image_width: int = 1280
    image_height: int = 960
    note: str = ''

    # ------------------------------------------------------------------ #
    @property
    def theta_max_rad(self) -> float:
        return math.radians(self.fov_deg) / 2.0

    @property
    def focal_px(self) -> float:
        """Brennweite in px; aus dem Bildkreis abgeleitet, wenn nicht gesetzt."""
        if self.f_px > 0.0:
            return self.f_px
        return self.radius_px / self.theta_max_rad

    def rotation_matrix(self) -> np.ndarray:
        """R mit P_cam = R @ (P_robot - t)."""
        cr, sr = math.cos(math.radians(self.roll_deg)), math.sin(math.radians(self.roll_deg))
        cp, sp = math.cos(math.radians(self.pitch_deg)), math.sin(math.radians(self.pitch_deg))
        cy_, sy = math.cos(math.radians(self.yaw_deg)), math.sin(math.radians(self.yaw_deg))
        rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], dtype=float)
        ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], dtype=float)
        rz = np.array([[cy_, -sy, 0], [sy, cy_, 0], [0, 0, 1]], dtype=float)
        return rx @ ry @ rz

    def translation(self) -> np.ndarray:
        return np.array([self.cam_x, self.cam_y, self.cam_z], dtype=float)

    # ------------------------------------------------------------------ #
    def to_yaml(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, 'w') as fh:
            yaml.safe_dump(asdict(self), fh, sort_keys=False, default_flow_style=False)

    @classmethod
    def from_dict(cls, data: dict) -> 'FisheyeCalib':
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in (data or {}).items() if k in known})

    @classmethod
    def load(cls, path: str, fallback: str = '') -> 'FisheyeCalib':
        """Laedt path; faellt auf fallback und dann auf die Defaults zurueck."""
        for candidate in (path, fallback):
            if candidate and os.path.exists(candidate):
                with open(candidate, 'r') as fh:
                    return cls.from_dict(yaml.safe_load(fh))
        return cls()


# ---------------------------------------------------------------------- #
def theta_to_radius(calib: FisheyeCalib, theta: np.ndarray) -> np.ndarray:
    """Winkel zur optischen Achse -> Bildradius in px."""
    if calib.poly_coeffs:
        acc = np.zeros_like(theta)
        for i, k in enumerate(calib.poly_coeffs):
            acc = acc + float(k) * theta ** (2 * i)
        return theta * acc
    return calib.focal_px * theta


def radius_to_theta(calib: FisheyeCalib, radius: np.ndarray) -> np.ndarray:
    """Umkehrung von theta_to_radius: Bildradius in px -> Winkel zur Achse."""
    radius = np.asarray(radius, dtype=float)
    if calib.poly_coeffs:
        grid = np.linspace(0.0, calib.theta_max_rad, 2048)
        return np.interp(radius, theta_to_radius(calib, grid), grid)
    return radius / calib.focal_px


def project(calib: FisheyeCalib, pts_robot: np.ndarray):
    """Projiziert 3D-Punkte (N,3) im Roboter-Frame in Pixelkoordinaten.

    Rueckgabe: u, v, theta, phi, in_fov  (jeweils Arrays der Laenge N).
    ``in_fov`` ist False fuer Punkte hinter dem Oeffnungswinkel des Objektivs;
    die Bildgrenzen pruefen die Nodes selbst.
    """
    pts = np.asarray(pts_robot, dtype=float).reshape(-1, 3)
    p_cam = (pts - calib.translation()) @ calib.rotation_matrix().T

    rho = np.hypot(p_cam[:, 0], p_cam[:, 1])
    theta = np.arctan2(rho, p_cam[:, 2])
    phi = np.arctan2(p_cam[:, 1], p_cam[:, 0])
    if calib.mirror:
        phi = -phi

    radius = theta_to_radius(calib, theta)
    u = calib.cx + radius * np.cos(phi)
    v = calib.cy + radius * np.sin(phi)
    return u, v, theta, phi, theta <= calib.theta_max_rad


def scan_to_points(ranges: np.ndarray, angle_min: float, angle_increment: float,
                   z_offset: float = 0.0):
    """LaserScan-Ranges -> (N,3)-Punkte im Roboter-Frame + zugehoerige Winkel.

    ``z_offset`` hebt die Punkte ueber die Lidar-Ebene an, damit nicht der
    Fusspunkt (Matte, Schatten) sondern die Mitte des Klotzes gesampelt wird.
    Skalar oder ein Wert je Punkt -- letzteres braucht der geneigte Ring, dessen
    Hoehe mit der Entfernung mitlaeuft.
    """
    ranges = np.asarray(ranges, dtype=float)
    angles = angle_min + np.arange(ranges.size, dtype=float) * angle_increment
    pts = np.column_stack([
        ranges * np.cos(angles),
        ranges * np.sin(angles),
        np.broadcast_to(np.asarray(z_offset, dtype=float), ranges.shape),
    ])
    return pts, angles


def visible_mask(angles_rad: np.ndarray, sectors_deg: list) -> np.ndarray:
    """True fuer Strahlen ausserhalb aller Blindsektoren.

    ``sectors_deg`` ist eine flache Liste [von1, bis1, von2, bis2, ...] in Grad.
    Ein Sektor mit von > bis laeuft ueber +-180 hinweg (z.B. 135 bis -153).
    """
    angles = np.degrees(np.asarray(angles_rad, dtype=float))
    keep = np.ones(angles.shape, dtype=bool)
    if not sectors_deg:
        return keep

    for i in range(0, len(sectors_deg) - 1, 2):
        lo, hi = float(sectors_deg[i]), float(sectors_deg[i + 1])
        if lo <= hi:
            keep &= ~((angles >= lo) & (angles <= hi))
        else:                       # laeuft ueber +-180
            keep &= ~((angles >= lo) | (angles <= hi))
    return keep


def find_blind_sectors(scans, angle_min: float, angle_increment: float,
                       near_m: float = 0.15, min_valid_frac: float = 0.35,
                       min_width_deg: float = 3.0):
    """Sucht dauerhaft blockierte Sektoren aus mehreren Scans.

    Blockiert heisst: der Median ueber alle Scans liegt unter ``near_m`` (der
    Scanner sieht sich selbst) oder es kommen kaum gueltige Messungen an.
    Rueckgabe: flache Liste [von1, bis1, ...] in Grad, absteigend nach Breite.
    """
    stack = np.asarray(scans, dtype=float)
    finite = np.isfinite(stack) & (stack > 0)
    valid_frac = finite.mean(0)

    # Strahlen ohne eine einzige gueltige Messung vor dem Median ausschliessen,
    # sonst warnt numpy ueber All-NaN-Spalten. Sie gelten ohnehin als blockiert.
    has_any = finite.any(0)
    median = np.zeros(stack.shape[1], dtype=float)
    if has_any.any():
        median[has_any] = np.nanmedian(
            np.where(finite[:, has_any], stack[:, has_any], np.nan), axis=0)

    blocked = ~has_any | (median < near_m) | (valid_frac < min_valid_frac)

    count = blocked.size
    angles = np.degrees(angle_min + np.arange(count) * angle_increment)
    step_deg = abs(np.degrees(angle_increment))
    min_beams = max(1, int(round(min_width_deg / max(step_deg, 1e-9))))

    hits = np.flatnonzero(blocked)
    if hits.size == 0:
        return []

    runs, start, prev = [], hits[0], hits[0]
    for i in hits[1:]:
        if i != prev + 1:
            runs.append((start, prev))
            start = i
        prev = i
    runs.append((start, prev))
    # Laeuft ein Bereich ueber den Index-Umbruch, die beiden Enden verbinden.
    if len(runs) > 1 and runs[0][0] == 0 and runs[-1][1] == count - 1:
        runs = [(runs[-1][0], runs[0][1] + count)] + runs[1:-1]

    runs = [(a, b) for a, b in runs if b - a + 1 >= min_beams]
    runs.sort(key=lambda r: r[1] - r[0], reverse=True)
    sectors = []
    for a, b in runs:
        sectors.extend([round(float(angles[a % count]), 1),
                        round(float(angles[b % count]), 1)])
    return sectors


def detect_image_circle(image_bgr: np.ndarray, threshold: int = 12):
    """Schaetzt (cx, cy, radius) des Fisheye-Bildkreises aus dem dunklen Rand.

    Nutzt die Bounding-Box des hellen Bereichs -- robuster gegen ueberstrahlte
    Lampen als ein umschliessender Kreis. Gibt None zurueck, wenn nichts passt.
    """
    import cv2  # lokal, damit das Modell selbst ohne OpenCV importierbar bleibt

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, int(threshold), 255, cv2.THRESH_BINARY)
    kernel = np.ones((15, 15), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(contour)
    if w < 50 or h < 50:
        return None
    return x + w / 2.0, y + h / 2.0, (w + h) / 4.0
