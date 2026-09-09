"""Farbklassifikation fuer die WRO-Klotzfarben (HSV, OpenCV-Wertebereiche).

OpenCV-HSV: H in 0..179, S und V in 0..255. Rot liegt um H=0 herum und
braucht deshalb zwei Intervalle.
"""

import cv2
import numpy as np

# name -> Liste von (h_lo, h_hi) plus gemeinsame S/V-Untergrenzen
DEFAULT_RANGES = {
    'rot':     {'hue': [(0, 10), (170, 179)], 's_min': 110, 'v_min': 50},
    'gruen':   {'hue': [(40, 90)],            's_min': 80,  'v_min': 45},
    'magenta': {'hue': [(140, 168)],          's_min': 90,  'v_min': 60},
}

# BGR-Farben fuer Debug-Overlays
LABEL_BGR = {
    'rot': (0, 0, 255),
    'gruen': (0, 220, 0),
    'magenta': (200, 0, 200),
    'schwarz': (60, 60, 60),
    'unbekannt': (180, 180, 180),
}


def classify_hsv(hsv: np.ndarray, ranges: dict = None, black_v_max: int = 45) -> list:
    """Klassifiziert ein (N,3)-HSV-Array zu Labels wie 'rot'/'gruen'/'unbekannt'."""
    ranges = ranges or DEFAULT_RANGES
    hsv = np.asarray(hsv).reshape(-1, 3).astype(np.int16)
    h, s, v = hsv[:, 0], hsv[:, 1], hsv[:, 2]

    labels = np.full(hsv.shape[0], 'unbekannt', dtype=object)
    labels[v <= black_v_max] = 'schwarz'

    for name, spec in ranges.items():
        hit = np.zeros(hsv.shape[0], dtype=bool)
        for lo, hi in spec['hue']:
            hit |= (h >= lo) & (h <= hi)
        hit &= (s >= spec['s_min']) & (v >= spec['v_min'])
        labels[hit] = name
    return labels.tolist()


def sample_colors(image_bgr: np.ndarray, u: np.ndarray, v: np.ndarray, patch: int = 5,
                  center=None, band_px=None, band_count: int = 5):
    """Liest an den Pixeln (u,v) Farbe aus. Gibt (bgr, hsv) als (N,3)-uint8 zurueck.

    Vorher wird das ganze Bild einmal median-gefiltert -- das ist deutlich
    schneller als pro Punkt ein Patch auszuschneiden und faengt Glanzlichter
    und Rauschen genauso weg.

    Sind ``center`` (cx, cy) und ``band_px`` gesetzt, wird nicht ein einzelnes
    Pixel gelesen, sondern ``band_count`` Stuetzstellen entlang der RADIALEN
    Linie durch (u,v) -- und davon der Median genommen. Radial nach aussen heisst
    im Fisheye "nach unten", die Linie liegt also laengs der Pylone. Der Median
    (nicht der Mittelwert) haelt das Ergebnis stabil, wenn ein Ende des Bandes
    ueber die Pylonenkante hinausrutscht.
    """
    if patch > 1:
        smooth = cv2.medianBlur(image_bgr, patch if patch % 2 else patch + 1)
    else:
        smooth = image_bgr

    height, width = image_bgr.shape[:2]
    u = np.asarray(u, dtype=float)
    v = np.asarray(v, dtype=float)

    if center is None or band_px is None or band_count < 2:
        offsets = np.zeros((1, u.size))
        dir_u = dir_v = np.zeros(u.size)
    else:
        cx, cy = center
        dir_u, dir_v = u - cx, v - cy
        radius = np.hypot(dir_u, dir_v)
        safe = np.where(radius > 1e-6, radius, 1.0)
        dir_u, dir_v = dir_u / safe, dir_v / safe
        steps = np.linspace(-1.0, 1.0, int(band_count))
        offsets = steps[:, None] * np.broadcast_to(
            np.asarray(band_px, dtype=float), u.shape)[None, :]

    stack = np.empty((offsets.shape[0], u.size, 3), dtype=np.uint8)
    for k in range(offsets.shape[0]):
        ui = np.clip(np.rint(u + offsets[k] * dir_u).astype(int), 0, width - 1)
        vi = np.clip(np.rint(v + offsets[k] * dir_v).astype(int), 0, height - 1)
        stack[k] = smooth[vi, ui]

    bgr = np.median(stack, axis=0).astype(np.uint8)
    hsv = cv2.cvtColor(bgr.reshape(-1, 1, 3), cv2.COLOR_BGR2HSV).reshape(-1, 3)
    return bgr, hsv


def find_color_blob(image_bgr: np.ndarray, ranges: dict = None, min_area: int = 300,
                    mask_circle=None, only_label: str = '', max_area: int = 0):
    """Sucht den groessten rot/gruen/magenta-Blob im Bild.

    ``mask_circle`` ist optional (cx, cy, radius) und blendet alles ausserhalb
    des Fisheye-Bildkreises aus.

    Rueckgabe: (u, v, label, area, r_innen, r_aussen) oder None. Die beiden
    Radien sind der kleinste und groesste Abstand der Blob-Kontur zum
    Bildkreismittelpunkt. Bei einer stehenden Pylone entspricht ``r_aussen``
    dem Fusspunkt auf der Matte und ``r_innen`` der Oberkante -- radial nach
    aussen heisst im Fisheye ja "nach unten". Daraus kalibriert
    ``rotation_calibration`` die Brennweite.
    """
    ranges = ranges or DEFAULT_RANGES
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)

    roi = None
    if mask_circle is not None:
        cx, cy, radius = mask_circle
        roi = np.zeros(image_bgr.shape[:2], np.uint8)
        cv2.circle(roi, (int(round(cx)), int(round(cy))), int(round(radius)), 255, -1)

    best = None
    for name, spec in ranges.items():
        # Auf eine Farbe festnageln, wenn gewuenscht -- sonst gewinnt der
        # groesste Fleck im Bild, und das ist oft irgendein Gegenstand im Raum
        # statt der Kalibrierpylone.
        if only_label and name != only_label:
            continue
        mask = np.zeros(image_bgr.shape[:2], np.uint8)
        for lo, hi in spec['hue']:
            mask |= cv2.inRange(
                hsv,
                np.array([lo, spec['s_min'], spec['v_min']], np.uint8),
                np.array([hi, 255, 255], np.uint8),
            )
        if roi is not None:
            mask = cv2.bitwise_and(mask, roi)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_area or (max_area > 0 and area > max_area):
                continue
            if best and area <= best[3]:
                continue
            moments = cv2.moments(contour)
            if moments['m00'] <= 0:
                continue

            if mask_circle is not None:
                points = contour.reshape(-1, 2).astype(float)
                radii = np.hypot(points[:, 0] - mask_circle[0], points[:, 1] - mask_circle[1])
                r_innen, r_aussen = float(radii.min()), float(radii.max())
            else:
                r_innen = r_aussen = float('nan')

            best = (moments['m10'] / moments['m00'], moments['m01'] / moments['m00'],
                    name, area, r_innen, r_aussen)
    return best


def ranges_from_params(node, prefix: str = 'color') -> dict:
    """Baut DEFAULT_RANGES aus ROS-Parametern, damit die Schwellen live passen."""
    ranges = {}
    for name, spec in DEFAULT_RANGES.items():
        flat = [bound for pair in spec['hue'] for bound in pair]
        hue = node.declare_parameter(f'{prefix}.{name}.hue', flat).value
        s_min = node.declare_parameter(f'{prefix}.{name}.s_min', spec['s_min']).value
        v_min = node.declare_parameter(f'{prefix}.{name}.v_min', spec['v_min']).value
        ranges[name] = {
            'hue': [(int(hue[i]), int(hue[i + 1])) for i in range(0, len(hue) - 1, 2)],
            's_min': int(s_min),
            'v_min': int(v_min),
        }
    return ranges
