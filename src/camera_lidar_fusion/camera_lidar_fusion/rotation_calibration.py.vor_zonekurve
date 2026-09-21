#!/usr/bin/env python3
"""Kalibriert die Verdrehung der liegenden 360-Grad-Kamera gegen das Lidar.

Ablauf
------
1. Bildkreis bestimmen (einmalig, rein aus dem Bild):
       ros2 topic pub --once /camera_lidar/calib_cmd std_msgs/msg/String "data: circle"

2. Verdrehung messen. Einen einzelnen roten/gruenen Klotz vor den Roboter
   stellen -- sonst nichts im Lidar-Nahbereich -- und je Position samplen:
       ros2 topic pub --once /camera_lidar/calib_cmd std_msgs/msg/String "data: sample"
   Klotz umstellen (moeglichst rundum verteilt, >= 3 Positionen) und wiederholen.
   Alternativ ``auto`` einschalten, dann sammelt die Node selbststaendig, sobald
   sich der Klotz weit genug bewegt hat.

3. Loesen und speichern:
       ros2 topic pub --once /camera_lidar/calib_cmd std_msgs/msg/String "data: solve"
       ros2 topic pub --once /camera_lidar/calib_cmd std_msgs/msg/String "data: save"

4. Optional die Kamerahoehe gegenpruefen (``height``): nachmessen mit dem Lineal
   ist genauer, aber so siehst du, ob dein Wert zu den Bildern passt.

Weitere Kommandos: ``list``, ``clear``, ``verify``, ``reload``, ``auto``.

Zum Feintunen von Hand laufen alle Groessen zusaetzlich als ROS-Parameter, die
live wirken -- das Debug-Bild aktualisiert sich sofort:
    ros2 param set /camera_rotation_calibration yaw_deg 12.5

Nur ``yaw_deg`` (Drehung um die optische Achse) laesst sich sauber allein aus
Peilungen bestimmen; dafuer gibt es eine geschlossene Loesung. Sollen auch
pitch/roll/cx/cy mit, uebernimmt scipy die Ausgleichsrechnung -- dann aber
deutlich mehr und besser verteilte Samples aufnehmen.

Start:
    ros2 run camera_lidar_fusion rotation_calibration
"""

import copy
import math
import os

import numpy as np
import rclpy
from cv_bridge import CvBridge
from rcl_interfaces.msg import SetParametersResult
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image, LaserScan
from std_msgs.msg import String

import cv2

from camera_lidar_fusion import colors
from camera_lidar_fusion.fisheye_model import (
    FisheyeCalib, detect_image_circle, find_blind_sectors, project, radius_to_theta,
    scan_to_points, visible_mask,
)

# Parameter, die live auf self.calib durchschlagen
LIVE_FIELDS = ('yaw_deg', 'pitch_deg', 'roll_deg', 'cx', 'cy', 'radius_px',
               'fov_deg', 'f_px', 'cam_x', 'cam_y', 'cam_z', 'mirror')


class RotationCalibration(Node):

    def __init__(self):
        super().__init__('camera_rotation_calibration')

        self.declare_parameter('scan_topic', '/scan')
        self.declare_parameter('image_topic', '/video_source/raw')
        self.declare_parameter('calib_file', '/workspace/config/fisheye_calib.yaml')
        self.declare_parameter('fit_params', ['yaw_deg'])
        self.declare_parameter('target_range_max_m', 1.5)
        self.declare_parameter('target_range_min_m', 0.10)
        self.declare_parameter('cluster_gap_m', 0.06)
        self.declare_parameter('cluster_min_points', 3)
        self.declare_parameter('cluster_max_width_deg', 60.0)
        self.declare_parameter('blob_min_area_px', 300)
        # Obergrenze gegen grosse Gegenstaende im Raum. 0 = aus.
        self.declare_parameter('blob_max_area_px', 0)
        # Auf die Farbe der Kalibrierpylone festnageln: 'rot', 'gruen' oder
        # 'magenta'. Leer = groesster Fleck egal welcher Farbe -- das greift in
        # einem moeblierten Raum fast immer daneben.
        self.declare_parameter('target_label', '')
        # Ab wieviel Grad Abweichung vom robusten Mittel ein Sample als falsch
        # zugeordnet gilt und aus solve/radial fliegt.
        self.declare_parameter('outlier_reject_deg', 20.0)
        # Hoehe des beobachteten Farb-Blob-Schwerpunkts ueber der Lidar-Ebene.
        # Der Lidar trifft den Klotz bei z=0; der Schwerpunkt der farbigen
        # Flaeche liegt etwas darueber. Einmal grob abschaetzen -- daraus loest
        # "height" dann cam_z.
        self.declare_parameter('target_height_m', 0.05)
        # Hoehe der Kalibrierpylone. Braucht "radial", um aus dem Fusspunkt des
        # Blobs die Brennweite und die Objektivhoehe ueber der Matte zu loesen.
        self.declare_parameter('pylon_height_m', 0.10)
        # Direktregler fuer den Horizontring: > 0 setzt f_px = radius/(pi/2).
        # Damit laesst sich der Ring live verschieben, ohne ueber die FOV zu
        # gehen. 0 = aus, dann gilt f_px bzw. radius_px/(fov/2).
        self.declare_parameter('horizon_radius_px', 0.0)
        # Wieviel naeher als der Referenzscan ein Strahl messen muss, damit er
        # als "da steht jetzt was Neues" gilt.
        self.declare_parameter('foreground_margin_m', 0.08)
        self.declare_parameter('background_scan_count', 25)
        self.declare_parameter('blind_scan_count', 25)
        self.declare_parameter('blind_near_m', 0.15)
        self.declare_parameter('blind_min_width_deg', 3.0)
        self.declare_parameter('auto_sample', False)
        self.declare_parameter('auto_min_bearing_step_deg', 20.0)
        self.declare_parameter('debug', True)
        self.declare_parameter('debug_rate_hz', 5.0)

        self.calib_path = self.get_parameter('calib_file').value
        self.calib = FisheyeCalib.load(self.calib_path, _packaged_default())
        for name in LIVE_FIELDS:
            self.declare_parameter(name, getattr(self.calib, name))
        self.add_on_set_parameters_callback(self._on_param_set)

        self.ranges = colors.ranges_from_params(self)
        self.bridge = CvBridge()
        self.latest_image = None
        self.latest_scan = None
        self.samples = []           # dicts: bearing_rad, range_m, u_obs, v_obs, label
        self.last_debug_stamp = 0.0
        self.blind_buffer = []      # [(ranges, (angle_min, angle_increment)), ...]
        self.blind_collecting = False
        # Referenzscan der leeren Umgebung. Alles, was naeher misst als diese
        # Referenz, ist neu -- also die Pylone. Damit fallen Kabel, Elektronik,
        # Tischkanten und Waende automatisch weg, ohne sie einzeln auszumessen.
        self.background = None
        self.background_buffer = []
        self.background_collecting = False

        self.commands = {
            'circle': self.cmd_circle, 'sample': self.cmd_sample, 'solve': self.cmd_solve,
            'save': self.cmd_save, 'reload': self.cmd_reload, 'clear': self.cmd_clear,
            'list': self.cmd_list, 'verify': self.cmd_verify, 'auto': self.cmd_auto,
            'height': self.cmd_height, 'radial': self.cmd_radial, 'ring': self.cmd_ring,
            'blind': self.cmd_blind, 'background': self.cmd_background,
        }

        self.create_subscription(Image, self.get_parameter('image_topic').value,
                                 self.on_image, qos_profile_sensor_data)
        self.create_subscription(LaserScan, self.get_parameter('scan_topic').value,
                                 self.on_scan, qos_profile_sensor_data)
        self.create_subscription(String, '/camera_lidar/calib_cmd', self.on_command, 10)
        self.pub_debug = self.create_publisher(Image, '/camera_lidar/calib_debug', 2)
        self.pub_status = self.create_publisher(String, '/camera_lidar/calib_status', 10)

        self.get_logger().info(
            'Kalibrier-Node bereit. Kommandos auf /camera_lidar/calib_cmd: '
            + ', '.join(sorted(self.commands))
            + f'\n  Aktuell: yaw={self.calib.yaw_deg:.2f} pitch={self.calib.pitch_deg:.2f} '
              f'roll={self.calib.roll_deg:.2f} Grad, Bildkreis '
              f'({self.calib.cx:.1f}, {self.calib.cy:.1f}) r={self.calib.radius_px:.1f}'
        )

    # ------------------------------------------------------------------ #
    def _on_param_set(self, params):
        for param in params:
            if param.name in LIVE_FIELDS:
                setattr(self.calib, param.name, param.value)
            elif param.name == 'horizon_radius_px' and param.value > 0.0:
                # Direktregler: Ringradius vorgeben, Brennweite ergibt sich.
                self.calib.f_px = float(param.value) / (math.pi / 2.0)
        return SetParametersResult(successful=True)

    def _push_to_params(self):
        """Uebernimmt geloeste Werte zurueck in die ROS-Parameter."""
        from rclpy.parameter import Parameter
        updates = [Parameter(name, value=getattr(self.calib, name)) for name in LIVE_FIELDS]
        self.set_parameters(updates)

    def _status(self, text: str, warn: bool = False):
        # warn und info MUESSEN in getrennten Zeilen stehen: rclpy merkt sich den
        # Log-Level pro Aufrufstelle und wirft "Logger severity cannot be changed
        # between calls", wenn beide ueber denselben Ausdruck laufen.
        if warn:
            self.get_logger().warn(text)
        else:
            self.get_logger().info(text)
        self.pub_status.publish(String(data=text))

    # ------------------------------------------------------------------ #
    def on_image(self, msg: Image):
        try:
            self.latest_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        except Exception as exc:  # noqa: BLE001
            self.get_logger().warn(f'Bild nicht dekodierbar: {exc}')
            return
        self.calib.image_height, self.calib.image_width = self.latest_image.shape[:2]
        self._publish_debug()

    def on_scan(self, msg: LaserScan):
        self.latest_scan = msg
        if self.background_collecting:
            self.background_buffer.append(np.asarray(msg.ranges, dtype=float))
            needed = self.get_parameter('background_scan_count').value
            if len(self.background_buffer) >= needed:
                self.background_collecting = False
                self._status(f'{len(self.background_buffer)} Referenzscans gesammelt -- '
                             'jetzt "background" nochmal senden.')
        if self.blind_collecting:
            self.blind_buffer.append(
                (np.asarray(msg.ranges, dtype=float), (msg.angle_min, msg.angle_increment)))
            needed = self.get_parameter('blind_scan_count').value
            if len(self.blind_buffer) >= needed:
                self.blind_collecting = False
                self._status(f'{len(self.blind_buffer)} Scans gesammelt -- jetzt "blind" '
                             'nochmal senden, um auszuwerten.')
        if self.get_parameter('auto_sample').value:
            self._try_auto_sample()

    def on_command(self, msg: String):
        command = msg.data.strip().lower()
        handler = self.commands.get(command)
        if handler is None:
            self._status(f'Unbekanntes Kommando "{msg.data}". Bekannt: '
                         + ', '.join(sorted(self.commands)), warn=True)
            return
        handler()

    # ------------------------------------------------------------------ #
    # Messung: naechstliegendes Lidar-Cluster + groesster Farb-Blob
    # ------------------------------------------------------------------ #
    def find_target_cluster(self):
        """Sucht das naechste isolierte Lidar-Cluster (= der Kalibrierklotz)."""
        msg = self.latest_scan
        if msg is None:
            return None

        ranges = np.asarray(msg.ranges, dtype=float)
        r_min = max(float(msg.range_min), self.get_parameter('target_range_min_m').value)
        r_max = min(float(msg.range_max), self.get_parameter('target_range_max_m').value)
        valid = np.isfinite(ranges) & (ranges >= r_min) & (ranges <= r_max)
        _, angles = scan_to_points(ranges, msg.angle_min, msg.angle_increment)
        # Verbaute Sektoren raus -- sonst ist der naechste Punkt immer das
        # eigene Kabel bzw. die Elektronik und nie die Pylone.
        valid &= visible_mask(angles, self.calib.lidar_blind_sectors_deg)

        # Referenzscan: nur was naeher misst als die leere Umgebung ist ein Ziel.
        if self.background is not None and self.background.size == ranges.size:
            margin = self.get_parameter('foreground_margin_m').value
            valid &= ranges < (self.background - margin)
        else:
            self.get_logger().warn(
                'Kein Referenzscan -- der naechste Punkt kann auch der eigene Aufbau '
                'sein. Erst "background" senden (ohne Pylone).',
                throttle_duration_sec=10.0)

        if not valid.any():
            return None

        seed = int(np.argmin(np.where(valid, ranges, np.inf)))

        # Vom naechsten Punkt aus nach beiden Seiten wachsen, solange die
        # Entfernung stetig bleibt -- der Scan ist zyklisch.
        gap = self.get_parameter('cluster_gap_m').value
        count = ranges.size
        members = [seed]
        for step in (1, -1):
            cursor = seed
            while True:
                nxt = (cursor + step) % count
                if nxt == seed or not valid[nxt] or abs(ranges[nxt] - ranges[cursor]) > gap:
                    break
                members.append(nxt)
                cursor = nxt
        if len(members) < self.get_parameter('cluster_min_points').value:
            return None

        # Ein Klotz ist schmal. Waechst das Cluster ueber eine Wand oder einen
        # durchgehenden Boden weiter, ist der Schwerpunkt wertlos -- verwerfen.
        max_width = math.radians(self.get_parameter('cluster_max_width_deg').value)
        if len(members) * abs(msg.angle_increment) > max_width:
            # Gedrosselt: das laeuft auch im Debug-Bild-Pfad mehrmals pro Sekunde.
            self.get_logger().warn(
                f'Cluster ist {math.degrees(len(members) * abs(msg.angle_increment)):.0f} Grad '
                f'breit -- das ist kein Klotz, sondern eine Flaeche. Freier stellen oder '
                f'cluster_max_width_deg anheben.', throttle_duration_sec=3.0)
            return None

        # Hinweis, wenn noch etwas fast genauso nah steht: dann koennte der
        # Farb-Blob zu einem anderen Objekt gehoeren als das Lidar-Cluster.
        # Nur warnen, nicht ablehnen -- in einem echten Raum steht praktisch
        # immer irgendwo etwas in aehnlicher Entfernung.
        member_set = set(members)
        others = [i for i in np.flatnonzero(valid) if i not in member_set]
        if others and ranges[others].min() < ranges[seed] + 0.15:
            self.get_logger().warn(
                f'Noch ein Objekt bei {ranges[others].min():.2f} m, Klotz bei '
                f'{ranges[seed]:.2f} m -- pruefe im Debug-Bild, ob das orange X auf '
                f'dem Farbring liegt.', throttle_duration_sec=5.0)

        # Zyklisch mitteln, damit ein Cluster ueber den Nulldurchgang nicht kippt.
        member_angles = angles[members]
        bearing = math.atan2(np.sin(member_angles).mean(), np.cos(member_angles).mean())
        return bearing, float(ranges[members].mean()), len(members)

    def find_target_blob(self):
        if self.latest_image is None:
            return None
        return colors.find_color_blob(
            self.latest_image, self.ranges,
            min_area=self.get_parameter('blob_min_area_px').value,
            mask_circle=(self.calib.cx, self.calib.cy, self.calib.radius_px),
            only_label=self.get_parameter('target_label').value,
            max_area=self.get_parameter('blob_max_area_px').value)

    # ------------------------------------------------------------------ #
    # Kommandos
    # ------------------------------------------------------------------ #
    def cmd_circle(self):
        if self.latest_image is None:
            self._status('Noch kein Kamerabild da.', warn=True)
            return
        found = detect_image_circle(self.latest_image)
        if found is None:
            self._status('Bildkreis nicht erkannt -- ist das Bild ueberall gleich hell?',
                         warn=True)
            return
        self.calib.cx, self.calib.cy, self.calib.radius_px = found
        self.calib.image_height, self.calib.image_width = self.latest_image.shape[:2]
        self._push_to_params()
        self._status(f'Bildkreis: cx={found[0]:.1f} cy={found[1]:.1f} r={found[2]:.1f} px '
                     f'-> Brennweite {self.calib.focal_px:.1f} px/rad. '
                     'Mit "save" festschreiben.')

    def cmd_sample(self):
        cluster = self.find_target_cluster()
        if cluster is None:
            self._status('Kein sauberes Lidar-Cluster gefunden.', warn=True)
            return
        blob = self.find_target_blob()
        if blob is None:
            self._status('Kein Farb-Blob im Bild gefunden.', warn=True)
            return

        bearing, distance, points = cluster
        u_obs, v_obs, label, area, r_innen, r_aussen = blob

        # Landet das Lidar zweimal fast auf demselben Punkt, obwohl die Pylone
        # umgestellt wurde, greift es in Wahrheit etwas Festes ab.
        for i, alt in enumerate(self.samples, 1):
            if (abs(math.degrees(_wrap(bearing - alt['bearing_rad']))) < 2.0
                    and abs(distance - alt['range_m']) < 0.03):
                self._status(
                    f'Dieses Ziel ist praktisch identisch mit Sample {i} '
                    f'({math.degrees(alt["bearing_rad"]):+.1f} Grad / {alt["range_m"]:.2f} m). '
                    'Wenn du die Pylone umgestellt hast, misst das Lidar hier etwas '
                    'Festes -- "background" aufnehmen und "clear".', warn=True)
                break

        # Dasselbe fuer die Kamera: bleibt der Blob stehen, waehrend die Peilung
        # wandert, ist es nicht die Pylone sondern ein Gegenstand im Raum.
        for i, alt in enumerate(self.samples, 1):
            weg_px = math.hypot(u_obs - alt['u_obs'], v_obs - alt['v_obs'])
            peilung_grad = abs(math.degrees(_wrap(bearing - alt['bearing_rad'])))
            if weg_px < 15.0 and peilung_grad > 10.0:
                self._status(
                    f'Der Farb-Blob steht fast unveraendert bei ({u_obs:.0f}, {v_obs:.0f}) px, '
                    f'obwohl sich die Peilung gegenueber Sample {i} um {peilung_grad:.0f} Grad '
                    f'geaendert hat. Die Kamera sieht hier NICHT die Pylone, sondern etwas '
                    f'Festes im Raum. Setze target_label auf die Pylonenfarbe und "clear".',
                    warn=True)
                break
        self.samples.append({'bearing_rad': bearing, 'range_m': distance,
                             'u_obs': u_obs, 'v_obs': v_obs, 'label': label,
                             'r_innen': r_innen, 'r_aussen': r_aussen})
        self._status(
            f'Sample {len(self.samples)}: Lidar {math.degrees(bearing):+7.2f} Grad / '
            f'{distance:.2f} m ({points} Punkte) <-> {label} bei '
            f'({u_obs:.1f}, {v_obs:.1f}) px, Flaeche {area:.0f}, '
            f'Radius {r_innen:.0f}..{r_aussen:.0f} px (Ober- bis Fusspunkt)')

    def _try_auto_sample(self):
        cluster = self.find_target_cluster()
        if cluster is None:
            return
        step = math.radians(self.get_parameter('auto_min_bearing_step_deg').value)
        for sample in self.samples:
            if abs(_wrap(cluster[0] - sample['bearing_rad'])) < step:
                return
        self.cmd_sample()

    def cmd_clear(self):
        self.samples.clear()
        self._status('Alle Samples verworfen.')

    def cmd_list(self):
        if not self.samples:
            self._status('Keine Samples.')
            return
        lines = [f'{len(self.samples)} Samples:']
        for i, sample in enumerate(self.samples, 1):
            lines.append(f'  {i:2d}  {math.degrees(sample["bearing_rad"]):+7.2f} Grad  '
                         f'{sample["range_m"]:.2f} m  -> ({sample["u_obs"]:7.1f}, '
                         f'{sample["v_obs"]:7.1f}) px  [{sample["label"]}]')
        self._status('\n'.join(lines))

    def cmd_verify(self):
        if not self.samples:
            self._status('Keine Samples zum Pruefen.', warn=True)
            return
        residuals = self._residuals(self.calib)
        rms_px = float(np.sqrt(np.mean(residuals ** 2) * 2))
        per_sample = np.hypot(residuals[0::2], residuals[1::2])
        self._status(f'Reprojektionsfehler: RMS {rms_px:.1f} px, max {per_sample.max():.1f} px '
                     f'ueber {len(self.samples)} Samples '
                     f'(Bildkreisradius {self.calib.radius_px:.0f} px)')

    def cmd_auto(self):
        from rclpy.parameter import Parameter
        new_state = not self.get_parameter('auto_sample').value
        self.set_parameters([Parameter('auto_sample', value=new_state)])
        self._status(f'Auto-Sampling {"AN" if new_state else "AUS"} -- Klotz jeweils '
                     f'mindestens {self.get_parameter("auto_min_bearing_step_deg").value:.0f} '
                     'Grad weiter stellen.')

    def cmd_save(self):
        self.calib.note = (f'rotation_calibration, {len(self.samples)} Samples, '
                           f'{_now_text()}')
        self.calib.to_yaml(self.calib_path)
        self._status(f'Kalibrierung gespeichert -> {self.calib_path}')

    def cmd_reload(self):
        self.calib = FisheyeCalib.load(self.calib_path, _packaged_default())
        self._push_to_params()
        self._status(f'Kalibrierung neu geladen: yaw={self.calib.yaw_deg:.2f} Grad')

    # ------------------------------------------------------------------ #
    # Loesung
    # ------------------------------------------------------------------ #
    def cmd_solve(self):
        fit = list(self.get_parameter('fit_params').value)
        if len(self.samples) < 2:
            self._status('Mindestens 2 Samples noetig (3+ sind deutlich besser).', warn=True)
            return
        unknown = [name for name in fit if name not in LIVE_FIELDS]
        if unknown:
            self._status(f'Unbekannte fit_params: {unknown}', warn=True)
            return

        before = self._rms(self.calib)
        if fit == ['yaw_deg']:
            self._solve_yaw_closed_form()
        elif not self._solve_least_squares(fit):
            return
        self._push_to_params()

        after = self._rms(self.calib)
        self._status(
            f'Geloest ueber {len(self.samples)} Samples: '
            f'yaw={self.calib.yaw_deg:.2f} pitch={self.calib.pitch_deg:.2f} '
            f'roll={self.calib.roll_deg:.2f} Grad, cx={self.calib.cx:.1f} cy={self.calib.cy:.1f}\n'
            f'  Reprojektionsfehler RMS: {before:.1f} px -> {after:.1f} px. '
            'Mit "save" festschreiben.')

    def _solve_yaw_closed_form(self):
        """yaw = zirkulaerer Mittelwert von (phi_beobachtet - Azimut).

        Der Drehwinkel um die optische Achse haengt nur vom Azimut ab, nicht von
        der Hoehe des Klotzes -- deshalb ist er ohne Startwert exakt loesbar.
        """
        keep, dropped = self._inliers()
        self._report_dropped(dropped)
        deltas = self._sample_yaw_deltas()[keep]
        yaw = math.atan2(np.sin(deltas).mean(), np.cos(deltas).mean())
        spread = np.degrees(np.abs(_wrap(deltas - yaw)))
        self.calib.yaw_deg = math.degrees(yaw)
        self._status(f'Yaw-Streuung der Einzelmessungen: max {spread.max():.2f} Grad, '
                     f'Mittel {spread.mean():.2f} Grad')
        if spread.max() > 10.0:
            self._status('Streuung > 10 Grad -- meist ist ein Sample falsch zugeordnet. '
                         'Mit "list" pruefen.', warn=True)

    def cmd_height(self):
        """Loest die Kamerahoehe cam_z geschlossen aus den Bildradien.

        Der Bildradius haengt nur an theta, und theta = atan2(rho, dz) mit
        rho = waagerechter Abstand (aus dem Lidar) und dz = Zielhoehe - cam_z.
        Also: dz = rho / tan(theta_gemessen), und cam_z = target_height_m - dz.

        Anders als yaw ist das an ``target_height_m`` gekoppelt -- messbar ist
        nur der Hoehenunterschied zwischen Kamera und Zielmarke, nicht beides
        getrennt. Nahe Samples (kleines rho) tragen am meisten bei, weil theta
        dort am weitesten von 90 Grad weg ist.
        """
        if not self.samples:
            self._status('Keine Samples -- erst sampeln.', warn=True)
            return

        target_height = self.get_parameter('target_height_m').value
        estimates, weights = [], []
        for sample in self.samples:
            radius = math.hypot(sample['u_obs'] - self.calib.cx,
                                sample['v_obs'] - self.calib.cy)
            theta = float(radius_to_theta(self.calib, radius))
            dx = sample['range_m'] * math.cos(sample['bearing_rad']) - self.calib.cam_x
            dy = sample['range_m'] * math.sin(sample['bearing_rad']) - self.calib.cam_y
            rho = math.hypot(dx, dy)

            tan_theta = math.tan(theta)
            if abs(tan_theta) < 1e-9:      # theta exakt 90 Grad -> dz = 0
                estimates.append(target_height)
                weights.append(1.0 / max(rho, 1e-3))
                continue
            estimates.append(target_height - rho / tan_theta)
            weights.append(1.0 / max(rho, 1e-3))

        estimates = np.asarray(estimates)
        cam_z = float(np.average(estimates, weights=np.asarray(weights)))
        spread = float(np.abs(estimates - cam_z).max())

        self.calib.cam_z = cam_z
        self._push_to_params()
        self._status(
            f'cam_z = {cam_z * 100:.1f} cm ueber der Lidar-Ebene '
            f'(bei target_height_m = {target_height * 100:.1f} cm), '
            f'Streuung der Einzelwerte max {spread * 100:.1f} cm ueber '
            f'{len(self.samples)} Samples. Mit "save" festschreiben.')
        if spread > 0.03:
            self._status('Streuung > 3 cm -- pruefe target_height_m und nimm Samples '
                         'naeher am Roboter auf (unter ~0.5 m ist die Hoehe am '
                         'besten bestimmt).', warn=True)

    def _sample_yaw_deltas(self):
        """Je Sample: Blob-Azimut minus Lidar-Azimut. Sollte konstant sein (= yaw)."""
        deltas = []
        for sample in self.samples:
            dx = sample['range_m'] * math.cos(sample['bearing_rad']) - self.calib.cam_x
            dy = sample['range_m'] * math.sin(sample['bearing_rad']) - self.calib.cam_y
            azimuth = math.atan2(dy, dx)
            phi = math.atan2(sample['v_obs'] - self.calib.cy,
                             sample['u_obs'] - self.calib.cx)
            if self.calib.mirror:
                phi = -phi
            deltas.append(float(_wrap(phi - azimuth)))
        return np.asarray(deltas)

    def _inliers(self):
        """Samples, deren Blob zur Lidar-Peilung passt.

        Ein einzelner falsch zugeordneter Blob zieht sonst yaw UND die
        Brennweite mit sich. Robuster Mittelpunkt ist der zirkulaere Median;
        alles, was weiter als ``outlier_reject_deg`` davon abweicht, fliegt
        raus. Die Radien eines falschen Blobs sind ohnehin wertlos.
        """
        if len(self.samples) < 3:
            return list(range(len(self.samples))), []

        deltas = self._sample_yaw_deltas()
        # Zirkulaerer Median: den Kandidaten nehmen, der die kleinste Summe
        # der Winkelabstaende hat -- unempfindlich gegen einzelne Ausreisser.
        spans = [np.abs(_wrap(deltas - d)).sum() for d in deltas]
        center = deltas[int(np.argmin(spans))]
        abweichung = np.abs(np.degrees(_wrap(deltas - center)))

        grenze = self.get_parameter('outlier_reject_deg').value
        keep = [i for i in range(len(self.samples)) if abweichung[i] <= grenze]
        drop = [i for i in range(len(self.samples)) if abweichung[i] > grenze]
        if len(keep) < 2:
            return list(range(len(self.samples))), []
        return keep, [(i, abweichung[i]) for i in drop]

    def _report_dropped(self, dropped):
        if not dropped:
            return
        lines = [f'{len(dropped)} Sample(s) als Ausreisser verworfen '
                 f'(Blob passt nicht zur Peilung):']
        for i, abw in dropped:
            s = self.samples[i]
            lines.append(f'  Sample {i + 1}: {math.degrees(s["bearing_rad"]):+7.2f} Grad / '
                         f'{s["range_m"]:.2f} m -> ({s["u_obs"]:.0f}, {s["v_obs"]:.0f}) px, '
                         f'{abw:.0f} Grad daneben')
        lines.append('Dauerhaft loswerden: "clear" und neu sampeln.')
        self._status('\n'.join(lines), warn=True)

    def cmd_background(self):
        """Nimmt einen Referenzscan der leeren Umgebung auf.

        Danach gilt als Ziel nur noch, was NAEHER misst als diese Referenz.
        Kabel, Elektronik, Tischkanten, Waende -- alles Feste steht in der
        Referenz und faellt damit von selbst raus. Das ist deutlich robuster
        als Blindsektoren, weil es auch die Randbereiche erwischt, in denen der
        Strahl den eigenen Aufbau nur streift.

        Wichtig: dabei die Pylone WEGNEHMEN und den Roboter nicht bewegen.
        """
        needed = self.get_parameter('background_scan_count').value
        if len(self.background_buffer) < needed:
            self.background_buffer.clear()
            self.background_collecting = True
            self._status(f'Nehme {needed} Referenzscans auf -- Pylone jetzt WEGNEHMEN '
                         'und Roboter still halten. Danach "background" nochmal senden.')
            return

        stack = np.asarray(self.background_buffer[-needed:], dtype=float)
        finite = np.isfinite(stack) & (stack > 0)
        reference = np.full(stack.shape[1], np.inf)
        has_any = finite.any(0)
        if has_any.any():
            reference[has_any] = np.nanmedian(
                np.where(finite[:, has_any], stack[:, has_any], np.nan), axis=0)

        self.background = reference
        self.background_buffer.clear()
        self.background_collecting = False

        messbar = np.isfinite(reference)
        self._status(
            f'Referenzscan steht ({needed} Scans, {messbar.sum()} von {reference.size} '
            f'Strahlen mit Wert).\n'
            f'  Naechster fester Punkt bei {reference[messbar].min():.2f} m.\n'
            f'  Ab jetzt zaehlt als Ziel nur, was mindestens '
            f'{self.get_parameter("foreground_margin_m").value * 100:.0f} cm naeher misst. '
            'Pylone hinstellen und "sample".')

    def cmd_blind(self):
        """Misst die verbauten Lidar-Sektoren aus (Kabel, Elektronik, Aufbau).

        Sammelt ein paar Sekunden Scans und sucht die Winkelbereiche, in denen
        der Scanner dauerhaft nur wenige Zentimeter misst -- da schaut er auf
        den eigenen Aufbau. Diese Bereiche werden danach in beiden Nodes
        ausgeblendet.
        """
        needed = self.get_parameter('blind_scan_count').value
        if len(self.blind_buffer) < needed:
            self.blind_collecting = True
            self._status(f'Sammle {needed} Scans fuer die Blindsektoren '
                         f'({len(self.blind_buffer)} bisher) -- Roboter dabei '
                         'still stehen lassen. Danach "blind" nochmal senden.')
            return

        scans = [s for s, _ in self.blind_buffer[-needed:]]
        angle_min, angle_increment = self.blind_buffer[-1][1]
        sectors = find_blind_sectors(
            scans, angle_min, angle_increment,
            near_m=self.get_parameter('blind_near_m').value,
            min_width_deg=self.get_parameter('blind_min_width_deg').value)

        self.calib.lidar_blind_sectors_deg = sectors
        self.blind_buffer.clear()
        self.blind_collecting = False

        if not sectors:
            self._status('Keine blockierten Sektoren gefunden -- Lidar sieht frei rundum.')
            return

        lines = [f'{len(sectors) // 2} blockierte Sektoren gefunden:']
        total = 0.0
        for i in range(0, len(sectors) - 1, 2):
            lo, hi = sectors[i], sectors[i + 1]
            width = (hi - lo) % 360.0
            total += width
            lines.append(f'  {lo:+7.1f} bis {hi:+7.1f} Grad  ({width:.1f} Grad breit)')
        lines.append(f'Summe {total:.0f} Grad blockiert -> nutzbar rund {360 - total:.0f} Grad.')
        lines.append('Mit "save" festschreiben.')
        self._status('\n'.join(lines))

    def cmd_ring(self):
        """Meldet, wo der Horizontring aktuell liegt, und gleicht f_px ab.

        Noetig, weil der Direktregler ``horizon_radius_px`` nur calib.f_px
        setzt -- der ROS-Parameter f_px hinkt sonst hinterher.
        """
        self._push_to_params()
        ring = self.calib.focal_px * math.pi / 2.0
        self._status(
            f'Horizontring bei r = {ring:.1f} px (Bildkreisradius '
            f'{self.calib.radius_px:.0f} px, also {ring / self.calib.radius_px * 100:.0f} '
            f'Prozent nach aussen).\n'
            f'  Brennweite f = {self.calib.focal_px:.1f} px/rad, entspricht einer '
            f'Objektiv-FOV von {math.degrees(2 * self.calib.radius_px / self.calib.focal_px):.0f} '
            f'Grad.\n'
            f'  Verschieben: ros2 param set {self.get_name()} horizon_radius_px <px>')

    def cmd_radial(self):
        """Loest Brennweite f und Objektivhoehe ueber der Matte aus den Pylonen.

        Der Fusspunkt der Pylone steht auf der Matte, also immer L unter dem
        Objektiv. Sein Winkel zur optischen Achse (die nach OBEN zeigt) ist

            theta_fuss(rho) = atan2(rho, -L)

        und der gemessene Bildradius r_aussen = f * theta_fuss. Nah steht die
        Pylone fast senkrecht unter der Kamera (theta gegen 180 Grad, grosser
        Radius), fern naehert sie sich dem Horizont (90 Grad, kleiner Radius).
        Diese Spreizung macht f und L gemeinsam bestimmbar. Die Oberkante
        liefert dieselbe Gleichung mit dz = Pylonenhoehe - L.

        Deswegen: Samples ueber einen WEITEN Entfernungsbereich aufnehmen, nicht
        alle auf demselben Abstand -- der Winkel unterscheidet sich sonst kaum.
        """
        keep, dropped = self._inliers()
        self._report_dropped(dropped)
        usable = [self.samples[i] for i in keep
                  if np.isfinite(self.samples[i].get('r_aussen', float('nan')))]
        if len(usable) < 3:
            self._status('Mindestens 3 Samples mit Blob-Radien noetig.', warn=True)
            return
        spread = max(s['range_m'] for s in usable) / max(min(s['range_m'] for s in usable), 1e-3)
        if spread < 2.0:
            self._status(f'Entfernungen liegen nur Faktor {spread:.1f} auseinander. '
                         'Fuer f brauchst du nah UND fern (z.B. 0.2 m bis 1.5 m).',
                         warn=True)

        try:
            from scipy.optimize import least_squares
        except ImportError:
            self._status('scipy fehlt -- radial nicht moeglich.', warn=True)
            return

        pylon = self.get_parameter('pylon_height_m').value
        rho = np.array([s['range_m'] for s in usable])
        r_fuss = np.array([s['r_aussen'] for s in usable])
        r_kopf = np.array([s['r_innen'] for s in usable])

        # NUR der Fusspunkt geht in den Fit. Er steht garantiert auf der Matte,
        # also immer genau lens unter dem Objektiv -- eine harte, bekannte
        # Groesse. Die Oberkante dagegen ist das Ende der FARBIGEN Flaeche, und
        # die reicht selten bis ganz oben. Nimmt man sie mit, zieht sie den Fit
        # kaputt (am echten Roboter: RMS 19.5 statt 1.4 px).
        def residuals(x):
            f_px, lens = float(x[0]), float(x[1])
            # theta = Winkel zur optischen Achse (zeigt nach OBEN).
            # arctan2(rho, -lens): nah fast senkrecht nach unten (gegen 180
            # Grad), fern gegen den Horizont (90 Grad).
            return f_px * np.arctan2(rho, -lens) - r_fuss

        start = np.array([self.calib.focal_px, max(self.calib.cam_z, 0.02)])
        result = least_squares(residuals, start,
                               bounds=([1.0, 0.001], [10000.0, 0.5]))
        f_px, lens = float(result.x[0]), float(result.x[1])
        rms = float(np.sqrt(np.mean(result.fun ** 2)))

        # Die Oberkante dient nur noch als Gegenprobe: wie hoch reicht die Farbe?
        theta_kopf = r_kopf / f_px
        farbhoehe = lens + rho / np.tan(theta_kopf)

        self.calib.f_px = f_px
        self.calib.fov_deg = math.degrees(2.0 * self.calib.radius_px / f_px)
        self._push_to_params()

        ring = f_px * math.pi / 2.0
        self._status(
            f'Brennweite f = {f_px:.1f} px/rad (vorher {start[0]:.1f}).\n'
            f'  Horizontring liegt damit bei r = {ring:.1f} px '
            f'(vorher {start[0] * math.pi / 2:.1f} px).\n'
            f'  Daraus folgende Objektiv-FOV: {self.calib.fov_deg:.0f} Grad '
            f'(Bildkreisradius {self.calib.radius_px:.0f} px).\n'
            f'  Objektiv steht {lens * 100:.1f} cm ueber der Matte.\n'
            f'  Farbige Flaeche der Pylone reicht bis {farbhoehe.mean() * 100:.1f} cm '
            f'ueber der Matte (Streuung {farbhoehe.std() * 100:.1f} cm).\n'
            f'  Restfehler RMS {rms:.1f} px ueber {len(usable)} Fusspunkte. '
            'Mit "save" festschreiben.')

        if rms > 8.0:
            self._status(
                f'RMS {rms:.1f} px ist hoch. Meist steckt ein Sample dahinter, bei dem '
                'der Fusspunkt verdeckt war oder der Blob mit etwas verschmolzen ist. '
                'Mit "list" pruefen.', warn=True)

        # Die farbige Flaeche ist der Bereich, den der Ring treffen kann.
        # Entscheidend ist nicht die Pylonenhoehe, sondern wie weit die FARBE
        # reicht -- nur dort kann der Ring etwas Brauchbares ablesen.
        oben = float(farbhoehe.mean())
        if lens >= oben:
            self._status(
                f'ACHTUNG: Objektiv sitzt {(lens - oben) * 100:.1f} cm UEBER dem oberen '
                f'Rand der farbigen Flaeche. Der Horizontring schaut damit darueber '
                'hinweg -- kein Kippwinkel rettet das fuer nah und fern zugleich. '
                'Kamera tiefer setzen oder sample_mode:=height nehmen.', warn=True)
        else:
            mitte = oben / 2.0
            self._status(
                f'Objektiv sitzt in der farbigen Flaeche, {(oben - lens) * 100:.1f} cm '
                f'unter deren Oberkante. Horizontring passt. Am meisten Luft nach '
                f'beiden Seiten haettest du auf {mitte * 100:.1f} cm '
                f'(aktuell {lens * 100:.1f} cm).')
        if abs(oben - pylon) > 0.03:
            self._status(
                f'Hinweis: die farbige Flaeche reicht nur {oben * 100:.1f} cm hoch, '
                f'pylon_height_m steht auf {pylon * 100:.0f} cm. Fuer den Ring zaehlt '
                'die Farbe, nicht die Bauhoehe -- der Parameter dient nur diesem '
                'Abgleich.')

    def _solve_least_squares(self, fit) -> bool:
        try:
            from scipy.optimize import least_squares
        except ImportError:
            self._status('scipy fehlt -- nur fit_params: ["yaw_deg"] moeglich.', warn=True)
            return False

        start = np.array([getattr(self.calib, name) for name in fit], dtype=float)

        def cost(values):
            trial = copy.deepcopy(self.calib)
            for name, value in zip(fit, values):
                setattr(trial, name, float(value))
            return self._residuals(trial)

        result = least_squares(cost, start, method='lm' if len(start) < len(self.samples) * 2
                               else 'trf')
        for name, value in zip(fit, result.x):
            setattr(self.calib, name, float(value))
        return True

    def _residuals(self, calib: FisheyeCalib) -> np.ndarray:
        """Aneinandergereihte (du, dv) je Sample.

        Die Zielmarke sitzt auf ``target_height_m`` ueber der Lidar-Ebene -- NICHT
        auf Kamerahoehe. Wuerde man sie auf cam_z legen, waere theta immer exakt
        90 Grad und der Bildradius damit fuer jedes Sample gleich; die radiale
        Richtung traegt dann keine Information mehr und cam_z liesse sich gar
        nicht bestimmen.
        """
        height = self.get_parameter('target_height_m').value
        pts, observed = [], []
        for sample in self.samples:
            pts.append([sample['range_m'] * math.cos(sample['bearing_rad']),
                        sample['range_m'] * math.sin(sample['bearing_rad']),
                        height])
            observed.append([sample['u_obs'], sample['v_obs']])
        u, v, _, _, _ = project(calib, np.asarray(pts))
        observed = np.asarray(observed)
        return np.column_stack([u - observed[:, 0], v - observed[:, 1]]).ravel()

    def _rms(self, calib: FisheyeCalib) -> float:
        if not self.samples:
            return float('nan')
        return float(np.sqrt(np.mean(self._residuals(calib) ** 2) * 2))

    # ------------------------------------------------------------------ #
    def _publish_debug(self):
        if not self.get_parameter('debug').value:
            return
        now = self.get_clock().now().nanoseconds * 1e-9
        rate = self.get_parameter('debug_rate_hz').value
        if rate > 0 and now - self.last_debug_stamp < 1.0 / rate:
            return
        self.last_debug_stamp = now

        canvas = self.latest_image.copy()
        center = (int(round(self.calib.cx)), int(round(self.calib.cy)))
        cv2.circle(canvas, center, int(round(self.calib.radius_px)), (255, 255, 0), 2)
        cv2.drawMarker(canvas, center, (255, 255, 0), cv2.MARKER_CROSS, 20, 2)

        # Horizontring: hier greift lidar_pixel_mapper im horizon-Modus ab.
        ring = self.calib.focal_px * math.pi / 2.0
        cv2.circle(canvas, center, int(round(ring)), (0, 140, 255), 2)
        cv2.putText(canvas, f'Horizont r={ring:.0f}px',
                    (center[0] - 70, center[1] - int(round(ring)) - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 140, 255), 2)

        # Peilrose: wo landen 0/90/180/270 Grad im Roboter-Frame?
        for degrees in range(0, 360, 30):
            angle = math.radians(degrees)
            point = np.array([[math.cos(angle), math.sin(angle), 0.0]]) * 1.0
            u, v, _, _, _ = project(self.calib, point)
            tip = (int(round(u[0])), int(round(v[0])))
            highlight = degrees == 0
            cv2.line(canvas, center, tip, (0, 255, 255) if highlight else (90, 90, 90),
                     2 if highlight else 1)
            cv2.putText(canvas, 'vorne' if highlight else f'{degrees}', tip,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 255, 255) if highlight else (160, 160, 160), 1)

        # Aktuelles Live-Paar: Lidar-Cluster (projiziert) vs. Farb-Blob (gemessen)
        cluster = self.find_target_cluster()
        if cluster is not None:
            bearing, distance, _ = cluster
            point = np.array([[distance * math.cos(bearing), distance * math.sin(bearing),
                               self.get_parameter('target_height_m').value]])
            u, v, _, _, _ = project(self.calib, point)
            cv2.drawMarker(canvas, (int(round(u[0])), int(round(v[0]))), (0, 165, 255),
                           cv2.MARKER_TILTED_CROSS, 26, 3)
        blob = self.find_target_blob()
        if blob is not None:
            colour = colors.LABEL_BGR.get(blob[2], (255, 255, 255))
            cv2.circle(canvas, (int(round(blob[0])), int(round(blob[1]))), 16, colour, 3)
            # Radiale Ausdehnung des Blobs = Ober- bis Fusspunkt der Pylone.
            # Genau daraus rechnet "radial" die Brennweite.
            if np.isfinite(blob[4]):
                for radius in (blob[4], blob[5]):
                    cv2.circle(canvas, center, int(round(radius)), colour, 1)

        # Bereits aufgenommene Samples
        for sample in self.samples:
            cv2.circle(canvas, (int(round(sample['u_obs'])), int(round(sample['v_obs']))),
                       7, (255, 255, 255), 2)

        cv2.putText(canvas,
                    f'yaw={self.calib.yaw_deg:.1f} pitch={self.calib.pitch_deg:.1f} '
                    f'roll={self.calib.roll_deg:.1f} | Samples={len(self.samples)}',
                    (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(canvas, 'orange X = Lidar projiziert, Farbring = Kamera-Blob',
                    (12, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        out = self.bridge.cv2_to_imgmsg(canvas, 'bgr8')
        out.header.frame_id = 'camera'
        self.pub_debug.publish(out)


def _wrap(angle):
    """Winkel nach (-pi, pi] normieren."""
    return (np.asarray(angle) + np.pi) % (2 * np.pi) - np.pi


def _now_text() -> str:
    import datetime
    return datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')


def _packaged_default() -> str:
    try:
        from ament_index_python.packages import get_package_share_directory
        return os.path.join(get_package_share_directory('camera_lidar_fusion'),
                            'config', 'fisheye_calib.yaml')
    except Exception:  # noqa: BLE001
        return ''


def main(args=None):
    rclpy.init(args=args)
    node = RotationCalibration()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
