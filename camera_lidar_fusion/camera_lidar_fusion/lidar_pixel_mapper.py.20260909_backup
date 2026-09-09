#!/usr/bin/env python3
"""Ordnet jedem Lidar-Punkt den Pixel bzw. die Farbe der 360-Grad-Kamera zu.

Fuer jeden Scan wird jeder gueltige Messpunkt ueber das Fisheye-Modell ins Bild
projiziert, dort die Farbe ausgelesen und als rot/gruen/magenta/schwarz
klassifiziert. Ergebnis geht raus als

  * CSV   (Hauptausgabe -- eine Zeile pro Lidar-Punkt),
  * PointCloud2 mit RGB fuer Foxglove,
  * Debug-Bild mit den eingezeichneten Projektionen.

Wo im Bild abgegriffen wird (Parameter ``sample_mode``):

  horizon  (Default) auf Objektivhoehe. Die Hoehendifferenz zur Kamera ist dann
           null, theta exakt 90 Grad, der Bildradius konstant f*pi/2 -- es
           bleibt nur der Azimut, also eine feste Kreislinie im Bild.
           Das reicht fuer Pylonen, SOLANGE das Objektiv zwischen Matte und
           Pylonenoberkante sitzt: eine Pylone, die die waagerechte Ebene durch
           die Linse durchstoesst, liegt in JEDER Entfernung auf diesem Ring.
           Vorteil: Entfernungsfehler des Lidars und ein falsches cam_z wirken
           sich radial gar nicht mehr aus, es zaehlt nur noch yaw.
           Sitzt die Linse ueber der Pylonenoberkante, greift der Ring dagegen
           an der Pylone vorbei -- dann height nehmen.

  height   auf fester Hoehe ``sample_height_m`` ueber der Lidar-Ebene. Der
           Bildradius haengt dann an der Entfernung.

Ring nach unten kippen (``sample_depression_deg``, nur bei horizon): aus der
waagerechten Ebene wird ein Kegel. Der greift in waagerechter Entfernung rho um
rho*tan(Winkel) unter der Linse ab -- die Tiefe waechst also MIT der Entfernung.
Bei 1 Grad sind das 0.5 cm auf 0.3 m, aber 3.5 cm auf 2 m. Fuer 10-cm-Pylonen
heisst das: nur Bruchteile eines Grades sind brauchbar, und sitzt die Linse
ueber der Pylonenoberkante, gibt es GAR KEINEN Winkel, der nah und fern
gleichzeitig trifft -- dann hilft nur ``height``.

Mitteln statt ein Pixel (``sample_band_m``, ``sample_band_count``): es werden
mehrere Stuetzstellen entlang der radialen Linie durch den Punkt gelesen -- die
liegt im Fisheye laengs der Pylone -- und davon der Median genommen. Die
Bandbreite ist in Metern Pylonenhoehe angegeben und wird je Punkt aus der
Entfernung in Pixel umgerechnet, schrumpft fern also von selbst mit und bleibt
damit innerhalb der Pylone. 0 schaltet auf ein einzelnes Pixel zurueck.

CSV-Modi (Parameter ``csv_mode``):
  trigger      pro Trigger eine Datei  -> ros2 topic pub --once \
                   /camera_lidar/capture std_msgs/msg/Empty '{}'
  continuous   haengt jeden Scan an eine Datei an
  off          keine CSV, nur Topics

Start:
    ros2 run camera_lidar_fusion lidar_pixel_mapper
"""

import csv
import datetime
import math
import os

import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image, LaserScan, PointCloud2, PointField
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Empty, String

import cv2

from camera_lidar_fusion import colors
from camera_lidar_fusion.fisheye_model import (
    FisheyeCalib, project, scan_to_points, theta_to_radius, visible_mask,
)

CLOUD_FIELDS = [
    PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
    PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
    PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
    PointField(name='rgb', offset=12, datatype=PointField.FLOAT32, count=1),
]

CSV_HEADER = [
    'stamp_sec', 'idx', 'angle_deg', 'range_m', 'x_m', 'y_m', 'z_m',
    'u_px', 'v_px', 'theta_deg', 'phi_deg', 'b', 'g', 'r', 'h', 's', 'v', 'label',
]


class LidarPixelMapper(Node):

    def __init__(self):
        super().__init__('lidar_pixel_mapper')

        self.declare_parameter('scan_topic', '/scan')
        self.declare_parameter('image_topic', '/video_source/raw')
        self.declare_parameter('calib_file', '/workspace/config/fisheye_calib.yaml')
        # horizon = auf dem Horizontring abgreifen (Default, siehe Modulkopf).
        # height  = auf fester Hoehe ueber der Lidar-Ebene, dann zaehlt
        #           sample_height_m. Nur noetig, wenn das Objektiv NICHT
        #           zwischen Matte und Pylonenoberkante sitzt.
        self.declare_parameter('sample_mode', 'horizon')
        self.declare_parameter('sample_height_m', 0.00)
        # Ring nach unten kippen (nur bei horizon). 0 = waagerecht durch die
        # Linse. Positiv blickt nach unten, der Ring wird groesser. ACHTUNG: der
        # Kegel greift dann in ENTFERNUNG*tan(Winkel) Tiefe -- fern also viel
        # tiefer als nah. Siehe Modulkopf.
        self.declare_parameter('sample_depression_deg', 0.0)
        # Statt eines Pixels laengs der Pylone mitteln: ueber +-sample_band_m
        # Pylonenhoehe, mit sample_band_count Stuetzstellen. 0 = ein Pixel.
        self.declare_parameter('sample_band_m', 0.03)
        self.declare_parameter('sample_band_count', 5)
        # Median-Blur ueber das GANZE Bild -- kostet auf 1280x960 rund 26 ms je
        # Scan, also bei 15 Hz gut 40 Prozent eines Kerns. Solange das Band aktiv
        # ist (sample_band_m > 0), ist der Blur ueberfluessig: der Median laengs
        # der Pylone faengt Ausreisser bereits ab. Nur hochdrehen, wenn du das
        # Band abschaltest.
        self.declare_parameter('patch_px', 1)
        self.declare_parameter('range_min_m', 0.05)
        self.declare_parameter('range_max_m', 3.0)
        self.declare_parameter('max_sync_age_s', 0.5)
        self.declare_parameter('csv_mode', 'trigger')
        self.declare_parameter('csv_dir', '/workspace/lidar_color_logs')
        self.declare_parameter('csv_only_labeled', False)
        # debug ist der Hauptschalter fuer alles, was nur zum Anschauen da ist:
        # die eingefaerbte PointCloud2 und das Debug-Bild. Im Wettkampflauf auf
        # false setzen, dann faellt das Zeichnen und Serialisieren komplett weg.
        self.declare_parameter('debug', True)
        self.declare_parameter('publish_cloud', True)
        self.declare_parameter('publish_debug_image', True)
        self.declare_parameter('debug_rate_hz', 5.0)

        self.scan_topic = self.get_parameter('scan_topic').value
        self.image_topic = self.get_parameter('image_topic').value
        self.calib_path = self.get_parameter('calib_file').value
        self.csv_dir = self.get_parameter('csv_dir').value
        self.ranges = colors.ranges_from_params(self)

        self.calib = FisheyeCalib.load(self.calib_path, _packaged_default())
        self.bridge = CvBridge()
        self.latest_image = None        # (cv_bild, stamp_sec)
        self.capture_pending = False
        self.continuous_writer = None   # (file, csv.writer) fuer csv_mode=continuous
        self.last_debug_stamp = 0.0

        self.create_subscription(LaserScan, self.scan_topic, self.on_scan,
                                 qos_profile_sensor_data)
        self.create_subscription(Image, self.image_topic, self.on_image,
                                 qos_profile_sensor_data)
        self.create_subscription(Empty, '/camera_lidar/capture', self.on_capture, 10)
        # Nach einem "save" in der Kalibrier-Node hier neu einlesen, statt die
        # Node neu starten zu muessen.
        self.create_subscription(Empty, '/camera_lidar/reload', self.on_reload, 10)

        self.pub_cloud = self.create_publisher(PointCloud2, '/camera_lidar/colored_scan', 5)
        self.pub_debug = self.create_publisher(Image, '/camera_lidar/debug_image', 2)
        self.pub_summary = self.create_publisher(String, '/camera_lidar/summary', 10)

        if self.get_parameter('csv_mode').value == 'continuous':
            self._open_continuous_csv()

        mode = self.get_parameter('sample_mode').value
        depression = self.get_parameter('sample_depression_deg').value
        if mode == 'horizon':
            radius = float(theta_to_radius(
                self.calib, np.array([np.pi / 2 + math.radians(depression)]))[0])
            abgriff = (f'horizon -- feste Kreislinie bei r={radius:.1f} px, '
                       f'entfernungsunabhaengig.\n'
                       f'    Setzt voraus, dass das Objektiv ZWISCHEN Matte und '
                       f'Pylonenoberkante sitzt. Mittig (ca. 5 cm bei 10-cm-Pylonen) '
                       f'ist der Abstand zu beiden Kanten am groessten.')
            if depression != 0.0:
                abgriff += (f'\n    Ring {depression:.2f} Grad nach unten gekippt: greift '
                            f'{math.tan(math.radians(depression)) * 30:.1f} cm unter der Linse '
                            f'ab auf 0.3 m, aber '
                            f'{math.tan(math.radians(depression)) * 200:.1f} cm auf 2 m.')
        else:
            abgriff = (f'height -- {self.get_parameter("sample_height_m").value * 100:.1f} cm '
                       f'ueber der Lidar-Ebene, Bildradius haengt an der Entfernung.')

        band_m = self.get_parameter('sample_band_m').value
        if band_m > 0.0:
            abgriff += (f'\n    Median ueber +-{band_m * 100:.1f} cm Pylonenhoehe '
                        f'({self.get_parameter("sample_band_count").value} Stuetzstellen '
                        f'laengs der Pylone).')
        else:
            patch = self.get_parameter('patch_px').value
            abgriff += f'\n    Ein einzelnes Pixel (sample_band_m = 0, patch_px = {patch}).'
            if patch <= 1:
                abgriff += (' ACHTUNG: weder Band noch Blur -- ungefiltert. '
                            'patch_px hochsetzen oder sample_band_m > 0.')

        self.get_logger().info(
            f'lidar_pixel_mapper laeuft. scan={self.scan_topic} image={self.image_topic}\n'
            f'  Kalibrierung: {self.calib_path}\n'
            f'  Bildkreis cx={self.calib.cx:.1f} cy={self.calib.cy:.1f} '
            f'r={self.calib.radius_px:.1f} FOV={self.calib.fov_deg:.0f} Grad\n'
            f'  Lage yaw={self.calib.yaw_deg:.2f} pitch={self.calib.pitch_deg:.2f} '
            f'roll={self.calib.roll_deg:.2f} (Grad), '
            f'Kamera {self.calib.cam_z * 100:.1f} cm ueber der Lidar-Ebene\n'
            f'  Abgriff: {abgriff}\n'
            f'  CSV-Modus: {self.get_parameter("csv_mode").value} -> {self.csv_dir}\n'
            f'  debug={self.get_parameter("debug").value} -> Foxglove: '
            f'/camera_lidar/colored_scan (Lidar-Punkte in Kamerafarbe) '
            f'und /camera_lidar/debug_image'
        )

    # ---------------------------------------------------------------- #
    def on_image(self, msg: Image):
        try:
            image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        except Exception as exc:  # noqa: BLE001
            self.get_logger().warn(f'Bild nicht dekodierbar: {exc}')
            return
        self.latest_image = (image, _stamp_sec(msg.header.stamp))

    def on_capture(self, _msg: Empty):
        self.capture_pending = True
        self.get_logger().info('Capture angefordert -- naechster Scan wird als CSV abgelegt.')

    def on_reload(self, _msg: Empty):
        self.calib = FisheyeCalib.load(self.calib_path, _packaged_default())
        ring = float(theta_to_radius(self.calib, np.array([np.pi / 2]))[0])
        self.get_logger().info(
            f'Kalibrierung neu geladen: yaw={self.calib.yaw_deg:.2f} Grad, '
            f'Horizontring r={ring:.1f} px, '
            f'{len(self.calib.lidar_blind_sectors_deg) // 2} Blindsektoren.')

    # ---------------------------------------------------------------- #
    def _sample_z(self, rho: np.ndarray):
        """Auf welcher Hoehe (Roboter-Frame) wird der Lidar-Punkt abgegriffen?

        Bei ``horizon`` genau auf Objektivhoehe. Dann ist die Hoehendifferenz
        zur Kamera null, theta damit exakt 90 Grad und der Bildradius konstant
        f*pi/2 -- unabhaengig von der Entfernung. Es bleibt nur der Azimut, also
        eine feste Kreislinie im Bild.

        ``sample_depression_deg`` kippt den Ring nach unten. Aus der Ebene wird
        dann ein Kegel: in waagerechter Entfernung ``rho`` liegt er
        ``rho*tan(Winkel)`` unter der Linse -- fern also viel tiefer als nah.
        """
        if self.get_parameter('sample_mode').value != 'horizon':
            return self.get_parameter('sample_height_m').value

        depression = math.radians(self.get_parameter('sample_depression_deg').value)
        if depression == 0.0:
            return self.calib.cam_z
        return self.calib.cam_z - rho * math.tan(depression)

    def on_scan(self, msg: LaserScan):
        if self.latest_image is None:
            self.get_logger().warn('Noch kein Kamerabild empfangen.', throttle_duration_sec=5.0)
            return

        image, image_stamp = self.latest_image
        scan_stamp = _stamp_sec(msg.header.stamp)
        age = abs(scan_stamp - image_stamp)
        if age > self.get_parameter('max_sync_age_s').value:
            self.get_logger().warn(
                f'Bild ist {age:.2f} s vom Scan entfernt -- Zuordnung unsicher.',
                throttle_duration_sec=5.0)

        ranges = np.asarray(msg.ranges, dtype=float)
        r_min = max(float(msg.range_min), self.get_parameter('range_min_m').value)
        r_max = min(float(msg.range_max), self.get_parameter('range_max_m').value)
        keep = np.isfinite(ranges) & (ranges >= r_min) & (ranges <= r_max)
        # Verbaute Sektoren raus (Kabel, Elektronik) -- dort misst das Lidar nur
        # sich selbst und wuerde die Kamerafarbe des eigenen Aufbaus liefern.
        _, all_angles = scan_to_points(ranges, msg.angle_min, msg.angle_increment)
        keep &= visible_mask(all_angles, self.calib.lidar_blind_sectors_deg)
        if not keep.any():
            return

        idx = np.flatnonzero(keep)
        flat, angles = scan_to_points(ranges, msg.angle_min, msg.angle_increment)
        rho = np.hypot(flat[:, 0] - self.calib.cam_x, flat[:, 1] - self.calib.cam_y)
        pts, angles = scan_to_points(ranges, msg.angle_min, msg.angle_increment,
                                     self._sample_z(rho))
        pts, angles, rho = pts[keep], angles[keep], rho[keep]

        u, v, theta, phi, in_fov = project(self.calib, pts)
        height, width = image.shape[:2]
        on_image = in_fov & (u >= 0) & (u < width) & (v >= 0) & (v < height)
        if not on_image.any():
            self.get_logger().warn(
                'Kein Lidar-Punkt landet im Bild -- Kalibrierung pruefen.',
                throttle_duration_sec=5.0)
            return

        idx, pts, angles, rho = idx[on_image], pts[on_image], angles[on_image], rho[on_image]
        u, v, theta, phi = u[on_image], v[on_image], theta[on_image], phi[on_image]

        # Bandbreite in px: +-sample_band_m Pylonenhoehe, aus der Entfernung
        # umgerechnet. Fern schrumpft das Band von selbst mit, bleibt also
        # automatisch innerhalb der Pylone.
        band_m = self.get_parameter('sample_band_m').value
        band_px = None
        if band_m > 0.0:
            band_px = self.calib.focal_px * np.arctan(band_m / np.maximum(rho, 1e-3))

        bgr, hsv = colors.sample_colors(
            image, u, v, self.get_parameter('patch_px').value,
            center=(self.calib.cx, self.calib.cy), band_px=band_px,
            band_count=self.get_parameter('sample_band_count').value)
        labels = colors.classify_hsv(hsv, self.ranges)

        self._publish_summary(labels)
        debug = self.get_parameter('debug').value
        if debug and self.get_parameter('publish_cloud').value:
            self._publish_cloud(msg.header, pts, bgr)
        if debug and self.get_parameter('publish_debug_image').value:
            self._publish_debug(image, u, v, labels, bgr)

        self._write_csv(scan_stamp, idx, angles, np.linalg.norm(pts[:, :2], axis=1),
                        pts, u, v, theta, phi, bgr, hsv, labels)

    # ---------------------------------------------------------------- #
    def _publish_summary(self, labels):
        counts = {}
        for label in labels:
            counts[label] = counts.get(label, 0) + 1
        text = ' '.join(f'{k}={v}' for k, v in sorted(counts.items()))
        self.pub_summary.publish(String(data=f'{len(labels)} Punkte: {text}'))

    def _publish_cloud(self, header, pts, bgr):
        packed = ((bgr[:, 2].astype(np.uint32) << 16)
                  | (bgr[:, 1].astype(np.uint32) << 8)
                  | bgr[:, 0].astype(np.uint32))
        rgb = packed.view(np.float32)
        cloud_points = np.column_stack([pts.astype(np.float32), rgb]).tolist()
        self.pub_cloud.publish(point_cloud2.create_cloud(header, CLOUD_FIELDS, cloud_points))

    def _publish_debug(self, image, u, v, labels, bgr):
        now = self.get_clock().now().nanoseconds * 1e-9
        rate = self.get_parameter('debug_rate_hz').value
        if rate > 0 and now - self.last_debug_stamp < 1.0 / rate:
            return
        self.last_debug_stamp = now

        canvas = image.copy()
        center = (int(round(self.calib.cx)), int(round(self.calib.cy)))
        cv2.circle(canvas, center, int(round(self.calib.radius_px)), (255, 255, 0), 2)
        # Horizontring zur Kontrolle: liegt er auf Hoehe der Pylonen?
        ring = self.calib.focal_px * math.pi / 2.0
        cv2.circle(canvas, center, int(round(ring)), (0, 140, 255), 1)
        for pu, pv, label, color in zip(u, v, labels, bgr):
            point = (int(round(pu)), int(round(pv)))
            cv2.circle(canvas, point, 4, tuple(int(c) for c in color), -1)
            cv2.circle(canvas, point, 4, colors.LABEL_BGR.get(label, (255, 255, 255)), 1)

        # Wo liegt "vorne"? Hilft beim Beurteilen der Yaw-Kalibrierung.
        front_u, front_v, _, _, _ = project(self.calib, np.array([[1.0, 0.0, 0.0]]))
        cv2.arrowedLine(canvas,
                        (int(round(self.calib.cx)), int(round(self.calib.cy))),
                        (int(round(front_u[0])), int(round(front_v[0]))),
                        (0, 255, 255), 2, tipLength=0.08)
        cv2.putText(canvas, 'vorne (+X)',
                    (int(round(front_u[0])) + 6, int(round(front_v[0]))),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        out = self.bridge.cv2_to_imgmsg(canvas, 'bgr8')
        out.header.frame_id = 'camera'
        self.pub_debug.publish(out)

    # ---------------------------------------------------------------- #
    def _rows(self, stamp, idx, angles, dists, pts, u, v, theta, phi, bgr, hsv, labels):
        only_labeled = self.get_parameter('csv_only_labeled').value
        for i in range(len(idx)):
            if only_labeled and labels[i] in ('unbekannt', 'schwarz'):
                continue
            yield [
                f'{stamp:.6f}', int(idx[i]), f'{np.degrees(angles[i]):.3f}',
                f'{dists[i]:.4f}', f'{pts[i, 0]:.4f}', f'{pts[i, 1]:.4f}', f'{pts[i, 2]:.4f}',
                f'{u[i]:.2f}', f'{v[i]:.2f}', f'{np.degrees(theta[i]):.3f}',
                f'{np.degrees(phi[i]):.3f}',
                int(bgr[i, 0]), int(bgr[i, 1]), int(bgr[i, 2]),
                int(hsv[i, 0]), int(hsv[i, 1]), int(hsv[i, 2]), labels[i],
            ]

    def _write_csv(self, *args):
        mode = self.get_parameter('csv_mode').value
        if mode == 'continuous' and self.continuous_writer:
            handle, writer = self.continuous_writer
            writer.writerows(self._rows(*args))
            handle.flush()
            return
        if mode != 'trigger' or not self.capture_pending:
            return

        self.capture_pending = False
        os.makedirs(self.csv_dir, exist_ok=True)
        tag = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        path = os.path.join(self.csv_dir, f'lidar_pixels_{tag}.csv')
        rows = list(self._rows(*args))
        with open(path, 'w', newline='') as handle:
            writer = csv.writer(handle)
            writer.writerow(CSV_HEADER)
            writer.writerows(rows)
        # Kalibrierung mitschreiben, damit die CSV spaeter nachvollziehbar bleibt.
        self.calib.to_yaml(os.path.join(self.csv_dir, f'lidar_pixels_{tag}_calib.yaml'))
        self.get_logger().info(f'{len(rows)} Punkte geschrieben -> {path}')

    def _open_continuous_csv(self):
        os.makedirs(self.csv_dir, exist_ok=True)
        tag = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        path = os.path.join(self.csv_dir, f'lidar_pixels_{tag}_continuous.csv')
        handle = open(path, 'w', newline='')
        writer = csv.writer(handle)
        writer.writerow(CSV_HEADER)
        self.continuous_writer = (handle, writer)
        self.get_logger().info(f'Schreibe fortlaufend nach {path}')

    def destroy_node(self):
        if self.continuous_writer:
            self.continuous_writer[0].close()
            self.continuous_writer = None
        return super().destroy_node()


def _stamp_sec(stamp) -> float:
    return stamp.sec + stamp.nanosec * 1e-9


def _packaged_default() -> str:
    try:
        from ament_index_python.packages import get_package_share_directory
        return os.path.join(get_package_share_directory('camera_lidar_fusion'),
                            'config', 'fisheye_calib.yaml')
    except Exception:  # noqa: BLE001
        return ''


def main(args=None):
    rclpy.init(args=args)
    node = LidarPixelMapper()
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
