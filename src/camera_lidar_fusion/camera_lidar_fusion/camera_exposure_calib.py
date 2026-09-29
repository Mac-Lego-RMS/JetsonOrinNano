#!/usr/bin/env python3
"""Calibrate the camera to the light on the field before the run.

The camera runs with fixed exposure, fixed gain and fixed white balance
(start_robot.sh). The values come from a room with daylight; under a
warm ceiling lamp the image had a strong yellow cast ((B-R)/max = -0.30 on the
white mat) and the colour detection did not find a single green point any more.

It measures on the white MAT, in the same ring just outside the wall band
that the white point of the fusion uses as well:

  1. Brightness: gain such that the mat sits at ~205 without blowing out.
     This first -- on a blown-out mat no colour cast can be measured.
  2. White balance: white_balance_temperature such that the mat is neutral,
     (B-R)/max ~ 0. The fusion subtracts the rest in (G-R) per sector anyway.
     Then adjust the brightness once more.
     The exposure time NEVER goes above 50 ms here -- above that the camera
     drops to 7.5 fps (see start_robot.sh). If the mat is already too bright
     at gain 0, the exposure is shortened.
  3. Check on the pylons: small, free-standing objects in colored_scan,
     and how many of their points get a colour.

Step 3 is the honest answer to "is the light enough". The mat lies flat
and gets the full ceiling light, the pylons show the camera their
vertical sides. Under pure ceiling light the mat was perfectly exposed at V 194
and the pylons were still almost black (V 25-40) -- no camera setting can
repair that, only more light. In that case the script says so.

The result goes to /workspace/config/camera_calib.env; start_robot.sh
reads it at the next start, so the calibration survives a restart.

Usage (in the container, with video_source and lidar_pixel_mapper running):
    python3 -m camera_lidar_fusion.camera_exposure_calib
    python3 -m camera_lidar_fusion.camera_exposure_calib --check-only
    python3 -m camera_lidar_fusion.camera_exposure_calib --no-save
"""
import argparse
import datetime
import os
import subprocess
import time

import numpy as np
import rclpy
import yaml
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image, PointCloud2
from sensor_msgs_py import point_cloud2

CALIB = '/workspace/config/fisheye_calib.yaml'
OUTPUT_FILE = '/workspace/config/camera_calib.env'
DEVICE = '/dev/picam'

# Limits from start_robot.sh: above 500 (50 ms) the camera drops to 7.5 fps.
EXPOSURE_MAX = 500
EXPOSURE_MIN = 100
GAIN_MAX = 60                  # above this the image blows out, eats saturation
WB_MIN, WB_MAX = 2800, 6500

MAT_TARGET = 205.0             # median of the brightest channel on the mat
MAT_TOL = 10.0
BLOWN_MAX = 0.05               # fraction of mat pixels with max >= 250
WB_TOL = 0.02                  # |(B-R)/max| on the mat

# Colours of the point cloud, exact (colors.CLOUD_BGR, packed as 0xRRGGBB)
CLOUD_RED = 0xFF0000
CLOUD_GREEN = 0x00FF00

# Pylon in the scan: small segment that stands in front of its surroundings
PYLON_WIDTH = (0.02, 0.09)
PYLON_DIST = (0.20, 1.60)
PYLON_STANDOUT = 0.10
PYLON_COLOUR_OK = 0.30         # this many points must have a colour
# Only count false colours from here on: the magenta bay walls reach up to ~0.3 m
# from the lidar and come out as red -- scan_processor filters that anyway
# (bay rule), it has nothing to do with the light.
FALSE_COLOUR_FROM = 0.35


def v4l2_set(**vals):
    args = ['v4l2-ctl', '-d', DEVICE]
    for k, v in vals.items():
        args += ['-c', f'{k}={int(v)}']
    subprocess.run(args, check=True, capture_output=True)


def v4l2_get(*names):
    out = subprocess.run(['v4l2-ctl', '-d', DEVICE, '--get-ctrl=' + ','.join(names)],
                         check=True, capture_output=True, text=True).stdout
    vals = {}
    for line in out.splitlines():
        k, _, v = line.partition(':')
        vals[k.strip()] = int(v.strip())
    return vals


class ExposureCalib(Node):
    def __init__(self):
        super().__init__('camera_exposure_calib')
        with open(CALIB) as f:
            c = yaml.safe_load(f)
        self.cx, self.cy = float(c['cx']), float(c['cy'])
        # Ring as colors.neutral_point in the fusion: from the bottom edge of the
        # wall band far away plus a margin, up to just before the image circle edge.
        self.r_min = float(c.get('zone_r0_out', 0.88 * c['radius_px'])) + 12.0
        self.r_max = float(c['radius_px']) - 15.0
        self.mask = None
        self.image = None
        self.image_t = 0.0
        self.clouds = []
        self.create_subscription(Image, '/video_source/raw', self._image_cb,
                                 qos_profile_sensor_data)
        self.create_subscription(PointCloud2, '/camera_lidar/colored_scan',
                                 self._cloud_cb, 10)

    # --------------------------------------------------------------- inputs
    def _image_cb(self, msg):
        if msg.encoding != 'bgr8':
            self.get_logger().error(f'Image format {msg.encoding}, expected bgr8')
            return
        self.image = np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.width, 3)
        self.image_t = time.monotonic()

    def _cloud_cb(self, msg):
        if self.clouds is not None:
            self.clouds.append(msg)

    def _wait(self, secs):
        end = time.monotonic() + secs
        while time.monotonic() < end:
            rclpy.spin_once(self, timeout_sec=0.05)

    # ------------------------------------------------------------------ mat
    def _ring(self, img_shape):
        if self.mask is None:
            h, w = img_shape[:2]
            yy, xx = np.mgrid[0:h:3, 0:w:3]
            r = np.hypot(xx - self.cx, yy - self.cy)
            m = (r >= self.r_min) & (r <= self.r_max)
            self.mask = (yy[m], xx[m])
        return self.mask

    def mat(self, n_images=5, settle=0.8):
        """(brightness, fraction blown out, (B-R)/max, (G-R)/max) of the mat,
        averaged over several fresh images after a change of a control value."""
        self._wait(settle)
        vals = []
        last = self.image_t
        while len(vals) < n_images:
            self._wait(0.05)
            if self.image is None or self.image_t == last:
                continue
            last = self.image_t
            ys, xs = self._ring(self.image.shape)
            px = self.image[ys, xs].astype(np.float32)
            b, g, r = px[:, 0], px[:, 1], px[:, 2]
            mx = np.maximum(np.maximum(b, g), r)
            lum = (b + g + r) / 3.0
            # Mat = the bright, weakly saturated pixels of the ring. NOT via
            # |G-R| < 30 as in colors.neutral_point: under cold light the mat is
            # turquoise (G far above R), dropped out there completely, and what
            # got measured were the dark leftovers -- mat "44", gain up to 59.
            # The saturation limit keeps magenta walls and pylons out,
            # but lets a colour cast of the mat through.
            mn = np.minimum(np.minimum(b, g), r)
            pale = (mx - mn) / np.maximum(mx, 1.0) < 0.40
            if pale.sum() < 200:
                continue
            m = pale & (lum >= np.percentile(lum[pale], 55.0))
            mx_m = np.maximum(mx[m], 1.0)
            vals.append((float(np.median(mx[m])), float(np.mean(mx[m] >= 250.0)),
                         float(np.mean((b[m] - r[m]) / mx_m)),
                         float(np.mean((g[m] - r[m]) / mx_m))))
        return tuple(float(np.median([w[i] for w in vals])) for i in range(4))

    # ------------------------------------------------------- control values
    def white_balance(self):
        """Bisection on the colour temperature. Higher = warmer (less blue):
        measured 4600 K -> (B-R)/max -0.30, 3000 K -> +0.07."""
        lo, hi = WB_MIN, WB_MAX
        t = v4l2_get('white_balance_temperature')['white_balance_temperature']
        for _ in range(8):
            v4l2_set(white_balance_temperature=t)
            _, _, br, _ = self.mat()
            self.get_logger().info(f'  white balance {t} K: (B-R)/max {br:+.3f}')
            if abs(br) <= WB_TOL:
                break
            if br < 0:          # too yellow -> set colder = smaller number
                hi = t
            else:
                lo = t
            t_new = int(round((lo + hi) / 2.0))
            if t_new == t:
                break
            t = t_new
        return t

    def brightness(self):
        """Gain first, and the exposure only if gain 0 is already too bright."""
        expo = min(v4l2_get('exposure_time_absolute')['exposure_time_absolute'],
                   EXPOSURE_MAX)
        v4l2_set(exposure_time_absolute=expo)

        def too_bright(v, blown):
            return v > MAT_TARGET + MAT_TOL or blown > BLOWN_MAX

        lo, hi = 0, GAIN_MAX
        gain = v4l2_get('gain')['gain']
        v = blown = 0.0
        for _ in range(8):
            v4l2_set(gain=gain)
            v, blown, _, _ = self.mat()
            self.get_logger().info(f'  gain {gain}: mat {v:.0f}, blown out {blown*100:.0f} %')
            if not too_bright(v, blown) and v >= MAT_TARGET - MAT_TOL:
                return expo, gain, v, blown
            if too_bright(v, blown):
                hi = gain
            else:
                lo = gain
            g_new = (lo + hi) // 2
            if g_new == gain:
                break
            gain = g_new

        if gain <= 1 and too_bright(v, blown):
            # Too bright even without gain: shorten the exposure.
            lo_b, hi_b = EXPOSURE_MIN, expo
            for _ in range(7):
                expo = (lo_b + hi_b) // 2
                v4l2_set(exposure_time_absolute=expo, gain=0)
                v, blown, _, _ = self.mat()
                self.get_logger().info(f'  exposure {expo/10:.0f} ms: mat {v:.0f}, '
                                       f'blown out {blown*100:.0f} %')
                if not too_bright(v, blown) and v >= MAT_TARGET - MAT_TOL:
                    break
                if too_bright(v, blown):
                    hi_b = expo
                else:
                    lo_b = expo
            return expo, 0, v, blown

        if not too_bright(v, blown) and v < MAT_TARGET - MAT_TOL and gain >= GAIN_MAX - 1:
            self.get_logger().warn(
                f'Too little light: mat at gain {GAIN_MAX} and {EXPOSURE_MAX/10:.0f} ms '
                f'only {v:.0f} (target {MAT_TARGET:.0f}). More light onto the field.')
        return expo, gain, v, blown

    # --------------------------------------------------------------- pylons
    def pylons(self, duration=3.0):
        """Small, free-standing objects in the point cloud and their colour."""
        self.clouds = []
        self._wait(duration)
        clouds, self.clouds = self.clouds, None
        finds = {}
        self.false_colour = 0       # coloured points that belong to no pylon
        for w in clouds:
            p = point_cloud2.read_points_numpy(w, field_names=('x', 'y', 'rgb'))
            if len(p) < 10:
                continue
            x, y = p[:, 0], p[:, 1]
            colour = p[:, 2].astype(np.float32).view(np.uint32) & 0xFFFFFF
            dist = np.hypot(x, y)
            order = np.argsort(np.arctan2(y, x))
            x, y, dist, colour = x[order], y[order], dist[order], colour[order]
            small = np.zeros(len(x), dtype=bool)     # segment as narrow as a pylon
            jump = np.hypot(np.diff(x), np.diff(y)) > 0.05
            bounds = np.concatenate([[0], np.nonzero(jump)[0] + 1, [len(x)]])
            for a, e in zip(bounds[:-1], bounds[1:]):
                if e - a < 3:
                    continue
                seg_w = float(np.hypot(x[e - 1] - x[a], y[e - 1] - y[a]))
                d = float(np.median(dist[a:e]))
                if seg_w <= PYLON_WIDTH[1] + 0.03:
                    small[a:e] = True
                if not (PYLON_WIDTH[0] <= seg_w <= PYLON_WIDTH[1]
                        and PYLON_DIST[0] <= d <= PYLON_DIST[1]):
                    continue
                # Free-standing: on at least one side the background is clearly
                # further away. Requiring both sides lost the pylon at the
                # rear diagonal, next to which a lidar blind sector lies.
                standout = any(dist[i] > d + PYLON_STANDOUT
                               for i in (a - 1, e % len(x)) if 0 <= i < len(x))
                if not standout:
                    continue
                cx, cy = float(np.mean(x[a:e])), float(np.mean(y[a:e]))
                cell = (round(cx / 0.08), round(cy / 0.08))
                f = finds.setdefault(cell, {'x': [], 'y': [], 'red': 0,
                                            'green': 0, 'n': 0, 'scans': 0})
                f['x'].append(cx)
                f['y'].append(cy)
                f['red'] += int(np.sum(colour[a:e] == CLOUD_RED))
                f['green'] += int(np.sum(colour[a:e] == CLOUD_GREEN))
                f['n'] += e - a
                f['scans'] += 1
            # Colour on WALLS (wide segments) is a false detection. Colour on
            # small objects is not -- even if the pylon search did not recognise
            # one as free-standing. The bay walls lie before
            # FALSE_COLOUR_FROM.
            stray = (~small & (dist >= FALSE_COLOUR_FROM) & (dist <= PYLON_DIST[1])
                     & ((colour == CLOUD_RED) | (colour == CLOUD_GREEN)))
            self.false_colour += int(stray.sum())
        self.false_colour_per_scan = self.false_colour / max(len(clouds), 1)
        n_scans = max(len(clouds), 1)
        # only what shows up in at least half of the scans
        return [f for f in finds.values() if f['scans'] >= 0.5 * n_scans], len(clouds)

    def gain_for_pylons(self, gain_mat):
        """Gain by the pylons instead of by the mat.

        The mat lies flat in the light, the pylons show the camera their
        vertical sides. Measured on the setup under cold ceiling light: mat
        regulated to 206 (gain 13) -> near pylon V 54, green index z +0.35,
        but too dark, 1 % of the points coloured. With gain 59 and a blown-out
        mat: 53 %. So the colour is there, brightness is missing -- and the
        mat may blow out for it, the colour detection does not need it.

        Tries a few steps from gain_mat to GAIN_MAX and takes the
        smallest that brings almost as many coloured points as the best one.
        None if no pylons can be seen -- then gain_mat stays."""
        steps = sorted({int(round(g)) for g in np.linspace(gain_mat, GAIN_MAX, 5)})
        results = []
        for g in steps:
            v4l2_set(gain=g)
            self._wait(1.0)
            finds, _ = self.pylons(2.0)
            if not finds:
                if g == steps[0]:
                    return None
                continue
            fracs = [(f['red'] + f['green']) / max(f['n'], 1) for f in finds]
            results.append((g, float(np.mean(fracs)), float(np.min(fracs))))
            self.get_logger().info(
                f'  gain {g}: {len(finds)} pylon(s), on average {np.mean(fracs)*100:.0f} % '
                f'found coloured, weakest {np.min(fracs)*100:.0f} %')
        if not results:
            return None
        best = max(e[1] for e in results)
        g = min(e[0] for e in results if e[1] >= best - 0.05)
        v4l2_set(gain=g)
        return g


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--check-only', action='store_true',
                    help='change nothing, only measure the mat and the pylons')
    ap.add_argument('--no-save', action='store_true',
                    help='do not write the values to %s' % OUTPUT_FILE)
    args = ap.parse_args()

    rclpy.init()
    node = ExposureCalib()
    log = node.get_logger()
    try:
        end = time.monotonic() + 5.0
        while node.image is None and time.monotonic() < end:
            node._wait(0.2)
        if node.image is None:
            log.error('No image on /video_source/raw -- is video_source running?')
            return 1
        v4l2_set(auto_exposure=1, white_balance_automatic=0)
        before = v4l2_get('exposure_time_absolute', 'gain', 'white_balance_temperature')
        v0, blown0, br0, gr0 = node.mat()
        log.info(f'Before: exposure {before["exposure_time_absolute"]/10:.0f} ms, '
                 f'gain {before["gain"]}, {before["white_balance_temperature"]} K -> '
                 f'mat {v0:.0f} ({blown0*100:.0f} % area blown out), '
                 f'(B-R)/max {br0:+.3f}, (G-R)/max {gr0:+.3f}')

        if args.check_only:
            expo, gain, wb = (before['exposure_time_absolute'], before['gain'],
                              before['white_balance_temperature'])
            v, blown, br, gr = v0, blown0, br0, gr0
        else:
            # Brightness first: on a blown-out mat all channels hit
            # 255, and the white balance cannot be measured. Then once
            # more, because the white balance shifts the channels.
            log.info('Brightness ...')
            node.brightness()
            log.info('White balance ...')
            wb = node.white_balance()
            log.info('Brightness ...')
            expo, gain, v, blown = node.brightness()
            log.info('Gain by the pylons ...')
            g_pyl = node.gain_for_pylons(gain)
            if g_pyl is None:
                log.info('  no pylons in sight -- gain stays as set by the mat.')
                v4l2_set(gain=gain)
            else:
                gain = g_pyl
            v, blown, br, gr = node.mat()
            log.info(f'After: exposure {expo/10:.0f} ms, gain {gain}, {wb} K -> '
                     f'mat {v:.0f} ({blown*100:.0f} % area blown out), '
                     f'(B-R)/max {br:+.3f}, (G-R)/max {gr:+.3f}')

        log.info('Checking pylons (3 s colored_scan) ...')
        finds, n = node.pylons()
        weak = 0
        if n == 0:
            log.warn('No /camera_lidar/colored_scan received -- is the fusion running?')
        elif not finds:
            log.info('No pylons found within 1.6 m -- colour check skipped.')
        for f in sorted(finds, key=lambda f: np.hypot(np.mean(f['x']), np.mean(f['y']))):
            x, y = float(np.mean(f['x'])), float(np.mean(f['y']))
            frac = (f['red'] + f['green']) / max(f['n'], 1)
            colour = ('green' if f['green'] > f['red'] else 'red') if frac > 0 else '-'
            # The lidar is mounted turned: robot front = lidar -x
            text = (f'  Pylon {np.hypot(x, y):.2f} m, {np.degrees(np.arctan2(-y, -x)):+4.0f} deg: '
                    f'{frac*100:3.0f} % detected as coloured ({colour}; red {f["red"]}, '
                    f'green {f["green"]}, total {f["n"]})')
            if frac >= PYLON_COLOUR_OK:
                log.info(text)
            else:
                weak += 1
                log.warn(text + ' -- too weak')
        if n:
            (log.warn if node.false_colour_per_scan >= 3 else log.info)(
                f'  Colour on walls (false hits): {node.false_colour_per_scan:.1f} points per scan')
        if weak:
            log.warn(f'{weak} pylon(s) hardly get any colour. If the mat is on target, '
                     f'there is not enough light on the SIDES of the pylons (ceiling light alone '
                     f'is not enough) -- no camera setting can make up for that.')

        if not args.check_only and not args.no_save:
            stamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            with open(OUTPUT_FILE, 'w') as fh:
                fh.write(f'# camera_exposure_calib, {stamp}\n'
                         f'# mat {v:.0f} ({blown*100:.0f} % area blown out), '
                         f'(B-R)/max {br:+.3f}, (G-R)/max {gr:+.3f}\n'
                         f'CAM_EXPOSURE={expo}\nCAM_GAIN={gain}\nCAM_WB_TEMP={wb}\n')
            log.info(f'Saved -> {OUTPUT_FILE} (start_robot.sh reads it at start-up)')
        return 0
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    raise SystemExit(main())
