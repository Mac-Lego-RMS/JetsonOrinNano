#!/usr/bin/env python3
"""M12 -- steering characteristic from steer_calib.json (no bag needed).

    python3 plot_steer_lut.py [--calib src/esp_bridge/esp_bridge/steer_calib.json]

The JSON (read by src/esp_bridge/esp_bridge/steer_lut.py) holds, per speed v,
measured points [servo, delta] separately for the left (servo > 0, delta > 0)
and the right side (servo < 0, delta < 0):
  servo  normalised servo command in [-1, 1] (the bridge sends servo x 100 %)
  delta  steering (wheel) angle in rad, measured at that speed
  wheelbase L [m]; curvature kappa = tan(delta) / L (Ackermann, as in the
  bridge: delta = atan(L * omega / v))
At run time SteerLUT inverts delta -> servo per side and interpolates
linearly between the calibrated speeds (clamped outside).

Figure steer_lut: (a) wheel angle vs servo command per speed, (b) curvature vs
servo command, (c) deviation from a straight line through the origin fitted
per side (the nonlinearity). Printed: gain per side and speed [deg per 10 %
servo], max |deviation|, left/right asymmetry at full lock.
"""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

import bagio
import style

DEFAULT_CALIB = bagio.REPO / 'src' / 'esp_bridge' / 'esp_bridge' / 'steer_calib.json'


def load_calib(path):
    data = json.loads(Path(path).read_text())
    L = float(data.get('wheelbase', 0.10))
    speeds = []
    for e in sorted(data['speeds'], key=lambda s: s['v']):
        sides = {}
        for side in ('left', 'right'):
            pts = np.array(sorted(e[side], key=lambda p: p[0]), dtype=float)
            sides[side] = pts
        speeds.append((float(e['v']), sides))
    return L, speeds


def side_fit(pts, centre):
    """Least-squares slope of delta vs (servo - servo_centre) through the
    trim point (the JSON centre point has delta = 0 at servo = centre)."""
    s, d = pts[:, 0] - centre, pts[:, 1]
    k = float(np.dot(s, d) / np.dot(s, s)) if np.dot(s, s) > 0 else math.nan
    return k, d - k * s


def run(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--calib', default=str(DEFAULT_CALIB))
    ap.add_argument('--out-dir', default=str(bagio.DEFAULT_FIG_DIR))
    a = ap.parse_args(argv)
    L, speeds = load_calib(a.calib)
    fig, (a1, a2, a3) = style.figure(1, 3, width=9.0, height=3.4)
    res = {'wheelbase': L, 'speeds': {}}
    for i, (v, sides) in enumerate(speeds):
        col = style.CAT[i % len(style.CAT)]
        pts = np.vstack([sides['right'], sides['left']])
        o = np.argsort(pts[:, 0])
        s, d = pts[o, 0], pts[o, 1]
        lbl = f'v = {v:.2f} m/s'
        a1.plot(s * 100, np.degrees(d), '-o', color=col, ms=4.5, mec=style.SURFACE, mew=1.2, label=lbl)
        a2.plot(s * 100, np.tan(d) / L, '-o', color=col, ms=4.5, mec=style.SURFACE, mew=1.2, label=lbl)
        zero = sides['left'][np.argmin(np.abs(sides['left'][:, 1])), 0]
        rv = {}
        for side in ('left', 'right'):
            k, resid = side_fit(sides[side], zero)
            ps = sides[side]
            a3.plot(ps[:, 0] * 100, np.degrees(resid), '-o', color=col, ms=4.5, mec=style.SURFACE,
                    mew=1.2, label=lbl if side == 'left' else None)
            rv[side] = {'deg_per_10pct': math.degrees(k) * 0.1, 'max_dev_deg': float(np.degrees(np.max(np.abs(resid)))),
                        'full_lock_deg': float(np.degrees(ps[np.argmax(np.abs(ps[:, 0])), 1]))}
        rv['asymmetry_full_lock_deg'] = abs(rv['left']['full_lock_deg']) - abs(rv['right']['full_lock_deg'])
        rv['servo_centre'] = float(zero)
        res['speeds'][v] = rv
    for ax in (a1, a2, a3):
        ax.axhline(0, color=style.AXIS, lw=0.8)
        ax.axvline(0, color=style.AXIS, lw=0.8)
        ax.set_xlabel('servo command [%]  (+ = left)')
    a1.set_ylabel('wheel angle delta [deg]')
    a1.set_title('Steering angle')
    a2.set_ylabel('curvature tan(delta)/L [1/m]')
    a2.set_title('Curvature')
    a3.set_ylabel('deviation from linear fit [deg]')
    a3.set_title('Nonlinearity per side')
    style.legend_below(a2, ncol=len(speeds))
    style.save(fig, a.out_dir, 'steer_lut', f'Source: {Path(a.calib).name} (L = {L:.3f} m)  |  plot_steer_lut.py')
    print(f'steer calibration {a.calib} (wheelbase {L} m):')
    for v, rv in res['speeds'].items():
        print(f'  v = {v:.2f} m/s: centre servo {rv["servo_centre"] * 100:+.1f} %')
        for side in ('left', 'right'):
            q = rv[side]
            print(f'    {side:5s} gain {q["deg_per_10pct"]:.2f} deg per 10 % servo, max deviation from linear '
                  f'{q["max_dev_deg"]:.2f} deg, full lock {q["full_lock_deg"]:+.1f} deg')
        print(f'    |left| - |right| at full lock: {rv["asymmetry_full_lock_deg"]:+.2f} deg')
    return res


def main(argv=None):
    run(argv)
    return 0


if __name__ == '__main__':
    sys.exit(main())
