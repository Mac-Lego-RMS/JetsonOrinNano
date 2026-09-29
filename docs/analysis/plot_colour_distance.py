#!/usr/bin/env python3
"""M4 -- pillar colour classification rate vs distance (camera-LiDAR fusion).

    python3 plot_colour_distance.py BAG [BAG ...] [--gate 0.08] [--max-clouds N]

Needs /camera_lidar/colored_scan (sensor_msgs/PointCloud2 from
lidar_pixel_mapper.py; fields x, y, z, rgb float32). In cloud_color_mode
'label' (the default) every point's rgb is an exact label colour
(colors.py CLOUD_BGR): 0xFF0000 red, 0x00FF00 green, 0xFF00FF magenta,
0x2D2D2D black, 0x555555 unknown. In 'raw' mode the rgb is the camera pixel
and cannot be decoded -- then all points count as unclassified.

Ground truth: each point is transformed LiDAR frame -> base_link (the LiDAR is
mounted turned by 180 deg: x_b = -x + 0.1101, y_b = -y, obstacle_detection.py)
-> map frame with the /ekf/odom pose at the cloud's header stamp, and assigned
to the nearest pillar of the LAST /obstacles message within --gate metres.
The pillar's colour from /obstacles is taken as the truth. This is only as
good as the final obstacle map and the EKF pose; for a clean measurement
place known pillars and check /obstacles in the bag.
Range = distance from the LiDAR, sqrt(x^2 + y^2) in the cloud frame.

Figure colour_distance_<bag>: per true colour, share of pillar points
classified correctly / as the other colour / not classified, per range bin.
Without /obstacles only the label shares of all points are plotted.
"""
import argparse
import math
import sys

import numpy as np
import pandas as pd

import bagio
import style
from bagio import T
from robot_constants import LIDAR_OFFSET_X

TOPICS = [T['cloud'], T['odom'], T['obstacles']]
OUTCOMES = [('correct', style.CAT[0]), ('other colour', style.CAT[1]), ('not classified', style.MUTED)]


def pose_at(odom, t_query, col):
    o = odom.sort_values(col)
    t = o[col].to_numpy(float)
    i = np.clip(np.searchsorted(t, t_query, side='right') - 1, 0, len(t) - 1)
    return (o[c].to_numpy(float)[i] for c in ('x', 'y', 'yaw'))


def classify(run, gate):
    cloud = run.get(T['cloud'])
    odom = run.get(T['odom'])
    obs = run.get(T['obstacles'])
    if len(obs):
        obs = obs[(obs['msg_index'] == obs['msg_index'].max()) & (obs['obst_idx'] >= 0)]
    use_hdr = cloud['t_header'].notna().all() and odom['t_header'].notna().all()
    tcol = 't_header' if use_hdr else 't_bag'
    xo, yo, th = pose_at(odom, cloud[tcol].to_numpy(float), tcol)
    xl, yl = cloud['x'].to_numpy(float), cloud['y'].to_numpy(float)
    ok = np.isfinite(xl) & np.isfinite(yl)
    xb, yb = -xl + LIDAR_OFFSET_X, -yl
    c, s = np.cos(th), np.sin(th)
    X, Y = xo + c * xb - s * yb, yo + s * xb + c * yb
    df = pd.DataFrame({'range': np.hypot(xl, yl), 'label': cloud.get('label', pd.Series(['other'] * len(cloud))),
                       'X': X, 'Y': Y})[ok]
    if not len(obs):
        return df, None
    P = obs[['x', 'y']].to_numpy(float)
    d = np.hypot(df['X'].to_numpy()[:, None] - P[None, :, 0], df['Y'].to_numpy()[:, None] - P[None, :, 1])
    k = np.argmin(d, axis=1)
    near = d[np.arange(len(k)), k] <= gate
    pil = df[near].copy()
    pil['truth'] = obs['color_name'].to_numpy()[k[near]]
    pil = pil[pil['truth'].isin(['red', 'green'])]
    other = {'red': 'green', 'green': 'red'}
    pil['outcome'] = np.where(pil['label'] == pil['truth'], 'correct',
                              np.where(pil['label'] == pil['truth'].map(other), 'other colour',
                                       'not classified'))
    return df, pil


def run(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('bags', nargs='+')
    ap.add_argument('--gate', type=float, default=0.08, help='association radius to a pillar [m]')
    ap.add_argument('--bin', type=float, default=0.25, help='range bin width [m]')
    ap.add_argument('--max-range', type=float, default=3.0)
    ap.add_argument('--max-clouds', type=int, default=None, help='use only the first N clouds')
    bagio.add_common_args(ap)
    a = ap.parse_args(argv)
    results = {}
    for r in bagio.iter_runs(a.bags, TOPICS, a.msg_dir, heavy=True, max_cloud_msgs=a.max_clouds):
        print(f'{r.name}:')
        if not r.has(T['cloud']):
            print('  no /camera_lidar/colored_scan in this bag -- M4 needs it (record it, or run the '
                  'fusion node while replaying). Skipped.')
            results[r.name] = None
            continue
        if not r.has(T['odom']):
            print('  no /ekf/odom -- cannot place the points on the field. Skipped.')
            results[r.name] = None
            continue
        pts, pil = classify(r, a.gate)
        bins = np.arange(0.0, a.max_range + a.bin, a.bin)
        centres = 0.5 * (bins[:-1] + bins[1:])
        cap = style.source_caption([r.name], 'plot_colour_distance.py')
        res = {}
        if pil is not None and len(pil):
            fig, axs = style.figure(1, 2, height=3.2, sharey=True)
            for ax, truth in zip(axs, ('red', 'green')):
                g = pil[pil['truth'] == truth]
                g = g.assign(b=pd.cut(g['range'], bins, labels=False))
                n = g.groupby('b').size().reindex(range(len(centres)), fill_value=0).to_numpy()
                tab = {'range_m': centres, 'n': n}
                for oc, col in OUTCOMES:
                    k = g[g['outcome'] == oc].groupby('b').size().reindex(range(len(centres)),
                                                                          fill_value=0).to_numpy()
                    frac = np.where(n >= 5, k / np.maximum(n, 1), np.nan)
                    tab[oc] = frac
                    ax.plot(centres, frac * 100, '-o', color=col, ms=5, mec=style.SURFACE, mew=1.2, label=oc)
                ax.set_title(f'True {truth} pillars ({len(g)} points)')
                ax.set_xlabel('range from the LiDAR [m]')
                ax.set_xlim(0, a.max_range)
                ax.set_ylim(-3, 103)
                res[truth] = pd.DataFrame(tab)
            axs[0].set_ylabel('share of pillar points [%]')
            style.legend_below(axs[0], ncol=3)
            style.save(fig, a.out_dir, f'colour_distance_{r.name}', cap)
            for truth, tab in res.items():
                print(f'  true {truth}: range bin -> correct / other colour / not classified (n)')
                for _, row in tab.iterrows():
                    if row['n']:
                        vals = ' / '.join(f'{100 * np.nan_to_num(row[oc]):5.1f} %' for oc, _ in OUTCOMES)
                        print(f'    {row["range_m"]:4.2f} m: {vals}  (n={int(row["n"])})')
        else:
            if pil is None:
                print('  no /obstacles -- no ground truth; plotting label shares of all points')
            else:
                print('  no cloud point within the gate of a known pillar; plotting label shares')
            fig, ax = style.figure(1, 1, height=3.2)
            pts = pts.assign(b=pd.cut(pts['range'], bins, labels=False))
            n = pts.groupby('b').size().reindex(range(len(centres)), fill_value=0).to_numpy()
            for lab, col in (('red', style.PILLAR['red']['color']), ('green', style.PILLAR['green']['color']),
                             ('unknown', style.MUTED), ('black', style.INK_2)):
                k = pts[pts['label'] == lab].groupby('b').size().reindex(range(len(centres)),
                                                                          fill_value=0).to_numpy()
                ax.plot(centres, np.where(n > 0, 100 * k / np.maximum(n, 1), np.nan), '-o', color=col,
                        ms=5, mec=style.SURFACE, mew=1.2, label=lab)
            ax.set_xlabel('range from the LiDAR [m]')
            ax.set_ylabel('share of all cloud points [%]')
            ax.set_title('Cloud labels vs range (no pillar ground truth)')
            style.legend_below(ax, ncol=4)
            style.save(fig, a.out_dir, f'colour_distance_{r.name}', cap)
        results[r.name] = res
    return results


def main(argv=None):
    run(argv)
    return 0


if __name__ == '__main__':
    sys.exit(main())
