#!/usr/bin/env python3
"""Top-down field map with the EKF trajectory (replaces an overhead camera).

    python3 plot_trajectory.py BAG [BAG ...] [--start-pose cw_pos1] [--out-dir ...]

Drawn in the start-anchored MAP frame (the frame of /ekf/odom and /obstacles),
metres, 1:1 aspect:
  * outer 3 x 3 m wall and inner 1 x 1 m wall, and the 24 obstacle seats
    (geometry from src/ekf/ekf/field_map.py, copied in robot_constants.py);
    placed with /corner_geometry (outer corners in the map frame) when the bag
    has it, else with --start-pose (cw_pos1, cw_pos2, ccw_pos1, ccw_pos2),
    else guessed from the bag name (e.g. cw_pos1_3) and /race_direction;
  * detected obstacles from the LAST /obstacles message: red squares / green
    triangles (colour + shape), drawn to scale (44 mm);
  * the EKF trajectory (/ekf/odom x, y) coloured by speed (twist.linear.x).
    NOTE: in a bag that starts in the parking bay the start-anchored frame is
    only defined after the start detection -- without /corner_geometry the
    field outline may be misplaced.

Figures: trajectory_<bag> (all laps) and trajectory_laps_<bag> (one panel per
lap, from /round1_controller/lap_state).
Printed: path length, mean / max speed, lap times, obstacles.
"""
import argparse
import math
import re
import sys

import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.patches import Rectangle

import bagio
import metrics
import style
from bagio import T
from robot_constants import (INNER_HALF, OUTER_HALF, PILLAR_EDGE, START_POSES, seats_field, square,
                             transform_from_corners, transform_from_start_pose)

TOPICS = [T['odom'], T['obstacles'], T['corner_geometry'], T['race_direction'], T['lap_state']]


def field_transform(run, start_pose=None):
    """(function field->map, description)."""
    cg = run.get(T['corner_geometry'])
    if start_pose:
        return transform_from_start_pose(START_POSES[start_pose]), f'start pose {start_pose}'
    if len(cg) and 'corner0_x' in cg:
        row = cg.iloc[-1]
        corners = [(row[f'corner{i}_x'], row[f'corner{i}_y']) for i in range(4)]
        return transform_from_corners(corners), '/corner_geometry'
    m = re.search(r'(ccw|cw)_?(pos[12])', run.name.lower())
    if m:
        key = f'{m.group(1)}_{m.group(2)}'
        return transform_from_start_pose(START_POSES[key]), f'start pose {key} (from bag name)'
    d = str(bagio.last_value(run.get(T['race_direction']), default='CW')).strip().lower()
    key = f'{"ccw" if d == "ccw" else "cw"}_pos1'
    bagio.warn_once(f'{run.name}: no /corner_geometry -- field drawn for {key} (use --start-pose)')
    return transform_from_start_pose(START_POSES[key]), f'start pose {key} (assumed!)'


def draw_field(ax, to_map, with_seats=True):
    for half in (OUTER_HALF, INNER_HALF):
        p = to_map(square(half))
        ax.plot(p[:, 0], p[:, 1], color=style.INK, lw=1.6, solid_capstyle='butt', zorder=2)
    if with_seats:
        s = to_map(np.array([q['p'] for q in seats_field()]))
        ax.scatter(s[:, 0], s[:, 1], s=22, facecolors='none', edgecolors=style.AXIS, linewidths=0.8,
                   zorder=2, label='obstacle seat')


def draw_obstacles(ax, obs):
    """obs: rows of the last /obstacles message."""
    for name, code in (('red', 1), ('green', 2), ('unknown', 0)):
        sel = obs[obs['color'] == code] if len(obs) else obs
        if not len(sel):
            continue
        st = style.PILLAR[name]
        for x, y in zip(sel['x'], sel['y']):
            ax.add_patch(Rectangle((x - PILLAR_EDGE / 2, y - PILLAR_EDGE / 2), PILLAR_EDGE, PILLAR_EDGE,
                                   facecolor=st['color'], edgecolor='none', zorder=4))
        ax.scatter(sel['x'], sel['y'], s=70, marker=st['marker'], facecolors='none',
                   edgecolors=st['color'], linewidths=1.4, zorder=5, label=st['label'])


def speed_line(ax, x, y, v, vmax):
    pts = np.column_stack([x, y]).reshape(-1, 1, 2)
    segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
    lc = LineCollection(segs, cmap=style.SEQ_CMAP, norm=style.plt.Normalize(0, vmax), lw=1.8,
                        capstyle='round', zorder=3)
    lc.set_array(0.5 * (v[:-1] + v[1:]))
    ax.add_collection(lc)
    return lc


def setup_axes(ax, to_map):
    c = to_map(square(OUTER_HALF + 0.12))
    ax.set_xlim(c[:, 0].min(), c[:, 0].max())
    ax.set_ylim(c[:, 1].min(), c[:, 1].max())
    ax.set_aspect('equal')
    ax.set_xlabel('x (map frame) [m]')
    ax.set_ylabel('y (map frame) [m]')


def lap_times(odom, lap_df):
    """Start = first EKF speed > 0.05 m/s; lap k ends when corner_count
    reaches 4k (lap_state data_1)."""
    if not len(odom) or not len(lap_df) or 'data_1' not in lap_df:
        return []
    mv = odom[odom['v'].abs() > 0.05]
    if not len(mv):
        return []
    t0 = float(mv['t_bag'].iloc[0])
    d = lap_df.sort_values('t_bag')
    out, prev = [], t0
    for k in range(1, int(d['data_1'].max()) // 4 + 1):
        t_end = float(d.loc[d['data_1'] >= 4 * k, 't_bag'].iloc[0])
        out.append(t_end - prev)
        prev = t_end
    return out


def run(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('bags', nargs='+')
    ap.add_argument('--start-pose', choices=sorted(START_POSES), default=None,
                    help='force the field placement (default: /corner_geometry)')
    ap.add_argument('--vmax', type=float, default=None, help='colour scale max [m/s]')
    bagio.add_common_args(ap)
    a = ap.parse_args(argv)
    results = {}
    for r in bagio.iter_runs(a.bags, TOPICS, a.msg_dir):
        print(f'{r.name}:')
        odom = r.get(T['odom'])
        if not len(odom):
            print('  no /ekf/odom -- skipped')
            continue
        to_map, how = field_transform(r, a.start_pose)
        obs_all = r.get(T['obstacles'])
        obs = obs_all[obs_all['msg_index'] == obs_all['msg_index'].max()] if len(obs_all) else obs_all
        if len(obs) and 'obst_idx' in obs:
            obs = obs[obs['obst_idx'] >= 0]
        x, y, v = (odom[c].to_numpy(float) for c in ('x', 'y', 'v'))
        vmax = a.vmax or max(0.1, float(np.nanpercentile(np.abs(v), 99.5)))
        cap = style.source_caption([r.name], f'plot_trajectory.py (field placed by {how})')

        fig, ax = style.figure(1, 1, width=6.2, height=5.8)
        draw_field(ax, to_map)
        draw_obstacles(ax, obs)
        lc = speed_line(ax, x, y, np.abs(v), vmax)
        ax.plot([x[0]], [y[0]], 'o', color=style.INK, ms=6, mec=style.SURFACE, mew=2, zorder=6,
                label='start of bag')
        setup_axes(ax, to_map)
        ax.grid(False)
        cb = fig.colorbar(lc, ax=ax, fraction=0.04, pad=0.02)
        cb.set_label('speed [m/s]', color=style.INK_2)
        cb.outline.set_visible(False)
        ax.set_title('EKF trajectory on the field')
        style.legend_below(ax, ncol=4)
        style.save(fig, a.out_dir, f'trajectory_{r.name}', cap)

        lap = r.get(T['lap_state'])
        laps = metrics.lap_of(odom['t_bag'], lap)
        uniq = [k for k in sorted(set(laps)) if k >= 0][:4]
        if len(uniq) > 1:
            n = len(uniq)
            fig, axs = style.figure(1, n, width=min(9.0, 2.6 * n), height=3.2)
            axs = np.atleast_1d(axs)
            for axl, k in zip(axs, uniq):
                sel = laps == k
                draw_field(axl, to_map, with_seats=False)
                draw_obstacles(axl, obs)
                speed_line(axl, x[sel], y[sel], np.abs(v[sel]), vmax)
                setup_axes(axl, to_map)
                axl.grid(False)
                axl.set_title(f'lap {k + 1}' if k < 3 else 'after lap 3')
                axl.tick_params(labelsize=7)
            for axl in axs[1:]:
                axl.set_ylabel('')
            style.save(fig, a.out_dir, f'trajectory_laps_{r.name}', cap)

        length = float(np.sum(np.hypot(np.diff(x), np.diff(y))))
        moving = np.abs(v) > 0.05
        res = {'path_length_m': length, 'v_mean_moving': float(np.mean(np.abs(v[moving]))) if moving.any()
               else math.nan, 'v_max': float(np.max(np.abs(v))), 'lap_times_s': lap_times(odom, lap),
               'obstacles': [(int(i), c) for i, c in zip(obs.get('id', []), obs.get('color_name', []))],
               'field_placement': how}
        print(f'  field placed by {how}')
        print(f'  path length {length:.2f} m, speed mean (moving) {res["v_mean_moving"]:.2f} m/s, '
              f'max {res["v_max"]:.2f} m/s')
        if res['lap_times_s']:
            print('  lap times: ' + ', '.join(f'{t:.2f} s' for t in res['lap_times_s']))
        if res['obstacles']:
            print('  obstacles (seat id, colour): ' + ', '.join(f'{i} {c}' for i, c in res['obstacles']))
        results[r.name] = res
    return results


def main(argv=None):
    run(argv)
    return 0


if __name__ == '__main__':
    sys.exit(main())
