#!/usr/bin/env python3
"""M2 -- localisation quality: state timeline, wall matches, innovations.

    python3 plot_localization.py BAG [BAG ...] [--out-dir docs/figures]

Topics (scan_processor_node.py, ekf_node.py):
  /localization_state   String, latched, published on change: ok | recovering | lost
                        (gate level of the wall association, see _publish_loc_state)
  /wall_matches         robot_msgs/WallMatchArray, one message per LiDAR scan;
                        each WallMatch = measured wall (alpha_meas, d_meas; robot
                        frame, HNF) and the map wall it was matched to
                        (alpha_map, d_map; map frame)
  /ekf/odom             pose used to compute the innovation
Innovation (metrics.wall_innovations): measurement minus prediction from the
latest EKF pose, with the EKF measurement model of ekf.py update_wall().

Figures: localization_<bag> (state timeline + matched walls per scan) and
innovations_<bag> (histograms of d and alpha innovations).
Printed: time share per state, matches per scan, share of scans without a
match, innovation median / std / p95(|.|).
"""
import argparse
import math
import sys

import numpy as np

import bagio
import metrics
import style
from bagio import T

TOPICS = [T['loc_state'], T['wall_matches'], T['odom'], T['lap_state']]
STATES = ['ok', 'recovering', 'lost']


def run(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('bags', nargs='+')
    bagio.add_common_args(ap)
    a = ap.parse_args(argv)
    results = {}
    for r in bagio.iter_runs(a.bags, TOPICS, a.msg_dir):
        print(f'{r.name}:')
        res = {}
        cap = style.source_caption([r.name], 'plot_localization.py')
        fig, (a1, a2) = style.figure(2, 1, height=4.6, sharex=True,
                                     gridspec_kw={'height_ratios': [1, 2]})
        iv = metrics.state_intervals(r.get(T['loc_state']), r.duration_s)
        states = STATES + sorted(set(iv['state']) - set(STATES)) if len(iv) else STATES
        for i, st in enumerate(states):
            sel = iv[iv['state'] == st]
            if len(sel):
                a1.broken_barh(list(zip(sel['start'], sel['end'] - sel['start'])), (i - 0.35, 0.7),
                               facecolors=style.LOC_STATE_COLORS.get(st, style.MUTED),
                               edgecolor=style.SURFACE, linewidth=0.5)
        a1.set_yticks(range(len(states)))
        a1.set_yticklabels(states)
        a1.set_ylim(len(states) - 0.5, -0.5)
        a1.grid(axis='y', visible=False)
        a1.set_title('Localisation state (/localization_state)')
        if len(iv) == 0:
            style.no_data(a1, 'no /localization_state in this bag')
        fr, total = metrics.loc_fractions(r.get(T['loc_state']), r.duration_s)
        res['fractions'] = fr

        mps = metrics.matches_per_scan(r.get(T['wall_matches']))
        if len(mps):
            t, n = mps['t_bag'].to_numpy(float), mps['n_matches'].to_numpy(float)
            a2.plot(t, n, color=style.BLUE_RAMP[4], lw=0.8, drawstyle='steps-post', label='per scan')
            win = max(1, int(round(1.0 / max(np.median(np.diff(t)) if t.size > 1 else 0.1, 1e-3))))
            roll = np.convolve(n, np.ones(win) / win, mode='same')
            a2.plot(t, roll, color=style.CAT[0], label='1 s mean')
            a2.set_ylim(-0.2, max(3.5, n.max() + 0.5))
            style.legend_below(a2, ncol=2)
            res['matches_mean'] = float(n.mean())
            res['scans'] = int(n.size)
            res['scans_without_match_frac'] = float((n == 0).mean())
        else:
            style.no_data(a2, 'no /wall_matches in this bag')
        a2.set_ylabel('matched walls [-]')
        a2.set_title('Walls matched per LiDAR scan')
        a2.set_xlabel('time since bag start [s]')
        style.save(fig, a.out_dir, f'localization_{r.name}', cap)

        inn = metrics.wall_innovations(r.get(T['wall_matches']), r.get(T['odom']))
        fig, (h1, h2) = style.figure(1, 2, height=3.0)
        if len(inn):
            d_cm = inn['innov_d'].to_numpy(float) * 100
            a_deg = np.degrees(inn['innov_alpha'].to_numpy(float))
            for ax, v, xl, title, key in ((h1, d_cm, 'distance innovation [cm]', 'Wall distance', 'd'),
                                          (h2, a_deg, 'angle innovation [deg]', 'Wall angle', 'alpha')):
                lim = np.percentile(np.abs(v), 99.5) * 1.3 or 1.0
                style.hist(ax, np.clip(v, -lim, lim), bins=40)
                ax.set_xlim(-lim, lim)
                ax.set_xlabel(xl)
                ax.set_ylabel('count')
                ax.set_title(title)
                unit = ' cm' if key == 'd' else ' deg'
                med, std = float(np.median(v)), float(np.std(v))
                p95 = float(np.percentile(np.abs(v), 95))
                ax.text(0.98, 0.95, f'median {med:+.2f}{unit}\nstd {std:.2f}{unit}\n'
                        f'p95 |.| {p95:.2f}{unit}\nn {v.size}', transform=ax.transAxes, ha='right',
                        va='top', fontsize=7.5, color=style.INK_2, linespacing=1.5,
                        bbox=dict(boxstyle='round,pad=0.35', fc=style.SURFACE, ec=style.GRID, lw=0.6))
                res[f'innov_{key}'] = {'median': med, 'std': std, 'p95_abs': p95, 'n': int(v.size)}
        else:
            style.no_data(h1, 'needs /wall_matches and /ekf/odom')
            style.no_data(h2, 'needs /wall_matches and /ekf/odom')
        style.save(fig, a.out_dir, f'innovations_{r.name}', cap)

        if fr:
            print('  state share: ' + ', '.join(f'{k} {100 * v:.1f} %' for k, v in fr.items())
                  + f' (of {total:.1f} s)')
        if 'matches_mean' in res:
            print(f'  matched walls per scan: mean {res["matches_mean"]:.2f}, scans without match '
                  f'{100 * res["scans_without_match_frac"]:.1f} % (n={res["scans"]})')
        for key, unit in (('d', 'cm'), ('alpha', 'deg')):
            s = res.get(f'innov_{key}')
            if s:
                print(f'  innovation {key:5s}: median {s["median"]:+.3f} {unit}, std {s["std"]:.3f} {unit}, '
                      f'p95 |.| {s["p95_abs"]:.3f} {unit}')
        results[r.name] = res
    return results


def main(argv=None):
    run(argv)
    return 0


if __name__ == '__main__':
    sys.exit(main())
