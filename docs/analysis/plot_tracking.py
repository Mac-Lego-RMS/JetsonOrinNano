#!/usr/bin/env python3
"""M9 -- path tracking errors of the round-1 controller.

    python3 plot_tracking.py BAG [BAG ...] [--out-dir docs/figures]

Topics (round1_controller_node.py, published every 30 Hz control tick):
  /round1_controller/dbg/e_ct         Float64 [m]   cross-track error
      TURN: distance to the planned circle minus R (>0 = outside, too wide)
      DRIVE (Stanley): offset LEFT of the target line -- but this publish is
      commented out in the current source, so straights usually have no e_ct
  /round1_controller/dbg/e_theta_deg  Float64 [deg] heading error (both phases)
  /round1_controller/dbg/k_h_eff      Float64       only on straights (Stanley)
  /round1_controller/dbg/arc_R        Float64 [m]   only in corners (TURN)
  /round1_controller/lap_state        Int32MultiArray [corner_idx, corner_count, lap]
Phase of each sample: metrics.tracking_samples() (same-tick arc_R -> corner,
same-tick k_h_eff -> straight).

Figures: tracking_<bag> (time series, lap boundaries marked) and
tracking_hist_<bag> (distributions per phase).
Printed: RMS / max per phase, and RMS per lap.
"""
import argparse
import sys

import numpy as np
import pandas as pd

import bagio
import metrics
import style
from bagio import T

TOPICS = [T['e_ct'], T['e_theta'], T['k_h'], T['arc_R'], T['lap_state']]
PHASE_STYLE = {'straight': (style.CAT[0], 'straight (Stanley)'),
               'arc': (style.CAT[1], 'corner (arc tracking)'),
               'unknown': (style.MUTED, 'phase unknown')}


def plot_segments(ax, t, v, color, label, gap=0.1):
    """Line plot that breaks where samples are further apart than gap."""
    t, v = np.asarray(t, float), np.asarray(v, float)
    if t.size == 0:
        return
    breaks = np.where(np.diff(t) > gap)[0] + 1
    first = True
    for seg in np.split(np.arange(t.size), breaks):
        ax.plot(t[seg], v[seg], color=color, lw=1.2, label=label if first else None)
        first = False


def lap_marks(ax, lap_df):
    if lap_df is None or len(lap_df) == 0 or 'data_2' not in lap_df:
        return
    d = lap_df.sort_values('t_bag')
    change = d[d['data_2'].diff().fillna(0) > 0]
    for t, lap in zip(change['t_bag'], change['data_2']):
        ax.axvline(t, color=style.AXIS, lw=0.9, zorder=1)
        ax.text(t, 1.0, f'lap {int(lap) + 1}', transform=ax.get_xaxis_transform(),
                fontsize=6.5, color=style.INK_2, ha='center', va='bottom')


def per_lap(samples, lap_df, scale):
    if len(samples) == 0:
        return pd.DataFrame()
    s = samples.copy()
    s['lap'] = metrics.lap_of(s['t_bag'], lap_df) + 1       # 1-based, 0 = before lap_state
    s['v'] = s['value'] * scale
    g = s.groupby(['lap', 'phase'])['v']
    return pd.DataFrame({'rms': g.apply(metrics.rms), 'max_abs': g.apply(metrics.absmax),
                         'n': g.size()}).reset_index()


def run(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('bags', nargs='+')
    bagio.add_common_args(ap)
    a = ap.parse_args(argv)
    results = {}
    for r in bagio.iter_runs(a.bags, TOPICS, a.msg_dir):
        print(f'{r.name}:')
        ect, eth = metrics.tracking_samples(r)
        lap = r.get(T['lap_state'])
        cap = style.source_caption([r.name], 'plot_tracking.py')

        fig, (a1, a2) = style.figure(2, 1, height=5.2, sharex=True)
        for ax, df, scale, ylabel, title in (
                (a1, ect, 100.0, 'cross-track error [cm]', 'Cross-track error e_ct'),
                (a2, eth, 1.0, 'heading error [deg]', 'Heading error e_theta')):
            if len(df) == 0:
                style.no_data(ax, 'topic not in bag')
            for ph in ('straight', 'arc', 'unknown'):
                sel = df[df['phase'] == ph]
                if len(sel):
                    c, lbl = PHASE_STYLE[ph]
                    plot_segments(ax, sel['t_bag'], sel['value'] * scale, c, lbl)
            ax.axhline(0, color=style.AXIS, lw=0.8)
            lap_marks(ax, lap)
            ax.set_ylabel(ylabel)
            ax.set_title(title)
        if len(ect) and not (ect['phase'] == 'straight').any():
            a1.text(0.99, 0.03, 'no e_ct on straights (publish commented out in the controller)',
                    transform=a1.transAxes, ha='right', va='bottom', fontsize=7, color=style.MUTED)
        a2.set_xlabel('time since bag start [s]')
        if len(ect) or len(eth):
            style.legend_below(a2, ncol=3)
        style.save(fig, a.out_dir, f'tracking_{r.name}', cap)

        fig, (h1, h2) = style.figure(1, 2, height=2.8)
        for ax, df, scale, xlabel, title, unit in ((h1, ect, 100.0, 'e_ct [cm]', 'Cross-track error', 'cm'),
                                                   (h2, eth, 1.0, 'e_theta [deg]', 'Heading error', 'deg')):
            any_ = False
            for ph in ('straight', 'arc'):
                v = df.loc[df['phase'] == ph, 'value'].to_numpy(float) * scale
                if v.size:
                    c, lbl = PHASE_STYLE[ph]
                    ax.hist(v, bins=30, histtype='step', lw=1.5, color=c,
                            label=f'{lbl}: RMS {metrics.rms(v):.2f} {unit}')
                    any_ = True
            if any_:
                ax.legend(loc='upper left', fontsize=7)
                ax.set_ylim(0, ax.get_ylim()[1] * 1.3)
            else:
                style.no_data(ax)
            ax.set_xlabel(xlabel)
            ax.set_ylabel('count')
            ax.set_title(title)
        style.save(fig, a.out_dir, f'tracking_hist_{r.name}', cap)

        stats = metrics.tracking_stats(r)
        laps_ct = per_lap(ect, lap, 100.0)
        laps_th = per_lap(eth, lap, 1.0)
        for k in ('straight', 'arc'):
            print(f'  {k:8s} e_ct RMS {stats[f"ect_{k}_rms_cm"]:.2f} cm, max {stats[f"ect_{k}_max_cm"]:.2f} cm '
                  f'(n={stats[f"ect_{k}_n"]});  e_theta RMS {stats[f"eth_{k}_rms_deg"]:.2f} deg, '
                  f'max {stats[f"eth_{k}_max_deg"]:.2f} deg')
        if len(laps_ct) or len(laps_th):
            print('  per lap (lap 0 = before the first lap_state):')
            for name, tab, unit in (('e_ct', laps_ct, 'cm'), ('e_theta', laps_th, 'deg')):
                for row in tab.itertuples():
                    print(f'    lap {row.lap} {row.phase:8s} {name:7s} RMS {row.rms:.2f} {unit}, '
                          f'max {row.max_abs:.2f} {unit} (n={row.n})')
        results[r.name] = {'stats': stats, 'per_lap_ect': laps_ct, 'per_lap_eth': laps_th}
    return results


def main(argv=None):
    run(argv)
    return 0


if __name__ == '__main__':
    sys.exit(main())
