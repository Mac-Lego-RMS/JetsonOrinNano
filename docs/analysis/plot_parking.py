#!/usr/bin/env python3
"""M17 -- parking results over all runs (from runs.csv of summarize_runs.py).

    python3 plot_parking.py [docs/data/runs.csv] [--range-size 10] [--out-dir ...]

Uses the parsed controller report ("EINGEPARKT. base_link X cm von der
Aussenbande (erwartet E), Kurs H grad zur Bande = A cm Achsdifferenz"):
  park_lateral_dev_cm = X - E  (distance to the outer wall minus expected)
  park_heading_deg    = H      (heading relative to the wall)
  park_axle_diff_cm   = A = |0.105 m x sin(H)|; WRO rule: at most 2 cm, i.e.
                        |H| <= asin(0.02 / 0.105) = 11.0 deg
Runs are grouped by their number (trailing digits of the bag name) into
ranges of --range-size and coloured light -> dark, so later iterations are
darker. Bag-name families (text before the number, e.g. parken_test vs
cw_pos1) get different marker shapes.

Figure parking: (a) heading error vs lateral deviation, (b) axle difference
per run with the 2 cm line. Printed: success rate per run range.
"""
import argparse
import math
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import bagio
import style
from robot_constants import PARK_AXLE_RULE_CM, PARK_WHEELBASE

MARKERS = ['o', 's', 'D', '^', 'v']


def family(name):
    return re.sub(r'[_-]?\d+$', '', str(name)) or str(name)


def run(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('runs_csv', nargs='?', default=str(bagio.DEFAULT_DATA_DIR / 'runs.csv'))
    ap.add_argument('--range-size', type=int, default=10)
    ap.add_argument('--out-dir', default=str(bagio.DEFAULT_FIG_DIR))
    a = ap.parse_args(argv)
    df = pd.read_csv(a.runs_csv)
    pk = df[df['parked'].fillna(False).astype(bool) & df['park_heading_deg'].notna()].copy()
    print(f'{len(df)} runs in {a.runs_csv}, {len(pk)} with a parsed parking result')
    if not len(pk):
        print('nothing to plot')
        return {}
    pk['family'] = pk['bag'].map(family)
    rs = a.range_size
    pk['range_lo'] = (np.floor((pk['run_no'].fillna(0) - 1) / rs) * rs + 1).astype(int)
    ranges = sorted(pk['range_lo'].unique())
    cols = dict(zip(ranges, style.seq_colors(len(ranges))))
    fams = sorted(pk['family'].unique())
    marks = {f: MARKERS[i % len(MARKERS)] for i, f in enumerate(fams)}
    h_rule = math.degrees(math.asin(PARK_AXLE_RULE_CM / 100 / PARK_WHEELBASE))

    fig, (a1, a2) = style.figure(1, 2, width=8.0, height=3.6)
    for (lo, fam), g in pk.groupby(['range_lo', 'family']):
        kw = dict(color=cols[lo], marker=marks[fam], s=46, edgecolors=style.SURFACE, linewidths=1.2,
                  zorder=3)
        a1.scatter(g['park_lateral_dev_cm'], g['park_heading_deg'], **kw)
        a2.scatter(g['run_no'], g['park_axle_diff_cm'], **kw)
    ylim = max(h_rule * 1.4, float(pk['park_heading_deg'].abs().max()) * 1.15)
    a1.set_ylim(-ylim, ylim)
    a1.axhspan(-h_rule, h_rule, color=style.GRID, alpha=0.5, lw=0, zorder=0)
    a1.axhline(0, color=style.AXIS, lw=0.8)
    a1.axvline(0, color=style.AXIS, lw=0.8)
    a1.text(a1.get_xlim()[0], h_rule, f' 2 cm rule: |heading| <= {h_rule:.1f} deg', fontsize=7,
            color=style.INK_2, va='bottom')
    a1.set_xlabel('lateral deviation from expected [cm]')
    a1.set_ylabel('heading error to the wall [deg]')
    a1.set_title('Final pose after parking')
    a2.axhline(PARK_AXLE_RULE_CM, color=style.INK_2, lw=0.9)
    a2.text(a2.get_xlim()[0], PARK_AXLE_RULE_CM, ' rule: 2 cm', fontsize=7, color=style.INK_2, va='bottom')
    a2.set_ylim(0, max(PARK_AXLE_RULE_CM * 1.6, float(pk['park_axle_diff_cm'].max()) * 1.15))
    from matplotlib.ticker import MaxNLocator
    a2.xaxis.set_major_locator(MaxNLocator(integer=True))
    a2.set_xlabel('run number')
    a2.set_ylabel('axle difference [cm]')
    a2.set_title('Axle difference per run')
    # legend: run ranges (colour) and families (marker)
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], ls='', marker='o', color=cols[lo], ms=7, label=f'runs {lo}-{lo + rs - 1}')
               for lo in ranges]
    if len(fams) > 1:
        handles += [Line2D([], [], ls='', marker=marks[f], color=style.MUTED, ms=7, label=f) for f in fams]
    a1.legend(handles=handles, loc='upper left', bbox_to_anchor=(0.0, -0.2), ncol=min(6, len(handles)),
              borderaxespad=0.0)
    style.save(fig, a.out_dir, 'parking',
               style.source_caption(pk['bag'].tolist(), f'plot_parking.py ({Path(a.runs_csv).name})'))

    res = {}
    print(f'  WRO rule: axle difference <= {PARK_AXLE_RULE_CM} cm (|heading| <= {h_rule:.1f} deg)')
    for lo in ranges:
        g = pk[pk['range_lo'] == lo]
        ok = int(g['park_within_2cm'].fillna(False).astype(bool).sum())
        res[lo] = {'n': len(g), 'within': ok,
                   'median_axle_cm': float(g['park_axle_diff_cm'].median()),
                   'median_abs_heading_deg': float(g['park_heading_deg'].abs().median()),
                   'median_abs_lateral_cm': float(g['park_lateral_dev_cm'].abs().median())}
        print(f'  runs {lo}-{lo + rs - 1}: {ok}/{len(g)} within 2 cm, median axle diff '
              f'{res[lo]["median_axle_cm"]:.2f} cm, median |heading| {res[lo]["median_abs_heading_deg"]:.1f} deg, '
              f'median |lateral| {res[lo]["median_abs_lateral_cm"]:.1f} cm')
    return res


def main(argv=None):
    run(argv)
    return 0


if __name__ == '__main__':
    sys.exit(main())
