#!/usr/bin/env python3
"""M1 / M3 -- manual measurements from the CSV templates in docs/data/manual/.

    python3 plot_manual.py [--manual-dir docs/data/manual] [--example]
    python3 plot_manual.py --gyro-integral BAG [--t0 S --t1 S]

Files (lines starting with '#' are comments and ignored):
  m1_pose_checkpoints.csv  run, lap, checkpoint, x_true_m, y_true_m, yaw_true_deg,
                           x_ekf_m, y_ekf_m, yaw_ekf_deg, notes
      -> position error sqrt(dx^2 + dy^2) [cm] and heading error
         wrap(yaw_ekf - yaw_true) [deg] per checkpoint and lap.
         True and EKF pose must be in the SAME frame (the start-anchored map
         frame of /ekf/odom). With --true-frame field --start-pose cw_pos1 the
         TRUE values may instead be given in the field frame (origin = field
         centre, x east, y north; e.g. x = distance to the west wall - 1.5 m)
         and are converted with the field_map.py start pose.
  m3_encoder_distance.csv  trial, distance_true_m, ticks
      -> r_eff = distance_true_m / (ticks x rad_per_tick) [m per rad of the
         drive shaft], compared with r_eff = 0.0150 m in src/ekf/ekf/ekf.py.
         rad_per_tick defaults to 2.41 / 10431 / 0.0150 (from the ekf.py
         calibration note) -- set --rad-per-tick, or put the shaft angle
         difference in rad from /esp_serial_bridge/joint_states position into
         an optional column shaft_rad (it then takes precedence over ticks).
  m3_gyro_turns.csv        trial, turns, integrated_deg
      -> scale = 360 x turns / integrated_deg, compared with |GYRO_SCALE| =
         0.9674 in src/ekf/ekf/ekf_node.py. integrated_deg = |integral of the
         RAW /bno055/imu angular_velocity.z| in degrees (no scale applied);
         --gyro-integral BAG prints it for a bag (between --t0 and --t1).
  m17_parking_ruler.csv    bag, front_axle_cm, rear_axle_cm, bay_front_cm,
                           bay_rear_cm, notes
      -> the parked car measured with a ruler: distance from the outer wall
         to the side of the car at the front and at the rear axle (the same
         edge of the chassis, the two points are PARK_WHEELBASE apart).
         axle difference = |front - rear| (WRO rule: at most 2 cm), heading
         = atan2(front - rear, wheelbase), positive = nose away from the
         outer wall. bay_front_cm / bay_rear_cm (optional): gap between the
         car and the front / rear magenta wall. Joined on `bag` with runs.csv
         (--runs-csv) to compare with the robot's own estimate. With
         --side-to-centre-cm (half the car width at the rear axle) the rear
         value is also compared with park_dist_outer_cm (base_link).
--example uses the *_example.csv files (fake numbers, for trying the tool).

Figures: manual_m1_pose, manual_m3_encoder, manual_m3_gyro,
manual_m17_parking. Printed: mean / max errors, r_eff and gyro scale with
their spread, parking ruler vs EKF.
"""
import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import bagio
import style
from bagio import T
from robot_constants import (GYRO_SCALE, PARK_AXLE_RULE_CM, PARK_WHEELBASE, R_EFF,
                             RAD_PER_TICK_DEFAULT, START_POSES, field_to_map)

MANUAL_DIR = bagio.DEFAULT_DATA_DIR / 'manual'


def read_csv(path):
    if not Path(path).exists():
        return None
    df = pd.read_csv(path, comment='#', skipinitialspace=True)
    df = df.dropna(how='all')
    return df if len(df) else None


def m1(df, out_dir, src, true_frame='map', start_pose=None):
    df = df.copy()
    if true_frame == 'field':
        sp = START_POSES[start_pose]
        p = field_to_map(df[['x_true_m', 'y_true_m']].to_numpy(float), sp)
        df['x_true_m'], df['y_true_m'] = p[:, 0], p[:, 1]
        df['yaw_true_deg'] = df['yaw_true_deg'] - math.degrees(sp[2])
        src = f'{src}, true pose converted from the field frame ({start_pose})'
    df['pos_err_cm'] = np.hypot(df['x_ekf_m'] - df['x_true_m'], df['y_ekf_m'] - df['y_true_m']) * 100
    df['yaw_err_deg'] = (df['yaw_ekf_deg'] - df['yaw_true_deg'] + 180) % 360 - 180
    laps = sorted(df['lap'].unique())
    cps = list(dict.fromkeys(df['checkpoint'].astype(str)))
    xpos = {c: i for i, c in enumerate(cps)}
    fig, (a1, a2) = style.figure(1, 2, width=8.0, height=3.4)
    width = 0.2 / max(len(laps), 1)
    for j, lap in enumerate(laps):
        g = df[df['lap'] == lap]
        col = style.CAT[j] if len(laps) <= 3 else style.seq_colors(len(laps))[j]
        x = np.array([xpos[str(c)] for c in g['checkpoint']]) + (j - (len(laps) - 1) / 2) * width
        for ax, ycol in ((a1, 'pos_err_cm'), (a2, 'yaw_err_deg')):
            ax.scatter(x, g[ycol], s=40, color=col, edgecolors=style.SURFACE, linewidths=1.2, zorder=3,
                       label=f'lap {lap}')
    for ax, yl, t in ((a1, 'position error [cm]', 'EKF position error'),
                      (a2, 'heading error EKF - true [deg]', 'EKF heading error')):
        ax.set_xticks(range(len(cps)))
        ax.set_xlim(-0.6, len(cps) - 0.4)
        ax.set_xticklabels(cps, rotation=0)
        ax.set_xlabel('checkpoint')
        ax.set_ylabel(yl)
        ax.set_title(t)
    a1.set_ylim(0, max(1.0, df['pos_err_cm'].max() * 1.2))
    a2.axhline(0, color=style.AXIS, lw=0.8)
    style.legend_below(a1, ncol=min(len(laps), 6))
    style.save(fig, out_dir, 'manual_m1_pose', f'Source: {src}  |  plot_manual.py')
    print(f'M1 pose checkpoints ({len(df)} rows):')
    for lap, g in df.groupby('lap'):
        print(f'  lap {lap}: position error mean {g["pos_err_cm"].mean():.1f} cm, max {g["pos_err_cm"].max():.1f} cm; '
              f'heading error mean |.| {g["yaw_err_deg"].abs().mean():.1f} deg, max |.| {g["yaw_err_deg"].abs().max():.1f} deg')
    return df


def strip(ax, values, ref, ref_label, xlabel, title):
    v = np.asarray(values, float)
    jitter = np.linspace(-0.12, 0.12, v.size) if v.size > 1 else np.zeros(1)
    ax.scatter(v, jitter, s=40, color=style.CAT[0], edgecolors=style.SURFACE, linewidths=1.2, zorder=3,
               label='trial')
    ax.axvline(ref, color=style.INK_2, lw=0.9)
    ax.text(ref, 0.3, f' {ref_label}', color=style.INK_2, fontsize=7, va='bottom')
    m, s = float(np.mean(v)), float(np.std(v, ddof=1)) if v.size > 1 else 0.0
    ax.axvspan(m - s, m + s, color=style.BLUE_RAMP[0], lw=0, zorder=0)
    ax.axvline(m, color=style.CAT[0], lw=1.5)
    ax.text(m, -0.32, f' mean {m:.5g} (std {s:.2g})', color=style.INK, fontsize=7, va='top')
    ax.set_ylim(-0.45, 0.45)
    ax.set_yticks([])
    ax.grid(axis='y', visible=False)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    return m, s


def m3_encoder(df, out_dir, src, rad_per_tick):
    df = df.copy()
    shaft = df['shaft_rad'] if 'shaft_rad' in df else pd.Series(np.nan, index=df.index)
    df['shaft_used_rad'] = shaft.where(shaft.notna(), df['ticks'] * rad_per_tick)
    df['r_eff_m'] = df['distance_true_m'] / df['shaft_used_rad']
    fig, ax = style.figure(1, 1, height=2.4)
    m, s = strip(ax, df['r_eff_m'] * 1000, R_EFF * 1000, f'ekf.py r_eff = {R_EFF * 1000:.2f} mm',
                 'r_eff [mm of travel per rad of drive shaft]', 'Encoder calibration r_eff')
    m, s = m / 1000, s / 1000
    style.save(fig, out_dir, 'manual_m3_encoder', f'Source: {src}  |  plot_manual.py '
               f'(rad per tick {rad_per_tick:.6g})')
    print(f'M3 encoder ({len(df)} trials): r_eff mean {m:.5f} m, std {s:.5f} m, '
          f'{100 * (m / R_EFF - 1):+.2f} % vs ekf.py ({R_EFF})')
    return m, s


def m3_gyro(df, out_dir, src):
    df = df.copy()
    df['scale'] = 360.0 * df['turns'] / df['integrated_deg'].abs()
    fig, ax = style.figure(1, 1, height=2.4)
    ref = abs(GYRO_SCALE)
    m, s = strip(ax, df['scale'], ref, f'ekf_node.py |GYRO_SCALE| = {ref}', 'gyro scale factor [-]',
                 'Gyro scale from full turns')
    style.save(fig, out_dir, 'manual_m3_gyro', f'Source: {src}  |  plot_manual.py')
    print(f'M3 gyro ({len(df)} trials): scale mean {m:.4f}, std {s:.4f}, {100 * (m / ref - 1):+.2f} % vs '
          f'ekf_node.py ({ref})')
    return m, s


def m17_parking(df, out_dir, src, runs_csv=None, side_to_centre_cm=None):
    df = df.copy()
    df['bag'] = df['bag'].astype(str).str.strip()
    d = df['front_axle_cm'] - df['rear_axle_cm']
    df['ruler_axle_diff_cm'] = d.abs()
    df['ruler_heading_deg'] = np.degrees(np.arctan2(d, PARK_WHEELBASE * 100))
    df['ruler_within_2cm'] = df['ruler_axle_diff_cm'] <= PARK_AXLE_RULE_CM + 1e-9
    ekf_cols = ['park_axle_diff_cm', 'park_heading_deg', 'park_dist_outer_cm',
                'park_bay_front_cm', 'park_bay_rear_cm']
    runs = pd.read_csv(runs_csv) if runs_csv and Path(runs_csv).exists() else None
    if runs is not None:
        runs = runs[['bag'] + [c for c in ekf_cols if c in runs]].copy()
        runs['bag'] = runs['bag'].astype(str).str.strip()
        df = df.merge(runs, on='bag', how='left')
    for c in ekf_cols:
        if c not in df:
            df[c] = np.nan
    have_ekf = df['park_axle_diff_cm'].notna()

    fig, (a1, a2) = style.figure(1, 2, width=8.0, height=3.4)
    x = np.arange(len(df))
    a1.axhline(PARK_AXLE_RULE_CM, color=style.INK_2, lw=0.9)
    a1.text(len(df) - 0.6, PARK_AXLE_RULE_CM, 'rule: 2 cm ', color=style.INK_2, fontsize=7, va='bottom', ha='right')
    a1.scatter(x, df['ruler_axle_diff_cm'], s=40, color=style.CAT[0], edgecolors=style.SURFACE,
               linewidths=1.2, zorder=3, label='ruler')
    if have_ekf.any():
        a1.scatter(x[have_ekf], df.loc[have_ekf, 'park_axle_diff_cm'], s=40, facecolors='none',
                   edgecolors=style.CAT[1], linewidths=1.4, zorder=3, label='robot estimate (EKF)')
    a1.set_xticks(x)
    a1.set_xlim(-0.6, len(df) - 0.4)
    a1.set_xticklabels(df['bag'], rotation=60, ha='right', fontsize=7)
    a1.set_ylabel('axle difference [cm]')
    a1.set_ylim(0, max(3.0, float(np.nanmax(df[['ruler_axle_diff_cm', 'park_axle_diff_cm']].to_numpy(float))) * 1.2))
    a1.set_title('Axle difference per run')
    a1.legend(loc='upper left', fontsize=7, frameon=False)
    if have_ekf.any():
        g = df[have_ekf]
        lim = max(3.0, float(np.nanmax(g[['ruler_axle_diff_cm', 'park_axle_diff_cm']].to_numpy(float))) * 1.15)
        a2.plot([0, lim], [0, lim], color=style.AXIS, lw=0.8)
        a2.axvline(PARK_AXLE_RULE_CM, color=style.INK_2, lw=0.6, ls='--')
        a2.axhline(PARK_AXLE_RULE_CM, color=style.INK_2, lw=0.6, ls='--')
        a2.scatter(g['ruler_axle_diff_cm'], g['park_axle_diff_cm'], s=40, color=style.CAT[0],
                   edgecolors=style.SURFACE, linewidths=1.2, zorder=3)
        a2.set_xlim(0, lim)
        a2.set_ylim(0, lim)
        a2.set_xlabel('ruler [cm]')
        a2.set_ylabel('robot estimate [cm]')
        a2.set_title('Robot estimate vs ruler')
    else:
        a2.set_axis_off()
        a2.text(0.5, 0.5, 'no matching runs in runs.csv', ha='center', va='center',
                color=style.INK_2, transform=a2.transAxes)
    style.save(fig, out_dir, 'manual_m17_parking', f'Source: {src}  |  plot_manual.py')

    n, ok = len(df), int(df['ruler_within_2cm'].sum())
    print(f'M17 parking by ruler ({n} runs): {ok}/{n} within the 2 cm rule, axle difference median '
          f'{df["ruler_axle_diff_cm"].median():.1f} cm, max {df["ruler_axle_diff_cm"].max():.1f} cm')
    if have_ekf.any():
        g = df[have_ekf]
        da = (g['park_axle_diff_cm'] - g['ruler_axle_diff_cm']).abs()
        dh = (g['park_heading_deg'].abs() - g['ruler_heading_deg'].abs()).abs()
        print(f'  vs robot estimate ({len(g)} runs): axle difference off by mean {da.mean():.1f} cm, '
              f'max {da.max():.1f} cm; |heading| off by mean {dh.mean():.1f} deg, max {dh.max():.1f} deg; '
              f'rule verdict agrees in {int((g["ruler_within_2cm"] == (g["park_axle_diff_cm"] <= PARK_AXLE_RULE_CM + 1e-9)).sum())}/{len(g)}')
        if side_to_centre_cm is not None and g['park_dist_outer_cm'].notna().any():
            dl = g['park_dist_outer_cm'] - (g['rear_axle_cm'] + side_to_centre_cm)
            print(f'  base_link to the outer wall: robot estimate - ruler = mean {dl.mean():+.1f} cm, '
                  f'max |.| {dl.abs().max():.1f} cm')
        for side in ('front', 'rear'):
            col, ecol = f'bay_{side}_cm', f'park_bay_{side}_cm'
            if col in g and g[col].notna().any() and g[ecol].notna().any():
                db = g[ecol] - g[col]
                print(f'  gap to the {side} magenta wall: robot estimate - ruler = mean {db.mean():+.1f} cm, '
                      f'max |.| {db.abs().max():.1f} cm')
    else:
        print('  no matching bag names in runs.csv -- ruler values only')
    return df


def gyro_integral(bag, t0=None, t1=None, msg_dir=None):
    r = bagio.load_run(bag, [T['imu']], msg_dir=msg_dir)
    d = r.get(T['imu'])
    if not len(d):
        print('no /bno055/imu in the bag')
        return math.nan
    t = d['t_header'].to_numpy(float) if d['t_header'].notna().all() else d['t_bag'].to_numpy(float)
    z = d['gyro_z'].to_numpy(float)
    sel = np.ones_like(t, bool)
    if t0 is not None:
        sel &= t >= t0
    if t1 is not None:
        sel &= t <= t1
    t, z = t[sel], z[sel]
    deg = float(np.degrees(np.trapezoid(z, t) if hasattr(np, 'trapezoid') else np.trapz(z, t)))
    print(f'{r.name}: raw /bno055/imu gyro_z integrated over {t[0]:.2f}..{t[-1]:.2f} s = {deg:+.1f} deg '
          f'(|.| = {abs(deg):.1f} deg -> column integrated_deg)')
    return deg


def run(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--manual-dir', default=str(MANUAL_DIR))
    ap.add_argument('--example', action='store_true', help='use the *_example.csv files')
    ap.add_argument('--rad-per-tick', type=float, default=RAD_PER_TICK_DEFAULT)
    ap.add_argument('--true-frame', choices=('map', 'field'), default='map',
                    help='frame of x_true_m / y_true_m / yaw_true_deg in m1 (default map)')
    ap.add_argument('--start-pose', choices=sorted(START_POSES), default='cw_pos1',
                    help='start pose for --true-frame field')
    ap.add_argument('--runs-csv', default=str(bagio.DEFAULT_DATA_DIR / 'runs.csv'),
                    help='runs.csv of summarize_runs.py, for the parking comparison (m17)')
    ap.add_argument('--side-to-centre-cm', type=float, default=None,
                    help='half the car width at the rear axle; compares base_link to the outer wall (m17)')
    ap.add_argument('--gyro-integral', metavar='BAG', default=None)
    ap.add_argument('--t0', type=float, default=None)
    ap.add_argument('--t1', type=float, default=None)
    bagio.add_common_args(ap)
    a = ap.parse_args(argv)
    if a.gyro_integral:
        return {'integrated_deg': gyro_integral(a.gyro_integral, a.t0, a.t1, a.msg_dir)}
    suffix = '_example' if a.example else ''
    d = Path(a.manual_dir)
    res = {}
    for key, fname, fn in (('m1', 'm1_pose_checkpoints', lambda df, src: m1(df, a.out_dir, src, a.true_frame, a.start_pose)),
                           ('m3_encoder', 'm3_encoder_distance',
                            lambda df, src: m3_encoder(df, a.out_dir, src, a.rad_per_tick)),
                           ('m3_gyro', 'm3_gyro_turns', lambda df, src: m3_gyro(df, a.out_dir, src)),
                           ('m17_parking', 'm17_parking_ruler',
                            lambda df, src: m17_parking(df, a.out_dir, src, a.runs_csv, a.side_to_centre_cm))):
        p = d / f'{fname}{suffix}.csv'
        df = read_csv(p)
        if df is None:
            print(f'{p.name}: no data rows yet -- skipped')
            continue
        res[key] = fn(df, p.name)
    return res


def main(argv=None):
    run(argv)
    return 0


if __name__ == '__main__':
    sys.exit(main())
