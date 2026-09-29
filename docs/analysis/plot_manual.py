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
--example uses the *_example.csv files (fake numbers, for trying the tool).

Figures: manual_m1_pose, manual_m3_encoder, manual_m3_gyro. Printed: mean /
max errors, r_eff and gyro scale with their spread.
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
from robot_constants import GYRO_SCALE, R_EFF, RAD_PER_TICK_DEFAULT, START_POSES, field_to_map

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
                           ('m3_gyro', 'm3_gyro_turns', lambda df, src: m3_gyro(df, a.out_dir, src))):
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
