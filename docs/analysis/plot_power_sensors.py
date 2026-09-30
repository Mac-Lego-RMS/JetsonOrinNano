"""Figures for the power and sensor chapter of the journal.

Reads the hand measurements in data/manual/ and the sweep results in data/
and writes four figures:

  power_budget     current per operating state, split into Jetson, the rest
                   of the system and motion (MP1)
  drive_current    supply current and derived winding current over PWM duty
                   (MP2)
  imu_noise        gyroscope noise per axis over drive-axle speed, and the
                   driven/coasting comparison that separates vibration from
                   electrical interference (MP3)
  gear_repair      pitch noise and duty before and after the gear repair

    python docs/analysis/plot_power_sensors.py
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np

import style

DOCS = Path(__file__).resolve().parents[1]
DATA = DOCS / 'data'
FIGURES = DOCS / 'figures'
SUPPLY_V = 14.8


def read(path):
    with open(path, newline='') as fh:
        return list(csv.DictReader(fh))


def mid(row):
    return (float(row['speed_lo']) + float(row['speed_hi'])) / 2


def power_budget(out_dir):
    rows = {r['state']: float(r['current_a'])
            for r in read(DATA / 'manual' / 'mp1_power_budget.csv')}
    jetson, idle = rows['jetson_idle'], rows['system_idle']
    states = [('Jetson alone, idle', 'jetson_idle'),
              ('Jetson boot, peak', 'jetson_boot_peak'),
              ('Full system idle', 'system_idle'),
              ('Servo holding against load', 'servo_holding'),
              ('Driving, 0.3 m/s', 'driving_0.3'),
              ('Driving, 1.0 m/s', 'driving_1.0')]
    parts = []
    for _, key in states:
        total = rows[key]
        if key.startswith('jetson'):
            parts.append((total, 0.0, 0.0))
        else:
            parts.append((jetson, idle - jetson, total - idle))
    parts = np.array(parts)

    fig, ax = style.figure(1, 1, width=7.0, height=3.0)
    y = np.arange(len(states))[::-1]
    left = np.zeros(len(states))
    labels = ['Jetson Orin Nano', 'LiDAR, ESP32-S3, regulators', 'Drive motor / servo']
    for k, (lbl, col) in enumerate(zip(labels, style.CAT[:3])):
        ax.barh(y, parts[:, k], left=left, height=0.62, color=col, label=lbl)
        left += parts[:, k]
    for yi, tot in zip(y, left):
        ax.text(tot + 0.02, yi, f'{tot:.3g} A  ·  {tot * SUPPLY_V:.1f} W',
                va='center', fontsize=8, color=style.INK_2)
    ax.set_yticks(y, [s for s, _ in states])
    ax.tick_params(axis='y', length=0)
    ax.set_xlim(0, 1.75)
    ax.set_xlabel(f'current at the battery rail [A]  ({SUPPLY_V} V)')
    ax.grid(axis='y', visible=False)
    ax.set_title('Power budget: the compute platform dominates')
    style.legend_below(ax, ncol=3)
    style.save(fig, out_dir, 'power_budget',
               'Source: data/manual/mp1_power_budget.csv (bench supply)  |  plot_power_sensors.py')


def drive_current(out_dir):
    rows = read(DATA / 'manual' / 'mp2_drive_current.csv')
    base = next(float(r['supply_current_ma']) for r in rows if r['condition'] == 'baseline')
    free = [(float(r['duty']), float(r['supply_current_ma'])) for r in rows
            if r['condition'] == 'free']
    duty = np.array([d for d, _ in free])
    delta = np.array([i for _, i in free]) - base
    winding = delta / (duty / 255.0)
    slope, icpt = np.polyfit(duty, delta, 1)
    flat = winding[duty >= 50].mean()

    fig, ax = style.figure(1, 1, width=7.0, height=3.2)
    xs = np.linspace(0, 255, 50)
    ax.plot(xs, slope * xs + icpt, color=style.CAT[0], lw=style.LINE_W, alpha=0.5)
    ax.plot(duty, delta, 'o', color=style.CAT[0], ms=style.MARKER_S, mec=style.SURFACE,
            mew=1.0, label='supply current above baseline (measured)')
    ax.axhline(flat, color=style.CAT[1], lw=style.LINE_W, alpha=0.5)
    ax.plot(duty, winding, 's', color=style.CAT[1], ms=style.MARKER_S, mec=style.SURFACE,
            mew=1.0, label='winding current = supply / D (derived)')
    ax.text(252, flat + 6, f'{flat:.1f} mA', ha='right', va='bottom', fontsize=8,
            color=style.INK_2)
    ax.text(150, slope * 150 + icpt - 14, f'{slope:.3f} mA per duty step', ha='left',
            va='top', fontsize=8, color=style.INK_2)
    ax.set_xlim(0, 260)
    ax.set_ylim(0, 160)
    ax.set_xlabel('PWM duty, ESP console scale [0-255]')
    ax.set_ylabel('current [mA]')
    ax.set_title('The H-bridge is a switching stage: supply current scales with duty')
    style.legend_below(ax, ncol=2)
    style.save(fig, out_dir, 'drive_current',
               f'Source: data/manual/mp2_drive_current.csv (wheels free, Jetson disconnected, '
               f'baseline {base:.0f} mA)  |  plot_power_sensors.py')


def imu_noise(out_dir):
    axes = read(DATA / 'mp3_noise_per_axis.csv')
    ctrl = read(DATA / 'mp3_noise_vs_speed.csv')
    fig, (a1, a2) = style.figure(1, 2, width=7.0, height=3.2)

    x = [mid(r) for r in axes]
    for key, lbl, col, mk in (('pitch_sd_dps', 'pitch', style.CAT[0], 'o'),
                              ('roll_sd_dps', 'roll', style.CAT[1], 's'),
                              ('yaw_sd_dps', 'yaw (used for heading)', style.CAT[2], '^')):
        a1.plot(x, [float(r[key]) for r in axes], '-' + mk, color=col, lw=style.LINE_W,
                ms=style.MARKER_S - 1, mec=style.SURFACE, mew=1.0, label=lbl)
    a1.set_yscale('log')
    a1.set_xlabel('drive-axle speed [rad/s]')
    a1.set_ylabel('gyroscope noise σ [°/s]')
    a1.set_title('Vibration hits pitch, not yaw')
    style.legend_below(a1, ncol=2)

    both = [r for r in ctrl if r['gyro_sd_coasting']]
    xc = [mid(r) for r in both]
    a2.plot(xc, [np.degrees(float(r['gyro_sd_driven'])) for r in both], '-o',
            color=style.CAT[0], lw=style.LINE_W, ms=style.MARKER_S - 1, mec=style.SURFACE,
            mew=1.0, label='motor driven (PWM on)')
    a2.plot(xc, [np.degrees(float(r['gyro_sd_coasting'])) for r in both], 's',
            color=style.CAT[1], ms=style.MARKER_S, mec=style.SURFACE, mew=1.0,
            label='coasting (PWM off)')
    a2.set_xlabel('drive-axle speed [rad/s]')
    a2.set_ylabel('gyroscope noise σ, all axes [°/s]')
    a2.set_title('Same speed, PWM on or off: same noise')
    style.legend_below(a2, ncol=1)
    style.save(fig, out_dir, 'imu_noise',
               'Source: bag PWM_Test via data/mp3_noise_per_axis.csv, data/mp3_noise_vs_speed.csv'
               '  |  plot_power_sensors.py')


def gear_repair(out_dir):
    rows = read(DATA / 'mp3_before_after.csv')
    v = [float(r['axle_rad_s']) for r in rows]
    fig, (a1, a2) = style.figure(1, 2, width=7.0, height=3.2)
    for ax, key, unit, title in (
            (a1, 'pitch_sd', 'pitch noise σ [°/s]', 'Pitch noise: −23 % to −86 %'),
            (a2, 'duty', 'duty, controller scale [0-1023]', 'Cost: 4 % to 20 % more duty')):
        before = [float(r[f'{key}_before' + ('_dps' if key == 'pitch_sd' else '')]) for r in rows]
        after = [float(r[f'{key}_after' + ('_dps' if key == 'pitch_sd' else '')]) for r in rows]
        ax.plot(v, before, '-o', color=style.CAT[1], lw=style.LINE_W, ms=style.MARKER_S - 1,
                mec=style.SURFACE, mew=1.0, label='improvised adapter')
        ax.plot(v, after, '-s', color=style.CAT[0], lw=style.LINE_W, ms=style.MARKER_S - 1,
                mec=style.SURFACE, mew=1.0, label='adapter repaired')
        ax.set_xlabel('drive-axle speed [rad/s]')
        ax.set_ylabel(unit)
        ax.set_title(title)
        ax.set_ylim(bottom=0)
        style.legend_below(ax, ncol=2)
    style.save(fig, out_dir, 'gear_repair',
               'Source: bags PWM_vorher, PWM_nachher via data/mp3_before_after.csv (identical sweep, '
               '14.8 V)  |  plot_power_sensors.py')


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out-dir', default=str(FIGURES))
    a = ap.parse_args(argv)
    style.apply_style()
    for fn in (power_budget, drive_current, imu_noise, gear_repair):
        fn(a.out_dir)
    return 0


if __name__ == '__main__':
    sys.exit(main())
