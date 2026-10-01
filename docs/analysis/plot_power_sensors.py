"""Figures for the power and sensor chapter of the journal.

Reads the hand measurements in data/manual/ and the sweep results in data/
and writes four figures:

  power_budget     current per operating state, split into Jetson, the rest
                   of the system and motion (MP1)
  drive_current    supply current and derived winding current over PWM duty
                   (MP2)
  discharge        pack voltage over time at a constant load, against the
                   low-voltage warning
  imu_noise        gyroscope noise per axis over drive-axle speed, and the
                   driven/coasting comparison that separates vibration from
                   electrical interference (MP3)
  vibration_orders pitch-rate spectrum per speed step over orders of the
                   wheel rotation: which part of the drivetrain shakes
  gear_repair      pitch and yaw noise and motor voltage over the three
                   drive-gear mountings, and with the wheels removed

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
    runs = (('adapter', 'improvised adapter', style.CAT[1], 'o', '-'),
            ('repaired', 'adapter repaired', style.CAT[3], 'D', '-'),
            ('newgear', 'new gear, no adapter', style.CAT[0], 's', '-'),
            ('newgear_nowheels', 'new gear, wheels off', style.CAT[0], '', ':'))
    fig, axes = style.figure(1, 3, width=7.0, height=3.0)
    for ax, key, unit, title in (
            (axes[0], 'pitch_sd_{}_dps', 'pitch noise σ [°/s]', 'Pitch: vibration'),
            (axes[1], 'yaw_sd_{}_dps', 'yaw noise σ [°/s]', 'Yaw: used for heading')):
        for run, lbl, col, mk, ls in runs:
            ax.plot(v, [float(r[key.format(run)]) for r in rows], ls + mk, color=col,
                    lw=style.LINE_W, ms=style.MARKER_S - 2, mec=style.SURFACE, mew=0.8,
                    label=lbl)
        ax.set_ylim(bottom=0)
        ax.set_ylabel(unit)
        ax.set_title(title)
    ref = np.array([float(r['motor_v_adapter']) for r in rows])
    for run, lbl, col, mk, ls in runs:
        volt = np.array([float(r[f'motor_v_{run}']) for r in rows])
        axes[2].plot(v, (volt / ref - 1) * 100, ls + mk, color=col, lw=style.LINE_W,
                     ms=style.MARKER_S - 2, mec=style.SURFACE, mew=0.8, label=lbl)
    axes[2].set_ylim(-10, 25)
    axes[2].set_ylabel('change against improvised adapter [%]')
    axes[2].set_title('Motor voltage: friction')
    for ax in axes:
        ax.set_xlabel('axle speed [rad/s]')
    axes[2].legend(loc='upper right', fontsize=6.5, handlelength=2.2, borderaxespad=0.3)
    style.save(fig, out_dir, 'gear_repair',
               'Source: bags PWM_vorher, PWM_nachher, PWM_neuesZahnrad(_ohneRaeder) via '
               'data/mp3_before_after.csv (sweep_noise.py)  |  plot_power_sensors.py')


def vibration_orders(out_dir):
    rows = read(DATA / 'mp3_order_spectrum.csv')
    runs = (('adapter', 'improvised adapter', style.CAT[1]),
            ('repaired', 'adapter repaired', style.CAT[3]),
            ('newgear', 'new gear', style.CAT[0]),
            ('newgear_nowheels', 'new gear, wheels off', style.CAT[6]))
    speeds = (0.6, 1.0)
    fig, axes = style.figure(len(runs), len(speeds), width=7.0, height=5.4,
                             sharex='col', sharey=True)
    for j, v in enumerate(speeds):
        for i, (run, lbl, col) in enumerate(runs):
            ax = axes[i, j]
            sel = [r for r in rows if r['run'] == run and float(r['v_cmd_mps']) == v
                   and r['pitch_amp_dps']]
            o = np.array([float(r['order']) for r in sel])
            amp = np.array([float(r['pitch_amp_dps']) for r in sel])
            ax.fill_between(o, amp, color=col, alpha=0.25, lw=0)
            ax.plot(o, amp, color=col, lw=1.0)
            for k in range(1, 8):
                ax.axvline(k, color=style.MUTED, lw=0.5, ls=(0, (2, 3)), zorder=0)
            ax.set_xlim(0, o.max())
            ax.grid(False)
            if j == 0:
                ax.set_ylabel(f'{lbl}\n[°/s]', fontsize=7.5)
        axes[0, j].set_title(f'{v} m/s  ({float(sel[0]["axle_rad_s"]) / (2 * np.pi):.1f} '
                             'wheel revolutions per second)', fontsize=8.5)
        axes[-1, j].set_xlabel('order = frequency / wheel rotation frequency')
        axes[-1, j].set_xticks(range(0, int(o.max()) + 1))
    axes[0, 0].set_ylim(0, 3.6)
    style.save(fig, out_dir, 'vibration_orders',
               'Pitch-rate amplitude spectrum. Source: bags PWM_vorher, PWM_nachher, '
               'PWM_neuesZahnrad(_ohneRaeder) via data/mp3_order_spectrum.csv  |  '
               'plot_power_sensors.py')


def discharge(out_dir):
    rows = read(DATA / 'discharge.csv')
    packs = (('race_450', '450 mAh race pack', style.CAT[0]),
             ('endurance_1150', '1150 mAh endurance pack', style.CAT[1]))
    fig, ax = style.figure(1, 1, width=7.0, height=3.2)
    ax.axvspan(0, 3, color=style.GRID, alpha=0.6, lw=0)
    ax.text(1.5, 16.75, 'one round\n(3 min)', ha='center', va='top', fontsize=7.5,
            color=style.INK_2)
    ax.axhline(15.2, color=style.STATUS['serious'], lw=1.0, ls='--')
    ax.text(0.2, 15.23, 'low-voltage warning, 3.8 V/cell', fontsize=7.5, va='bottom',
            color=style.STATUS['serious'])
    t_end = 0
    for key, lbl, col in packs:
        sel = [r for r in rows if r['pack'] == key]
        if not sel:
            continue
        t = np.array([float(r['t_s']) for r in sel]) / 60
        v = np.array([float(r['voltage_v']) for r in sel])
        ax.plot(t, v, color=col, lw=style.LINE_W, label=lbl)
        below = t[v < 15.2]
        if len(below):
            ax.plot(below[0], 15.2, 'o', color=col, ms=style.MARKER_S, mec=style.SURFACE)
            ax.annotate(f'{below[0]:.1f} min', (below[0], 15.2), xytext=(4, -12),
                        textcoords='offset points', fontsize=8, color=col)
        t_end = max(t_end, t[-1])
    ax.set_xlim(0, np.ceil(t_end + 0.5))
    ax.set_ylim(14.4, 16.8)
    ax.set_xlabel('time [min], vehicle standing, motor off')
    ax.set_ylabel('pack voltage under load [V]')
    cell = ax.secondary_yaxis('right', functions=(lambda x: x / 4, lambda x: x * 4))
    cell.set_ylabel('per cell [V]')
    ax.set_title('Discharge of the race pack')
    ax.legend(loc='upper right')
    style.save(fig, out_dir, 'discharge',
               'Source: bag Entladung_450 via data/discharge.csv (battery_curve.py)  |  '
               'plot_power_sensors.py')


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out-dir', default=str(FIGURES))
    a = ap.parse_args(argv)
    style.apply_style()
    for fn in (power_budget, drive_current, discharge, imu_noise, gear_repair,
               vibration_orders):
        fn(a.out_dir)
    return 0


if __name__ == '__main__':
    sys.exit(main())
