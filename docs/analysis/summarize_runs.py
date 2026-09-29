#!/usr/bin/env python3
"""One summary row per bag -> runs.csv, plus a short text summary.

    python3 summarize_runs.py BAG_OR_DIR [...] [-o docs/data/runs.csv]

Columns (units in the name):
  bag, run_no (trailing number of the bag name), start_utc, duration_s
  race_direction           last /race_direction (CW / CCW)
  corners, laps            max corner_count / lap from /round1_controller/lap_state
                           (data = [corner_idx, corner_count, lap]); laps =
                           corners // 4 = completed laps
  three_laps_logged, finish_logged   controller log lines (see logpatterns.py)
  parked, park_*           parsed "EINGEPARKT. ..." / "PARKED. ..." line:
                           park_dist_outer_cm   base_link to the outer wall
                           park_expected_cm     expected value in the same line
                           park_lateral_dev_cm  dist - expected
                           park_heading_deg     heading relative to the wall
                           park_axle_diff_cm    |0.105 m * sin(heading)| (logged)
                           park_within_2cm      axle difference <= 2 cm (WRO rule)
                           park_bay_rear_cm / park_bay_front_cm  bay clearances
  emergency_stop, estop_t, estop_kind, estop_reason   first NOTSTOP / emergency stop
  abort, abort_reason      "Einparken/Ausparken abgebrochen: ..."
  n_manoeuvres, manoeuvre_reasons, manoeuvre_gave_up
                           "NOTFALL-RANGIEREN n/m: reason -- setzt X cm zurueck"
                           (back up and re-plan instead of stopping; newer controller)
  gyro_fail                GYRO AUSGEFALLEN logged or /ekf/gyro_ok false
  loc_ok_frac, loc_recovering_frac, loc_lost_frac, loc_lost_s
                           time share of /localization_state values from its
                           first message to the end of the bag
  run_time_s               last /viz/run_time
  ect_straight_rms_cm/max  Stanley cross-track error on straights (only if the
                           bag has e_ct in Stanley ticks -- the current source
                           does NOT publish it there, then NaN)
  eth_straight_rms_deg/max Stanley heading error on straights
  ect_arc_rms_cm/max       cross-track error to the planned circle in corners
  eth_arc_rms_deg/max      heading error to the circle tangent in corners
                           (phase detection: metrics.tracking_samples())
  battery_min_v            min /esp_serial_bridge/battery voltage
  cpu_mean_pct, cpu_max_pct   /jtop/cpu_total
  temp_max_c               max over /jtop/temp/*
  latency_median_ms, latency_p95_ms   /esp_serial_bridge/latency_ms
  n_errors                 rosout lines with level >= ERROR
"""
import argparse
import math
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import bagio
import metrics
from bagio import T

SUMMARY_TOPICS = [T[k] for k in ('race_direction', 'lap_state', 'rosout', 'loc_state', 'run_time',
                                 'e_ct', 'e_theta', 'k_h', 'arc_R', 'battery', 'cpu', 'gyro_ok',
                                 'latency')] + [T['temps']]


def run_number(name):
    m = re.search(r'(\d+)$', name)
    return int(m.group(1)) if m else math.nan


def summarize_run(run):
    row = {'bag': run.name, 'run_no': run_number(run.name),
           'start_utc': run.start_utc.isoformat() if run.start_utc else '',
           'duration_s': round(float(run.duration_s), 2)}
    row['race_direction'] = bagio.last_value(run.get(T['race_direction']), default='')
    lap = run.get(T['lap_state'])
    if len(lap) and 'data_1' in lap:
        row['corners'] = int(lap['data_1'].max())
        row['laps'] = int(lap['data_1'].max()) // 4
    else:
        row['corners'] = row['laps'] = math.nan
    events = metrics.log_events(run)
    keys = {e['key'] for e in events}
    row['three_laps_logged'] = 'three_laps' in keys
    row['finish_logged'] = 'finish' in keys
    row.update(metrics.parking_result(events))
    row.update(metrics.estop_result(events))
    gyro = run.get(T['gyro_ok'])
    row['gyro_fail'] = ('gyro_fail' in keys) or (len(gyro) > 0 and not gyro['data'].astype(bool).all())
    fr, total = metrics.loc_fractions(run.get(T['loc_state']), run.duration_s)
    for st in ('ok', 'recovering', 'lost'):
        row[f'loc_{st}_frac'] = round(fr.get(st, 0.0), 4) if fr else math.nan
    row['loc_lost_s'] = round(fr.get('lost', 0.0) * total, 2) if fr else math.nan
    row['run_time_s'] = bagio.last_value(run.get(T['run_time']), default=math.nan)
    for k, v in metrics.tracking_stats(run).items():
        row[k] = round(v, 3) if isinstance(v, float) and math.isfinite(v) else v
    bat = run.get(T['battery'])
    row['battery_min_v'] = float(bat['voltage'].min()) if len(bat) else math.nan
    cpu = run.get(T['cpu'])
    row['cpu_mean_pct'] = round(float(cpu['data'].mean()), 1) if len(cpu) else math.nan
    row['cpu_max_pct'] = round(float(cpu['data'].max()), 1) if len(cpu) else math.nan
    temps = [df['temperature'].max() for df in run.matching(T['temps']).values() if len(df)]
    row['temp_max_c'] = round(float(max(temps)), 1) if temps else math.nan
    lat = run.get(T['latency'])
    row['latency_median_ms'] = round(float(lat['data'].median()), 3) if len(lat) else math.nan
    row['latency_p95_ms'] = round(float(lat['data'].quantile(0.95)), 3) if len(lat) else math.nan
    log = run.get(T['rosout'])
    row['n_errors'] = int((log['level'] >= 40).sum()) if len(log) else 0
    row['rosout_present'] = bool(len(log))
    return row


def text_summary(df):
    n = len(df)
    lines = [f'{n} runs']
    if n == 0:
        return '\n'.join(lines)

    def cnt(col):
        return int(df[col].fillna(False).astype(bool).sum()) if col in df else 0

    parked = cnt('parked')
    within = cnt('park_within_2cm')
    judged = int(df['park_within_2cm'].notna().sum()) if 'park_within_2cm' in df else 0
    lines.append(f'reached 3 laps (lap_state >= 12 corners): '
                 f'{int((df["corners"].fillna(0) >= 12).sum())}/{n}')
    lines.append(f'parked: {parked}/{n} ({100 * parked / n:.0f} %)')
    if judged:
        lines.append(f'within the 2 cm axle rule: {within}/{judged} parked runs with numbers '
                     f'({100 * within / judged:.0f} %)')
        pk = df[df['park_axle_diff_cm'].notna()]
        lines.append(f'axle difference: median {pk["park_axle_diff_cm"].median():.2f} cm, '
                     f'max {pk["park_axle_diff_cm"].max():.2f} cm; heading error median '
                     f'{pk["park_heading_deg"].abs().median():.1f} deg')
    es = cnt('emergency_stop')
    lines.append(f'emergency stops: {es}/{n}')
    if es:
        kinds = df.loc[df['emergency_stop'].fillna(False).astype(bool), 'estop_kind'].value_counts()
        lines.append('  by kind: ' + ', '.join(f'{k} {v}' for k, v in kinds.items()))
    if 'abort' in df:
        lines.append(f'park manoeuvre aborts: {cnt("abort")}/{n}')
    if 'n_manoeuvres' in df and df['n_manoeuvres'].fillna(0).sum() > 0:
        lines.append(f'emergency manoeuvres (back up + re-plan): {int(df["n_manoeuvres"].sum())} in '
                     f'{int((df["n_manoeuvres"] > 0).sum())} runs')
    if 'loc_ok_frac' in df and df['loc_ok_frac'].notna().any():
        lines.append(f'localisation ok: mean {100 * df["loc_ok_frac"].mean():.1f} % of the time, '
                     f'lost in {int((df["loc_lost_frac"].fillna(0) > 0).sum())} runs')
    if df['ect_arc_rms_cm'].notna().any():
        lines.append(f'corner cross-track RMS: median {df["ect_arc_rms_cm"].median():.2f} cm')
    if df['ect_straight_rms_cm'].notna().any():
        lines.append(f'straight cross-track RMS: median {df["ect_straight_rms_cm"].median():.2f} cm')
    if df['eth_straight_rms_deg'].notna().any():
        lines.append(f'straight heading error RMS: median {df["eth_straight_rms_deg"].median():.2f} deg')
    if not df['rosout_present'].all():
        miss = df.loc[~df['rosout_present'], 'bag'].tolist()
        lines.append(f'WARNING: no /rosout in {len(miss)} bags -> parking/estop columns empty '
                     f'there ({", ".join(miss[:5])}{" ..." if len(miss) > 5 else ""})')
    return '\n'.join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('bags', nargs='+', help='bags, dirs of bags, or bag_export.py dirs')
    ap.add_argument('-o', '--out', default=str(bagio.DEFAULT_DATA_DIR / 'runs.csv'))
    ap.add_argument('--msg-dir', default=None)
    a = ap.parse_args(argv)
    paths = bagio.find_bags(a.bags)
    if not paths:
        print('no bags found', file=sys.stderr)
        return 1
    store = bagio.make_typestore(a.msg_dir)
    rows = []
    for p in paths:
        try:
            run = bagio.load_run(p, SUMMARY_TOPICS, typestore=store)
            rows.append(summarize_run(run))
            print(f'  {run.name}: done')
        except Exception as exc:                          # noqa: BLE001
            print(f'  {p}: FAILED ({exc})', file=sys.stderr)
    rows.sort(key=lambda r: bagio.natural_key(r['bag']))
    df = pd.DataFrame(rows)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f'wrote {out}')
    print(text_summary(df))
    return 0


if __name__ == '__main__':
    sys.exit(main())
