"""Metric definitions shared by summarize_runs.py and the plot scripts.

Keeping them in one place guarantees that a number quoted from a figure and
the same number in runs.csv are computed the same way.
"""
import math

import numpy as np
import pandas as pd

import logpatterns
from bagio import T, nearest_dt
from robot_constants import GYRO_SCALE, PARK_AXLE_RULE_CM, wrap

# Messages published in the same 30 Hz control tick arrive within a few ms of
# each other; the tick period is 33 ms. Two dbg samples closer than this are
# treated as coming from the same tick.
SAME_TICK_S = 0.012


# --------------------------------------------------------------------------
# Localisation state
# --------------------------------------------------------------------------
def state_intervals(df, t_end, col='data'):
    """Latched String state topic -> DataFrame(start, end, state).
    Each state lasts until the next message (the last one until t_end)."""
    if df is None or len(df) == 0:
        return pd.DataFrame(columns=['start', 'end', 'state'])
    d = df.sort_values('t_bag')
    t = d['t_bag'].to_numpy(float)
    s = d[col].astype(str).str.strip().str.lower().to_numpy()
    end = np.append(t[1:], max(t_end, t[-1]))
    return pd.DataFrame({'start': t, 'end': end, 'state': s})


def loc_fractions(df, t_end):
    """Time fraction of each /localization_state value, from the first state
    message to the end of the bag. Returns ({state: fraction}, seconds)."""
    iv = state_intervals(df, t_end)
    if len(iv) == 0:
        return {}, 0.0
    dur = (iv['end'] - iv['start']).clip(lower=0)
    total = float(dur.sum())
    if total <= 0:
        return {iv['state'].iloc[-1]: 1.0}, 0.0
    fr = (dur.groupby(iv['state']).sum() / total).to_dict()
    return {k: float(v) for k, v in fr.items()}, total


# --------------------------------------------------------------------------
# Controller tracking errors
# --------------------------------------------------------------------------
def tracking_samples(run):
    """Controller dbg samples with their control phase.

    Returns (ect, eth): DataFrames with t_bag, value, phase in
    {'straight', 'arc', 'unknown'}.

    Phase definition (from round1_controller_node.py):
      * arc (TURN): _turn() publishes e_ct (= distance to the planned circle
        minus R, >0 = outside), arc_dist, arc_R, e_theta_deg and delta_deg in
        the same tick  -> a sample is 'arc' if an arc_R message lies within
        SAME_TICK_S.
      * straight (DRIVE, Stanley): _stanley_steer() publishes e_theta_deg,
        delta_deg and k_h_eff -> 'straight' if a k_h_eff message lies within
        SAME_TICK_S. In the current source the Stanley e_ct publish is
        commented out, so e_ct normally only exists in arcs.
    """
    t_arc = run.get(T['arc_R'])['t_bag'].to_numpy(float) if run.has(T['arc_R']) else np.array([])
    t_str = run.get(T['k_h'])['t_bag'].to_numpy(float) if run.has(T['k_h']) else np.array([])
    t_arc, t_str = np.sort(t_arc), np.sort(t_str)

    def classify(df):
        if df is None or len(df) == 0:
            return pd.DataFrame(columns=['t_bag', 'value', 'phase'])
        t = df['t_bag'].to_numpy(float)
        da, ds = nearest_dt(t, t_arc), nearest_dt(t, t_str)
        phase = np.where((da <= SAME_TICK_S) & (da <= ds), 'arc',
                         np.where(ds <= SAME_TICK_S, 'straight', 'unknown'))
        return pd.DataFrame({'t_bag': t, 'value': df['data'].to_numpy(float), 'phase': phase})

    return classify(run.get(T['e_ct'])), classify(run.get(T['e_theta']))


def lap_of(t, lap_df):
    """Lap index (lap_state data_2) at each time t; -1 before the first
    lap_state message. data = [corner_idx, corner_count, lap]."""
    t = np.asarray(t, float)
    if lap_df is None or len(lap_df) == 0 or 'data_2' not in lap_df:
        return np.full(t.shape, -1, dtype=int)
    d = lap_df.sort_values('t_bag')
    tl, lap = d['t_bag'].to_numpy(float), d['data_2'].to_numpy(int)
    i = np.searchsorted(tl, t, side='right') - 1
    out = np.where(i >= 0, lap[np.clip(i, 0, None)], -1)
    return out.astype(int)


def rms(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return float(np.sqrt(np.mean(x ** 2))) if x.size else math.nan


def absmax(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return float(np.max(np.abs(x))) if x.size else math.nan


def tracking_stats(run):
    """RMS / max of e_ct [cm] and e_theta [deg] per phase."""
    ect, eth = tracking_samples(run)
    out = {}
    for phase in ('straight', 'arc'):
        e = ect.loc[ect['phase'] == phase, 'value'] * 100.0
        h = eth.loc[eth['phase'] == phase, 'value']
        out[f'ect_{phase}_rms_cm'] = rms(e)
        out[f'ect_{phase}_max_cm'] = absmax(e)
        out[f'ect_{phase}_n'] = int(e.size)
        out[f'eth_{phase}_rms_deg'] = rms(h)
        out[f'eth_{phase}_max_deg'] = absmax(h)
    return out


# --------------------------------------------------------------------------
# Dead time (command -> measured yaw rate)
# --------------------------------------------------------------------------
def measured_yaw_rate(run, source='auto'):
    """(t, yaw rate REP-103 [rad/s], label). 'gyro' = /bno055/imu gyro_z *
    GYRO_SCALE (the same correction ekf_node.py applies); 'odom' =
    /ekf/odom omega."""
    if source in ('auto', 'gyro') and run.has(T['imu']):
        d = run.get(T['imu'])
        return (d['t_bag'].to_numpy(float), d['gyro_z'].to_numpy(float) * GYRO_SCALE,
                'gyro (/bno055/imu x GYRO_SCALE)')
    if source in ('auto', 'odom') and run.has(T['odom']):
        d = run.get(T['odom'])
        return d['t_bag'].to_numpy(float), d['omega'].to_numpy(float), 'EKF omega (/ekf/odom)'
    return np.array([]), np.array([]), 'none'


def estimate_dead_time(t_cmd, u_cmd, t_meas, y_meas, v_cmd=None, fs=100.0,
                       max_lag=1.0, min_lag=-0.2, v_min=0.05):
    """Lag between commanded and measured yaw rate by cross-correlation.

    Both signals are put on a common fs grid (command: zero-order hold, as the
    bridge holds the last /cmd_vel; measurement: linear interpolation). Only
    samples where the commanded speed |v_cmd| >= v_min are used (standstill
    carries no steering information). For each lag L the Pearson correlation
    corr(u(t), y(t + L)) is computed over the valid samples; the best lag is
    refined with a parabola through the peak and its neighbours.

    Returns dict(lag_s, corr, gain, n, lags, corrs). gain is the least-squares
    slope y(t+lag) ~ gain * u(t) (compare steer_gain_pred = 0.84).
    """
    t_cmd, u_cmd = np.asarray(t_cmd, float), np.asarray(u_cmd, float)
    t_meas, y_meas = np.asarray(t_meas, float), np.asarray(y_meas, float)
    res = {'lag_s': math.nan, 'corr': math.nan, 'gain': math.nan, 'n': 0,
           'lags': np.array([]), 'corrs': np.array([])}
    if t_cmd.size < 10 or t_meas.size < 10:
        return res
    o = np.argsort(t_cmd)
    t_cmd, u_cmd = t_cmd[o], u_cmd[o]
    v = np.asarray(v_cmd, float)[o] if v_cmd is not None else np.ones_like(u_cmd)
    o = np.argsort(t_meas)
    t_meas, y_meas = t_meas[o], y_meas[o]
    t0, t1 = max(t_cmd[0], t_meas[0]), min(t_cmd[-1], t_meas[-1])
    if t1 - t0 < 2 * max_lag:
        return res
    grid = np.arange(t0, t1, 1.0 / fs)
    idx = np.clip(np.searchsorted(t_cmd, grid, side='right') - 1, 0, t_cmd.size - 1)
    u = u_cmd[idx]
    moving = np.abs(v[idx]) >= v_min
    # a stale command (bridge timeout 0.5 s) means the motor is off
    moving &= (grid - t_cmd[idx]) < 0.5
    y = np.interp(grid, t_meas, y_meas)
    lags = np.arange(int(round(min_lag * fs)), int(round(max_lag * fs)) + 1)
    corrs = np.full(lags.size, np.nan)
    n = grid.size
    for k, L in enumerate(lags):
        if L >= 0:
            a, b, m = u[:n - L], y[L:], moving[:n - L]
        else:
            a, b, m = u[-L:], y[:n + L], moving[-L:]
        a, b = a[m], b[m]
        if a.size < 50 or np.std(a) < 1e-9 or np.std(b) < 1e-9:
            continue
        corrs[k] = np.corrcoef(a, b)[0, 1]
    if not np.isfinite(corrs).any():
        return res
    k = int(np.nanargmax(corrs))
    lag = lags[k] / fs
    if 0 < k < lags.size - 1 and np.all(np.isfinite(corrs[k - 1:k + 2])):
        c0, c1, c2 = corrs[k - 1:k + 2]
        den = c0 - 2 * c1 + c2
        if abs(den) > 1e-12:
            lag += 0.5 * (c0 - c2) / den / fs
    L = lags[k]
    if L >= 0:
        a, b, m = u[:n - L], y[L:], moving[:n - L]
    else:
        a, b, m = u[-L:], y[:n + L], moving[-L:]
    a, b = a[m], b[m]
    gain = float(np.dot(a, b) / np.dot(a, a)) if np.dot(a, a) > 0 else math.nan
    res.update(lag_s=float(lag), corr=float(corrs[k]), gain=gain, n=int(a.size),
               lags=lags / fs, corrs=corrs)
    return res


# --------------------------------------------------------------------------
# Wall-match innovations
# --------------------------------------------------------------------------
def wall_innovations(wm_df, odom_df):
    """Innovation of every wall match against the EKF pose.

    For each match the latest /ekf/odom pose at or before the match's scan
    time (t_bag) is used with the EKF measurement model of
    src/ekf/ekf/ekf.py update_wall():
        alpha_pred = alpha_map - theta
        d_pred     = d_map - (x cos(alpha_map) + y sin(alpha_map))
        innov_alpha = wrap(alpha_meas - alpha_pred),  innov_d = d_meas - d_pred
    The pose already contains the corrections of earlier scans, so this is the
    pre-update innovation of the current scan (up to the odom publish period,
    20 ms at 50 Hz).
    """
    cols = ['t_bag', 'msg_index', 'innov_d', 'innov_alpha']
    if wm_df is None or len(wm_df) == 0 or odom_df is None or len(odom_df) == 0:
        return pd.DataFrame(columns=cols)
    m = wm_df[wm_df['match_idx'] >= 0].copy()
    if len(m) == 0:
        return pd.DataFrame(columns=cols)
    od = odom_df.sort_values('t_bag')
    to = od['t_bag'].to_numpy(float)
    i = np.clip(np.searchsorted(to, m['t_bag'].to_numpy(float), side='right') - 1, 0, None)
    x, y, th = (od[c].to_numpy(float)[i] for c in ('x', 'y', 'yaw'))
    am, dm = m['alpha_map'].to_numpy(float), m['d_map'].to_numpy(float)
    a_pred = wrap(am - th)
    d_pred = dm - (x * np.cos(am) + y * np.sin(am))
    m['innov_alpha'] = wrap(m['alpha_meas'].to_numpy(float) - a_pred)
    m['innov_d'] = m['d_meas'].to_numpy(float) - d_pred
    return m[cols + ['alpha_map', 'd_map']]


def matches_per_scan(wm_df):
    """(t_bag, n_matches) per /wall_matches message."""
    if wm_df is None or len(wm_df) == 0:
        return pd.DataFrame(columns=['t_bag', 'n_matches'])
    g = wm_df.groupby('msg_index', sort=True)
    return pd.DataFrame({'t_bag': g['t_bag'].first(), 'n_matches': g['n_matches'].first()})


# --------------------------------------------------------------------------
# Log events
# --------------------------------------------------------------------------
def log_events(run):
    return logpatterns.scan_log(run.get(T['rosout']))


def parking_result(events):
    """Parsed parking result from the log (last report wins)."""
    num = logpatterns.parse_number
    res = {'parked': False, 'park_dist_outer_cm': math.nan, 'park_expected_cm': math.nan,
           'park_lateral_dev_cm': math.nan, 'park_heading_deg': math.nan,
           'park_axle_diff_cm': math.nan, 'park_within_2cm': None,
           'park_bay_rear_cm': math.nan, 'park_bay_front_cm': math.nan, 'park_log_lang': ''}
    for ev in events:
        if ev['key'] == 'parked':
            res.update(parked=True, park_dist_outer_cm=num(ev.get('dist_cm')),
                       park_expected_cm=num(ev.get('expected')),
                       park_heading_deg=num(ev.get('heading_deg')),
                       park_axle_diff_cm=abs(num(ev.get('axle_cm'))), park_log_lang=ev['lang'])
        elif ev['key'] == 'parked_at' and not res['parked']:
            res.update(parked=True, park_log_lang=ev['lang'])
        elif ev['key'] == 'bay_clearance':
            res.update(park_bay_rear_cm=num(ev.get('rear_cm')),
                       park_bay_front_cm=num(ev.get('front_cm')))
    if math.isfinite(res['park_dist_outer_cm']) and math.isfinite(res['park_expected_cm']):
        res['park_lateral_dev_cm'] = res['park_dist_outer_cm'] - res['park_expected_cm']
    if math.isfinite(res['park_axle_diff_cm']):
        res['park_within_2cm'] = bool(res['park_axle_diff_cm'] <= PARK_AXLE_RULE_CM + 1e-9)
    return res


def estop_result(events):
    """First emergency stop / abort found in the log, and the emergency
    manoeuvres (NOTFALL-RANGIEREN: back up and re-plan, controller >= e740c8a)."""
    res = {'emergency_stop': False, 'estop_t': math.nan, 'estop_kind': '', 'estop_reason': '',
           'abort': False, 'abort_reason': '', 'n_manoeuvres': 0, 'manoeuvre_reasons': '',
           'manoeuvre_gave_up': False}
    reasons = []
    for ev in events:
        if ev['key'] == 'manoeuvre':
            res['n_manoeuvres'] += 1
            reasons.append(ev.get('reason', ''))
        elif ev['key'] == 'manoeuvre_giveup':
            res['manoeuvre_gave_up'] = True
        if ev['key'] == 'estop' and not res['emergency_stop']:
            reason = ev.get('reason') or ev['msg']
            res.update(emergency_stop=True, estop_t=ev['t_bag'], estop_reason=reason,
                       estop_kind=logpatterns.estop_kind(reason))
        elif ev['key'] == 'abort' and not res['abort']:
            res.update(abort=True, abort_reason=f"{ev.get('phase', '')}: {ev.get('reason', '')}")
    res['manoeuvre_reasons'] = ' | '.join(reasons)
    return res


def configured_dead_time(events):
    for ev in events:
        if ev['key'] == 'dead_time':
            return logpatterns.parse_number(ev.get('dead_time_s'))
    return math.nan
