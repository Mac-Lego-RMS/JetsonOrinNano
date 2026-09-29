#!/usr/bin/env python3
"""M11 -- steering / actuator dead time from command vs measured yaw rate.

    python3 plot_dead_time.py BAG [BAG ...] [--source gyro|odom] [--max-lag 1.0]

Signals:
  command   /cmd_vel angular.z [rad/s]. The esp_serial_bridge converts it with
            the calibrated Ackermann inverse delta = atan(L * omega / v_ist)
            and the steer LUT into a servo command (REP 103: yaw rate, CCW +).
            NOTE: with the bridge parameter steer_raw_bypass:=true angular.z
            is a raw servo fraction instead -- the lag is still valid, the
            gain is not.
  measured  /bno055/imu angular_velocity.z x GYRO_SCALE (-0.9674, the
            correction applied in ekf_node.py) = REP-103 yaw rate, or with
            --source odom the EKF yaw rate /ekf/odom twist.angular.z.
Both on the recorder receive clock (t_bag).

Method: metrics.estimate_dead_time() -- 100 Hz grid, command zero-order hold,
Pearson correlation for every lag in [min-lag, max-lag] using only samples
where the commanded speed is >= 0.05 m/s, parabolic refinement of the peak.
The controller assumes steer_dead_time = 0.26 s (value taken from the
controller's start-up log line when present).

Figure: dead_time_<bag> (correlation vs lag; time-series excerpt with the
command shifted by the estimated lag).
Printed: lag, peak correlation, yaw-rate gain (controller: steer_gain_pred 0.84).
"""
import argparse
import math
import sys

import numpy as np

import bagio
import metrics
import style
from bagio import T
from robot_constants import STEER_DEAD_TIME, STEER_GAIN_PRED

TOPICS = [T['cmd_vel'], T['imu'], T['odom'], T['rosout']]


def busiest_window(t, u, width=8.0):
    """Start time of the window with the most steering activity."""
    if t.size < 2 or t[-1] - t[0] <= width:
        return (t[0] if t.size else 0.0)
    starts = np.arange(t[0], t[-1] - width, 0.5)
    act = [np.sum(np.abs(np.diff(u[(t >= s) & (t < s + width)]))) for s in starts]
    return float(starts[int(np.argmax(act))])


def run(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('bags', nargs='+')
    ap.add_argument('--source', choices=('auto', 'gyro', 'odom'), default='auto')
    ap.add_argument('--max-lag', type=float, default=1.0)
    ap.add_argument('--min-lag', type=float, default=-0.2)
    bagio.add_common_args(ap)
    a = ap.parse_args(argv)
    results = {}
    for r in bagio.iter_runs(a.bags, TOPICS, a.msg_dir):
        print(f'{r.name}:')
        cmd = r.get(T['cmd_vel'])
        tm, ym, label = metrics.measured_yaw_rate(r, a.source)
        if len(cmd) == 0 or tm.size == 0:
            print('  no /cmd_vel or no yaw-rate measurement -- skipped')
            continue
        tc, uc, vc = (cmd[c].to_numpy(float) for c in ('t_bag', 'angular_z', 'linear_x'))
        res = metrics.estimate_dead_time(tc, uc, tm, ym, v_cmd=vc, max_lag=a.max_lag,
                                         min_lag=a.min_lag)
        cfg = metrics.configured_dead_time(metrics.log_events(r))
        assumed = cfg if math.isfinite(cfg) else STEER_DEAD_TIME
        res['assumed_s'] = assumed
        res['source'] = label
        cap = style.source_caption([r.name], 'plot_dead_time.py')

        fig, (a1, a2) = style.figure(2, 1, height=5.6)
        if res['lags'].size:
            a1.plot(res['lags'], res['corrs'], color=style.CAT[0])
            a1.axvline(res['lag_s'], color=style.INK, lw=0.9)
            a1.axvline(assumed, color=style.MUTED, lw=0.9)
            right = res['lag_s'] < 0.5 * (res['lags'][0] + res['lags'][-1])
            a1.annotate(f'estimated {res["lag_s"]:.3f} s (r = {res["corr"]:.3f})',
                        (res['lag_s'], res['corr']), xytext=(10 if right else -10, -4),
                        textcoords='offset points', ha='left' if right else 'right',
                        va='top', fontsize=7.5, color=style.INK)
            a1.text(assumed, 0.04, f' controller assumes {assumed:.2f} s ' if right else
                    f'controller assumes {assumed:.2f} s ',
                    transform=a1.get_xaxis_transform(), fontsize=7, color=style.INK_2,
                    ha='left' if right and assumed >= res['lag_s'] else 'right', va='bottom')
            a1.plot([res['lag_s']], [res['corr']], 'o', color=style.CAT[0], mec=style.SURFACE, mew=2)
        else:
            style.no_data(a1, 'not enough moving samples')
        a1.set_xlabel('lag of the measured yaw rate behind the command [s]')
        a1.set_ylabel('correlation [-]')
        a1.set_title('Cross-correlation command vs measured yaw rate')

        t0 = busiest_window(tc, uc)
        w = (tc >= t0) & (tc <= t0 + 8)
        wm = (tm >= t0) & (tm <= t0 + 8)
        gain = res['gain'] if math.isfinite(res['gain']) else 1.0
        a2.step(tc[w], uc[w] * gain, where='post', color=style.CAT[1], lw=1.2,
                label=f'command x gain {gain:.2f}')
        if math.isfinite(res['lag_s']):
            a2.step(tc[w] + res['lag_s'], uc[w] * gain, where='post', color=style.CAT[2], lw=1.2,
                    label=f'command shifted by {res["lag_s"]:.3f} s')
        a2.plot(tm[wm], ym[wm], color=style.CAT[0], lw=1.2, label=f'measured: {label}')
        a2.set_xlabel('time since bag start [s]')
        a2.set_ylabel('yaw rate [rad/s]')
        a2.set_title('Excerpt with the most steering activity')
        style.legend_below(a2, ncol=2)
        style.save(fig, a.out_dir, f'dead_time_{r.name}', cap)

        diff_ms = (res['lag_s'] - assumed) * 1000 if math.isfinite(res['lag_s']) else math.nan
        print(f'  measured: {label}')
        print(f'  dead time {res["lag_s"]:.3f} s (peak r = {res["corr"]:.3f}, n = {res["n"]}); '
              f'controller assumes {assumed:.3f} s -> difference {diff_ms:+.0f} ms')
        print(f'  yaw-rate gain {res["gain"]:.3f} (controller steer_gain_pred {STEER_GAIN_PRED})')
        results[r.name] = {k: v for k, v in res.items() if k not in ('lags', 'corrs')}
    return results


def main(argv=None):
    run(argv)
    return 0


if __name__ == '__main__':
    sys.exit(main())
