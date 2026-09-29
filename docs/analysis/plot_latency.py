#!/usr/bin/env python3
"""M14 -- serial link latency and ESP32 <-> Jetson time sync.

    python3 plot_latency.py BAG [BAG ...] [--out-dir docs/figures] [--pool]

Topics (esp_serial_bridge.py):
  /esp_serial_bridge/latency_ms  Float32, one value per stamped ESP frame:
                                 transport time ESP send -> Jetson read [ms]
  /esp_serial_bridge/rtt_ms      Float32, 1 Hz: best round trip of the time sync [ms]
  /esp_serial_bridge/offset_ms   Float64, 1 Hz: ESP clock offset [ms] (millions of ms,
                                 only its change over time matters)
  /esp_serial_bridge/drift_ppm   Float32, 1 Hz: drift estimate of the time sync [ppm]

Figures: latency_<bag> (histograms with median / p95 / max) and
clocksync_<bag> (offset change and drift over the run). --pool additionally
writes latency_pooled from all given bags.
Printed: median / p95 / max / n of latency and rtt, mean drift, and the drift
implied by the slope of the offset (ms per s x 1000 = ppm).
"""
import argparse
import sys

import numpy as np

import bagio
import style
from bagio import T

TOPICS = [T['latency'], T['rtt'], T['offset'], T['drift']]


def pstats(v):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return {'n': 0, 'median': np.nan, 'p95': np.nan, 'max': np.nan}
    return {'n': int(v.size), 'median': float(np.median(v)), 'p95': float(np.percentile(v, 95)),
            'max': float(v.max())}


def hist_figure(lat, rtt, name, out_dir, stem, caption):
    fig, (a1, a2) = style.figure(1, 2, height=3.0)
    res = {}
    for ax, v, title, key in ((a1, lat, 'Frame latency ESP32 to Jetson', 'latency'),
                              (a2, rtt, 'Time-sync round trip', 'rtt')):
        if len(v) == 0:
            style.no_data(ax, f'no {key} data')
            ax.set_title(title)
            continue
        lo_q, hi_q = np.percentile(v, [0.5, 99.5])
        span = max(hi_q - lo_q, 1e-3)
        lo, hi = max(0.0, lo_q - 0.3 * span), hi_q + 0.6 * span
        style.hist(ax, np.clip(v, None, hi), bins=40)
        ax.set_xlim(lo, hi)
        ax.set_title(title)
        ax.set_xlabel(f'{key} [ms]')
        ax.set_ylabel('count')
        style.stat_lines(ax, v, ' ms')
        res[key] = pstats(v)
    style.save(fig, out_dir, stem, caption)
    return res


def run(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('bags', nargs='+')
    ap.add_argument('--pool', action='store_true', help='also one pooled histogram of all bags')
    bagio.add_common_args(ap)
    a = ap.parse_args(argv)
    results, pooled_lat, pooled_rtt, names = {}, [], [], []
    for r in bagio.iter_runs(a.bags, TOPICS, a.msg_dir):
        print(f'{r.name}:')
        lat = r.get(T['latency'])['data'].to_numpy(float) if r.has(T['latency']) else np.array([])
        rtt = r.get(T['rtt'])['data'].to_numpy(float) if r.has(T['rtt']) else np.array([])
        cap = style.source_caption([r.name], 'plot_latency.py')
        res = hist_figure(lat, rtt, r.name, a.out_dir, f'latency_{r.name}', cap)
        pooled_lat.append(lat)
        pooled_rtt.append(rtt)
        names.append(r.name)

        # clock sync over time
        fig, (b1, b2) = style.figure(2, 1, height=4.4, sharex=True)
        off = r.get(T['offset'])
        if len(off):
            t, o = off['t_bag'].to_numpy(float), off['data'].to_numpy(float)
            d = o - o[0]
            b1.plot(t, d, color=style.CAT[0])
            slope = np.polyfit(t, o, 1)[0] if t.size > 2 else np.nan   # ms per s
            res['offset_slope_ppm'] = float(slope * 1000.0)
            res['offset_change_ms'] = float(d[-1])
            b1.set_ylabel('offset change [ms]')
            b1.set_title('ESP32 clock offset (relative to first sample)')
        else:
            style.no_data(b1, 'no /esp_serial_bridge/offset_ms')
        dr = r.get(T['drift'])
        if len(dr):
            b2.plot(dr['t_bag'], dr['data'], color=style.CAT[0])
            b2.set_ylabel('drift [ppm]')
            b2.set_title('Drift estimate of the time sync')
            res['drift_mean_ppm'] = float(dr['data'].mean())
        else:
            style.no_data(b2, 'no /esp_serial_bridge/drift_ppm')
        b2.set_xlabel('time since bag start [s]')
        style.save(fig, a.out_dir, f'clocksync_{r.name}', cap)

        for k in ('latency', 'rtt'):
            if k in res:
                s = res[k]
                print(f'  {k:8s} median {s["median"]:.3f} ms  p95 {s["p95"]:.3f} ms  '
                      f'max {s["max"]:.3f} ms  (n={s["n"]})')
        if 'drift_mean_ppm' in res:
            print(f'  drift mean {res["drift_mean_ppm"]:.1f} ppm')
        if 'offset_slope_ppm' in res:
            print(f'  offset slope {res["offset_slope_ppm"]:.1f} ppm '
                  f'(offset changed by {res["offset_change_ms"]:.3f} ms over the bag)')
        results[r.name] = res
    if a.pool and len(names) > 1:
        lat, rtt = np.concatenate(pooled_lat), np.concatenate(pooled_rtt)
        res = hist_figure(lat, rtt, 'pooled', a.out_dir, 'latency_pooled',
                          style.source_caption(names, 'plot_latency.py --pool'))
        results['_pooled'] = res
        print(f'pooled over {len(names)} bags: latency median {res["latency"]["median"]:.3f} ms, '
              f'p95 {res["latency"]["p95"]:.3f} ms, max {res["latency"]["max"]:.3f} ms')
    return results


def main(argv=None):
    run(argv)
    return 0


if __name__ == '__main__':
    sys.exit(main())
