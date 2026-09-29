#!/usr/bin/env python3
"""M15 -- Jetson CPU / GPU / RAM load, power and temperatures over a run.

    python3 plot_cpu.py BAG [BAG ...] [--out-dir docs/figures]

Topics (foxglove_overlay_node.py, jetson-stats, ~1 Hz):
  /jtop/cpu_total, /jtop/gpu_load, /jtop/ram_percent  Float32 [%]
  /jtop/power_total                                   Float32 [W]
  /jtop/temp/<zone>                                   sensor_msgs/Temperature [degC]
                                                      (zones cpu, gpu, soc, tj)
Figure: cpu_<bag> with three panels (load %, power, temperatures).
Printed: mean / max of every signal.
"""
import argparse
import sys

import numpy as np

import bagio
import style
from bagio import T

TOPICS = [T['cpu'], T['gpu'], T['ram'], T['power'], T['temps'], T['run_state']]


def end_label(ax, x, y, text):
    ax.annotate(text, (x, y), xytext=(4, 0), textcoords='offset points', va='center',
                fontsize=7, color=style.INK_2)


def run(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('bags', nargs='+')
    bagio.add_common_args(ap)
    a = ap.parse_args(argv)
    results = {}
    for r in bagio.iter_runs(a.bags, TOPICS, a.msg_dir):
        print(f'{r.name}:')
        res = {}
        fig, (a1, a2, a3) = style.figure(3, 1, height=6.6, sharex=True)
        x_end = r.duration_s
        for i, (key, label) in enumerate((('cpu', 'CPU total'), ('gpu', 'GPU'), ('ram', 'RAM'))):
            df = r.get(T[key])
            if len(df):
                a1.plot(df['t_bag'], df['data'], color=style.CAT[i], label=label)
                end_label(a1, df['t_bag'].iloc[-1], df['data'].iloc[-1], label)
                res[key] = {'mean': float(df['data'].mean()), 'max': float(df['data'].max())}
        if res:
            a1.set_ylim(0, 100)
            a1.legend(loc='upper left', ncol=3)
        else:
            style.no_data(a1, 'no /jtop load topics')
        a1.set_ylabel('load [%]')
        a1.set_title('Jetson load')
        pw = r.get(T['power'])
        if len(pw):
            a2.plot(pw['t_bag'], pw['data'], color=style.CAT[0])
            res['power'] = {'mean': float(pw['data'].mean()), 'max': float(pw['data'].max())}
            a2.set_ylim(0, max(1.0, float(pw['data'].max()) * 1.15))
        else:
            style.no_data(a2, 'no /jtop/power_total')
        a2.set_ylabel('power [W]')
        a2.set_title('Board power')
        temps = sorted(r.matching(T['temps']).items())
        for i, (topic, df) in enumerate(temps[:8]):
            if not len(df):
                continue
            zone = topic.rsplit('/', 1)[-1]
            a3.plot(df['t_bag'], df['temperature'], color=style.CAT[i], label=zone)
            end_label(a3, df['t_bag'].iloc[-1], df['temperature'].iloc[-1], zone)
            res[f'temp_{zone}'] = {'mean': float(df['temperature'].mean()),
                                   'max': float(df['temperature'].max())}
        if temps:
            a3.legend(loc='upper left', ncol=4)
        else:
            style.no_data(a3, 'no /jtop/temp/*')
        a3.set_ylabel('temperature [°C]')
        a3.set_title('Thermal zones')
        a3.set_xlabel('time since bag start [s]')
        for ax in (a1, a2, a3):
            ax.set_xlim(0, x_end * 1.06 if np.isfinite(x_end) and x_end > 0 else None)
        style.save(fig, a.out_dir, f'cpu_{r.name}', style.source_caption([r.name], 'plot_cpu.py'))
        for k, v in res.items():
            unit = 'W' if k == 'power' else ('degC' if k.startswith('temp') else '%')
            print(f'  {k:10s} mean {v["mean"]:6.1f} {unit}   max {v["max"]:6.1f} {unit}')
        results[r.name] = res
    return results


def main(argv=None):
    run(argv)
    return 0


if __name__ == '__main__':
    sys.exit(main())
