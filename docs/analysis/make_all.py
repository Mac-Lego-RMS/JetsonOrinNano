#!/usr/bin/env python3
"""Run the whole toolkit for a list of bags.

    python3 make_all.py BAG_OR_DIR [...] [--data-dir docs/data] [--fig-dir docs/figures]
                        [--no-export] [--heavy] [--skip colour,trajectory,...]

Steps (each one is isolated: a failure is reported and the rest continues):
  1. bag_export.py        docs/data/<bag>/*.csv            (skip with --no-export)
  2. summarize_runs.py    docs/data/runs.csv + text summary
  3. per bag: plot_latency, plot_cpu, plot_tracking, plot_dead_time,
     plot_localization, plot_trajectory, plot_colour_distance
     -> docs/figures/<bag>/
  4. plot_parking.py (from runs.csv), plot_steer_lut.py, plot_manual.py
     -> docs/figures/
The full console output is also written to docs/figures/make_all_log.txt,
so the printed key numbers can be quoted later.
"""
import argparse
import contextlib
import io
import sys
import time
import traceback
from pathlib import Path

import bagio

PER_BAG = ['latency', 'cpu', 'tracking', 'dead_time', 'localization', 'trajectory', 'colour']


class Tee(io.TextIOBase):
    def __init__(self, *streams):
        self.streams = streams

    def write(self, s):
        for st in self.streams:
            st.write(s)
        return len(s)

    def flush(self):
        for st in self.streams:
            st.flush()


def step(name, fn, *args):
    t0 = time.time()
    print(f'\n=== {name}')
    try:
        fn(*args)
        print(f'--- {name}: ok ({time.time() - t0:.1f} s)')
        return True
    except SystemExit as exc:
        ok = not exc.code
        print(f'--- {name}: exit {exc.code}')
        return ok
    except Exception:                                    # noqa: BLE001
        traceback.print_exc(file=sys.stdout)
        print(f'--- {name}: FAILED')
        return False


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('bags', nargs='+')
    ap.add_argument('--data-dir', default=str(bagio.DEFAULT_DATA_DIR))
    ap.add_argument('--fig-dir', default=str(bagio.DEFAULT_FIG_DIR))
    ap.add_argument('--msg-dir', default=None)
    ap.add_argument('--no-export', action='store_true', help='skip the per-topic CSV export')
    ap.add_argument('--heavy', action='store_true', help='export /scan and point clouds too')
    ap.add_argument('--skip', default='', help=f'comma-separated plots to skip: {",".join(PER_BAG)}')
    a = ap.parse_args(argv)

    import bag_export
    import plot_colour_distance
    import plot_cpu
    import plot_dead_time
    import plot_latency
    import plot_localization
    import plot_manual
    import plot_parking
    import plot_steer_lut
    import plot_tracking
    import plot_trajectory
    import summarize_runs
    tools = {'latency': plot_latency, 'cpu': plot_cpu, 'tracking': plot_tracking,
             'dead_time': plot_dead_time, 'localization': plot_localization,
             'trajectory': plot_trajectory, 'colour': plot_colour_distance}

    data_dir, fig_dir = Path(a.data_dir), Path(a.fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    skip = {s.strip() for s in a.skip.split(',') if s.strip()}
    msg = ['--msg-dir', a.msg_dir] if a.msg_dir else []
    bags = [b for b in bagio.find_bags(a.bags) if not bagio.is_export_dir(b)]
    if not bags:
        print('no bags found', file=sys.stderr)
        return 1

    log = io.StringIO()
    failures = []
    with contextlib.redirect_stdout(Tee(sys.stdout, log)):
        print(f'make_all: {len(bags)} bags -> {data_dir}, {fig_dir}')
        if not a.no_export:
            for b in bags:
                args = [str(b), '-o', str(data_dir / bagio.bag_name(b))] + msg + (['--heavy'] if a.heavy else [])
                if not step(f'export {bagio.bag_name(b)}', bag_export.main, args):
                    failures.append(f'export {b}')
        runs_csv = data_dir / 'runs.csv'
        if not step('summarize_runs', summarize_runs.main, [str(b) for b in bags] + ['-o', str(runs_csv)] + msg):
            failures.append('summarize_runs')
        for b in bags:
            name = bagio.bag_name(b)
            for key in PER_BAG:
                if key in skip:
                    continue
                args = [str(b), '--out-dir', str(fig_dir / name)] + msg
                if not step(f'{key} {name}', tools[key].run, args):
                    failures.append(f'{key} {name}')
        if runs_csv.exists():
            if not step('plot_parking', plot_parking.run, [str(runs_csv), '--out-dir', str(fig_dir)]):
                failures.append('plot_parking')
        if not step('plot_steer_lut', plot_steer_lut.run, ['--out-dir', str(fig_dir)]):
            failures.append('plot_steer_lut')
        if not step('plot_manual', plot_manual.run, ['--out-dir', str(fig_dir)]):
            failures.append('plot_manual')
        print(f'\nmake_all finished: {len(failures)} failed step(s)'
              + (': ' + ', '.join(failures) if failures else ''))
    (fig_dir / 'make_all_log.txt').write_text(log.getvalue())
    return 1 if failures else 0


if __name__ == '__main__':
    sys.exit(main())
