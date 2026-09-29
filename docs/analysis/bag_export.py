#!/usr/bin/env python3
"""Export ROS 2 bags to one CSV per topic.

    python3 bag_export.py BAG [BAG ...] [-o OUT] [--topics GLOB,GLOB] [--heavy]

BAG is a bag directory, its metadata.yaml, a .db3 file, or a directory of bags.
Default output: docs/data/<bagname>/ (with -o and several bags:
OUT/<bagname>/). Each topic becomes <topic with '/' -> '__'>.csv, e.g.
/ekf/odom -> ekf__odom.csv. export_info.json lists topic, type, file and row
count; the plot scripts accept such an export dir instead of the bag.

Columns: t_bag (s since bag start, recorder receive time), t_header (header
stamp, s since bag start, NaN if none), msg_index, then the message fields
(see bagio.py). Lightweight topics only by default; /scan, point clouds and
images need --heavy (point clouds become one row per point).
"""
import argparse
import json
import sys
from pathlib import Path

import bagio


def export_bag(path, out_dir, topics=None, heavy=False, msg_dir=None, typestore=None, quiet=False):
    """Export one bag; returns the output directory."""
    with bagio.Bag(path, msg_dir=msg_dir, typestore=typestore) as bag:
        tables = bagio.read_tables(bag, topics, heavy=heavy)
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        info = {'bag': bag.name, 'source': str(Path(path).resolve()), 'start_ns': bag.start_ns,
                'start_utc': bag.start_utc.isoformat(), 'duration_s': bag.duration_s,
                'topics': {}}
        for topic, df in sorted(tables.items()):
            fname = bagio.topic_to_filename(topic)
            df.to_csv(out / fname, index=False)
            info['topics'][topic] = {'type': bag.topics.get(topic, ''), 'file': fname,
                                     'rows': int(len(df))}
            if not quiet:
                print(f'  {topic:45s} {len(df):8d} rows -> {fname}')
        skipped = sorted(set(bag.topics) - set(tables))
        info['skipped_topics'] = {t: bag.topics[t] for t in skipped}
        (out / 'export_info.json').write_text(json.dumps(info, indent=2))
        if skipped and not quiet:
            print(f'  skipped (heavy or not selected): {", ".join(skipped)}')
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('bags', nargs='+', help='bag dir / metadata.yaml / .db3 / dir of bags')
    ap.add_argument('-o', '--out', default=None,
                    help='output dir (default docs/data/<bagname>/; with several bags '
                         'OUT/<bagname>/)')
    ap.add_argument('--topics', default=None,
                    help='comma-separated topic globs, e.g. "/ekf/*,/cmd_vel"')
    ap.add_argument('--heavy', action='store_true', help='also export /scan and point clouds')
    ap.add_argument('--msg-dir', default=None, help='robot_msgs/msg directory')
    a = ap.parse_args(argv)
    topics = [t.strip() for t in a.topics.split(',')] if a.topics else None
    bags = [b for b in bagio.find_bags(a.bags) if not bagio.is_export_dir(b)]
    if not bags:
        print('no bags found', file=sys.stderr)
        return 1
    store = bagio.make_typestore(a.msg_dir)
    for b in bags:
        name = bagio.bag_name(b)
        if a.out is None:
            out = bagio.DEFAULT_DATA_DIR / name
        elif len(bags) == 1:
            out = Path(a.out)
        else:
            out = Path(a.out) / name
        print(f'{name} -> {out}')
        export_bag(b, out, topics, a.heavy, typestore=store)
    return 0


if __name__ == '__main__':
    sys.exit(main())
