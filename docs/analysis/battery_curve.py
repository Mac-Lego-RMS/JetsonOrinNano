"""Discharge curve of a battery pack from a recording of /esp_serial_bridge/battery.

The ESP32-S3 measures the pack voltage on IO1 and the bridge publishes it once
per second. This script writes the curve of one or more recordings into one
table:

    python docs/analysis/battery_curve.py \\
        race_450=Entladung_450 endurance_1150=Entladung_1150 \\
        --out docs/data/discharge.csv

Readings above the 4S maximum of 16.8 V cannot come from the pack: they
appear while a second source is plugged into the other power input, which
the ideal-diode OR then selects. Those readings are dropped and reported.
The margin covers the reading itself, which is up to 1 % high at the top of
the range (data/manual/battery_divider.csv). Time starts at the first reading kept.

Each path is a bag directory or a single .db3 file; only sqlite3 and the
standard library are needed.
"""
import argparse
import csv
import sqlite3
import struct
import sys
from pathlib import Path

PACK_MAX_V = 16.8
MARGIN_V = 0.25


def voltage(b):
    """sensor_msgs/BatteryState: header, then float32 voltage."""
    e = '<' if b[1] == 1 else '>'
    o = 4 + 8                                          # stamp
    n = struct.unpack_from(e + 'I', b, o)[0]           # frame_id length
    o += 4 + n
    o += (4 - (o - 4) % 4) % 4
    return struct.unpack_from(e + 'f', b, o)[0]


def db3_of(path):
    p = Path(path)
    if p.is_dir():
        found = sorted(p.glob('*.db3'))
        if not found:
            sys.exit(f'no .db3 in {p}')
        return found[0]
    return p


def curve(path):
    db = sqlite3.connect(f'file:{db3_of(path)}?mode=ro', uri=True)
    tid = db.execute("select id from topics where name='/esp_serial_bridge/battery'").fetchone()
    if tid is None:
        sys.exit(f'/esp_serial_bridge/battery not in {path}')
    rows = [(ts / 1e9, voltage(bytes(d))) for ts, d in
            db.execute('select timestamp, data from messages where topic_id=? '
                       'order by timestamp', tid)]
    kept = [(t, v) for t, v in rows if v <= PACK_MAX_V + MARGIN_V]
    return kept, len(rows) - len(kept)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('runs', nargs='+', metavar='label=bag')
    ap.add_argument('--out', required=True)
    a = ap.parse_args(argv)
    with open(a.out, 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['pack', 't_s', 'voltage_v'])
        for label, path in (r.split('=', 1) for r in a.runs):
            kept, dropped = curve(path)
            t0 = kept[0][0]
            for t, v in kept:
                w.writerow([label, round(t - t0, 1), round(v, 3)])
            print(f'{label}: {len(kept)} readings over {(kept[-1][0] - t0) / 60:.1f} min, '
                  f'{kept[0][1]:.2f} V -> {kept[-1][1]:.2f} V; '
                  f'{dropped} readings above {PACK_MAX_V} V dropped')
    return 0


if __name__ == '__main__':
    sys.exit(main())
