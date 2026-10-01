"""Per-step results of the bench speed sweep (MP3): duty and gyro noise.

The sweep (`ros2 run esp_bridge pwm_sweep`) drives the wheel axle through
eight speed setpoints with a coasting phase after each and publishes the
current setpoint on /pwm_test/stage. For every drive step this script takes
the steady part (from 1.5 s after the setpoint change to the next mark) and
computes the mean axle speed, the mean commanded duty and the standard
deviation of the gyroscope on each axis.

Several recordings of the same sweep are compared step by step and written
as one table, one column group per recording. Duty alone is only comparable
at the same supply voltage; with --supply the table also gets the mean motor
voltage, duty / 1023 x supply, which is what the speed actually depends on:

    python docs/analysis/sweep_noise.py \\
        adapter=PWM_vorher repaired=PWM_nachher \\
        newgear=PWM_neuesZahnrad newgear_nowheels=PWM_neuesZahnrad_ohneRaeder \\
        --supply adapter=14.8 repaired=14.8 newgear=15.5 newgear_nowheels=15.5 \\
        --out docs/data/mp3_before_after.csv

Each path is a bag directory or a single .db3 file. Only sqlite3 and the
standard library are needed: the few message types used are decoded here.
"""
import argparse
import bisect
import csv
import math
import sqlite3
import statistics as st
import struct
import sys
from pathlib import Path

SETTLE_S = 1.5


class Cdr:
    def __init__(self, b):
        self.b, self.o = b, 4
        self.e = '<' if b[1] == 1 else '>'

    def _align(self, n):
        m = (self.o - 4) % n
        if m:
            self.o += n - m

    def _get(self, fmt, n):
        self._align(n)
        v = struct.unpack_from(self.e + fmt, self.b, self.o)[0]
        self.o += n
        return v

    def u32(self):
        return self._get('I', 4)

    def f32(self):
        return self._get('f', 4)

    def f64(self):
        return self._get('d', 8)

    def string(self):
        n = self.u32()
        self.o += n
        return n

    def header(self):
        self.u32(), self.u32(), self.string()

    def f64s(self, n=None):
        return [self.f64() for _ in range(self.u32() if n is None else n)]


def dec_f32(b):
    return Cdr(b).f32()


def dec_f32_array(b):
    r = Cdr(b)
    for _ in range(r.u32()):          # layout.dim
        r.string(), r.u32(), r.u32()
    r.u32()                           # data_offset
    return [r.f32() for _ in range(r.u32())]


def dec_joint_velocity(b):
    r = Cdr(b)
    r.header()
    for _ in range(r.u32()):          # name
        r.string()
    r.f64s()                          # position
    vel = r.f64s()
    return abs(vel[0]) if vel else 0.0


def dec_imu_gyro(b):
    r = Cdr(b)
    r.header()
    r.f64s(4), r.f64s(9)              # orientation, covariance
    return r.f64s(3)


def read_topic(db, topic, decode):
    tid = db.execute('select id from topics where name=?', (topic,)).fetchone()
    if tid is None:
        sys.exit(f'{topic} not in bag')
    rows = db.execute('select timestamp, data from messages where topic_id=? '
                      'order by timestamp', tid)
    return [(ts / 1e9, decode(bytes(d))) for ts, d in rows]


def window(series, t0, t1):
    ts = [t for t, _ in series]
    return [v for _, v in series[bisect.bisect_left(ts, t0):bisect.bisect_left(ts, t1)]]


def db3_of(path):
    p = Path(path)
    if p.is_dir():
        found = sorted(p.glob('*.db3'))
        if not found:
            sys.exit(f'no .db3 in {p}')
        return found[0]
    return p


def analyse(path):
    db = sqlite3.connect(f'file:{db3_of(path)}?mode=ro', uri=True)
    stage = read_topic(db, '/pwm_test/stage', dec_f32)
    speed = read_topic(db, '/esp_serial_bridge/joint_states', dec_joint_velocity)
    duty = read_topic(db, '/esp_serial_bridge/motor_state', dec_f32_array)
    gyro = read_topic(db, '/bno055/imu', dec_imu_gyro)
    steps = {}
    for (t, v), (t_next, _) in zip(stage, stage[1:]):
        if v <= 0:
            continue
        a, b = t + SETTLE_S, t_next
        g = window(gyro, a, b)
        steps[round(v, 2)] = dict(
            axle=st.mean(window(speed, a, b)),
            duty=st.mean(abs(d[0]) for d in window(duty, a, b)),
            **{ax: math.degrees(st.pstdev(x[k] for x in g))
               for k, ax in enumerate(('roll', 'pitch', 'yaw'))})
    return steps


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('runs', nargs='+', metavar='label=bag')
    ap.add_argument('--supply', nargs='*', default=[], metavar='label=volts',
                    help='bench supply voltage per run, adds motor_v_<label> columns')
    ap.add_argument('--out', required=True)
    a = ap.parse_args(argv)
    runs = [r.split('=', 1) for r in a.runs]
    supply = {k: float(v) for k, v in (s.split('=', 1) for s in a.supply)}
    res = {label: analyse(path) for label, path in runs}
    first = res[runs[0][0]]
    setpoints = sorted(set.intersection(*(set(r) for r in res.values())))
    head = ['v_cmd_mps', 'axle_rad_s']
    for key in ('duty', 'pitch_sd', 'roll_sd', 'yaw_sd'):
        head += [f'{key}_{label}' + ('' if key == 'duty' else '_dps') for label, _ in runs]
        if key == 'duty':
            head += [f'motor_v_{label}' for label, _ in runs if label in supply]
    with open(a.out, 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(head)
        for v in setpoints:
            row = [v, round(first[v]['axle'], 1)]
            row += [round(res[l][v]['duty']) for l, _ in runs]
            row += [round(res[l][v]['duty'] / 1023 * supply[l], 3) for l, _ in runs
                    if l in supply]
            for ax in ('pitch', 'roll', 'yaw'):
                row += [round(res[l][v][ax], 3) for l, _ in runs]
            w.writerow(row)
    print(f'{len(setpoints)} steps, {len(runs)} runs -> {a.out}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
