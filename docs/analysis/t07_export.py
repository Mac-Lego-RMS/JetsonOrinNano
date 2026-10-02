"""Export one full-throttle step (test T07) from a bag to a CSV.

The bag is recorded while `ros2 run esp_bridge step_test` runs:

    ros2 bag record -o t07_16v8_1 /esp_serial_bridge/joint_states \\
        /esp_serial_bridge/motor_state /esp_serial_bridge/battery \\
        /bno055/imu /cmd_vel /t07/stage /ekf/odom

and exported with

    python docs/analysis/t07_export.py t07_16v8_1 --out docs/data/t07/t07_16v8_1.csv

Time base: the ESP's send time of each telemetry frame (header stamp of
joint_states, synchronised to ROS time by the bridge). Encoder, duty and
motor current come from the same 100 Hz frame, so they share that stamp
exactly. The IMU (100 Hz, stamped when the Jetson reads it over I2C) is
interpolated linearly onto those stamps. t = 0 is the first frame after the
/t07/stage step mark whose duty is no longer zero, i.e. the PWM step as the
ESP applied it.

Columns:
  t_s          time since the PWM step [s]
  v_mps        wheel speed from the encoder, w_rad_s x wheel radius [m/s]
  w_rad_s      wheel speed from the encoder [rad/s]
  s_m          encoder distance since t = 0 [m]
  ax_mps2 ...  BNO055 linear acceleration (gravity removed), IMU frame [m/s^2]
  duty         motor duty, -1023..1023 (1023 = 100 % PWM)
  current_A    motor current as reported by the ESP [A]
  voltage_V    last battery reading before the step [V] (published every 5 s)

The export ends --after seconds (default 3) after the motor is cut; the
encoder distance reported is the one at that point.

The wheel radius is the effective rolling radius under load (chapter 1,
tyres): 15.0 mm. A summary line per run is appended to notes.csv next to
the output, with the fields to be filled in by hand left empty.
"""
import argparse
import bisect
import csv
import sqlite3
import sys
from pathlib import Path

from sweep_noise import Cdr, db3_of, dec_f32, dec_f32_array

WHEEL_RADIUS_M = 0.015


def dec_joint(b):
    r = Cdr(b)
    sec, nsec = r.u32(), r.u32()
    r.string()
    for _ in range(r.u32()):
        r.string()
    pos = r.f64s()
    vel = r.f64s()
    return sec + nsec * 1e-9, pos[0], vel[0] if vel else 0.0


def dec_imu(b):
    r = Cdr(b)
    sec, nsec = r.u32(), r.u32()
    r.string()
    r.f64s(4), r.f64s(9)                 # orientation
    r.f64s(3), r.f64s(9)                 # angular velocity
    return sec + nsec * 1e-9, r.f64s(3)  # linear acceleration


def dec_battery(b):
    r = Cdr(b)
    r.u32(), r.u32(), r.string()
    return r.f32()


def read(db, topic, decode):
    tid = db.execute('select id from topics where name=?', (topic,)).fetchone()
    if tid is None:
        sys.exit(f'{topic} not in bag')
    return [(ts / 1e9, decode(bytes(d))) for ts, d in db.execute(
        'select timestamp, data from messages where topic_id=? order by timestamp', tid)]


def interp(ts, values, t):
    i = bisect.bisect_left(ts, t)
    if i == 0:
        return values[0]
    if i >= len(ts):
        return values[-1]
    w = (t - ts[i - 1]) / (ts[i] - ts[i - 1])
    return [a + w * (b - a) for a, b in zip(values[i - 1], values[i])]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('bag')
    ap.add_argument('--out', required=True)
    ap.add_argument('--before', type=float, default=0.5, help='s kept before the step')
    ap.add_argument('--after', type=float, default=3.0,
                    help='s kept after the motor is cut (the car should stand by then)')
    ap.add_argument('--radius', type=float, default=WHEEL_RADIUS_M, help='wheel radius [m]')
    a = ap.parse_args(argv)

    db = sqlite3.connect(f'file:{db3_of(a.bag)}?mode=ro', uri=True)
    stage = read(db, '/t07/stage', dec_f32)
    joints = read(db, '/esp_serial_bridge/joint_states', dec_joint)
    motor = read(db, '/esp_serial_bridge/motor_state', dec_f32_array)
    imu = read(db, '/bno055/imu', dec_imu)
    battery = read(db, '/esp_serial_bridge/battery', dec_battery)

    step = next((t for t, v in stage if v > 0.5), None)
    if step is None:
        sys.exit('no step mark (/t07/stage = 1) in bag')
    cut = next((t for t, v in stage if t > step and v < 0.5), None)

    # duty and current belong to the telemetry frame published right before
    # the joint state; pair them by receive time
    m_rx = [t for t, _ in motor]
    frames = []
    for rx, (stamp, pos, vel) in joints:
        i = bisect.bisect_right(m_rx, rx) - 1
        if i < 0 or rx - m_rx[i] > 0.005:
            continue
        duty, current = motor[i][1][0], motor[i][1][1]
        frames.append((rx, stamp, pos, vel, duty, current))

    start = next((f for f in frames if f[0] >= step - 0.05 and abs(f[4]) > 0), None)
    if start is None:
        sys.exit('duty never left zero after the step mark')
    t0, pos0 = start[1], start[2]
    volts = [v for t, v in battery if t <= step]
    voltage = volts[-1] if volts else float('nan')

    imu_t = [s for _, (s, _) in imu]
    imu_v = [acc for _, (_, acc) in imu]

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    t_end = (cut if cut is not None else frames[-1][0]) + a.after
    rows, s_end = 0, 0.0
    with open(out, 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['t_s', 'v_mps', 'w_rad_s', 's_m', 'ax_mps2', 'ay_mps2', 'az_mps2',
                    'duty', 'current_A', 'voltage_V'])
        for rx, stamp, pos, vel, duty, current in frames:
            t = stamp - t0
            if t < -a.before:
                continue
            if rx > t_end:
                break
            ax, ay, az = interp(imu_t, imu_v, stamp)
            w.writerow([f'{t:.4f}', f'{vel * a.radius:.4f}', f'{vel:.3f}',
                        f'{(pos - pos0) * a.radius:.4f}', f'{ax:.3f}', f'{ay:.3f}',
                        f'{az:.3f}', int(duty), f'{current:.3f}', f'{voltage:.2f}'])
            rows += 1
            s_end = (pos - pos0) * a.radius
    on = [f for f in frames if cut is None or f[0] < cut]
    full = sum(1 for f in on if f[0] >= start[0] and abs(f[4]) >= 1023)
    print(f'{out}: {rows} rows, battery {voltage:.2f} V before the step, '
          f'encoder distance {s_end:.3f} m, '
          f'{full} of {sum(1 for f in on if f[0] >= start[0])} frames at full duty')

    notes = out.parent / 'notes.csv'
    new = not notes.exists()
    with open(notes, 'a', newline='') as fh:
        w = csv.writer(fh)
        if new:
            w.writerow(['run', 'battery_v_before', 'pwm_mode', 'encoder_m', 'tape_m',
                        'remarks'])
        w.writerow([out.stem, f'{voltage:.2f}', '', f'{s_end:.3f}', '', ''])
    return 0


if __name__ == '__main__':
    sys.exit(main())
