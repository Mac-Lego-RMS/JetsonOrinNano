#!/usr/bin/env python3
"""
Offline evaluation of the dead-reckoning EKF against a rosbag2 recording.
Runs WITHOUT a ROS graph: reads the bag, sends gyro/encoder in stamp order
through the pure filter class, plots the state history.

Usage:  python3 eval_ekf.py <bag_folder>     e.g. python3 eval_ekf.py stillstand
"""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')          # headless (SSH/Jetson) -> into a PNG instead of a window
import matplotlib.pyplot as plt

import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message

from ekf.ekf import DeadReckoningEKF   # your ROS-free class

# ---- Configuration ----
IMU_TOPIC = '/bno055/imu'
ENC_TOPIC = '/esp_serial_bridge/joint_states'
R_EFF   = 0.0150       # m, distance-calibrated
R_GYRO  = 2.83e-7      # (rad/s)^2
R_ENC   = 9.3e-4       # (m/s)^2
STORAGE = 'sqlite3'    # Humble default; if you record mcap -> 'mcap'

def stamp_to_sec(stamp):
    return stamp.sec + stamp.nanosec * 1e-9

def read_bag(path):
    """Generator: yields (topic, deserialised_msg) in bag order."""
    reader = rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=path, storage_id=STORAGE),
                rosbag2_py.ConverterOptions('', ''))
    type_map = {t.name: t.type for t in reader.get_all_topics_and_types()}
    while reader.has_next():
        topic, data, _bag_t = reader.read_next()
        msg = deserialize_message(data, get_message(type_map[topic]))
        yield topic, msg

def main():
    zupt_count = 0
    last_gyro_z = 0.0
    v_thresh, w_thresh = 0.03, 5e-3
    GYRO_SCALE = 0.9674   # 5x360deg at a race-typical turn rate: 1800/1860.6

    bag = sys.argv[1] if len(sys.argv) > 1 else 'stillstand'
    bag = os.path.abspath(os.path.expanduser(bag))
    name = os.path.basename(os.path.normpath(bag))   # only the bag name, without the path

    # 1) Collect measurements, WITH header.stamp (not bag time!)
    meas = []
    for topic, msg in read_bag(bag):
        if topic == IMU_TOPIC:
            meas.append((stamp_to_sec(msg.header.stamp), 'gyro',
                         msg.angular_velocity.z * GYRO_SCALE))
        elif topic == ENC_TOPIC:
            v = msg.velocity[0] if len(msg.velocity) else 0.0
            meas.append((stamp_to_sec(msg.header.stamp), 'enc', v * R_EFF))

    if not meas:
        print('No measurements found - check topics/bag.')
        return
    meas.sort(key=lambda m: m[0])     # strictly by stamp -> no negative dt

    # 2) Send through the filter (same predict/update logic as the node,
    #    but without queue/window -> isolates the pure filter maths)
    ekf = DeadReckoningEKF()
    ekf.r_gyro, ekf.r_enc = R_GYRO, R_ENC

    hist = {k: [] for k in ('t', 'x', 'y', 'th', 'v', 'w', 'bg')}
    last = None
    t0 = meas[0][0]
    for t, kind, z in meas:
        if last is not None:
            dt = t - last
            if dt > 0:
                ekf.predict(dt)
        last = t
        if kind == 'gyro':
            ekf.update_gyro(z)
            last_gyro_z = z                  # remember for the standstill check
        else:
            ekf.update_encoder(z)
            if abs(z) < v_thresh and abs(last_gyro_z) < w_thresh:
                ekf.update_zero_motion()     # <-- the new part
                zupt_count += 1

        hist['t'].append(t - t0)
        hist['x'].append(ekf.x[0]);  hist['y'].append(ekf.x[1])
        hist['th'].append(ekf.x[2]); hist['v'].append(ekf.x[3])
        hist['w'].append(ekf.x[4]);  hist['bg'].append(ekf.x[5])

    # 3) Key figures
    dur = hist['t'][-1]
    pos_err = np.hypot(hist['x'][-1], hist['y'][-1])
    th_unwrapped = np.unwrap(hist['th'])          # undoes the +-180 deg jumps
    total_rotation = np.degrees(th_unwrapped[-1] - th_unwrapped[0])
    print(f'Duration:         {dur:.1f} s   ({len(meas)} measurements)')
    print(f"End position:     x: {hist['x'][-1]*1000:.1f} mm  y: {hist['y'][-1]*1000:.1f} mm")
    print(f'Total rotation (unwrapped): {total_rotation:+.1f} deg')
    print(f'Bias estimate:    {np.degrees(hist["bg"][-1]):+.4f} deg/s (final)')
    print(f'Zero-motion fired: {zupt_count} / {sum(1 for m in meas if m[1]=="enc")} encoder measurements')

    # 4) Plots
    fig, ax = plt.subplots(4, 1, figsize=(10, 11), sharex=True)
    ax[0].plot(hist['t'], np.array(hist['x'])*1000, label='x')
    ax[0].plot(hist['t'], np.array(hist['y'])*1000, label='y')
    ax[0].set_ylabel('Position [mm]'); ax[0].legend(); ax[0].grid(alpha=.3)
    ax[1].plot(hist['t'], np.degrees(hist['th']), color='crimson')
    ax[1].set_ylabel('Heading [deg]'); ax[1].grid(alpha=.3)
    ax[2].plot(hist['t'], hist['v'], label='v [m/s]')
    ax[2].plot(hist['t'], hist['w'], label='omega [rad/s]')
    ax[2].set_ylabel('Speed'); ax[2].legend(); ax[2].grid(alpha=.3)
    ax[3].plot(hist['t'], np.degrees(hist['bg']), color='teal')
    ax[3].set_ylabel('Bias b_g [deg/s]'); ax[3].set_xlabel('Time [s]')
    ax[3].grid(alpha=.3)
    fig.suptitle(f'Dead-Reckoning EKF - {name}')
    fig.tight_layout()
    out = f'ekf_eval_{name}.png'
    fig.savefig(out, dpi=110)
    print(f'Plot: {os.path.abspath(out)}')

if __name__ == '__main__':
    main()