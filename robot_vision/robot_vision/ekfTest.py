#!/usr/bin/env python3
"""
Offline-Auswertung des Dead-Reckoning-EKF gegen eine rosbag2-Aufnahme.
Laeuft OHNE ROS-Graph: liest die Bag, schickt Gyro/Encoder in Stamp-Reihenfolge
durch die reine Filter-Klasse, plottet den State-Verlauf.

Aufruf:  python3 eval_ekf.py <bag_ordner>     z.B. python3 eval_ekf.py stillstand
"""
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')          # headless (SSH/Jetson) -> in PNG statt Fenster
import matplotlib.pyplot as plt

import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message

from ekf import DeadReckoningEKF   # deine ROS-freie Klasse

# ---- Konfiguration ----
IMU_TOPIC = '/bno055/imu'
ENC_TOPIC = '/esp_serial_bridge/joint_states'
R_EFF   = 0.0150       # m, Strecken-kalibriert
R_GYRO  = 2.83e-7      # (rad/s)^2
R_ENC   = 9.3e-4       # (m/s)^2
STORAGE = 'sqlite3'    # Humble-Default; falls ihr mcap aufnehmt -> 'mcap'

def stamp_to_sec(stamp):
    return stamp.sec + stamp.nanosec * 1e-9

def read_bag(path):
    """Generator: liefert (topic, deserialisierte_msg) in Bag-Reihenfolge."""
    reader = rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=path, storage_id=STORAGE),
                rosbag2_py.ConverterOptions('', ''))
    type_map = {t.name: t.type for t in reader.get_all_topics_and_types()}
    while reader.has_next():
        topic, data, _bag_t = reader.read_next()
        msg = deserialize_message(data, get_message(type_map[topic]))
        yield topic, msg

def main():
    bag = sys.argv[1] if len(sys.argv) > 1 else 'stillstand'

    # 1) Messungen einsammeln, MIT header.stamp (nicht Bag-Zeit!)
    meas = []
    for topic, msg in read_bag(bag):
        if topic == IMU_TOPIC:
            meas.append((stamp_to_sec(msg.header.stamp), 'gyro',
                         msg.angular_velocity.z))
        elif topic == ENC_TOPIC:
            v = msg.velocity[0] if len(msg.velocity) else 0.0
            meas.append((stamp_to_sec(msg.header.stamp), 'enc', v * R_EFF))

    if not meas:
        print('Keine Messungen gefunden - Topics/Bag pruefen.')
        return
    meas.sort(key=lambda m: m[0])     # strikt nach Stamp -> kein negatives dt

    # 2) Durch den Filter schicken (gleiche predict/update-Logik wie der Node,
    #    aber ohne Queue/Window -> isoliert die reine Filter-Mathe)
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
        else:
            ekf.update_encoder(z)
        hist['t'].append(t - t0)
        hist['x'].append(ekf.x[0]);  hist['y'].append(ekf.x[1])
        hist['th'].append(ekf.x[2]); hist['v'].append(ekf.x[3])
        hist['w'].append(ekf.x[4]);  hist['bg'].append(ekf.x[5])

    # 3) Kennzahlen
    dur = hist['t'][-1]
    pos_err = np.hypot(hist['x'][-1], hist['y'][-1])
    th_drift = np.degrees(hist['th'][-1] - hist['th'][0])
    print(f'Dauer:            {dur:.1f} s   ({len(meas)} Messungen)')
    print(f'End-Position:     {pos_err*1000:.1f} mm  (soll ~0)')
    print(f'Heading-Drift:    {th_drift:+.2f} grad  ->  {th_drift/dur:+.3f} grad/s')
    print(f'Bias-Schaetzung:  {np.degrees(hist["bg"][-1]):+.4f} grad/s (final)')

    # 4) Plots
    fig, ax = plt.subplots(4, 1, figsize=(10, 11), sharex=True)
    ax[0].plot(hist['t'], np.array(hist['x'])*1000, label='x')
    ax[0].plot(hist['t'], np.array(hist['y'])*1000, label='y')
    ax[0].set_ylabel('Position [mm]'); ax[0].legend(); ax[0].grid(alpha=.3)
    ax[1].plot(hist['t'], np.degrees(hist['th']), color='crimson')
    ax[1].set_ylabel('Heading [grad]'); ax[1].grid(alpha=.3)
    ax[2].plot(hist['t'], hist['v'], label='v [m/s]')
    ax[2].plot(hist['t'], hist['w'], label='omega [rad/s]')
    ax[2].set_ylabel('Geschw.'); ax[2].legend(); ax[2].grid(alpha=.3)
    ax[3].plot(hist['t'], np.degrees(hist['bg']), color='teal')
    ax[3].set_ylabel('Bias b_g [grad/s]'); ax[3].set_xlabel('Zeit [s]')
    ax[3].grid(alpha=.3)
    fig.suptitle(f'Dead-Reckoning EKF - {bag}')
    fig.tight_layout()
    out = f'ekf_eval_{bag}.png'
    fig.savefig(out, dpi=110)
    print(f'Plot: {out}')

if __name__ == '__main__':
    main()