#!/usr/bin/env python3
"""
Offline bag analysis for push/drive tests.

Mode 1 -- pose timeline:
    python3 plot_pose_bag.py <bag>
    Plots x, y, theta from /ekf/odom over time. Use it to spot jumps/drift.

Mode 2 -- per-frame extraction at a chosen time:
    python3 plot_pose_bag.py <bag> <t_seconds>
    Finds the /scan nearest t_seconds (from bag start), runs the extraction
    pipeline, and plots the clusters + HNF fits. Use it to inspect a frame
    where the pose jumped (e.g. a corner fitted as one cluster).

Imports extraction from wall_extraction so this matches the live nodes exactly.
"""
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message

from ekf.wall_extraction import (
    scan_to_points, cluster_points, merge_wraparound, split_at_corners,
    fit_wall_hnf, lidar_to_base_link,
)

ODOM_TOPIC = '/ekf/odom'
SCAN_TOPIC = '/scan'
STORAGE = 'sqlite3'


def stamp_to_sec(stamp):
    return stamp.sec + stamp.nanosec * 1e-9


def yaw_from_quaternion(q):
    siny = 2.0 * (q.w * q.z + q.x * q.y)
    cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return np.arctan2(siny, cosy)


def _reader(bag_path):
    reader = rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=bag_path, storage_id=STORAGE),
                rosbag2_py.ConverterOptions('', ''))
    type_map = {t.name: t.type for t in reader.get_all_topics_and_types()}
    return reader, type_map


def read_odom(bag_path):
    """Return arrays t, x, y, theta from /ekf/odom (t relative to first msg)."""
    reader, type_map = _reader(bag_path)
    ts, xs, ys, ths = [], [], [], []
    while reader.has_next():
        topic, data, _ = reader.read_next()
        if topic != ODOM_TOPIC:
            continue
        msg = deserialize_message(data, get_message(type_map[topic]))
        ts.append(stamp_to_sec(msg.header.stamp))
        xs.append(msg.pose.pose.position.x)
        ys.append(msg.pose.pose.position.y)
        ths.append(yaw_from_quaternion(msg.pose.pose.orientation))
    if not ts:
        return None
    t0 = ts[0]
    t = np.array(ts) - t0
    return t, np.array(xs), np.array(ys), np.array(ths)


def read_scan_near(bag_path, t_target):
    """Return the LaserScan whose stamp is nearest t_target (sec from bag start)."""
    reader, type_map = _reader(bag_path)
    first_stamp = None
    best_msg, best_dt = None, np.inf
    while reader.has_next():
        topic, data, _ = reader.read_next()
        if topic != SCAN_TOPIC:
            continue
        msg = deserialize_message(data, get_message(type_map[topic]))
        s = stamp_to_sec(msg.header.stamp)
        if first_stamp is None:
            first_stamp = s
        rel = s - first_stamp
        dt = abs(rel - t_target)
        if dt < best_dt:
            best_dt, best_msg = dt, msg
    return best_msg, best_dt


def plot_pose(bag_path):
    result = read_odom(bag_path)
    if result is None:
        print(f'No {ODOM_TOPIC} messages in bag.')
        return
    t, x, y, th = result
    print(f'{len(t)} pose samples over {t[-1]:.1f} s')

    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    axes[0].plot(t, x, '-')
    axes[0].set_ylabel('x [m]')
    axes[0].grid(True)
    axes[1].plot(t, y, '-')
    axes[1].set_ylabel('y [m]')
    axes[1].grid(True)
    axes[2].plot(t, np.degrees(th), '-')
    axes[2].set_ylabel('theta [deg]')
    axes[2].set_xlabel('time [s]')
    axes[2].grid(True)
    fig.suptitle('EKF pose over time')
    fig.tight_layout()
    out = 'pose_timeline.png'
    fig.savefig(out, dpi=110)
    print(f'saved {out}')


def plot_frame(bag_path, t_target):
    msg, dt = read_scan_near(bag_path, t_target)
    if msg is None:
        print(f'No {SCAN_TOPIC} messages in bag.')
        return
    print(f'nearest scan is {dt*1e3:.1f} ms from t={t_target:.2f} s')

    pts = scan_to_points(msg)
    clusters = merge_wraparound(cluster_points(pts))
    # split any cluster that contains a corner (L-shape) into straight walls
    split = []
    for c in clusters:
        split.extend(split_at_corners(c))
    clusters = split
    print(f'{len(clusters)} clusters')

    fig, ax = plt.subplots(figsize=(9, 9))
    for i, c in enumerate(clusters):
        ax.scatter(c[:, 0], c[:, 1], s=5, label=f'cluster {i} (n={len(c)})')
        hnf = fit_wall_hnf(c)
        if hnf is not None:
            alpha, d, p_start, p_end = lidar_to_base_link(*hnf)
            print(f'  cluster {i}: alpha={np.degrees(alpha):+7.1f} deg  '
                  f'd={d:+.3f} m  n={len(c)}')
    ax.plot(0, 0, 'r^', markersize=12, label='robot')
    ax.set_aspect('equal')
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.grid(True)
    ax.legend(loc='upper right', fontsize=8)
    fig.suptitle(f'Extraction at t={t_target:.2f} s')
    out = f'frame_{t_target:.2f}.png'
    fig.savefig(out, dpi=110)
    print(f'saved {out}')


def main():
    if len(sys.argv) < 2:
        print('usage: plot_pose_bag.py <bag> [t_seconds]')
        return
    bag = sys.argv[1]
    if len(sys.argv) >= 3:
        plot_frame(bag, float(sys.argv[2]))
    else:
        plot_pose(bag)


if __name__ == '__main__':
    main()