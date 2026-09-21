#!/usr/bin/env python3
"""
Offline matching test against a verification bag.

    python3 -m ekf.test_matching <bag> <pose_key> [t_seconds]

pose_key is one of: pos1_cw, pos2_cw, pos1_ccw, pos2_ccw  (selects the start
pose whose generated map is matched against the scan). Prints, per measured
wall, whether it matched a map wall (and which) or was rejected -- so overlap
gating can be verified: park walls / foreign clusters should be rejected.
"""
import sys
import numpy as np

import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message

from ekf.wall_extraction import (
    scan_to_points, cluster_points, merge_wraparound, split_at_corners,
    fit_wall_hnf, lidar_to_base_link, match_walls,
)
from ekf.field_map import generate_map, START_POSES_CW, START_POSES_CCW

SCAN_TOPIC = '/scan'
STORAGE = 'sqlite3'

POSE_KEYS = {
    'pos1_cw': START_POSES_CW['pos1'],
    'pos2_cw': START_POSES_CW['pos2'],
    'pos1_ccw': START_POSES_CCW['pos1'],
    'pos2_ccw': START_POSES_CCW['pos2'],
}


def read_scan_near(bag_path, t_target):
    reader = rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=bag_path, storage_id=STORAGE),
                rosbag2_py.ConverterOptions('', ''))
    type_map = {t.name: t.type for t in reader.get_all_topics_and_types()}
    first, best, best_dt = None, None, np.inf
    while reader.has_next():
        topic, data, _ = reader.read_next()
        if topic != SCAN_TOPIC:
            continue
        msg = deserialize_message(data, get_message(type_map[topic]))
        s = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        if first is None:
            first = s
        if abs((s - first) - t_target) < best_dt:
            best_dt, best = abs((s - first) - t_target), msg
    return best


def main():
    if len(sys.argv) < 3:
        print('usage: test_matching.py <bag> <pose_key> [t_seconds]')
        print('pose_key:', ', '.join(POSE_KEYS))
        return
    bag = sys.argv[1]
    pose_key = sys.argv[2]
    t = float(sys.argv[3]) if len(sys.argv) > 3 else 0.5

    field_start = POSE_KEYS[pose_key]
    map_walls = generate_map(field_start)
    # the robot's own pose in ITS map frame is always (0,0,0) at start
    robot_pose = (0.0, 0.0, 0.0)

    msg = read_scan_near(bag, t)
    pts = scan_to_points(msg)
    clusters = merge_wraparound(cluster_points(pts))
    split = []
    for c in clusters:
        split.extend(split_at_corners(c))
    clusters = split

    measured = []
    for c in clusters:
        hnf = fit_wall_hnf(c)
        if hnf is not None:
            measured.append(lidar_to_base_link(*hnf))

    print(f'{len(measured)} measured walls, {len(map_walls)} map walls\n')

    matches = match_walls(measured, map_walls, robot_pose)
    matched_meas = {(round(m['measured'][0], 4), round(m['measured'][1], 4))
                    for m in matches}

    print('MEASURED WALLS:')
    for a, d, ps, pe in measured:
        key = (round(a, 4), round(d, 4))
        status = 'MATCHED' if key in matched_meas else 'rejected'
        print(f'  alpha={np.degrees(a):+7.1f}  d={d:+.3f}  '
              f'extent=({ps[0]:+.2f},{ps[1]:+.2f})->({pe[0]:+.2f},{pe[1]:+.2f})  '
              f'-> {status}')

    print('\nMATCHES:')
    for m in matches:
        print(f"  meas alpha={np.degrees(m['measured'][0]):+7.1f} d={m['measured'][1]:+.3f}"
              f"  -> map#{m['map_index']}"
              f"  innov: a={np.degrees(m['innov_alpha']):+.2f} d={m['innov_d']:+.3f}")


if __name__ == '__main__':
    main()