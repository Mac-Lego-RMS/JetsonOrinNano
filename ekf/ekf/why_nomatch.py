#!/usr/bin/env python3
"""
Why does nothing match? Innovation inspector at a chosen time.

    python3 -m ekf.why_nomatch <bag> <pose_key> <t_seconds>

For the scan nearest t, prints each measured wall and its innovation (d_alpha,
d_d) against the NEAREST map wall -- ignoring the gate. Small innovations =
near-miss (pose slightly off, gate too tight). Huge innovations = gross
mismatch (pose/map fundamentally wrong on this straight).
"""
import sys
import numpy as np

import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message

from ekf.wall_extraction import (
    scan_to_points, cluster_points, merge_wraparound, split_at_corners,
    fit_wall_hnf, lidar_to_base_link,
)
from ekf.ekf import wrap
from ekf.wall_extraction import predict_wall_in_robot_frame
from ekf.field_map import generate_map, START_POSES_CW, START_POSES_CCW

POSE_KEYS = {
    'pos1_cw': START_POSES_CW['pos1'], 'pos2_cw': START_POSES_CW['pos2'],
    'pos1_ccw': START_POSES_CCW['pos1'], 'pos2_ccw': START_POSES_CCW['pos2'],
}


def yaw(q):
    return np.arctan2(2.0 * (q.w * q.z + q.x * q.y),
                      1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def main():
    bag, pose_key, t_target = sys.argv[1], sys.argv[2], float(sys.argv[3])
    map_walls = generate_map(POSE_KEYS[pose_key])

    reader = rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=bag, storage_id='sqlite3'),
                rosbag2_py.ConverterOptions('', ''))
    tm = {t.name: t.type for t in reader.get_all_topics_and_types()}

    poses, scans, t0 = [], [], None
    while reader.has_next():
        topic, data, _ = reader.read_next()
        if topic == '/ekf/odom':
            m = deserialize_message(data, get_message(tm[topic]))
            t = m.header.stamp.sec + m.header.stamp.nanosec * 1e-9
            p = m.pose.pose
            poses.append((t, (p.position.x, p.position.y, yaw(p.orientation))))
        elif topic == '/scan':
            m = deserialize_message(data, get_message(tm[topic]))
            t = m.header.stamp.sec + m.header.stamp.nanosec * 1e-9
            scans.append((t, m))
        if t0 is None and (poses or scans):
            t0 = (poses or scans)[0][0]

    pose_t = np.array([t for t, _ in poses])
    # find scan nearest t_target (relative to t0)
    best = min(scans, key=lambda s: abs((s[0] - t0) - t_target))
    t, msg = best
    i = int(np.argmin(np.abs(pose_t - t)))
    pose = poses[i][1]
    print(f'scan at t={t - t0:.2f}, pose=({pose[0]:+.2f},{pose[1]:+.2f},'
          f'{np.degrees(pose[2]):+.1f} deg)\n')

    pts = scan_to_points(msg)
    clusters = merge_wraparound(cluster_points(pts))
    split = []
    for c in clusters:
        split.extend(split_at_corners(c))
    measured = [lidar_to_base_link(*h) for c in split
                if (h := fit_wall_hnf(c)) is not None]

    # predicted map walls in robot frame at this pose
    pred = []
    for j, mw in enumerate(map_walls):
        a_p, d_p = predict_wall_in_robot_frame(mw['alpha'], mw['d'], pose)
        pred.append((j, a_p, d_p))

    for (a_m, d_m, ps, pe) in measured:
        # nearest map wall by combined innovation
        best_j, best_da, best_dd, best_cost = None, None, None, np.inf
        for (j, a_p, d_p) in pred:
            da = np.degrees(wrap(a_m - a_p))
            dd = d_m - d_p
            cost = (da / 20.0) ** 2 + (dd / 0.12) ** 2
            if cost < best_cost:
                best_cost, best_j, best_da, best_dd = cost, j, da, dd
        gate = 'IN ' if (abs(best_da) < 20 and abs(best_dd) < 0.12) else 'OUT'
        print(f'  meas a={np.degrees(a_m):+7.1f} d={d_m:+.3f}  -> '
              f'nearest map#{best_j}  innov: da={best_da:+6.1f} deg  '
              f'dd={best_dd:+.3f} m  [{gate}]')


if __name__ == '__main__':
    main()