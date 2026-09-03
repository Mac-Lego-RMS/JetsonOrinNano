#!/usr/bin/env python3
"""
Check FULL-map matching over a whole run (obstacle mode).

    python3 -m ekf.check_full <bag> <pose_key>

pose_key in {pos1_cw, pos2_cw, pos1_ccw, pos2_ccw}. Matches every scan against
generate_map(start_pose) using the /ekf/odom pose at that time, and prints the
match count plus which map-wall indices matched. Run it on a GOOD and a BAD run
and diff the output to find where matching diverges on the second straight.
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

POSE_KEYS = {
    'pos1_cw': START_POSES_CW['pos1'], 'pos2_cw': START_POSES_CW['pos2'],
    'pos1_ccw': START_POSES_CCW['pos1'], 'pos2_ccw': START_POSES_CCW['pos2'],
}


def yaw(q):
    return np.arctan2(2.0 * (q.w * q.z + q.x * q.y),
                      1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def main():
    bag, pose_key = sys.argv[1], sys.argv[2]
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

    if not poses:
        print('no /ekf/odom in bag')
        return

    pose_t = np.array([t for t, _ in poses])
    for t, msg in scans:
        i = int(np.argmin(np.abs(pose_t - t)))
        pose = poses[i][1]
        pts = scan_to_points(msg)
        clusters = merge_wraparound(cluster_points(pts))
        split = []
        for c in clusters:
            split.extend(split_at_corners(c))
        measured = [lidar_to_base_link(*h) for c in split
                    if (h := fit_wall_hnf(c)) is not None]
        matches = match_walls(measured, map_walls, pose, d_tol=0.12)
        idx = sorted(m['map_index'] for m in matches)
        print(f't={t - t0:6.2f}  pose=({pose[0]:+.2f},{pose[1]:+.2f},'
              f'{np.degrees(pose[2]):+6.1f})  measured={len(measured)}  '
              f'matched={len(matches)}  map_idx={idx}')


if __name__ == '__main__':
    main()