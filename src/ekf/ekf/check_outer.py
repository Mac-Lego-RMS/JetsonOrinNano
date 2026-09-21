#!/usr/bin/env python3
"""
Check outer-band matching over a whole run.

    python3 -m ekf.check_outer <bag> <pose_key>

pose_key in {pos1_cw, pos2_cw, pos1_ccw, pos2_ccw}. For every scan it extracts
walls, matches them against outer_walls_map(start_pose) using the pose from
/ekf/odom at (nearest) that time, and prints the match count. A stable outer
match count through the corner means the outer-band map holds up; a drop to 0
or wild swings flags where matching breaks.
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
from ekf.field_map import outer_walls_map, START_POSES_CW, START_POSES_CCW

POSE_KEYS = {
    'pos1_cw': START_POSES_CW['pos1'], 'pos2_cw': START_POSES_CW['pos2'],
    'pos1_ccw': START_POSES_CCW['pos1'], 'pos2_ccw': START_POSES_CCW['pos2'],
}


def yaw(q):
    return np.arctan2(2.0 * (q.w * q.z + q.x * q.y),
                      1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def main():
    bag, pose_key = sys.argv[1], sys.argv[2]
    start_pose = POSE_KEYS[pose_key]
    map_walls = outer_walls_map(start_pose)

    reader = rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=bag, storage_id='sqlite3'),
                rosbag2_py.ConverterOptions('', ''))
    tm = {t.name: t.type for t in reader.get_all_topics_and_types()}

    # collect odom poses with timestamps
    poses = []            # (t, (x,y,theta))
    scans = []            # (t, msg)
    t0 = None
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
        print('no /ekf/odom in bag -- run this with the EKF running')
        return

    pose_t = np.array([t for t, _ in poses])
    for t, msg in scans:
        # nearest pose in time
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
        print(f't={t - t0:6.2f}  pose=({pose[0]:+.2f},{pose[1]:+.2f},'
              f'{np.degrees(pose[2]):+6.1f})  measured={len(measured)}  '
              f'outer_matched={len(matches)}')


if __name__ == '__main__':
    main()