#!/usr/bin/env python3
"""
Stage 1 test: LaserScan -> point cloud in ROBOT frame (REP-103), plotted.
Reads a rosbag, converts one scan, and shows the (x, y) scatter so the
coordinate convention (mirroring + blocked-zone removal) can be verified visually.

Usage:  python3 stage1_test.py <bag_dir>
"""
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message

from ekf.ekf import DeadReckoningEKF

SCAN_TOPIC = '/scan'
STORAGE = 'sqlite3'

# Rear blocked zone: a PCB at scan height blocks the back of the LiDAR.
# These angles are in the RAW LiDAR frame (radians), near +/- pi ("behind").
# Drop any point whose |raw angle| exceeds this threshold.
BLOCK_ANGLE = np.radians(60.0)
LIDAR_OFFSET_X = 0.1101   # m, LiDAR rotation centre ahead of rear axle (measured)

def scan_to_points(msg):
    """LaserScan -> (N,2) point cloud in ROBOT frame (REP-103: +x fwd, +y left).

    This is the ONLY place the LiDAR convention (left-hand / CW) is mirrored
    into REP-103 (right-hand / CCW). Downstream code never touches a LiDAR angle.
    """
    ranges = np.asarray(msg.ranges, dtype=np.float64)
    n = len(ranges)
    angles = msg.angle_min + np.arange(n) * msg.angle_increment  # rad, LiDAR frame

    valid = (
        np.isfinite(ranges)
        & (ranges >= msg.range_min)
        & (ranges <= msg.range_max)
        & (np.abs(angles) > BLOCK_ANGLE)      # drop rear blocked zone
    )
    r = ranges[valid]
    a = angles[valid]

    # Mirror LiDAR(CW) -> REP-103(CCW): x stays, y is negated.
    x = -r * np.cos(a)          # cos negated by the 180 deg rotation
    y =  -r * np.sin(a)          # -sin (mirror) then negated again (rotation) = +sin
    return np.column_stack((x, y))

def cluster_points(points, gap_threshold=0.15, min_cluster_size=45):
    """Split an ordered point cloud into clusters at large gaps.

    Uses Manhattan distance (|dx|+|dy|) rather than Euclidean: WRO walls are
    axis-aligned, so points across a 90-degree corner sit diagonally and
    Manhattan measures their gap larger (up to sqrt(2)x), giving sharper corner
    separation. Trade-off: on walls seen at an angle (robot not parallel),
    Manhattan slightly over-fragments. Acceptable at this threshold.
    """
    if len(points) < min_cluster_size:
        return []

    diffs = np.abs(np.diff(points, axis=0))
    dist = diffs[:, 0] + diffs[:, 1]                # Manhattan
    split_idx = np.where(dist >= gap_threshold)[0] + 1
    

    clusters = np.split(points, split_idx)
    return [c for c in clusters if len(c) >= min_cluster_size]

def merge_wraparound(clusters, gap_threshold=0.15):
    """Merge the first and last cluster if they are spatially adjacent.

    The scan wraps at the +/-pi boundary: a wall crossing that boundary is
    split into the last cluster (angles near +pi) and the first (near -pi),
    though the points are physically continuous. If the end of the last
    cluster is within gap_threshold of the start of the first, they are one
    wall — merge them.
    """
    if len(clusters) < 2:
        return clusters

    first, last = clusters[0], clusters[-1]
    # distance between end of last cluster and start of first (Manhattan, matching cluster_points)
    gap = np.abs(last[-1, 0] - first[0, 0]) + np.abs(last[-1, 1] - first[0, 1])

    if gap < gap_threshold:
        clusters[0] = np.vstack((last, first))   # prepend last to first
        clusters.pop()                            # remove the now-merged last
    return clusters

def fit_wall_hnf(cluster):
    """Fit a line to a cluster and return it in Hesse normal form (alpha, d).

    Convention: the normal points toward the field interior. In the robot
    frame the robot (LiDAR origin) is inside the field, so the normal points
    toward the origin. d = signed distance of the line from the origin along
    the normal; it is NEGATIVE under this convention (origin lies on the
    positive-normal side). alpha in radians, in [-pi, pi].

    Returns (alpha, d) or None if the cluster is too small / degenerate.
    """
    if cluster is None or len(cluster) < 3:
        return None

    centroid = cluster.mean(axis=0)
    centered = cluster - centroid
    # SVD: last right-singular vector is the direction of least variance = normal.
    _, _, Vh = np.linalg.svd(centered, full_matrices=False)
    normal = Vh[-1]                          # arbitrary orientation from SVD

    # Orient the normal toward the field interior (= toward the origin, since
    # the robot sits inside). Vector from wall centroid to origin is -centroid.
    if np.dot(normal, -centroid) < 0:
        normal = -normal

    alpha = np.arctan2(normal[1], normal[0])
    d = np.dot(centroid, normal)             # signed; negative under this convention
    return alpha, d

def wrap(a):
    """Wrap angle to [-pi, pi]."""
    return (a + np.pi) % (2 * np.pi) - np.pi


def predict_wall_in_robot_frame(alpha_map, d_map, pose):
    """Measurement model h(x): where a known map wall should appear in the
    robot frame, given the current pose estimate.

        alpha_robot = alpha_map - theta
        d_robot     = d_map - (x*cos(alpha_map) + y*sin(alpha_map))

    pose = (x, y, theta) of the robot in the map frame.
    Returns (alpha_robot, d_robot).
    """
    x, y, theta = pose
    alpha_robot = wrap(alpha_map - theta)
    d_robot = d_map - (x * np.cos(alpha_map) + y * np.sin(alpha_map))
    return alpha_robot, d_robot


def match_walls(measured, map_walls, pose, alpha_tol=np.radians(20.0), d_tol=0.30):
    """Match each measured wall to the nearest map wall via the measurement model.

    For each measured (alpha, d) in the robot frame, predict how every map wall
    should look in the robot frame (via h(x)), then assign the measured wall to
    the map wall with the smallest combined innovation, if within tolerance.

    Args:
        measured:  list of (alpha, d), robot frame (from fit_wall_hnf).
        map_walls: list of (alpha, d), map frame (the fixed map).
        pose:      (x, y, theta) current estimate.
        alpha_tol, d_tol: gates — reject a match whose innovation exceeds these.

    Returns:
        list of dicts: {measured, map, map_index, innov_alpha, innov_d}
    """
    matches = []
    for (a_meas, d_meas) in measured:
        best = None
        best_cost = np.inf
        for j, (a_map, d_map) in enumerate(map_walls):
            a_pred, d_pred = predict_wall_in_robot_frame(a_map, d_map, pose)
            innov_a = wrap(a_meas - a_pred)          # angle innovation, wrapped
            innov_d = d_meas - d_pred                # distance innovation

            # gate: reject implausible matches
            if abs(innov_a) > alpha_tol or abs(innov_d) > d_tol:
                continue

            # combined cost: normalise each term by its tolerance so they are comparable
            cost = (innov_a / alpha_tol) ** 2 + (innov_d / d_tol) ** 2
            if cost < best_cost:
                best_cost = cost
                best = {'measured': (a_meas, d_meas),
                        'map': (a_map, d_map),
                        'map_index': j,
                        'innov_alpha': innov_a,
                        'innov_d': innov_d}
        if best is not None:
            matches.append(best)
    return matches

def lidar_to_base_link(alpha, d, offset_x=LIDAR_OFFSET_X):
    """Shift a wall (alpha, d) from LiDAR frame to rear-axle (base_link) frame.
    The LiDAR sits offset_x ahead of the rear axle (+x). Observing from the rear
    axle shifts the origin by (-offset_x, 0), so d changes by +offset_x*cos(alpha).
    Alpha is translation-invariant (unchanged).
    """
    return alpha, d + offset_x * np.cos(alpha)

def measure_wall_noise(bag_path, map_walls, pose=(0.0, 0.0, 0.0)):
    """Estimate wall-fit measurement noise (R_alpha, R_d) from a static bag.

    Robot stationary -> true walls constant -> variation in fitted (alpha, d)
    across frames is measurement noise. Walls are identified per frame via
    map matching, then grouped by map_index so each map wall's variance is
    measured separately.
    """
    reader = rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=bag_path, storage_id=STORAGE),
                rosbag2_py.ConverterOptions('', ''))
    type_map = {t.name: t.type for t in reader.get_all_topics_and_types()}

    # collect (alpha, d) per map wall across all frames
    samples = {j: {'alpha': [], 'd': []} for j in range(len(map_walls))}
    sizes = {j: [] for j in range(len(map_walls))}

    n_frames = 0
    while reader.has_next():
        topic, data, _ = reader.read_next()
        if topic != SCAN_TOPIC:
            continue
        msg = deserialize_message(data, get_message(type_map[topic]))
        pts = scan_to_points(msg)
        clusters = merge_wraparound(cluster_points(pts))
        measured = [lidar_to_base_link(*hnf)
                    for c in clusters
                    if (hnf := fit_wall_hnf(c)) is not None]
        for m in match_walls(measured, map_walls, pose):
            j = m['map_index']
            samples[j]['alpha'].append(m['measured'][0])
            samples[j]['d'].append(m['measured'][1])

        measured = []
        for c in clusters:
            hnf = fit_wall_hnf(c)
            if hnf is not None:
                a, d = lidar_to_base_link(*hnf)
                measured.append((a, d, len(c)))          # carry point count

        for m in match_walls([(a, d) for (a, d, _) in measured], map_walls, pose):
            j = m['map_index']
            samples[j]['alpha'].append(m['measured'][0])
            samples[j]['d'].append(m['measured'][1])
            # find the cluster size for this matched wall (match by alpha+d identity)
            for (a, d, n) in measured:
                if a == m['measured'][0] and d == m['measured'][1]:
                    sizes[j].append(n)
                    break
        n_frames += 1

    print(f'Frames processed: {n_frames}\n')
    for j, s in samples.items():
        if len(s['d']) < 2:
            print(f'map#{j}: too few matches ({len(s["d"])})')
            continue
        # unwrap alpha before variance (avoid the +-pi jump corrupting it)
        a = np.unwrap(np.array(s['alpha']))
        d = np.array(s['d'])
        var_a = np.var(a)
        var_d = np.var(d)
        print(f'map#{j}  (n={len(d):4d}):  '
              f'var_alpha={var_a:.3e} rad^2 (std={np.degrees(np.sqrt(var_a)):.3f} deg)   '
              f'var_d={var_d:.3e} m^2 (std={np.sqrt(var_d)*1000:.2f} mm)')

    print()
    for j, n_list in sizes.items():
        if n_list:
            arr = np.array(n_list)
            print(f'map#{j}: cluster size  mean={arr.mean():.0f}  min={arr.min()}  '
                f'max={arr.max()}  std={arr.std():.0f}')
    return samples


def read_first_scan(path):
    reader = rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=path, storage_id=STORAGE),
                rosbag2_py.ConverterOptions('', ''))
    type_map = {t.name: t.type for t in reader.get_all_topics_and_types()}
    while reader.has_next():
        topic, data, _ = reader.read_next()
        if topic == SCAN_TOPIC:
            return deserialize_message(data, get_message(type_map[topic]))
    return None


def main():
    bag = sys.argv[1] if len(sys.argv) > 1 else 'scan_bag'
    msg = read_first_scan(bag)
    if msg is None:
        print(f'No {SCAN_TOPIC} messages found in bag.')
        return

    pts = scan_to_points(msg)
    print(f'Raw ranges:   {len(msg.ranges)}')
    print(f'Valid points: {len(pts)}')

    clusters = cluster_points(pts)
    clusters = merge_wraparound(clusters)

    for c in clusters:
        hnf = fit_wall_hnf(c)
        if hnf is None:
            continue
        alpha, d = lidar_to_base_link(*hnf)
        centroid_y = c[:, 1].mean()
        side = 'LEFT (+y)' if centroid_y > 0 else 'RIGHT (-y)'
        print(f'alpha={np.degrees(alpha):+7.1f} deg  d={d:+.3f}  '
              f'centroid_y={centroid_y:+.3f}  -> physically {side}')

    for i, c in enumerate(clusters):
        hnf = fit_wall_hnf(c)
        if hnf:
            alpha, d = hnf
            print(f'cluster {i}: alpha={np.degrees(alpha):+.1f} deg, d={d:+.3f} m, n={len(c)}')

    MAP_WALLS = [
        (np.radians(-90.0), -0.50),   # inner band, left  (unchanged: alpha=+-90 -> cos=0)
        (np.radians(+90.0), -0.50),   # outer band, right (unchanged)
        (np.radians(180.0), -1.84),   # front band: -1.73 (LiDAR) shifted to base_link
    ]                                  # -1.73 + 0.1101*cos(180) = -1.84

    measured = [lidar_to_base_link(*hnf)
        for c in clusters
        if (hnf := fit_wall_hnf(c)) is not None]
    measured = [m for m in measured if m is not None]

    measure_wall_noise(bag, MAP_WALLS)

    # Test 1: true pose (0,0,0) -> innovations should be ~0
    print("Pose (0,0,0):")
    for m in match_walls(measured, MAP_WALLS, (0.0, 0.0, 0.0)):
        print(f"  meas a={np.degrees(m['measured'][0]):+.1f} d={m['measured'][1]:+.3f}"
            f"  -> map#{m['map_index']}"
            f"  innov: a={np.degrees(m['innov_alpha']):+.2f} d={m['innov_d']:+.3f}")

    # Test 2: perturbed pose -> innovations should reflect the offset
    print("Pose (0.1, 0.05, 5deg):")
    for m in match_walls(measured, MAP_WALLS, (0.1, 0.05, np.radians(5.0))):
        print(f"  meas a={np.degrees(m['measured'][0]):+.1f} d={m['measured'][1]:+.3f}"
            f"  -> map#{m['map_index']}"
            f"  innov: a={np.degrees(m['innov_alpha']):+.2f} d={m['innov_d']:+.3f}")

    # start from a deliberately wrong pose, then correct with the matched walls
    ekf = DeadReckoningEKF()
    ekf.x[0], ekf.x[1], ekf.x[2] = 0.1, 0.05, np.radians(5.0)
    print(f"before: x={ekf.x[0]:+.3f} y={ekf.x[1]:+.3f} th={np.degrees(ekf.x[2]):+.2f}")

    for m in match_walls(measured, MAP_WALLS, tuple(ekf.x[:3])):
        (a_meas, d_meas) = m['measured']
        (a_map, d_map)   = m['map']
        ekf.update_wall(a_meas, d_meas, a_map, d_map)

    print(f"after:  x={ekf.x[0]:+.3f} y={ekf.x[1]:+.3f} th={np.degrees(ekf.x[2]):+.2f}")

    fig, ax = plt.subplots(figsize=(9, 9))
    print(f'Clusters found: {len(clusters)}')
    for i, c in enumerate(clusters):
        print(f'  cluster {i}: {len(c)} points')
        ax.scatter(c[:, 0], c[:, 1], s=3, label=f'cluster {i}')
    #ax.scatter(pts[:, 0], pts[:, 1], s=2, c='teal')
    ax.plot(0, 0, 'r^', markersize=12, label='robot (base_link)')  # robot at origin, facing +x
    ax.axhline(0, color='gray', lw=0.5)
    ax.axvline(0, color='gray', lw=0.5)
    # annotate directions to catch mirror errors at a glance
    ax.annotate('+x (forward)', xy=(0.5, 0.05))
    ax.annotate('+y (left)',    xy=(0.05, 0.5))
    ax.set_xlabel('x [m]'); ax.set_ylabel('y [m]')
    ax.set_aspect('equal'); ax.grid(alpha=0.3); ax.legend()
    ax.set_title('Stage 1 - point cloud in robot frame')
    out = 'stage1_points.png'
    fig.savefig(out, dpi=110)
    print(f'Plot: {out}')


if __name__ == '__main__':
    main()