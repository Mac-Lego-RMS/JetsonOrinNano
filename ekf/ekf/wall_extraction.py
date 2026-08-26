#!/usr/bin/env python3
"""
Wall extraction pipeline (ROS-free) for LiDAR-based wall correction.

Stages:
  1. scan_to_points        LaserScan  -> (N,2) point cloud, robot frame (REP-103)
  2. cluster_points        point cloud -> list of clusters (gap split)
     merge_wraparound      merge a wall split across the +/-pi scan boundary
  3. fit_wall_hnf          cluster     -> (alpha, d) Hesse normal form, robot frame
     lidar_to_base_link    shift (alpha, d) from LiDAR to rear-axle frame
  4. match_walls           measured walls + map walls + pose -> matched pairs
     predict_wall_in_robot_frame   measurement model h(x)

All angles in radians. Convention: normal points into the field interior,
d is signed (negative when the robot/origin is on the positive-normal side).

This module has NO ROS dependencies so it can be imported by the nodes AND
exercised offline against rosbags in the test scripts.
"""
import numpy as np

from ekf.ekf import wrap   # single source of truth for angle wrapping

# --- calibration constants -------------------------------------------------
# Rear blocked zone: a PCB at scan height blocks part of the LiDAR. Angles are
# in the RAW LiDAR frame (radians). Points with |raw angle| <= BLOCK_ANGLE are
# dropped. Measured PCB span was -33..+59 deg (asymmetric); symmetric 60 deg cut
# is safe for now. RE-MEASURE after the next chassis rebuild.
BLOCK_ANGLE = np.radians(60.0)

# LiDAR rotation centre sits this far ahead of the rear axle on +x (measured).
LIDAR_OFFSET_X = 0.1101


# --- stage 1 ---------------------------------------------------------------
def scan_to_points(msg):
    """LaserScan -> (N,2) point cloud in ROBOT frame (REP-103: +x fwd, +y left).

    The ONLY place the LiDAR convention is converted to REP-103. The S3 is
    mounted rotated 180 deg about its z-axis, and uses a left-hand / CW frame.
    Combined transform: x = -r*cos(a), y = +r*sin(a). Downstream code never
    touches a raw LiDAR angle again.
    """
    ranges = np.asarray(msg.ranges, dtype=np.float64)
    n = len(ranges)
    angles = msg.angle_min + np.arange(n) * msg.angle_increment  # rad, LiDAR frame

    valid = (
        np.isfinite(ranges)
        & (ranges >= msg.range_min)
        & (ranges <= msg.range_max)
        & (np.abs(angles) > BLOCK_ANGLE)      # drop rear PCB blocked zone
    )
    r = ranges[valid]
    a = angles[valid]

    x = -r * np.cos(a)          # cos negated by the 180 deg mount rotation
    y = r * np.sin(a)           # -sin (CW->CCW mirror) negated again (rotation) = +sin
    return np.column_stack((x, y))


# --- stage 2 ---------------------------------------------------------------
def cluster_points(points, gap_threshold=0.15, min_cluster_size=45):
    """Split an ordered point cloud into clusters at large gaps.

    Manhattan distance (|dx|+|dy|), not Euclidean: WRO walls are axis-aligned,
    so points across a 90-degree corner sit diagonally and Manhattan measures
    their gap larger (up to sqrt(2)x), giving sharper corner separation.
    """
    if len(points) < min_cluster_size:
        return []

    diffs = np.abs(np.diff(points, axis=0))
    dist = diffs[:, 0] + diffs[:, 1]                # Manhattan
    split_idx = np.where(dist >= gap_threshold)[0] + 1

    clusters = np.split(points, split_idx)
    return [c for c in clusters if len(c) >= min_cluster_size]


def merge_wraparound(clusters, gap_threshold=0.15):
    """Merge first and last cluster if spatially adjacent across the +/-pi seam.

    A wall crossing the scan's +/-pi boundary is split into the last cluster
    (angles near +pi) and the first (near -pi) though physically continuous.
    """
    if len(clusters) < 2:
        return clusters

    first, last = clusters[0], clusters[-1]
    gap = np.abs(last[-1, 0] - first[0, 0]) + np.abs(last[-1, 1] - first[0, 1])

    if gap < gap_threshold:
        clusters[0] = np.vstack((last, first))
        clusters.pop()
    return clusters


# --- stage 3 ---------------------------------------------------------------
def fit_wall_hnf(cluster):
    """Fit a line to a cluster, return Hesse normal form (alpha, d), robot frame.

    Normal is oriented toward the field interior (= toward the origin, since the
    robot sits inside). d is signed and negative under this convention. Returns
    (alpha, d) or None if the cluster is too small.
    """
    if cluster is None or len(cluster) < 3:
        return None

    centroid = cluster.mean(axis=0)
    centered = cluster - centroid
    _, _, Vh = np.linalg.svd(centered, full_matrices=False)
    normal = Vh[-1]                          # arbitrary orientation from SVD

    if np.dot(normal, -centroid) < 0:        # orient toward origin (field interior)
        normal = -normal

    alpha = np.arctan2(normal[1], normal[0])
    d = np.dot(centroid, normal)             # signed; negative under this convention
    return alpha, d


def lidar_to_base_link(alpha, d, offset_x=LIDAR_OFFSET_X):
    """Shift a wall (alpha, d) from LiDAR frame to rear-axle (base_link) frame.
    Observing from the rear axle shifts the origin by (-offset_x, 0), so d
    changes by +offset_x*cos(alpha). Alpha is translation-invariant.
    """
    return alpha, d + offset_x * np.cos(alpha)


# --- stage 4 ---------------------------------------------------------------
def predict_wall_in_robot_frame(alpha_map, d_map, pose):
    """Measurement model h(x): how a map wall should appear in the robot frame.

        alpha_robot = alpha_map - theta
        d_robot     = d_map - (x*cos(alpha_map) + y*sin(alpha_map))

    pose = (x, y, theta) in the map frame. Returns (alpha_robot, d_robot).
    """
    x, y, theta = pose
    alpha_robot = wrap(alpha_map - theta)
    d_robot = d_map - (x * np.cos(alpha_map) + y * np.sin(alpha_map))
    return alpha_robot, d_robot


def match_walls(measured, map_walls, pose, alpha_tol=np.radians(20.0), d_tol=0.30):
    """Match each measured wall to the nearest map wall via the measurement model.

    Args:
        measured:  list of (alpha, d), robot frame (from fit_wall_hnf).
        map_walls: list of (alpha, d), map frame.
        pose:      (x, y, theta) current estimate.
        alpha_tol, d_tol: gates rejecting implausible matches.

    Returns list of dicts: {measured, map, map_index, innov_alpha, innov_d}.
    """
    matches = []
    for (a_meas, d_meas) in measured:
        best = None
        best_cost = np.inf
        for j, (a_map, d_map) in enumerate(map_walls):
            a_pred, d_pred = predict_wall_in_robot_frame(a_map, d_map, pose)
            innov_a = wrap(a_meas - a_pred)
            innov_d = d_meas - d_pred

            if abs(innov_a) > alpha_tol or abs(innov_d) > d_tol:
                continue

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