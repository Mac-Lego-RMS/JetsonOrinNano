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
MAX_RANGE = 4.0

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
        & (ranges <= MAX_RANGE)               # drop points beyond the field
        & (np.abs(angles) > BLOCK_ANGLE)      # drop rear PCB blocked zone
    )
    r = ranges[valid]
    a = angles[valid]

    x = -r * np.cos(a)          # cos negated by the 180 deg mount rotation
    y = -r * np.sin(a)           # -sin (CW->CCW mirror) negated again (rotation) = +sin
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

def split_at_corners(cluster, max_dev=0.04, min_segment_size=65):
    """Split a cluster at corners using iterative split-and-merge.

    A straight wall's points lie within a few mm of the line through its first
    and last point. An L-shaped cluster (two walls meeting at a corner without a
    gap) has a point far from that line -- the corner. Split there and repeat on
    both halves.

    Uses an explicit stack (no recursion). Segments are kept in scan order.

    Args:
        cluster: (N, 2) points in scan order.
        max_dev: max perpendicular distance (m) of a point from the first-last
                 line before the segment is considered bent (a corner). Above
                 the wall-fit noise (~2 mm), below a real corner (>0.1 m).
        min_segment_size: segments shorter than this are not split further and
                 are dropped if produced by a split (matches cluster_points).

    Returns:
        list of (M, 2) arrays, one per straight wall segment.
    """
    if cluster is None or len(cluster) < min_segment_size:
        return [cluster] if cluster is not None and len(cluster) >= 3 else []

    segments = []
    stack = [cluster]                      # segments still to check

    while stack:
        seg = stack.pop()
        if len(seg) < min_segment_size:
            continue                       # too short to be a reliable wall

        p_first = seg[0]
        p_last = seg[-1]
        line = p_last - p_first
        line_len = np.hypot(line[0], line[1])

        if line_len < 1e-6:
            # first and last coincide (degenerate) -> keep as-is
            segments.append(seg)
            continue

        # perpendicular distance of every point to the first-last line.
        # normal to the line direction, normalised:
        normal = np.array([-line[1], line[0]]) / line_len
        dev = np.abs((seg - p_first) @ normal)   # (M,) distances

        idx = np.argmax(dev)
        if dev[idx] > max_dev:
            # corner at idx -> split into [0..idx] and [idx..end].
            # include idx in both so neither segment loses the corner point.
            stack.append(seg[:idx + 1])
            stack.append(seg[idx:])
        else:
            segments.append(seg)           # straight enough -> a wall

    return segments


# --- stage 3 ---------------------------------------------------------------
def fit_wall_hnf(cluster):
    """Fit a line to a cluster and return HNF (alpha, d) plus its endpoints.

    Convention: normal points toward the field interior (toward the origin,
    since the robot sits inside). d is signed, negative under this convention.

    Endpoints are the first and last cluster points PROJECTED onto the fitted
    line -- the clean extent of the wall along its own direction, free of
    cross-noise. They let match_walls check that a measured wall overlaps the
    map segment it is matched to.

    Returns (alpha, d, p_start, p_end) or None if the cluster is too small.
    p_start, p_end are (2,) arrays in the robot/base_link frame.
    """
    if cluster is None or len(cluster) < 3:
        return None

    centroid = cluster.mean(axis=0)
    centered = cluster - centroid
    _, _, Vh = np.linalg.svd(centered, full_matrices=False)
    normal = Vh[-1]
    direction = Vh[0]                        # first singular vector = along the wall

    if np.dot(normal, -centroid) < 0:        # orient toward origin (field interior)
        normal = -normal

    alpha = np.arctan2(normal[1], normal[0])
    d = np.dot(centroid, normal)

    # project first and last point onto the fitted line to get clean endpoints
    t_first = np.dot(cluster[0] - centroid, direction)
    t_last = np.dot(cluster[-1] - centroid, direction)
    p_start = centroid + t_first * direction
    p_end = centroid + t_last * direction

    return alpha, d, p_start, p_end


def lidar_to_base_link(alpha, d, p_start, p_end, offset_x=LIDAR_OFFSET_X):
    """Shift a wall from LiDAR frame to rear-axle (base_link) frame.

    Observing from the rear axle shifts the origin by (-offset_x, 0), so:
      - d changes by +offset_x*cos(alpha); alpha is translation-invariant
      - endpoints shift by +offset_x in x (LiDAR sits offset_x ahead on +x)
    """
    d_bl = d + offset_x * np.cos(alpha)
    shift = np.array([offset_x, 0.0])
    return alpha, d_bl, p_start + shift, p_end + shift


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


def _overlap_along_direction(a1, a2, b1, b2, tol=0.05):
    """Do segments [a1,a2] and [b1,b2] overlap when projected onto the line
    through a1->a2? Returns True if the projected intervals overlap (with a
    small tolerance tol in metres at the ends).

    a1,a2 = measured endpoints; b1,b2 = map segment endpoints (all (2,) arrays).
    """
    d = a2 - a1
    length = np.hypot(d[0], d[1])
    if length < 1e-6:
        return True                      # degenerate measured wall: don't gate
    u = d / length                       # unit direction along the measured wall

    # project all four points onto u
    ta = np.array([np.dot(a1, u), np.dot(a2, u)])
    tb = np.array([np.dot(b1, u), np.dot(b2, u)])
    a_lo, a_hi = ta.min(), ta.max()
    b_lo, b_hi = tb.min(), tb.max()

    # intervals overlap if each starts before the other ends (with tolerance)
    return (a_lo <= b_hi + tol) and (b_lo <= a_hi + tol)


def match_walls(measured, map_walls, pose,
                alpha_tol=np.radians(20.0), d_tol=0.30, overlap_tol=0.05):
    """Match each measured wall to the nearest map wall, with overlap gating.

    Args:
        measured:  list of (alpha, d, p_start, p_end), robot frame.
        map_walls: list of dicts {'alpha','d','p1','p2'} in the map frame
                   (from generate_map), OR list of (alpha, d) tuples (legacy;
                   then overlap gating is skipped for that wall).
        pose:      (x, y, theta) current estimate.
        alpha_tol, d_tol: innovation gates.
        overlap_tol: end tolerance (m) for the overlap check.

    Returns list of dicts: {measured, map, map_index, innov_alpha, innov_d}.
    """
    matches = []
    for meas in measured:
        a_meas, d_meas = meas[0], meas[1]
        has_endpoints = len(meas) >= 4
        if has_endpoints:
            m_start, m_end = meas[2], meas[3]

        best = None
        best_cost = np.inf
        for j, mw in enumerate(map_walls):
            # support both dict map walls (with endpoints) and legacy tuples
            if isinstance(mw, dict):
                a_map, d_map = mw['alpha'], mw['d']
                map_p1, map_p2 = mw['p1'], mw['p2']
            else:
                a_map, d_map = mw[0], mw[1]
                map_p1 = map_p2 = None

            a_pred, d_pred = predict_wall_in_robot_frame(a_map, d_map, pose)
            innov_a = wrap(a_meas - a_pred)
            innov_d = d_meas - d_pred

            if abs(innov_a) > alpha_tol or abs(innov_d) > d_tol:
                continue

            # overlap gate: only if both sides carry endpoints. Map endpoints
            # are in the MAP frame, measured endpoints in the robot frame, so
            # project the map segment into the robot frame first via the pose.
            if has_endpoints and map_p1 is not None:
                mp1 = _map_point_to_robot(map_p1, pose)
                mp2 = _map_point_to_robot(map_p2, pose)
                if not _overlap_along_direction(m_start, m_end, mp1, mp2,
                                                tol=overlap_tol):
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


def _map_point_to_robot(p_map, pose):
    """Transform a point from the map frame into the robot/base_link frame."""
    x, y, th = pose
    dx, dy = p_map[0] - x, p_map[1] - y
    c, s = np.cos(th), np.sin(th)
    return np.array([c * dx + s * dy, -s * dx + c * dy])