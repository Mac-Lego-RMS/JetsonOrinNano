#!/usr/bin/env python3
"""
Obstacle (traffic sign) detection from the colour-classified point cloud.

Input is /camera_lidar/colored_scan, where the camera-LiDAR node has already
CLASSIFIED each point: pillars come out as pure (255,0,0) red or (0,255,0)
green, parking walls as (255,0,255) magenta, everything else keeps its original
colour. Detection here is an exact byte match -- no colour thresholds.

Frames: the cloud is in the RAW LiDAR frame. Verified with a pillar placed 1 m
straight ahead: it appears at bearing +-180 deg, x = -0.96. So the same mirror
correction as scan_to_points applies (x_robot = -x_cloud, y_robot = -y_cloud),
plus the rear-axle offset.

Clustering is SPATIAL (region growing), not bearing-sorted: bearing sorting
breaks at the +-180 deg wraparound and merges objects that share a bearing but
sit at different distances.

Noise rejection is by COMPACTNESS, not point count: a 44 mm pillar fits in a
~5 cm circle, while camera noise misclassified on a wall smears along it.

Each detection also carries the angular SECTOR it occupies, so the wall
extraction can mask those directions out (see mask_sectors). Pillars standing
near a corner otherwise corrupt the front-wall fit and delay the direction
latch by seconds.
"""
import struct
import numpy as np

from ekf.ekf import wrap
from ekf.wall_extraction import LIDAR_OFFSET_X

RED = (255, 0, 0)
GREEN = (0, 255, 0)
MAGENTA = (255, 0, 255)

PILLAR_HALF_WIDTH = 0.022      # 44 mm pillars: LiDAR sees the front face
CLUSTER_RADIUS = 0.04          # region growing: neighbour distance (m)
MIN_OBSTACLE_POINTS = 5        # fewer points -> stray misclassification
MAX_OBSTACLE_EXTENT = 0.09     # a pillar spans <= this; longer = noise on a wall
SECTOR_MARGIN = np.radians(3.0)  # widen each masked sector by this on each side


def colored_points(msg, colors=(RED, GREEN)):
    """Classified points as (x, y, colour) in the BASE_LINK frame."""
    wanted = {(r << 16) | (g << 8) | b for (r, g, b) in colors}
    out = []
    for i in range(msg.width):
        off = i * msg.point_step
        rgb = struct.unpack_from('<I', msg.data, off + 12)[0] & 0xFFFFFF
        if rgb not in wanted:
            continue
        x, y, _ = struct.unpack_from('<fff', msg.data, off)
        if not (np.isfinite(x) and np.isfinite(y)):
            continue
        # mirror correction (LiDAR mounted 180 deg) + rear-axle offset
        out.append((-x + LIDAR_OFFSET_X, -y,
                    ((rgb >> 16) & 255, (rgb >> 8) & 255, rgb & 255)))
    return out


def _grow_clusters(points, radius):
    """Spatial region growing. points: (N,2) array. Returns list of index lists.
    Order-independent, so no +-180 deg wraparound problem."""
    n = len(points)
    unassigned = set(range(n))
    clusters = []
    while unassigned:
        seed = unassigned.pop()
        cluster = [seed]
        frontier = [seed]
        while frontier:
            i = frontier.pop()
            close = [j for j in unassigned
                     if np.hypot(*(points[j] - points[i])) <= radius]
            for j in close:
                unassigned.discard(j)
                cluster.append(j)
                frontier.append(j)
        clusters.append(cluster)
    return clusters


def _sector(arr, margin=SECTOR_MARGIN):
    """Angular sector a cluster occupies, as (centre, half_width), in the frame
    scan_to_points produces -- mirror-corrected but WITHOUT the rear-axle
    offset, so it lines up with the raw scan points.

    Offsets are taken relative to the cluster's mean direction, so a cluster
    straddling +-180 deg is handled without a special case.
    """
    sx = arr[:, 0] - LIDAR_OFFSET_X          # base_link -> scan frame
    sy = arr[:, 1]
    ang = np.arctan2(sy, sx)
    centre = float(np.arctan2(np.sin(ang).mean(), np.cos(ang).mean()))
    delta = np.array([wrap(a - centre) for a in ang])
    half = float(max(abs(delta.min()), abs(delta.max()))) + margin
    return (centre, half)


def detect_obstacles(msg, min_points=MIN_OBSTACLE_POINTS,
                     radius=CLUSTER_RADIUS, max_extent=MAX_OBSTACLE_EXTENT):
    """Detect pillars in one colour-classified scan.

    Returns a list of dicts:
        {'x', 'y', 'color': 'red'|'green', 'n', 'dist', 'extent', 'sector'}
    with (x, y) the estimated pillar CENTRE in the base_link frame -- the
    measured front face pushed back by half the pillar width along the line of
    sight -- and 'sector' = (centre, half_width) in the scan frame, for
    mask_sectors.
    """
    pts = colored_points(msg)
    obstacles = []

    for color_rgb, name in ((RED, 'red'), (GREEN, 'green')):
        group = np.array([[x, y] for (x, y, c) in pts if c == color_rgb])
        if len(group) == 0:
            continue

        for idx in _grow_clusters(group, radius):
            if len(idx) < min_points:
                continue                        # stray misclassified points
            arr = group[idx]
            # compactness: a pillar is small; noise smeared along a wall is not
            extent = float(max(arr[:, 0].ptp(), arr[:, 1].ptp()))
            if extent > max_extent:
                continue

            cx, cy = arr[:, 0].mean(), arr[:, 1].mean()
            d = np.hypot(cx, cy)
            if d < 1e-6:
                continue
            # front face -> centre: push away from the sensor by half a width
            cx += PILLAR_HALF_WIDTH * cx / d
            cy += PILLAR_HALF_WIDTH * cy / d
            obstacles.append({'x': float(cx), 'y': float(cy), 'color': name,
                              'n': len(idx), 'dist': float(np.hypot(cx, cy)),
                              'extent': extent, 'sector': _sector(arr)})

    return obstacles


def sector_from_robot_point(x, y, margin=SECTOR_MARGIN):
    """Angular sector a pillar at base_link (x, y) occupies, in the scan frame.

    For pillars whose position is ALREADY KNOWN (from the obstacle map), so the
    sector can be recomputed at wall-extraction rate from the current pose.
    That matters on a close pass: /obstacles_live arrives at ~6 Hz, and at
    0.1-0.2 m the bearing sweeps ~25 deg between messages while the pillar is
    only ~20 deg wide -- the previous message's sector no longer overlaps the
    current position, however generously it is widened. A pillar is also
    invisible to the fusion below its range floor (0.15 m), exactly in the
    window where it wrecks the wall fit.

    The half-width is geometric rather than measured, so it grows correctly as
    the pillar gets closer: ~1.3 deg at 1 m, ~11 deg at 0.11 m.
    """
    sx = x - LIDAR_OFFSET_X          # base_link -> scan frame
    sy = y
    d = float(np.hypot(sx, sy))
    if d < 1e-3:
        return None
    centre = float(np.arctan2(sy, sx))
    half = float(np.arctan2(PILLAR_HALF_WIDTH, d)) + margin
    return (centre, half)


def mask_sectors(pts, sectors):
    """Drop scan points whose bearing falls inside an obstacle sector.

    Applied to the output of scan_to_points, before clustering, so pillars do
    not end up inside wall clusters. Losing a slice of a wall is harmless --
    the gap clustering handles it, and both fragments still match the same map
    wall -- while a pillar merged into a wall corrupts its fit.

    pts: (N, 2) array. sectors: list of (centre, half_width) from detections.
    """
    if not sectors or len(pts) == 0:
        return pts
    ang = np.arctan2(pts[:, 1], pts[:, 0])
    keep = np.ones(len(pts), dtype=bool)
    for centre, half in sectors:
        keep &= np.abs((ang - centre + np.pi) % (2 * np.pi) - np.pi) > half
    return pts[keep]