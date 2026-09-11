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
"""
import struct
import numpy as np

from ekf.wall_extraction import LIDAR_OFFSET_X

RED = (255, 0, 0)
GREEN = (0, 255, 0)
MAGENTA = (255, 0, 255)

PILLAR_HALF_WIDTH = 0.022      # 44 mm pillars: LiDAR sees the front face
CLUSTER_RADIUS = 0.04          # region growing: neighbour distance (m)
MIN_OBSTACLE_POINTS = 5        # fewer points -> stray misclassification
MAX_OBSTACLE_EXTENT = 0.15     # Farblauf einer Pylone: auf 1.93 m am Rohbild 12.4 cm gemessen


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


def detect_obstacles(msg, min_points=MIN_OBSTACLE_POINTS,
                     radius=CLUSTER_RADIUS, max_extent=MAX_OBSTACLE_EXTENT):
    """Detect pillars in one colour-classified scan.

    Returns a list of dicts:
        {'x', 'y', 'color': 'red'|'green', 'n', 'dist', 'extent'}
    with (x, y) the estimated pillar CENTRE in the base_link frame -- the
    measured front face pushed back by half the pillar width along the line of
    sight.
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
            # compactness: largest span in either axis. A pillar is small;
            # noise smeared along a wall is not.
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
                              'extent': extent})

    return obstacles