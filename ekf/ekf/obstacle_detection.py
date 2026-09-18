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

# --- parking lot -----------------------------------------------------------
# The two magenta boundary walls are a different shape from the pillars: 20 cm
# long, 2 cm thick, standing perpendicular to the outer band. Long and thin
# where a pillar is compact, so MAX_OBSTACLE_EXTENT would throw them away.
# They get their own path: cluster, fit a line, keep the ENDPOINTS -- it is the
# ends that define the bay.
#
# Note on timing: while the robot is parked the walls sit about 4 cm ahead and
# behind it, far below the fusion range floor, so they are invisible. The bay
# can only be measured after pulling out.

PARK_CLUSTER_RADIUS = 0.05     # region growing for the magenta points (m)
PARK_MIN_POINTS = 8
PARK_LEN_MIN = 0.10            # a 20 cm wall seen at an angle still spans this
PARK_LEN_MAX = 0.30
PARK_PARALLEL_TOL = np.radians(25.0)   # the two walls face the same way


def _fit_segment(arr):
    """Least-squares line through a cluster, returned as its two endpoints.

    Endpoints come from projecting every point onto the fitted direction and
    taking the extremes -- unlike the wall fit there is no scan ordering to
    rely on here, the points come from a cloud.
    """
    centroid = arr.mean(axis=0)
    _, _, Vh = np.linalg.svd(arr - centroid, full_matrices=False)
    direction = Vh[0]
    t = (arr - centroid) @ direction
    return centroid + t.min() * direction, centroid + t.max() * direction


def detect_parking_walls(msg, min_points=PARK_MIN_POINTS,
                         radius=PARK_CLUSTER_RADIUS):
    """Find the magenta parking-lot walls in one colour-classified scan.

    Returns a list of dicts:
        {'p1', 'p2', 'length', 'n', 'dist'}
    with p1/p2 the segment endpoints in the base_link frame.
    """
    pts = colored_points(msg, colors=(MAGENTA,))
    if not pts:
        return []
    group = np.array([[x, y] for (x, y, _) in pts])

    walls = []
    for idx in _grow_clusters(group, radius):
        if len(idx) < min_points:
            continue
        arr = group[idx]
        p1, p2 = _fit_segment(arr)
        length = float(np.hypot(*(p2 - p1)))
        if not (PARK_LEN_MIN <= length <= PARK_LEN_MAX):
            continue
        centre = 0.5 * (p1 + p2)
        walls.append({'p1': p1, 'p2': p2, 'length': length, 'n': len(idx),
                      'dist': float(np.hypot(*centre))})
    return walls


def parking_pair(walls, expected_gap, tol=0.06):
    """Pick the two walls that actually form the bay, or None.

    Checks what the rules fix: two roughly parallel segments, separated by
    1.5 x robot length. Anything else -- a single wall, a magenta smear, three
    candidates -- fails rather than producing a plausible-looking bay.
    """
    if len(walls) < 2:
        return None

    best, best_err = None, np.inf
    for i in range(len(walls)):
        for j in range(i + 1, len(walls)):
            a, b = walls[i], walls[j]
            da = a['p2'] - a['p1']
            db = b['p2'] - b['p1']
            # segments have no head/tail, so fold the angle into [0, pi/2]
            dang = abs(wrap(np.arctan2(da[1], da[0]) - np.arctan2(db[1], db[0])))
            if dang > np.pi / 2:
                dang = np.pi - dang
            if dang > PARK_PARALLEL_TOL:
                continue

            # gap: perpendicular distance from B's centre to A's line
            u = da / np.hypot(*da)
            n = np.array([-u[1], u[0]])
            gap = abs(float(np.dot(0.5 * (b['p1'] + b['p2']) - a['p1'], n)))
            err = abs(gap - expected_gap)
            if err <= tol and err < best_err:
                best_err, best = err, (a, b, gap)
    return best