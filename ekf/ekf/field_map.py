#!/usr/bin/env python3
"""
Field-fixed map of the WRO Future Engineers game field.

Frame: origin at the field centre (centre of the inner square), axes X_field
(east) and Y_field (north). This is the neutral geometric description -- no
driving direction, no start pose. Everything variable (start position, CW/CCW)
lives in the transformation into the start-anchored map frame (separate step).

Field geometry (obstacle challenge, fixed):
  - Outer wall: 3.0 x 3.0 m square, corners at (+-1.5, +-1.5)
  - Inner wall: 1.0 x 1.0 m square, corners at (+-0.5, +-0.5), centred

Walls are stored as SEGMENTS: pairs of corner points (p1, p2). Corners are
listed counter-clockwise so the inward normal (into the drivable lane) is
consistent: for a CCW-ordered polygon, the left-hand normal of each edge
(p2 - p1 rotated +90 deg) points into the polygon interior.
  - Outer square: interior = the lane -> left normal points inward (toward
    field centre) = toward the lane. Correct.
  - Inner square: we want the normal to point OUTWARD (into the lane, away
    from the centre). So the inner square is listed CLOCKWISE, making its
    left-hand normal point outward.
"""
import numpy as np

from ekf.ekf import wrap

# --- field dimensions (metres) ---
OUTER_HALF = 1.5     # outer wall: 3x3 m -> half-size 1.5
INNER_HALF = 0.5     # inner wall: 1x1 m -> half-size 0.5


def _square_segments(half, clockwise=False):
    """Return the 4 edges of an axis-aligned square of the given half-size,
    as (p1, p2) corner-point pairs. CCW by default; set clockwise=True to
    reverse the winding (flips the inward/outward normal sense).
    """
    # corners CCW starting bottom-left
    c = [
        np.array([-half, -half]),
        np.array([+half, -half]),
        np.array([+half, +half]),
        np.array([-half, +half]),
    ]
    if clockwise:
        c = c[::-1]
    segments = []
    for i in range(4):
        segments.append((c[i], c[(i + 1) % 4]))
    return segments


# Outer wall CCW: left-hand normal points inward (toward centre / into lane).
OUTER_SEGMENTS = _square_segments(OUTER_HALF, clockwise=False)

# Inner wall CW: left-hand normal points outward (away from centre / into lane).
INNER_SEGMENTS = _square_segments(INNER_HALF, clockwise=True)

# Full field: 8 wall segments.
FIELD_SEGMENTS = OUTER_SEGMENTS + INNER_SEGMENTS


def segment_to_hnf(p1, p2):
    """Convert a segment (p1, p2) to Hesse normal form (alpha, d) in the SAME
    frame the points are in. Normal is the left-hand normal of (p2 - p1):
    direction rotated +90 deg. Returns (alpha, d) with alpha in [-pi, pi].

    d is the signed distance of the line from the frame origin along the normal.
    """
    d_vec = p2 - p1
    length = np.hypot(d_vec[0], d_vec[1])
    # left-hand normal (rotate direction +90 deg): (dx, dy) -> (-dy, dx)
    normal = np.array([-d_vec[1], d_vec[0]]) / length
    alpha = np.arctan2(normal[1], normal[0])
    d = np.dot(p1, normal)          # signed distance of the line from origin
    return alpha, d


# Robot drives along the north lane in +X (east). Inner wall is to the right
# (CW). Positions differ only in distance from the front wall.
#   Pos 1: front wall 1.45 m ahead -> x_start = 1.5 - 1.45 = 0.05
#   Pos 2: front wall 1.95 m ahead -> x_start = 1.5 - 1.95 = -0.45
# y_start = 1.0 (centred in the 1 m north lane), theta = 0 (facing +X).
START_POSES_CW = {
    'pos1': (0.05, 1.0, 0.0),
    'pos2': (-0.45, 1.0, 0.0),
}

# CCW: robot drives the north lane facing -X (west), inner wall to the left.
#   Pos 1: front wall 1.45 m ahead -> x_start = -1.5 + 1.45 = -0.05
#   Pos 2: front wall 1.95 m ahead -> x_start = -1.5 + 1.95 = 0.45
START_POSES_CCW = {
    'pos1': (-0.05, 1.0, np.pi),
    'pos2': (0.45, 1.0, np.pi),
}


def _transform_point(p, start_pose):
    """Map a field-frame point into the start-anchored map frame."""
    xs, ys, th = start_pose
    dx, dy = p[0] - xs, p[1] - ys
    c, s = np.cos(th), np.sin(th)
    # inverse rotation (map frame is field rotated by +theta about start)
    x_map = c * dx + s * dy
    y_map = -s * dx + c * dy
    return np.array([x_map, y_map])


def _segments_to_map(segments, start_pose):
    """Transform a list of (p1, p2) field-frame segments into map-frame wall
    dicts {'alpha', 'd', 'p1', 'p2'}. Shared by generate_map, outer_walls_map
    and inner_walls_map so the transformation lives in exactly one place."""
    xs, ys, th = start_pose
    walls = []
    for (p1, p2) in segments:
        alpha_f, d_f = segment_to_hnf(p1, p2)
        walls.append({
            'alpha': wrap(alpha_f - th),
            'd': d_f - (xs * np.cos(alpha_f) + ys * np.sin(alpha_f)),
            'p1': _transform_point(np.asarray(p1), start_pose),
            'p2': _transform_point(np.asarray(p2), start_pose),
        })
    return walls


def generate_map(start_pose, segments=FIELD_SEGMENTS):
    """Transform field-fixed wall segments into the map frame for a start pose.

    Returns a list of dicts, one per wall:
        {'alpha', 'd', 'p1', 'p2'}
    where alpha, d are the map-frame HNF (for matching) and p1, p2 are the
    transformed endpoints (for visibility gating).
    """
    return _segments_to_map(segments, start_pose)


def start_map_3wall(front_dist, left_d, right_d):
    """Reduced start-straight map: the three visible walls as (alpha, d).

    For the open challenge the full field geometry is unknown at the start
    (inner band is variable), so the map is just the three walls the robot
    actually sees, in the map frame (robot start pose = origin, facing +x):

        front wall : alpha = 180 deg, d = -front_dist
        left  wall : alpha = -90 deg, d = -left_d
        right wall : alpha = +90 deg, d = -right_d

    Returned as plain (alpha, d) tuples (no endpoints): match_walls handles
    these via its legacy path (innovation gating only, no overlap check).
    Left/right are just the two sides in the robot frame; inner vs outer is
    not resolved here (direction is unknown until the first corner).

    Args:
        front_dist, left_d, right_d: positive distances (m) from detect_start.

    Returns list of (alpha, d) tuples.
    """
    return [
        (np.pi, -abs(front_dist)),           # front
        (np.radians(-90.0), -abs(left_d)),   # left  (+y side)
        (np.radians(90.0), -abs(right_d)),   # right (-y side)
    ]


def outer_box_map(start_pose):
    """Outer track box (fixed 3x3 rim) in the start-anchored map frame.

    Returns (corners, walls, edge_length):
      corners: list of 4 (x, y) np arrays, CCW winding, index 0 = largest x
               (furthest ahead in driving direction).
      walls:   list of 4 (nx, ny, d) tuples, HNF in the map frame, ordered to
               match the edge starting at corners[i] -> corners[i+1].
      edge_length: 3.0 (nominal).

    Direction-agnostic: the winding/index rule is purely geometric; the caller
    uses the separate latched direction to traverse the indices.
    """
    # transform the 4 outer corners (field frame) into the map frame
    corners_field = [p1 for (p1, p2) in OUTER_SEGMENTS]  # the 4 corners
    corners_map = [_transform_point(np.array(c), start_pose) for c in corners_field]

    # sort CCW about the box centre, then rotate so index 0 = largest x
    centre = np.mean(corners_map, axis=0)
    order = sorted(range(4),
                   key=lambda i: np.arctan2(corners_map[i][1] - centre[1],
                                            corners_map[i][0] - centre[0]))
    ccw = [corners_map[i] for i in order]
    start = int(np.argmax([c[0] for c in ccw]))          # largest x
    ccw = ccw[start:] + ccw[:start]

    # build the wall HNF for each edge corners[i] -> corners[i+1], normal
    # pointing inward (toward the box centre)
    walls = []
    for i in range(4):
        p1 = ccw[i]
        p2 = ccw[(i + 1) % 4]
        edge = p2 - p1
        n = np.array([-edge[1], edge[0]])
        n = n / np.hypot(n[0], n[1])
        # orient inward: normal should point toward the centre
        if np.dot(n, centre - p1) < 0:
            n = -n
        d = np.dot(n, p1)
        walls.append((float(n[0]), float(n[1]), float(d)))

    return ccw, walls, 3.0


def outer_walls_map(start_pose):
    """The 4 outer-band walls (3x3 rim) as match-ready walls in the map frame.

    Same dict format as generate_map ({'alpha','d','p1','p2'} with endpoints),
    so match_walls applies overlap gating. Inner band is omitted (unknown in the
    open challenge until round-1 learning). Used as the EKF matching map after
    the direction latch on the open challenge.
    """
    return _segments_to_map(OUTER_SEGMENTS, start_pose)


def inner_walls_map(start_pose):
    """The 4 inner-band walls (fixed 1x1 square) in the map frame.

    OBSTACLE CHALLENGE ONLY. There the lane is always 1000 mm (+-10 mm), so the
    inner band is known exactly and must NOT be measured -- measuring would
    replace a known constant with a noisy estimate.

    Returns (walls, corners) in the same shape as inner_band_from_widths, so
    both challenges can feed the same publisher:
        walls:   list of 4 dicts {'alpha','d','p1','p2'}, normals pointing
                 OUTWARD (into the lane)
        corners: list of 4 (2,) arrays, with wall i spanning
                 corners[i] -> corners[i+1]
    """
    walls = _segments_to_map(INNER_SEGMENTS, start_pose)
    corners = [w['p1'] for w in walls]
    return walls, corners


def inner_band_from_widths(start_pose, widths):
    """Inner-band walls and corners in the map frame, from measured lane widths.

    OPEN CHALLENGE: the inner band is variable, so it is reconstructed from the
    widths learned while driving. The inner edge of straight i is the outer wall
    i shifted inward by that straight's measured lane width; intersecting
    adjacent inner lines gives the inner corners. Uses the MEASURED width (not
    snapped to 0.6/1.0) because the rules allow +-100 mm at the international
    final.

    Args:
        start_pose: field-frame start pose (same one used for outer_box_map).
        widths: dict {outer_wall_index (0..3): measured lane width in m}.
                All four indices must be present.

    Returns (walls, corners) or None if the geometry is degenerate. Same shape
    as inner_walls_map.
    """
    if any(i not in widths for i in range(4)):
        return None

    _, outer_walls, _ = outer_box_map(start_pose)

    # inner LINES: n . p = d_out + w, with n the outer wall's inward normal
    lines = []
    for i in range(4):
        nx, ny, d_out = outer_walls[i]
        lines.append((np.array([nx, ny]), d_out + widths[i]))

    # inner corner j = intersection of line[j-1] and line[j]
    corners = []
    for j in range(4):
        n1, c1 = lines[(j - 1) % 4]
        n2, c2 = lines[j]
        A = np.array([n1, n2])
        if abs(np.linalg.det(A)) < 1e-9:
            return None                      # parallel -> degenerate
        corners.append(np.linalg.solve(A, np.array([c1, c2])))

    # wall i spans corners[i] -> corners[i+1]; normal flipped to point outward
    walls = []
    for i in range(4):
        n_out = -lines[i][0]                 # outward = into the lane
        d_out = -lines[i][1]
        walls.append({
            'alpha': float(np.arctan2(n_out[1], n_out[0])),
            'd': float(d_out),
            'p1': corners[i],
            'p2': corners[(i + 1) % 4],
        })
    return walls, corners


# --- obstacle (traffic sign) seats -----------------------------------------
# Six seats per straight: two columns and three rows.
#   columns: 0.4 m inside the outer wall, 0.4 m outside the inner wall
#            (for a 1 m lane that is 0.1 m either side of the lane centre,
#             so the two columns are only 0.2 m apart)
#   rows:    -0.5 / 0 / +0.5 along the straight -- the ends and the middle of
#            the inner band, i.e. the T- and X-intersections in the rules
#
# Measured on the WEST straight (GeoGebra):
#   outer column x = -1.1, inner column x = -0.9, rows y = -0.5 / 0 / +0.5
# The other three straights follow by 90 deg rotation (4-fold symmetric field).
#
# NOTE: on the START straight the rules move all signs to the positions closer
# to the inner wall, so only the INNER column is legal there.

SEAT_OUTER_INSET = 0.4         # from the outer wall, into the lane
SEAT_INNER_INSET = 0.4         # from the inner wall, into the lane
SEAT_ROWS = (-0.5, 0.0, 0.5)   # along the straight


def _rot90(p, k):
    """Rotate a point k times by 90 deg CCW about the origin."""
    x, y = p
    for _ in range(k % 4):
        x, y = -y, x
    return np.array([x, y])


def _west_straight_seats():
    """The six seats of the west straight in the field frame, as
    (point, column) with column 'outer' or 'inner'."""
    x_outer = -OUTER_HALF + SEAT_OUTER_INSET      # -1.1
    x_inner = -INNER_HALF - SEAT_INNER_INSET      # -0.9
    seats = []
    for y in SEAT_ROWS:
        seats.append((np.array([x_outer, y]), 'outer'))
        seats.append((np.array([x_inner, y]), 'inner'))
    return seats


def obstacle_seats_field():
    """All 24 seats in the FIELD frame, grouped by straight.

    Returns a list of 4 lists (one per straight), each holding dicts:
        {'p': (2,) field-frame point, 'column': 'outer'|'inner', 'row': index}

    Straight k is the west straight rotated k times by 90 deg CCW. Which
    straight belongs to which outer-wall index in the map depends on the start
    pose, so the caller pairs them up via seat_group_to_wall_index().
    """
    base = _west_straight_seats()
    out = []
    for k in range(4):
        straight = []
        for i, (p, col) in enumerate(base):
            straight.append({'p': _rot90(p, k), 'column': col,
                             'row': i // 2})
        out.append(straight)
    return out


def obstacle_seats_map(start_pose):
    """All 24 seats transformed into the start-anchored MAP frame.

    Same grouping as obstacle_seats_field(); 'p' is now a map-frame point.
    """
    out = []
    for straight in obstacle_seats_field():
        out.append([{'p': _transform_point(s['p'], start_pose),
                     'column': s['column'], 'row': s['row']}
                    for s in straight])
    return out


def seat_group_to_wall_index(start_pose):
    """Map each seat group (0..3 from obstacle_seats_*) to the outer-wall index
    used by outer_box_map / corner_geometry.

    The two indexings are independent: seat groups are rotations of the west
    straight in the FIELD frame, wall indices are a CCW ordering in the MAP
    frame anchored at max x. They are paired geometrically: a straight's seats
    sit 0.4 m from their own outer wall and much further from the other three,
    so the nearest wall line is unambiguous.

    Returns a list of 4 wall indices, one per seat group.
    """
    _, walls, _ = outer_box_map(start_pose)
    out = []
    for group in obstacle_seats_map(start_pose):
        c = np.mean([s['p'] for s in group], axis=0)
        best, best_d = -1, np.inf
        for i, (nx, ny, d) in enumerate(walls):
            dist = abs(nx * c[0] + ny * c[1] - d)
            if dist < best_d:
                best_d, best = dist, i
        out.append(best)
    return out


if __name__ == '__main__':
    pose = START_POSES_CW['pos2']
    print('Pos 2 CW -- outer walls in map frame:')
    for w in outer_walls_map(pose):
        print(f"  alpha={np.degrees(w['alpha']):+7.1f} deg  d={w['d']:+.3f}  "
              f"p1=({w['p1'][0]:+.2f},{w['p1'][1]:+.2f})  "
              f"p2=({w['p2'][0]:+.2f},{w['p2'][1]:+.2f})")
    print('Pos 2 CW -- inner walls in map frame (fixed 1x1):')
    walls, corners = inner_walls_map(pose)
    for i, w in enumerate(walls):
        edge = float(np.hypot(*(w['p2'] - w['p1'])))
        print(f"  alpha={np.degrees(w['alpha']):+7.1f} deg  d={w['d']:+.3f}  "
              f"p1=({w['p1'][0]:+.2f},{w['p1'][1]:+.2f})  "
              f"p2=({w['p2'][0]:+.2f},{w['p2'][1]:+.2f})  len={edge:.3f}")