#!/usr/bin/env python3
"""
Driving-direction detection (CW / CCW) at the first corner (ROS-free).

Principle (robust, topological): the front wall and a side wall define an
intersection = the corner point. On the OUTER side the band is continuous
(front and side wall meet), so no front-wall points reach beyond the
intersection. On the INNER side the lane opens up, so front-wall points reach
beyond it. "Open" is positively provable; "closed" is only inferred, so a
single closed-looking side is NOT conclusive on its own.

Inner side right  -> CW.
Inner side left   -> CCW.

Works with a SKEWED front wall (general HNF line intersection). Cross-checks
both sides when available; falls back to one side + known lane width otherwise.

All inputs in the robot/base_link frame. Uses your verified convention:
alpha ~ -90 deg = left (+y), alpha ~ +90 deg = right (-y), alpha ~ +-180 = front.
"""
import numpy as np

FRONT_ALPHA_TOL = np.radians(30.0)
SIDE_ALPHA_TOL = np.radians(30.0)
BEYOND_DIST = 0.10        # front points must reach this far (m) past the corner
BEYOND_MIN_PTS = 15       # ...and at least this many, to count as "open"


def _wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def _line_intersection(alpha1, d1, alpha2, d2):
    """Intersection of two HNF lines x*cos(a)+y*sin(a)=d. None if ~parallel."""
    A = np.array([[np.cos(alpha1), np.sin(alpha1)],
                  [np.cos(alpha2), np.sin(alpha2)]])
    b = np.array([d1, d2])
    det = np.linalg.det(A)
    if abs(det) < 1e-6:
        return None
    return np.linalg.solve(A, b)


def _classify_side(front_wall, side_wall, side_sign):
    """Is this side OPEN (inner) or CLOSED (outer)?

    front_wall, side_wall: (alpha, d, p_start, p_end) in base_link frame.
    side_sign: +1 for the left side (+y), -1 for the right side (-y).

    Returns 'open', 'closed', or None (couldn't decide).
    """
    a_f, d_f, fp1, fp2 = front_wall[:4]
    a_s, d_s = side_wall[0], side_wall[1]

    corner = _line_intersection(a_f, d_f, a_s, d_s)
    if corner is None:
        return None

    # direction ALONG the front wall (perpendicular to its normal)
    u = np.array([-np.sin(a_f), np.cos(a_f)])
    # orient u so it points toward the side under test (away from lane centre):
    # the side is at +y (left) or -y (right); pick u's sign by its y component
    if np.sign(u[1]) != np.sign(side_sign):
        u = -u

    # project front-wall endpoints and corner onto u
    t_corner = np.dot(corner, u)
    # reconstruct approximate front-wall point extent from its endpoints
    t_p1 = np.dot(fp1, u)
    t_p2 = np.dot(fp2, u)
    t_far = max(t_p1, t_p2)          # how far the front wall reaches toward this side

    # "open" if the front wall reaches clearly beyond the corner on this side
    if t_far > t_corner + BEYOND_DIST:
        return 'open'
    return 'closed'


def detect_direction(measured, lane_width=None):
    """Detect CW / CCW from walls at the corner.

    Args:
        measured: list of (alpha, d, p_start, p_end), base_link frame.
        lane_width: known lane width (m) for one-sided fallback; may be None.

    Returns dict:
        {'direction': 'CW'|'CCW'|None, 'confident': bool, 'reason': str}
    """
    front = None
    left = None       # alpha ~ -90 (+y side)
    right = None      # alpha ~ +90 (-y side)
    for w in measured:
        a = w[0]
        if abs(_wrap(a - np.pi)) < FRONT_ALPHA_TOL:
            if front is None or abs(w[1]) < abs(front[1]):
                front = w
        elif abs(_wrap(a + np.radians(90.0))) < SIDE_ALPHA_TOL:
            if left is None or abs(w[1]) < abs(left[1]):
                left = w
        elif abs(_wrap(a - np.radians(90.0))) < SIDE_ALPHA_TOL:
            if right is None or abs(w[1]) < abs(right[1]):
                right = w

    if front is None:
        return {'direction': None, 'confident': False,
                'reason': 'no front wall'}

    left_state = _classify_side(front, left, +1) if left is not None else None
    right_state = _classify_side(front, right, -1) if right is not None else None

    # --- both sides available: cross-check ---
    if left_state is not None and right_state is not None:
        if right_state == 'open' and left_state == 'closed':
            return {'direction': 'CW', 'confident': True,
                    'reason': 'right open, left closed'}
        if left_state == 'open' and right_state == 'closed':
            return {'direction': 'CCW', 'confident': True,
                    'reason': 'left open, right closed'}
        # both same -> contradictory, don't guess
        return {'direction': None, 'confident': False,
                'reason': f'ambiguous (left {left_state}, right {right_state})'}

    # --- one side only: 'open' is conclusive; 'closed' is not ---
    single = left_state if left_state is not None else right_state
    single_sign = +1 if left_state is not None else -1
    if single == 'open':
        direction = 'CCW' if single_sign > 0 else 'CW'
        return {'direction': direction, 'confident': True,
                'reason': 'one side open (conclusive)'}

    # single side looks closed -> could just be the unseen side that's inner
    return {'direction': None, 'confident': False,
            'reason': 'only one side, looks closed (inconclusive)'}