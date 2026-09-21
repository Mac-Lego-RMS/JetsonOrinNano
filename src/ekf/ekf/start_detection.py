#!/usr/bin/env python3
"""
Start-situation detection (ROS-free).

From the walls extracted at the start (list of (alpha, d, p_start, p_end)),
determine the robot's start position so the right map can be generated.

Obstacle challenge: lane width is always 1.0 m; front distance is 1.45 m
(position 1) or 1.95 m (position 2). Direction (CW/CCW) is NOT determined here
-- it defaults to CW and is resolved at the first corner.

This is a pure function so it can be tested offline against bags. Temporal
robustness (averaging / majority vote over several scans) is the caller's job.
"""
import numpy as np

# expected geometry (obstacle challenge)
LANE_WIDTH_OBSTACLE = 1.0
FRONT_POS1 = 1.45
FRONT_POS2 = 1.95
FRONT_SPLIT = 0.5 * (FRONT_POS1 + FRONT_POS2)   # 1.70 m threshold

# tolerances
LANE_WIDTH_TOL = 0.15      # lane width must be within this of 1.0 m
FRONT_ALPHA_TOL = np.radians(25.0)
SIDE_ALPHA_TOL = np.radians(25.0)


def _wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def detect_start_obstacle(measured):
    """Determine start position from measured walls (obstacle challenge).

    Args:
        measured: list of (alpha, d, p_start, p_end), base_link frame.

    Returns a dict:
        {'position': 1|2|None,
         'front_dist': float|None,
         'lane_width': float|None,
         'valid': bool,
         'reason': str}
    'valid' is False when the geometry doesn't match a plausible start (e.g.
    lane width off, no front wall) -- the caller should then not commit.
    """
    front_d = None
    left_d = None      # wall on the +y (left) side: alpha ~ -90 deg
    right_d = None     # wall on the -y (right) side: alpha ~ +90 deg

    for (a, d, ps, pe) in measured:
        if abs(_wrap(a - np.pi)) < FRONT_ALPHA_TOL:
            # front wall (alpha ~ +-180). take the nearest if several.
            if front_d is None or abs(d) < abs(front_d):
                front_d = d
        elif abs(_wrap(a - np.radians(-90.0))) < SIDE_ALPHA_TOL:
            if left_d is None or abs(d) < abs(left_d):
                left_d = d
        elif abs(_wrap(a - np.radians(90.0))) < SIDE_ALPHA_TOL:
            if right_d is None or abs(d) < abs(right_d):
                right_d = d

    if front_d is None:
        return {'position': None, 'front_dist': None, 'lane_width': None,
                'valid': False, 'reason': 'no front wall found'}
    if left_d is None or right_d is None:
        return {'position': None, 'front_dist': abs(front_d), 'lane_width': None,
                'valid': False, 'reason': 'missing a side wall'}

    lane_width = abs(left_d) + abs(right_d)
    if abs(lane_width - LANE_WIDTH_OBSTACLE) > LANE_WIDTH_TOL:
        return {'position': None, 'front_dist': abs(front_d),
                'lane_width': lane_width, 'valid': False,
                'reason': f'lane width {lane_width:.2f} not ~1.0 m'}

    position = 1 if abs(front_d) < FRONT_SPLIT else 2
    return {'position': position, 'front_dist': abs(front_d),
            'lane_width': lane_width, 'valid': True, 'reason': 'ok'}

# --- open challenge --------------------------------------------------------
# Lane width is 60 or 100 cm (+-100 mm at the international final). Unlike the
# obstacle challenge it is not fixed but must be REPORTED, since it drives the
# start-map geometry. Front distance still separates pos 1 (1.45) / pos 2 (1.95).

LANE_WIDTH_NARROW = 0.60
LANE_WIDTH_WIDE = 1.00
LANE_WIDTH_SPLIT = 0.80        # threshold between 60 and 100 cm (20 cm margin each side)
LANE_WIDTH_MAX_DEV = 0.15      # measured width must be within this of a nominal value


def detect_start_open(measured):
    """Determine start position AND lane width for the open challenge.

    Args:
        measured: list of (alpha, d, p_start, p_end), base_link frame.

    Returns a dict:
        {'position': 1|2|None,
         'front_dist': float|None,
         'lane_width': float|None,        # measured sum of side distances
         'lane_nominal': 0.60|1.00|None,  # snapped to the nearest legal width
         'left_d': float|None,
         'right_d': float|None,
         'valid': bool,
         'reason': str}
    Note: left/right here are just the two sides in the robot frame; which is
    inner vs outer is NOT resolved (direction is unknown until the first corner).
    """
    front_d = None
    left_d = None      # +y side (alpha ~ -90)
    right_d = None     # -y side (alpha ~ +90)

    for (a, d, ps, pe) in measured:
        if abs(_wrap(a - np.pi)) < FRONT_ALPHA_TOL:
            if front_d is None or abs(d) < abs(front_d):
                front_d = d
        elif abs(_wrap(a - np.radians(-90.0))) < SIDE_ALPHA_TOL:
            if left_d is None or abs(d) < abs(left_d):
                left_d = d
        elif abs(_wrap(a - np.radians(90.0))) < SIDE_ALPHA_TOL:
            if right_d is None or abs(d) < abs(right_d):
                right_d = d

    if front_d is None:
        return {'position': None, 'front_dist': None, 'lane_width': None,
                'lane_nominal': None, 'left_d': None, 'right_d': None,
                'valid': False, 'reason': 'no front wall'}
    if left_d is None or right_d is None:
        return {'position': None, 'front_dist': abs(front_d), 'lane_width': None,
                'lane_nominal': None, 'left_d': left_d, 'right_d': right_d,
                'valid': False, 'reason': 'missing a side wall'}

    lane_width = abs(left_d) + abs(right_d)
    lane_nominal = LANE_WIDTH_NARROW if lane_width < LANE_WIDTH_SPLIT else LANE_WIDTH_WIDE
    if abs(lane_width - lane_nominal) > LANE_WIDTH_MAX_DEV:
        return {'position': None, 'front_dist': abs(front_d),
                'lane_width': lane_width, 'lane_nominal': None,
                'left_d': left_d, 'right_d': right_d, 'valid': False,
                'reason': f'lane width {lane_width:.2f} not near 0.6 or 1.0 m'}

    position = 1 if abs(front_d) < FRONT_SPLIT else 2
    return {'position': position, 'front_dist': abs(front_d),
            'lane_width': lane_width, 'lane_nominal': lane_nominal,
            'left_d': abs(left_d), 'right_d': abs(right_d),
            'valid': True, 'reason': 'ok'}