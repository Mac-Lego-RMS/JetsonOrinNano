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