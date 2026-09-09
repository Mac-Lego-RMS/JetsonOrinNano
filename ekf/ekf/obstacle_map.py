#!/usr/bin/env python3
"""
Obstacle map accumulation (ROS-free).

Detections from obstacle_detection are transformed into the map frame and
SNAPPED to the nearest legal seat. Seats are 0.2 m apart across the lane and
0.5 m along it, so snapping turns a noisy measurement into a discrete decision
and a plausibility check in one step: anything that does not land near a seat
is not an obstacle.

Votes are accumulated per (seat, colour) over the whole drive rather than
decided per scan. This solves the occlusion case: when two pillars share a row
(20 cm apart in depth) the near one hides the far one head-on, and only a later
viewpoint reveals both.
"""
import numpy as np
from collections import defaultdict

SNAP_MAX_DIST = 0.12       # a detection further than this from any seat is
                           # not an obstacle (seats are 0.2 m apart)
MIN_SEAT_VOTES = 3         # votes before a seat counts as occupied


def robot_to_map(x, y, pose):
    """Transform a point from the robot/base_link frame into the map frame."""
    px, py, th = pose
    c, s = np.cos(th), np.sin(th)
    return np.array([px + c * x - s * y, py + s * x + c * y])


class ObstacleMap:
    """Accumulates obstacle detections onto the fixed seat grid."""

    def __init__(self, seats_by_straight, snap_max=SNAP_MAX_DIST):
        """seats_by_straight: output of obstacle_seats_map(start_pose)."""
        self.seats = []            # flat list of (straight, index_in_straight, point)
        for si, straight in enumerate(seats_by_straight):
            for k, s in enumerate(straight):
                self.seats.append((si, k, np.asarray(s['p'], dtype=float),
                                   s['column'], s['row']))
        self.snap_max = snap_max
        self.votes = defaultdict(lambda: defaultdict(int))   # seat_id -> colour -> n
        self.rejected = 0          # detections that matched no seat

    def add_detections(self, detections, pose, allowed=None):
        """Snap detections (robot frame) to seats using the current pose.

        allowed: optional predicate(straight, column) -> bool, to encode the
        start-straight rule that only the inner column is legal there.
        """
        for det in detections:
            p = robot_to_map(det['x'], det['y'], pose)
            best, best_d = None, np.inf
            for sid, (si, k, sp, col, row) in enumerate(self.seats):
                if allowed is not None and not allowed(si, col):
                    continue
                d = float(np.hypot(*(p - sp)))
                if d < best_d:
                    best_d, best = d, sid
            if best is None or best_d > self.snap_max:
                self.rejected += 1
                continue
            self.votes[best][det['color']] += 1

    def occupied_seats(self, min_votes=MIN_SEAT_VOTES):
        """Seats with enough votes, as a list of dicts:
            {'straight', 'column', 'row', 'p', 'color', 'votes'}
        Colour is the majority of that seat's votes."""
        out = []
        for sid, colours in self.votes.items():
            total = sum(colours.values())
            if total < min_votes:
                continue
            color = max(colours.items(), key=lambda kv: kv[1])[0]
            si, k, sp, col, row = self.seats[sid]
            out.append({'straight': si, 'column': col, 'row': row,
                        'p': sp, 'color': color, 'votes': total})
        return out

    def straight_is_plausible(self, straight, min_votes=MIN_SEAT_VOTES):
        """The rules allow exactly 1 or 2 obstacles per straight -- a useful
        sanity check before trusting a straight's result."""
        n = len([s for s in self.occupied_seats(min_votes)
                 if s['straight'] == straight])
        return 1 <= n <= 2, n