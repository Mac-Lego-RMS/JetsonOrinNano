#!/usr/bin/env python3
"""
Obstacle-avoidance path planner for the WRO obstacle round.

Pure geometry, no ROS -- so it can be verified offline (see test at the bottom).

Track facts this relies on (from the perception spec):
  * Seats sit 0.10 m either side of the lane centre (two columns, 0.20 m apart).
  * Per straight there are 1 or 2 obstacles; two are ALWAYS 1.0 m apart
    (row 0 at the start of the straight, row 2 at the end) -- never the same row.
  * Block edge length 0.044 m, robot width 0.12 m.
  * Rule: RED  -> pass on the robot's RIGHT (block stays LEFT of the robot)
          GREEN-> pass on the robot's LEFT  (block stays RIGHT of the robot)

Output is a polyline in lane coordinates:
    s  = distance along the straight (0 at its start)
    q  = lateral offset from the OUTER wall (same convention as o_in/o_out)
The caller maps (s,q) to map coordinates using the straight's wall geometry.
"""

import math

BLOCK_HALF = 0.022          # 44 mm / 2
ROBOT_HALF = 0.06           # 120 mm / 2

COLOR_UNKNOWN, COLOR_RED, COLOR_GREEN = 0, 1, 2


class ObstaclePathPlanner:
    def __init__(self, lane_width=1.00, clear_before=0.20, clear_after=0.20,
                 transition_pref=0.60, transition_min=0.40, wall_margin=0.12):
        """
        lane_width      : outer wall -> inner wall [m]
        clear_before    : be ON the new offset this far BEFORE the obstacle [m]
        clear_after     : hold the offset this far AFTER the obstacle [m]
        transition_pref : preferred lane-change length [m] (used if room allows)
        transition_min  : shortest lane change we dare (measured ~0.40 m @0.45 m/s)
        wall_margin     : never plan closer than this to a wall (robot centre) [m]
        """
        self.lane_width = lane_width
        self.clear_before = clear_before
        self.clear_after = clear_after
        self.transition_pref = transition_pref
        self.transition_min = transition_min
        self.wall_margin = wall_margin

    # ---------------------------------------------------------------- offsets
    def pass_offset(self, obstacle_q, color, ccw):
        """Lateral offset (from the OUTER wall) to pass this obstacle.

        The robot drives MIDWAY between the block and the wall it passes on --
        safety first.

        RULE: red -> pass on the robot's RIGHT (block stays LEFT of the robot),
              green-> pass on the robot's LEFT.
        WHICH LANE SIDE that is depends on the DRIVE DIRECTION:
          CCW: field centre is left  -> inner band LEFT,  outer wall RIGHT
          CW : field centre is right -> inner band RIGHT, outer wall LEFT
        So red+CCW and green+CW both mean "pass on the OUTER side" (small q),
        the other two mean "pass on the INNER side" (large q).
        """
        pass_outer = ((color != COLOR_GREEN) == bool(ccw))
        if pass_outer:
            near, far = 0.0, obstacle_q - BLOCK_HALF
        else:
            near, far = obstacle_q + BLOCK_HALF, self.lane_width
        q = 0.5 * (near + far)
        return min(max(q, self.wall_margin), self.lane_width - self.wall_margin)

    def gap_width(self, obstacle_q, color, ccw):
        """Free width of the gap we plan to drive through [m] (for diagnostics)."""
        if ((color != COLOR_GREEN) == bool(ccw)):
            return obstacle_q - BLOCK_HALF
        return self.lane_width - (obstacle_q + BLOCK_HALF)

    # ---------------------------------------------------------------- planning
    def plan(self, obstacles, straight_length, ccw, q_start=None, s_start=0.0,
             q_default=None):
        """Build the (s, q) polyline for one straight.

        obstacles : list of (s_obs, q_obs, color), s along the straight
        ccw       : True if driving counter-clockwise (decides the pass side!)
        q_start   : lateral offset the robot is on right now (None -> use first
                    obstacle's offset from the very beginning)
        s_start   : where the plan starts (robot's current s; >0 when replanning
                    mid-straight after a late detection)
        q_default : offset to use where no obstacle dictates one
        """
        obs = sorted(obstacles, key=lambda o: o[0])
        if q_default is None:
            q_default = 0.5 * self.lane_width

        # target offset per obstacle
        targets = [(s, self.pass_offset(q, c, ccw)) for (s, q, c) in obs]

        pts = []
        if not targets:
            q0 = q_default if q_start is None else q_start
            return [(s_start, q0), (straight_length, q_default)]

        # start: either where we are, or already on the first target
        q_cur = targets[0][1] if q_start is None else q_start
        s_cur = s_start
        pts.append((s_cur, q_cur))

        for i, (s_obs, q_tgt) in enumerate(targets):
            # must be on q_tgt by this s:
            s_need = s_obs - self.clear_before
            if q_tgt != q_cur:
                room = max(s_need - s_cur, 0.0)
                length = min(self.transition_pref, room) if room > 0 else 0.0
                if length < self.transition_min:
                    # late detection: use whatever room is left, even if steep
                    length = room
                s_ramp_end = s_need
                s_ramp_start = max(s_cur, s_need - length)
                if s_ramp_start > s_cur:
                    pts.append((s_ramp_start, q_cur))       # hold until ramp
                pts.append((s_ramp_end, q_tgt))             # S-curve added later
                q_cur = q_tgt
                s_cur = s_ramp_end
            # hold the offset past the obstacle
            s_hold = s_obs + self.clear_after
            pts.append((s_hold, q_cur))
            s_cur = s_hold

        # run out to the end of the straight on the last offset
        if s_cur < straight_length:
            pts.append((straight_length, q_cur))
        return pts

    @staticmethod
    def max_slope(pts):
        """Steepest lane change in the plan: lateral metres per longitudinal metre.
        The controller uses this to slow down before a steep swap (measured limit
        was ~1.0 at 0.45 m/s, so anything approaching that wants less speed)."""
        worst = 0.0
        for i in range(len(pts) - 1):
            ds = pts[i + 1][0] - pts[i][0]
            dq = abs(pts[i + 1][1] - pts[i][1])
            if ds > 1e-6 and dq > 1e-6:
                worst = max(worst, dq / ds)
        return worst

    # ------------------------------------------------------------- smoothing
    @staticmethod
    def densify(pts, step=0.05):
        """Turn the corner points into a dense polyline with SMOOTH (cosine)
        transitions -- a straight ramp would leave a kink that Stanley has to
        absorb; the cosine blend is tangent-continuous at both ends."""
        out = []
        for i in range(len(pts) - 1):
            s0, q0 = pts[i]
            s1, q1 = pts[i + 1]
            if s1 <= s0:
                continue
            n = max(int((s1 - s0) / step), 1)
            for k in range(n):
                t = k / n
                if abs(q1 - q0) < 1e-9:
                    q = q0                                   # straight section
                else:
                    q = q0 + (q1 - q0) * 0.5 * (1.0 - math.cos(math.pi * t))
                out.append((s0 + (s1 - s0) * t, q))
        out.append(pts[-1])
        return out