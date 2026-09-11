#!/usr/bin/env python3
"""Offline check of the obstacle path planner. No ROS, no robot.

Verifies for every legal obstacle constellation:
  1. the planned offset really clears the block (robot half-width + block half)
  2. the path stays inside the lane (wall margin)
  3. the lane change is not steeper than what the robot demonstrated
     (~0.40 m lateral per 0.40 m longitudinal at 0.45 m/s)
"""

import math
from obstacle_path import (ObstaclePathPlanner, COLOR_RED, COLOR_GREEN,
                           BLOCK_HALF, ROBOT_HALF)

LANE = 1.00
Q_OUTER_SEAT = 0.40          # outer column: 0.40 m from the outer wall
Q_INNER_SEAT = 0.60          # inner column: 0.40 m from the inner wall
S_ROW0, S_ROW2 = 1.00, 2.00  # two obstacles are 1.0 m apart
STRAIGHT = 3.00

pl = ObstaclePathPlanner(lane_width=LANE)


def clearance_ok(q_robot, q_block):
    """Free space between robot edge and block edge [m]."""
    return abs(q_robot - q_block) - (ROBOT_HALF + BLOCK_HALF)


def check(name, obstacles):
    pts = pl.plan(obstacles, STRAIGHT)
    dense = pl.densify(pts, 0.02)
    ok = True
    msgs = []

    # 1 + 2: clearance at each obstacle, and lane containment everywhere
    for (s_o, q_o, c) in obstacles:
        q_at = min(dense, key=lambda p: abs(p[0] - s_o))[1]
        gap = clearance_ok(q_at, q_o)
        side = "links" if q_at > q_o else "rechts"
        want = "links" if c == COLOR_GREEN else "rechts"
        if side != want:
            ok = False; msgs.append(f"FALSCHE SEITE bei s={s_o}: {side}, erwartet {want}")
        if gap < 0.02:
            ok = False; msgs.append(f"ZU ENG bei s={s_o}: nur {gap*100:.1f} cm frei")
        else:
            msgs.append(f"s={s_o:.1f} {('gruen' if c==COLOR_GREEN else 'rot')}: "
                        f"q={q_at:.2f} ({side}), {gap*100:.1f} cm frei")

    for s, q in dense:
        if q < ROBOT_HALF or q > LANE - ROBOT_HALF:
            ok = False; msgs.append(f"AUSSERHALB Gasse bei s={s:.2f}: q={q:.2f}")
            break

    # 3: steepest lane change
    max_slope, seg = 0.0, ""
    for i in range(len(pts) - 1):
        ds = pts[i+1][0] - pts[i][0]
        dq = abs(pts[i+1][1] - pts[i][1])
        if ds > 1e-6 and dq > 1e-6:
            sl = dq / ds
            if sl > max_slope:
                max_slope, seg = sl, f"{dq*100:.0f}cm quer in {ds*100:.0f}cm laengs"
    if max_slope > 1.05:
        ok = False; msgs.append(f"ZU STEIL: {seg} (Faktor {max_slope:.2f})")
    elif seg:
        msgs.append(f"steilster Wechsel: {seg} (Faktor {max_slope:.2f})")

    print(f"\n{name}: {'OK' if ok else 'FEHLER <<<'}")
    for m in msgs:
        print("   ", m)
    return ok


print("=== Pfadplanung: alle legalen Konstellationen ===")
print(f"Gasse {LANE} m, Sitze bei q={Q_OUTER_SEAT}/{Q_INNER_SEAT}, "
      f"Roboter {ROBOT_HALF*2*100:.0f} cm breit, Kloetze {BLOCK_HALF*2*100:.1f} cm\n")

all_ok = True
# --- ein Hindernis ---
for q, qn in ((Q_OUTER_SEAT, "aussen"), (Q_INNER_SEAT, "innen")):
    for c, cn in ((COLOR_RED, "rot"), (COLOR_GREEN, "gruen")):
        all_ok &= check(f"1 Hindernis {qn} {cn}", [(S_ROW0, q, c)])

# --- zwei Hindernisse, 1 m auseinander, alle 4 Spalten-Kombis x 4 Farb-Kombis ---
for q1, n1 in ((Q_OUTER_SEAT, "a"), (Q_INNER_SEAT, "i")):
    for q2, n2 in ((Q_OUTER_SEAT, "a"), (Q_INNER_SEAT, "i")):
        for c1, cn1 in ((COLOR_RED, "rot"), (COLOR_GREEN, "gruen")):
            for c2, cn2 in ((COLOR_RED, "rot"), (COLOR_GREEN, "gruen")):
                all_ok &= check(f"2 Hindernisse {n1}{cn1[0]} -> {n2}{cn2[0]}",
                                [(S_ROW0, q1, c1), (S_ROW2, q2, c2)])

print("\n" + "="*60)
print("ALLE KONSTELLATIONEN OK" if all_ok else "MINDESTENS EINE KONSTELLATION PROBLEMATISCH")