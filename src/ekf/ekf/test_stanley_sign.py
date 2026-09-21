#!/usr/bin/env python3
"""
Offline sign test for the Stanley cross-track law.

No ROS, no robot. Constructs cases: robot offset LEFT and RIGHT of a target
line, for all four travel directions (+x, -y, -x, +y). Checks that omega steers
the robot BACK toward the line (correct sign) in every case.

Convention: positive omega = LEFT.
  - robot LEFT of line  -> must steer RIGHT (omega < 0)
  - robot RIGHT of line -> must steer LEFT  (omega > 0)
"""

import math


def wrap(a):
    return math.atan2(math.sin(a), math.cos(a))


# ---- paste the CURRENT _stanley_steer body here as a plain function ----
def stanley(x, y, theta, target_line, u_dir, v_ist=0.35,
            k_stanley=1.4, max_steer=math.radians(25), wheelbase=0.10):
    ux, uy = u_dir
    un = math.hypot(ux, uy) or 1e-9
    ux, uy = ux / un, uy / un
    lx, ly = -uy, ux
    nx, ny, d = target_line
    dist_along_n = (nx * x + ny * y) - d
    n_dot_left = nx * lx + ny * ly
    e_ct = -dist_along_n * (1.0 if n_dot_left >= 0 else -1.0)
    heading_line = math.atan2(uy, ux)
    e_theta = wrap(heading_line - theta)
    v = max(abs(v_ist), 0.05)
    delta = e_theta + math.atan2(k_stanley * e_ct, v)
    delta = max(-max_steer, min(max_steer, delta))
    omega = v * math.tan(delta) / wheelbase
    return omega, e_ct
# -----------------------------------------------------------------------


def line_through(px, py, ux, uy):
    """HNF of the line through (px,py) with direction (ux,uy). Normal = left of dir."""
    n = math.hypot(ux, uy)
    ux, uy = ux / n, uy / n
    nx, ny = -uy, ux           # left normal (arbitrary sign choice)
    d = nx * px + ny * py
    return (nx, ny, d)


# four travel directions and a line along each through the origin
dirs = {
    "+x": (1.0, 0.0),
    "-y": (0.0, -1.0),
    "-x": (-1.0, 0.0),
    "+y": (0.0, 1.0),
}

print(f"{'travel':>6} {'side':>6} {'e_ct':>8} {'omega':>8}  {'verdict'}")
print("-" * 45)
all_ok = True
for name, (ux, uy) in dirs.items():
    line = line_through(0.0, 0.0, ux, uy)
    # robot heading = travel direction (aligned), offset 0.1 m to LEFT and RIGHT
    theta = math.atan2(uy, ux)
    lx, ly = -uy, ux           # left of travel
    for side, sgn in (("LEFT", +1.0), ("RIGHT", -1.0)):
        rx, ry = 0.1 * sgn * lx, 0.1 * sgn * ly     # robot 10cm to that side
        omega, e_ct = stanley(rx, ry, theta, line, (ux, uy))
        # correct: LEFT -> omega<0 (steer right), RIGHT -> omega>0 (steer left)
        want_negative = (side == "LEFT")
        ok = (omega < 0) == want_negative or abs(omega) < 1e-6
        all_ok = all_ok and ok
        print(f"{name:>6} {side:>6} {e_ct:+8.3f} {omega:+8.3f}  {'OK' if ok else 'FALSCH <<<'}")
print("-" * 45)
print("ALLE RICHTUNGEN OK" if all_ok else "MINDESTENS EINE RICHTUNG FALSCH -- Vorzeichenfehler!")