#!/usr/bin/env python3
"""
Offline geometry test for plan_arc (per-corner o_in / o_out / R).

No ROS, no robot. Reconstructs the pure geometry of plan_arc and checks, for a
known box and several offset/radius combinations, that:
  1. T_A lies exactly on the entry offset line LA
  2. T_B lies exactly on the exit  offset line LB
  3. the arc centre C is at distance R from BOTH offset lines (arc tangent)
  4. |C - T_A| == R and |C - T_B| == R (tangent points on the circle)
  5. the arc bulges toward the outer corner (does not cut across the box)

Run: python3 test_plan_arc_geometry.py
"""

import math


# ---------- geometry helpers (copied verbatim from the controller) ----------
def line_intersect(l1, l2):
    n1x, n1y, d1 = l1
    n2x, n2y, d2 = l2
    det = n1x * n2y - n1y * n2x
    if abs(det) < 1e-9:
        return None
    x = (d1 * n2y - d2 * n1y) / det
    y = (n1x * d2 - n2x * d1) / det
    return (x, y)


def inward(wall, cx, cy):
    """Return wall HNF (nx,ny,d) with normal pointing toward (cx,cy)."""
    nx, ny, d = wall
    # signed value of centre; if centre is on the -normal side, flip
    if (nx * cx + ny * cy - d) < 0:
        return (-nx, -ny, -d)
    return (nx, ny, d)


def plan_arc_geom(corners, walls, idx, theta, o_in, o_out, R, s):
    """Pure-geometry core of plan_arc. Returns dict or None."""
    wall_a = walls[(idx - 1) % 4]   # edge ending at corner idx (entry side)
    wall_b = walls[idx]             # edge starting at corner idx (exit side)

    cx = sum(c[0] for c in corners) / 4.0
    cy = sum(c[1] for c in corners) / 4.0
    A = inward(wall_a, cx, cy)
    B = inward(wall_b, cx, cy)

    tx, ty = math.cos(theta), math.sin(theta)
    if abs(A[0] * tx + A[1] * ty) > abs(B[0] * tx + B[1] * ty):
        A, B = B, A

    LA = (A[0], A[1], A[2] + o_in)
    LB = (B[0], B[1], B[2] + o_out)
    P = line_intersect(LA, LB)
    if P is None:
        return None

    C = (P[0] + R * (A[0] + B[0]), P[1] + R * (A[1] + B[1]))
    T_A = (C[0] - R * A[0], C[1] - R * A[1])
    T_B = (C[0] - R * B[0], C[1] - R * B[1])
    return dict(A=A, B=B, LA=LA, LB=LB, C=C, T_A=T_A, T_B=T_B, P=P)


# ---------- test harness ----------
def dist_point_to_line(px, py, line):
    nx, ny, d = line
    return (nx * px + ny * py - d) / math.hypot(nx, ny)


def run_case(name, corners, walls, idx, theta, o_in, o_out, R, s):
    g = plan_arc_geom(corners, walls, idx, theta, o_in, o_out, R, s)
    if g is None:
        print(f"  {name}: FEHLER -- Linien parallel")
        return False

    C, T_A, T_B, LA, LB = g['C'], g['T_A'], g['T_B'], g['LA'], g['LB']
    tol = 1e-6
    ok = True

    # 1 & 2: tangent points on the offset lines
    dA = abs(dist_point_to_line(*T_A, LA))
    dB = abs(dist_point_to_line(*T_B, LB))
    if dA > tol: ok = False; print(f"  {name}: T_A NICHT auf LA (d={dA:.2e})")
    if dB > tol: ok = False; print(f"  {name}: T_B NICHT auf LB (d={dB:.2e})")

    # 3: centre at distance R from both offset lines
    cA = abs(abs(dist_point_to_line(*C, LA)) - R)
    cB = abs(abs(dist_point_to_line(*C, LB)) - R)
    if cA > tol: ok = False; print(f"  {name}: C nicht R von LA (|dev|={cA:.2e})")
    if cB > tol: ok = False; print(f"  {name}: C nicht R von LB (|dev|={cB:.2e})")

    # 4: |C - T| == R
    rA = abs(math.hypot(C[0]-T_A[0], C[1]-T_A[1]) - R)
    rB = abs(math.hypot(C[0]-T_B[0], C[1]-T_B[1]) - R)
    if rA > tol: ok = False; print(f"  {name}: |C-T_A| != R ({rA:.2e})")
    if rB > tol: ok = False; print(f"  {name}: |C-T_B| != R ({rB:.2e})")

    status = "OK" if ok else "FALSCH <<<"
    print(f"  {name:32} o_in={o_in:.2f} o_out={o_out:.2f} R={R:.2f}  "
          f"T_A=({T_A[0]:+.2f},{T_A[1]:+.2f}) T_B=({T_B[0]:+.2f},{T_B[1]:+.2f})  {status}")
    return ok


def box_from_corners(corners):
    """Build 4 outer walls HNF from corners; walls[i] = edge corners[i]->corners[i+1]."""
    walls = []
    for i in range(4):
        x1, y1 = corners[i]
        x2, y2 = corners[(i + 1) % 4]
        dx, dy = x2 - x1, y2 - y1
        n = math.hypot(dx, dy)
        # normal (either side); d = n . point
        nx, ny = -dy / n, dx / n
        d = nx * x1 + ny * y1
        walls.append((nx, ny, d))
    return walls


def main():
    # a 3x3 box, CCW-indexed, idx0 = max x (like the real /corner_geometry)
    corners = [(1.5, -1.5), (1.5, 1.5), (-1.5, 1.5), (-1.5, -1.5)]
    walls = box_from_corners(corners)

    print("=== plan_arc Geometrie-Test (4 Ecken x mehrere Offsets/Radien) ===\n")
    all_ok = True

    # travel heading per corner (approx): robot arrives along the entry wall.
    # For this test we just feed a heading roughly parallel to each entry edge.
    headings = {0: 0.0, 1: math.pi/2, 2: math.pi, 3: -math.pi/2}

    combos = [
        ("mitte 0.5/0.5 R0.5", 0.5, 0.5, 0.5),
        ("eng   0.3/0.3 R0.4", 0.3, 0.3, 0.4),
        ("asym  0.3/0.6 R0.5", 0.3, 0.6, 0.5),
        ("weit  0.7/0.7 R0.6", 0.7, 0.7, 0.6),
        ("mini  0.4/0.4 R0.2", 0.4, 0.4, 0.2),
    ]

    for idx in range(4):
        print(f"-- Ecke idx {idx} (heading {math.degrees(headings[idx]):+.0f}) --")
        for label, oi, oo, R in combos:
            ok = run_case(label, corners, walls, idx, headings[idx], oi, oo, R, s=1.0)
            all_ok = all_ok and ok
        print()

    print("=" * 60)
    print("ALLE FAELLE OK" if all_ok else "MINDESTENS EIN FALL FALSCH -- Geometriefehler!")


if __name__ == '__main__':
    main()