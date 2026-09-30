#!/usr/bin/env python3
"""Which obstacle seats of the NEXT straight can the camera see before the corner?

Pure geometry, no bag needed:

    python3 plot_fov_coverage.py [--out-dir docs/figures]

The robot drives along the centre of a 1 m lane towards the corner. For every
distance to the front wall it counts the 6 seats of the next straight that
are inside the horizontal field of view AND not hidden behind the 1 x 1 m
inner block, once for the old forward camera (120 deg) and once for the
fisheye (240 deg; a board blocks the rear for LiDAR and camera, and the
software cuts +-60 deg around the rear, wall_extraction.BLOCK_ANGLE). A second
pair of curves only counts seats within the range the colour is trusted
(obstacle_map.py, 1.60 m).

Simplifications: camera at the LiDAR position, pillars as points, the robot
exactly on the lane centre and parallel to the walls.
"""
import argparse
import math

import numpy as np

import bagio
import style
from robot_constants import INNER_HALF, OUTER_HALF, seats_field, square

FOV_OLD = 120.0
FOV_NEW = 240.0     # 360 - 2 * wall_extraction.BLOCK_ANGLE
COLOUR_RANGE = 1.60     # obstacle_map.py: colour only trusted up to here
SCAN_HALT_FRONT = 1.10  # round1_controller: scan halt in front of the wall
LOOKAHEAD_HALT_FRONT = 1.85
LANE_X = -1.0           # centre of the west lane (outer -1.5, inner -0.5)
HEADING = -math.pi / 2  # CCW on the west straight = driving towards -y


def blocked_by_inner(a, b, n=60):
    """True if the straight line a -> b passes through the inner block."""
    t = np.linspace(0.0, 1.0, n)[1:-1, None]
    pts = a + t * (b - a)
    return bool(np.any((np.abs(pts[:, 0]) < INNER_HALF) & (np.abs(pts[:, 1]) < INNER_HALF)))


def visible(robot, heading, seat, fov_deg, max_range=None):
    d = seat - robot
    if max_range is not None and np.hypot(*d) > max_range:
        return False
    bearing = math.atan2(d[1], d[0]) - heading
    bearing = (bearing + math.pi) % (2 * math.pi) - math.pi
    if abs(math.degrees(bearing)) > fov_deg / 2:
        return False
    return not blocked_by_inner(robot, seat)


def next_straight_seats():
    # straight 1 = the west straight rotated by 90 deg CCW = the south one,
    # which is the next one when driving CCW down the west straight
    return [s for s in seats_field() if s['straight'] == 1]


def count_visible(front_dist, fov, max_range=None):
    robot = np.array([LANE_X, -OUTER_HALF + front_dist])
    return sum(visible(robot, HEADING, s['p'], fov, max_range) for s in next_straight_seats())


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--out-dir', default=str(bagio.DEFAULT_FIG_DIR))
    args = ap.parse_args()
    style.apply_style()

    fronts = np.linspace(0.30, 2.40, 211)
    curves = {
        (FOV_OLD, None): [count_visible(f, FOV_OLD) for f in fronts],
        (FOV_NEW, None): [count_visible(f, FOV_NEW) for f in fronts],
        (FOV_OLD, COLOUR_RANGE): [count_visible(f, FOV_OLD, COLOUR_RANGE) for f in fronts],
        (FOV_NEW, COLOUR_RANGE): [count_visible(f, FOV_NEW, COLOUR_RANGE) for f in fronts],
    }

    fig, (ax_map, ax) = style.figure(1, 2, width=8.0, height=3.6,
                                     gridspec_kw={'width_ratios': [1, 1.4]})

    # --- left: field view at the scan halt ---------------------------------
    robot = np.array([LANE_X, -OUTER_HALF + SCAN_HALT_FRONT])
    for half in (OUTER_HALF, INNER_HALF):
        sq = square(half)
        ax_map.plot(sq[:, 0], sq[:, 1], color=style.INK_2, lw=style.LINE_W)
    for fov, colour in ((FOV_NEW, style.CAT[0]), (FOV_OLD, style.CAT[1])):
        a0 = HEADING - math.radians(fov / 2)
        a1 = HEADING + math.radians(fov / 2)
        ang = np.linspace(a0, a1, 80)
        r = 0.55 if fov == FOV_NEW else 0.75
        wedge = np.vstack([robot, robot + r * np.c_[np.cos(ang), np.sin(ang)], robot])
        ax_map.fill(wedge[:, 0], wedge[:, 1], color=colour, alpha=0.18, lw=0)
        ax_map.plot(wedge[:, 0], wedge[:, 1], color=colour, lw=1.0,
                    label=f'{fov:.0f} deg')
    for s in seats_field():
        nxt = s['straight'] == 1
        vis_new = nxt and visible(robot, HEADING, s['p'], FOV_NEW)
        vis_old = nxt and visible(robot, HEADING, s['p'], FOV_OLD)
        if not nxt:
            ax_map.plot(*s['p'], 'o', ms=3, color=style.AXIS)
            continue
        face = style.CAT[1] if vis_old else (style.CAT[0] if vis_new else 'white')
        ax_map.plot(*s['p'], 's', ms=6, mfc=face, mec=style.INK_2, mew=0.8)
    ax_map.plot(*robot, marker=(3, 0, 180), ms=9, color=style.INK)
    ax_map.set_aspect('equal')
    ax_map.set_xlim(-1.6, 1.6)
    ax_map.set_ylim(-1.6, 1.6)
    ax_map.set_xticks([])
    ax_map.set_yticks([])
    for sp in ax_map.spines.values():
        sp.set_visible(False)
    ax_map.set_title(f'View at the scan halt ({SCAN_HALT_FRONT:.2f} m)')
    ax_map.plot([], [], 's', ms=6, mfc=style.CAT[1], mec=style.INK_2, label='seen with both')
    ax_map.plot([], [], 's', ms=6, mfc=style.CAT[0], mec=style.INK_2, label=f'seen with {FOV_NEW:.0f} deg only')
    ax_map.legend(loc='upper center', bbox_to_anchor=(0.5, -0.02), ncol=2, fontsize=7, frameon=False)

    # --- right: count over the approach ------------------------------------
    for (fov, rng), vals in sorted(curves.items(), key=lambda kv: -kv[0][0]):
        colour = style.CAT[0] if fov == FOV_NEW else style.CAT[1]
        ls = '-' if rng is None else '--'
        lw = 2.6 if fov == FOV_NEW else 1.3
        lab = f'{fov:.0f} deg' + ('' if rng is None else f', colour range {rng:.2f} m')
        ax.step(fronts, vals, where='mid', color=colour, ls=ls, lw=lw, label=lab)
    for x, txt in ((SCAN_HALT_FRONT, 'scan halt'), (LOOKAHEAD_HALT_FRONT, 'look-ahead halt')):
        style.ref_line(ax, x=x, label=txt)
    ax.set_xlabel('distance to the front wall [m]')
    ax.set_ylabel('seats of the next straight visible')
    ax.set_ylim(-0.3, 6.5)
    ax.set_yticks(range(7))
    ax.invert_xaxis()
    ax.set_title('Seats seen before the corner')
    style.legend_below(ax, ncol=2)

    style.save(fig, args.out_dir, 'fov_coverage',
               caption='Geometry only: lane centre, camera at the LiDAR, pillars as points  |  plot_fov_coverage.py')

    for f in (SCAN_HALT_FRONT, LOOKAHEAD_HALT_FRONT):
        print(f'front wall {f:.2f} m: 120 deg sees {count_visible(f, FOV_OLD)}/6 '
              f'({count_visible(f, FOV_OLD, COLOUR_RANGE)} within {COLOUR_RANGE} m), '
              f'{FOV_NEW:.0f} deg sees {count_visible(f, FOV_NEW)}/6 '
              f'({count_visible(f, FOV_NEW, COLOUR_RANGE)} within {COLOUR_RANGE} m)')


if __name__ == '__main__':
    main()
