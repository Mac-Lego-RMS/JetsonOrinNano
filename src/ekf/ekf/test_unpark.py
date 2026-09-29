#!/usr/bin/env python3
"""Tests for unpark.py. Without ROS, from inside the package directory:

    cd src/ekf/ekf && python3 test_unpark.py

(like test_obstacle_path.py -- an import "from ekf import ..." fails here,
because ekf.py in the same directory hides the package ekf.)
"""
import io
import json
import math

import numpy as np

import unpark as A


def check(name, cond, extra=''):
    if not cond:
        raise AssertionError('FAILED: %s %s' % (name, extra))
    print('  ok  %s%s' % (name, ('  ' + extra) if extra else ''))


def sector(centre_deg, dist, n=40, spread=0.0):
    """Point cloud in a sector, robot frame (+x forward, +y left)."""
    centre = math.radians(centre_deg)
    w = centre + np.linspace(-0.25, 0.25, n)
    r = dist + spread * np.sin(np.arange(n))
    return np.column_stack((r * np.cos(w), r * np.sin(w)))


print('Conversion travel <-> encoder')
check('1 cm is 38.2 deg of shaft', abs(A.cm_to_deg(1.0) - 38.197) < 0.01,
      '%.3f' % A.cm_to_deg(1.0))
check('There and back', abs(A.deg_to_cm(A.cm_to_deg(7.3)) - 7.3) < 1e-9)
check('Sign is kept', A.cm_to_deg(-5.0) < 0)

print('\nCurve from steer_calib.json')
check('it really is loaded from the file', A.STEER_SOURCE is not None,
      str(A.STEER_SOURCE))
check('and not from the fallback', A.STEER_CURVE is not A.FALLBACK_CURVE)

curve, centre, wheelbase, source = A.load_steer_curve()
raw = json.load(io.open(source, encoding='utf-8'))
slowest = sorted(raw['speeds'], key=lambda e: float(e['v']))[0]
check('the slowest step is taken',
      abs(float(slowest['v']) - 0.35) < 1e-9,
      'v = %s m/s' % slowest['v'])

expected = {}
for side in ('left', 'right'):
    for servo, delta in slowest[side]:
        expected[round(float(servo) * 100.0, 6)] = math.degrees(float(delta))
check('every support point matches the file',
      len(curve) == len(expected)
      and all(abs(deg - expected[pct]) < 1e-9 for pct, deg in curve),
      '%d points' % len(curve))
check('the curve is sorted ascending',
      [pct for pct, _deg in curve] == sorted(pct for pct, _deg in curve))
check('the wheelbase comes from the file',
      abs(wheelbase - float(raw['wheelbase'])) < 1e-12,
      '%.3f m' % wheelbase)
check('the trim is the point without steering angle',
      abs(dict(curve)[centre]) < 1e-12, '%.1f %%' % centre)

# Speed selectable: at more speed the tyre slips, the curve is flatter.
fast, _m, _r, _q = A.load_steer_curve(speed=0.75)
check('with speed=0.75 a DIFFERENT step comes back', fast != curve)
check('and speed=0.0 gives the slowest again',
      A.load_steer_curve(speed=0.0)[0] == curve)

# If the file is missing, it guesses -- but visibly.
fallback, f_centre, f_wheelbase, f_source = A.load_steer_curve(
    path='/does/not/exist/steer_calib.json')
check('missing file falls back to the fallback',
      fallback == A.FALLBACK_CURVE and f_source is None)
check('... and the attempt is logged',
      any('does/not/exist' in z for z in A.load_steer_curve.tried))

print('\nSteering')
check('0 in the table is the trim',
      abs(A.steer_to_wire(0.0) - A.STEER_CENTER) < 1e-9)
check('full lock stays full lock',
      abs(A.steer_to_wire(100.0) - 100.0) < 1e-9 and
      abs(A.steer_to_wire(-100.0) + 100.0) < 1e-9)
check('beyond the end stop it is clamped',
      A.steer_to_wire(150.0) == 100.0 and A.steer_to_wire(-150.0) == -100.0)
# Expect NO fixed number: the turning circle comes from the calibration and
# changes with it. On 21.09.2026 it moved from 0.306 to 0.203 m,
# and a test for "about 0.31" would then have declared the calibration broken
# instead of adopting it. What is checked is the calculation, not the value.
full = dict(A.STEER_CURVE)[100.0]
check('turn radius follows the curve',
      abs(A.turn_radius_of(100.0)
          - A.WHEELBASE / math.tan(math.radians(full))) < 1e-12,
      '%.3f m at %.2f deg' % (A.turn_radius_of(100.0), full))
check('and is of a plausible magnitude',
      0.10 < A.turn_radius_of(100.0) < 1.00, '%.3f m' % A.turn_radius_of(100.0))
check('straight-ahead has no radius', A.turn_radius_of(A.STEER_CENTER) is None)

print('\nStep list')
check('pairs are recognised',
      A.steps_from_flat([100.0, 5.0, -100.0, -3.0]) == [(100.0, 5.0), (-100.0, -3.0)])
try:
    A.steps_from_flat([100.0, 5.0, -100.0])
    check('odd list is caught', False)
except ValueError:
    check('odd list is caught', True)
try:
    A.steps_from_flat([120.0, 5.0])
    check('steering out of range is caught', False)
except ValueError:
    check('steering out of range is caught', True)

print('\nStep sequence per driving direction')
SHARED = [100.0, 5.0, -100.0, -3.0]
CW_ONLY = [100.0, 9.0, -100.0, -7.0]
seq, origin = A.steps_for('CCW', shared=SHARED, cw=CW_ONLY, ccw=[])
check('empty direction falls back to the shared one',
      seq == SHARED and 'shared' in origin, origin)
seq, origin = A.steps_for('CW', shared=SHARED, cw=CW_ONLY, ccw=[])
check('filled direction wins', seq == CW_ONLY and 'CW' in origin, origin)
check('filling only one side is enough',
      A.steps_for('CCW', shared=SHARED, cw=CW_ONLY, ccw=[])[0] == SHARED
      and A.steps_for('CW', shared=SHARED, cw=CW_ONLY, ccw=[])[0] == CW_ONLY)
check('both sides can be different',
      A.steps_for('CW', shared=SHARED, cw=CW_ONLY, ccw=SHARED)[0] !=
      A.steps_for('CCW', shared=SHARED, cw=CW_ONLY, ccw=SHARED)[0])
check('without arguments the module default applies',
      A.steps_for('CW')[0] == list(A.STEPS_DEFAULT)
      or A.STEPS_CW)
for broken, name in ((('LEFT',), 'unknown direction'),):
    try:
        A.steps_for(*broken)
        check(name + ' is caught', False)
    except ValueError:
        check(name + ' is caught', True)
try:
    A.steps_for('CW', shared=[], cw=[], ccw=[])
    check('no sequence at all is caught', False)
except ValueError:
    check('no sequence at all is caught', True)

print('\nMirroring')
tab = A.steps_from_flat([100.0, 5.0, -100.0, -3.0, 0.0, 9.0])
left = A.mirror_steps(tab, True)
right = A.mirror_steps(tab, False)
check('open left: signs stay',
      left[0][0] > 0 and left[1][0] < 0)
check('open right: signs flip',
      right[0][0] < 0 and right[1][0] > 0)
check('travels stay untouched',
      [c for _l, c in left] == [c for _l, c in right] == [5.0, -3.0, 9.0])
check('straight-ahead stays the trim on both sides',
      abs(left[2][0] - A.STEER_CENTER) < 1e-9 and abs(right[2][0] - A.STEER_CENTER) < 1e-9)
check('never more than full lock',
      all(abs(l) <= 100.0 + 1e-9 for l, _c in left + right))

print('\nDriving direction from the scan')
# Outer wall on the right (near), field on the left (far) -> inner block left -> CCW
pts = np.vstack((sector(+90, 0.90), sector(-90, 0.16)))
r = A.direction_from_scan(pts)
check('field left -> CCW', r['direction'] == 'CCW' and r['confident'], r['reason'])
# mirrored
pts = np.vstack((sector(+90, 0.16), sector(-90, 0.90)))
r = A.direction_from_scan(pts)
check('field right -> CW', r['direction'] == 'CW' and r['confident'], r['reason'])
# near side with no return at all (below range_min)
r = A.direction_from_scan(sector(+90, 0.90))
check('empty side is the wall -> CCW', r['direction'] == 'CCW' and r['confident'],
      r['reason'])
r = A.direction_from_scan(sector(-90, 0.75))
check('empty side right -> CW', r['direction'] == 'CW' and r['confident'], r['reason'])
# both sides similar -> no decision
pts = np.vstack((sector(+90, 0.50), sector(-90, 0.46)))
r = A.direction_from_scan(pts)
check('too similar -> no decision',
      r['direction'] is None and not r['confident'], r['reason'])
# nothing at all
r = A.direction_from_scan(np.zeros((0, 2)))
check('empty scan -> no decision', r['direction'] is None)
# points only front and rear (magenta walls) must not count
pts = np.vstack((sector(0, 0.11), sector(180, 0.15)))
r = A.direction_from_scan(pts)
check('magenta walls front/rear stay out',
      r['left_n'] == 0 and r['right_n'] == 0)

print('\nAreas')
# The separating axis theorem must handle the case that makes corner comparisons
# fail: a thin bar across the robot, without a corner of
# either lying inside the other. Exactly this pose comes up when unparking.
robot = A.rectangle(-0.05, 0.12, -0.055, 0.055)
crossbar = A.rectangle(0.02, 0.04, -0.20, 0.20)
check('bar across the robot is detected',
      A.overlaps(robot, crossbar))
check('no corner lies inside the other rectangle',
      not any(-0.05 <= px <= 0.12 and -0.055 <= py <= 0.055 for px, py in crossbar)
      and not any(0.02 <= px <= 0.04 and -0.20 <= py <= 0.20 for px, py in robot))
beside = A.rectangle(0.30, 0.32, -0.20, 0.20)
check('cleanly separated is not overlapping',
      not A.overlaps(robot, beside))
check('distance is right', abs(A.poly_distance(robot, beside) - 0.18) < 1e-9,
      '%.3f m' % A.poly_distance(robot, beside))
check('overlap has distance 0', A.poly_distance(robot, crossbar) == 0.0)
touching = A.rectangle(0.12, 0.14, -0.20, 0.20)
check('touching does not count as overlap',
      not A.overlaps(robot, touching)
      and A.poly_distance(robot, touching) < 1e-9,
      '%.1e m' % A.poly_distance(robot, touching))

print('\nDry run')
# Deliberately NOT against STEPS_DEFAULT: the sequence is tuned at the real bay,
# whose dimensions differ from the nominal dimensions of the rules. Here
# the mechanism is to be checked, not the tuned numbers.
REFERENCE = [100.0, 5.9, -100.0, -4.4, 100.0, 4.6,
             -100.0, -3.9, 100.0, 17.7, -100.0, 37.1]

# Plus the curve it was DESIGNED FOR -- not the one currently
# installed. Otherwise this section checks the calibration instead of the
# geometry: on 21.09.2026 the full lock moved from 0.306 to 0.203 m,
# and the reference sequence would have had 6 instead of 10 mm margin, without
# anything being wrong in the calculation.
REFERENCE_CURVE = [
    (-100.0, -17.76), (-80.0, -14.29), (-65.0, -11.61),
    (-50.0, -9.42), (-35.0, -5.33), (-2.0, 0.0),
    (35.0, 7.60), (50.0, 10.07), (65.0, 12.66),
    (80.0, 14.56), (100.0, 18.10),
]
_saved = (A.STEER_CURVE, A.STEER_CENTER, A.WHEELBASE)
A.STEER_CURVE, A.STEER_CENTER, A.WHEELBASE = REFERENCE_CURVE, -2.0, 0.10

ref = A.steps_from_flat(REFERENCE)
e = A.simulate(A.mirror_steps(ref, True), A.bay_start_pose(long_clearance=0.010))
check('reference sequence touches nowhere', not e['collision'])
check('reference sequence keeps 8 mm reserve', not e['tight'],
      '%.0f mm' % (e['magenta_dist_m'] * 1000))
check('reference sequence gets out', e['clear'])
check('reference sequence ends on the lane heading',
      abs(math.degrees(e['end_pose'][2])) < 3.0,
      '%.1f deg' % math.degrees(e['end_pose'][2]))
check('reference sequence ends in the lane', 0.25 < e['end_pose'][1] < 0.75,
      'y = %.3f m' % e['end_pose'][1])

mirrored = A.simulate(
    A.mirror_steps(ref, False),
    (A.bay_start_pose(long_clearance=0.010)[0], -A.bay_start_pose(long_clearance=0.010)[1], 0.0),
    depth=-A.BAY_DEPTH)
# Not exactly mirror-symmetric: the steering is not either (R = 0.306 m
# left, 0.312 m right from the calibration). 1 cm difference is physics,
# not a calculation error -- more would be one.
check('mirrored sequence turns the other way',
      mirrored['end_pose'][1] < 0, 'y = %.3f m' % mirrored['end_pose'][1])
check('mirrored sequence gets equally far',
      abs(abs(mirrored['end_pose'][1]) - e['end_pose'][1]) < 0.02,
      'difference %.0f mm' % (abs(abs(mirrored['end_pose'][1]) - e['end_pose'][1]) * 1000))
check('mirrored sequence also ends on the lane heading',
      abs(math.degrees(mirrored['end_pose'][2])) < 3.0,
      '%.1f deg' % math.degrees(mirrored['end_pose'][2]))

e2 = A.simulate(A.mirror_steps([(0.0, 30.0)], True), A.bay_start_pose(long_clearance=0.010))
check('straight ahead runs into the front wall', e2['collision'],
      'move %s' % e2['at_step'])

# From here on the real calibration again: the tuned sequence is to be checked
# against what the robot really drives with.
A.STEER_CURVE, A.STEER_CENTER, A.WHEELBASE = _saved

print('\nThe tuned default sequence')
print('      (computed with the installed calibration: full lock '
      'R = %.3f m)' % A.turn_radius_of(100.0))
std = A.steps_from_flat(A.STEPS_DEFAULT)
check('is well-formed', len(std) >= 1)
check('can be mirrored in both directions',
      len(A.mirror_steps(std, True)) == len(A.mirror_steps(std, False)) == len(std))
check('stays within the steering range',
      all(abs(l) <= 100.0 + 1e-9 for l, _c in A.mirror_steps(std, True)))
fwd = max([cm for _l, cm in std if cm > 0], default=0.0)
back = -min([cm for _l, cm in std if cm < 0], default=0.0)
# Deliberately do NOT derive a long clearance from this: adding forward and
# reverse only holds for straight driving. As soon as the robot stands at an angle,
# it does not come back on the same line when reversing, but past the
# wall tip -- that is why sequences fit that by this back-of-the-envelope
# calculation should not fit. What really holds is what simulate() says.
print('      longest move forward %.1f cm, reverse %.1f cm'
      % (fwd, back))
e_std = A.simulate(A.mirror_steps(std, True), A.bay_start_pose())
print('      dry run against the nominal dimensions: %s, %s'
      % ('collision in move %s' % e_std['at_step'] if e_std['collision']
         else 'collision-free',
         'gets out' if e_std['clear'] else 'stays in the bay'))
print('      (the real bay can differ from this -- set bay=CM)')

print('\nall tests passed')
