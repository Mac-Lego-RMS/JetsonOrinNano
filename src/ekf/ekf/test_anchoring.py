#!/usr/bin/env python3
"""Anchoring of the map: field pose of the robot -> field pose of the odom origin.

    python3 -m ekf.test_anchoring

generate_map() expects the pose of the point where the odometry was zeroed,
not that of the robot. On a normal start that is the same -- the
robot stands still until detection is done. After unparking there are 50 cm
in between, and without the conversion the whole map is offset by the unpark
travel. Exactly that drove a run into the wall (1.67 m
lateral deviation in the first control step).
"""
import math

import numpy as np

from ekf.scan_processor_node import ScanProcessor
from ekf.field_map import generate_map, START_POSES_CW


def check(name, cond, extra=''):
    if not cond:
        raise AssertionError('FAILED: %s %s' % (name, extra))
    print('  ok  %s%s' % (name, ('  ' + extra) if extra else ''))


class Stub:
    _anchored = ScanProcessor._anchored

    def __init__(self, commit_pose=(0.0, 0.0, 0.0)):
        self.commit_pose = commit_pose


def robot_in_field(anchor, odom_pose):
    """Where is the robot in the field if the origin lies at ``anchor``?"""
    xa, ya, tha = anchor
    xo, yo, tho = odom_pose
    c, s = math.cos(tha), math.sin(tha)
    return (xa + c * xo - s * yo, ya + s * xo + c * yo, tha + tho)


print('Anchoring')

f = Stub()
field = START_POSES_CW['pos1']
check('at the origin nothing changes',
       all(abs(a - b) < 1e-12 for a, b in zip(f._anchored(field), field)),
       '%s' % (f._anchored(field),))

# The robot has moved since the zeroing: the anchor must lie so that
# it stands on the detected field pose NOW.
for odom in ((0.25, -0.33, math.radians(-34.6)),
             (0.5, 0.0, 0.0),
             (-0.2, 0.7, math.radians(120.0)),
             (1.3, -0.4, math.radians(-175.0))):
    f = Stub(odom)
    anchor = f._anchored(field)
    back = robot_in_field(anchor, odom)
    fits = (abs(back[0] - field[0]) < 1e-9 and abs(back[1] - field[1]) < 1e-9
            and abs(math.atan2(math.sin(back[2] - field[2]),
                               math.cos(back[2] - field[2]))) < 1e-9)
    check('robot lands on the detected field pose (odom %+.2f,%+.2f,%+.0f)'
          % (odom[0], odom[1], math.degrees(odom[2])), fits,
          'anchor (%+.3f,%+.3f,%+.0f)' % (anchor[0], anchor[1], math.degrees(anchor[2])))

print('\nEffect on the map')
# Without anchoring the map would be offset by the distance driven.
odom = (0.25, -0.33, math.radians(-34.6))
f = Stub(odom)
without = generate_map(field)
with_anchor = generate_map(f._anchored(field))
offset = max(abs(a['d'] - b['d']) for a, b in zip(without, with_anchor))
check('without anchoring the map is clearly off', offset > 0.20,
      'largest distance error %.2f m' % offset)

f0 = Stub()
same = generate_map(f0._anchored(field))
check('at the origin the map is unchanged',
      all(abs(a['d'] - b['d']) < 1e-12 for a, b in zip(without, same)))

print('\nall tests passed')
