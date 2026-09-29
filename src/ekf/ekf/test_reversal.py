#!/usr/bin/env python3
"""The way back must cancel the way out exactly, kinematically.

    python3 -m ekf.test_reversal

This is the basis of the unpark test (ekf/unpark_test_node.py): if it does
not hold on paper already, the test on the robot measures nothing useful.
"""
import math
import sys
import threading
import time
import types

from ekf import unpark as A
from ekf.unpark_test_node import (UnparkTest, in_start_frame, pose_text,
                                  reverse_steps, wrap)


def check(name, cond, extra=''):
    if not cond:
        raise AssertionError('FAILED: %s %s' % (name, extra))
    print('  ok  %s%s' % (name, ('  ' + extra) if extra else ''))


print('Reversal of the step sequence')

steps_out = A.mirror_steps(A.steps_from_flat(A.STEPS_DEFAULT), True)
steps_back = reverse_steps(steps_out)

check('same number of moves', len(steps_back) == len(steps_out))
check('order is reversed',
      [l for l, _c in steps_back] == [l for l, _c in reversed(steps_out)])
check('every travel is negated',
      [c for _l, c in steps_back] == [-c for _l, c in reversed(steps_out)])
check('the steering stays the same per move',
      [l for l, _c in steps_back] == [l for l, _c in reversed(steps_out)])
check('reversing twice gives the original', reverse_steps(steps_back) == steps_out)

print('\nCancellation in the kinematics')
start = (0.0, 0.0, 0.0)
after_out = A.trajectory(start, steps_out)[-1][0]
after_back = A.trajectory(after_out, steps_back)[-1][0]
long, lat, yaw = in_start_frame(start, after_back)
check('the way back lands at the start again',
      abs(long) < 1e-9 and abs(lat) < 1e-9 and abs(yaw) < 1e-9,
      '%.1e m / %.1e m / %.1e rad' % (long, lat, yaw))

# Also from a slanted start pose, and for the mirrored sequence.
for start in ((0.4, -0.2, math.radians(37.0)),
              (-1.1, 0.8, math.radians(-160.0))):
    for tab in (steps_out, A.mirror_steps(A.steps_from_flat(A.STEPS_DEFAULT), False)):
        end = A.trajectory(A.trajectory(start, tab)[-1][0], reverse_steps(tab))[-1][0]
        l, q, g = in_start_frame(start, end)
        check('also from (%.1f, %.1f, %+.0f deg)'
              % (start[0], start[1], math.degrees(start[2])),
              abs(l) < 1e-9 and abs(q) < 1e-9 and abs(g) < 1e-9,
              '%.1e m' % math.hypot(l, q))

print('\nDeviation in the start frame')
start = (1.0, 2.0, math.radians(90.0))
# 10 cm in the facing direction of the start = +y in the world frame when it faces north
l, q, g = in_start_frame(start, (1.0, 2.1, math.radians(90.0)))
check('long points in the facing direction of the start',
      abs(l - 0.1) < 1e-12 and abs(q) < 1e-12, '%.3f / %.3f' % (l, q))
l, q, g = in_start_frame(start, (0.9, 2.0, math.radians(90.0)))
check('lat points to the left', abs(q - 0.1) < 1e-12 and abs(l) < 1e-12,
      '%.3f / %.3f' % (l, q))
l, q, g = in_start_frame(start, (1.0, 2.0, math.radians(-175.0)))
check('the yaw deviation is computed the short way round',
      abs(math.degrees(g) - 95.0) < 1e-9, '%.1f deg' % math.degrees(g))

print('\nMeasure the driving direction instead of guessing')
from sensor_msgs.msg import LaserScan


def build_scan(left_m, right_m, n=360):
    """A scan that sees different distances left and right.

    Built through the real conversion in scan_to_points, so that the test does
    not lay down the convention a second time.
    """
    msg = LaserScan()
    msg.angle_min = -math.pi
    msg.angle_increment = 2.0 * math.pi / n
    msg.range_min, msg.range_max = 0.05, 12.0
    vals = []
    for i in range(n):
        a = msg.angle_min + i * msg.angle_increment
        # like scan_to_points: x = -r*cos(a), y = -r*sin(a)
        phi = math.atan2(-math.sin(a), -math.cos(a))
        vals.append(left_m if phi > 0.0 else right_m)
    msg.ranges = vals
    return msg


class DirectionStub:
    scan_cb = UnparkTest.scan_cb

    def __init__(self, state='DIRECTION'):
        self.state = state
        self.sector_deg = 20.0
        self.votes = []
        self.last_reason = None


# Wall right, field left -> CCW (field_map: CW has the inner block on the right)
f = DirectionStub()
for _ in range(5):
    f.scan_cb(build_scan(left_m=0.86, right_m=0.15))
check('field left is counted as CCW',
      f.votes == ['CCW'] * 5, str(f.votes))

f = DirectionStub()
for _ in range(5):
    f.scan_cb(build_scan(left_m=0.15, right_m=0.86))
check('field right is counted as CW', f.votes == ['CW'] * 5,
      str(f.votes))

# A contradiction resets -- do not let a narrow majority win.
f = DirectionStub()
for _ in range(3):
    f.scan_cb(build_scan(0.86, 0.15))
f.scan_cb(build_scan(0.15, 0.86))
check('a contradiction resets the votes', f.votes == ['CW'],
      str(f.votes))

# Undecided scans do not count at all.
f = DirectionStub()
f.scan_cb(build_scan(0.86, 0.15))
f.scan_cb(build_scan(0.50, 0.46))
check('uncertain scans reset', f.votes == [], f.last_reason)

# Outside the search nothing is counted.
f = DirectionStub(state='OUTBOUND')
f.scan_cb(build_scan(0.86, 0.15))
check('votes only during the search', f.votes == [])


class SequenceStub:
    _choose_sequence = UnparkTest._choose_sequence

    def __init__(self, table):
        self.table = table
        self.lines = []

    def get_parameter(self, name):
        return types.SimpleNamespace(value=self.table)

    def get_logger(self):
        add = lambda t, **kw: self.lines.append(t)
        return types.SimpleNamespace(info=add, warn=add, error=add)


TAB = [100.0, 6.0, -100.0, -4.0, 0.0, 9.0]
a, b = SequenceStub(TAB), SequenceStub(TAB)
a._choose_sequence('CW', 'Test')
b._choose_sequence('CCW', 'Test')
check('CW and CCW mirror the steering against each other',
      all(x * y < 0 for (x, _c1), (y, _c2) in zip(a.steps_out, b.steps_out)
          if abs(x) > 5.0),
      '%s vs %s' % ([round(l) for l, _c in a.steps_out],
                    [round(l) for l, _c in b.steps_out]))
check('the travels stay the same in both',
      [c for _l, c in a.steps_out] == [c for _l, c in b.steps_out])
check('the way back is built along with it',
      a.steps_back == reverse_steps(a.steps_out) and b.steps_back == reverse_steps(b.steps_out))
check('the direction is in the log',
      any('CW' in z for z in a.lines) and any('CCW' in z for z in b.lines))

print('\nPose output')
check('pose is printed readably',
      pose_text((0.1234, -0.5678, math.radians(12.34)))
      == 'x=+0.123 m  y=-0.568 m  heading=+12.3 deg',
      pose_text((0.1234, -0.5678, math.radians(12.34))))


class PauseStub:
    _pause_over = UnparkTest._pause_over
    _wait_for_key = UnparkTest._wait_for_key

    def __init__(self, with_key, tty):
        self.pause_on_key = with_key
        self.pause_s = 2.0
        self.proceed = False
        self.key_wait_active = False
        self.t0 = 100.0
        self._tty = tty
        self.lines = []

    def get_logger(self):
        add = lambda text, **kw: self.lines.append(text)
        return types.SimpleNamespace(info=add, warn=add, error=add)


print('\nPause at the turnaround')
real_stdin = sys.stdin

# With a terminal: it waits until the key came -- and not for the clock.
# readline() must BLOCK like a real terminal, otherwise the thread sets
# the flag in the very same moment and the test measures nothing.
key = threading.Event()
sys.stdin = types.SimpleNamespace(
    isatty=lambda: True,
    readline=lambda: (key.wait(5.0), '\n')[1])
f = PauseStub(with_key=True, tty=True)
check('without a key press it does not go on', not f._pause_over(100.0))
check('the prompt is in the log',
      any('ENTER' in z for z in f.lines))
check('not even after a long time', not f._pause_over(1e6))
key.set()
for _ in range(200):                      # give the thread time
    if f.proceed:
        break
    time.sleep(0.01)
check('after the key press it goes on', f._pause_over(100.0))
check('the thread really set the flag', f.proceed is True)

# Without a terminal it must not hang forever.
sys.stdin = types.SimpleNamespace(isatty=lambda: False)
f = PauseStub(with_key=True, tty=False)
check('without a terminal it falls back to the clock',
      not f._pause_over(100.0) and f._pause_over(102.5))
check('and says that it does so',
      any('no terminal' in z for z in f.lines))

# Switched off it also just waits out the time.
sys.stdin = types.SimpleNamespace(isatty=lambda: True)
f = PauseStub(with_key=False, tty=True)
check('pause_on_key=False uses the clock again',
      not f._pause_over(101.0) and f._pause_over(102.1)
      and not any('ENTER' in z for z in f.lines))

sys.stdin = real_stdin

print('\nall tests passed')
