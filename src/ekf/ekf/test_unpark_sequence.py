#!/usr/bin/env python3
"""Sequence test of the unpark sequence -- without the ROS core, without the robot.

    python3 -m ekf.test_unpark_sequence

Borrows the methods of Round1Controller and drives them with stubs for
clock, logger and publishers. What is checked is the SEQUENCE: first steer, then drive,
wait for the ack, move by move -- and what happens when something goes
wrong. That is exactly what you do not want to find out on the robot.
"""
import math
import types

from ekf.round1_controller_node import Round1Controller
from ekf import unpark as A


def check(name, cond, extra=''):
    if not cond:
        raise AssertionError('FAILED: %s %s' % (name, extra))
    print('  ok  %s%s' % (name, ('  ' + extra) if extra else ''))


class Collector:
    """Publisher stub. Also writes into a shared trace,
    so that the ORDER across all topics can be checked."""

    def __init__(self, name='?', trace=None, subscribers=1):
        self.name = name
        self.values = []
        self.trace = trace if trace is not None else []
        self.subscribers = subscribers

    def publish(self, msg):
        self.values.append(msg)
        val = msg.data if not isinstance(msg.data, (list, tuple)) else list(msg.data)
        self.trace.append((self.name, val))

    def get_subscription_count(self):
        return self.subscribers


class Logbook:
    def __init__(self):
        self.lines = []

    def _add(self, level):
        return lambda text, **kw: self.lines.append((level, text))

    def __getattr__(self, name):
        return self._add(name)

    def text(self):
        return '\n'.join(t for _s, t in self.lines)

    def at_level(self, level):
        return [t for s, t in self.lines if s == level]


class Stub:
    _unpark_step = Round1Controller._unpark_step
    _unpark_plan = Round1Controller._unpark_plan
    _unpark_done = Round1Controller._unpark_done
    _unpark_pid = Round1Controller._unpark_pid
    _unpark_abort = Round1Controller._unpark_abort
    unpark_move_done_cb = Round1Controller.unpark_move_done_cb
    _unpark_scan_hold = Round1Controller._unpark_scan_hold
    _unpark_handover = Round1Controller._unpark_handover
    _unpark_adopt_direction = \
        Round1Controller._unpark_adopt_direction

    def __init__(self, **kw):
        self.t = 100.0
        self.x = self.y = self.th = 0.0     # stub of the pose
        self.state = 'UNPARK_BUTTON'
        self.require_button = False
        self.button_pressed = False
        self.unpark_only = False
        self.unpark_invert_direction = False
        self.unpark_steps = list(A.STEPS_DEFAULT)
        self.unpark_steps_cw = []
        self.unpark_steps_ccw = []
        self.unpark_pid = [4.0, 140.0, 8.0, 90.0]   # like the real default
        self.unpark_pid_after = [4.0, 1023.0]
        self.unpark_scans = 5
        self.unpark_sector_deg = 20.0
        self.unpark_direction_timeout = 8.0
        self.unpark_steer_wait_s = 0.6
        self.unpark_move_timeout = 15.0
        self.unpark_travel_tol_cm = 1.0
        self.unpark_hold_s = 2.0
        self.race_direction = None
        self.unpark_sets_direction = True
        self.unpark_votes = ['CW'] * 5
        self.unpark_last_reason = 'left 0.14 m, right 0.87 m'
        self.unpark_steps_run = None
        self.unpark_direction = None
        self.unpark_index = 0
        self.unpark_phase = 'steer'
        self.unpark_steer_sent = False
        self.unpark_t0 = 0.0
        self.unpark_sent_t = None
        self.unpark_move_done = None
        self.unpark_theta0 = 0.0
        self.unpark_pose0 = None
        self.trace = []
        self.pub_steer = Collector('steer', self.trace)
        self.pub_move = Collector('move', self.trace)
        self.pub_pid = Collector('pid', self.trace)
        self.pub_motor = Collector('motor', self.trace)
        self.pub_park_dir = Collector('park_dir', self.trace)
        self.stops = 0
        self._log = Logbook()
        self.__dict__.update(kw)

    def now_s(self):
        return self.t

    def get_logger(self):
        return self._log

    def publish_stop(self):
        self.stops += 1

    # --- Helpers for the tests ---
    def tick(self, n=1, dt=0.1):
        for _ in range(n):
            self._unpark_step(self.x, self.y, self.th)
            self.t += dt

    def ack(self, status=0, pos_decideg=0):
        self.unpark_move_done_cb(
            types.SimpleNamespace(data=[1, status, pos_decideg]))

    def run_through(self, status=0, travel_ok=True, limit=4000):
        """Drive the sequence. travel_ok=True lets the stub pose follow the
        plan -- then a timeout of the ESP is harmless."""
        for _ in range(limit):
            before = len(self.pub_move.values)
            self._unpark_step(self.x, self.y, self.th)
            self.t += 0.1
            if len(self.pub_move.values) > before:
                if travel_ok:
                    steer, cm = self.unpark_steps_run[self.unpark_index]
                    ex, ey, eth = A.trajectory((0.0, 0.0, 0.0), [(steer, cm)])[-1][0]
                    c, si = math.cos(self.th), math.sin(self.th)
                    self.x += c * ex - si * ey
                    self.y += si * ex + c * ey
                    self.th += eth
                self.t += 0.5
                self.ack(status)
            if self.state == 'UNPARK_SCAN':
                self.t += self.unpark_hold_s + 0.1
            if self.state in ('WAIT_INPUTS', 'DONE'):
                return
        raise AssertionError('sequence hangs -- no end after %d ticks' % limit)


print('Button')
f = Stub(require_button=True)
f.tick(5)
check('without the button it stays put',
      f.state == 'UNPARK_BUTTON' and f.stops == 5 and not f.pub_move.values)
f.button_pressed = True
f.tick(1)
check('with the button it goes on to the direction search', f.state == 'UNPARK_DIRECTION')

print('\nDirection search')
f = Stub(unpark_votes=[])
f.tick(1)                                  # UNPARK_BUTTON -> UNPARK_DIRECTION
f.tick(10)
check('without votes nothing is driven',
      f.state == 'UNPARK_DIRECTION' and not f.pub_move.values)
f.t += 20.0
f.tick(1)
check('timeout aborts', f.state == 'DONE')
check('abort actively stops the motor',
      len(f.pub_motor.values) == 1 and f.pub_motor.values[0].data == 0)
check('abort resets the control parameters',
      [list(m.data) for m in f.pub_pid.values] == [[4.0, 1023.0]])
check('abort is logged as an error', f._log.at_level('error'))

# As long as the bridge has not subscribed to the topics, nothing may go out:
# DDS discovery swallows the first messages on a fresh connection,
# and those would be maxduty and the first steering command of all things.
f = Stub()
f.pub_steer.subscribers = 0
f.tick(4)
check('without a subscriber nothing is planned',
      f.state == 'UNPARK_DIRECTION' and not f.pub_pid.values
      and not f.pub_steer.values)
f.pub_steer.subscribers = 1
f.tick(1)
check('as soon as the bridge is there, it goes on',
      f.state == 'UNPARK_DRIVE' and len(f.pub_pid.values) == 2)

f = Stub()
f.pub_move.subscribers = 0
f.tick(2)
f.t += f.unpark_direction_timeout + 1.0
f.tick(1)
check('if the bridge stays away, it aborts', f.state == 'DONE')
check('... and the message names the missing topic',
      'move' in f._log.text() and 'esp_serial_bridge' in f._log.text())

print('\nPlanning')
f = Stub()
f.tick(2)
check('CW mirrors the table to the right',
      f.unpark_steps_run[0][0] < 0 and A.STEPS_DEFAULT[0] > 0,
      'first move %+.0f %%' % f.unpark_steps_run[0][0])
check('travels stay unchanged',
      [cm for _l, cm in f.unpark_steps_run] == list(A.STEPS_DEFAULT[1::2]))
check('control parameters are set before the sequence',
      [list(m.data) for m in f.pub_pid.values] == [[4.0, 140.0], [8.0, 90.0]])
check('dry run is in the log', 'dry run' in f._log.text())

f2 = Stub(unpark_votes=['CCW'] * 5)
f2.tick(2)
check('CCW does not mirror', f2.unpark_steps_run[0][0] > 0,
      'first move %+.0f %%' % f2.unpark_steps_run[0][0])
f3 = Stub(unpark_invert_direction=True)
f3.tick(2)
check('invert switch flips the side',
      f3.unpark_steps_run[0][0] * f.unpark_steps_run[0][0] < 0)

# An own sequence for the detected direction (here CW) beats the shared one.
OWN = [100.0, 3.0, -100.0, -2.0]
f = Stub(unpark_steps_cw=OWN)
f.tick(2)
check('CW takes its own sequence',
      [cm for _l, cm in f.unpark_steps_run] == [3.0, -2.0],
      '%d moves' % len(f.unpark_steps_run))
check('and says so in the log', 'own sequence for CW' in f._log.text())

f = Stub(unpark_steps_ccw=OWN)     # filled, but CW is detected
f.tick(2)
check('the sequence of the OTHER direction is ignored',
      [cm for _l, cm in f.unpark_steps_run]
      == list(A.STEPS_DEFAULT[1::2]))
check('and the shared one is named', 'shared sequence' in f._log.text())

print('\nA single move')
f = Stub()
f.tick(2)                                   # planned, now UNPARK_DRIVE
f.stops = 0            # the halt from the direction search does not count here
steer_target, cm_target = f.unpark_steps_run[0]
f.tick(1)
check('first it steers, does not drive yet',
      len(f.pub_steer.values) == 1 and not f.pub_move.values)
check('steering value is right',
      abs(f.pub_steer.values[0].data - steer_target) < 1e-6)
f.tick(2)
check('during the wait time nothing is driven', not f.pub_move.values)
f.t += f.unpark_steer_wait_s
f.tick(1)
check('after the wait time the move goes out', len(f.pub_move.values) == 1)
check('travel in encoder degrees',
      abs(f.pub_move.values[0].data - A.cm_to_deg(cm_target)) < 1e-3,
      '%.0f deg for %.1f cm' % (f.pub_move.values[0].data, cm_target))
# During a position move NO /cmd_vel may go out: the bridge
# would then set the steering again.
check('no /cmd_vel during the move', f.stops == 0)
f.tick(5)
check('without an ack it does not go on',
      f.unpark_index == 0 and len(f.pub_move.values) == 1)

print('\nAcks')
f_old = Stub()
f_old.tick(2)
f_old.unpark_move_done = (f_old.t - 50.0, 1, 0, 0.0)     # ack from BEFORE
f_old.t += f_old.unpark_steer_wait_s
f_old.tick(3)
check('old ack does not count', f_old.unpark_index == 0)

f = Stub()
f.tick(2)
f.stops = 0
f.run_through()
# During the moves no /cmd_vel may go out; afterwards it may: once
# at completion and then in the scan hold.
check('no halt during the moves, afterwards yes', f.stops >= 2,
      '%d halts' % f.stops)
check('complete sequence runs through',
      len(f.pub_move.values) == len(f.unpark_steps_run),
      '%d moves sent' % len(f.pub_move.values))
# The steering command is repeated during the wait time, so do not count,
# but check the order: before EVERY move the last thing sent must be the
# steering value of exactly this move.
last_steered, seen = None, []
for name, val in f.trace:
    if name == 'steer':
        last_steered = val
    elif name == 'move':
        seen.append(last_steered)
check('before every move there is the right steering value',
      all(a is not None and abs(a - b) < 1e-6
          for a, b in zip(seen, [l for l, _cm in f.unpark_steps_run]))
      and len(seen) == len(f.unpark_steps_run),
      '%s' % ['%+.0f' % g for g in seen])
check('the control parameters come BEFORE the first move',
      [n for n, _w in f.trace].index('pid')
      < [n for n, _w in f.trace].index('move'))
check('then on to the race', f.state == 'WAIT_INPUTS')
check('button counts as pressed', f.button_pressed is True)
check('control parameters reset at the end',
      list(f.pub_pid.values[-1].data) == [4.0, 1023.0])

print('\nScan hold after unparking')
f = Stub()
f.tick(2)
for _ in range(4000):
    before = len(f.pub_move.values)
    f._unpark_step(f.x, f.y, f.th)
    f.t += 0.1
    if len(f.pub_move.values) > before:
        f.t += 0.5
        f.ack(0)
    if f.state == 'UNPARK_SCAN':
        break
check('after the last move it holds', f.state == 'UNPARK_SCAN')
check('the direction is out already BEFORE the hold',
      'CW' in [w for n, w in f.trace if n == 'park_dir'])
halts_before = f.stops
f.tick(5)
check('during the hold it stays put',
      f.state == 'UNPARK_SCAN' and f.stops == halts_before + 5)
check('and does not drive on', len(f.pub_move.values) == len(f.unpark_steps_run))
f.t += f.unpark_hold_s
f.tick(1)
check('after the time it goes on to the race', f.state == 'WAIT_INPUTS')

f = Stub(unpark_hold_s=0.0)
f.run_through()
check('hold_s = 0 switches the pause off', f.state == 'WAIT_INPUTS')

# The latch of the perception can come from the time IN the bay and then
# carry the wrong direction. That must be noticed, not silently
# overruled.
f = Stub(race_direction='CCW')      # unparking measures CW
f.run_through()
check('the parking enforces the direction', f.race_direction == 'CW')
# Not latched, so it is repeated during the hold -- what is checked is
# the content, not the count.
sent = [w for n, w in f.trace if n == 'park_dir']
check('... and sends it to the scan_processor',
      sent and set(sent) == {'CW'}, '%dx' % len(sent))
check('... and says that it contradicts the latch',
      any('/race_direction reports' in t for t in f._log.at_level('warn')))
check('... and the run goes on', f.state == 'WAIT_INPUTS')

f = Stub(race_direction='CW')
f.run_through()
check('matching direction produces no warning',
      not any('reports' in t for t in f._log.at_level('warn')
              if 'direction' in t or 'race_direction' in t))
sent = [w for n, w in f.trace if n == 'park_dir']
check('... but is published anyway',
      sent and set(sent) == {'CW'}, '%dx' % len(sent))

# Switch off: the corner geometry keeps the say.
f = Stub(race_direction='CCW', unpark_sets_direction=False)
f.run_through()
check('switch off -> /race_direction stays', f.race_direction == 'CCW')
check('... nothing is published',
      not [w for n, w in f.trace if n == 'park_dir'])
check('... the conflict is reported anyway',
      any('CONFLICT' in t for t in f._log.at_level('error')))

# Without any latch the parking carries the direction alone.
f = Stub(race_direction=None)
f.run_through()
check('without a latch the parking applies', f.race_direction == 'CW')

f = Stub(unpark_only=True)
f.run_through()
check('unpark_only stops', f.state == 'DONE')
check('unpark_only still drives the whole sequence',
      len(f.pub_move.values) == len(f.unpark_steps_run))

# Status 2 means: something else has taken over the motor. Always fatal.
f = Stub()
f.run_through(status=2)
check('replaced move aborts', f.state == 'DONE')
check('abort after the FIRST bad move', len(f.pub_move.values) == 1)

# Status 1 only means "not settled". What matters is the travel.
f = Stub()
f.run_through(status=1, travel_ok=True)
check('timeout with the right travel carries on',
      f.state == 'WAIT_INPUTS' and len(f.pub_move.values) == len(f.unpark_steps_run),
      '%d moves sent' % len(f.pub_move.values))
check('... but warns on every move',
      len([t for t in f._log.at_level('warn') if 'timeout' in t])
      == len(f.unpark_steps_run))
check('... and the warning names the way out',
      any('minduty' in t for t in f._log.at_level('warn')))

f = Stub()
f.run_through(status=1, travel_ok=False)
check('timeout WITHOUT travel aborts', f.state == 'DONE')
check('... after the first move', len(f.pub_move.values) == 1)

print('\nHangs')
f = Stub()
f.tick(2)                                    # planned
f.tick(1)                                    # steering command out, clock runs from here
f.t += f.unpark_steer_wait_s
f.tick(1)                                    # drive command out
assert f.unpark_phase == 'drive', 'test setup is wrong'
f.t += f.unpark_move_timeout + 1.0
f.tick(1)
check('missing ack aborts', f.state == 'DONE')
check('error message names the bridge',
      'esp_serial_bridge' in f._log.text())

print('\nBad inputs')
f = Stub(unpark_steps=[100.0, 5.0, -100.0])      # odd
f.tick(2)
check('odd step list aborts cleanly', f.state == 'DONE')
f = Stub(unpark_steps=[])
f.tick(2)
check('empty step list aborts cleanly', f.state == 'DONE')
f = Stub(unpark_pid=[4.0])                          # odd
f.tick(2)
check('odd PID list is only reported, not driven',
      f.state == 'UNPARK_DRIVE' and f._log.at_level('warn'))

print('\nall tests passed')
