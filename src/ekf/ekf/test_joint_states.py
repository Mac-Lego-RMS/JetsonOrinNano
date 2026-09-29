#!/usr/bin/env python3
"""The encoder callback must survive JointState messages WITHOUT a velocity.

    python3 -m ekf.test_joint_states

The bridge sends three kinds of messages on /esp_serial_bridge/joint_states.
Only CMD_TELEMETRY carries a velocity; MOVE_DONE and
PROGRESS_RSP only know the position and leave velocity empty on purpose
(see _publish_joint in esp_serial_bridge.py). Exactly those come during
a position move -- i.e. on every unpark.

Without the check ekf_node died on the FIRST such message with
"IndexError: array index out of range", and with it the whole
state estimation: /ekf/odom went silent, the bridge switched the motor off
and the controller never got past "if self.pose is None". Found on the
robot, not here -- that is why the test is here now.
"""
from sensor_msgs.msg import JointState

from ekf.ekf_node import EKFNode


def check(name, cond, extra=''):
    if not cond:
        raise AssertionError('FAILED: %s %s' % (name, extra))
    print('  ok  %s%s' % (name, ('  ' + extra) if extra else ''))


class Stub:
    enc_cb = EKFNode.enc_cb

    def __init__(self):
        self.r_eff = 0.0150
        self.pushed = []
        self.drained = []

    def _push(self, t, kind, z):
        self.pushed.append((t, kind, z))

    def _drain(self, t):
        self.drained.append(t)


def message(seconds, velocity=None, position=0.0):
    msg = JointState()
    msg.header.stamp.sec = int(seconds)
    msg.header.stamp.nanosec = int((seconds % 1) * 1e9)
    msg.name = ['drive_axle']
    msg.position = [position]
    if velocity is not None:
        msg.velocity = [velocity]
    return msg


print('Encoder callback')

f = Stub()
f.enc_cb(message(10.0, velocity=2.0))
check('telemetry with velocity is processed',
      len(f.pushed) == 1 and f.pushed[0][1] == 'enc')
check('rad/s are converted to m/s',
      abs(f.pushed[0][2] - 2.0 * 0.0150) < 1e-12,
      '%.4f m/s' % f.pushed[0][2])

f = Stub()
f.enc_cb(message(11.0))            # MOVE_DONE: only position, no velocity
check('message without velocity does not crash', True)
check('... and is silently dropped', f.pushed == [] and f.drained == [])

f = Stub()
for i, v in enumerate([1.0, None, 2.0, None, None, 3.0]):
    f.enc_cb(message(20.0 + i, velocity=v))
check('mixed stream: only the real measurements arrive',
      [k for _t, k, _z in f.pushed] == ['enc'] * 3,
      '%d of 6' % len(f.pushed))
check('the timestamps stay those of the real measurements',
      [round(t) for t, _k, _z in f.pushed] == [20, 22, 25])

print('\nall tests passed')
