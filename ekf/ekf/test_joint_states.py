#!/usr/bin/env python3
"""Der Encoder-Callback muss JointState-Nachrichten OHNE Geschwindigkeit
ueberstehen.

    python3 -m ekf.test_joint_states

Die Bruecke schickt auf /esp_serial_bridge/joint_states drei Sorten
Nachrichten. Nur CMD_TELEMETRY traegt eine Geschwindigkeit; MOVE_DONE und
PROGRESS_RSP kennen nur die Position und lassen velocity absichtlich leer
(siehe _publish_joint in esp_serial_bridge.py). Genau die kommen waehrend
einer Positionsfahrt -- also bei jedem Ausparken.

Ohne die Pruefung starb ekf_node an der ERSTEN solchen Nachricht mit
"IndexError: array index out of range", und mit ihm die ganze
Zustandsschaetzung: /ekf/odom verstummte, die Bruecke schaltete den Motor ab
und der Regler kam nicht ueber "if self.pose is None" hinaus. Am Roboter
gefunden, nicht hier -- deshalb steht der Test jetzt da.
"""
from sensor_msgs.msg import JointState

from ekf.ekf_node import EKFNode


def pruefe(name, bedingung, zusatz=''):
    if not bedingung:
        raise AssertionError('FEHLGESCHLAGEN: %s %s' % (name, zusatz))
    print('  ok  %s%s' % (name, ('  ' + zusatz) if zusatz else ''))


class Attrappe:
    enc_cb = EKFNode.enc_cb

    def __init__(self):
        self.r_eff = 0.0150
        self.gepusht = []
        self.gedraint = []

    def _push(self, t, kind, z):
        self.gepusht.append((t, kind, z))

    def _drain(self, t):
        self.gedraint.append(t)


def nachricht(sekunden, velocity=None, position=0.0):
    msg = JointState()
    msg.header.stamp.sec = int(sekunden)
    msg.header.stamp.nanosec = int((sekunden % 1) * 1e9)
    msg.name = ['drive_axle']
    msg.position = [position]
    if velocity is not None:
        msg.velocity = [velocity]
    return msg


print('Encoder-Callback')

f = Attrappe()
f.enc_cb(nachricht(10.0, velocity=2.0))
pruefe('Telemetrie mit Geschwindigkeit wird verarbeitet',
       len(f.gepusht) == 1 and f.gepusht[0][1] == 'enc')
pruefe('rad/s werden in m/s umgerechnet',
       abs(f.gepusht[0][2] - 2.0 * 0.0150) < 1e-12,
       '%.4f m/s' % f.gepusht[0][2])

f = Attrappe()
f.enc_cb(nachricht(11.0))            # MOVE_DONE: nur Position, kein velocity
pruefe('Nachricht ohne Geschwindigkeit stuerzt nicht ab', True)
pruefe('... und wird still verworfen', f.gepusht == [] and f.gedraint == [])

f = Attrappe()
for i, v in enumerate([1.0, None, 2.0, None, None, 3.0]):
    f.enc_cb(nachricht(20.0 + i, velocity=v))
pruefe('gemischter Strom: nur die echten Messungen kommen an',
       [k for _t, k, _z in f.gepusht] == ['enc'] * 3,
       '%d von 6' % len(f.gepusht))
pruefe('die Zeitstempel bleiben die der echten Messungen',
       [round(t) for t, _k, _z in f.gepusht] == [20, 22, 25])

print('\nalle Tests bestanden')
