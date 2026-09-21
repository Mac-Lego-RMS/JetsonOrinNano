#!/usr/bin/env python3
"""Verankerung der Karte: Feldpose des Roboters -> Feldpose des Odom-Ursprungs.

    python3 -m ekf.test_verankerung

generate_map() erwartet die Pose des Punktes, an dem die Odometrie genullt
wurde, nicht die des Roboters. Bei einem normalen Start ist das dasselbe -- der
Roboter steht still, bis erkannt ist. Nach dem Ausparken liegen 50 cm
dazwischen, und ohne die Verrechnung ist die ganze Karte um die Ausparkstrecke
versetzt. Genau das hat einen Lauf in die Wand gefahren (1,67 m
Querabweichung im ersten Regelschritt).
"""
import math

import numpy as np

from ekf.scan_processor_node import ScanProcessor
from ekf.field_map import generate_map, START_POSES_CW


def pruefe(name, bedingung, zusatz=''):
    if not bedingung:
        raise AssertionError('FEHLGESCHLAGEN: %s %s' % (name, zusatz))
    print('  ok  %s%s' % (name, ('  ' + zusatz) if zusatz else ''))


class Attrappe:
    _verankert = ScanProcessor._verankert

    def __init__(self, commit_pose=(0.0, 0.0, 0.0)):
        self.commit_pose = commit_pose


def roboter_im_feld(anker, odom_pose):
    """Wo liegt der Roboter im Feld, wenn der Ursprung bei ``anker`` liegt?"""
    xa, ya, tha = anker
    xo, yo, tho = odom_pose
    c, s = math.cos(tha), math.sin(tha)
    return (xa + c * xo - s * yo, ya + s * xo + c * yo, tha + tho)


print('Verankerung')

f = Attrappe()
feld = START_POSES_CW['pos1']
pruefe('im Ursprung aendert sich nichts',
       all(abs(a - b) < 1e-12 for a, b in zip(f._verankert(feld), feld)),
       '%s' % (f._verankert(feld),))

# Der Roboter hat sich seit dem Nullen bewegt: der Anker muss so liegen, dass
# er JETZT auf der erkannten Feldpose steht.
for odom in ((0.25, -0.33, math.radians(-34.6)),
             (0.5, 0.0, 0.0),
             (-0.2, 0.7, math.radians(120.0)),
             (1.3, -0.4, math.radians(-175.0))):
    f = Attrappe(odom)
    anker = f._verankert(feld)
    zurueck = roboter_im_feld(anker, odom)
    passt = (abs(zurueck[0] - feld[0]) < 1e-9 and abs(zurueck[1] - feld[1]) < 1e-9
             and abs(math.atan2(math.sin(zurueck[2] - feld[2]),
                                math.cos(zurueck[2] - feld[2]))) < 1e-9)
    pruefe('Roboter landet auf der erkannten Feldpose (odom %+.2f,%+.2f,%+.0f)'
           % (odom[0], odom[1], math.degrees(odom[2])), passt,
           'Anker (%+.3f,%+.3f,%+.0f)' % (anker[0], anker[1], math.degrees(anker[2])))

print('\nWirkung auf die Karte')
# Ohne Verankerung waere die Karte um die gefahrene Strecke versetzt.
odom = (0.25, -0.33, math.radians(-34.6))
f = Attrappe(odom)
ohne = generate_map(feld)
mit = generate_map(f._verankert(feld))
versatz = max(abs(a['d'] - b['d']) for a, b in zip(ohne, mit))
pruefe('ohne Verankerung liegt die Karte deutlich daneben', versatz > 0.20,
       'groesster Abstandsfehler %.2f m' % versatz)

f0 = Attrappe()
gleich = generate_map(f0._verankert(feld))
pruefe('im Ursprung ist die Karte unveraendert',
       all(abs(a['d'] - b['d']) < 1e-12 for a, b in zip(ohne, gleich)))

print('\nalle Tests bestanden')
