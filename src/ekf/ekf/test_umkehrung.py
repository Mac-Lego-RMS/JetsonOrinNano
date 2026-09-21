#!/usr/bin/env python3
"""Der Rueckweg muss den Hinweg kinematisch genau aufheben.

    python3 -m ekf.test_umkehrung

Das ist die Grundlage des Ausparktests (ekf/ausparken_test_node.py): gilt es
nicht schon auf dem Papier, misst der Test am Roboter nichts Brauchbares.
"""
import math

from ekf import ausparken as A
from ekf.ausparken_test_node import umkehren, im_startrahmen, wrap


def pruefe(name, bedingung, zusatz=''):
    if not bedingung:
        raise AssertionError('FEHLGESCHLAGEN: %s %s' % (name, zusatz))
    print('  ok  %s%s' % (name, ('  ' + zusatz) if zusatz else ''))


print('Umkehrung der Schrittfolge')

hin = A.spiegeln(A.schritte_aus_flach(A.SCHRITTE_STANDARD), True)
zurueck = umkehren(hin)

pruefe('gleich viele Zuege', len(zurueck) == len(hin))
pruefe('Reihenfolge ist gedreht',
       [l for l, _c in zurueck] == [l for l, _c in reversed(hin)])
pruefe('jede Strecke ist negiert',
       [c for _l, c in zurueck] == [-c for _l, c in reversed(hin)])
pruefe('die Lenkung bleibt je Zug dieselbe',
       [l for l, _c in zurueck] == [l for l, _c in reversed(hin)])
pruefe('zweimal umkehren ergibt das Original', umkehren(zurueck) == hin)

print('\nAufhebung in der Kinematik')
start = (0.0, 0.0, 0.0)
nach_hin = A.bahn(start, hin)[-1][0]
nach_zurueck = A.bahn(nach_hin, zurueck)[-1][0]
laengs, quer, gier = im_startrahmen(start, nach_zurueck)
pruefe('der Rueckweg landet wieder im Start',
       abs(laengs) < 1e-9 and abs(quer) < 1e-9 and abs(gier) < 1e-9,
       '%.1e m / %.1e m / %.1e rad' % (laengs, quer, gier))

# Auch aus einer schraegen Startlage, und fuer die gespiegelte Folge.
for start in ((0.4, -0.2, math.radians(37.0)),
              (-1.1, 0.8, math.radians(-160.0))):
    for tab in (hin, A.spiegeln(A.schritte_aus_flach(A.SCHRITTE_STANDARD), False)):
        ende = A.bahn(A.bahn(start, tab)[-1][0], umkehren(tab))[-1][0]
        l, q, g = im_startrahmen(start, ende)
        pruefe('auch aus (%.1f, %.1f, %+.0f grad)'
               % (start[0], start[1], math.degrees(start[2])),
               abs(l) < 1e-9 and abs(q) < 1e-9 and abs(g) < 1e-9,
               '%.1e m' % math.hypot(l, q))

print('\nAbweichung im Startrahmen')
start = (1.0, 2.0, math.radians(90.0))
# 10 cm in Blickrichtung des Starts = +y im Weltrahmen, wenn er nach Norden schaut
l, q, g = im_startrahmen(start, (1.0, 2.1, math.radians(90.0)))
pruefe('laengs zeigt in Blickrichtung des Starts',
       abs(l - 0.1) < 1e-12 and abs(q) < 1e-12, '%.3f / %.3f' % (l, q))
l, q, g = im_startrahmen(start, (0.9, 2.0, math.radians(90.0)))
pruefe('quer zeigt nach links', abs(q - 0.1) < 1e-12 and abs(l) < 1e-12,
       '%.3f / %.3f' % (l, q))
l, q, g = im_startrahmen(start, (1.0, 2.0, math.radians(-175.0)))
pruefe('die Gierabweichung wird kurz herum gerechnet',
       abs(math.degrees(g) - 95.0) < 1e-9, '%.1f grad' % math.degrees(g))

print('\nalle Tests bestanden')
