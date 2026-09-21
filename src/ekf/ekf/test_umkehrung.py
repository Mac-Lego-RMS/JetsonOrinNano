#!/usr/bin/env python3
"""Der Rueckweg muss den Hinweg kinematisch genau aufheben.

    python3 -m ekf.test_umkehrung

Das ist die Grundlage des Ausparktests (ekf/ausparken_test_node.py): gilt es
nicht schon auf dem Papier, misst der Test am Roboter nichts Brauchbares.
"""
import math
import sys
import threading
import time
import types

from ekf import ausparken as A
from ekf.ausparken_test_node import (AusparkTest, im_startrahmen, pose_text,
                                     umkehren, wrap)


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

print('\nFahrtrichtung messen statt raten')
from sensor_msgs.msg import LaserScan


def scan_bauen(links_m, rechts_m, n=360):
    """Ein Scan, der links und rechts unterschiedlich weit sieht.

    Gebaut ueber die echte Umrechnung in scan_to_points, damit der Test die
    Konvention nicht ein zweites Mal festschreibt.
    """
    msg = LaserScan()
    msg.angle_min = -math.pi
    msg.angle_increment = 2.0 * math.pi / n
    msg.range_min, msg.range_max = 0.05, 12.0
    werte = []
    for i in range(n):
        a = msg.angle_min + i * msg.angle_increment
        # wie scan_to_points: x = -r*cos(a), y = -r*sin(a)
        phi = math.atan2(-math.sin(a), -math.cos(a))
        werte.append(links_m if phi > 0.0 else rechts_m)
    msg.ranges = werte
    return msg


class RichtungAttrappe:
    scan_cb = AusparkTest.scan_cb

    def __init__(self, zustand='RICHTUNG'):
        self.zustand = zustand
        self.sektor_grad = 20.0
        self.stimmen = []
        self.letzter_grund = None


# Wand rechts, Feld links -> CCW (field_map: CW hat den Innenblock rechts)
f = RichtungAttrappe()
for _ in range(5):
    f.scan_cb(scan_bauen(links_m=0.86, rechts_m=0.15))
pruefe('Feld links wird als CCW gezaehlt',
       f.stimmen == ['CCW'] * 5, str(f.stimmen))

f = RichtungAttrappe()
for _ in range(5):
    f.scan_cb(scan_bauen(links_m=0.15, rechts_m=0.86))
pruefe('Feld rechts wird als CW gezaehlt', f.stimmen == ['CW'] * 5,
       str(f.stimmen))

# Ein Widerspruch setzt zurueck -- keine knappe Mehrheit gewinnen lassen.
f = RichtungAttrappe()
for _ in range(3):
    f.scan_cb(scan_bauen(0.86, 0.15))
f.scan_cb(scan_bauen(0.15, 0.86))
pruefe('ein Widerspruch setzt die Stimmen zurueck', f.stimmen == ['CW'],
       str(f.stimmen))

# Unentschiedene Scans zaehlen gar nicht.
f = RichtungAttrappe()
f.scan_cb(scan_bauen(0.86, 0.15))
f.scan_cb(scan_bauen(0.50, 0.46))
pruefe('unsichere Scans setzen zurueck', f.stimmen == [], f.letzter_grund)

# Ausserhalb der Suche wird nicht gezaehlt.
f = RichtungAttrappe(zustand='HIN')
f.scan_cb(scan_bauen(0.86, 0.15))
pruefe('nur waehrend der Suche wird gestimmt', f.stimmen == [])


class FolgeAttrappe:
    _folge_festlegen = AusparkTest._folge_festlegen

    def __init__(self, tabelle):
        self.tabelle = tabelle
        self.zeilen = []

    def get_parameter(self, name):
        return types.SimpleNamespace(value=self.tabelle)

    def get_logger(self):
        an = lambda t, **kw: self.zeilen.append(t)
        return types.SimpleNamespace(info=an, warn=an, error=an)


TAB = [100.0, 6.0, -100.0, -4.0, 0.0, 9.0]
a, b = FolgeAttrappe(TAB), FolgeAttrappe(TAB)
a._folge_festlegen('CW', 'Test')
b._folge_festlegen('CCW', 'Test')
pruefe('CW und CCW spiegeln die Lenkung gegeneinander',
       all(x * y < 0 for (x, _c1), (y, _c2) in zip(a.hin, b.hin)
           if abs(x) > 5.0),
       '%s vs %s' % ([round(l) for l, _c in a.hin],
                     [round(l) for l, _c in b.hin]))
pruefe('die Strecken bleiben in beiden gleich',
       [c for _l, c in a.hin] == [c for _l, c in b.hin])
pruefe('der Rueckweg wird gleich mitgebaut',
       a.zurueck == umkehren(a.hin) and b.zurueck == umkehren(b.hin))
pruefe('die Richtung steht im Log',
       any('CW' in z for z in a.zeilen) and any('CCW' in z for z in b.zeilen))

print('\nPose-Ausgabe')
pruefe('Pose wird lesbar ausgegeben',
       pose_text((0.1234, -0.5678, math.radians(12.34)))
       == 'x=+0.123 m  y=-0.568 m  Kurs=+12.3 grad',
       pose_text((0.1234, -0.5678, math.radians(12.34))))


class PauseAttrappe:
    _pause_vorbei = AusparkTest._pause_vorbei
    _auf_taste_warten = AusparkTest._auf_taste_warten

    def __init__(self, mit_taste, tty):
        self.pause_mit_taste = mit_taste
        self.pause_s = 2.0
        self.weiter = False
        self.taste_laeuft = False
        self.t0 = 100.0
        self._tty = tty
        self.zeilen = []

    def get_logger(self):
        an = lambda text, **kw: self.zeilen.append(text)
        return types.SimpleNamespace(info=an, warn=an, error=an)


print('\nPause an der Wende')
echtes_stdin = sys.stdin

# Mit Terminal: es wartet, bis die Taste kam -- und nicht auf die Uhr.
# readline() muss BLOCKIEREN wie ein echtes Terminal, sonst setzt der Faden
# das Flag schon im selben Augenblick und der Test misst nichts.
taste = threading.Event()
sys.stdin = types.SimpleNamespace(
    isatty=lambda: True,
    readline=lambda: (taste.wait(5.0), '\n')[1])
f = PauseAttrappe(mit_taste=True, tty=True)
pruefe('ohne Tastendruck geht es nicht weiter', not f._pause_vorbei(100.0))
pruefe('die Aufforderung steht im Log',
       any('ENTER' in z for z in f.zeilen))
pruefe('auch nach langer Zeit nicht', not f._pause_vorbei(1e6))
taste.set()
for _ in range(200):                      # dem Faden Zeit lassen
    if f.weiter:
        break
    time.sleep(0.01)
pruefe('nach dem Tastendruck geht es weiter', f._pause_vorbei(100.0))
pruefe('der Faden hat das Flag wirklich gesetzt', f.weiter is True)

# Ohne Terminal darf er nicht ewig haengen.
sys.stdin = types.SimpleNamespace(isatty=lambda: False)
f = PauseAttrappe(mit_taste=True, tty=False)
pruefe('ohne Terminal faellt er auf die Uhr zurueck',
       not f._pause_vorbei(100.0) and f._pause_vorbei(102.5))
pruefe('und sagt, dass er das tut',
       any('kein Terminal' in z for z in f.zeilen))

# Abgeschaltet wartet er ebenfalls nur die Zeit ab.
sys.stdin = types.SimpleNamespace(isatty=lambda: True)
f = PauseAttrappe(mit_taste=False, tty=True)
pruefe('pause_mit_taste=False nimmt wieder die Uhr',
       not f._pause_vorbei(101.0) and f._pause_vorbei(102.1)
       and not any('ENTER' in z for z in f.zeilen))

sys.stdin = echtes_stdin

print('\nalle Tests bestanden')
