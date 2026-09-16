#!/usr/bin/env python3
"""Tests fuer ausparken.py. Ohne ROS, aus dem Paketverzeichnis heraus:

    cd src/ekf/ekf && python3 test_ausparken.py

(wie test_obstacle_path.py -- ein Import "from ekf import ..." scheitert hier,
weil ekf.py im selben Verzeichnis das Paket ekf verdeckt.)
"""
import math

import numpy as np

import ausparken as A


def pruefe(name, bedingung, zusatz=''):
    if not bedingung:
        raise AssertionError('FEHLGESCHLAGEN: %s %s' % (name, zusatz))
    print('  ok  %s%s' % (name, ('  ' + zusatz) if zusatz else ''))


def sektor(mitte_grad, reichweite, n=40, streuung=0.0):
    """Punktwolke in einem Sektor, Roboterrahmen (+x vor, +y links)."""
    mitte = math.radians(mitte_grad)
    w = mitte + np.linspace(-0.25, 0.25, n)
    r = reichweite + streuung * np.sin(np.arange(n))
    return np.column_stack((r * np.cos(w), r * np.sin(w)))


print('Umrechnung Weg <-> Encoder')
pruefe('1 cm sind 38.2 Grad Welle', abs(A.cm_zu_grad(1.0) - 38.197) < 0.01,
       '%.3f' % A.cm_zu_grad(1.0))
pruefe('Hin und zurueck', abs(A.grad_zu_cm(A.cm_zu_grad(7.3)) - 7.3) < 1e-9)
pruefe('Vorzeichen bleibt', A.cm_zu_grad(-5.0) < 0)

print('\nLenkung')
pruefe('0 in der Tabelle ist der Trimm',
       abs(A.lenk_auf_leitung(0.0) - A.LENK_MITTE) < 1e-9)
pruefe('Vollausschlag bleibt Vollausschlag',
       abs(A.lenk_auf_leitung(100.0) - 100.0) < 1e-9 and
       abs(A.lenk_auf_leitung(-100.0) + 100.0) < 1e-9)
pruefe('ueber den Anschlag hinaus wird geklemmt',
       A.lenk_auf_leitung(150.0) == 100.0 and A.lenk_auf_leitung(-150.0) == -100.0)
pruefe('Wenderadius bei Vollausschlag rund 0.31 m',
       0.29 < A.wenderadius(100.0) < 0.33, '%.3f m' % A.wenderadius(100.0))
pruefe('Geradeausstellung hat keinen Radius', A.wenderadius(A.LENK_MITTE) is None)

print('\nSchrittliste')
pruefe('Paare werden erkannt',
       A.schritte_aus_flach([100.0, 5.0, -100.0, -3.0]) == [(100.0, 5.0), (-100.0, -3.0)])
try:
    A.schritte_aus_flach([100.0, 5.0, -100.0])
    pruefe('ungerade Liste faellt auf', False)
except ValueError:
    pruefe('ungerade Liste faellt auf', True)
try:
    A.schritte_aus_flach([120.0, 5.0])
    pruefe('Lenkung ausserhalb faellt auf', False)
except ValueError:
    pruefe('Lenkung ausserhalb faellt auf', True)

print('\nSpiegelung')
tab = A.schritte_aus_flach([100.0, 5.0, -100.0, -3.0, 0.0, 9.0])
links = A.spiegeln(tab, True)
rechts = A.spiegeln(tab, False)
pruefe('offen links: Vorzeichen bleiben',
       links[0][0] > 0 and links[1][0] < 0)
pruefe('offen rechts: Vorzeichen kippen',
       rechts[0][0] < 0 and rechts[1][0] > 0)
pruefe('Strecken bleiben unangetastet',
       [c for _l, c in links] == [c for _l, c in rechts] == [5.0, -3.0, 9.0])
pruefe('Geradeaus bleibt beidseitig der Trimm',
       abs(links[2][0] - A.LENK_MITTE) < 1e-9 and abs(rechts[2][0] - A.LENK_MITTE) < 1e-9)
pruefe('nie mehr als Vollausschlag',
       all(abs(l) <= 100.0 + 1e-9 for l, _c in links + rechts))

print('\nFahrtrichtung aus dem Scan')
# Aussenwall rechts (nah), Feld links (fern) -> Innenblock links -> CCW
pts = np.vstack((sektor(+90, 0.90), sektor(-90, 0.16)))
r = A.richtung_aus_scan(pts)
pruefe('Feld links -> CCW', r['richtung'] == 'CCW' and r['sicher'], r['grund'])
# gespiegelt
pts = np.vstack((sektor(+90, 0.16), sektor(-90, 0.90)))
r = A.richtung_aus_scan(pts)
pruefe('Feld rechts -> CW', r['richtung'] == 'CW' and r['sicher'], r['grund'])
# nahe Seite ganz ohne Rueckgabe (unter range_min)
r = A.richtung_aus_scan(sektor(+90, 0.90))
pruefe('leere Seite ist die Wand -> CCW', r['richtung'] == 'CCW' and r['sicher'],
       r['grund'])
r = A.richtung_aus_scan(sektor(-90, 0.75))
pruefe('leere Seite rechts -> CW', r['richtung'] == 'CW' and r['sicher'], r['grund'])
# beide Seiten aehnlich -> keine Entscheidung
pts = np.vstack((sektor(+90, 0.50), sektor(-90, 0.46)))
r = A.richtung_aus_scan(pts)
pruefe('zu aehnlich -> keine Entscheidung',
       r['richtung'] is None and not r['sicher'], r['grund'])
# gar nichts
r = A.richtung_aus_scan(np.zeros((0, 2)))
pruefe('leerer Scan -> keine Entscheidung', r['richtung'] is None)
# Punkte nur vorn und hinten (Magenta-Waende) duerfen nicht zaehlen
pts = np.vstack((sektor(0, 0.11), sektor(180, 0.15)))
r = A.richtung_aus_scan(pts)
pruefe('Magenta-Waende vorn/hinten bleiben draussen',
       r['links_n'] == 0 and r['rechts_n'] == 0)

print('\nTrockenlauf')
std = A.schritte_aus_flach(A.SCHRITTE_STANDARD)
e = A.simuliere(A.spiegeln(std, True), A.startpose(laengsspiel=0.010))
pruefe('Standardfolge setzt nirgends auf', not e['kollision'])
pruefe('Standardfolge haelt 8 mm Reserve', not e['knapp'],
       '%.0f mm' % (e['magenta_abstand_m'] * 1000))
pruefe('Standardfolge kommt heraus', e['frei'])
pruefe('Standardfolge endet auf Bahnkurs',
       abs(math.degrees(e['endpose'][2])) < 3.0,
       '%.1f grad' % math.degrees(e['endpose'][2]))
pruefe('Standardfolge endet in der Spur', 0.25 < e['endpose'][1] < 0.75,
       'y = %.3f m' % e['endpose'][1])
# gespiegelt muss es genauso gut gehen, nur andersherum
gespiegelt = A.simuliere(
    A.spiegeln(std, False),
    (A.startpose(laengsspiel=0.010)[0], -A.startpose(laengsspiel=0.010)[1], 0.0),
    tiefe=-A.LUECKE_TIEFE)
# Nicht exakt spiegelbildlich: die Lenkung ist es auch nicht (R = 0.306 m
# links, 0.312 m rechts aus der Kalibrierung). 1 cm Unterschied ist Physik,
# nicht Rechenfehler -- mehr waere einer.
pruefe('gespiegelte Folge dreht andersherum',
       gespiegelt['endpose'][1] < 0, 'y = %.3f m' % gespiegelt['endpose'][1])
pruefe('gespiegelte Folge kommt gleich weit',
       abs(abs(gespiegelt['endpose'][1]) - e['endpose'][1]) < 0.02,
       'Unterschied %.0f mm' % (abs(abs(gespiegelt['endpose'][1]) - e['endpose'][1]) * 1000))
pruefe('gespiegelte Folge endet ebenfalls auf Bahnkurs',
       abs(math.degrees(gespiegelt['endpose'][2])) < 3.0,
       '%.1f grad' % math.degrees(gespiegelt['endpose'][2]))
# blind geradeaus muss auffallen
e2 = A.simuliere(A.spiegeln([(0.0, 30.0)], True), A.startpose(laengsspiel=0.010))
pruefe('geradeaus laeuft in die vordere Wand', e2['kollision'], 'Zug %s' % e2['bei_schritt'])

print('\nalle Tests bestanden')
