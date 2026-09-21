#!/usr/bin/env python3
"""Tests fuer ausparken.py. Ohne ROS, aus dem Paketverzeichnis heraus:

    cd src/ekf/ekf && python3 test_ausparken.py

(wie test_obstacle_path.py -- ein Import "from ekf import ..." scheitert hier,
weil ekf.py im selben Verzeichnis das Paket ekf verdeckt.)
"""
import io
import json
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

print('\nKennlinie aus steer_calib.json')
pruefe('sie wird wirklich aus der Datei geladen', A.LENK_QUELLE is not None,
       str(A.LENK_QUELLE))
pruefe('und nicht vom Notnagel', A.LENK_KENNLINIE is not A.NOT_KENNLINIE)

kennlinie, mitte, radstand, quelle = A.lade_lenkkennlinie()
roh = json.load(io.open(quelle, encoding='utf-8'))
langsamste = sorted(roh['speeds'], key=lambda e: float(e['v']))[0]
pruefe('die langsamste Stufe wird genommen',
       abs(float(langsamste['v']) - 0.35) < 1e-9,
       'v = %s m/s' % langsamste['v'])

erwartet = {}
for seite in ('left', 'right'):
    for servo, delta in langsamste[seite]:
        erwartet[round(float(servo) * 100.0, 6)] = math.degrees(float(delta))
pruefe('jeder Stuetzpunkt stimmt mit der Datei ueberein',
       len(kennlinie) == len(erwartet)
       and all(abs(g - erwartet[pz]) < 1e-9 for pz, g in kennlinie),
       '%d Punkte' % len(kennlinie))
pruefe('die Kennlinie ist aufsteigend sortiert',
       [pz for pz, _g in kennlinie] == sorted(pz for pz, _g in kennlinie))
pruefe('der Radstand kommt aus der Datei',
       abs(radstand - float(roh['wheelbase'])) < 1e-12,
       '%.3f m' % radstand)
pruefe('der Trimm ist der Punkt ohne Lenkwinkel',
       abs(dict(kennlinie)[mitte]) < 1e-12, '%.1f %%' % mitte)

# Tempo waehlbar: bei mehr Tempo schmiert der Reifen, die Kennlinie ist flacher.
schnell, _m, _r, _q = A.lade_lenkkennlinie(tempo=0.75)
pruefe('mit tempo=0.75 kommt eine ANDERE Stufe', schnell != kennlinie)
pruefe('und tempo=0.0 liefert wieder die langsamste',
       A.lade_lenkkennlinie(tempo=0.0)[0] == kennlinie)

# Fehlt die Datei, wird geraten -- aber sichtbar.
ersatz, e_mitte, e_radstand, e_quelle = A.lade_lenkkennlinie(
    pfad='/gibt/es/nicht/steer_calib.json')
pruefe('fehlende Datei faellt auf den Notnagel zurueck',
       ersatz == A.NOT_KENNLINIE and e_quelle is None)
pruefe('... und der Versuch wird protokolliert',
       any('gibt/es/nicht' in z for z in A.lade_lenkkennlinie.versucht))

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

print('\nFlaechen')
# Der Trennachsensatz muss den Fall koennen, der Eckenvergleiche schlagen
# laesst: ein duenner Balken quer durch den Roboter, ohne dass eine Ecke von
# beiden im jeweils anderen liegt. Genau diese Lage entsteht beim Ausparken.
roboter = A.rechteck(-0.05, 0.12, -0.055, 0.055)
quer = A.rechteck(0.02, 0.04, -0.20, 0.20)
pruefe('Balken quer durch den Roboter wird erkannt',
       A.ueberlappen(roboter, quer))
pruefe('keine Ecke liegt dabei im anderen Rechteck',
       not any(-0.05 <= px <= 0.12 and -0.055 <= py <= 0.055 for px, py in quer)
       and not any(0.02 <= px <= 0.04 and -0.20 <= py <= 0.20 for px, py in roboter))
daneben = A.rechteck(0.30, 0.32, -0.20, 0.20)
pruefe('sauber getrennt ist nicht ueberlappend',
       not A.ueberlappen(roboter, daneben))
pruefe('Abstand stimmt', abs(A.abstand(roboter, daneben) - 0.18) < 1e-9,
       '%.3f m' % A.abstand(roboter, daneben))
pruefe('Ueberlappung hat Abstand 0', A.abstand(roboter, quer) == 0.0)
beruehrt = A.rechteck(0.12, 0.14, -0.20, 0.20)
pruefe('Beruehrung zaehlt nicht als Ueberlappung',
       not A.ueberlappen(roboter, beruehrt)
       and A.abstand(roboter, beruehrt) < 1e-9,
       '%.1e m' % A.abstand(roboter, beruehrt))

print('\nTrockenlauf')
# Bewusst NICHT gegen SCHRITTE_STANDARD: die Folge wird an der echten Luecke
# eingestellt, deren Masse von den Sollmassen des Reglements abweichen. Hier
# soll der Mechanismus geprueft werden, nicht die eingestellten Zahlen.
REFERENZ = [100.0, 5.9, -100.0, -4.4, 100.0, 4.6,
            -100.0, -3.9, 100.0, 17.7, -100.0, 37.1]
ref = A.schritte_aus_flach(REFERENZ)
e = A.simuliere(A.spiegeln(ref, True), A.startpose(laengsspiel=0.010))
pruefe('Referenzfolge setzt nirgends auf', not e['kollision'])
pruefe('Referenzfolge haelt 8 mm Reserve', not e['knapp'],
       '%.0f mm' % (e['magenta_abstand_m'] * 1000))
pruefe('Referenzfolge kommt heraus', e['frei'])
pruefe('Referenzfolge endet auf Bahnkurs',
       abs(math.degrees(e['endpose'][2])) < 3.0,
       '%.1f grad' % math.degrees(e['endpose'][2]))
pruefe('Referenzfolge endet in der Spur', 0.25 < e['endpose'][1] < 0.75,
       'y = %.3f m' % e['endpose'][1])

gespiegelt = A.simuliere(
    A.spiegeln(ref, False),
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

e2 = A.simuliere(A.spiegeln([(0.0, 30.0)], True), A.startpose(laengsspiel=0.010))
pruefe('geradeaus laeuft in die vordere Wand', e2['kollision'],
       'Zug %s' % e2['bei_schritt'])

print('\nDie eingestellte Standardfolge')
std = A.schritte_aus_flach(A.SCHRITTE_STANDARD)
pruefe('ist wohlgeformt', len(std) >= 1)
pruefe('laesst sich in beide Richtungen spiegeln',
       len(A.spiegeln(std, True)) == len(A.spiegeln(std, False)) == len(std))
pruefe('bleibt im Lenkbereich',
       all(abs(l) <= 100.0 + 1e-9 for l, _c in A.spiegeln(std, True)))
vor = max([cm for _l, cm in std if cm > 0], default=0.0)
zurueck = -min([cm for _l, cm in std if cm < 0], default=0.0)
# Bewusst KEIN Laengsspiel daraus ableiten: vorwaerts und rueckwaerts zu
# addieren gilt nur fuer Geradeausfahrt. Sobald der Roboter schraeg steht,
# kommt er beim Zurueckfahren nicht auf derselben Linie zurueck, sondern an der
# Wandspitze vorbei -- deshalb passen Folgen, die nach dieser Milchmaedchen-
# rechnung nicht passen duerften. Was wirklich gilt, sagt simuliere().
print('      laengster Zug vorwaerts %.1f cm, rueckwaerts %.1f cm'
      % (vor, zurueck))
e_std = A.simuliere(A.spiegeln(std, True), A.startpose())
print('      Trockenlauf gegen die Sollmasse: %s, %s'
      % ('Kollision in Zug %s' % e_std['bei_schritt'] if e_std['kollision']
         else 'kollisionsfrei',
         'kommt heraus' if e_std['frei'] else 'bleibt in der Luecke'))
print('      (die echte Luecke kann davon abweichen -- luecke=CM setzen)')

print('\nalle Tests bestanden')
