#!/usr/bin/env python3
"""
Ausparken aus der Startluecke (ROS-frei, testbar).

GEOMETRIE. Die beiden Magenta-Waende stehen SENKRECHT auf dem Aussenwall und
ragen 20 cm ins Feld. Die Luecke ist der Spalt zwischen ihnen, 1,5 x
Fahrzeuglaenge = 26,25 cm. Der Roboter steht laengs darin, seine Laengsachse
also parallel zum Aussenwall. Er muss damit SEITLICH heraus -- und das geht
bei Ackermann-Lenkung nur ueber Rangieren. Mit 17,5 cm Fahrzeuglaenge bleiben
8,75 cm Laengsspiel; ein Zug dreht bei Vollausschlag rund 13 Grad.

Hier drin stecken drei Dinge:

  richtung_aus_scan()  Welche Seite ist offen? Die NAHE Seite ist der
                       Aussenwall, die FERNE das Spielfeld. Feld rechts -> CW,
                       Feld links -> CCW. Das folgt aus field_map.py:
                       START_POSES_CW setzt den Roboter auf y=+1,0 mit Kurs
                       +x, also Aussenwall (y=+1,5) links und Innenblock
                       (y=+0,5) rechts; der Kommentar darunter sagt fuer CCW
                       ausdruecklich "inner wall to the left".

  Ausparkplan          Die Schrittfolge in cm und Lenkprozent. Positive
                       Lenkung heisst IMMER "zur offenen Seite" -- die Tabelle
                       ist damit richtungsfrei, gespiegelt wird erst beim
                       Ausfuehren.

  simuliere()          Trockenlauf: faehrt die Tabelle im Kopf und prueft jede
                       Zwischenlage gegen die Lueckenmasse. Damit laesst sich
                       eine neue Schrittfolge pruefen, ohne den Roboter gegen
                       eine Wand zu setzen.

Der Weg wird ueber den Encoder gemessen, nicht ueber den Lidar: unterhalb
0,15 m (range_min) liefert der Lidar keine Punkte, und in der Luecke ist die
naechste Wand genau dort.
"""
import io
import json
import math
import os

import numpy as np


# --- Fahrzeug ------------------------------------------------------------
# base_link sitzt auf der HINTERACHSE. Der Ueberhang nach hinten ist deshalb
# nur 3,5 cm, was den Heckausschlag beim Einlenken auf 1,7 mm begrenzt.
FZ_BREITE = 0.110
FZ_LAENGE = 0.175
FZ_NASE = 0.140                      # nose_offset aus round1_controller_node
FZ_HECK = FZ_NASE - FZ_LAENGE        # = -0.035

# --- Luecke --------------------------------------------------------------
LUECKE_LAENGE = 1.5 * FZ_LAENGE      # 0.2625 m, Vorgabe aus dem Reglement
LUECKE_TIEFE = 0.200                 # wie weit die Magenta-Waende ins Feld ragen
# Dicke der Magenta-Waende. Entscheidend, auch wenn sie klein ist: die
# Waende sind BALKEN, keine Mauern. Sobald der Roboter an einer vorbei
# ist, ist er frei -- er muss sie nicht seitlich umfahren. Wer sie als
# Halbebene rechnet, verbietet Folgen, die in Wirklichkeit passen.
LUECKE_WANDDICKE = 0.020

# --- Lenkung -------------------------------------------------------------
# Kennlinie, Trimm und Radstand kommen aus der GEMESSENEN Kalibrierung,
# esp_bridge/steer_calib.json -- derselben Datei, aus der auch die
# Bruecke ihre Lenkung speist. Nichts davon wird hier abgeschrieben: wer neu
# kalibriert, soll nicht daran denken muessen, es an zweiter Stelle
# nachzutragen.
#
# Nur wenn die Datei fehlt, greifen die Werte unten -- sie sind ein Abzug vom
# 08.09.2026 und stehen ausdruecklich als Notnagel da. Der Trockenlauf sagt
# dann auch, dass er raet.
STEER_CALIB_UMGEBUNG = 'STEER_CALIB'     # Pfad per Umgebungsvariable

NOT_KENNLINIE = [
    (-100.0, -17.76), (-80.0, -14.29), (-65.0, -11.61),
    (-50.0, -9.42), (-35.0, -5.33), (-2.0, 0.0),
    (35.0, 7.60), (50.0, 10.07), (65.0, 12.66),
    (80.0, 14.56), (100.0, 18.10),
]
NOT_MITTE = -2.0
NOT_RADSTAND = 0.10


def steer_calib_pfade():
    """Wo nach steer_calib.json gesucht wird, in dieser Reihenfolge."""
    pfade = []
    aus_umgebung = os.environ.get(STEER_CALIB_UMGEBUNG)
    if aus_umgebung:
        pfade.append(aus_umgebung)
    # Nachbarpaket im selben Workspace. __file__ aufloesen, weil dieses Modul
    # ueber den colcon-Symlink build/ekf/ekf/ geladen wird.
    hier = os.path.dirname(os.path.realpath(__file__))
    src = os.path.dirname(os.path.dirname(hier))          # .../src
    pfade.append(os.path.join(src, 'esp_bridge', 'esp_bridge',
                              'steer_calib.json'))
    pfade.append('/workspace/src/esp_bridge/esp_bridge/steer_calib.json')
    # Altlast: bis September 2026 lag die Datei im wall_follower_robot-Paket.
    pfade.append(os.path.join(src, 'wall_follower_robot',
                              'wall_follower_robot', 'steer_calib.json'))
    pfade.append('/workspace/src/wall_follower_robot/wall_follower_robot/'
                 'steer_calib.json')
    return pfade


def lade_lenkkennlinie(pfad=None, tempo=None):
    """steer_calib.json einlesen.

    ``tempo`` waehlt die Geschwindigkeitsstufe; ohne Angabe die LANGSAMSTE.
    Beim Ausparken kriecht der Roboter, und die Kennlinie haengt vom Tempo ab
    (bei mehr Tempo schmiert der Reifen und der wirksame Lenkwinkel sinkt).

    Rueckgabe: (kennlinie, mitte, radstand, quelle) mit der Kennlinie als
    aufsteigende Liste (prozent, grad). ``quelle`` ist der benutzte Pfad oder
    None, wenn nichts gelesen werden konnte.
    """
    versucht = []
    for kandidat in ([pfad] if pfad else steer_calib_pfade()):
        try:
            with io.open(kandidat, encoding='utf-8') as f:
                daten = json.load(f)
            stufen = sorted(daten['speeds'], key=lambda e: float(e['v']))
            if not stufen:
                raise ValueError('keine Geschwindigkeitsstufe enthalten')
            if tempo is None:
                stufe = stufen[0]
            else:
                stufe = min(stufen, key=lambda e: abs(float(e['v']) - tempo))
            punkte = {}
            for seite in ('left', 'right'):
                for servo, delta_rad in stufe[seite]:
                    # servo -1..1 -> Prozent; die Mitte steht in beiden Seiten
                    punkte[round(float(servo) * 100.0, 6)] = \
                        math.degrees(float(delta_rad))
            if len(punkte) < 3:
                raise ValueError('zu wenige Stuetzpunkte')
            kennlinie = sorted(punkte.items())
            # Der Trimm ist der Punkt, an dem die Lenkung wirklich gerade
            # steht -- nicht 0 Prozent.
            mitte = min(kennlinie, key=lambda pd: abs(pd[1]))[0]
            radstand = float(daten.get('wheelbase', NOT_RADSTAND))
            return kennlinie, mitte, radstand, kandidat
        except Exception as fehler:
            versucht.append('%s: %s' % (kandidat, fehler))

    lade_lenkkennlinie.versucht = versucht
    return NOT_KENNLINIE, NOT_MITTE, NOT_RADSTAND, None


LENK_KENNLINIE, LENK_MITTE, RADSTAND, LENK_QUELLE = lade_lenkkennlinie()


# --- Encoder -------------------------------------------------------------
# r_eff aus ekf.py: 0,0150 m pro rad der Ausgangswelle (Strecken-Kalibrierung
# 2,41 m / 10431 Ticks). 1 cm sind damit 38,2 Grad Wellendrehung; die
# Aufloesung auf der Leitung ist 0,1 Grad = 26 Mikrometer.
R_EFF = 0.0150


def cm_zu_grad(cm, r_eff=R_EFF):
    """Fahrweg in cm -> Drehung der Ausgangswelle in Grad."""
    return (cm / 100.0) / r_eff * 180.0 / math.pi


def grad_zu_cm(grad, r_eff=R_EFF):
    return grad * math.pi / 180.0 * r_eff * 100.0


def lenkwinkel(prozent):
    """Lenkprozent -> Lenkwinkel in rad, aus der gemessenen Kennlinie."""
    xs = [p for p, _ in LENK_KENNLINIE]
    ys = [math.radians(d) for _, d in LENK_KENNLINIE]
    return float(np.interp(float(prozent), xs, ys))


def wenderadius(prozent, radstand=RADSTAND):
    """Wenderadius in m. Unendlich (None) bei Geradeausstellung."""
    delta = lenkwinkel(prozent)
    if abs(delta) < 1e-4:
        return None
    return radstand / math.tan(abs(delta))


# =========================================================================
# Fahrtrichtung aus einem einzelnen Scan
# =========================================================================

def richtung_aus_scan(punkte, halbwinkel_grad=20.0, min_punkte=5,
                      max_verhaeltnis=2.0):
    """CW/CCW aus einem Scan in der Parkluecke.

    ``punkte``: (N,2)-Feld im Roboterrahmen (REP-103, +x vorwaerts, +y links),
    also genau das, was ``wall_extraction.scan_to_points`` liefert.

    Verglichen werden zwei schmale Sektoren um +-90 Grad. Der Sektor bleibt
    schmal, damit die beiden Magenta-Waende vorn und hinten nicht hineinragen.

    Die nahe Seite ist der Aussenwall. Sie liefert oft GAR KEINE Punkte, weil
    der Lidar unter range_min (0,15 m) nichts zurueckgibt und die Wand in der
    Luecke etwa 0,145 m entfernt steht -- knapp darunter. Eine leere Seite ist
    deshalb kein Fehler, sondern das Signal "hier ist die Wand".

    Rueckgabe:
        {'richtung': 'CW'|'CCW'|None, 'sicher': bool,
         'links_m': float|None, 'rechts_m': float|None,
         'links_n': int, 'rechts_n': int, 'grund': str}
    """
    leer = {'richtung': None, 'sicher': False, 'links_m': None,
            'rechts_m': None, 'links_n': 0, 'rechts_n': 0}

    pts = np.asarray(punkte, dtype=float)
    if pts.ndim != 2 or pts.shape[0] == 0:
        return dict(leer, grund='kein Punkt im Scan')

    winkel = np.arctan2(pts[:, 1], pts[:, 0])
    reichweite = np.hypot(pts[:, 0], pts[:, 1])
    tol = math.radians(halbwinkel_grad)

    def seite(mitte):
        d = np.abs(np.arctan2(np.sin(winkel - mitte), np.cos(winkel - mitte)))
        treffer = reichweite[d <= tol]
        if treffer.size == 0:
            return None, 0
        return float(np.median(treffer)), int(treffer.size)

    links_m, links_n = seite(math.pi / 2.0)
    rechts_m, rechts_n = seite(-math.pi / 2.0)
    mess = dict(leer, links_m=links_m, rechts_m=rechts_m,
                links_n=links_n, rechts_n=rechts_n)

    genug_l = links_n >= min_punkte
    genug_r = rechts_n >= min_punkte

    if not genug_l and not genug_r:
        return dict(mess, grund='beide Seiten leer -- steht der Roboter frei?')

    # Genau eine Seite leer: die leere ist die Wand, die andere das Feld.
    if genug_l != genug_r:
        feld_links = genug_l
        return dict(mess, richtung='CCW' if feld_links else 'CW', sicher=True,
                    grund=('links %.2f m, rechts ohne Rueckgabe (Wand unter range_min)'
                           % links_m) if feld_links else
                          ('rechts %.2f m, links ohne Rueckgabe (Wand unter range_min)'
                           % rechts_m))

    # Beide Seiten sichtbar: die deutlich fernere ist das Feld.
    fern, nah = max(links_m, rechts_m), min(links_m, rechts_m)
    feld_links = links_m > rechts_m
    sicher = fern >= max_verhaeltnis * nah
    return dict(mess, richtung=('CCW' if feld_links else 'CW') if sicher else None,
                sicher=sicher,
                grund='links %.2f m, rechts %.2f m%s'
                      % (links_m, rechts_m,
                         '' if sicher else ' -- zu aehnlich, keine Entscheidung'))


# =========================================================================
# Schrittfolge
# =========================================================================

def schritte_aus_flach(flach):
    """[lenk1, cm1, lenk2, cm2, ...] -> [(lenk, cm), ...].

    Eine flache Liste, weil ROS-Parameter nur homogene Felder koennen.
    """
    werte = [float(v) for v in flach]
    if len(werte) % 2 != 0:
        raise ValueError('Schrittliste braucht Paare aus Lenkung und cm, '
                         'bekam %d Werte' % len(werte))
    schritte = []
    for i in range(0, len(werte), 2):
        lenk, cm = werte[i], werte[i + 1]
        if not -100.0 <= lenk <= 100.0:
            raise ValueError('Lenkung %.1f %% ausserhalb -100..100' % lenk)
        schritte.append((lenk, cm))
    return schritte


def lenk_auf_leitung(anteil):
    """Tabellenwert (-100..100, Anteil des Vollausschlags) -> Servoprozent.

    Der Trimm LENK_MITTE ist der NULLPUNKT der Lenkung, kein Versatz: 0 in der
    Tabelle muss als -2 % rausgehen, +-100 aber als genau +-100, sonst
    verlangen wir mehr als den Anschlag und der ESP klemmt stillschweigend.
    Also linear vom Trimm zum jeweiligen Anschlag skalieren.
    """
    a = max(-100.0, min(100.0, float(anteil)))
    spanne = (100.0 - LENK_MITTE) if a >= 0.0 else (100.0 + LENK_MITTE)
    return LENK_MITTE + spanne * a / 100.0


def spiegeln(schritte, offen_links):
    """Tabelle auf die tatsaechliche Seite drehen und auf die Leitung bringen.

    In der Tabelle heisst positive Lenkung "zur offenen Seite". Liegt die
    offene Seite links (CCW), stimmt das Vorzeichen schon; liegt sie rechts
    (CW), wird gespiegelt.
    """
    vz = 1.0 if offen_links else -1.0
    return [(lenk_auf_leitung(vz * lenk), cm) for lenk, cm in schritte]


# =========================================================================
# Trockenlauf
# =========================================================================

def ecken(pose, breite=FZ_BREITE, nase=FZ_NASE, heck=FZ_HECK):
    """Die vier Fahrzeugecken in Weltkoordinaten."""
    x, y, th = pose
    c, s = math.cos(th), math.sin(th)
    halb = breite / 2.0
    return [(x + c * lx - s * ly, y + s * lx + c * ly)
            for lx, ly in ((nase, -halb), (nase, halb),
                           (heck, halb), (heck, -halb))]


def _bogen(pose, strecke, radius, links):
    """Eine Teilstrecke fahren. ``radius`` None = geradeaus."""
    x, y, th = pose
    if radius is None:
        return (x + math.cos(th) * strecke, y + math.sin(th) * strecke, th)
    vz = 1.0 if links else -1.0
    dth = vz * strecke / radius
    px = x - vz * radius * math.sin(th)
    py = y + vz * radius * math.cos(th)
    nth = th + dth
    return (px + vz * radius * math.sin(nth),
            py - vz * radius * math.cos(nth), nth)


def bahn(startpose, schritte, feinheit=0.002):
    """Alle Zwischenlagen als [(pose, schritt_nr)]. schritt_nr zaehlt ab 1,
    die Startpose bekommt 0."""
    pose = tuple(startpose)
    posen = [(pose, 0)]
    for nr, (lenk, cm) in enumerate(schritte, 1):
        strecke = cm / 100.0
        R = wenderadius(lenk)
        links = lenk > LENK_MITTE
        n = max(1, int(abs(strecke) / feinheit))
        for i in range(1, n + 1):
            posen.append((_bogen(pose, strecke * i / n, R, links), nr))
        pose = posen[-1][0]
    return posen


def startpose(rueckstand=0.0, laengsspiel=0.004,
              tiefe=LUECKE_TIEFE, breite=FZ_BREITE):
    """Abstellpose in der Luecke.

    Nullpunkt: Aussenwall bei y=0, INNENKANTE der hinteren Magenta-Wand bei
    x=0, Kurs +x (also entlang der Bahn). ``rueckstand`` ist der Abstand der
    Innenflanke von den Wandspitzen, ``laengsspiel`` die Luft zwischen Heck
    und hinterer Wand.
    """
    return (-FZ_HECK + laengsspiel, tiefe - breite / 2.0 - rueckstand, 0.0)


# --- Flaechen und ihre Ueberschneidung -----------------------------------

def rechteck(x0, x1, y0, y1):
    return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]


def hindernisse(laenge=LUECKE_LAENGE, tiefe=LUECKE_TIEFE,
                dicke=LUECKE_WANDDICKE):
    """Die beiden Magenta-Balken als Rechtecke, links und rechts der Luecke."""
    return [rechteck(-dicke, 0.0, 0.0, tiefe),
            rechteck(laenge, laenge + dicke, 0.0, tiefe)]


def schlitz(laenge=LUECKE_LAENGE, tiefe=LUECKE_TIEFE):
    """Der Raum ZWISCHEN den Waenden. Wer ihn verlassen hat, ist ausgeparkt."""
    return rechteck(0.0, laenge, 0.0, tiefe)


def ueberlappen(a, b):
    """Schneiden sich zwei konvexe Vierecke? Trennachsensatz.

    Eckenvergleiche allein reichen hier NICHT: ein 2 cm dicker Balken kann
    quer durch den Roboter gehen, ohne dass eine Ecke von beiden im jeweils
    anderen liegt -- genau die Lage, die beim Ausparken entsteht.
    """
    for poly in (a, b):
        n = len(poly)
        for i in range(n):
            (x1, y1), (x2, y2) = poly[i], poly[(i + 1) % n]
            achse = (-(y2 - y1), x2 - x1)
            betrag = math.hypot(*achse)
            if betrag < 1e-12:
                continue
            achse = (achse[0] / betrag, achse[1] / betrag)
            amin = min(px * achse[0] + py * achse[1] for px, py in a)
            amax = max(px * achse[0] + py * achse[1] for px, py in a)
            bmin = min(px * achse[0] + py * achse[1] for px, py in b)
            bmax = max(px * achse[0] + py * achse[1] for px, py in b)
            if amax <= bmin + 1e-12 or bmax <= amin + 1e-12:
                return False
    return True


def _punkt_strecke(p, a, b):
    px, py = p
    ax, ay = a
    bx, by = b
    dx, dy = bx - ax, by - ay
    laenge2 = dx * dx + dy * dy
    if laenge2 < 1e-18:
        return math.hypot(px - ax, py - ay)
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / laenge2))
    return math.hypot(px - (ax + t * dx), py - (ay + t * dy))


def abstand(a, b):
    """Kleinster Abstand zweier konvexer Vierecke. 0 bei Ueberschneidung."""
    if ueberlappen(a, b):
        return 0.0
    kleinster = float('inf')
    for erst, zweit in ((a, b), (b, a)):
        n = len(zweit)
        for punkt in erst:
            for i in range(n):
                kleinster = min(kleinster,
                                _punkt_strecke(punkt, zweit[i], zweit[(i + 1) % n]))
    return kleinster


def simuliere(schritte, start=None, tiefe=LUECKE_TIEFE,
              laenge=LUECKE_LAENGE, rand=0.008, dicke=LUECKE_WANDDICKE):
    """Trockenlauf gegen die Lueckenmasse.

    Die Magenta-Waende sind Rechtecke der Dicke ``dicke`` -- Balken, keine
    Mauern. Der Aussenwall bei y=0 ist eine harte Grenze.

    ``kollision`` meint ECHTES Ueberlappen, nicht "zu wenig Reserve". Die
    Reserve steht getrennt in den Abstaenden: wer den Roboter mit dem Heck an
    die hintere Wand stellt, faengt eben mit 0 mm an -- das ist knapp, aber
    keine Kollision. ``rand`` ist nur die Schwelle, ab der ``knapp`` gesetzt
    wird.

    Rueckgabe:
        {'frei': bool, 'kollision': bool, 'knapp': bool, 'endpose',
         'wand_abstand_m'    kleinster Abstand einer Ecke zum Aussenwall,
         'magenta_abstand_m' kleinster Abstand zu einer Magenta-Wand,
         'bei_schritt'       Nummer des Zuges, in dem es zuerst aufsetzt,
         'posen'}
    """
    if start is None:
        start = startpose(tiefe=tiefe)
    posen = bahn(start, schritte)
    balken = hindernisse(laenge, tiefe, dicke)
    raum = schlitz(laenge, tiefe)
    wand = float('inf')
    magenta = float('inf')
    bei = None

    for pose, nr in posen:
        auto = ecken(pose)
        wand = min(wand, min(py for (_px, py) in auto))
        d = min(abstand(auto, b) for b in balken)
        magenta = min(magenta, d)
        if bei is None and (wand < 0.0 or d <= 0.0):
            bei = nr

    ende = posen[-1][0]
    frei = not ueberlappen(ecken(ende), raum)
    return {'frei': frei, 'kollision': bei is not None,
            'knapp': magenta < rand or wand < rand, 'endpose': ende,
            'wand_abstand_m': wand, 'magenta_abstand_m': magenta,
            'bei_schritt': bei, 'posen': posen}


# --- Standardfolge -------------------------------------------------------
# Ausgerechnet fuer 20 cm tiefe Waende, Innenflanke buendig mit den Spitzen,
# 1 cm Luft zur hinteren Wand, 8 mm Reserve zu den Magenta-Waenden.
#
# Vier Rangierzuege drehen den Roboter auf 35 Grad -- mehr geht in 26,25 cm
# Luecke nicht, weil er SEITLICH heraus muss und dafuer nur 8,75 cm
# Laengsspiel hat. Dann traegt ihn ein Bogen aus der Luecke, und ein
# Gegenbogen legt ihn wieder auf Bahnkurs. Er endet bei y = 0,47 m, also
# fast auf der Spurmitte (Spur 1,00 m breit).
#
# Positive Lenkung heisst ZUR OFFENEN SEITE, negative Strecke rueckwaerts.
SCHRITTE_STANDARD = [
    0.0,  0.0,
     100.0,   6.9,     # vorwaerts, voll zur offenen Seite
    -100.0,  -5.4,     # rueckwaerts, voll zur Wandseite
     100.0,   9.6,
     0.0, 5.0,
    -100.0, 21.0,     # Bogen aus der Luecke heraus    # Gegenbogen zurueck auf Bahnkurs
     0.0,  0.0,
]


# Je Fahrtrichtung eine eigene Folge, wenn sie gebraucht wird.
#
# Gespiegelt wird ohnehin (positive Lenkung heisst "zur offenen Seite"), aber
# das reicht nur, solange der Roboter in beiden Faellen GLEICH in der Luecke
# steht. Tut er das nicht, sind es andere Wege, nicht nur andere Vorzeichen.
#
# Leer heisst: SCHRITTE_STANDARD gilt. Wer nur eine Richtung anders braucht,
# fuellt nur diese -- die andere bleibt leer und folgt weiter dem Standard.
# Gemessen am 11.09.2026 (je 5-7 Laeufe, Handmessung an den Radnaben):
# Die Rangierzuege 1-4 sind in beiden Richtungen gleich, nur der Schlussbogen
# unterscheidet sich -- die Lenkung ist im Rangiertempo asymmetrisch, und das
# Spiegeln allein gleicht das nicht aus. Beide Folgen enden bei 0 grad.
#   CW : Schlussbogen 27,0 cm -> Kurs +0,6 grad, base_link 37,0 cm zur Aussenbande
#   CCW: Schlussbogen 21,0 cm -> Kurs  0,0 grad, base_link 34,5 cm zur Aussenbande
SCHRITTE_CW = [
    0.0,  0.0,
     100.0,   6.9,
    -100.0,  -5.4,
     100.0,   9.6,
     0.0, 5.0,
    -100.0, 27.0,
     0.0,  0.0,
]
SCHRITTE_CCW = [
    0.0,  0.0,
     100.0,   6.9,
    -100.0,  -5.4,
     100.0,   9.6,
     0.0, 5.0,
    -100.0, 21.0,
     0.0,  0.0,
]


def schritte_fuer(richtung, gemeinsam=None, cw=None, ccw=None):
    """Welche Schrittfolge gilt fuer diese Fahrtrichtung?

    Rueckgabe: (flache Liste, Herkunft als Text fuers Protokoll).
    """
    if richtung not in ('CW', 'CCW'):
        raise ValueError('Fahrtrichtung "%s" ist weder CW noch CCW' % richtung)
    eigen = (cw if cw is not None else SCHRITTE_CW) if richtung == 'CW' \
        else (ccw if ccw is not None else SCHRITTE_CCW)
    if eigen:
        return list(eigen), 'eigene Folge fuer %s' % richtung
    geteilt = gemeinsam if gemeinsam is not None else SCHRITTE_STANDARD
    if not geteilt:
        raise ValueError('weder eine Folge fuer %s noch eine gemeinsame'
                         % richtung)
    return list(geteilt), 'gemeinsame Folge'


def _trockenlauf(flach=None, laenge=LUECKE_LAENGE, tiefe=LUECKE_TIEFE,
                 spalt=0.004, dicke=LUECKE_WANDDICKE, richtung=None):
    """Tabelle im Kopf fahren und das Ergebnis ausgeben.

    laenge/tiefe sind die gemessenen Lueckenmasse, spalt die Luft zwischen
    Heck und hinterer Wand beim Abstellen.
    """
    # Gespiegelt wird mit offen_links=True: das laesst die Vorzeichen, wie
    # sie in der Tabelle stehen, addiert aber den Trimm -- der Trockenlauf
    # faehrt damit genau die Lenkwerte, die spaeter auf die Leitung gehen.
    if flach:
        roh, herkunft = flach, 'Kommandozeile'
    elif richtung:
        roh, herkunft = schritte_fuer(richtung)
    else:
        roh, herkunft = SCHRITTE_STANDARD, 'gemeinsame Folge'
    schritte = spiegeln(schritte_aus_flach(roh), True)
    start = startpose(laengsspiel=spalt, tiefe=tiefe)
    balken = hindernisse(laenge, tiefe, dicke)
    print('Luecke %.1f cm lang, Waende %.0f cm tief und %.1f cm dick, '
          'Fahrzeug %.1f x %.1f cm.'
          % (laenge * 100, tiefe * 100, dicke * 100,
             FZ_BREITE * 100, FZ_LAENGE * 100))
    print('Start base_link (%.3f, %.3f), %.0f mm Luft nach hinten.'
          % (start[0], start[1], spalt * 1000))
    print('Schrittfolge: %s%s.'
          % (herkunft, ' (%s)' % richtung if richtung else ''))
    if LENK_QUELLE:
        print('Lenkung aus %s: Trimm %.1f %%, Radstand %.3f m, '
              'Vollausschlag R = %.3f m.'
              % (LENK_QUELLE, LENK_MITTE, RADSTAND, wenderadius(100.0)))
    else:
        print('ACHTUNG: steer_calib.json nicht gefunden -- gerechnet wird mit '
              'dem Notnagel vom 08.09.2026, nicht mit eurer Kalibrierung.')
        for zeile in getattr(lade_lenkkennlinie, 'versucht', []):
            print('  versucht: %s' % zeile)
    print()
    pose = start
    for i, (lenk, cm) in enumerate(schritte, 1):
        # Engste Stelle NUR in diesem Zug -- so sieht man, welcher Zug die
        # Grenze setzt und wo noch Luft ist.
        eng = min(abstand(ecken(p), b)
                  for (p, _nr) in bahn(pose, [(lenk, cm)])
                  for b in balken)
        pose = bahn(pose, [(lenk, cm)])[-1][0]
        R = wenderadius(lenk)
        print('  %d. Lenkung %+6.1f %% (R %s)  %+6.1f cm = %+7.0f grad Welle'
              '  -> Kurs %+6.1f grad, y=%.3f   Rand %s'
              % (i, lenk, '%.2f m' % R if R else 'gerade', cm, cm_zu_grad(cm),
                 math.degrees(pose[2]), pose[1],
                 'beruehrt' if eng <= 0.0 else '%3.0f mm' % (eng * 1000)))
    e = simuliere(schritte, start, tiefe=tiefe, laenge=laenge, dicke=dicke)
    print()
    print('  Gesamtweg %.1f cm, %d Positionsfahrten.'
          % (sum(abs(cm) for _l, cm in schritte), len(schritte)))
    print('  Engster Abstand zu einer Magenta-Wand: %.0f mm.'
          % (e['magenta_abstand_m'] * 1000))
    print('  Engster Abstand zum Aussenwall:         %.0f mm.'
          % (e['wand_abstand_m'] * 1000))
    if e['kollision']:
        print('  KOLLISION in Zug %s' % e['bei_schritt'])
    elif e['knapp']:
        print('  kollisionsfrei, aber knapp (unter 8 mm Reserve)')
    else:
        print('  kollisionsfrei')
    print('  %s' % ('aus der Luecke heraus' if e['frei']
                    else 'ACHTUNG: am Ende noch in der Luecke'))
    return 0 if (e['frei'] and not e['kollision']) else 1


if __name__ == '__main__':
    import sys

    HILFE = """Trockenlauf einer Ausparkfolge.

  python3 ausparken.py [cw|ccw] [luecke=CM] [tiefe=CM] [dicke=CM] [spalt=MM]
                       [lenk cm lenk cm ...]

Ohne Zahlen wird SCHRITTE_STANDARD gefahren. Die Masse sind die GEMESSENEN
der echten Luecke -- stimmen sie nicht, sagt der Trockenlauf das Falsche.

  luecke  Abstand zwischen den beiden Magenta-Waenden (Standard %.2f cm)
  tiefe   wie weit sie vom Aussenwall ins Feld ragen (Standard %.0f cm)
  dicke   Dicke der Balken laengs der Bahn (Standard %.1f cm) -- sie sind
          BALKEN, keine Mauern: hinter ihnen ist wieder frei
  spalt   Luft zwischen Heck und hinterer Wand beim Abstellen (Standard 4 mm)
  cw/ccw  die fuer diese Fahrtrichtung hinterlegte Folge fahren (SCHRITTE_CW
          bzw. SCHRITTE_CCW, sonst SCHRITTE_STANDARD)

Beispiel:
  python3 ausparken.py luecke=32 100 9 -100 -6 100 7 -100 -5 100 18 -100 37
""" % (LUECKE_LAENGE * 100, LUECKE_TIEFE * 100, LUECKE_WANDDICKE * 100)

    if '-h' in sys.argv or '--help' in sys.argv:
        print(HILFE)
        raise SystemExit(0)

    masse = {'luecke': LUECKE_LAENGE, 'tiefe': LUECKE_TIEFE,
             'dicke': LUECKE_WANDDICKE, 'spalt': 0.004}
    zahlen = []
    richtung = None
    for arg in sys.argv[1:]:
        if arg.upper() in ('CW', 'CCW'):
            richtung = arg.upper()
        elif '=' in arg:
            name, _, wert = arg.partition('=')
            if name not in masse:
                print('Unbekanntes Mass "%s".\n' % name)
                print(HILFE)
                raise SystemExit(2)
            teiler = 1000.0 if name == 'spalt' else 100.0
            masse[name] = float(wert) / teiler
        else:
            zahlen.append(float(arg))

    raise SystemExit(_trockenlauf(zahlen or None, laenge=masse['luecke'],
                                  tiefe=masse['tiefe'], spalt=masse['spalt'],
                                  dicke=masse['dicke'], richtung=richtung))