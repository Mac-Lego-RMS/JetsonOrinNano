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
import math

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

# --- Lenkung -------------------------------------------------------------
# Gemessene Kennlinie aus wall_follower_robot/steer_calib.json bei v=0,35 m/s,
# der langsamsten kalibrierten Stufe. Prozent -> Lenkwinkel in Grad. Der
# Nullpunkt liegt bei -2 % (Trimm), nicht bei 0.
LENK_MITTE = -2.0
LENK_KENNLINIE = [
    (-100.0, -17.76), (-80.0, -14.29), (-65.0, -11.61),
    (-50.0, -9.42), (-35.0, -5.33), (LENK_MITTE, 0.0),
    (35.0, 7.60), (50.0, 10.07), (65.0, 12.66),
    (80.0, 14.56), (100.0, 18.10),
]
RADSTAND = 0.10

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

    Nullpunkt: Aussenwall bei y=0, hintere Magenta-Wand bei x=0, Kurs +x
    (also entlang der Bahn). ``rueckstand`` ist der Abstand der Innenflanke
    von den Wandspitzen, ``laengsspiel`` die Luft zwischen Heck und hinterer
    Wand.
    """
    return (-FZ_HECK + laengsspiel, tiefe - breite / 2.0 - rueckstand, 0.0)


def simuliere(schritte, start=None, tiefe=LUECKE_TIEFE,
              laenge=LUECKE_LAENGE, rand=0.008):
    """Trockenlauf gegen die Lueckenmasse.

    Die Magenta-Waende sind als Halbebenen x<=0 bzw. x>=laenge modelliert,
    jeweils nur bis zur Hoehe ``tiefe`` -- darueber ist frei.

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
        start = startpose()
    posen = bahn(start, schritte)
    wand = float('inf')
    magenta = float('inf')
    bei = None

    for pose, nr in posen:
        for (px, py) in ecken(pose):
            wand = min(wand, py)
            if py < tiefe:                       # auf Hoehe der Magenta-Waende
                magenta = min(magenta, min(px, laenge - px))
        if bei is None and (wand < 0.0 or magenta < 0.0):
            bei = nr

    ende = posen[-1][0]
    frei = all(py > tiefe for (_px, py) in ecken(ende))
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
     100.0,   5.9,     # vorwaerts, voll zur offenen Seite
    -100.0,  -4.4,     # rueckwaerts, voll zur Wandseite
     100.0,   4.6,
    -100.0,  -3.9,
     100.0,  17.7,     # Bogen aus der Luecke heraus
    -100.0,  37.1,     # Gegenbogen zurueck auf Bahnkurs
]


def _trockenlauf(flach=None):
    """Tabelle im Kopf fahren und das Ergebnis ausgeben."""
    # Gespiegelt wird mit offen_links=True: das laesst die Vorzeichen, wie
    # sie in der Tabelle stehen, addiert aber den Trimm -- der Trockenlauf
    # faehrt damit genau die Lenkwerte, die spaeter auf die Leitung gehen.
    schritte = spiegeln(schritte_aus_flach(flach or SCHRITTE_STANDARD), True)
    start = startpose()
    print('Luecke %.1f cm lang, Waende %.0f cm tief, Fahrzeug %.1f x %.1f cm.'
          % (LUECKE_LAENGE * 100, LUECKE_TIEFE * 100,
             FZ_BREITE * 100, FZ_LAENGE * 100))
    print('Start base_link (%.3f, %.3f), Kurs %+.1f grad.'
          % (start[0], start[1], math.degrees(start[2])))
    print()
    pose = start
    for i, (lenk, cm) in enumerate(schritte, 1):
        pose = bahn(pose, [(lenk, cm)])[-1][0]
        R = wenderadius(lenk)
        print('  %d. Lenkung %+6.1f %% (R %s)  %+6.1f cm = %+7.0f grad Welle'
              '  -> Kurs %+6.1f grad, y=%.3f'
              % (i, lenk, '%.2f m' % R if R else 'gerade', cm, cm_zu_grad(cm),
                 math.degrees(pose[2]), pose[1]))
    e = simuliere(schritte, start)
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
    werte = [float(a) for a in sys.argv[1:]] or None
    raise SystemExit(_trockenlauf(werte))
