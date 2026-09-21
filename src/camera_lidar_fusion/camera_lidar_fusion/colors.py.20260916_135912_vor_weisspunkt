"""Farbklassifikation fuer die WRO-Klotzfarben (HSV, OpenCV-Wertebereiche).

OpenCV-HSV: H in 0..179, S und V in 0..255. Rot liegt um H=0 herum und
braucht deshalb zwei Intervalle.
"""

import cv2
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

# name -> Liste von (h_lo, h_hi) plus gemeinsame S/V-Untergrenzen
#
# Zu den v_min-Werten: die Pylonen sind im Randbereich des Fisheye deutlich
# dunkler als in der Bildmitte. An einer gruenen Pylone bei 0.99 m gemessen
# (21 Lidar-Punkte) lag V zwischen 39 und 106, der Median bei 44 -- die alte
# Schwelle von 45 schnitt also mitten durch die Pylone und erkannte nur 8 der
# 21 Punkte. Der Farbton war dabei bei ALLEN 21 sauber gruen (H 44..68).
#
# GRUEN GEGEN AUSGEBRANNTE WAND. Der Hue-Bereich hiess frueher (40, 90) und
# fing damit ueberbelichtete weisse Flaechen mit ein. Brennt eine Wand aus
# (V=255), bekommt sie einen Cyanstich und landet bei H 85 bis 94 -- also
# mitten im alten Bereich. Im Horizontring gemessen:
#     H 35..79 ->   59 Wandpixel, 4572 Pylonenpixel
#     H 85..94 -> 3811 Wandpixel,  309 Pylonenpixel
# Die Pylonen liegen bei H 56 bis 82, die ausgebrannten Waende bei 85 bis 94.
#
# Das war auch der Grund, warum s_min nicht weiter runter konnte: die Waende
# haben S 52 bis 58, die dunklen Pylonen S 62 bis 73 -- kaum Abstand. Ueber den
# Farbton getrennt geht es dagegen sauber, bei gleichem s_min=50:
#     hue (40,90) -> 72 Prozent der Pylonenpixel, 63 Prozent der Wandpixel
#     hue (35,85) -> 74 Prozent der Pylonenpixel,  4 Prozent der Wandpixel
# Deshalb zunaechst 85 statt 90 oben. Seit die Zone sitzt und v_min tief darf,
# musste der Bereich aber noch enger: bei H 80 bis 94 tauchten 45 Punkte auf,
# ueber Entfernungen von 0.60 bis 2.41 m verteilt -- das ist TUERKIS (H 83
# entspricht 166 Grad im Farbkreis), also Reflexe auf der Wand, kein Gruen.
# Die echten Pylonen liegen kompakt bei H 45 bis 64 und jeweils in EINER
# Entfernung. Am unteren Ende dasselbe Bild: H 30 bis 39 streute ueber 0.16
# bis 2.50 m, das sind gelbliche Holztoene.
#
# Deshalb jetzt 40 bis 72. Die Faustregel dahinter: echte Pylonenpunkte
# klumpen in Farbton UND Entfernung, Fehltreffer streuen in beidem.
#
# ENTFERNUNG ist der zweite grosse Faktor, und sie wirkt ueber die Saettigung,
# nicht ueber die Helligkeit. An vier gruenen Pylonen gemessen:
#     0.70 m -> S 101, 90 Prozent der Punkte erkannt
#     0.99 m -> S  76, 43 Prozent
#     2.08 m -> S  36,  9 Prozent
# V lag dabei ueberall bei 53 bis 74, war also nie das Problem. Bei 2 m ist die
# Pylone im Bild so klein, dass der Band-Median sie mit dem Hintergrund
# verwaschet -- dagegen hilft keine Schwelle. Rot hat das Problem nicht, dort
# liegt S bei 200 und mehr.
#
# SEIT DIE ZONE SITZT darf v_min noch viel tiefer. Solange der Abgriff quer
# ueber Bande, Matte und Wand lief, war v_min die einzige Bremse gegen dunkles
# Rauschen. Mit der kalibrierten Zone (siehe README) wird nur noch die Bande
# angeschnitten -- und die hat, ueber die Zone gemittelt, eine Saettigung von
# glatt NULL. Eine Pylone davor kommt im selben Mass auf 90 bis 113. s_min
# allein trennt das also sauber, und v_min darf so tief, dass auch eine Pylone
# im Schatten (dort gemessen: V=15) noch durchkommt.
#
# Und die Helligkeit ist nicht stabil: dieselbe Pylone, vier Minuten spaeter,
# hatte einen V-Median von 32 statt 44 -- Wolken reichen dafuer. Der Farbton
# blieb bei 51, die Saettigung stieg sogar auf 136. V ist also der wackelige
# Kanal, S der belastbare. Deshalb steht v_min bewusst tief bei 20: gemessen
# 18 der 21 Pylonenpunkte bei NULL Fehltreffern im ganzen Scan. Die drei
# fehlenden scheitern nicht an V, sondern am Farbton (H 27, 39, 39 -- knapp
# unter dem Bereich 40..90).
#
# Dass v_min so tief gefahrlos ist, liegt allein an s_min: die Pylonenpunkte
# haben S zwischen 114 und 165, alles Stoerende scheitert vorher an der
# Saettigung. v_min ist hier kein Rauschfilter, das ist s_min.
#
# s_min ist der eigentliche Rauschfilter, deshalb faellt es nicht beliebig
# tief. Die Stoerungen im Scan sind naemlich nicht dunkel, sondern HELL
# (Waende und Decke bei 2 bis 3 m, V ueber 200) mit leichtem Farbstich und
# geringer Saettigung -- dagegen hilft nur s_min.
#
# Wie tief es darf, an vier gruenen Pylonen in 0.70 bis 2.08 m gemessen
# (162 Punkte Ground Truth, v_min 20):
#     s_min 80 ->  64 erkannt,  0 Fehltreffer
#     s_min 70 ->  83 erkannt,  0 Fehltreffer
#     s_min 65 ->  92 erkannt,  1 Fehltreffer
#     s_min 60 ->  93 erkannt,  3 Fehltreffer
#     s_min 40 -> 102 erkannt, 29 Fehltreffer   <- Knick, ab hier unbrauchbar
# Diese Messung galt noch fuer den alten Hue-Bereich (40, 90). Seit der oben
# auf (35, 85) eingeengt ist, fallen die ausgebrannten Waende schon am Farbton
# raus und s_min darf auf 50 -- das holt rund 15 Prozent mehr Pylonenpixel bei
# 4 statt 63 Prozent Wandkontamination.
#
# Die restlichen zwei Pylonenpunkte scheitern an s_min -- es sind ausgerechnet
# die hellsten (V 82 und 106) am ueberstrahlten Rand, wo Ueberbelichtung die
# Saettigung frisst (S faellt dort auf 51 und 65). Das ist ein Belichtungs-,
# kein Schwellenproblem.
#
# ROT ist bewusst konservativer als gruen. OpenCV liefert fuer entsaettigte
# Pixel H=0, und das faellt genau in den Rot-Bereich -- dunkles Grau kann also
# als Rot durchgehen. Gemessen stieg die Zahl roter Punkte von 34 (v_min 25)
# auf 78 (v_min 15), ohne dass mehr rote Flaeche da war. Deshalb v_min 25 und
# das strengere s_min 110. Bei gruen gibt es diesen Effekt nicht.
#
# ROT GEGEN MAGENTA. Der obere Rot-Ast hiess frueher (170, 179) und hat damit
# eine magenta Wand eingefangen: die misst H 172 bis 176 (Median 174) bei
# S 114 bis 184, lag also mittendrin. 130 Wandpunkte wurden rot.
#
# Echtes Rot liegt weit davon entfernt. An zwei roten Pylonen gemessen:
# H 0 bis 7, Median 4 bis 5. Zwischen H 23 und H 160 ist im ganzen Scan
# ueberhaupt nichts -- die Luecke ist also breit und die Trennung eindeutig:
#     H   0.. 7  ->  59 Pylonenpunkte,   0 Wandpunkte
#     H 172..175 ->   0 Pylonenpunkte, 129 Wandpunkte
# Deshalb reicht der obere Ast nur noch von 177 bis 179 und magenta bis 176.
# Ergebnis: rot 140 -> 4 Fehltreffer, magenta 15 -> 150 erkannte Wandpunkte.
#
# Der untere Ast bleibt bei 10, obwohl alle Pylonenpunkte unter 8 liegen --
# das sind drei Stufen Reserve gegen Weissabgleich-Drift und kostet nur 6
# zusaetzliche Fehltreffer (Holztoene ab H 8, siehe Tisch im Testaufbau).
#
# Die Reihenfolge im Dict ist nicht egal: classify_hsv laeuft sie der Reihe
# nach durch und ueberschreibt, bei Ueberlappung gewinnt also der letzte
# Eintrag (magenta). Aktuell ueberlappen die Bereiche nicht.
DEFAULT_RANGES = {
    'rot':     {'hue': [(0, 10), (177, 179)], 's_min': 110, 'v_min': 12},
    'gruen':   {'hue': [(40, 72)],            's_min': 50,  'v_min': 12},
    'magenta': {'hue': [(140, 176)],          's_min': 90,  'v_min': 15},
}

# BGR-Farben fuer Debug-Overlays
LABEL_BGR = {
    'rot': (0, 0, 255),
    'gruen': (0, 220, 0),
    'magenta': (200, 0, 200),
    'schwarz': (60, 60, 60),
    'unbekannt': (180, 180, 180),
}

# Kraeftige BGR-Farben fuer die PointCloud (/camera_lidar/colored_scan).
# LABEL_BGR zeichnet im Debug-Bild nur einen duennen Rand um einen Punkt, der
# innen die gemessene Farbe behaelt -- da darf es dezent sein. Hier wird der
# ganze Punkt eingefaerbt, also sind die erkannten Farben voll ausgesteuert und
# alles Unklassifizierte bewusst dunkelgrau: rot und gruen sollen im 3D-Panel
# sofort ins Auge springen. Die Werte sind exakt, ein Konsument kann also auf
# 0x0000FF / 0x00FF00 / 0xFF00FF pruefen statt Farbbereiche zu raten.
CLOUD_BGR = {
    'rot': (0, 0, 255),
    'gruen': (0, 255, 0),
    'magenta': (255, 0, 255),
    'schwarz': (45, 45, 45),
    'unbekannt': (85, 85, 85),
}


def label_colors(labels, palette: dict = None) -> np.ndarray:
    """Labels -> (N,3)-BGR-uint8 in den kraeftigen Farben aus CLOUD_BGR."""
    palette = palette or CLOUD_BGR
    fallback = palette.get('unbekannt', (85, 85, 85))
    return np.array([palette.get(l, fallback) for l in labels],
                    dtype=np.uint8).reshape(-1, 3)


def classify_hsv(hsv: np.ndarray, ranges: dict = None, black_v_max: int = 45) -> list:
    """Klassifiziert ein (N,3)-HSV-Array zu Labels wie 'rot'/'gruen'/'unbekannt'."""
    ranges = ranges or DEFAULT_RANGES
    hsv = np.asarray(hsv).reshape(-1, 3).astype(np.int16)
    h, s, v = hsv[:, 0], hsv[:, 1], hsv[:, 2]

    labels = np.full(hsv.shape[0], 'unbekannt', dtype=object)
    labels[v <= black_v_max] = 'schwarz'

    for name, spec in ranges.items():
        hit = np.zeros(hsv.shape[0], dtype=bool)
        for lo, hi in spec['hue']:
            hit |= (h >= lo) & (h <= hi)
        hit &= (s >= spec['s_min']) & (v >= spec['v_min'])
        labels[hit] = name
    return labels.tolist()


def sample_colors(image_bgr: np.ndarray, u: np.ndarray, v: np.ndarray, patch: int = 5,
                  center=None, band_px=None, band_count: int = 5):
    """Liest an den Pixeln (u,v) Farbe aus. Gibt (bgr, hsv) als (N,3)-uint8 zurueck.

    Vorher wird das ganze Bild einmal median-gefiltert -- das ist deutlich
    schneller als pro Punkt ein Patch auszuschneiden und faengt Glanzlichter
    und Rauschen genauso weg.

    Sind ``center`` (cx, cy) und ``band_px`` gesetzt, wird nicht ein einzelnes
    Pixel gelesen, sondern ``band_count`` Stuetzstellen entlang der RADIALEN
    Linie durch (u,v) -- und davon der Median genommen. Radial nach aussen heisst
    im Fisheye "nach unten", die Linie liegt also laengs der Pylone. Der Median
    (nicht der Mittelwert) haelt das Ergebnis stabil, wenn ein Ende des Bandes
    ueber die Pylonenkante hinausrutscht.
    """
    if patch > 1:
        smooth = cv2.medianBlur(image_bgr, patch if patch % 2 else patch + 1)
    else:
        smooth = image_bgr

    height, width = image_bgr.shape[:2]
    u = np.asarray(u, dtype=float)
    v = np.asarray(v, dtype=float)

    if center is None or band_px is None or band_count < 2:
        offsets = np.zeros((1, u.size))
        dir_u = dir_v = np.zeros(u.size)
    else:
        cx, cy = center
        dir_u, dir_v = u - cx, v - cy
        radius = np.hypot(dir_u, dir_v)
        safe = np.where(radius > 1e-6, radius, 1.0)
        dir_u, dir_v = dir_u / safe, dir_v / safe
        steps = np.linspace(-1.0, 1.0, int(band_count))
        offsets = steps[:, None] * np.broadcast_to(
            np.asarray(band_px, dtype=float), u.shape)[None, :]

    stack = np.empty((offsets.shape[0], u.size, 3), dtype=np.uint8)
    for k in range(offsets.shape[0]):
        ui = np.clip(np.rint(u + offsets[k] * dir_u).astype(int), 0, width - 1)
        vi = np.clip(np.rint(v + offsets[k] * dir_v).astype(int), 0, height - 1)
        stack[k] = smooth[vi, ui]

    bgr = np.median(stack, axis=0).astype(np.uint8)
    hsv = cv2.cvtColor(bgr.reshape(-1, 1, 3), cv2.COLOR_BGR2HSV).reshape(-1, 3)
    return bgr, hsv


def rg_kennzahl(stack: np.ndarray):
    """(G-R)/max(B,G,R) je Pixel, dazu max(B,G,R).

    Positiv heisst gruenlich, negativ roetlich, um null herum farblos.

    Warum nicht ueber den Farbton: am Aufbau gemessen (Rohbild + CSV, beide
    Pylonen auf 0.85 m) liegt die ROTE Pylone bei H 5..9 mit S 156..219 --
    bilderbuchmaessig. Die GRUENE dagegen bei H 33..67 mit S nur 64..133, also
    quer ueber die untere Fenstergrenze (H=40) und dicht an der oberen (H=72).
    Das ist kein Zufall: der Farbton wird bei niedriger Saettigung numerisch
    instabil, und genau dort lebt gruen. Deshalb verliert ein Farbton-Fenster
    gruen reihenweise und nimmt dafuer die dunkle Bande mit.

    Das Verhaeltnis von Gruen- zu Rotkanal ist dagegen eindeutig getrennt
    (Median, 5..95 Perzentil, dieselbe Messung):

        rote Pylone     -0.60   (-0.63 .. -0.48)
        gruene Pylone   +0.44   (+0.17 .. +0.52)
        Holz und Moebel -0.08   (-0.16 .. +0.01)
        Bande und Rest   0.00   (-0.07 .. +0.16)

    Die Normierung auf den hellsten Kanal macht die Kennzahl unabhaengig von
    Helligkeit und Belichtung -- eine im Schatten stehende Pylone hat dieselbe
    Kennzahl wie eine in der Sonne, nur mit mehr Rauschen.
    """
    b = stack[..., 0].astype(np.int16)
    g = stack[..., 1].astype(np.int16)
    r = stack[..., 2].astype(np.int16)
    mx = np.maximum(np.maximum(b, g), r)
    return (g - r) / np.maximum(mx, 1).astype(np.float32), mx, (g - r)


def classify_zone(image_bgr: np.ndarray, phi: np.ndarray, r_innen: np.ndarray,
                  r_aussen: np.ndarray, center, min_frac: float = 0.20,
                  ranges: dict = None, steps: int = 13, black_v_max: int = 45,
                  nutz_anteil: float = 1.0, adaptiv_faktor: float = 0.0,
                  adaptiv_grad: float = 20.0,
                  rg_z_min: float = 0.15, rg_s_min: int = 60,
                  rg_d_min: int = 20):
    """Farbe je Punkt per Abstimmung ueber ein radiales Segment.

    Das Segment ist NICHT konstant breit, sondern wird je Punkt aus zwei
    Hoehen berechnet und in Bildradien uebergeben (``r_innen`` = obere Kante,
    ``r_aussen`` = untere Kante; radial nach aussen heisst im Fisheye "nach
    unten"). Genau das ist der Punkt: eine Bande fester Hoehe erscheint im
    Fisheye nicht als Kreisband konstanter Dicke.

    Sitzt das Objektiv auf Hoehe der Bandenoberkante, dann ist fuer die
    Oberkante die Hoehendifferenz null, theta also exakt 90 Grad und der
    Bildradius konstant -- die Oberkante laeuft als gerade Linie. Die
    Unterkante liegt die Bandenhoehe tiefer und wandert mit der Entfernung
    nach oben, weil theta sich von unten an 90 Grad annaehert:

        Bandenhoehe 9 cm, f=262 px/rad:
        0.3 m -> Unterkante bei r=489   (Zone 77 px dick)
        1.0 m -> r=436                  (Zone 24 px)
        3.0 m -> r=420                  (Zone  8 px)

    Eine konstante Pixelbreite ist deshalb nah viel zu schmal und fern zu
    breit -- fern ragt sie ueber die Bande hinaus und sammelt die helle Wand
    dahinter mit ein, was die Punkte faelschlich auf "unbekannt" zieht.

    Abgestimmt statt gemittelt: gezaehlt wird, welcher Anteil der Pixel im
    Segment zu welcher Farbe passt; ab ``min_frac`` gewinnt eine Farbe. Ein
    Median ueber ein Segment, das halb auf der Pylone und halb auf der Wand
    liegt, ergaebe dagegen Mischmasch. (Gegenprobe am Aufbau: nimmt man statt
    der Abstimmung das gesaettigtste Pixel, findet man in fast jeder Linie
    irgendwas und erzeugt Cluster von 30 Grad Breite, wo eine Pylone 5 Grad
    haette.)

    Rueckgabe: ``(labels, bgr, hsv)``. Die Farbe ist der Median der Pixel, die
    fuer das Gewinnerlabel gestimmt haben (sonst der Median des ganzen
    Segments), damit CSV und der raw-Modus der PointCloud etwas Sinnvolles
    zeigen.
    """
    ranges = ranges or DEFAULT_RANGES
    height, width = image_bgr.shape[:2]
    cx, cy = center
    phi = np.asarray(phi, dtype=float)
    r_innen = np.asarray(r_innen, dtype=float)
    r_aussen = np.asarray(r_aussen, dtype=float)
    steps = max(int(steps), 2)

    cos_p, sin_p = np.cos(phi), np.sin(phi)
    # Nur den mittleren Teil der Zone abtasten. Sitzen die Zonengrenzen sauber,
    # ist die Mitte die beste Stelle: maximaler Abstand zur hellen Matte unten
    # und zur Wand oben. Die Raender tragen dann nur noch Mischpixel bei.
    # 1.0 = ganze Zone, 0.33 = mittleres Drittel. Ganz auf eine Linie zu gehen
    # ist allerdings riskant -- dann haengt alles daran, dass die Zone auf ein
    # paar Pixel genau sitzt, und genau das war vorher das Problem.
    nutz = min(max(float(nutz_anteil), 0.02), 1.0)
    rand = (1.0 - nutz) / 2.0
    anteile = np.linspace(rand, 1.0 - rand, steps)

    stack = np.empty((steps, phi.size, 3), dtype=np.uint8)
    for k, t in enumerate(anteile):
        r = r_innen + t * (r_aussen - r_innen)
        ui = np.clip(np.rint(cx + r * cos_p).astype(int), 0, width - 1)
        vi = np.clip(np.rint(cy + r * sin_p).astype(int), 0, height - 1)
        stack[k] = image_bgr[vi, ui]

    hsv_stack = cv2.cvtColor(stack.reshape(-1, 1, 3),
                             cv2.COLOR_BGR2HSV).reshape(stack.shape)
    hh = hsv_stack[..., 0].astype(np.int16)
    ss = hsv_stack[..., 1].astype(np.int16)
    vv = hsv_stack[..., 2].astype(np.int16)

    # --- Saettigungsschwelle: absolut oder relativ zur Umgebung -------- #
    # Absolute Schwellen scheitern an dunklen Pylonen: am Aufbau hatte eine im
    # Schatten S=66 bei V=15, die schwarze Bande daneben S=28 bei V=17 -- in
    # BEIDEN Kanaelen ueberlappend, also mit keiner festen Schwelle trennbar.
    # Im Verhaeltnis ist die Sache dagegen eindeutig: die Pylone ist 2.4- bis
    # 2.9-mal so gesaettigt wie die Bande neben ihr, und das gilt im Schatten
    # wie in der Sonne. Deshalb kann die Schwelle mitwandern.
    #
    # Der Hintergrund ist der gleitende Median der Saettigung ueber ein
    # Azimutfenster. Es muss deutlich breiter sein als eine Pylone, sonst
    # hebt sie ihre eigene Schwelle an: bei 0.8 m ist eine Pylone rund 7 Grad
    # breit, mit 20 Grad Fenster macht sie also gut ein Sechstel aus und der
    # Median bleibt fest bei der Bande.
    schwellen = {name: float(spec['s_min']) for name, spec in ranges.items()}
    if adaptiv_faktor > 0.0 and phi.size >= 16:
        sat_pkt = np.median(ss, axis=0)
        ordnung = np.argsort(phi)
        sortiert = sat_pkt[ordnung]
        breite = max(int(round(phi.size * adaptiv_grad / 360.0)), 3)
        if breite % 2 == 0:
            breite += 1
        halb = breite // 2
        # zyklisch, der Azimut laeuft rundum
        lang = np.concatenate([sortiert[-halb:], sortiert, sortiert[:halb]])
        grund_sortiert = np.median(sliding_window_view(lang, breite), axis=1)
        grund = np.empty_like(grund_sortiert)
        grund[ordnung] = grund_sortiert
        for name, spec in ranges.items():
            # Absolute Untergrenze bleibt als Rauschsperre bestehen, sie ist
            # aber unkritisch, weil die relative Schwelle meist hoeher liegt.
            schwellen[name] = np.maximum(grund * adaptiv_faktor,
                                         float(spec['s_min']) * 0.5)

    zz, _, dd = rg_kennzahl(stack)

    labels = np.full(phi.size, 'unbekannt', dtype=object)
    best = np.full(phi.size, float(min_frac) - 1e-9)
    hits = {}
    for name, spec in ranges.items():
        if rg_z_min > 0.0 and name in ('rot', 'gruen'):
            # Rot und Gruen ueber das Kanalverhaeltnis (siehe rg_kennzahl).
            hit = (zz >= rg_z_min) if name == 'gruen' else (zz <= -rg_z_min)
            hit &= ss >= rg_s_min
            # ABSOLUTES Tor. Ohne das reicht ein Farbstich: ein dunkles,
            # fast neutrales Bandenpixel BGR(30,35,25) hat S=73 und z=+0.29 --
            # beide relativen Tore offen, obwohl der Kanalunterschied nur 10
            # Zaehlwerte betraegt. Am Aufbau gemessen liegt die Bande bei
            # |G-R| = 0 (5..95 Perzentil -2..+9), die gruene Pylone bei 38,
            # die rote bei 110.
            hit &= np.abs(dd) >= rg_d_min
        else:
            # magenta (Parkzone) bleibt auf dem Farbton-Weg: dort ist der
            # Farbton eindeutig und es gibt keine Messreihe fuer eine bessere
            # Kennzahl.
            hit = np.zeros(hh.shape, dtype=bool)
            for lo, hi in spec['hue']:
                hit |= (hh >= lo) & (hh <= hi)
            hit &= (ss >= np.asarray(schwellen[name])) & (vv >= spec['v_min'])
        hits[name] = hit
        frac = hit.mean(0)
        take = frac > best
        labels[take] = name
        best[take] = frac[take]

    # Keine Farbe hat die Mehrheit: schwarz, wenn das Segment ueberwiegend
    # dunkel ist (Bande, Schatten), sonst unbekannt.
    offen = best < float(min_frac)
    labels[offen & ((vv <= black_v_max).mean(0) >= 0.5)] = 'schwarz'

    gewinner = np.zeros(hh.shape, dtype=bool)
    for name, hit in hits.items():
        gewinner |= hit & (labels == name)[None, :]

    arr = stack.astype(float)
    med = np.median(arr, axis=0)
    hat = gewinner.any(0)
    if hat.any():
        maskiert = np.where(gewinner[..., None], arr, np.nan)
        med[hat] = np.nanmedian(maskiert[:, hat], axis=0)
    bgr = med.astype(np.uint8)
    hsv = cv2.cvtColor(bgr.reshape(-1, 1, 3), cv2.COLOR_BGR2HSV).reshape(-1, 3)
    return labels.tolist(), bgr, hsv


def find_color_blob(image_bgr: np.ndarray, ranges: dict = None, min_area: int = 300,
                    mask_circle=None, only_label: str = '', max_area: int = 0):
    """Sucht den groessten rot/gruen/magenta-Blob im Bild.

    ``mask_circle`` ist optional (cx, cy, radius) und blendet alles ausserhalb
    des Fisheye-Bildkreises aus.

    Rueckgabe: (u, v, label, area, r_innen, r_aussen) oder None. Die beiden
    Radien sind der kleinste und groesste Abstand der Blob-Kontur zum
    Bildkreismittelpunkt. Bei einer stehenden Pylone entspricht ``r_aussen``
    dem Fusspunkt auf der Matte und ``r_innen`` der Oberkante -- radial nach
    aussen heisst im Fisheye ja "nach unten". Daraus kalibriert
    ``rotation_calibration`` die Brennweite.
    """
    ranges = ranges or DEFAULT_RANGES
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)

    roi = None
    if mask_circle is not None:
        cx, cy, radius = mask_circle
        roi = np.zeros(image_bgr.shape[:2], np.uint8)
        cv2.circle(roi, (int(round(cx)), int(round(cy))), int(round(radius)), 255, -1)

    best = None
    for name, spec in ranges.items():
        # Auf eine Farbe festnageln, wenn gewuenscht -- sonst gewinnt der
        # groesste Fleck im Bild, und das ist oft irgendein Gegenstand im Raum
        # statt der Kalibrierpylone.
        if only_label and name != only_label:
            continue
        mask = np.zeros(image_bgr.shape[:2], np.uint8)
        for lo, hi in spec['hue']:
            mask |= cv2.inRange(
                hsv,
                np.array([lo, spec['s_min'], spec['v_min']], np.uint8),
                np.array([hi, 255, 255], np.uint8),
            )
        if roi is not None:
            mask = cv2.bitwise_and(mask, roi)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_area or (max_area > 0 and area > max_area):
                continue
            if best and area <= best[3]:
                continue
            moments = cv2.moments(contour)
            if moments['m00'] <= 0:
                continue

            if mask_circle is not None:
                points = contour.reshape(-1, 2).astype(float)
                radii = np.hypot(points[:, 0] - mask_circle[0], points[:, 1] - mask_circle[1])
                r_innen, r_aussen = float(radii.min()), float(radii.max())
            else:
                r_innen = r_aussen = float('nan')

            best = (moments['m10'] / moments['m00'], moments['m01'] / moments['m00'],
                    name, area, r_innen, r_aussen)
    return best


def ranges_from_params(node, prefix: str = 'color') -> dict:
    """Baut DEFAULT_RANGES aus ROS-Parametern, damit die Schwellen live passen."""
    ranges = {}
    for name, spec in DEFAULT_RANGES.items():
        flat = [bound for pair in spec['hue'] for bound in pair]
        hue = node.declare_parameter(f'{prefix}.{name}.hue', flat).value
        s_min = node.declare_parameter(f'{prefix}.{name}.s_min', spec['s_min']).value
        v_min = node.declare_parameter(f'{prefix}.{name}.v_min', spec['v_min']).value
        ranges[name] = {
            'hue': [(int(hue[i]), int(hue[i + 1])) for i in range(0, len(hue) - 1, 2)],
            's_min': int(s_min),
            'v_min': int(v_min),
        }
    return ranges
