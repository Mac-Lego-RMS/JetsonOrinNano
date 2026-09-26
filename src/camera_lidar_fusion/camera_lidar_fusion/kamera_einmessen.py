#!/usr/bin/env python3
"""Kamera vor dem Lauf auf das Licht am Feld einmessen.

Die Kamera laeuft mit fester Belichtung, festem Gain und festem Weissabgleich
(start_robot.sh). Die Werte stammen aus einem Raum mit Tageslicht; unter einer
warmen Deckenlampe war das Bild stark gelbstichig ((B-R)/max = -0,30 auf der
weissen Matte) und die Farberkennung fand keinen einzigen gruenen Punkt mehr.

Gemessen wird an der weissen MATTE, im selben Ring knapp ausserhalb der Bande,
den auch der Weisspunkt der Fusion benutzt:

  1. Helligkeit: gain so, dass die Matte bei ~205 liegt, ohne auszubrennen.
     Das zuerst -- auf ausgebrannter Matte ist kein Farbstich messbar.
  2. Weissabgleich: white_balance_temperature so, dass die Matte neutral ist,
     (B-R)/max ~ 0. Den Rest in (G-R) zieht die Fusion ohnehin je Sektor ab.
     Danach die Helligkeit noch einmal nachziehen.
     Die Belichtungszeit geht dabei NIE ueber 50 ms -- darueber faellt die
     Kamera auf 7,5 fps (siehe start_robot.sh). Ist die Matte schon bei gain 0
     zu hell, wird die Belichtung verkuerzt.
  3. Pruefung an den Pylonen: kleine, frei stehende Objekte im colored_scan,
     und wie viele ihrer Punkte eine Farbe bekommen.

Punkt 3 ist die ehrliche Antwort auf "reicht das Licht". Die Matte liegt flach
und bekommt das Deckenlicht voll ab, die Pylonen zeigen der Kamera ihre
senkrechten Seiten. Unter reinem Deckenlicht war die Matte bei V 194 bestens
belichtet und die Pylonen trotzdem fast schwarz (V 25-40) -- das kann keine
Kameraeinstellung reparieren, nur mehr Licht. Dann sagt das Skript das.

Das Ergebnis landet in /workspace/config/kamera_einmessung.env; start_robot.sh
liest es beim naechsten Start, die Einmessung ueberlebt also einen Neustart.

Aufruf (im Container, video_source und lidar_pixel_mapper laufen):
    python3 -m camera_lidar_fusion.kamera_einmessen
    python3 -m camera_lidar_fusion.kamera_einmessen --nur-pruefen
    python3 -m camera_lidar_fusion.kamera_einmessen --nicht-speichern
"""
import argparse
import datetime
import os
import subprocess
import time

import numpy as np
import rclpy
import yaml
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image, PointCloud2
from sensor_msgs_py import point_cloud2

CALIB = '/workspace/config/fisheye_calib.yaml'
AUSGABE = '/workspace/config/kamera_einmessung.env'
GERAET = '/dev/picam'

# Grenzen aus start_robot.sh: ueber 500 (50 ms) faellt die Kamera auf 7,5 fps.
BELICHTUNG_MAX = 500
BELICHTUNG_MIN = 100
GAIN_MAX = 60                  # darueber brennt das Bild aus, frisst Saettigung
WB_MIN, WB_MAX = 2800, 6500

MATTE_ZIEL = 205.0             # Median des hellsten Kanals auf der Matte
MATTE_TOL = 10.0
AUSGEBRANNT_MAX = 0.05         # Anteil Mattenpixel mit max >= 250
WB_TOL = 0.02                  # |(B-R)/max| auf der Matte

# Farben der Punktwolke, exakt (colors.CLOUD_BGR, als 0xRRGGBB gepackt)
WOLKE_ROT = 0xFF0000
WOLKE_GRUEN = 0x00FF00

# Pylon im Scan: kleines Segment, das vor seiner Umgebung steht
PYLON_BREITE = (0.02, 0.09)
PYLON_ABSTAND = (0.20, 1.60)
PYLON_VORSPRUNG = 0.10
PYLON_FARBE_OK = 0.30          # so viele Punkte muessen eine Farbe haben
# Fehlfarben erst ab hier zaehlen: die Magenta-Buchtwaende reichen bis ~0,3 m
# an den Lidar heran und kommen als rot heraus -- das filtert der
# scan_processor ohnehin (Buchtregel), mit dem Licht hat es nichts zu tun.
FEHLFARBE_AB = 0.35


def v4l2_setzen(**werte):
    args = ['v4l2-ctl', '-d', GERAET]
    for k, v in werte.items():
        args += ['-c', f'{k}={int(v)}']
    subprocess.run(args, check=True, capture_output=True)


def v4l2_lesen(*namen):
    out = subprocess.run(['v4l2-ctl', '-d', GERAET, '--get-ctrl=' + ','.join(namen)],
                         check=True, capture_output=True, text=True).stdout
    werte = {}
    for zeile in out.splitlines():
        k, _, v = zeile.partition(':')
        werte[k.strip()] = int(v.strip())
    return werte


class Einmessen(Node):
    def __init__(self):
        super().__init__('kamera_einmessen')
        with open(CALIB) as f:
            c = yaml.safe_load(f)
        self.cx, self.cy = float(c['cx']), float(c['cy'])
        # Ring wie colors.neutralpunkt in der Fusion: ab der Bandenunterkante
        # in der Ferne plus Abstand bis kurz vor den Bildkreisrand.
        self.r_min = float(c.get('zone_r0_out', 0.88 * c['radius_px'])) + 12.0
        self.r_max = float(c['radius_px']) - 15.0
        self.maske = None
        self.bild = None
        self.bild_t = 0.0
        self.wolken = []
        self.create_subscription(Image, '/video_source/raw', self._bild_cb,
                                 qos_profile_sensor_data)
        self.create_subscription(PointCloud2, '/camera_lidar/colored_scan',
                                 self._wolke_cb, 10)

    # ------------------------------------------------------------ Eingaenge
    def _bild_cb(self, msg):
        if msg.encoding != 'bgr8':
            self.get_logger().error(f'Bildformat {msg.encoding}, erwartet bgr8')
            return
        self.bild = np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.width, 3)
        self.bild_t = time.monotonic()

    def _wolke_cb(self, msg):
        if self.wolken is not None:
            self.wolken.append(msg)

    def _warte(self, sek):
        ende = time.monotonic() + sek
        while time.monotonic() < ende:
            rclpy.spin_once(self, timeout_sec=0.05)

    # --------------------------------------------------------------- Matte
    def _ring(self, form):
        if self.maske is None:
            h, w = form[:2]
            yy, xx = np.mgrid[0:h:3, 0:w:3]
            r = np.hypot(xx - self.cx, yy - self.cy)
            m = (r >= self.r_min) & (r <= self.r_max)
            self.maske = (yy[m], xx[m])
        return self.maske

    def matte(self, bilder=5, beruhigen=0.8):
        """(Helligkeit, Anteil ausgebrannt, (B-R)/max, (G-R)/max) der Matte,
        gemittelt ueber mehrere frische Bilder nach einer Stellgroessenaenderung."""
        self._warte(beruhigen)
        werte = []
        letzt = self.bild_t
        while len(werte) < bilder:
            self._warte(0.05)
            if self.bild is None or self.bild_t == letzt:
                continue
            letzt = self.bild_t
            ys, xs = self._ring(self.bild.shape)
            px = self.bild[ys, xs].astype(np.float32)
            b, g, r = px[:, 0], px[:, 1], px[:, 2]
            mx = np.maximum(np.maximum(b, g), r)
            lum = (b + g + r) / 3.0
            # Matte = die hellen, schwach gesaettigten Pixel des Rings. NICHT
            # ueber |G-R| < 30 wie colors.neutralpunkt: unter kaltem Licht ist
            # die Matte tuerkis (G weit ueber R), fiel dort komplett heraus, und
            # gemessen wurden die dunklen Reste -- Matte "44", gain auf 59.
            # Die Saettigungsgrenze haelt Magenta-Waende und Pylonen draussen,
            # laesst einen Farbstich der Matte aber durch.
            mn = np.minimum(np.minimum(b, g), r)
            blass = (mx - mn) / np.maximum(mx, 1.0) < 0.40
            if blass.sum() < 200:
                continue
            m = blass & (lum >= np.percentile(lum[blass], 55.0))
            mx_m = np.maximum(mx[m], 1.0)
            werte.append((float(np.median(mx[m])), float(np.mean(mx[m] >= 250.0)),
                          float(np.mean((b[m] - r[m]) / mx_m)),
                          float(np.mean((g[m] - r[m]) / mx_m))))
        return tuple(float(np.median([w[i] for w in werte])) for i in range(4))

    # -------------------------------------------------------- Stellgroessen
    def weissabgleich(self):
        """Bisektion auf die Farbtemperatur. Hoeher = waermer (weniger Blau):
        gemessen 4600 K -> (B-R)/max -0,30, 3000 K -> +0,07."""
        lo, hi = WB_MIN, WB_MAX
        t = v4l2_lesen('white_balance_temperature')['white_balance_temperature']
        for _ in range(8):
            v4l2_setzen(white_balance_temperature=t)
            _, _, br, _ = self.matte()
            self.get_logger().info(f'  Weissabgleich {t} K: (B-R)/max {br:+.3f}')
            if abs(br) <= WB_TOL:
                break
            if br < 0:          # zu gelb -> kaelter einstellen = kleinere Zahl
                hi = t
            else:
                lo = t
            t_neu = int(round((lo + hi) / 2.0))
            if t_neu == t:
                break
            t = t_neu
        return t

    def helligkeit(self):
        """Erst gain, und nur wenn gain 0 schon zu hell ist, die Belichtung."""
        bel = min(v4l2_lesen('exposure_time_absolute')['exposure_time_absolute'],
                  BELICHTUNG_MAX)
        v4l2_setzen(exposure_time_absolute=bel)

        def zu_hell(v, aus):
            return v > MATTE_ZIEL + MATTE_TOL or aus > AUSGEBRANNT_MAX

        lo, hi = 0, GAIN_MAX
        gain = v4l2_lesen('gain')['gain']
        v = aus = 0.0
        for _ in range(8):
            v4l2_setzen(gain=gain)
            v, aus, _, _ = self.matte()
            self.get_logger().info(f'  gain {gain}: Matte {v:.0f}, ausgebrannt {aus*100:.0f} %')
            if not zu_hell(v, aus) and v >= MATTE_ZIEL - MATTE_TOL:
                return bel, gain, v, aus
            if zu_hell(v, aus):
                hi = gain
            else:
                lo = gain
            g_neu = (lo + hi) // 2
            if g_neu == gain:
                break
            gain = g_neu

        if gain <= 1 and zu_hell(v, aus):
            # Selbst ohne Verstaerkung zu hell: Belichtung verkuerzen.
            lo_b, hi_b = BELICHTUNG_MIN, bel
            for _ in range(7):
                bel = (lo_b + hi_b) // 2
                v4l2_setzen(exposure_time_absolute=bel, gain=0)
                v, aus, _, _ = self.matte()
                self.get_logger().info(f'  Belichtung {bel/10:.0f} ms: Matte {v:.0f}, '
                                       f'ausgebrannt {aus*100:.0f} %')
                if not zu_hell(v, aus) and v >= MATTE_ZIEL - MATTE_TOL:
                    break
                if zu_hell(v, aus):
                    hi_b = bel
                else:
                    lo_b = bel
            return bel, 0, v, aus

        if not zu_hell(v, aus) and v < MATTE_ZIEL - MATTE_TOL and gain >= GAIN_MAX - 1:
            self.get_logger().warn(
                f'Zu wenig Licht: Matte bei gain {GAIN_MAX} und {BELICHTUNG_MAX/10:.0f} ms '
                f'erst {v:.0f} (Ziel {MATTE_ZIEL:.0f}). Mehr Licht ins Feld.')
        return bel, gain, v, aus

    # --------------------------------------------------------------- Pylonen
    def pylonen(self, dauer=3.0):
        """Kleine, frei stehende Objekte in der Punktwolke und ihre Farbe."""
        self.wolken = []
        self._warte(dauer)
        wolken, self.wolken = self.wolken, None
        funde = {}
        self.fehlfarbe = 0          # farbige Punkte, die zu keinem Pylon gehoeren
        for w in wolken:
            p = point_cloud2.read_points_numpy(w, field_names=('x', 'y', 'rgb'))
            if len(p) < 10:
                continue
            x, y = p[:, 0], p[:, 1]
            farbe = p[:, 2].astype(np.float32).view(np.uint32) & 0xFFFFFF
            ab = np.hypot(x, y)
            ordnung = np.argsort(np.arctan2(y, x))
            x, y, ab, farbe = x[ordnung], y[ordnung], ab[ordnung], farbe[ordnung]
            klein = np.zeros(len(x), dtype=bool)     # Segment schmal wie ein Pylon
            sprung = np.hypot(np.diff(x), np.diff(y)) > 0.05
            grenzen = np.concatenate([[0], np.nonzero(sprung)[0] + 1, [len(x)]])
            for a, e in zip(grenzen[:-1], grenzen[1:]):
                if e - a < 3:
                    continue
                breite = float(np.hypot(x[e - 1] - x[a], y[e - 1] - y[a]))
                d = float(np.median(ab[a:e]))
                if breite <= PYLON_BREITE[1] + 0.03:
                    klein[a:e] = True
                if not (PYLON_BREITE[0] <= breite <= PYLON_BREITE[1]
                        and PYLON_ABSTAND[0] <= d <= PYLON_ABSTAND[1]):
                    continue
                # Frei stehend: wenigstens auf einer Seite liegt der Hintergrund
                # deutlich weiter weg. Beide Seiten zu verlangen verlor den
                # Pylon schraeg hinten, neben dem ein Lidar-Blindsektor liegt.
                vorn = any(ab[i] > d + PYLON_VORSPRUNG
                           for i in (a - 1, e % len(x)) if 0 <= i < len(x))
                if not vorn:
                    continue
                cx, cy = float(np.mean(x[a:e])), float(np.mean(y[a:e]))
                schluessel = (round(cx / 0.08), round(cy / 0.08))
                f = funde.setdefault(schluessel, {'x': [], 'y': [], 'rot': 0,
                                                  'gruen': 0, 'n': 0, 'scans': 0})
                f['x'].append(cx)
                f['y'].append(cy)
                f['rot'] += int(np.sum(farbe[a:e] == WOLKE_ROT))
                f['gruen'] += int(np.sum(farbe[a:e] == WOLKE_GRUEN))
                f['n'] += e - a
                f['scans'] += 1
            # Farbe auf WAENDEN (breite Segmente) ist eine Fehlerkennung. Farbe
            # auf kleinen Objekten nicht -- auch wenn die Pylonsuche eins nicht
            # als frei stehend erkannt hat. Die Buchtwaende liegen vor
            # FEHLFARBE_AB.
            weg = (~klein & (ab >= FEHLFARBE_AB) & (ab <= PYLON_ABSTAND[1])
                   & ((farbe == WOLKE_ROT) | (farbe == WOLKE_GRUEN)))
            self.fehlfarbe += int(weg.sum())
        self.fehlfarbe_je_scan = self.fehlfarbe / max(len(wolken), 1)
        n_scans = max(len(wolken), 1)
        # nur, was in mindestens der Haelfte der Scans auftaucht
        return [f for f in funde.values() if f['scans'] >= 0.5 * n_scans], len(wolken)

    def gain_fuer_pylonen(self, gain_matte):
        """Gain nach den Pylonen statt nach der Matte.

        Die Matte liegt flach im Licht, die Pylonen zeigen der Kamera ihre
        senkrechten Seiten. Unter kaltem Deckenlicht am Aufbau gemessen: Matte
        auf 206 eingeregelt (gain 13) -> naher Pylon V 54, Gruenanteil z +0,35,
        aber zu dunkel, 1 % der Punkte farbig. Mit gain 59 bei ausgebrannter
        Matte: 53 %. Die Farbe ist also da, es fehlt Helligkeit -- und die
        Matte darf dafuer ausbrennen, die Farberkennung braucht sie nicht.

        Probiert einige Stufen von gain_matte bis GAIN_MAX und nimmt die
        kleinste, die fast so viele farbige Punkte bringt wie die beste.
        None, wenn keine Pylonen zu sehen sind -- dann bleibt gain_matte."""
        stufen = sorted({int(round(g)) for g in np.linspace(gain_matte, GAIN_MAX, 5)})
        ergebnis = []
        for g in stufen:
            v4l2_setzen(gain=g)
            self._warte(1.0)
            funde, _ = self.pylonen(2.0)
            if not funde:
                if g == stufen[0]:
                    return None
                continue
            anteile = [(f['rot'] + f['gruen']) / max(f['n'], 1) for f in funde]
            ergebnis.append((g, float(np.mean(anteile)), float(np.min(anteile))))
            self.get_logger().info(
                f'  gain {g}: {len(funde)} Pylon(en), im Mittel {np.mean(anteile)*100:.0f} % '
                f'farbig, schwaechster {np.min(anteile)*100:.0f} %')
        if not ergebnis:
            return None
        bestes = max(e[1] for e in ergebnis)
        g = min(e[0] for e in ergebnis if e[1] >= bestes - 0.05)
        v4l2_setzen(gain=g)
        return g


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--nur-pruefen', action='store_true',
                    help='nichts verstellen, nur Matte und Pylonen messen')
    ap.add_argument('--nicht-speichern', action='store_true',
                    help='Werte nicht nach %s schreiben' % AUSGABE)
    args = ap.parse_args()

    rclpy.init()
    node = Einmessen()
    log = node.get_logger()
    try:
        ende = time.monotonic() + 5.0
        while node.bild is None and time.monotonic() < ende:
            node._warte(0.2)
        if node.bild is None:
            log.error('Kein Bild auf /video_source/raw -- laeuft video_source?')
            return 1
        v4l2_setzen(auto_exposure=1, white_balance_automatic=0)
        vorher = v4l2_lesen('exposure_time_absolute', 'gain', 'white_balance_temperature')
        v0, aus0, br0, gr0 = node.matte()
        log.info(f'Vorher: Belichtung {vorher["exposure_time_absolute"]/10:.0f} ms, '
                 f'gain {vorher["gain"]}, {vorher["white_balance_temperature"]} K -> '
                 f'Matte {v0:.0f} ({aus0*100:.0f} % ausgebrannt), '
                 f'(B-R)/max {br0:+.3f}, (G-R)/max {gr0:+.3f}')

        if args.nur_pruefen:
            bel, gain, wb = (vorher['exposure_time_absolute'], vorher['gain'],
                             vorher['white_balance_temperature'])
            v, aus, br, gr = v0, aus0, br0, gr0
        else:
            # Erst die Helligkeit: auf ausgebrannter Matte stossen alle Kanaele
            # bei 255 an, und der Weissabgleich ist nicht messbar. Danach noch
            # einmal, weil der Weissabgleich die Kanaele verschiebt.
            log.info('Helligkeit ...')
            node.helligkeit()
            log.info('Weissabgleich ...')
            wb = node.weissabgleich()
            log.info('Helligkeit ...')
            bel, gain, v, aus = node.helligkeit()
            log.info('Gain nach den Pylonen ...')
            g_pyl = node.gain_fuer_pylonen(gain)
            if g_pyl is None:
                log.info('  keine Pylonen zu sehen -- gain bleibt nach der Matte.')
                v4l2_setzen(gain=gain)
            else:
                gain = g_pyl
            v, aus, br, gr = node.matte()
            log.info(f'Nachher: Belichtung {bel/10:.0f} ms, gain {gain}, {wb} K -> '
                     f'Matte {v:.0f} ({aus*100:.0f} % ausgebrannt), '
                     f'(B-R)/max {br:+.3f}, (G-R)/max {gr:+.3f}')

        log.info('Pylonen pruefen (3 s colored_scan) ...')
        funde, n = node.pylonen()
        schwach = 0
        if n == 0:
            log.warn('Keine /camera_lidar/colored_scan empfangen -- laeuft die Fusion?')
        elif not funde:
            log.info('Keine Pylonen im Umkreis von 1,6 m gefunden -- Farbpruefung entfaellt.')
        for f in sorted(funde, key=lambda f: np.hypot(np.mean(f['x']), np.mean(f['y']))):
            x, y = float(np.mean(f['x'])), float(np.mean(f['y']))
            anteil = (f['rot'] + f['gruen']) / max(f['n'], 1)
            farbe = ('gruen' if f['gruen'] > f['rot'] else 'rot') if anteil > 0 else '-'
            # Lidar ist gedreht montiert: Roboter-vorn = Lidar -x
            text = (f'  Pylon {np.hypot(x, y):.2f} m, {np.degrees(np.arctan2(-y, -x)):+4.0f} Grad: '
                    f'{anteil*100:3.0f} % der Punkte farbig ({farbe}; rot {f["rot"]}, '
                    f'gruen {f["gruen"]}, gesamt {f["n"]})')
            if anteil >= PYLON_FARBE_OK:
                log.info(text)
            else:
                schwach += 1
                log.warn(text + ' -- zu schwach')
        if n:
            (log.warn if node.fehlfarbe_je_scan >= 3 else log.info)(
                f'  Farbe auf Waenden (Fehlerkennung): {node.fehlfarbe_je_scan:.1f} Punkte je Scan')
        if schwach:
            log.warn(f'{schwach} Pylon(en) bekommen kaum Farbe. Liegt die Matte im Ziel, '
                     f'fehlt Licht auf den SEITEN der Pylonen (Deckenlicht allein reicht '
                     f'nicht) -- das kann keine Kameraeinstellung ausgleichen.')

        if not args.nur_pruefen and not args.nicht_speichern:
            stempel = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            with open(AUSGABE, 'w') as fh:
                fh.write(f'# kamera_einmessen, {stempel}\n'
                         f'# Matte {v:.0f} ({aus*100:.0f} % ausgebrannt), '
                         f'(B-R)/max {br:+.3f}, (G-R)/max {gr:+.3f}\n'
                         f'CAM_EXPOSURE={bel}\nCAM_GAIN={gain}\nCAM_WB_TEMP={wb}\n')
            log.info(f'Gespeichert -> {AUSGABE} (start_robot.sh liest das beim Start)')
        return 0
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    raise SystemExit(main())
