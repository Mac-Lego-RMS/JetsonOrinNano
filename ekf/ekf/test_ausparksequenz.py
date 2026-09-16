#!/usr/bin/env python3
"""Ablauftest der Ausparksequenz -- ohne ROS-Kern, ohne Roboter.

    python3 -m ekf.test_ausparksequenz

Borgt sich die Methoden von Round1Controller und faehrt sie mit Attrappen fuer
Uhr, Logger und Publisher. Geprueft wird der ABLAUF: erst lenken, dann fahren,
auf die Quittung warten, Zug fuer Zug -- und was passiert, wenn etwas schief
geht. Genau das soll man nicht am Roboter herausfinden.
"""
import types

from ekf.round1_controller_node import Round1Controller
from ekf import ausparken as A


def pruefe(name, bedingung, zusatz=''):
    if not bedingung:
        raise AssertionError('FEHLGESCHLAGEN: %s %s' % (name, zusatz))
    print('  ok  %s%s' % (name, ('  ' + zusatz) if zusatz else ''))


class Sammler:
    def __init__(self):
        self.werte = []

    def publish(self, msg):
        self.werte.append(msg)


class Logbuch:
    def __init__(self):
        self.zeilen = []

    def _an(self, stufe):
        return lambda text, **kw: self.zeilen.append((stufe, text))

    def __getattr__(self, name):
        return self._an(name)

    def text(self):
        return '\n'.join(t for _s, t in self.zeilen)

    def stufen(self, stufe):
        return [t for s, t in self.zeilen if s == stufe]


class Attrappe:
    _ausparken_schritt = Round1Controller._ausparken_schritt
    _ausparken_planen = Round1Controller._ausparken_planen
    _ausparken_fertig = Round1Controller._ausparken_fertig
    _ausparken_pid = Round1Controller._ausparken_pid
    _ausparken_abbruch = Round1Controller._ausparken_abbruch
    ausparken_move_done_cb = Round1Controller.ausparken_move_done_cb

    def __init__(self, **kw):
        self.t = 100.0
        self.state = 'AUSPARK_BUTTON'
        self.require_button = False
        self.button_pressed = False
        self.nur_ausparken = False
        self.ausparken_richtung_invertieren = False
        self.ausparken_schritte = list(A.SCHRITTE_STANDARD)
        self.ausparken_pid = [4.0, 300.0]
        self.ausparken_pid_nachher = [4.0, 1023.0]
        self.ausparken_scans = 5
        self.ausparken_sektor_grad = 20.0
        self.ausparken_richtung_timeout = 8.0
        self.ausparken_lenk_wartezeit = 0.6
        self.ausparken_zug_timeout = 15.0
        self.ausp_stimmen = ['CW'] * 5
        self.ausp_letzter_grund = 'links 0.14 m, rechts 0.87 m'
        self.ausp_schritte = None
        self.ausp_richtung = None
        self.ausp_index = 0
        self.ausp_phase = 'lenken'
        self.ausp_lenk_gesendet = False
        self.ausp_t0 = 0.0
        self.ausp_gesendet_t = None
        self.ausp_move_done = None
        self.ausp_theta0 = 0.0
        self.ausp_pose0 = None
        self.pub_steer = Sammler()
        self.pub_move = Sammler()
        self.pub_pid = Sammler()
        self.pub_motor = Sammler()
        self.stopps = 0
        self._log = Logbuch()
        self.__dict__.update(kw)

    def now_s(self):
        return self.t

    def get_logger(self):
        return self._log

    def publish_stop(self):
        self.stopps += 1

    # --- Hilfen fuer die Tests ---
    def takt(self, n=1, dt=0.1):
        for _ in range(n):
            self._ausparken_schritt(0.0, 0.0, 0.0)
            self.t += dt

    def quittiere(self, status=0, pos_zehntelgrad=0):
        self.ausparken_move_done_cb(
            types.SimpleNamespace(data=[1, status, pos_zehntelgrad]))

    def durchfahren(self, status=0, grenze=4000):
        for _ in range(grenze):
            vorher = len(self.pub_move.werte)
            self._ausparken_schritt(0.0, 0.0, 0.0)
            self.t += 0.1
            if len(self.pub_move.werte) > vorher:
                self.t += 0.5
                self.quittiere(status)
            if self.state in ('WAIT_INPUTS', 'DONE'):
                return
        raise AssertionError('Sequenz haengt -- kein Ende nach %d Takten' % grenze)


print('Taster')
f = Attrappe(require_button=True)
f.takt(5)
pruefe('ohne Taster bleibt er stehen',
       f.state == 'AUSPARK_BUTTON' and f.stopps == 5 and not f.pub_move.werte)
f.button_pressed = True
f.takt(1)
pruefe('mit Taster geht es zur Richtungssuche', f.state == 'AUSPARK_RICHTUNG')

print('\nRichtungssuche')
f = Attrappe(ausp_stimmen=[])
f.takt(1)                                  # AUSPARK_BUTTON -> AUSPARK_RICHTUNG
f.takt(10)
pruefe('ohne Stimmen wird nicht gefahren',
       f.state == 'AUSPARK_RICHTUNG' and not f.pub_move.werte)
f.t += 20.0
f.takt(1)
pruefe('Zeitueberschreitung bricht ab', f.state == 'DONE')
pruefe('Abbruch stoppt den Motor aktiv',
       len(f.pub_motor.werte) == 1 and f.pub_motor.werte[0].data == 0)
pruefe('Abbruch stellt die Regelparameter zurueck',
       [list(m.data) for m in f.pub_pid.werte] == [[4.0, 1023.0]])
pruefe('Abbruch wird als Fehler protokolliert', f._log.stufen('error'))

print('\nPlanung')
f = Attrappe()
f.takt(2)
pruefe('CW spiegelt die Tabelle nach rechts',
       f.ausp_schritte[0][0] < 0 and A.SCHRITTE_STANDARD[0] > 0,
       'erster Zug %+.0f %%' % f.ausp_schritte[0][0])
pruefe('Strecken bleiben unveraendert',
       [cm for _l, cm in f.ausp_schritte] == list(A.SCHRITTE_STANDARD[1::2]))
pruefe('Regelparameter werden vor der Sequenz gesetzt',
       [list(m.data) for m in f.pub_pid.werte] == [[4.0, 300.0]])
pruefe('Trockenlauf steht im Log', 'Trockenlauf' in f._log.text())

f2 = Attrappe(ausp_stimmen=['CCW'] * 5)
f2.takt(2)
pruefe('CCW spiegelt nicht', f2.ausp_schritte[0][0] > 0,
       'erster Zug %+.0f %%' % f2.ausp_schritte[0][0])
f3 = Attrappe(ausparken_richtung_invertieren=True)
f3.takt(2)
pruefe('Invertierschalter dreht die Seite',
       f3.ausp_schritte[0][0] * f.ausp_schritte[0][0] < 0)

print('\nEin einzelner Zug')
f = Attrappe()
f.takt(2)                                   # geplant, jetzt AUSPARK_FAHREN
f.stopps = 0            # der Halt aus der Richtungssuche zaehlt hier nicht mit
lenk_soll, cm_soll = f.ausp_schritte[0]
f.takt(1)
pruefe('zuerst wird gelenkt, noch nicht gefahren',
       len(f.pub_steer.werte) == 1 and not f.pub_move.werte)
pruefe('Lenkwert stimmt',
       abs(f.pub_steer.werte[0].data - lenk_soll) < 1e-6)
f.takt(2)
pruefe('waehrend der Wartezeit wird nicht gefahren', not f.pub_move.werte)
f.t += f.ausparken_lenk_wartezeit
f.takt(1)
pruefe('nach der Wartezeit geht die Fahrt raus', len(f.pub_move.werte) == 1)
pruefe('Fahrstrecke in Encodergrad',
       abs(f.pub_move.werte[0].data - A.cm_zu_grad(cm_soll)) < 1e-3,
       '%.0f grad fuer %.1f cm' % (f.pub_move.werte[0].data, cm_soll))
# Waehrend einer Positionsfahrt darf KEIN /cmd_vel rausgehen: die Bruecke
# wuerde daraufhin die Lenkung neu stellen.
pruefe('kein /cmd_vel waehrend der Fahrt', f.stopps == 0)
f.takt(5)
pruefe('ohne Quittung geht es nicht weiter',
       f.ausp_index == 0 and len(f.pub_move.werte) == 1)

print('\nQuittungen')
f_alt = Attrappe()
f_alt.takt(2)
f_alt.ausp_move_done = (f_alt.t - 50.0, 1, 0, 0.0)     # Quittung von VORHER
f_alt.t += f_alt.ausparken_lenk_wartezeit
f_alt.takt(3)
pruefe('alte Quittung zaehlt nicht', f_alt.ausp_index == 0)

f = Attrappe()
f.takt(2)
f.stopps = 0
f.durchfahren()
pruefe('ueber die ganze Sequenz nur der Schlusshalt', f.stopps == 1,
       '%d Halte' % f.stopps)
pruefe('vollstaendige Sequenz laeuft durch',
       len(f.pub_move.werte) == len(f.ausp_schritte),
       '%d Fahrten' % len(f.pub_move.werte))
pruefe('jede Fahrt hat ihre Lenkung davor',
       len(f.pub_steer.werte) == len(f.ausp_schritte))
pruefe('danach weiter zum Rennen', f.state == 'WAIT_INPUTS')
pruefe('Taster gilt als gedrueckt', f.button_pressed is True)
pruefe('Regelparameter am Ende zurueckgestellt',
       list(f.pub_pid.werte[-1].data) == [4.0, 1023.0])

f = Attrappe(nur_ausparken=True)
f.durchfahren()
pruefe('nur_ausparken haelt an', f.state == 'DONE')
pruefe('nur_ausparken faehrt trotzdem die ganze Folge',
       len(f.pub_move.werte) == len(f.ausp_schritte))

f = Attrappe()
f.durchfahren(status=1)                      # 1 = Zeitueberschreitung im ESP
pruefe('schlechter Status bricht ab', f.state == 'DONE')
pruefe('Abbruch nach dem ERSTEN schlechten Zug', len(f.pub_move.werte) == 1)

print('\nHaenger')
f = Attrappe()
f.takt(2)                                    # geplant
f.takt(1)                                    # Lenkbefehl raus, Uhr laeuft ab hier
f.t += f.ausparken_lenk_wartezeit
f.takt(1)                                    # Fahrbefehl raus
assert f.ausp_phase == 'fahren', 'Aufbau des Tests stimmt nicht'
f.t += f.ausparken_zug_timeout + 1.0
f.takt(1)
pruefe('ausbleibende Quittung bricht ab', f.state == 'DONE')
pruefe('Fehlermeldung nennt die Bruecke',
       'esp_serial_bridge' in f._log.text())

print('\nSchlechte Eingaben')
f = Attrappe(ausparken_schritte=[100.0, 5.0, -100.0])      # ungerade
f.takt(2)
pruefe('ungerade Schrittliste bricht sauber ab', f.state == 'DONE')
f = Attrappe(ausparken_schritte=[])
f.takt(2)
pruefe('leere Schrittliste bricht sauber ab', f.state == 'DONE')
f = Attrappe(ausparken_pid=[4.0])                          # ungerade
f.takt(2)
pruefe('ungerade PID-Liste wird nur gemeldet, nicht gefahren',
       f.state == 'AUSPARK_FAHREN' and f._log.stufen('warn'))

print('\nalle Tests bestanden')
