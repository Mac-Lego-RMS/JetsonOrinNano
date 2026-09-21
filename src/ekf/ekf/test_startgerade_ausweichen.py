"""Randfaelle von _start_ausweich_y -- gegen den echten Codetext aus r1c.py."""
import collections
import math
import types

quelle = open('r1c.py').read()
start = quelle.index('    def _start_ausweich_y(')
ende = quelle.index('    def publish_lap_state(')
src = '\n'.join(l[4:] if l.startswith('    ') else l
                for l in quelle[start:ende].split('\n'))
ns = {'math': math, 'OBST_GRUEN': 2, 'OBST_ROT': 1, 'BLOCK_HALB': 0.022}
exec(src, ns)

ROT, GRUEN, UNBEKANNT = 1, 2, 0


class Fake:
    _start_ausweich_y = ns['_start_ausweich_y']
    _start_ausweich_halten = ns['_start_ausweich_halten']

    def __init__(self, **kw):
        self.start_dodge = True
        self.start_dodge_look = 1.20
        self.start_dodge_back = 0.25
        self.start_dodge_lane = 0.35
        self.start_dodge_margin = 0.12
        self.start_dodge_votes = 3
        self.start_dodge_window_s = 1.0
        self.live_obs = collections.deque(maxlen=60)
        self._start_halt = None
        self.pose = (0.0, 0.0, 0.0)
        self._t = 10.0
        self.__dict__.update(kw)

    def get_clock(self):
        s = self
        return types.SimpleNamespace(
            now=lambda: types.SimpleNamespace(nanoseconds=s._t * 1e9))

    def sieh(self, mx, my, farbe, n=5):
        for i in range(n):
            self.live_obs.append((self._t - 0.05 * i, mx, my, farbe))


BREITE = 1.00      # Gasse -0.50 .. +0.50 um die Mitte

def test(name, bedingung):
    assert bedingung, 'FEHLGESCHLAGEN: ' + name
    print('  ok  ' + name)


f = Fake(); f.sieh(0.95, -0.10, GRUEN)
z, i = f._start_ausweich_y(0.0, BREITE)
test('gruen -> links vorbei, Ziel +0.211', abs(z - 0.211) < 0.001 and i[3] == 'links')

f = Fake(); f.sieh(0.95, -0.10, ROT)
z, i = f._start_ausweich_y(0.0, BREITE)
test('rot -> rechts vorbei, Ziel -0.311', abs(z - (-0.311)) < 0.001 and i[3] == 'rechts')

# 0.5*((0.30+0.022)+0.50) = 0.411, begrenzt auf 0.50-0.12 = 0.38
f = Fake(); f.sieh(0.95, +0.30, GRUEN)
z, i = f._start_ausweich_y(0.0, BREITE)
test('gruen weit aussen -> auf Wandabstand begrenzt', abs(z - 0.38) < 1e-9)

f = Fake(); f.sieh(0.95, -0.10, UNBEKANNT)
z, i = f._start_ausweich_y(0.0, BREITE)
test('Farbe unklar -> Seite mit mehr Platz (links)', z > 0 and 'unklar' in i[3])

f = Fake(); f.sieh(0.95, +0.10, UNBEKANNT)
z, i = f._start_ausweich_y(0.0, BREITE)
test('Farbe unklar, Klotz links -> rechts vorbei', z < 0 and 'unklar' in i[3])

f = Fake(); f.sieh(0.95, -0.10, GRUEN, n=2)
z, i = f._start_ausweich_y(0.0, BREITE)
test('zu wenige Sichtungen -> keine Lenkung', z == 0.0 and i is None)

f = Fake(); f.sieh(2.00, -0.10, GRUEN)
z, i = f._start_ausweich_y(0.0, BREITE)
test('zu weit voraus -> keine Lenkung', z == 0.0 and i is None)

f = Fake(); f.sieh(0.95, -0.45, GRUEN)
z, i = f._start_ausweich_y(0.0, BREITE)
test('seitlich ausserhalb der Gasse -> ignoriert', z == 0.0 and i is None)

f = Fake(); f.sieh(0.95, -0.10, GRUEN)
f._t += 2.0                                   # alle Sichtungen veraltet
z, i = f._start_ausweich_y(0.0, BREITE)
test('veraltete Sichtungen -> keine Lenkung', z == 0.0 and i is None)

f = Fake(); f.sieh(0.95, -0.10, GRUEN)
f._start_ausweich_y(0.0, BREITE)         # Halt setzen
f.live_obs.clear()
f.pose = (0.90, 0.0, 0.0)                # direkt neben dem Klotz
z, _ = f._start_ausweich_y(0.0, BREITE)
test('Halten waehrend der Vorbeifahrt', abs(z - 0.211) < 0.001)
f.pose = (1.21, 0.0, 0.0)                # 0.26 m dahinter
z, i = f._start_ausweich_y(0.0, BREITE)
test('Loslassen ab 0.25 m dahinter', z == 0.0 and i is None)

f = Fake(start_dodge=False); f.sieh(0.95, -0.10, GRUEN)
z, i = f._start_ausweich_y(0.0, BREITE)
test('Schalter aus -> unveraendert', z == 0.0 and i is None)

f = Fake(pose=None)
z, i = f._start_ausweich_y(0.0, BREITE)
test('ohne Pose -> unveraendert', z == 0.0 and i is None)

print('\nalle Faelle bestanden')
