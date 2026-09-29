"""Edge cases of _start_dodge_y -- against the real code text from r1c.py."""
import collections
import math
import types

source = open('r1c.py').read()
start = source.index('    def _start_dodge_y(')
end = source.index('    def publish_lap_state(')
src = '\n'.join(l[4:] if l.startswith('    ') else l
                for l in source[start:end].split('\n'))
ns = {'math': math, 'OBST_GREEN': 2, 'OBST_RED': 1, 'BLOCK_HALF': 0.022}
exec(src, ns)

RED, GREEN, UNKNOWN = 1, 2, 0


class Fake:
    _start_dodge_y = ns['_start_dodge_y']
    _start_dodge_hold = ns['_start_dodge_hold']

    def __init__(self, **kw):
        self.start_dodge = True
        self.start_dodge_look = 1.20
        self.start_dodge_back = 0.25
        self.start_dodge_lane = 0.35
        self.start_dodge_margin = 0.12
        self.start_dodge_votes = 3
        self.start_dodge_window_s = 1.0
        self.live_obs = collections.deque(maxlen=60)
        self._start_hold = None
        self.pose = (0.0, 0.0, 0.0)
        self._t = 10.0
        self.__dict__.update(kw)

    def get_clock(self):
        s = self
        return types.SimpleNamespace(
            now=lambda: types.SimpleNamespace(nanoseconds=s._t * 1e9))

    def see(self, mx, my, colour, n=5):
        for i in range(n):
            self.live_obs.append((self._t - 0.05 * i, mx, my, colour))


WIDTH = 1.00      # lane -0.50 .. +0.50 around the centre

def test(name, cond):
    assert cond, 'FAILED: ' + name
    print('  ok  ' + name)


f = Fake(); f.see(0.95, -0.10, GREEN)
z, i = f._start_dodge_y(0.0, WIDTH)
test('green -> pass on the left, target +0.211', abs(z - 0.211) < 0.001 and i[3] == 'left')

f = Fake(); f.see(0.95, -0.10, RED)
z, i = f._start_dodge_y(0.0, WIDTH)
test('red -> pass on the right, target -0.311', abs(z - (-0.311)) < 0.001 and i[3] == 'right')

# 0.5*((0.30+0.022)+0.50) = 0.411, limited to 0.50-0.12 = 0.38
f = Fake(); f.see(0.95, +0.30, GREEN)
z, i = f._start_dodge_y(0.0, WIDTH)
test('green far outside -> limited to the wall margin', abs(z - 0.38) < 1e-9)

f = Fake(); f.see(0.95, -0.10, UNKNOWN)
z, i = f._start_dodge_y(0.0, WIDTH)
test('colour unknown -> side with more room (left)', z > 0 and 'unknown' in i[3])

f = Fake(); f.see(0.95, +0.10, UNKNOWN)
z, i = f._start_dodge_y(0.0, WIDTH)
test('colour unknown, block on the left -> pass on the right', z < 0 and 'unknown' in i[3])

f = Fake(); f.see(0.95, -0.10, GREEN, n=2)
z, i = f._start_dodge_y(0.0, WIDTH)
test('too few sightings -> no steering', z == 0.0 and i is None)

f = Fake(); f.see(2.00, -0.10, GREEN)
z, i = f._start_dodge_y(0.0, WIDTH)
test('too far ahead -> no steering', z == 0.0 and i is None)

f = Fake(); f.see(0.95, -0.45, GREEN)
z, i = f._start_dodge_y(0.0, WIDTH)
test('sideways outside the lane -> ignored', z == 0.0 and i is None)

f = Fake(); f.see(0.95, -0.10, GREEN)
f._t += 2.0                                   # all sightings stale
z, i = f._start_dodge_y(0.0, WIDTH)
test('stale sightings -> no steering', z == 0.0 and i is None)

f = Fake(); f.see(0.95, -0.10, GREEN)
f._start_dodge_y(0.0, WIDTH)         # set the hold
f.live_obs.clear()
f.pose = (0.90, 0.0, 0.0)                # right next to the block
z, _ = f._start_dodge_y(0.0, WIDTH)
test('hold while passing', abs(z - 0.211) < 0.001)
f.pose = (1.21, 0.0, 0.0)                # 0.26 m behind it
z, i = f._start_dodge_y(0.0, WIDTH)
test('release from 0.25 m behind it', z == 0.0 and i is None)

f = Fake(start_dodge=False); f.see(0.95, -0.10, GREEN)
z, i = f._start_dodge_y(0.0, WIDTH)
test('switch off -> unchanged', z == 0.0 and i is None)

f = Fake(pose=None)
z, i = f._start_dodge_y(0.0, WIDTH)
test('without pose -> unchanged', z == 0.0 and i is None)

print('\nall cases passed')
