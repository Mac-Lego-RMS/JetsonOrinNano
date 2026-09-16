"""Prueft auf_bildzeit() gegen eine unabhaengig gerechnete Wahrheit.

Vorgehen: ein Weltpunkt wird fest gesetzt, daraus werden die Lidar-Koordinaten
zu BEIDEN Zeitpunkten direkt ausgerechnet. Die Funktion bekommt nur die
Scan-Koordinaten und muss die Bild-Koordinaten reproduzieren.
"""
import math, numpy as np, sys, types

# lpm.py ohne ROS-Importe laden: nur die freie Funktion herausschneiden.
quelle = open('lpm.py').read()
start = quelle.index('def auf_bildzeit(')
ende = quelle.index('def _packaged_default(')
mod = types.ModuleType('m')
mod.__dict__.update({'math': math, 'np': np})
exec(quelle[start:ende], mod.__dict__)
auf_bildzeit = mod.auf_bildzeit

OFF_X, OFF_Y, LYAW = 0.110, 0.0, math.pi

def welt_zu_lidar(W, pose):
    x, y, th = pose
    ox = x + math.cos(th) * OFF_X - math.sin(th) * OFF_Y
    oy = y + math.sin(th) * OFF_X + math.cos(th) * OFF_Y
    a = th + LYAW
    dx, dy = W[0] - ox, W[1] - oy
    return (math.cos(a) * dx + math.sin(a) * dy,
            -math.sin(a) * dx + math.cos(a) * dy)

rng = np.random.default_rng(7)
schlimmster = 0.0
for _ in range(2000):
    pose_i = (rng.uniform(-2, 2), rng.uniform(-2, 2), rng.uniform(-math.pi, math.pi))
    d_th = rng.uniform(-0.8, 0.8)             # bis 45 Grad Drehung
    pose_s = (pose_i[0] + rng.uniform(-0.3, 0.3),
              pose_i[1] + rng.uniform(-0.3, 0.3),
              pose_i[2] + d_th)
    W = (rng.uniform(-3, 3), rng.uniform(-3, 3))
    P_s = welt_zu_lidar(W, pose_s)
    P_i_soll = welt_zu_lidar(W, pose_i)
    out, dyaw, dtrans = auf_bildzeit(np.array([[P_s[0], P_s[1], 0.027]]),
                                     pose_s, pose_i, OFF_X, OFF_Y, LYAW)
    fehler = math.hypot(out[0, 0] - P_i_soll[0], out[0, 1] - P_i_soll[1])
    schlimmster = max(schlimmster, fehler)
    assert abs(math.atan2(math.sin(dyaw - d_th), math.cos(dyaw - d_th))) < 1e-9, 'dyaw falsch'
    assert abs(out[0, 2] - 0.027) < 1e-12, 'z veraendert'
print('2000 Zufallsfaelle, groesster Lagefehler: %.2e m' % schlimmster)
assert schlimmster < 1e-9

# Identitaet: gleiche Pose -> Punkte unveraendert
p = (0.4, -1.2, 0.9)
pts = rng.uniform(-2, 2, size=(50, 3))
out, dyaw, dtrans = auf_bildzeit(pts, p, p, OFF_X, OFF_Y, LYAW)
assert np.allclose(out, pts, atol=1e-12) and abs(dyaw) < 1e-12 and dtrans < 1e-12
print('Identitaet ok')

# Reine Drehung um den Lidar-Ursprung: Azimut muss sich exakt um dyaw aendern
for d_th in (0.1, 0.5, -0.7):
    # base_link so legen, dass der Lidar-Ursprung bei beiden Posen gleich liegt
    th_i = 0.0
    o = (0.110, 0.0)
    pose_i = (o[0] - math.cos(th_i) * OFF_X, o[1] - math.sin(th_i) * OFF_X, th_i)
    th_s = th_i + d_th
    pose_s = (o[0] - math.cos(th_s) * OFF_X, o[1] - math.sin(th_s) * OFF_X, th_s)
    P = np.array([[1.5, 0.3, 0.027]])
    out, dyaw, dtrans = auf_bildzeit(P, pose_s, pose_i, OFF_X, OFF_Y, LYAW)
    a0 = math.atan2(P[0, 1], P[0, 0]); a1 = math.atan2(out[0, 1], out[0, 0])
    da = math.atan2(math.sin(a1 - a0), math.cos(a1 - a0))
    # Dreht sich der Roboter zwischen Bild und Scan um +dyaw, dann lag derselbe
    # Weltpunkt im Bild um +dyaw weiter herum -- genau diese Drehung soll die
    # Kompensation nachholen.
    assert abs(da - d_th) < 1e-9, (da, d_th)
    assert abs(np.hypot(*out[0, :2]) - np.hypot(*P[0, :2])) < 1e-9, 'Entfernung veraendert'
    assert dtrans < 1e-9, 'reine Drehung darf keinen Versatz erzeugen'
print('reine Drehung ok: Azimut dreht um +dyaw, Entfernung bleibt')
