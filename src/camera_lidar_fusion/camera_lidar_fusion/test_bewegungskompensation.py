"""Checks to_image_time() against an independently computed ground truth.

Approach: a world point is fixed, and from it the lidar coordinates at BOTH
times are computed directly. The function only gets the scan coordinates
and has to reproduce the image coordinates.
"""
import math, numpy as np, sys, types

# Load lpm.py without the ROS imports: cut out only the free function.
source = open('lpm.py').read()
start = source.index('def to_image_time(')
end = source.index('def _packaged_default(')
mod = types.ModuleType('m')
mod.__dict__.update({'math': math, 'np': np})
exec(source[start:end], mod.__dict__)
to_image_time = mod.to_image_time

OFF_X, OFF_Y, LYAW = 0.110, 0.0, math.pi

def world_to_lidar(W, pose):
    x, y, th = pose
    ox = x + math.cos(th) * OFF_X - math.sin(th) * OFF_Y
    oy = y + math.sin(th) * OFF_X + math.cos(th) * OFF_Y
    a = th + LYAW
    dx, dy = W[0] - ox, W[1] - oy
    return (math.cos(a) * dx + math.sin(a) * dy,
            -math.sin(a) * dx + math.cos(a) * dy)

rng = np.random.default_rng(7)
worst = 0.0
for _ in range(2000):
    pose_i = (rng.uniform(-2, 2), rng.uniform(-2, 2), rng.uniform(-math.pi, math.pi))
    d_th = rng.uniform(-0.8, 0.8)             # up to 45 degrees of rotation
    pose_s = (pose_i[0] + rng.uniform(-0.3, 0.3),
              pose_i[1] + rng.uniform(-0.3, 0.3),
              pose_i[2] + d_th)
    W = (rng.uniform(-3, 3), rng.uniform(-3, 3))
    P_s = world_to_lidar(W, pose_s)
    P_i_target = world_to_lidar(W, pose_i)
    out, dyaw, dtrans = to_image_time(np.array([[P_s[0], P_s[1], 0.027]]),
                                      pose_s, pose_i, OFF_X, OFF_Y, LYAW)
    error = math.hypot(out[0, 0] - P_i_target[0], out[0, 1] - P_i_target[1])
    worst = max(worst, error)
    assert abs(math.atan2(math.sin(dyaw - d_th), math.cos(dyaw - d_th))) < 1e-9, 'dyaw wrong'
    assert abs(out[0, 2] - 0.027) < 1e-12, 'z changed'
print('2000 random cases, largest position error: %.2e m' % worst)
assert worst < 1e-9

# Identity: same pose -> points unchanged
p = (0.4, -1.2, 0.9)
pts = rng.uniform(-2, 2, size=(50, 3))
out, dyaw, dtrans = to_image_time(pts, p, p, OFF_X, OFF_Y, LYAW)
assert np.allclose(out, pts, atol=1e-12) and abs(dyaw) < 1e-12 and dtrans < 1e-12
print('identity ok')

# Pure rotation about the lidar origin: the azimuth must change by exactly dyaw
for d_th in (0.1, 0.5, -0.7):
    # place base_link so that the lidar origin is at the same spot for both poses
    th_i = 0.0
    o = (0.110, 0.0)
    pose_i = (o[0] - math.cos(th_i) * OFF_X, o[1] - math.sin(th_i) * OFF_X, th_i)
    th_s = th_i + d_th
    pose_s = (o[0] - math.cos(th_s) * OFF_X, o[1] - math.sin(th_s) * OFF_X, th_s)
    P = np.array([[1.5, 0.3, 0.027]])
    out, dyaw, dtrans = to_image_time(P, pose_s, pose_i, OFF_X, OFF_Y, LYAW)
    a0 = math.atan2(P[0, 1], P[0, 0]); a1 = math.atan2(out[0, 1], out[0, 0])
    da = math.atan2(math.sin(a1 - a0), math.cos(a1 - a0))
    # If the robot turns by +dyaw between image and scan, the same world point
    # lay +dyaw further round in the image -- exactly this rotation is what
    # the compensation has to catch up on.
    assert abs(da - d_th) < 1e-9, (da, d_th)
    assert abs(np.hypot(*out[0, :2]) - np.hypot(*P[0, :2])) < 1e-9, 'range changed'
    assert dtrans < 1e-9, 'a pure rotation must not produce a shift'
print('pure rotation ok: azimuth turns by +dyaw, range stays')
