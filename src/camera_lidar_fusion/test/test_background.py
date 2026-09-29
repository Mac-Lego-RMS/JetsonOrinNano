"""Self-test of the reference scan logic (separating foreground from our fixed build)."""

import math

import numpy as np

from camera_lidar_fusion.fisheye_model import visible_mask


def _world(count=3240):
    """Surroundings as on the real robot: cable, electronics, grazing sector, wall."""
    angles = np.degrees(-math.pi + np.arange(count) * (2 * math.pi / count))
    ranges = np.full(count, 2.5)                       # wall all round
    ranges[(angles >= 135.2) | (angles <= -153.8)] = 0.08    # electronics
    ranges[(angles >= -17.9) & (angles <= -8.2)] = 0.06      # cable
    # The grazing area right next to the electronics: 0.23 m, i.e. above the
    # 0.15 m threshold of the blind sector detection -- exactly the spot where
    # the cluster search used to get stuck.
    ranges[(angles > -153.8) & (angles <= -150.0)] = 0.23
    return ranges, angles


def _reference(count=3240):
    ranges, _ = _world(count)
    return ranges.copy()


def test_blind_sectors_alone_do_not_catch_the_grazing_edge():
    """The grazing area stays visible and is the nearest point."""
    ranges, angles = _world()
    keep = visible_mask(np.radians(angles), [135.2, -153.8, -17.9, -8.2])

    nearest = angles[keep][np.argmin(ranges[keep])]
    assert abs(nearest - (-152.0)) < 3.0            # lands on the grazing area
    assert math.isclose(ranges[keep].min(), 0.23, abs_tol=1e-6)


def test_background_subtraction_finds_the_pylon_instead():
    """With the reference scan the pylon wins, although it stands further away."""
    reference = _reference()
    ranges, angles = _world()
    # Pylon at +40 deg at 0.35 m -- further away than the grazing area (0.23).
    pylon = (angles > 37) & (angles < 43)
    ranges[pylon] = 0.35

    keep = visible_mask(np.radians(angles), [135.2, -153.8, -17.9, -8.2])
    keep &= ranges < (reference - 0.08)

    assert keep.sum() > 0
    assert math.isclose(ranges[keep].min(), 0.35, abs_tol=1e-6)
    assert abs(angles[keep][np.argmin(ranges[keep])] - 40.0) < 3.0
    # And only the pylon is left, nothing else.
    assert np.array_equal(keep, pylon & keep)


def test_background_subtraction_works_without_any_blind_sectors():
    """The reference scan alone is enough -- blind sectors are only an extra now."""
    reference = _reference()
    ranges, angles = _world()
    ranges[(angles > 100) & (angles < 106)] = 0.5

    keep = ranges < (reference - 0.08)

    assert math.isclose(ranges[keep].min(), 0.5, abs_tol=1e-6)
    assert abs(angles[keep][np.argmin(ranges[keep])] - 103.0) < 3.0


def test_moving_the_pylon_gives_distinct_targets():
    """Two positions must give two clearly different targets."""
    reference = _reference()
    hits = []
    for deg, dist in ((40.0, 0.35), (-60.0, 0.8)):
        ranges, angles = _world()
        ranges[(angles > deg - 3) & (angles < deg + 3)] = dist
        keep = ranges < (reference - 0.08)
        i = np.argmin(np.where(keep, ranges, np.inf))
        hits.append((angles[i], ranges[i]))

    assert abs(hits[0][0] - 40.0) < 3.0 and math.isclose(hits[0][1], 0.35, abs_tol=1e-6)
    assert abs(hits[1][0] - (-60.0)) < 3.0 and math.isclose(hits[1][1], 0.8, abs_tol=1e-6)
    assert abs(hits[0][0] - hits[1][0]) > 50.0


def test_empty_scene_yields_no_target():
    """Without a pylon nothing at all may pass as a target."""
    reference = _reference()
    ranges, _ = _world()
    assert not (ranges < (reference - 0.08)).any()


def _delta(bearing_deg, u, v, cx=677.5, cy=454.0):
    phi = math.degrees(math.atan2(v - cy, u - cx))
    return (phi - bearing_deg + 180.0) % 360.0 - 180.0


def test_outlier_rejection_on_the_real_measurement():
    """The real 6 samples: one is wrong, five agree to within 1 deg."""
    # (bearing, u, v) from the log on the robot
    samples = [(48.07, 953.9, 760.4), (9.67, 302.2, 221.1), (-40.57, 981.5, 178.4),
               (-64.35, 838.1, 68.9), (61.35, 881.0, 816.2), (94.92, 649.3, 864.0)]
    deltas = np.array([_delta(b, u, v) for b, u, v in samples])

    # Circular median as in _inliers: the candidate with the smallest
    # sum of angular distances.
    def wrap(a):
        return (a + 180.0) % 360.0 - 180.0
    spans = [np.abs(wrap(deltas - d)).sum() for d in deltas]
    center = deltas[int(np.argmin(spans))]
    deviation = np.abs(wrap(deltas - center))

    keep = deviation <= 20.0
    assert keep.sum() == 5
    assert not keep[1]                       # sample 2 is thrown out

    yaw = deltas[keep].mean()
    assert abs(yaw - (-1.28)) < 0.2
    assert deltas[keep].std() < 1.5

    # With the outlier the solution would be off by several degrees.
    assert abs(deltas.mean() - yaw) > 20.0


def test_a_single_outlier_does_not_survive_a_clean_set():
    deltas = np.array([-1.2, -1.5, -0.9, -1.1, 160.0])

    def wrap(a):
        return (a + 180.0) % 360.0 - 180.0
    spans = [np.abs(wrap(deltas - d)).sum() for d in deltas]
    center = deltas[int(np.argmin(spans))]
    keep = np.abs(wrap(deltas - center)) <= 20.0

    assert keep.sum() == 4
    assert not keep[-1]
