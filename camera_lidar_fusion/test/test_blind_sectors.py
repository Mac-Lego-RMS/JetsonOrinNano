"""Selbsttest der Blindsektoren (verbaute Lidar-Bereiche)."""

import math

import numpy as np

from camera_lidar_fusion.fisheye_model import find_blind_sectors, visible_mask


def _scan(count=3240, blocked=(), value=0.08, far=2.5):
    """Ein Scan: ueberall ``far``, in den Sektoren ``blocked`` nur ``value``."""
    angles = np.degrees(-math.pi + np.arange(count) * (2 * math.pi / count))
    ranges = np.full(count, far)
    for lo, hi in blocked:
        if lo <= hi:
            ranges[(angles >= lo) & (angles <= hi)] = value
        else:
            ranges[(angles >= lo) | (angles <= hi)] = value
    return ranges, angles


def test_visible_mask_blocks_a_plain_sector():
    _, angles = _scan()
    keep = visible_mask(np.radians(angles), [-20.0, -10.0])
    assert not keep[(angles > -19) & (angles < -11)].any()
    assert keep[(angles > 0) & (angles < 90)].all()


def test_visible_mask_handles_a_sector_across_180():
    _, angles = _scan()
    keep = visible_mask(np.radians(angles), [135.0, -153.0])
    assert not keep[angles > 140].any()
    assert not keep[angles < -160].any()
    assert keep[(angles > -90) & (angles < 90)].all()


def test_visible_mask_without_sectors_keeps_everything():
    _, angles = _scan()
    assert visible_mask(np.radians(angles), []).all()


def test_find_blind_sectors_recovers_the_real_measurement():
    """Die drei Sektoren, die am S3 tatsaechlich gemessen wurden."""
    truth = [(135.2, -153.4), (-133.8, -119.1), (-17.8, -8.2)]
    ranges, _ = _scan(blocked=truth)
    scans = [ranges + np.random.default_rng(i).normal(0, 0.002, ranges.size)
             for i in range(25)]

    found = find_blind_sectors(scans, -math.pi, 2 * math.pi / ranges.size)

    assert len(found) == 6                       # drei Paare
    # Breitester Sektor zuerst -- das ist der ueber +-180.
    assert abs(found[0] - 135.2) < 0.5 and abs(found[1] - (-153.4)) < 0.5
    breiten = [(found[i + 1] - found[i]) % 360.0 for i in range(0, 6, 2)]
    assert breiten == sorted(breiten, reverse=True)
    assert abs(breiten[0] - 71.4) < 1.0


def test_found_sectors_actually_mask_the_short_returns():
    """Der Kreis schliesst sich: messen -> maskieren -> nur noch echte Ziele."""
    truth = [(135.2, -153.4), (-17.8, -8.2)]
    ranges, angles = _scan(blocked=truth)
    # Eine Pylone bei +40 Grad, 0.35 m -- weiter weg als die Kurz-Returns.
    ranges[(angles > 37) & (angles < 43)] = 0.35

    scans = [ranges] * 25
    found = find_blind_sectors(scans, -math.pi, 2 * math.pi / ranges.size)
    keep = visible_mask(np.radians(angles), found)

    # Ohne Maske ist der naechste Punkt der eigene Aufbau ...
    assert ranges.min() < 0.1
    # ... mit Maske ist es die Pylone.
    assert math.isclose(ranges[keep].min(), 0.35, abs_tol=1e-6)
    assert abs(angles[keep][np.argmin(ranges[keep])] - 40.0) < 3.0


def test_narrow_glitches_are_ignored():
    """Einzelne Ausreisser duerfen keinen Sektor erzeugen."""
    ranges, _ = _scan()
    ranges[100] = 0.05
    ranges[2000:2003] = 0.05
    found = find_blind_sectors([ranges] * 25, -math.pi, 2 * math.pi / ranges.size,
                               min_width_deg=3.0)
    assert found == []
