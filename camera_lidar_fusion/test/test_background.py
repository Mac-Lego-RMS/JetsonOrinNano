"""Selbsttest der Referenzscan-Logik (Vordergrund vom festen Aufbau trennen)."""

import math

import numpy as np

from camera_lidar_fusion.fisheye_model import visible_mask


def _welt(count=3240):
    """Umgebung wie am echten Roboter: Kabel, Elektronik, Streifsektor, Wand."""
    angles = np.degrees(-math.pi + np.arange(count) * (2 * math.pi / count))
    ranges = np.full(count, 2.5)                       # Wand rundum
    ranges[(angles >= 135.2) | (angles <= -153.8)] = 0.08    # Elektronik
    ranges[(angles >= -17.9) & (angles <= -8.2)] = 0.06      # Kabel
    # Der Streifbereich direkt neben der Elektronik: 0.23 m, also ueber der
    # 0.15-m-Schwelle der Blindsektor-Erkennung -- genau die Stelle, an der
    # die Cluster-Suche vorher haengen blieb.
    ranges[(angles > -153.8) & (angles <= -150.0)] = 0.23
    return ranges, angles


def _referenz(count=3240):
    ranges, _ = _welt(count)
    return ranges.copy()


def test_blind_sectors_alone_do_not_catch_the_grazing_edge():
    """Der Streifbereich bleibt sichtbar und ist der naechste Punkt."""
    ranges, angles = _welt()
    keep = visible_mask(np.radians(angles), [135.2, -153.8, -17.9, -8.2])

    nearest = angles[keep][np.argmin(ranges[keep])]
    assert abs(nearest - (-152.0)) < 3.0            # landet am Streifbereich
    assert math.isclose(ranges[keep].min(), 0.23, abs_tol=1e-6)


def test_background_subtraction_finds_the_pylon_instead():
    """Mit Referenzscan gewinnt die Pylone, obwohl sie weiter weg steht."""
    reference = _referenz()
    ranges, angles = _welt()
    # Pylone bei +40 Grad in 0.35 m -- weiter weg als der Streifbereich (0.23).
    pylon = (angles > 37) & (angles < 43)
    ranges[pylon] = 0.35

    keep = visible_mask(np.radians(angles), [135.2, -153.8, -17.9, -8.2])
    keep &= ranges < (reference - 0.08)

    assert keep.sum() > 0
    assert math.isclose(ranges[keep].min(), 0.35, abs_tol=1e-6)
    assert abs(angles[keep][np.argmin(ranges[keep])] - 40.0) < 3.0
    # Und nur die Pylone bleibt uebrig, sonst nichts.
    assert np.array_equal(keep, pylon & keep)


def test_background_subtraction_works_without_any_blind_sectors():
    """Der Referenzscan allein reicht -- Blindsektoren sind nur noch Beiwerk."""
    reference = _referenz()
    ranges, angles = _welt()
    ranges[(angles > 100) & (angles < 106)] = 0.5

    keep = ranges < (reference - 0.08)

    assert math.isclose(ranges[keep].min(), 0.5, abs_tol=1e-6)
    assert abs(angles[keep][np.argmin(ranges[keep])] - 103.0) < 3.0


def test_moving_the_pylon_gives_distinct_targets():
    """Zwei Positionen muessen zwei klar verschiedene Ziele ergeben."""
    reference = _referenz()
    treffer = []
    for grad, dist in ((40.0, 0.35), (-60.0, 0.8)):
        ranges, angles = _welt()
        ranges[(angles > grad - 3) & (angles < grad + 3)] = dist
        keep = ranges < (reference - 0.08)
        i = np.argmin(np.where(keep, ranges, np.inf))
        treffer.append((angles[i], ranges[i]))

    assert abs(treffer[0][0] - 40.0) < 3.0 and math.isclose(treffer[0][1], 0.35, abs_tol=1e-6)
    assert abs(treffer[1][0] - (-60.0)) < 3.0 and math.isclose(treffer[1][1], 0.8, abs_tol=1e-6)
    assert abs(treffer[0][0] - treffer[1][0]) > 50.0


def test_empty_scene_yields_no_target():
    """Ohne Pylone darf gar nichts als Ziel durchgehen."""
    reference = _referenz()
    ranges, _ = _welt()
    assert not (ranges < (reference - 0.08)).any()


def _delta(bearing_deg, u, v, cx=677.5, cy=454.0):
    phi = math.degrees(math.atan2(v - cy, u - cx))
    return (phi - bearing_deg + 180.0) % 360.0 - 180.0


def test_outlier_rejection_on_the_real_measurement():
    """Die echten 6 Samples: einer ist falsch, fuenf stimmen auf 1 Grad."""
    # (Peilung, u, v) aus dem Log am Roboter
    samples = [(48.07, 953.9, 760.4), (9.67, 302.2, 221.1), (-40.57, 981.5, 178.4),
               (-64.35, 838.1, 68.9), (61.35, 881.0, 816.2), (94.92, 649.3, 864.0)]
    deltas = np.array([_delta(b, u, v) for b, u, v in samples])

    # Zirkulaerer Median wie in _inliers: der Kandidat mit der kleinsten
    # Summe der Winkelabstaende.
    def wrap(a):
        return (a + 180.0) % 360.0 - 180.0
    spans = [np.abs(wrap(deltas - d)).sum() for d in deltas]
    center = deltas[int(np.argmin(spans))]
    abweichung = np.abs(wrap(deltas - center))

    keep = abweichung <= 20.0
    assert keep.sum() == 5
    assert not keep[1]                       # Sample 2 fliegt raus

    yaw = deltas[keep].mean()
    assert abs(yaw - (-1.28)) < 0.2
    assert deltas[keep].std() < 1.5

    # Mit dem Ausreisser waere die Loesung um mehrere Grad daneben.
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
