# -*- coding: utf-8 -*-
"""Tests fuer den Neutralpunkt (Weissabgleich am Spielfeld)."""
import numpy as np
import pytest

from camera_lidar_fusion import colors


def _ring_bild(bgr_matte, groesse=400, r_min=150, r_max=190):
    """Kunstbild: farbloser Hintergrund, im Ring die vorgegebene 'Matte'."""
    bild = np.full((groesse, groesse, 3), 20, np.uint8)
    yy, xx = np.mgrid[0:groesse, 0:groesse]
    rad = np.hypot(xx - groesse / 2.0, yy - groesse / 2.0)
    ring = (rad >= r_min) & (rad <= r_max)
    for k in range(3):
        bild[..., k][ring] = bgr_matte[k]
    return bild


def test_neutrale_matte_ergibt_null():
    bild = _ring_bild((200, 200, 200))
    z0 = colors.neutralpunkt(bild, (200, 200), 150, 190, sektoren=8, schritt=1)
    assert z0.shape == (8,)
    assert np.allclose(z0, 0.0, atol=1e-3)


def test_gruenstich_wird_gemessen():
    # B=167 G=212 R=194 -- die am Aufbau gemessene Matte, z soll +0.085 sein
    bild = _ring_bild((167, 212, 194))
    z0 = colors.neutralpunkt(bild, (200, 200), 150, 190, sektoren=8, schritt=1)
    assert np.allclose(z0, (212 - 194) / 212.0, atol=5e-3)


def test_leerer_ring_gibt_null_statt_muell():
    """Kein brauchbares Pixel -> 0, also 'keine Korrektur'. Nie NaN."""
    bild = np.zeros((400, 400, 3), np.uint8)
    z0 = colors.neutralpunkt(bild, (200, 200), 900, 950, sektoren=8)
    assert np.all(np.isfinite(z0)) and np.allclose(z0, 0.0)


def test_z0_je_punkt_ist_zyklisch():
    z0 = np.array([0.1, 0.2, 0.3, 0.4], np.float32)
    phi = np.array([0.0, 2 * np.pi, -2 * np.pi], np.float32)
    w = colors.z0_je_punkt(phi, z0)
    assert np.allclose(w, w[0])


def test_z0_je_punkt_trifft_sektormitten():
    z0 = np.array([0.0, 0.5, 1.0, 0.5], np.float32)
    mitten = (np.arange(4) + 0.5) * (2 * np.pi / 4)
    assert np.allclose(colors.z0_je_punkt(mitten, z0), z0, atol=1e-5)


def test_z0_skalar_wird_akzeptiert():
    w = colors.z0_je_punkt(np.zeros(5), [0.07])
    assert np.allclose(w, 0.07)


def test_classify_zone_z0_rettet_rot():
    """Ein leicht rotes Feld auf gruenstichiger Kamera.

    Roh liegt z knapp ueber der Schwelle -0.15 und faellt durch; mit dem
    gemessenen Neutralpunkt wird daraus sauber 'rot'.
    """
    # BGR so gewaehlt, dass z = (G-R)/max ungefaehr -0.09 ist
    bild = np.zeros((400, 400, 3), np.uint8)
    bild[..., 0] = 60      # B
    bild[..., 1] = 155     # G
    bild[..., 2] = 170     # R   -> z = -15/170 = -0.088, S kraeftig
    phi = np.linspace(0, 2 * np.pi, 16, endpoint=False)
    r_in = np.full(16, 60.0)
    r_out = np.full(16, 90.0)
    gem = dict(center=(200, 200), min_frac=0.5, steps=9,
               ranges={k: v for k, v in colors.DEFAULT_RANGES.items()
                       if k in ('rot', 'gruen')})

    ohne, _, _ = colors.classify_zone(bild, phi, r_in, r_out, **gem)
    assert set(ohne) == {'unbekannt'}, ohne[:3]

    mit, _, _ = colors.classify_zone(bild, phi, r_in, r_out, z0=0.085, **gem)
    assert set(mit) == {'rot'}, mit[:3]


def test_classify_zone_z0_entfernt_phantomgruen():
    """Die weisse Matte selbst darf nicht als gruen durchgehen."""
    bild = np.zeros((400, 400, 3), np.uint8)
    bild[..., 0], bild[..., 1], bild[..., 2] = 167, 212, 194   # gemessene Matte
    phi = np.linspace(0, 2 * np.pi, 16, endpoint=False)
    r_in, r_out = np.full(16, 60.0), np.full(16, 90.0)
    gem = dict(center=(200, 200), min_frac=0.5, steps=9,
               ranges={k: v for k, v in colors.DEFAULT_RANGES.items()
                       if k in ('rot', 'gruen')})
    mit, _, _ = colors.classify_zone(bild, phi, r_in, r_out,
                                     z0=(212 - 194) / 212.0, **gem)
    assert 'gruen' not in set(mit), mit[:3]


def test_ring_cache_liefert_gleiches_ergebnis():
    bild = _ring_bild((167, 212, 194))
    a = colors.neutralpunkt(bild, (200, 200), 150, 190, sektoren=8, schritt=2)
    b = colors.neutralpunkt(bild, (200, 200), 150, 190, sektoren=8, schritt=2)
    assert np.allclose(a, b)
