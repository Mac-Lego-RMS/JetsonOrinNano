# -*- coding: utf-8 -*-
"""Tests for the neutral point (white balance on the field)."""
import numpy as np
import pytest

from camera_lidar_fusion import colors


def _ring_image(bgr_mat, size=400, r_min=150, r_max=190):
    """Synthetic image: colourless background, the given 'mat' in the ring."""
    image = np.full((size, size, 3), 20, np.uint8)
    yy, xx = np.mgrid[0:size, 0:size]
    rad = np.hypot(xx - size / 2.0, yy - size / 2.0)
    ring = (rad >= r_min) & (rad <= r_max)
    for k in range(3):
        image[..., k][ring] = bgr_mat[k]
    return image


def test_neutral_mat_gives_zero():
    image = _ring_image((200, 200, 200))
    z0 = colors.neutral_point(image, (200, 200), 150, 190, sectors=8, step=1)
    assert z0.shape == (8,)
    assert np.allclose(z0, 0.0, atol=1e-3)


def test_green_cast_is_measured():
    # B=167 G=212 R=194 -- the mat measured on the setup, z should be +0.085
    image = _ring_image((167, 212, 194))
    z0 = colors.neutral_point(image, (200, 200), 150, 190, sectors=8, step=1)
    assert np.allclose(z0, (212 - 194) / 212.0, atol=5e-3)


def test_empty_ring_gives_zero_not_garbage():
    """No usable pixel -> 0, i.e. 'no correction'. Never NaN."""
    image = np.zeros((400, 400, 3), np.uint8)
    z0 = colors.neutral_point(image, (200, 200), 900, 950, sectors=8)
    assert np.all(np.isfinite(z0)) and np.allclose(z0, 0.0)


def test_z0_per_point_is_cyclic():
    z0 = np.array([0.1, 0.2, 0.3, 0.4], np.float32)
    phi = np.array([0.0, 2 * np.pi, -2 * np.pi], np.float32)
    w = colors.z0_per_point(phi, z0)
    assert np.allclose(w, w[0])


def test_z0_per_point_hits_sector_centres():
    z0 = np.array([0.0, 0.5, 1.0, 0.5], np.float32)
    mids = (np.arange(4) + 0.5) * (2 * np.pi / 4)
    assert np.allclose(colors.z0_per_point(mids, z0), z0, atol=1e-5)


def test_z0_scalar_is_accepted():
    w = colors.z0_per_point(np.zeros(5), [0.07])
    assert np.allclose(w, 0.07)


def test_classify_zone_z0_rescues_red():
    """A slightly red field on a camera with a green cast.

    Raw, z lies just above the threshold -0.15 and fails; with the
    measured neutral point it cleanly becomes 'red'.
    """
    # BGR chosen so that z = (G-R)/max is about -0.09
    image = np.zeros((400, 400, 3), np.uint8)
    image[..., 0] = 60      # B
    image[..., 1] = 155     # G
    image[..., 2] = 170     # R   -> z = -15/170 = -0.088, S strong
    phi = np.linspace(0, 2 * np.pi, 16, endpoint=False)
    r_in = np.full(16, 60.0)
    r_out = np.full(16, 90.0)
    common = dict(center=(200, 200), min_frac=0.5, steps=9,
                  ranges={k: v for k, v in colors.DEFAULT_RANGES.items()
                          if k in ('red', 'green')})

    without, _, _ = colors.classify_zone(image, phi, r_in, r_out, **common)
    assert set(without) == {'unknown'}, without[:3]

    with_z0, _, _ = colors.classify_zone(image, phi, r_in, r_out, z0=0.085, **common)
    assert set(with_z0) == {'red'}, with_z0[:3]


def test_classify_zone_z0_removes_phantom_green():
    """The white mat itself must not pass as green."""
    image = np.zeros((400, 400, 3), np.uint8)
    image[..., 0], image[..., 1], image[..., 2] = 167, 212, 194   # measured mat
    phi = np.linspace(0, 2 * np.pi, 16, endpoint=False)
    r_in, r_out = np.full(16, 60.0), np.full(16, 90.0)
    common = dict(center=(200, 200), min_frac=0.5, steps=9,
                  ranges={k: v for k, v in colors.DEFAULT_RANGES.items()
                          if k in ('red', 'green')})
    with_z0, _, _ = colors.classify_zone(image, phi, r_in, r_out,
                                         z0=(212 - 194) / 212.0, **common)
    assert 'green' not in set(with_z0), with_z0[:3]


def test_ring_cache_gives_same_result():
    image = _ring_image((167, 212, 194))
    a = colors.neutral_point(image, (200, 200), 150, 190, sectors=8, step=2)
    b = colors.neutral_point(image, (200, 200), 150, 190, sectors=8, step=2)
    assert np.allclose(a, b)
