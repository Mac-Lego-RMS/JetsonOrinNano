"""Selbsttest der Farbabtastung: laeuft ohne ROS, Kamera und Lidar."""

import numpy as np

from camera_lidar_fusion import colors


def _stripe_image():
    """Bild mit einem roten Balken; darueber und darunter schwarz.

    Die radiale Richtung von (100, 100) nach (100, 200) zeigt nach unten, das
    Band laeuft also senkrecht durch den Balken -- wie an einer Pylone.
    """
    image = np.zeros((300, 200, 3), np.uint8)
    image[180:220, :] = (0, 0, 255)      # BGR: rot
    return image


def test_single_pixel_sampling_reads_the_stripe():
    image = _stripe_image()
    bgr, hsv = colors.sample_colors(image, np.array([100.0]), np.array([200.0]), patch=1)
    assert tuple(bgr[0]) == (0, 0, 255)
    assert colors.classify_hsv(hsv) == ['rot']


def test_band_median_ignores_a_single_outlier_pixel():
    """Ein Ausreisser mitten im Balken darf das Ergebnis nicht kippen."""
    image = _stripe_image()
    image[200, 100] = (255, 255, 255)    # ein weisses Stoerpixel genau im Treffer

    einzeln, _ = colors.sample_colors(image, np.array([100.0]), np.array([200.0]), patch=1)
    band, hsv = colors.sample_colors(image, np.array([100.0]), np.array([200.0]), patch=1,
                                     center=(100.0, 100.0), band_px=np.array([15.0]),
                                     band_count=5)

    assert tuple(einzeln[0]) == (255, 255, 255)   # Einzelpixel faellt drauf rein
    assert tuple(band[0]) == (0, 0, 255)          # Median nicht
    assert colors.classify_hsv(hsv) == ['rot']


def test_band_median_survives_overhanging_one_edge():
    """Rutscht ein Ende des Bandes ueber die Kante, haelt der Median dagegen."""
    image = _stripe_image()
    # Treffer bei v=210, Balken endet bei 220 -> das obere Ende haengt raus.
    band, hsv = colors.sample_colors(image, np.array([100.0]), np.array([210.0]), patch=1,
                                     center=(100.0, 100.0), band_px=np.array([18.0]),
                                     band_count=5)
    assert tuple(band[0]) == (0, 0, 255)
    assert colors.classify_hsv(hsv) == ['rot']


def test_band_runs_along_the_radial_direction():
    """Das Band muss radial laufen, nicht achsparallel."""
    image = np.zeros((300, 300, 3), np.uint8)
    # Radialer Streifen von der Mitte (150,150) nach rechts oben.
    for step in range(0, 120):
        u = int(150 + step * np.cos(np.radians(-45)))
        v = int(150 + step * np.sin(np.radians(-45)))
        image[v - 1:v + 2, u - 1:u + 2] = (0, 255, 0)

    point_u = 150 + 80 * np.cos(np.radians(-45))
    point_v = 150 + 80 * np.sin(np.radians(-45))
    band, hsv = colors.sample_colors(image, np.array([point_u]), np.array([point_v]),
                                     patch=1, center=(150.0, 150.0),
                                     band_px=np.array([20.0]), band_count=5)
    assert colors.classify_hsv(hsv) == ['gruen']
    assert tuple(band[0]) == (0, 255, 0)


def test_band_disabled_matches_single_pixel():
    image = _stripe_image()
    einzeln, _ = colors.sample_colors(image, np.array([100.0]), np.array([200.0]), patch=1)
    aus, _ = colors.sample_colors(image, np.array([100.0]), np.array([200.0]), patch=1,
                                  center=(100.0, 100.0), band_px=None)
    assert tuple(einzeln[0]) == tuple(aus[0])


def test_many_points_at_once():
    """Vektorisierung: viele Punkte mit je eigener Bandbreite."""
    image = _stripe_image()
    n = 50
    u = np.full(n, 100.0)
    v = np.full(n, 200.0)
    band_px = np.linspace(2.0, 18.0, n)
    bgr, hsv = colors.sample_colors(image, u, v, patch=1, center=(100.0, 100.0),
                                    band_px=band_px, band_count=5)
    assert bgr.shape == (n, 3)
    assert colors.classify_hsv(hsv) == ['rot'] * n


def test_only_label_ignores_a_bigger_blob_of_another_colour():
    """Der Fall aus der Praxis: grosses rotes Objekt im Raum, kleine gruene Pylone."""
    image = np.zeros((400, 400, 3), np.uint8)
    image[300:380, 40:160] = (0, 0, 255)      # grosser roter Stoerer
    image[150:190, 250:280] = (0, 255, 0)     # kleine gruene Pylone
    kreis = (200.0, 200.0, 195.0)

    ohne = colors.find_color_blob(image, min_area=100, mask_circle=kreis)
    assert ohne[2] == 'rot'                   # der Groessere gewinnt

    mit = colors.find_color_blob(image, min_area=100, mask_circle=kreis, only_label='gruen')
    assert mit[2] == 'gruen'
    assert 250 < mit[0] < 280 and 150 < mit[1] < 190


def test_max_area_rejects_the_oversized_background_object():
    image = np.zeros((400, 400, 3), np.uint8)
    image[300:380, 40:160] = (0, 0, 255)
    image[150:190, 250:280] = (0, 0, 255)     # zweite, kleine rote Flaeche
    kreis = (200.0, 200.0, 195.0)

    gross = colors.find_color_blob(image, min_area=100, mask_circle=kreis)
    klein = colors.find_color_blob(image, min_area=100, mask_circle=kreis, max_area=3000)
    assert gross[3] > klein[3]
    assert 250 < klein[0] < 280


def test_blob_radii_bracket_the_pylon_in_the_image():
    """r_innen/r_aussen umschliessen den Blob -- Grundlage fuer cal radial."""
    image = np.zeros((400, 400, 3), np.uint8)
    image[250:330, 190:210] = (0, 255, 0)     # senkrechter Streifen unter der Mitte
    kreis = (200.0, 200.0, 195.0)

    blob = colors.find_color_blob(image, min_area=100, mask_circle=kreis, only_label='gruen')
    _, _, label, _, r_innen, r_aussen = blob
    assert label == 'gruen'
    assert r_innen < r_aussen
    assert 45 < r_innen < 60        # Oberkante, rund 50 px unter der Mitte
    assert 125 < r_aussen < 140     # Fusspunkt, rund 130 px
