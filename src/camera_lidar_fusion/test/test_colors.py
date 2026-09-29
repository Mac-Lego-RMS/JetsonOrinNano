"""Self-test of the colour sampling: runs without ROS, camera and lidar."""

import numpy as np

from camera_lidar_fusion import colors


def _stripe_image():
    """Image with a red bar; black above and below it.

    The radial direction from (100, 100) to (100, 200) points downwards, so the
    band runs vertically through the bar -- as on a pylon.
    """
    image = np.zeros((300, 200, 3), np.uint8)
    image[180:220, :] = (0, 0, 255)      # BGR: red
    return image


def test_single_pixel_sampling_reads_the_stripe():
    image = _stripe_image()
    bgr, hsv = colors.sample_colors(image, np.array([100.0]), np.array([200.0]), patch=1)
    assert tuple(bgr[0]) == (0, 0, 255)
    assert colors.classify_hsv(hsv) == ['red']


def test_band_median_ignores_a_single_outlier_pixel():
    """An outlier in the middle of the bar must not flip the result."""
    image = _stripe_image()
    image[200, 100] = (255, 255, 255)    # a white noise pixel exactly at the hit

    single, _ = colors.sample_colors(image, np.array([100.0]), np.array([200.0]), patch=1)
    band, hsv = colors.sample_colors(image, np.array([100.0]), np.array([200.0]), patch=1,
                                     center=(100.0, 100.0), band_px=np.array([15.0]),
                                     band_count=5)

    assert tuple(single[0]) == (255, 255, 255)   # single pixel falls for it
    assert tuple(band[0]) == (0, 0, 255)          # median does not
    assert colors.classify_hsv(hsv) == ['red']


def test_band_median_survives_overhanging_one_edge():
    """If one end of the band slips over the edge, the median holds out."""
    image = _stripe_image()
    # hit at v=210, bar ends at 220 -> the upper end hangs out.
    band, hsv = colors.sample_colors(image, np.array([100.0]), np.array([210.0]), patch=1,
                                     center=(100.0, 100.0), band_px=np.array([18.0]),
                                     band_count=5)
    assert tuple(band[0]) == (0, 0, 255)
    assert colors.classify_hsv(hsv) == ['red']


def test_band_runs_along_the_radial_direction():
    """The band must run radially, not parallel to the axes."""
    image = np.zeros((300, 300, 3), np.uint8)
    # Radial stripe from the centre (150,150) to the top right.
    for step in range(0, 120):
        u = int(150 + step * np.cos(np.radians(-45)))
        v = int(150 + step * np.sin(np.radians(-45)))
        image[v - 1:v + 2, u - 1:u + 2] = (0, 255, 0)

    point_u = 150 + 80 * np.cos(np.radians(-45))
    point_v = 150 + 80 * np.sin(np.radians(-45))
    band, hsv = colors.sample_colors(image, np.array([point_u]), np.array([point_v]),
                                     patch=1, center=(150.0, 150.0),
                                     band_px=np.array([20.0]), band_count=5)
    assert colors.classify_hsv(hsv) == ['green']
    assert tuple(band[0]) == (0, 255, 0)


def test_band_disabled_matches_single_pixel():
    image = _stripe_image()
    single, _ = colors.sample_colors(image, np.array([100.0]), np.array([200.0]), patch=1)
    off, _ = colors.sample_colors(image, np.array([100.0]), np.array([200.0]), patch=1,
                                  center=(100.0, 100.0), band_px=None)
    assert tuple(single[0]) == tuple(off[0])


def test_many_points_at_once():
    """Vectorisation: many points, each with its own band width."""
    image = _stripe_image()
    n = 50
    u = np.full(n, 100.0)
    v = np.full(n, 200.0)
    band_px = np.linspace(2.0, 18.0, n)
    bgr, hsv = colors.sample_colors(image, u, v, patch=1, center=(100.0, 100.0),
                                    band_px=band_px, band_count=5)
    assert bgr.shape == (n, 3)
    assert colors.classify_hsv(hsv) == ['red'] * n


def test_only_label_ignores_a_bigger_blob_of_another_colour():
    """The real-world case: large red object in the room, small green pylon."""
    image = np.zeros((400, 400, 3), np.uint8)
    image[300:380, 40:160] = (0, 0, 255)      # large red distractor
    image[150:190, 250:280] = (0, 255, 0)     # small green pylon
    circle = (200.0, 200.0, 195.0)

    without = colors.find_color_blob(image, min_area=100, mask_circle=circle)
    assert without[2] == 'red'                   # the larger one wins

    with_label = colors.find_color_blob(image, min_area=100, mask_circle=circle, only_label='green')
    assert with_label[2] == 'green'
    assert 250 < with_label[0] < 280 and 150 < with_label[1] < 190


def test_max_area_rejects_the_oversized_background_object():
    image = np.zeros((400, 400, 3), np.uint8)
    image[300:380, 40:160] = (0, 0, 255)
    image[150:190, 250:280] = (0, 0, 255)     # second, small red area
    circle = (200.0, 200.0, 195.0)

    big = colors.find_color_blob(image, min_area=100, mask_circle=circle)
    small = colors.find_color_blob(image, min_area=100, mask_circle=circle, max_area=3000)
    assert big[3] > small[3]
    assert 250 < small[0] < 280


def test_blob_radii_bracket_the_pylon_in_the_image():
    """r_inner/r_outer enclose the blob -- basis for cal radial."""
    image = np.zeros((400, 400, 3), np.uint8)
    image[250:330, 190:210] = (0, 255, 0)     # vertical stripe below the centre
    circle = (200.0, 200.0, 195.0)

    blob = colors.find_color_blob(image, min_area=100, mask_circle=circle, only_label='green')
    _, _, label, _, r_inner, r_outer = blob
    assert label == 'green'
    assert r_inner < r_outer
    assert 45 < r_inner < 60        # top edge, about 50 px below the centre
    assert 125 < r_outer < 140     # foot point, about 130 px
