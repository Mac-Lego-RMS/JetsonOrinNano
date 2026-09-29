"""Self-test of the fisheye model: runs without ROS, camera and lidar."""

import math

import numpy as np

from camera_lidar_fusion.fisheye_model import (
    FisheyeCalib, project, radius_to_theta, scan_to_points,
)


def _calib(**kwargs):
    base = dict(cx=678.5, cy=451.0, radius_px=452.0, fov_deg=270.0, cam_z=0.05)
    base.update(kwargs)
    return FisheyeCalib(**base)


def test_focal_from_circle():
    calib = _calib()
    assert math.isclose(calib.focal_px, 452.0 / math.radians(135.0), rel_tol=1e-9)


def test_point_straight_up_hits_the_centre():
    """A point exactly above the camera lies at the image centre."""
    calib = _calib()
    u, v, theta, _, in_fov = project(calib, np.array([[0.0, 0.0, 2.0]]))
    assert in_fov[0]
    assert math.isclose(theta[0], 0.0, abs_tol=1e-12)
    assert math.isclose(u[0], calib.cx, abs_tol=1e-9)
    assert math.isclose(v[0], calib.cy, abs_tol=1e-9)


def test_ground_points_land_just_outside_the_horizon_ring():
    """The lidar plane lies below the camera -> theta just above 90 degrees."""
    calib = _calib(cam_z=0.05)
    pts = np.array([[1.0, 0.0, 0.0], [0.5, 0.0, 0.0], [0.2, 0.0, 0.0]])
    _, _, theta, _, in_fov = project(calib, pts)
    assert in_fov.all()
    assert (np.degrees(theta) > 90.0).all()
    # The closer the point, the steeper the camera looks down.
    assert np.all(np.diff(theta) > 0)
    assert math.isclose(math.degrees(theta[0]), 90.0 + math.degrees(math.atan(0.05)), abs_tol=1e-9)


def test_yaw_rotates_the_image_azimuth_one_to_one():
    calib_zero = _calib(yaw_deg=0.0)
    calib_yaw = _calib(yaw_deg=30.0)
    point = np.array([[1.0, 0.0, 0.0]])

    _, _, _, phi_zero, _ = project(calib_zero, point)
    _, _, _, phi_yaw, _ = project(calib_yaw, point)
    assert math.isclose(math.degrees(phi_yaw[0] - phi_zero[0]), 30.0, abs_tol=1e-9)


def test_closed_form_yaw_recovers_a_synthetic_rotation():
    """Core of the calibration: compute the rotation angle back from bearing + pixel."""
    truth = _calib(yaw_deg=-37.5)
    bearings = np.radians([0.0, 45.0, 130.0, -80.0, 175.0])
    distances = np.array([0.4, 0.8, 1.1, 0.6, 0.9])

    pts = np.column_stack([distances * np.cos(bearings),
                           distances * np.sin(bearings),
                           np.full(bearings.size, truth.cam_z)])
    u_obs, v_obs, _, _, in_fov = project(truth, pts)
    assert in_fov.all()

    # Back-calculation as in rotation_calibration._solve_yaw_closed_form
    guess = _calib(yaw_deg=0.0)
    phi_obs = np.arctan2(v_obs - guess.cy, u_obs - guess.cx)
    azimuth = np.arctan2(distances * np.sin(bearings) - guess.cam_y,
                         distances * np.cos(bearings) - guess.cam_x)
    deltas = (phi_obs - azimuth + np.pi) % (2 * np.pi) - np.pi
    yaw = math.degrees(math.atan2(np.sin(deltas).mean(), np.cos(deltas).mean()))

    assert math.isclose(yaw, truth.yaw_deg, abs_tol=1e-6)


def test_closed_form_yaw_is_independent_of_target_height():
    """The rotation angle must not depend on how tall the block is."""
    truth = _calib(yaw_deg=22.0)
    bearings = np.radians([10.0, 100.0, -140.0])
    distances = np.array([0.5, 0.7, 1.0])

    yaws = []
    for height in (0.0, 0.05, 0.12):
        pts = np.column_stack([distances * np.cos(bearings),
                               distances * np.sin(bearings),
                               np.full(bearings.size, height)])
        u_obs, v_obs, _, _, _ = project(truth, pts)
        phi_obs = np.arctan2(v_obs - truth.cy, u_obs - truth.cx)
        azimuth = bearings
        deltas = (phi_obs - azimuth + np.pi) % (2 * np.pi) - np.pi
        yaws.append(math.degrees(math.atan2(np.sin(deltas).mean(), np.cos(deltas).mean())))

    assert all(math.isclose(y, truth.yaw_deg, abs_tol=1e-6) for y in yaws)


def test_a_full_scan_stays_inside_the_image_circle():
    """360 degrees of lidar must land completely on the sensor."""
    calib = _calib()
    ranges = np.full(360, 1.0)
    pts, _ = scan_to_points(ranges, -math.pi, math.radians(1.0), z_offset=0.03)
    u, v, _, _, in_fov = project(calib, pts)

    assert in_fov.all()
    radius = np.hypot(u - calib.cx, v - calib.cy)
    assert radius.max() <= calib.radius_px
    assert (u >= 0).all() and (u < calib.image_width).all()
    assert (v >= 0).all() and (v < calib.image_height).all()


def test_radius_to_theta_inverts_theta_to_radius():
    calib = _calib()
    theta = np.radians([0.0, 30.0, 90.0, 120.0, 135.0])
    _, _, _, _, _ = project(calib, np.array([[1.0, 0.0, 0.0]]))
    radius = calib.focal_px * theta
    assert np.allclose(radius_to_theta(calib, radius), theta, atol=1e-9)


def test_radius_to_theta_also_inverts_the_polynomial_model():
    calib = _calib(poly_coeffs=[190.0, -6.0, 0.8])
    theta = np.radians([5.0, 45.0, 95.0, 130.0])
    from camera_lidar_fusion.fisheye_model import theta_to_radius
    assert np.allclose(radius_to_theta(calib, theta_to_radius(calib, theta)), theta, atol=1e-3)


def test_target_height_changes_the_image_radius():
    """The regression test for the cam_z bug.

    If the target mark lies exactly at camera height, theta is always 90 degrees
    and the image radius the same for every distance -- then cam_z cannot be
    determined from images. At another height the radius must vary with the
    distance, otherwise the radial direction carries no information.
    """
    calib = _calib(cam_z=0.08)
    distances = np.array([0.3, 0.6, 1.2])

    def radii(height):
        pts = np.column_stack([distances, np.zeros(3), np.full(3, height)])
        u, v, _, _, _ = project(calib, pts)
        return np.hypot(u - calib.cx, v - calib.cy)

    at_cam_height = radii(0.08)
    assert np.allclose(at_cam_height, at_cam_height[0])   # degenerate

    below = radii(0.0)
    assert np.ptp(below) > 5.0                             # informative
    assert np.all(np.diff(below) < 0)   # further = flatter = smaller radius


def test_closed_form_height_recovers_the_camera_height():
    """Core of cmd_height: cam_z from image radius and lidar distance."""
    truth = _calib(cam_z=0.083, yaw_deg=17.0)
    target_height = 0.05
    bearings = np.radians([0.0, 70.0, -120.0, 160.0])
    distances = np.array([0.25, 0.40, 0.55, 0.30])

    pts = np.column_stack([distances * np.cos(bearings), distances * np.sin(bearings),
                           np.full(bearings.size, target_height)])
    u_obs, v_obs, _, _, in_fov = project(truth, pts)
    assert in_fov.all()

    # Back-calculation as in rotation_calibration.cmd_height
    radius = np.hypot(u_obs - truth.cx, v_obs - truth.cy)
    theta = radius_to_theta(truth, radius)
    rho = np.hypot(distances * np.cos(bearings) - truth.cam_x,
                   distances * np.sin(bearings) - truth.cam_y)
    estimates = target_height - rho / np.tan(theta)

    assert np.allclose(estimates, truth.cam_z, atol=1e-9)


def test_height_error_stays_small_at_typical_distances():
    """How bad is it to have cam_z off by 2 cm?"""
    distances = np.array([0.3, 1.0, 2.0])
    pts = np.column_stack([distances, np.zeros(3), np.zeros(3)])

    u_a, v_a, _, _, _ = project(_calib(cam_z=0.05), pts)
    u_b, v_b, _, _, _ = project(_calib(cam_z=0.07), pts)
    shift = np.hypot(u_a - u_b, v_a - v_b)

    # Near the error counts most, far away it tends to zero.
    assert shift[0] > shift[1] > shift[2]
    assert shift[0] < 25.0 and shift[2] < 3.0


def test_horizon_ring_radius_is_independent_of_distance():
    """sample_mode=horizon: at lens height the image radius becomes constant."""
    calib = _calib(cam_z=0.05, yaw_deg=23.0)
    distances = np.array([0.15, 0.4, 1.0, 2.5, 6.0])
    pts = np.column_stack([distances, np.zeros(5), np.full(5, calib.cam_z)])

    u, v, theta, _, _ = project(calib, pts)
    radius = np.hypot(u - calib.cx, v - calib.cy)

    assert np.allclose(np.degrees(theta), 90.0)
    assert np.allclose(radius, radius[0])
    assert math.isclose(radius[0], calib.focal_px * math.pi / 2, rel_tol=1e-9)


def test_horizon_ring_is_immune_to_range_and_cam_z_error():
    """The actual gain: neither range nor cam_z errors have a radial effect."""
    bearing = math.radians(40.0)

    def sample(distance, cam_z):
        calib = _calib(cam_z=cam_z)
        point = np.array([[distance * math.cos(bearing), distance * math.sin(bearing), cam_z]])
        u, v, _, _, _ = project(calib, point)
        return u[0], v[0]

    reference = sample(1.0, 0.05)
    # Lidar measures 20 cm off, cam_z is 3 cm wrong -- the pixel stays the same.
    assert np.allclose(sample(1.2, 0.05), reference, atol=1e-9)
    assert np.allclose(sample(1.0, 0.08), reference, atol=1e-9)


def test_horizon_ring_hits_the_pylon_whenever_the_lens_is_below_its_top():
    """The condition: lens between the mat and the pylon top.

    Then the pylon pierces the horizontal plane through the lens and lies on the
    ring at EVERY distance. If the lens sits above it, the ring misses at every
    distance.
    """
    pylon_height = 0.10

    def ring_inside_pylon(lens_height, distance):
        # theta of the top and bottom edge of the pylon, seen from the lens
        theta_top = math.atan2(distance, pylon_height - lens_height)
        theta_bottom = math.atan2(distance, -lens_height)
        return theta_top < math.pi / 2 < theta_bottom

    for distance in (0.2, 0.5, 1.0, 2.0, 5.0):
        for lens_height in (0.01, 0.05, 0.09):
            assert ring_inside_pylon(lens_height, distance)
        for lens_height in (0.105, 0.15):
            assert not ring_inside_pylon(lens_height, distance)


def test_mid_pylon_mounting_maximises_the_margin():
    """At half the pylon height the distance to both edges is largest."""
    calib = _calib()
    pylon_height, distance = 0.10, 1.0

    def smallest_margin_px(lens_height):
        theta_top = math.atan2(distance, pylon_height - lens_height)
        theta_bottom = math.atan2(distance, -lens_height)
        return min(math.pi / 2 - theta_top, theta_bottom - math.pi / 2) * calib.focal_px

    centred = smallest_margin_px(0.05)
    assert centred > smallest_margin_px(0.02)
    assert centred > smallest_margin_px(0.09)
    assert centred > 9.0     # approx. 9.6 px margin to both sides at 1 m


def test_depression_grows_the_ring_but_dives_with_distance():
    """The cone samples at a depth of rho*tan(angle) -- so far away much lower."""
    calib = _calib(cam_z=0.05)
    depression = math.radians(1.0)
    distances = np.array([0.3, 1.0, 2.0])

    z = calib.cam_z - distances * math.tan(depression)
    pts = np.column_stack([distances, np.zeros(3), z])
    u, v, theta, _, _ = project(calib, pts)

    # A cone: constant image radius, but larger than the horizon ring.
    radius = np.hypot(u - calib.cx, v - calib.cy)
    assert np.allclose(np.degrees(theta), 91.0)
    assert np.allclose(radius, radius[0])
    assert radius[0] > calib.focal_px * math.pi / 2

    # ... paid for with a sampling depth that grows with the distance.
    depth_cm = (calib.cam_z - z) * 100
    assert np.allclose(depth_cm, [0.52, 1.75, 3.49], atol=0.01)


def test_no_single_cone_works_when_the_lens_sits_above_the_pylon():
    """Lens above the pylon top: near and far exclude each other."""
    pylon_height, lens_height = 0.10, 0.13

    def usable_depressions(distance):
        # The depth below the lens must lie between the pylon top and the mat.
        depth_min = lens_height - pylon_height
        depth_max = lens_height
        return (math.degrees(math.atan(depth_min / distance)),
                math.degrees(math.atan(depth_max / distance)))

    near_lo, near_hi = usable_depressions(0.3)
    far_lo, far_hi = usable_depressions(2.0)

    # The two windows do not overlap -> one ring cannot do both.
    assert far_hi < near_lo

    # If the lens sits IN the pylon instead, angle 0 covers every distance.
    for distance in (0.3, 1.0, 2.0, 5.0):
        theta_top = math.atan2(distance, pylon_height - 0.05)
        theta_bottom = math.atan2(distance, -0.05)
        assert theta_top < math.pi / 2 < theta_bottom


def test_band_width_shrinks_with_distance_and_stays_inside_the_pylon():
    """sample_band_m: far away the band gets narrower by itself."""
    calib = _calib()
    band_m = 0.03
    distances = np.array([0.3, 1.0, 2.0])
    band_px = calib.focal_px * np.arctan(band_m / distances)

    assert np.all(np.diff(band_px) < 0)          # shrinks with the distance

    # Must stay inside the pylon: lens centred at 5 cm, 10 cm pylon.
    margin_px = np.array([
        (math.pi / 2 - math.atan2(d, 0.05)) * calib.focal_px for d in distances])
    assert np.all(band_px <= margin_px + 1e-9)


def test_mirror_flips_the_azimuth():
    point = np.array([[1.0, 0.0, 0.0]])
    _, _, _, phi_normal, _ = project(_calib(yaw_deg=25.0), point)
    _, _, _, phi_mirror, _ = project(_calib(yaw_deg=25.0, mirror=True), point)
    assert math.isclose(phi_mirror[0], -phi_normal[0], abs_tol=1e-12)


def test_wrong_fov_puts_the_horizon_ring_at_the_wrong_radius():
    """The cause of a ring sitting too high: f depends on the FOV."""
    radius_px = 452.0
    rings = {}
    for fov in (270.0, 240.0, 220.0, 200.0, 180.0):
        rings[fov] = _calib(radius_px=radius_px, fov_deg=fov).focal_px * math.pi / 2

    # Smaller FOV -> larger focal length -> ring further out (lower in the room).
    assert rings[270.0] < rings[240.0] < rings[220.0] < rings[200.0] < rings[180.0]
    # At 180 degrees the horizon falls exactly on the edge of the image circle.
    assert math.isclose(rings[180.0], radius_px, rel_tol=1e-9)
    # Assuming 270 instead of 220 shifts the ring almost 70 px inwards.
    assert 60.0 < rings[220.0] - rings[270.0] < 75.0


def _pylon_edges(calib, lens_height, distances, pylon_height=0.10):
    """Image radii of foot and top edge -- via project(), not via a formula of
    its own. Otherwise the test only checks its own derivation against itself
    (exactly that is how the first version missed the bug)."""
    foot = np.column_stack([distances, np.zeros(distances.size),
                            np.full(distances.size, calib.cam_z - lens_height)])
    top = np.column_stack([distances, np.zeros(distances.size),
                            np.full(distances.size,
                                    calib.cam_z - lens_height + pylon_height)])
    u_f, v_f, _, _, _ = project(calib, foot)
    u_k, v_k, _, _, _ = project(calib, top)
    return (np.hypot(u_f - calib.cx, v_f - calib.cy),
            np.hypot(u_k - calib.cx, v_k - calib.cy))


def test_pylon_base_radius_shrinks_with_distance():
    """The basic fact on which the old formula failed: near = large radius."""
    calib = _calib(cam_z=0.05)
    distances = np.array([0.2, 0.5, 1.0, 2.0])
    r_foot, r_top = _pylon_edges(calib, lens_height=0.07, distances=distances)

    assert np.all(np.diff(r_foot) < 0)       # further -> smaller radius
    assert np.all(r_top < r_foot)           # top edge lies further in

    # The foot point always lies below the horizon, and the lower the closer
    # the pylon stands: theta = atan2(rho, -L), i.e. 109 degrees at 0.2 m
    # and 7 cm lens height, towards 90 degrees far away.
    theta_foot = np.degrees(r_foot / calib.focal_px)
    assert np.all(theta_foot > 90.0)
    assert abs(theta_foot[0] - math.degrees(math.atan2(0.2, -0.07))) < 0.5
    assert theta_foot[-1] < 93.0             # practically at the horizon at 2 m


def test_radial_calibration_recovers_focal_length_and_lens_height():
    """cmd_radial: f and lens height from the foot and top point of the pylon."""
    from scipy.optimize import least_squares

    radius_px, pylon = 452.0, 0.10
    fov_truth = 212.0                                  # real FOV, not the 270
    f_truth = radius_px / math.radians(fov_truth / 2)
    lens_truth = 0.07
    rho = np.array([0.2, 0.35, 0.6, 1.0, 1.5])

    calib = _calib(radius_px=radius_px, fov_deg=fov_truth, cam_z=0.05)
    r_foot, r_top = _pylon_edges(calib, lens_truth, rho, pylon)
    rng = np.random.default_rng(1)
    r_foot = r_foot + rng.normal(0, 2.0, rho.size)     # 2 px segmentation noise
    r_top = r_top + rng.normal(0, 2.0, rho.size)

    # Only the foot point goes into the fit -- that is how cmd_radial does it.
    def residuals(x):
        return x[0] * np.arctan2(rho, -x[1]) - r_foot

    start = [radius_px / math.radians(135.0), 0.05]    # wrong 270 degree assumption
    f_px, lens = least_squares(residuals, start, bounds=([1, 0.001], [10000, 0.5])).x

    assert abs(f_px - f_truth) < 3.0
    assert abs(lens - lens_truth) < 0.005
    # And the result must be physically possible.
    assert math.degrees(2 * radius_px / f_px) < 360.0
    assert r_top[0] < r_foot[0]       # top edge lies further in


def test_a_short_coloured_area_ruins_the_fit_if_the_top_is_used():
    """Why cmd_radial only takes the foot point.

    On the real robot the green area only reached 5.3 cm high, not the
    assumed 10 cm. Including the top edge distorts the fit.
    """
    from scipy.optimize import least_squares

    radius_px, assumed, real = 452.0, 0.10, 0.053
    f_truth, lens_truth = radius_px / math.radians(212.0 / 2), 0.03
    rho = np.array([0.23, 0.24, 0.51, 0.55])

    calib = _calib(radius_px=radius_px, fov_deg=212.0, cam_z=0.05)
    r_foot, r_top = _pylon_edges(calib, lens_truth, rho, real)   # colour ends earlier

    def foot_only(x):
        return x[0] * np.arctan2(rho, -x[1]) - r_foot

    def with_top(x):
        return np.concatenate([x[0] * np.arctan2(rho, -x[1]) - r_foot,
                               x[0] * np.arctan2(rho, assumed - x[1]) - r_top])

    bounds = ([1, 0.001], [10000, 0.5])
    f_good = least_squares(foot_only, [200.0, 0.05], bounds=bounds).x[0]
    result = least_squares(with_top, [200.0, 0.05], bounds=bounds)
    f_bad, rms_bad = result.x[0], np.sqrt(np.mean(result.fun ** 2))

    assert abs(f_good - f_truth) < 1.0                 # foot point hits
    assert abs(f_bad - f_truth) > 10.0           # with a wrong top height it does not
    assert rms_bad > 5.0                         # and stands out through the RMS


def test_the_old_swapped_formula_is_rejected():
    """Regression: rho and L swapped gives an impossible FOV."""
    from scipy.optimize import least_squares

    radius_px, pylon, rho = 452.0, 0.10, np.array([0.2, 0.35, 0.6, 1.0, 1.5])
    calib = _calib(radius_px=radius_px, fov_deg=212.0, cam_z=0.05)
    r_foot, r_top = _pylon_edges(calib, 0.07, rho, pylon)

    def wrong(x):
        f_px, lens = x
        return np.concatenate([
            f_px * (np.pi - np.arctan2(lens, rho)) - r_foot,
            f_px * (np.pi - np.arctan2(lens - pylon, rho)) - r_top,
        ])

    f_px, _ = least_squares(wrong, [radius_px / math.radians(135.0), 0.05],
                            bounds=([1, 0.001], [10000, 1.0])).x
    assert math.degrees(2 * radius_px / f_px) > 360.0   # impossible -> that was the bug
