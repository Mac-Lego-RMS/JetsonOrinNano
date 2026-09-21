"""Selbsttest des Fisheye-Modells: laeuft ohne ROS, Kamera und Lidar."""

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
    """Ein Punkt genau ueber der Kamera liegt im Bildmittelpunkt."""
    calib = _calib()
    u, v, theta, _, in_fov = project(calib, np.array([[0.0, 0.0, 2.0]]))
    assert in_fov[0]
    assert math.isclose(theta[0], 0.0, abs_tol=1e-12)
    assert math.isclose(u[0], calib.cx, abs_tol=1e-9)
    assert math.isclose(v[0], calib.cy, abs_tol=1e-9)


def test_ground_points_land_just_outside_the_horizon_ring():
    """Die Lidar-Ebene liegt unter der Kamera -> theta knapp ueber 90 Grad."""
    calib = _calib(cam_z=0.05)
    pts = np.array([[1.0, 0.0, 0.0], [0.5, 0.0, 0.0], [0.2, 0.0, 0.0]])
    _, _, theta, _, in_fov = project(calib, pts)
    assert in_fov.all()
    assert (np.degrees(theta) > 90.0).all()
    # Je naeher der Punkt, desto steiler nach unten schaut die Kamera.
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
    """Kernstueck der Kalibrierung: aus Peilung + Pixel den Drehwinkel zurueckrechnen."""
    truth = _calib(yaw_deg=-37.5)
    bearings = np.radians([0.0, 45.0, 130.0, -80.0, 175.0])
    distances = np.array([0.4, 0.8, 1.1, 0.6, 0.9])

    pts = np.column_stack([distances * np.cos(bearings),
                           distances * np.sin(bearings),
                           np.full(bearings.size, truth.cam_z)])
    u_obs, v_obs, _, _, in_fov = project(truth, pts)
    assert in_fov.all()

    # Rueckrechnung wie in rotation_calibration._solve_yaw_closed_form
    guess = _calib(yaw_deg=0.0)
    phi_obs = np.arctan2(v_obs - guess.cy, u_obs - guess.cx)
    azimuth = np.arctan2(distances * np.sin(bearings) - guess.cam_y,
                         distances * np.cos(bearings) - guess.cam_x)
    deltas = (phi_obs - azimuth + np.pi) % (2 * np.pi) - np.pi
    yaw = math.degrees(math.atan2(np.sin(deltas).mean(), np.cos(deltas).mean()))

    assert math.isclose(yaw, truth.yaw_deg, abs_tol=1e-6)


def test_closed_form_yaw_is_independent_of_target_height():
    """Der Drehwinkel darf nicht davon abhaengen, wie hoch der Klotz ist."""
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
    """360 Grad Lidar muessen komplett auf dem Sensor landen."""
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
    """Der Regressionstest zum cam_z-Fehler.

    Liegt die Zielmarke exakt auf Kamerahoehe, ist theta immer 90 Grad und der
    Bildradius fuer jede Entfernung gleich -- dann ist cam_z aus Bildern nicht
    bestimmbar. Auf einer anderen Hoehe muss der Radius mit der Entfernung
    variieren, sonst traegt die radiale Richtung keine Information.
    """
    calib = _calib(cam_z=0.08)
    distances = np.array([0.3, 0.6, 1.2])

    def radii(height):
        pts = np.column_stack([distances, np.zeros(3), np.full(3, height)])
        u, v, _, _, _ = project(calib, pts)
        return np.hypot(u - calib.cx, v - calib.cy)

    auf_kamerahoehe = radii(0.08)
    assert np.allclose(auf_kamerahoehe, auf_kamerahoehe[0])   # degeneriert

    darunter = radii(0.0)
    assert np.ptp(darunter) > 5.0                             # informativ
    assert np.all(np.diff(darunter) < 0)   # ferner = flacher = kleinerer Radius


def test_closed_form_height_recovers_the_camera_height():
    """Kernstueck von cmd_height: cam_z aus Bildradius und Lidar-Abstand."""
    truth = _calib(cam_z=0.083, yaw_deg=17.0)
    target_height = 0.05
    bearings = np.radians([0.0, 70.0, -120.0, 160.0])
    distances = np.array([0.25, 0.40, 0.55, 0.30])

    pts = np.column_stack([distances * np.cos(bearings), distances * np.sin(bearings),
                           np.full(bearings.size, target_height)])
    u_obs, v_obs, _, _, in_fov = project(truth, pts)
    assert in_fov.all()

    # Rueckrechnung wie in rotation_calibration.cmd_height
    radius = np.hypot(u_obs - truth.cx, v_obs - truth.cy)
    theta = radius_to_theta(truth, radius)
    rho = np.hypot(distances * np.cos(bearings) - truth.cam_x,
                   distances * np.sin(bearings) - truth.cam_y)
    estimates = target_height - rho / np.tan(theta)

    assert np.allclose(estimates, truth.cam_z, atol=1e-9)


def test_height_error_stays_small_at_typical_distances():
    """Wie schlimm ist es, cam_z um 2 cm daneben zu haben?"""
    distances = np.array([0.3, 1.0, 2.0])
    pts = np.column_stack([distances, np.zeros(3), np.zeros(3)])

    u_a, v_a, _, _, _ = project(_calib(cam_z=0.05), pts)
    u_b, v_b, _, _, _ = project(_calib(cam_z=0.07), pts)
    shift = np.hypot(u_a - u_b, v_a - v_b)

    # Nah zaehlt der Fehler am meisten, fern laeuft er gegen null.
    assert shift[0] > shift[1] > shift[2]
    assert shift[0] < 25.0 and shift[2] < 3.0


def test_horizon_ring_radius_is_independent_of_distance():
    """sample_mode=horizon: auf Objektivhoehe wird der Bildradius konstant."""
    calib = _calib(cam_z=0.05, yaw_deg=23.0)
    distances = np.array([0.15, 0.4, 1.0, 2.5, 6.0])
    pts = np.column_stack([distances, np.zeros(5), np.full(5, calib.cam_z)])

    u, v, theta, _, _ = project(calib, pts)
    radius = np.hypot(u - calib.cx, v - calib.cy)

    assert np.allclose(np.degrees(theta), 90.0)
    assert np.allclose(radius, radius[0])
    assert math.isclose(radius[0], calib.focal_px * math.pi / 2, rel_tol=1e-9)


def test_horizon_ring_is_immune_to_range_and_cam_z_error():
    """Der eigentliche Gewinn: radial wirken weder Entfernungs- noch cam_z-Fehler."""
    bearing = math.radians(40.0)

    def sample(distance, cam_z):
        calib = _calib(cam_z=cam_z)
        point = np.array([[distance * math.cos(bearing), distance * math.sin(bearing), cam_z]])
        u, v, _, _, _ = project(calib, point)
        return u[0], v[0]

    referenz = sample(1.0, 0.05)
    # Lidar misst 20 cm daneben, cam_z ist 3 cm falsch -- Pixel bleibt derselbe.
    assert np.allclose(sample(1.2, 0.05), referenz, atol=1e-9)
    assert np.allclose(sample(1.0, 0.08), referenz, atol=1e-9)


def test_horizon_ring_hits_the_pylon_whenever_the_lens_is_below_its_top():
    """Die Bedingung: Objektiv zwischen Matte und Pylonenoberkante.

    Dann durchstoesst die Pylone die waagerechte Ebene durch die Linse und liegt
    in JEDER Entfernung auf dem Ring. Sitzt die Linse darueber, geht der Ring in
    jeder Entfernung daneben.
    """
    pylon_height = 0.10

    def ring_inside_pylon(lens_height, distance):
        # theta der Ober- und Unterkante der Pylone, von der Linse aus gesehen
        theta_top = math.atan2(distance, pylon_height - lens_height)
        theta_bottom = math.atan2(distance, -lens_height)
        return theta_top < math.pi / 2 < theta_bottom

    for distance in (0.2, 0.5, 1.0, 2.0, 5.0):
        for lens_height in (0.01, 0.05, 0.09):
            assert ring_inside_pylon(lens_height, distance)
        for lens_height in (0.105, 0.15):
            assert not ring_inside_pylon(lens_height, distance)


def test_mid_pylon_mounting_maximises_the_margin():
    """Auf halber Pylonenhoehe ist der Abstand zu beiden Kanten am groessten."""
    calib = _calib()
    pylon_height, distance = 0.10, 1.0

    def smallest_margin_px(lens_height):
        theta_top = math.atan2(distance, pylon_height - lens_height)
        theta_bottom = math.atan2(distance, -lens_height)
        return min(math.pi / 2 - theta_top, theta_bottom - math.pi / 2) * calib.focal_px

    mittig = smallest_margin_px(0.05)
    assert mittig > smallest_margin_px(0.02)
    assert mittig > smallest_margin_px(0.09)
    assert mittig > 9.0     # ca. 9.6 px Luft nach beiden Seiten auf 1 m


def test_depression_grows_the_ring_but_dives_with_distance():
    """Der Kegel greift in rho*tan(Winkel) Tiefe ab -- fern also viel tiefer."""
    calib = _calib(cam_z=0.05)
    depression = math.radians(1.0)
    distances = np.array([0.3, 1.0, 2.0])

    z = calib.cam_z - distances * math.tan(depression)
    pts = np.column_stack([distances, np.zeros(3), z])
    u, v, theta, _, _ = project(calib, pts)

    # Ein Kegel: konstanter Bildradius, aber groesser als der Horizontring.
    radius = np.hypot(u - calib.cx, v - calib.cy)
    assert np.allclose(np.degrees(theta), 91.0)
    assert np.allclose(radius, radius[0])
    assert radius[0] > calib.focal_px * math.pi / 2

    # ... erkauft mit einer Abgriffstiefe, die mit der Entfernung waechst.
    tiefe_cm = (calib.cam_z - z) * 100
    assert np.allclose(tiefe_cm, [0.52, 1.75, 3.49], atol=0.01)


def test_no_single_cone_works_when_the_lens_sits_above_the_pylon():
    """Linse ueber der Pylonenoberkante: nah und fern schliessen sich aus."""
    pylon_height, lens_height = 0.10, 0.13

    def usable_depressions(distance):
        # Tiefe unter der Linse muss zwischen Pylonenoberkante und Matte liegen.
        tief_min = lens_height - pylon_height
        tief_max = lens_height
        return (math.degrees(math.atan(tief_min / distance)),
                math.degrees(math.atan(tief_max / distance)))

    nah_lo, nah_hi = usable_depressions(0.3)
    fern_lo, fern_hi = usable_depressions(2.0)

    # Die beiden Fenster ueberlappen sich nicht -> ein Ring kann nicht beides.
    assert fern_hi < nah_lo

    # Sitzt die Linse dagegen IN der Pylone, deckt Winkel 0 jede Entfernung ab.
    for distance in (0.3, 1.0, 2.0, 5.0):
        theta_top = math.atan2(distance, pylon_height - 0.05)
        theta_bottom = math.atan2(distance, -0.05)
        assert theta_top < math.pi / 2 < theta_bottom


def test_band_width_shrinks_with_distance_and_stays_inside_the_pylon():
    """sample_band_m: das Band wird fern von selbst schmaler."""
    calib = _calib()
    band_m = 0.03
    distances = np.array([0.3, 1.0, 2.0])
    band_px = calib.focal_px * np.arctan(band_m / distances)

    assert np.all(np.diff(band_px) < 0)          # schrumpft mit der Entfernung

    # Muss innerhalb der Pylone bleiben: Linse mittig auf 5 cm, 10-cm-Pylone.
    marge_px = np.array([
        (math.pi / 2 - math.atan2(d, 0.05)) * calib.focal_px for d in distances])
    assert np.all(band_px <= marge_px + 1e-9)


def test_mirror_flips_the_azimuth():
    point = np.array([[1.0, 0.0, 0.0]])
    _, _, _, phi_normal, _ = project(_calib(yaw_deg=25.0), point)
    _, _, _, phi_mirror, _ = project(_calib(yaw_deg=25.0, mirror=True), point)
    assert math.isclose(phi_mirror[0], -phi_normal[0], abs_tol=1e-12)


def test_wrong_fov_puts_the_horizon_ring_at_the_wrong_radius():
    """Die Ursache fuer einen zu hoch sitzenden Ring: f haengt an der FOV."""
    radius_px = 452.0
    ringe = {}
    for fov in (270.0, 240.0, 220.0, 200.0, 180.0):
        ringe[fov] = _calib(radius_px=radius_px, fov_deg=fov).focal_px * math.pi / 2

    # Kleinere FOV -> groessere Brennweite -> Ring weiter aussen (tiefer im Raum).
    assert ringe[270.0] < ringe[240.0] < ringe[220.0] < ringe[200.0] < ringe[180.0]
    # Bei 180 Grad faellt der Horizont genau auf den Rand des Bildkreises.
    assert math.isclose(ringe[180.0], radius_px, rel_tol=1e-9)
    # 270 statt 220 angenommen schiebt den Ring um knapp 70 px nach innen.
    assert 60.0 < ringe[220.0] - ringe[270.0] < 75.0


def _pylon_edges(calib, lens_height, distances, pylon_height=0.10):
    """Bildradien von Fuss- und Oberkante -- ueber project(), nicht ueber eine
    eigene Formel. Sonst prueft der Test nur seine eigene Herleitung gegen sich
    selbst (genau daran ist die erste Fassung vorbeigelaufen)."""
    fuss = np.column_stack([distances, np.zeros(distances.size),
                            np.full(distances.size, calib.cam_z - lens_height)])
    kopf = np.column_stack([distances, np.zeros(distances.size),
                            np.full(distances.size,
                                    calib.cam_z - lens_height + pylon_height)])
    u_f, v_f, _, _, _ = project(calib, fuss)
    u_k, v_k, _, _, _ = project(calib, kopf)
    return (np.hypot(u_f - calib.cx, v_f - calib.cy),
            np.hypot(u_k - calib.cx, v_k - calib.cy))


def test_pylon_base_radius_shrinks_with_distance():
    """Grundtatsache, an der die alte Formel scheiterte: nah = grosser Radius."""
    calib = _calib(cam_z=0.05)
    distances = np.array([0.2, 0.5, 1.0, 2.0])
    r_fuss, r_kopf = _pylon_edges(calib, lens_height=0.07, distances=distances)

    assert np.all(np.diff(r_fuss) < 0)       # ferner -> kleinerer Radius
    assert np.all(r_kopf < r_fuss)           # Oberkante liegt weiter innen

    # Der Fusspunkt liegt immer unter dem Horizont, und zwar umso tiefer, je
    # naeher die Pylone steht: theta = atan2(rho, -L), also 109 Grad bei 0.2 m
    # und 7 cm Objektivhoehe, gegen 90 Grad in der Ferne.
    theta_fuss = np.degrees(r_fuss / calib.focal_px)
    assert np.all(theta_fuss > 90.0)
    assert abs(theta_fuss[0] - math.degrees(math.atan2(0.2, -0.07))) < 0.5
    assert theta_fuss[-1] < 93.0             # auf 2 m praktisch am Horizont


def test_radial_calibration_recovers_focal_length_and_lens_height():
    """cmd_radial: f und Objektivhoehe aus Fuss- und Kopfpunkt der Pylone."""
    from scipy.optimize import least_squares

    radius_px, pylon = 452.0, 0.10
    fov_truth = 212.0                                  # echte FOV, nicht die 270
    f_truth = radius_px / math.radians(fov_truth / 2)
    lens_truth = 0.07
    rho = np.array([0.2, 0.35, 0.6, 1.0, 1.5])

    calib = _calib(radius_px=radius_px, fov_deg=fov_truth, cam_z=0.05)
    r_fuss, r_kopf = _pylon_edges(calib, lens_truth, rho, pylon)
    rng = np.random.default_rng(1)
    r_fuss = r_fuss + rng.normal(0, 2.0, rho.size)     # 2 px Segmentierrauschen
    r_kopf = r_kopf + rng.normal(0, 2.0, rho.size)

    # Nur der Fusspunkt geht in den Fit -- so macht es cmd_radial.
    def residuals(x):
        return x[0] * np.arctan2(rho, -x[1]) - r_fuss

    start = [radius_px / math.radians(135.0), 0.05]    # falsche 270-Grad-Annahme
    f_px, lens = least_squares(residuals, start, bounds=([1, 0.001], [10000, 0.5])).x

    assert abs(f_px - f_truth) < 3.0
    assert abs(lens - lens_truth) < 0.005
    # Und das Ergebnis muss physikalisch moeglich sein.
    assert math.degrees(2 * radius_px / f_px) < 360.0
    assert r_kopf[0] < r_fuss[0]       # Oberkante liegt weiter innen


def test_a_short_coloured_area_ruins_the_fit_if_the_top_is_used():
    """Warum cmd_radial nur den Fusspunkt nimmt.

    Am echten Roboter reichte die gruene Flaeche nur 5.3 cm hoch, nicht die
    angenommenen 10 cm. Nimmt man die Oberkante mit, verzieht das den Fit.
    """
    from scipy.optimize import least_squares

    radius_px, angenommen, echt = 452.0, 0.10, 0.053
    f_truth, lens_truth = radius_px / math.radians(212.0 / 2), 0.03
    rho = np.array([0.23, 0.24, 0.51, 0.55])

    calib = _calib(radius_px=radius_px, fov_deg=212.0, cam_z=0.05)
    r_fuss, r_kopf = _pylon_edges(calib, lens_truth, rho, echt)   # Farbe endet frueher

    def nur_fuss(x):
        return x[0] * np.arctan2(rho, -x[1]) - r_fuss

    def mit_kopf(x):
        return np.concatenate([x[0] * np.arctan2(rho, -x[1]) - r_fuss,
                               x[0] * np.arctan2(rho, angenommen - x[1]) - r_kopf])

    bounds = ([1, 0.001], [10000, 0.5])
    f_gut = least_squares(nur_fuss, [200.0, 0.05], bounds=bounds).x[0]
    ergebnis = least_squares(mit_kopf, [200.0, 0.05], bounds=bounds)
    f_schlecht, rms_schlecht = ergebnis.x[0], np.sqrt(np.mean(ergebnis.fun ** 2))

    assert abs(f_gut - f_truth) < 1.0                 # Fusspunkt trifft
    assert abs(f_schlecht - f_truth) > 10.0           # mit falscher Kopfhoehe nicht
    assert rms_schlecht > 5.0                         # und faellt durch den RMS auf


def test_the_old_swapped_formula_is_rejected():
    """Regression: rho und L vertauscht ergibt eine unmoegliche FOV."""
    from scipy.optimize import least_squares

    radius_px, pylon, rho = 452.0, 0.10, np.array([0.2, 0.35, 0.6, 1.0, 1.5])
    calib = _calib(radius_px=radius_px, fov_deg=212.0, cam_z=0.05)
    r_fuss, r_kopf = _pylon_edges(calib, 0.07, rho, pylon)

    def falsch(x):
        f_px, lens = x
        return np.concatenate([
            f_px * (np.pi - np.arctan2(lens, rho)) - r_fuss,
            f_px * (np.pi - np.arctan2(lens - pylon, rho)) - r_kopf,
        ])

    f_px, _ = least_squares(falsch, [radius_px / math.radians(135.0), 0.05],
                            bounds=([1, 0.001], [10000, 1.0])).x
    assert math.degrees(2 * radius_px / f_px) > 360.0   # unmoeglich -> war der Fehler
