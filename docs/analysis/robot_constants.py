"""Robot and field constants used by the analysis toolkit.

Everything here is COPIED from the ROS 2 sources (src/ is not importable
without a ROS installation: src/ekf/ekf/field_map.py imports ekf.ekf, the nodes
import rclpy). Each constant names the file it comes from -- if the source
changes, update the value here.

Frames
------
field frame  origin at the field centre, X east, Y north (field_map.py).
map frame    "start-anchored": the frame of /ekf/odom, /obstacles and
             /corner_geometry. It is the field frame expressed relative to the
             start pose (x_start, y_start, theta_start), see field_to_map().
"""
import math

import numpy as np

# --------------------------------------------------------------------------
# Calibration constants
# --------------------------------------------------------------------------
# src/ekf/ekf/ekf_node.py: GYRO_SCALE. The raw BNO055 yaw rate is multiplied by
# this to get the REP-103 yaw rate (CCW positive). Sign flips the axis,
# magnitude 0.9674 is the 5 x 360 deg calibration.
GYRO_SCALE = -0.9674

# src/ekf/ekf/ekf.py: DeadReckoningEKF.r_eff, metres of travel per radian of the
# drive output shaft (the encoder velocity on /esp_serial_bridge/joint_states is
# in rad/s of that shaft). Comment in ekf.py: "Strecken-Kalibrierung
# (2,41 m / 10431 Ticks)".
R_EFF = 0.0150

# Radians of output shaft per encoder "tick" implied by the ekf.py calibration
# note above (2.41 m / 10431 ticks / r_eff). ASSUMPTION -- confirm what the
# team counted as a "tick" when filling m3_encoder_distance.csv.
RAD_PER_TICK_DEFAULT = 2.41 / 10431.0 / R_EFF

# src/ekf/ekf/round1_controller_node.py: parameter steer_dead_time (s) and
# steer_gain_pred (measured yaw rate per commanded yaw rate).
STEER_DEAD_TIME = 0.260
STEER_GAIN_PRED = 0.84

# src/ekf/ekf/round1_controller_node.py, _einparken_fertig(): the parking report
# converts the heading error to an axle difference with this wheelbase
# (achsdiff = 0.105 * sin(kurs)). The WRO rule allows at most 2 cm.
PARK_WHEELBASE = 0.105
PARK_AXLE_RULE_CM = 2.0

# src/esp_bridge/esp_bridge/esp_serial_bridge.py: parameter wheelbase (Ackermann
# inverse delta = atan(L * omega / v)); also "wheelbase" in steer_calib.json.
BRIDGE_WHEELBASE = 0.10

# src/ekf/ekf/wall_extraction.py: LIDAR_OFFSET_X. The LiDAR is mounted turned
# by 180 deg: x_base = -x_lidar + LIDAR_OFFSET_X, y_base = -y_lidar
# (src/ekf/ekf/obstacle_detection.py, _cloud_arrays()).
LIDAR_OFFSET_X = 0.1101

# src/robot_msgs/msg/Obstacle.msg colour codes.
COLOR_UNKNOWN, COLOR_RED, COLOR_GREEN = 0, 1, 2
COLOR_NAMES = {COLOR_UNKNOWN: 'unknown', COLOR_RED: 'red', COLOR_GREEN: 'green'}

# src/ekf/ekf/obstacle_detection.py: pillars are 44 mm wide.
PILLAR_EDGE = 0.044

# src/camera_lidar_fusion/camera_lidar_fusion/colors.py CLOUD_BGR, packed as
# 0xRRGGBB by lidar_pixel_mapper._publish_cloud() into the 'rgb' float32 field
# of /camera_lidar/colored_scan (only in cloud_color_mode 'label', the default).
CLOUD_LABEL_RGB = {
    0xFF0000: 'red',       # 'rot'
    0x00FF00: 'green',     # 'gruen'
    0xFF00FF: 'magenta',   # 'magenta' (parking bay walls)
    0x2D2D2D: 'black',     # 'schwarz' (the black wall band)
    0x555555: 'unknown',   # 'unbekannt'
}

# --------------------------------------------------------------------------
# Field geometry -- src/ekf/ekf/field_map.py
# --------------------------------------------------------------------------
OUTER_HALF = 1.5     # outer wall 3 x 3 m
INNER_HALF = 0.5     # inner wall 1 x 1 m (obstacle challenge, fixed)
SEAT_OUTER_INSET = 0.4
SEAT_INNER_INSET = 0.4
SEAT_ROWS = (-0.5, 0.0, 0.5)

# Start poses in the FIELD frame (x, y, theta), field_map.py.
START_POSES = {
    'cw_pos1': (0.05, 1.0, 0.0),
    'cw_pos2': (-0.45, 1.0, 0.0),
    'ccw_pos1': (-0.05, 1.0, math.pi),
    'ccw_pos2': (0.45, 1.0, math.pi),
}


def square(half):
    """Corners of an axis-aligned square (closed polygon, 5 x 2 array)."""
    c = np.array([[-half, -half], [half, -half], [half, half],
                  [-half, half], [-half, -half]], dtype=float)
    return c


def _rot90(p, k):
    x, y = p
    for _ in range(k % 4):
        x, y = -y, x
    return np.array([x, y])


def seats_field():
    """The 24 obstacle seats in the field frame.

    Returns a list of dicts {'id', 'p', 'column', 'row', 'straight'} in the
    same order and with the same id rule as scan_processor_node._seat_id():
    id = straight * 6 + row * 2 + (0 outer / 1 inner). Straight k is the west
    straight rotated k times by 90 deg CCW (field_map.obstacle_seats_field()).
    """
    x_outer = -OUTER_HALF + SEAT_OUTER_INSET      # -1.1
    x_inner = -INNER_HALF - SEAT_INNER_INSET      # -0.9
    base = []
    for y in SEAT_ROWS:
        base.append((np.array([x_outer, y]), 'outer'))
        base.append((np.array([x_inner, y]), 'inner'))
    out = []
    for k in range(4):
        for i, (p, col) in enumerate(base):
            row = i // 2
            out.append({'id': k * 6 + row * 2 + (0 if col == 'outer' else 1),
                        'p': _rot90(p, k), 'column': col, 'row': row,
                        'straight': k})
    return out


def field_to_map(points, start_pose):
    """Field-frame points (N x 2) -> start-anchored map frame.

    Same maths as field_map._transform_point(): translate by the start
    position, rotate by -theta_start.
    """
    pts = np.atleast_2d(np.asarray(points, dtype=float))
    xs, ys, th = start_pose
    dx, dy = pts[:, 0] - xs, pts[:, 1] - ys
    c, s = math.cos(th), math.sin(th)
    return np.column_stack([c * dx + s * dy, -s * dx + c * dy])


def transform_from_corners(corners_map):
    """Field -> map transform from the 4 outer corners seen in the map frame
    (/corner_geometry). Returns a function points(N x 2) -> map points.

    The field is 4-fold symmetric (outer box, inner box and the 24 seats all
    map onto themselves under a 90 deg rotation), so the rotation is only
    needed modulo 90 deg: it is taken from the direction of one box edge.
    """
    c = np.asarray(corners_map, dtype=float)[:, :2]
    centre = c.mean(axis=0)
    edge = c[1] - c[0]
    phi = math.atan2(edge[1], edge[0]) % (math.pi / 2)
    cp, sp = math.cos(phi), math.sin(phi)
    rot = np.array([[cp, -sp], [sp, cp]])

    def to_map(points):
        pts = np.atleast_2d(np.asarray(points, dtype=float))
        return pts @ rot.T + centre
    return to_map


def transform_from_start_pose(start_pose):
    """Field -> map transform for a known start pose (fallback when the bag
    has no /corner_geometry)."""
    return lambda points: field_to_map(points, start_pose)


def wrap(a):
    """Wrap an angle (rad, scalar or array) to [-pi, pi)."""
    return (np.asarray(a) + np.pi) % (2 * np.pi) - np.pi
