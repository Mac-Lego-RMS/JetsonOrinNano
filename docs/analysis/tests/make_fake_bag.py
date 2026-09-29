#!/usr/bin/env python3
"""Write a realistic SYNTHETIC ROS 2 Humble bag for testing the toolkit.

    python3 tests/make_fake_bag.py OUT_DIR [--name parken_test_7] [--lang de|en]
                                   [--straight-ect] [--lag 0.25] [--laps 3]

What is inside (all topics the tools use, same names/types as on the robot):
  * 3 laps (default) clockwise around the 3 x 3 m field from start pose
    cw_pos1, lane centre line with 0.5 m corner radius, 0.6 m/s;
    /ekf/odom (50 Hz), /bno055/imu (100 Hz, RAW sign/scale, i.e. divided by
    GYRO_SCALE), /esp_serial_bridge/joint_states + speed (100 Hz);
  * /cmd_vel (30 Hz) whose angular.z LEADS the measured yaw rate by exactly
    --lag seconds (default 0.25 s) with gain 0.84 -> dead-time check;
  * controller dbg topics: e_ct / arc_dist / arc_R in arcs, k_h_eff on
    straights, e_theta_deg + delta_deg in both (e_ct also on straights with
    --straight-ect, amplitude 1.5 cm);
  * /round1_controller/lap_state [corner_idx, corner_count, lap];
  * /wall_matches (10 Hz, 2-3 matches from the true pose + noise),
    /localization_state ok -> recovering (2 s) -> ok -> lost (1 s) -> ok,
    /race_direction, /corner_geometry, /inner_geometry, /front_wall_x,
    /start_scan_state, /obstacles (5 pillars), /obstacles_live;
  * /scan (ray cast, 10 Hz) and /camera_lidar/colored_scan (7 Hz, pillar
    points whose colour-classification rate falls with range);
  * jtop topics (1 Hz), serial-link latency/rtt/offset/drift, battery, run
    timer, /ekf/gyro_ok;
  * /rosout with the controller's dead-time line, corner lines, one
    emergency manoeuvre, a parking report and an emergency-stop line, in German (--lang de, the current
    source strings) or English (--lang en, the expected translation).

The bag is written with rosbags' Writer and then converted to the Humble
layout (metadata.yaml version 5, sqlite schema 3, no embedded message
definitions -- so the reader MUST register robot_msgs from the .msg files).

Known values (checked by tests/test_toolkit.py) are in EXPECTED[lang].
"""
import argparse
import math
import sqlite3
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from bagio import make_typestore  # noqa: E402
from robot_constants import (GYRO_SCALE, INNER_HALF, LIDAR_OFFSET_X, OUTER_HALF,  # noqa: E402
                             R_EFF, START_POSES, field_to_map, seats_field, wrap)

EPOCH_NS = 1_789_900_000 * 10**9       # 2026-09-20 ~10:13 UTC
V_CRUISE = 0.6
R_TURN = 0.5
GAIN = 0.84
OBSTACLES = [(1, 1), (8, 2), (13, 1), (21, 2), (4, 2)]     # (seat id, colour)
LOC_EVENTS = [(0.5, 'ok'), (15.0, 'recovering'), (17.0, 'ok'), (25.0, 'lost'), (26.0, 'ok')]
BATTERY_V = (8.20, 7.80)

EXPECTED = {
    'de': {'parked': True, 'dist_cm': 4.3, 'expected_cm': 4.0, 'heading_deg': 0.8,
           'axle_cm': 0.1, 'within': True, 'estop': True, 'estop_kind': 'localisation_lost',
           'dead_time_cfg': 0.26},
    'en': {'parked': True, 'dist_cm': 6.1, 'expected_cm': 4.0, 'heading_deg': -12.5,
           'axle_cm': 2.3, 'within': False, 'estop': True, 'estop_kind': 'turn_in_point',
           'dead_time_cfg': 0.26},
}

LOG_TEXT = {
    'de': {
        'dead_time': 'Regler: Totzeit-Vorausberechnung 0.260 s (Verstaerkung 0.84), k_heading 1.00, '
                     'k_stanley 1.20, k_ct 2.0, k_th 1.5, Bogen verankern bei gestoerter Einfahrt: an.',
        'start': 'Start.',
        'corner': 'TURN fertig Ecke {n} (theta={th:.1f}, ziel={th:.1f}).',
        'loc': 'Lokalisierung: {old} -> {new}{extra}.',
        'three': 'Drei Runden fertig ({n} Ecken) -- Seiten frei, faehrt ohne Halt zur Einpark-Startpose.',
        'bay': 'Endlage laut Buchtmessung: Heck 1.2 cm, Front 2.5 cm Luft.',
        'parked': 'EINGEPARKT. base_link 4.3 cm von der Aussenbande (erwartet 4.0 cm), Kurs +0.8 grad '
                  'zur Bande = 0.1 cm Achsdifferenz (Regel: hoechstens 2 cm).',
        'estop': "NOTSTOP: Lokalisierung seit 2.1 s 'lost' -- ueber ~0,35 m Fehler faengt sie sich "
                 "nicht mehr, Weiterfahren waere Blindflug.",
        'harmless': 'Bridge bereit',
        'manoeuvre': 'NOTFALL-RANGIEREN 1/2: etwas 3 cm vor der Nase -- setzt 12 cm zurueck '
                     '(Lenkung +40 %, Kurs +8 grad zur Geraden), dann neu planen.',
    },
    'en': {
        'dead_time': 'Controller: dead-time prediction 0.260 s (gain 0.84), k_heading 1.00, '
                     'k_stanley 1.20, k_ct 2.0, k_th 1.5, anchor arc on disturbed entry: on.',
        'start': 'Start.',
        'corner': 'TURN done corner {n} (theta={th:.1f}, target={th:.1f}).',
        'loc': 'Localization: {old} -> {new}{extra}.',
        'three': 'Three laps done ({n} corners) -- sides clear, driving to the park-in start pose.',
        'bay': 'Final position from bay measurement: rear 1.0 cm, front 3.1 cm clearance.',
        'parked': 'PARKED. base_link 6.1 cm from the outer wall (expected 4.0 cm), heading -12.5 deg '
                  'to the wall = 2.3 cm axle difference (rule: at most 2 cm).',
        'estop': 'EMERGENCY STOP: turn-in point missed laterally (lat=0.65). Wrong corner? idx 2.',
        'harmless': 'emergency stop service ready',     # INFO level -> must NOT count
        'manoeuvre': 'EMERGENCY MANOEUVRE 1/2: something 3 cm in front of the nose -- backs up 12 cm '
                     '(steering +40 %, heading +8 deg to the straight), then re-plans.',
    },
}


# --------------------------------------------------------------------------
# Trajectory
# --------------------------------------------------------------------------
def path_segments(start_x=0.05):
    """CW lane-centre path in the field frame as (length, curvature) list for
    one lap, starting at (start_x, 1.0) heading +x."""
    quarter = math.pi / 2 * R_TURN
    segs = [(0.5 - start_x, 0.0)]
    for _ in range(3):
        segs += [(quarter, -1.0 / R_TURN), (1.0, 0.0)]
    segs += [(quarter, -1.0 / R_TURN), (start_x + 0.5, 0.0)]
    return segs


def wiggle(t):
    return (0.25 * np.sin(2 * np.pi * 0.7 * t) + 0.15 * np.sin(2 * np.pi * 1.3 * t + 1.0)
            + 0.10 * np.sin(2 * np.pi * 0.31 * t + 2.0))


def simulate(laps, dt=0.001, t_start=2.0, lag=0.25):
    """Integrate the true motion. Returns dict of arrays on a 1 kHz grid."""
    segs = path_segments() * laps
    bounds = np.cumsum([0.0] + [s[0] for s in segs])
    total = bounds[-1]
    # speed profile: 0.5 s ramp up, cruise, 1 s ramp down at the end
    t_drive = total / V_CRUISE + 1.0
    t_end_drive = t_start + t_drive
    T_total = t_end_drive + 6.0
    t = np.arange(0.0, T_total, dt)
    v = np.zeros_like(t)
    s = np.zeros_like(t)
    kappa = np.zeros_like(t)
    seg_idx = np.full(t.shape, -1)
    for i in range(1, t.size):
        tt = t[i]
        if t_start <= tt and s[i - 1] < total:
            ramp_up = min(1.0, (tt - t_start) / 0.5)
            rest = total - s[i - 1]
            v_brake = math.sqrt(max(2 * 0.4 * rest, 0.0))
            v[i] = max(0.05, min(V_CRUISE * ramp_up, v_brake))
        s[i] = min(total, s[i - 1] + v[i] * dt)
        if s[i] >= total:
            v[i] = 0.0
    k = np.clip(np.searchsorted(bounds, s, side='right') - 1, 0, len(segs) - 1)
    kappa = np.array([segs[j][1] for j in k])
    seg_idx = k
    moving = v > 0.01
    omega_geo = v * kappa + np.where(moving, wiggle(t) * np.minimum(v / V_CRUISE, 1.0), 0.0)
    # The controller publishes /cmd_vel at 30 Hz; the bridge holds each command
    # (zero-order hold) and the yaw rate follows it after `lag` seconds:
    #     omega_true(t) = GAIN * cmd_zoh(t - lag)
    # The command samples are chosen from the geometric path so that the robot
    # still drives the lane-centre line.
    t_cmd = np.arange(0.0, T_total, 1.0 / 30.0)
    w_cmd = np.interp(t_cmd + lag, t, omega_geo, right=0.0) / GAIN
    v_cmd = np.interp(t_cmd + lag, t, v, right=0.0)
    k_cmd = np.clip(np.searchsorted(t_cmd, t - lag, side='right') - 1, 0, t_cmd.size - 1)
    omega = np.where(t - lag >= 0.0, GAIN * w_cmd[k_cmd], 0.0)
    # integrate pose in the FIELD frame
    x0, y0, th0 = START_POSES['cw_pos1']
    th = th0 + np.cumsum(omega) * dt
    x = x0 + np.cumsum(v * np.cos(th)) * dt
    y = y0 + np.cumsum(v * np.sin(th)) * dt
    # to the map frame
    pm = field_to_map(np.column_stack([x, y]), START_POSES['cw_pos1'])
    yaw_m = wrap(th - th0)
    # corner completion times (end of each arc)
    corner_end = []
    acc = 0.0
    for L, kap in segs:
        acc += L
        if kap != 0.0:
            corner_end.append(t[np.searchsorted(s, acc - 1e-9)])
    return dict(t=t, v=v, s=s, kappa=kappa, seg=seg_idx, omega=omega, x=pm[:, 0], y=pm[:, 1],
                yaw=yaw_m, t_cmd=t_cmd, w_cmd=w_cmd, v_cmd=v_cmd, corner_end=corner_end,
                t_start=t_start, t_end_drive=t_end_drive, T_total=T_total, dt=dt)


def map_walls():
    """The 8 field walls in the map frame: list of (alpha, d, p1, p2),
    same formula as field_map._segments_to_map()."""
    xs, ys, th = START_POSES['cw_pos1']
    out = []
    for half, cw in ((OUTER_HALF, False), (INNER_HALF, True)):
        c = [np.array([-half, -half]), np.array([half, -half]), np.array([half, half]),
             np.array([-half, half])]
        if cw:
            c = c[::-1]
        for i in range(4):
            p1, p2 = c[i], c[(i + 1) % 4]
            d = p2 - p1
            n = np.array([-d[1], d[0]]) / np.hypot(*d)
            af = math.atan2(n[1], n[0])
            df = float(np.dot(p1, n))
            out.append((float(wrap(af - th)), df - (xs * math.cos(af) + ys * math.sin(af)),
                        field_to_map(p1, (xs, ys, th))[0], field_to_map(p2, (xs, ys, th))[0]))
    return out


def box_corners_map(half):
    c = field_to_map(np.array([[-half, -half], [half, -half], [half, half], [-half, half]]),
                     START_POSES['cw_pos1'])
    centre = c.mean(axis=0)
    order = sorted(range(4), key=lambda i: math.atan2(c[i, 1] - centre[1], c[i, 0] - centre[0]))
    c = c[order]
    k = int(np.argmax(c[:, 0]))
    return np.roll(c, -k, axis=0)


# --------------------------------------------------------------------------
# Message building
# --------------------------------------------------------------------------
class Builder:
    def __init__(self, ts):
        self.ts = ts
        self.msgs = []          # (t_ns, topic, typ, msg)
        self.ty = ts.types

    def T(self, name):
        return self.ty[name]

    def time(self, t):
        ns = EPOCH_NS + int(round(t * 1e9))
        return self.T('builtin_interfaces/msg/Time')(sec=ns // 10**9, nanosec=ns % 10**9)

    def header(self, t, frame='map'):
        return self.T('std_msgs/msg/Header')(stamp=self.time(t), frame_id=frame)

    def add(self, t, topic, typ, msg):
        self.msgs.append((EPOCH_NS + int(round(t * 1e9)), topic, typ, msg))

    def f64(self, t, topic, v):
        self.add(t, topic, 'std_msgs/msg/Float64', self.T('std_msgs/msg/Float64')(data=float(v)))

    def f32(self, t, topic, v):
        self.add(t, topic, 'std_msgs/msg/Float32', self.T('std_msgs/msg/Float32')(data=float(v)))

    def string(self, t, topic, s):
        self.add(t, topic, 'std_msgs/msg/String', self.T('std_msgs/msg/String')(data=s))

    def boolean(self, t, topic, b):
        self.add(t, topic, 'std_msgs/msg/Bool', self.T('std_msgs/msg/Bool')(data=bool(b)))

    def layout(self):
        return self.T('std_msgs/msg/MultiArrayLayout')(dim=[], data_offset=0)

    def i32arr(self, t, topic, data):
        self.add(t, topic, 'std_msgs/msg/Int32MultiArray', self.T('std_msgs/msg/Int32MultiArray')(
            layout=self.layout(), data=np.asarray(data, dtype=np.int32)))

    def f32arr(self, t, topic, data):
        self.add(t, topic, 'std_msgs/msg/Float32MultiArray', self.T('std_msgs/msg/Float32MultiArray')(
            layout=self.layout(), data=np.asarray(data, dtype=np.float32)))

    def vec3(self, x=0.0, y=0.0, z=0.0):
        return self.T('geometry_msgs/msg/Vector3')(x=float(x), y=float(y), z=float(z))

    def point(self, x=0.0, y=0.0, z=0.0):
        return self.T('geometry_msgs/msg/Point')(x=float(x), y=float(y), z=float(z))

    def quat_yaw(self, yaw):
        return self.T('geometry_msgs/msg/Quaternion')(x=0.0, y=0.0, z=math.sin(yaw / 2),
                                                      w=math.cos(yaw / 2))

    def log(self, t, level, name, text):
        self.add(t, '/rosout', 'rcl_interfaces/msg/Log', self.T('rcl_interfaces/msg/Log')(
            stamp=self.time(t), level=level, name=name, msg=text, file='synthetic.py',
            function='fake', line=1))


def build(args):
    rng = np.random.default_rng(args.seed)
    ts = make_typestore(args.msg_dir)
    b = Builder(ts)
    sim = simulate(args.laps, lag=args.lag)
    t, dt = sim['t'], sim['dt']
    L = LOG_TEXT[args.lang]

    def at(tq):
        return min(int(round(tq / dt)), t.size - 1)

    T_total = sim['T_total']
    moving_any = sim['v'] > 0.01

    # --- odometry 50 Hz, EKF error ~3 mm / 0.2 deg -------------------------------
    cov = np.zeros(36)
    cov[0] = cov[7] = 1e-5
    cov[35] = 1e-5
    for tq in np.arange(0.02, T_total, 0.02):
        i = at(tq)
        ex, ey = 0.003 * math.sin(0.9 * tq), 0.003 * math.cos(0.7 * tq)
        pose = b.T('geometry_msgs/msg/Pose')(position=b.point(sim['x'][i] + ex, sim['y'][i] + ey),
                                             orientation=b.quat_yaw(sim['yaw'][i] + 0.003 * math.sin(tq)))
        tw = b.T('geometry_msgs/msg/Twist')(linear=b.vec3(sim['v'][i]), angular=b.vec3(z=sim['omega'][i]))
        msg = b.T('nav_msgs/msg/Odometry')(
            header=b.header(tq, 'odom'), child_frame_id='base_link',
            pose=b.T('geometry_msgs/msg/PoseWithCovariance')(pose=pose, covariance=cov.copy()),
            twist=b.T('geometry_msgs/msg/TwistWithCovariance')(twist=tw, covariance=cov.copy()))
        b.add(tq + 0.001, '/ekf/odom', 'nav_msgs/msg/Odometry', msg)

    # --- IMU 100 Hz (raw BNO055 sign/scale), joint states + speed 100 Hz --------
    shaft = 0.0
    zero_q = b.quat_yaw(0.0)
    for tq in np.arange(0.01, T_total, 0.01):
        i = at(tq)
        raw_z = sim['omega'][i] / GYRO_SCALE + rng.normal(0, 0.01)
        imu = b.T('sensor_msgs/msg/Imu')(
            header=b.header(tq, 'imu'), orientation=zero_q, orientation_covariance=np.zeros(9),
            angular_velocity=b.vec3(rng.normal(0, 0.002), rng.normal(0, 0.002), raw_z),
            angular_velocity_covariance=np.zeros(9),
            linear_acceleration=b.vec3(0.0, 0.0, 9.81), linear_acceleration_covariance=np.zeros(9))
        b.add(tq + 0.002, '/bno055/imu', 'sensor_msgs/msg/Imu', imu)
        w_shaft = sim['v'][i] / R_EFF
        shaft += w_shaft * 0.01
        js = b.T('sensor_msgs/msg/JointState')(header=b.header(tq - 0.004, 'esp'), name=['drive_axle'],
                                               position=np.array([shaft]), velocity=np.array([w_shaft]),
                                               effort=np.array([], dtype=float))
        b.add(tq + 0.003, '/esp_serial_bridge/joint_states', 'sensor_msgs/msg/JointState', js)
        b.f32(tq + 0.003, '/esp_serial_bridge/speed', math.degrees(w_shaft))
        lat = 1.6 + rng.gamma(2.0, 0.35)
        b.f32(tq + 0.0031, '/esp_serial_bridge/latency_ms', lat)

    # --- cmd_vel + controller dbg 30 Hz -------------------------------------------
    ctrl = '/round1_controller'
    for k_c, tq in enumerate(sim['t_cmd']):
        i = at(tq)
        driving = sim['t_start'] - 0.5 <= tq <= sim['t_end_drive'] + 0.3
        v_c = sim['v_cmd'][k_c] if driving else 0.0
        w_c = sim['w_cmd'][k_c] if driving else 0.0
        tw = b.T('geometry_msgs/msg/Twist')(linear=b.vec3(v_c), angular=b.vec3(z=w_c))
        b.add(tq, '/cmd_vel', 'geometry_msgs/msg/Twist', tw)
        if not (driving and sim['v'][i] > 0.01):
            continue
        in_arc = sim['kappa'][i] != 0.0
        e_th = 3.0 * math.sin(2 * math.pi * 0.4 * tq) + rng.normal(0, 0.3)
        delta = math.degrees(math.atan(0.1 * w_c / max(sim['v'][i], 0.05)))
        if in_arc:
            e_ct = 0.02 * math.sin(2 * math.pi * 0.8 * tq) + rng.normal(0, 0.002)
            b.f64(tq + 0.0005, f'{ctrl}/dbg/e_ct', e_ct)
            b.f64(tq + 0.0007, f'{ctrl}/dbg/arc_dist', R_TURN + e_ct)
            b.f64(tq + 0.0009, f'{ctrl}/dbg/arc_R', R_TURN)
            b.f64(tq + 0.0011, f'{ctrl}/dbg/e_theta_deg', e_th)
            b.f64(tq + 0.0013, f'{ctrl}/dbg/delta_deg', delta)
        else:
            if args.straight_ect:
                b.f64(tq + 0.0003, f'{ctrl}/dbg/e_ct', 0.015 * math.sin(2 * math.pi * 0.5 * tq))
            b.f64(tq + 0.0005, f'{ctrl}/dbg/e_theta_deg', e_th)
            b.f64(tq + 0.0007, f'{ctrl}/dbg/delta_deg', delta)
            b.f64(tq + 0.0009, f'{ctrl}/dbg/k_h_eff', 1.0)

    # --- lap_state + corner log lines --------------------------------------------
    corner_idx = 0
    b.i32arr(sim['t_start'], f'{ctrl}/lap_state', [corner_idx, 0, 0])
    for n, tc in enumerate(sim['corner_end'], start=1):
        corner_idx = (corner_idx - 1) % 4                 # CW: index steps -1
        b.i32arr(tc, f'{ctrl}/lap_state', [corner_idx, n, n // 4])
        b.log(tc, 20, 'round1_controller', L['corner'].format(n=n, th=-90.0 * n))
    n_corners = len(sim['corner_end'])

    # --- rosout: config, start, localisation, laps, parking, estop ---------------
    b.log(0.3, 20, 'round1_controller', L['dead_time'])
    b.log(0.35, 20, 'esp_serial_bridge', L['harmless'])
    b.log(sim['t_start'], 20, 'round1_controller', L['start'])
    prev = '-'
    for tl, st in LOC_EVENTS:
        extra = '' if st == 'ok' else ' (max 0.35 m/s, no parking)'
        b.log(tl + 0.01, 30 if st != 'ok' else 20, 'round1_controller',
              L['loc'].format(old=prev, new=st, extra=extra))
        prev = st
    b.log(sim['corner_end'][5] - 1.0, 30, 'round1_controller', L['manoeuvre'])
    te = sim['t_end_drive']
    b.log(te - 0.5, 20, 'round1_controller', L['three'].format(n=n_corners))
    b.log(te + 2.0, 20, 'round1_controller', L['bay'])
    b.log(te + 2.05, 20, 'round1_controller', L['parked'])
    b.log(te + 3.0, 40, 'round1_controller', L['estop'])

    # --- latched perception topics ---------------------------------------------
    b.string(0.5, '/start_scan_state', 'scanning')
    b.string(1.2, '/start_scan_state', 'complete')
    b.string(1.0, '/race_direction', 'CW')
    b.f64(1.0, '/front_wall_x', OUTER_HALF - START_POSES['cw_pos1'][0])
    for topic, half in (('/corner_geometry', OUTER_HALF), ('/inner_geometry', INNER_HALF)):
        c = box_corners_map(half)
        walls = []
        for k in range(4):
            p1, p2 = c[k], c[(k + 1) % 4]
            e = p2 - p1
            nrm = np.array([-e[1], e[0]]) / np.hypot(*e)
            if np.dot(nrm, c.mean(axis=0) - p1) < 0:
                nrm = -nrm
            walls.append(b.T('robot_msgs/msg/WallHNF')(nx=float(nrm[0]), ny=float(nrm[1]),
                                                        d=float(np.dot(nrm, p1))))
        msg = b.T('robot_msgs/msg/CornerGeometry')(
            header=b.header(1.0), corners=[b.point(*p) for p in c], walls=walls,
            edge_length=2 * half)
        b.add(1.0, topic, 'robot_msgs/msg/CornerGeometry', msg)
    for tl, st in LOC_EVENTS:
        b.string(tl, '/localization_state', st)
    b.boolean(2.0, '/ekf/gyro_ok', True)

    seats = {s['id']: s for s in seats_field()}
    obst_map = []
    for sid, col in OBSTACLES:
        p = field_to_map(seats[sid]['p'], START_POSES['cw_pos1'])[0]
        obst_map.append((sid, col, p, seats[sid]['straight']))

    def obstacle_array(tq, items, frame):
        obs = [b.T('robot_msgs/msg/Obstacle')(id=int(sid), position=b.point(p[0], p[1]),
                                              color=int(col), wall_idx=int(w))
               for sid, col, p, w in items]
        return b.T('robot_msgs/msg/ObstacleArray')(header=b.header(tq, frame), obstacles=obs)

    b.add(1.5, '/obstacles', 'robot_msgs/msg/ObstacleArray', obstacle_array(1.5, obst_map[:2], 'map'))
    b.add(6.0, '/obstacles', 'robot_msgs/msg/ObstacleArray', obstacle_array(6.0, obst_map, 'map'))

    # --- wall matches, obstacles_live, scan, colored cloud ------------------------
    walls = map_walls()
    beams = np.linspace(-math.pi, math.pi, 360, endpoint=False)
    for tq in np.arange(0.05, T_total, 0.1):
        i = at(tq)
        x, y, th = sim['x'][i], sim['y'][i], sim['yaw'][i]
        # wall matches (none while 'recovering' / 'lost')
        state = [s for tl, s in LOC_EVENTS if tl <= tq]
        state = state[-1] if state else 'ok'
        cand = []
        for a_map, d_map, p1, p2 in walls:
            d_pred = d_map - (x * math.cos(a_map) + y * math.sin(a_map))
            seg = p2 - p1
            u = np.dot(np.array([x, y]) - p1, seg) / np.dot(seg, seg)
            # EKF convention (ekf.py / field_map.start_map_3wall): a wall the
            # robot sees has a NEGATIVE d in the robot frame
            if -0.2 <= u <= 1.2 and d_pred < -0.05:
                cand.append((abs(d_pred), d_pred, a_map, d_map))
        cand.sort()
        k = 0 if state != 'ok' else min(len(cand), 2 + (int(tq * 10) % 2))
        matches = []
        for _, d_pred, a_map, d_map in cand[:k]:
            matches.append(b.T('robot_msgs/msg/WallMatch')(
                header=b.header(tq, 'base_link'),
                alpha_meas=float(wrap(a_map - th) + rng.normal(0, math.radians(0.3))),
                d_meas=float(d_pred + rng.normal(0, 0.003)), alpha_map=a_map, d_map=d_map))
        b.add(tq + 0.03, '/wall_matches', 'robot_msgs/msg/WallMatchArray',
              b.T('robot_msgs/msg/WallMatchArray')(header=b.header(tq, 'base_link'), matches=matches))
        # obstacles in base_link
        c, s_ = math.cos(th), math.sin(th)
        live = []
        for sid, col, p, w in obst_map:
            dx, dy = p[0] - x, p[1] - y
            xb, yb = c * dx + s_ * dy, -s_ * dx + c * dy
            if math.hypot(xb, yb) < 1.5:
                live.append((-1, col, (xb, yb), -1))
        b.add(tq + 0.04, '/obstacles_live', 'robot_msgs/msg/ObstacleArray',
              obstacle_array(tq, live, 'base_link'))
        # laser scan: ray cast from the LiDAR (base x = LIDAR_OFFSET_X), mounted 180 deg
        lx, ly = x + c * LIDAR_OFFSET_X, y + s_ * LIDAR_OFFSET_X
        dirs = th + math.pi + beams
        dxs, dys = np.cos(dirs), np.sin(dirs)
        best = np.full(beams.size, np.inf)
        for _, _, p1, p2 in walls:
            ex, ey = p2[0] - p1[0], p2[1] - p1[1]
            den = dxs * ey - dys * ex
            with np.errstate(divide='ignore', invalid='ignore'):
                tr = ((p1[0] - lx) * ey - (p1[1] - ly) * ex) / den
                us = ((p1[0] - lx) * dys - (p1[1] - ly) * dxs) / den
            ok = (np.abs(den) > 1e-9) & (tr > 0) & (us >= 0) & (us <= 1)
            best = np.where(ok & (tr < best), tr, best)
        ranges = np.where(np.isfinite(best), best + rng.normal(0, 0.005, best.size), np.inf)
        scan = b.T('sensor_msgs/msg/LaserScan')(
            header=b.header(tq, 'laser'), angle_min=float(beams[0]), angle_max=float(beams[-1]),
            angle_increment=float(beams[1] - beams[0]), time_increment=0.0, scan_time=0.1,
            range_min=0.15, range_max=12.0, ranges=ranges.astype(np.float32),
            intensities=np.zeros(0, dtype=np.float32))
        b.add(tq + 0.01, '/scan', 'sensor_msgs/msg/LaserScan', scan)

    # colored cloud 7 Hz: pillar points (label depends on range) + wall points
    pf = b.T('sensor_msgs/msg/PointField')
    fields = [pf(name=n, offset=o, datatype=7, count=1) for n, o in
              (('x', 0), ('y', 4), ('z', 8), ('rgb', 12))]
    codes = {1: 0xFF0000, 2: 0x00FF00}
    for tq in np.arange(0.1, T_total, 1.0 / 7.0):
        i = at(tq)
        x, y, th = sim['x'][i], sim['y'][i], sim['yaw'][i]
        c, s_ = math.cos(th), math.sin(th)
        pts, rgbs = [], []
        for sid, col, p, w in obst_map:
            dx, dy = p[0] - x, p[1] - y
            xb, yb = c * dx + s_ * dy, -s_ * dx + c * dy
            xl, yl = -(xb - LIDAR_OFFSET_X), -yb
            r = math.hypot(xl, yl)
            if r > 3.0:
                continue
            p_ok = float(np.clip(1.05 - 0.3 * r, 0.15, 0.97))
            for _ in range(8):
                ang = math.atan2(yl, xl) + rng.normal(0, 0.012)
                rr = r - 0.022 + rng.normal(0, 0.004)
                pts.append((rr * math.cos(ang), rr * math.sin(ang)))
                u = rng.random()
                if u < p_ok:
                    rgbs.append(codes[col])
                elif u < p_ok + 0.03:
                    rgbs.append(codes[3 - col])
                else:
                    rgbs.append(0x555555)
        for a in rng.uniform(-math.pi, math.pi, 40):
            rr = rng.uniform(0.3, 2.5)
            pts.append((rr * math.cos(a), rr * math.sin(a)))
            rgbs.append(0x2D2D2D if rng.random() < 0.7 else 0x555555)
        arr = np.zeros(len(pts), dtype=[('x', '<f4'), ('y', '<f4'), ('z', '<f4'), ('rgb', '<u4')])
        arr['x'] = [q[0] for q in pts]
        arr['y'] = [q[1] for q in pts]
        arr['rgb'] = rgbs
        cloud = b.T('sensor_msgs/msg/PointCloud2')(
            header=b.header(tq, 'laser'), height=1, width=len(pts), fields=fields,
            is_bigendian=False, point_step=16, row_step=16 * len(pts),
            data=np.frombuffer(arr.tobytes(), dtype=np.uint8), is_dense=True)
        b.add(tq + 0.05, '/camera_lidar/colored_scan', 'sensor_msgs/msg/PointCloud2', cloud)

    # --- 1 Hz topics: jtop, serial link, battery ----------------------------------
    for k, tq in enumerate(np.arange(0.5, T_total, 1.0)):
        b.f32(tq, '/jtop/cpu_total', 45 + 12 * math.sin(tq / 7) + rng.normal(0, 2))
        b.f32arr(tq, '/jtop/cpu_load', 45 + rng.normal(0, 8, 6))
        b.f32(tq, '/jtop/gpu_load', 20 + 10 * math.sin(tq / 5) ** 2)
        b.f32(tq, '/jtop/ram_percent', 55 + 3 * tq / T_total)
        b.f32(tq, '/jtop/power_total', 9.5 + 1.5 * math.sin(tq / 9) + rng.normal(0, 0.2))
        for z, base in (('cpu', 46), ('gpu', 44), ('soc', 45), ('tj', 48)):
            temp = b.T('sensor_msgs/msg/Temperature')(header=b.header(tq, z),
                                                      temperature=base + 6 * tq / T_total, variance=0.0)
            b.add(tq, f'/jtop/temp/{z}', 'sensor_msgs/msg/Temperature', temp)
        b.f32(tq + 0.2, '/esp_serial_bridge/rtt_ms', 2.2 + rng.gamma(2.0, 0.15))
        b.f64(tq + 0.2, '/esp_serial_bridge/offset_ms', 1_234_567.0 + 0.1 * tq + rng.normal(0, 0.01))
        b.f32(tq + 0.2, '/esp_serial_bridge/drift_ppm', 100.0 + rng.normal(0, 1.0))
    for tq in np.arange(1.0, T_total, 5.0):
        frac = tq / T_total
        volt = BATTERY_V[0] + (BATTERY_V[1] - BATTERY_V[0]) * frac
        bat = b.T('sensor_msgs/msg/BatteryState')(
            header=b.header(tq, 'esp'), voltage=volt, temperature=math.nan, current=math.nan,
            charge=math.nan, capacity=math.nan, design_capacity=math.nan,
            percentage=(volt / 2 - 3.3) / 0.9, power_supply_status=2, power_supply_health=1,
            power_supply_technology=0, present=True,
            cell_voltage=np.full(4, volt / 2, dtype=np.float32),
            cell_temperature=np.zeros(0, dtype=np.float32), location='', serial_number='')
        b.add(tq, '/esp_serial_bridge/battery', 'sensor_msgs/msg/BatteryState', bat)
    b.f32arr(0.4, '/esp_serial_bridge/pid', [4.0, 140.0, 8.0])

    # --- run timer 10 Hz -----------------------------------------------------------
    t_move = t[moving_any]
    t0m, t1m = (t_move[0], t_move[-1]) if t_move.size else (0, 0)
    for tq in np.arange(0.0, T_total, 0.1):
        if tq < t0m:
            st, el = 'idle', 0.0
        elif tq < t1m + 1.0:
            st, el = 'running', min(tq, t1m) - t0m
        else:
            st, el = 'stopped', t1m - t0m
        b.f32(tq, '/viz/run_time', el)
        b.string(tq, '/viz/run_state', st)
    return b, sim


def humbleize(bagdir):
    """Convert a rosbags-written bag (metadata v8, sqlite schema 4) to the
    layout ros2 bag record writes on Humble (metadata v5, sqlite schema 3,
    no message definitions / type hashes)."""
    from ruamel.yaml import YAML
    db = next(Path(bagdir).glob('*.db3'))
    con = sqlite3.connect(db)
    con.executescript("""
        DROP TABLE IF EXISTS message_definitions;
        CREATE TABLE topics_new(id INTEGER PRIMARY KEY, name TEXT NOT NULL, type TEXT NOT NULL,
            serialization_format TEXT NOT NULL, offered_qos_profiles TEXT NOT NULL);
        INSERT INTO topics_new SELECT id, name, type, serialization_format, offered_qos_profiles FROM topics;
        DROP TABLE topics;
        ALTER TABLE topics_new RENAME TO topics;
        UPDATE schema SET schema_version = 3, ros_distro = 'humble';
    """)
    con.commit()
    con.execute('VACUUM')
    con.close()
    yaml = YAML()
    meta_p = Path(bagdir) / 'metadata.yaml'
    meta = yaml.load(meta_p.read_text())
    info = meta['rosbag2_bagfile_information']
    info['version'] = 5
    for k in ('ros_distro', 'custom_data'):
        info.pop(k, None)
    for tm in info['topics_with_message_count']:
        tm['topic_metadata'].pop('type_description_hash', None)
    with meta_p.open('w') as f:
        yaml.dump(meta, f)


def write_bag(out_dir, name='parken_test_7', lang='de', straight_ect=False, lag=0.25, laps=3,
              seed=1, msg_dir=None, humble=True):
    from rosbags.rosbag2 import Writer
    args = argparse.Namespace(lang=lang, straight_ect=straight_ect, lag=lag, laps=laps, seed=seed,
                              msg_dir=msg_dir)
    b, sim = build(args)
    bagdir = Path(out_dir) / name
    if bagdir.exists():
        raise FileExistsError(bagdir)
    b.msgs.sort(key=lambda m: m[0])
    with Writer(bagdir, version=8) as w:
        conns = {}
        for _, topic, typ, _ in b.msgs:
            if topic not in conns:
                conns[topic] = w.add_connection(topic, typ, typestore=b.ts)
        for t_ns, topic, typ, msg in b.msgs:
            w.write(conns[topic], t_ns, b.ts.serialize_cdr(msg, typ))
    if humble:
        humbleize(bagdir)
    return bagdir, sim


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('out_dir')
    ap.add_argument('--name', default='parken_test_7')
    ap.add_argument('--lang', choices=('de', 'en'), default='de')
    ap.add_argument('--straight-ect', action='store_true')
    ap.add_argument('--lag', type=float, default=0.25)
    ap.add_argument('--laps', type=int, default=3)
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--msg-dir', default=None)
    a = ap.parse_args(argv)
    p, sim = write_bag(a.out_dir, a.name, a.lang, a.straight_ect, a.lag, a.laps, a.seed, a.msg_dir)
    print(f'wrote {p} ({sim["T_total"]:.1f} s)')


if __name__ == '__main__':
    main()
