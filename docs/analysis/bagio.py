"""Shared bag I/O for the analysis toolkit (no ROS installation needed).

Reads ROS 2 bags (sqlite3 .db3 + metadata.yaml, as written by `ros2 bag
record` on Humble) with the pure-Python `rosbags` package, registers the
custom robot_msgs types from the .msg files in the repo, and flattens messages
into pandas tables.

Main entry points
-----------------
find_bags(paths)            expand files / bag dirs / dirs-of-bags to bag paths
Bag(path)                   open one bag (dir, metadata.yaml or .db3 path)
read_tables(bag, topics)    one pass over the bag -> {topic: DataFrame}
load_run(path, topics)      same, but also accepts a CSV export dir written by
                            bag_export.py; returns a Run object

Table conventions (identical in memory and in the exported CSVs)
----------------------------------------------------------------
t_bag     receive time of the message in the recorder, seconds since the
          first message of the bag
t_header  header.stamp (or Log.stamp) in seconds since the same bag start;
          NaN when the message has no header or the stamp is 0
other columns: see the flatten_* functions below.

A bag without metadata.yaml (e.g. the recorder was killed) is read directly
from the .db3 file(s) with sqlite3.
"""
import dataclasses
import fnmatch
import json
import math
import re
import sqlite3
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from logpatterns import LEVEL_NAMES
from robot_constants import CLOUD_LABEL_RGB, COLOR_NAMES

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
DEFAULT_MSG_DIR = REPO / 'src' / 'robot_msgs' / 'msg'
DEFAULT_DATA_DIR = REPO / 'docs' / 'data'
DEFAULT_FIG_DIR = REPO / 'docs' / 'figures'

# --------------------------------------------------------------------------
# Topic names (from the node sources, see README "Topic assumptions")
# --------------------------------------------------------------------------
T = {
    'odom': '/ekf/odom',                              # ekf_node.py
    'gyro_ok': '/ekf/gyro_ok',
    'imu': '/bno055/imu',
    'cmd_vel': '/cmd_vel',                            # round1_controller_node.py
    'lap_state': '/round1_controller/lap_state',
    'e_ct': '/round1_controller/dbg/e_ct',
    'e_theta': '/round1_controller/dbg/e_theta_deg',
    'delta': '/round1_controller/dbg/delta_deg',
    'k_h': '/round1_controller/dbg/k_h_eff',
    'arc_dist': '/round1_controller/dbg/arc_dist',
    'arc_R': '/round1_controller/dbg/arc_R',
    'wall_matches': '/wall_matches',                  # scan_processor_node.py
    'wall_distances': '/wall_distances',
    'loc_state': '/localization_state',
    'race_direction': '/race_direction',
    'obstacles': '/obstacles',
    'obstacles_live': '/obstacles_live',
    'front_wall_x': '/front_wall_x',
    'start_scan_state': '/start_scan_state',
    'corner_geometry': '/corner_geometry',
    'inner_geometry': '/inner_geometry',
    'latency': '/esp_serial_bridge/latency_ms',       # esp_serial_bridge.py
    'rtt': '/esp_serial_bridge/rtt_ms',
    'offset': '/esp_serial_bridge/offset_ms',
    'drift': '/esp_serial_bridge/drift_ppm',
    'joints': '/esp_serial_bridge/joint_states',
    'battery': '/esp_serial_bridge/battery',
    'speed': '/esp_serial_bridge/speed',
    'pid': '/esp_serial_bridge/pid',
    'cpu': '/jtop/cpu_total',                         # foxglove_overlay_node.py
    'gpu': '/jtop/gpu_load',
    'ram': '/jtop/ram_percent',
    'power': '/jtop/power_total',
    'temps': '/jtop/temp/*',
    'run_time': '/viz/run_time',
    'run_state': '/viz/run_state',
    'scan': '/scan',
    'cloud': '/camera_lidar/colored_scan',            # lidar_pixel_mapper.py
    'rosout': '/rosout',
}

# Message types that are only exported with --heavy.
HEAVY_TYPES = {
    'sensor_msgs/msg/LaserScan', 'sensor_msgs/msg/PointCloud2',
    'sensor_msgs/msg/Image', 'sensor_msgs/msg/CompressedImage',
}

_warned = set()


def warn_once(text):
    if text not in _warned:
        _warned.add(text)
        print(f'WARNING: {text}', file=sys.stderr)


# --------------------------------------------------------------------------
# Typestore
# --------------------------------------------------------------------------
def make_typestore(msg_dir=None):
    """ROS 2 Humble typestore plus every robot_msgs/*.msg found in msg_dir."""
    from rosbags.typesys import Stores, get_types_from_msg, get_typestore
    store = get_typestore(Stores.ROS2_HUMBLE)
    msg_dir = Path(msg_dir) if msg_dir else DEFAULT_MSG_DIR
    types = {}
    if msg_dir.is_dir():
        for p in sorted(msg_dir.glob('*.msg')):
            types.update(get_types_from_msg(p.read_text(), f'robot_msgs/msg/{p.stem}'))
    else:
        warn_once(f'msg dir {msg_dir} not found -- robot_msgs topics will be skipped '
                  f'(use --msg-dir)')
    if types:
        store.register(types)
    return store


# --------------------------------------------------------------------------
# Finding and opening bags
# --------------------------------------------------------------------------
def natural_key(s):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', str(s))]


def is_bag_dir(p):
    p = Path(p)
    return p.is_dir() and ((p / 'metadata.yaml').exists() or any(p.glob('*.db3')))


def is_export_dir(p):
    return (Path(p) / 'export_info.json').exists()


def find_bags(paths):
    """Expand CLI paths: a .db3 file, a bag dir, a metadata.yaml, or a dir that
    contains bag dirs (searched recursively). Sorted naturally
    (parken_test_2 before parken_test_10)."""
    found = []
    for raw in paths:
        p = Path(raw)
        if p.is_file() and p.suffix == '.db3':
            found.append(p)
        elif p.is_file() and p.name == 'metadata.yaml':
            found.append(p.parent)
        elif is_bag_dir(p) or is_export_dir(p):
            found.append(p)
        elif p.is_dir():
            subs = sorted({q.parent for q in p.rglob('metadata.yaml')}
                          | {q.parent for q in p.rglob('*.db3')}
                          | {q.parent for q in p.rglob('export_info.json')},
                          key=natural_key)
            found.extend(subs)
        else:
            warn_once(f'{p}: not a bag, skipped')
    # de-duplicate, keep order
    out, seen = [], set()
    for f in found:
        k = str(Path(f).resolve())
        if k not in seen:
            seen.add(k)
            out.append(Path(f))
    return out


def bag_name(path):
    p = Path(path)
    if p.suffix == '.db3':
        return re.sub(r'_\d+$', '', p.stem)
    return p.name


class _SqliteFallbackReader:
    """Minimal reader for .db3 files without metadata.yaml."""

    def __init__(self, files):
        self.files = [Path(f) for f in files]
        self.topics = {}          # name -> (type, count)
        for f in self.files:
            con = sqlite3.connect(f'file:{f}?mode=ro', uri=True)
            try:
                rows = con.execute(
                    'SELECT topics.name, topics.type, count(messages.id) FROM topics '
                    'LEFT JOIN messages ON topics.id = messages.topic_id '
                    'GROUP BY topics.id').fetchall()
                for name, typ, n in rows:
                    t0, n0 = self.topics.get(name, (typ, 0))
                    self.topics[name] = (typ, n0 + n)
                lo, hi = con.execute('SELECT min(timestamp), max(timestamp) FROM messages').fetchone()
            finally:
                con.close()
            if lo is not None:
                self._lo = min(getattr(self, '_lo', lo), lo)
                self._hi = max(getattr(self, '_hi', hi), hi)
        self.start_ns = getattr(self, '_lo', 0)
        self.end_ns = getattr(self, '_hi', 0)

    def messages(self, topics):
        for f in self.files:
            con = sqlite3.connect(f'file:{f}?mode=ro', uri=True)
            try:
                idmap = {i: (n, t) for i, n, t in con.execute('SELECT id, name, type FROM topics')}
                ids = [i for i, (n, _) in idmap.items() if n in topics]
                if not ids:
                    continue
                q = ('SELECT topic_id, timestamp, data FROM messages WHERE topic_id IN (%s) '
                     'ORDER BY timestamp' % ','.join('?' * len(ids)))
                for tid, ts, data in con.execute(q, ids):
                    name, typ = idmap[tid]
                    yield name, typ, ts, data
            finally:
                con.close()


class Bag:
    """One ROS 2 bag. Use as a context manager or call close()."""

    def __init__(self, path, msg_dir=None, typestore=None):
        self.path = Path(path)
        self.name = bag_name(self.path)
        self.typestore = typestore or make_typestore(msg_dir)
        self._reader = None
        self._fallback = None
        if self.path.is_file() and self.path.name == 'metadata.yaml':
            self.path = self.path.parent
        if self.path.is_file():                          # a .db3 file
            meta = self.path.parent / 'metadata.yaml'
            if meta.exists():
                self._open_reader(self.path.parent)
            else:
                self._fallback = _SqliteFallbackReader([self.path])
        elif (self.path / 'metadata.yaml').exists():
            self._open_reader(self.path)
        else:
            files = sorted(self.path.glob('*.db3'), key=natural_key)
            if not files:
                raise FileNotFoundError(f'{self.path}: no metadata.yaml and no .db3 file')
            warn_once(f'{self.path}: no metadata.yaml, reading the .db3 directly')
            self._fallback = _SqliteFallbackReader(files)

    def _open_reader(self, bagdir):
        from rosbags.rosbag2 import Reader
        self._reader = Reader(bagdir)
        self._reader.open()

    # -- metadata ----------------------------------------------------------
    @property
    def topics(self):
        """{topic: msgtype}"""
        if self._reader:
            return {c.topic: c.msgtype for c in self._reader.connections}
        return {n: t for n, (t, _) in self._fallback.topics.items()}

    @property
    def counts(self):
        if self._reader:
            out = {}
            for c in self._reader.connections:
                out[c.topic] = out.get(c.topic, 0) + c.msgcount
            return out
        return {n: c for n, (_, c) in self._fallback.topics.items()}

    @property
    def start_ns(self):
        return self._reader.start_time if self._reader else self._fallback.start_ns

    @property
    def end_ns(self):
        return self._reader.end_time if self._reader else self._fallback.end_ns

    @property
    def duration_s(self):
        return max(0.0, (self.end_ns - self.start_ns) * 1e-9)

    @property
    def start_utc(self):
        return datetime.fromtimestamp(self.start_ns * 1e-9, tz=timezone.utc)

    def select(self, patterns=None, heavy=True):
        """Topics matching glob patterns (None = all). heavy=False drops
        LaserScan / PointCloud2 / images."""
        out = []
        for topic, typ in self.topics.items():
            if patterns and not any(fnmatch.fnmatchcase(topic, p) for p in patterns):
                continue
            if not heavy and typ in HEAVY_TYPES:
                continue
            out.append(topic)
        return sorted(out)

    # -- messages ------------------------------------------------------------
    def messages(self, topics):
        """Yield (topic, msgtype, t_ns, msg) for the given topic names.
        Unknown or undecodable types are skipped with one warning each."""
        topics = set(topics)
        known = set(self.typestore.types)

        def decode(topic, typ, t, raw):
            if typ not in known:
                warn_once(f'{self.name}: type {typ} of {topic} is not registered -- skipped')
                return None
            try:
                return self.typestore.deserialize_cdr(raw, typ)
            except Exception as exc:                          # noqa: BLE001
                warn_once(f'{self.name}: cannot decode {topic} ({typ}): {exc} -- skipped')
                return None

        if self._reader:
            conns = [c for c in self._reader.connections if c.topic in topics]
            if not conns:
                return
            for conn, t, raw in self._reader.messages(connections=conns):
                msg = decode(conn.topic, conn.msgtype, t, raw)
                if msg is not None:
                    yield conn.topic, conn.msgtype, t, msg
        else:
            for topic, typ, t, raw in self._fallback.messages(topics):
                msg = decode(topic, typ, t, raw)
                if msg is not None:
                    yield topic, typ, t, msg

    def close(self):
        if self._reader:
            self._reader.close()
            self._reader = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


# --------------------------------------------------------------------------
# Flattening messages to rows
# --------------------------------------------------------------------------
def stamp_s(stamp):
    """builtin_interfaces/Time -> float seconds (NaN for a zero stamp)."""
    if stamp is None:
        return math.nan
    v = stamp.sec + stamp.nanosec * 1e-9
    return v if v > 0 else math.nan


def yaw_of(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def _is_field(name):
    return not (name.startswith('__') or name.isupper() or name[:1].isupper())


def _fields(msg):
    return [f.name for f in dataclasses.fields(msg) if _is_field(f.name)]


def flatten_generic(msg, prefix='', out=None, depth=0, max_elems=16):
    """Any message -> flat dict of scalars (dotted names joined by '_').
    Short arrays (<= max_elems) are expanded, longer ones become <name>_len."""
    out = {} if out is None else out
    for name in _fields(msg):
        v = getattr(msg, name)
        key = prefix + name
        if name == 'header' and depth == 0:
            out['frame_id'] = v.frame_id
            continue
        if isinstance(v, (bool, int, float, str, np.integer, np.floating)):
            out[key] = v.item() if hasattr(v, 'item') else v
        elif isinstance(v, np.ndarray):
            if v.size <= max_elems:
                for i, e in enumerate(v.ravel()):
                    out[f'{key}_{i}'] = e.item()
            else:
                out[f'{key}_len'] = int(v.size)
        elif isinstance(v, (list, tuple)):
            if len(v) <= max_elems and depth < 3:
                for i, e in enumerate(v):
                    if dataclasses.is_dataclass(e):
                        flatten_generic(e, f'{key}_{i}_', out, depth + 1, max_elems)
                    else:
                        out[f'{key}_{i}'] = e
            else:
                out[f'{key}_len'] = len(v)
        elif dataclasses.is_dataclass(v):
            f = _fields(v)
            if f == ['sec', 'nanosec']:
                out[key] = stamp_s(v)
            elif depth < 4:
                flatten_generic(v, key + '_', out, depth + 1, max_elems)
    return out


def flatten_odometry(msg):
    p, q = msg.pose.pose.position, msg.pose.pose.orientation
    tw = msg.twist.twist
    pc, tc = np.asarray(msg.pose.covariance), np.asarray(msg.twist.covariance)
    return [{'x': p.x, 'y': p.y, 'yaw': yaw_of(q), 'v': tw.linear.x, 'omega': tw.angular.z,
             'cov_x': pc[0], 'cov_y': pc[7], 'cov_yaw': pc[35],
             'cov_v': tc[0], 'cov_omega': tc[35], 'frame_id': msg.header.frame_id}]


def flatten_twist(msg):
    return [{'linear_x': msg.linear.x, 'linear_y': msg.linear.y, 'linear_z': msg.linear.z,
             'angular_x': msg.angular.x, 'angular_y': msg.angular.y, 'angular_z': msg.angular.z}]


def flatten_data(msg):
    v = msg.data
    return [{'data': v.item() if hasattr(v, 'item') else v}]


def flatten_multiarray(msg):
    d = np.asarray(msg.data).ravel()
    row = {f'data_{i}': e.item() for i, e in enumerate(d[:64])}
    row['n'] = int(d.size)
    return [row]


def flatten_imu(msg):
    w, a, q = msg.angular_velocity, msg.linear_acceleration, msg.orientation
    return [{'gyro_x': w.x, 'gyro_y': w.y, 'gyro_z': w.z,
             'acc_x': a.x, 'acc_y': a.y, 'acc_z': a.z, 'yaw': yaw_of(q),
             'frame_id': msg.header.frame_id}]


def flatten_jointstate(msg):
    row = {'n_joints': len(msg.name), 'name': ';'.join(msg.name)}
    for key in ('position', 'velocity', 'effort'):
        arr = np.asarray(getattr(msg, key)).ravel()
        for i, e in enumerate(arr):
            row[key if i == 0 else f'{key}_{i}'] = float(e)
        if arr.size == 0:
            row[key] = math.nan          # e.g. MOVE_DONE frames carry no velocity
    return [row]


def flatten_battery(msg):
    cells = np.asarray(msg.cell_voltage, dtype=float)
    return [{'voltage': msg.voltage, 'current': msg.current, 'percentage': msg.percentage,
             'cell_voltage_mean': float(np.nanmean(cells)) if cells.size else math.nan,
             'present': bool(msg.present), 'power_supply_status': msg.power_supply_status,
             'power_supply_health': msg.power_supply_health}]


def flatten_temperature(msg):
    return [{'temperature': msg.temperature, 'variance': msg.variance}]


def flatten_log(msg):
    return [{'level': int(msg.level), 'level_name': LEVEL_NAMES.get(int(msg.level), str(msg.level)),
             'name': msg.name, 'msg': msg.msg, 'file': msg.file, 'function': msg.function,
             'line': int(msg.line)}]


def flatten_wallmatches(msg):
    """One row per match; a scan with no match gives one row with
    match_idx = -1 so the per-scan count is not lost."""
    n = len(msg.matches)
    if n == 0:
        return [{'n_matches': 0, 'match_idx': -1}]
    return [{'n_matches': n, 'match_idx': i, 'alpha_meas': m.alpha_meas, 'd_meas': m.d_meas,
             'alpha_map': m.alpha_map, 'd_map': m.d_map} for i, m in enumerate(msg.matches)]


def flatten_obstacles(msg):
    """One row per obstacle; an empty set gives one row with obst_idx = -1."""
    n = len(msg.obstacles)
    if n == 0:
        return [{'n_obstacles': 0, 'obst_idx': -1, 'frame_id': msg.header.frame_id}]
    return [{'n_obstacles': n, 'obst_idx': i, 'id': o.id, 'x': o.position.x, 'y': o.position.y,
             'color': int(o.color), 'color_name': COLOR_NAMES.get(int(o.color), str(o.color)),
             'wall_idx': o.wall_idx, 'frame_id': msg.header.frame_id}
            for i, o in enumerate(msg.obstacles)]


def flatten_corner_geometry(msg):
    row = {'edge_length': msg.edge_length, 'frame_id': msg.header.frame_id}
    for i, c in enumerate(msg.corners):
        row[f'corner{i}_x'], row[f'corner{i}_y'] = c.x, c.y
    for i, w in enumerate(msg.walls):
        row[f'wall{i}_nx'], row[f'wall{i}_ny'], row[f'wall{i}_d'] = w.nx, w.ny, w.d
    return [row]


def flatten_laserscan(msg):
    r = np.asarray(msg.ranges, dtype=float)
    return [{'angle_min': msg.angle_min, 'angle_max': msg.angle_max,
             'angle_increment': msg.angle_increment, 'range_min': msg.range_min,
             'range_max': msg.range_max, 'n': int(r.size),
             'ranges': ' '.join(f'{v:.4f}' for v in r)}]


_PF_DTYPES = {1: 'i1', 2: 'u1', 3: '<i2', 4: '<u2', 5: '<i4', 6: '<u4', 7: '<f4', 8: '<f8'}


def pointcloud_to_array(msg):
    """sensor_msgs/PointCloud2 -> numpy structured array (one record per point),
    using the offsets and datatypes in msg.fields."""
    names, formats, offsets = [], [], []
    for f in msg.fields:
        dt = _PF_DTYPES.get(int(f.datatype))
        if dt is None:
            continue
        if msg.is_bigendian:
            dt = dt.replace('<', '>')
        names.append(f.name)
        formats.append(dt if f.count <= 1 else (dt, int(f.count)))
        offsets.append(int(f.offset))
    dtype = np.dtype({'names': names, 'formats': formats, 'offsets': offsets,
                      'itemsize': int(msg.point_step)})
    raw = np.asarray(msg.data, dtype=np.uint8).tobytes()
    return np.frombuffer(raw, dtype=dtype, count=int(msg.width) * int(msg.height))


def cloud_label(rgb_u32):
    """Packed 0xRRGGBB (uint32 array) -> label strings (see CLOUD_LABEL_RGB)."""
    rgb = np.asarray(rgb_u32, dtype=np.uint32) & 0xFFFFFF
    out = np.full(rgb.shape, 'other', dtype=object)
    for code, name in CLOUD_LABEL_RGB.items():
        out[rgb == code] = name
    return out


def flatten_pointcloud(msg):
    """Point cloud -> DataFrame, one row per point: x, y, z, rgb (packed
    uint32), label (decoded class, see cloud_label())."""
    arr = pointcloud_to_array(msg)
    df = pd.DataFrame({k: arr[k].astype(float) for k in ('x', 'y', 'z') if k in arr.dtype.names})
    if 'rgb' in arr.dtype.names:
        rgb = arr['rgb'].copy().view(np.uint32) if arr['rgb'].dtype.itemsize == 4 else arr['rgb'].astype(np.uint32)
        df['rgb'] = rgb & 0xFFFFFF
        df['label'] = cloud_label(rgb)
    return df


FLATTENERS = {
    'nav_msgs/msg/Odometry': flatten_odometry,
    'geometry_msgs/msg/Twist': flatten_twist,
    'std_msgs/msg/Float32': flatten_data,
    'std_msgs/msg/Float64': flatten_data,
    'std_msgs/msg/Int32': flatten_data,
    'std_msgs/msg/Bool': flatten_data,
    'std_msgs/msg/String': flatten_data,
    'std_msgs/msg/Int32MultiArray': flatten_multiarray,
    'std_msgs/msg/Float32MultiArray': flatten_multiarray,
    'std_msgs/msg/Float64MultiArray': flatten_multiarray,
    'sensor_msgs/msg/Imu': flatten_imu,
    'sensor_msgs/msg/JointState': flatten_jointstate,
    'sensor_msgs/msg/BatteryState': flatten_battery,
    'sensor_msgs/msg/Temperature': flatten_temperature,
    'sensor_msgs/msg/LaserScan': flatten_laserscan,
    'sensor_msgs/msg/PointCloud2': flatten_pointcloud,
    'rcl_interfaces/msg/Log': flatten_log,
    'robot_msgs/msg/WallMatchArray': flatten_wallmatches,
    'robot_msgs/msg/ObstacleArray': flatten_obstacles,
    'robot_msgs/msg/CornerGeometry': flatten_corner_geometry,
}


def header_time(msg):
    h = getattr(msg, 'header', None)
    if h is not None and hasattr(h, 'stamp'):
        return stamp_s(h.stamp)
    st = getattr(msg, 'stamp', None)            # rcl_interfaces/Log
    if st is not None and hasattr(st, 'sec'):
        return stamp_s(st)
    return math.nan


def flatten(msgtype, msg):
    """-> list of dict rows, or a DataFrame (point clouds)."""
    fn = FLATTENERS.get(msgtype)
    if fn is None:
        return [flatten_generic(msg)]
    return fn(msg)


# --------------------------------------------------------------------------
# Reading whole tables
# --------------------------------------------------------------------------
def read_tables(bag, topics=None, heavy=False, max_cloud_msgs=None):
    """One pass over the bag. topics: list of names or glob patterns (None =
    all lightweight topics; heavy types only when heavy=True or when a heavy
    topic is named explicitly). Returns {topic: DataFrame} with t_bag and
    t_header in seconds since bag start, plus msg_index (0-based per topic)."""
    if topics is None:
        selected = bag.select(None, heavy=heavy)
    else:
        selected = bag.select(topics, heavy=True)
        if not heavy:
            explicit = {t for t in topics if not any(c in t for c in '*?[')}
            selected = [t for t in selected
                        if bag.topics[t] not in HEAVY_TYPES or t in explicit]
    t0 = bag.start_ns
    rows = {t: [] for t in selected}
    frames = {t: [] for t in selected}
    index = {t: 0 for t in selected}
    for topic, typ, t_ns, msg in bag.messages(selected):
        i = index[topic]
        index[topic] += 1
        if typ == 'sensor_msgs/msg/PointCloud2' and max_cloud_msgs and i >= max_cloud_msgs:
            continue
        t_bag = (t_ns - t0) * 1e-9
        th = header_time(msg)
        t_hdr = th - t0 * 1e-9 if not math.isnan(th) else math.nan
        try:
            flat = flatten(typ, msg)
        except Exception as exc:                          # noqa: BLE001
            warn_once(f'{bag.name}: cannot flatten {topic} ({typ}): {exc}')
            continue
        if isinstance(flat, pd.DataFrame):
            flat.insert(0, 'msg_index', i)
            flat.insert(0, 't_header', t_hdr)
            flat.insert(0, 't_bag', t_bag)
            frames[topic].append(flat)
        else:
            for r in flat:
                rows[topic].append({'t_bag': t_bag, 't_header': t_hdr, 'msg_index': i, **r})
    out = {}
    for t in selected:
        parts = list(frames[t])
        if rows[t]:
            parts.insert(0, pd.DataFrame(rows[t]))
        if parts:
            out[t] = pd.concat(parts, ignore_index=True) if len(parts) > 1 else parts[0]
        else:
            out[t] = pd.DataFrame(columns=['t_bag', 't_header', 'msg_index'])
    return out


def topic_to_filename(topic):
    return topic.strip('/').replace('/', '__') + '.csv'


class Run:
    """Tables of one bag (or of one CSV export dir)."""

    def __init__(self, name, tables, types, start_ns, duration_s, source):
        self.name = name
        self.tables = tables
        self.types = types
        self.start_ns = start_ns
        self.duration_s = duration_s
        self.source = str(source)

    @property
    def start_utc(self):
        if not self.start_ns:
            return None
        return datetime.fromtimestamp(self.start_ns * 1e-9, tz=timezone.utc)

    def has(self, topic):
        df = self.tables.get(topic)
        return df is not None and len(df) > 0

    def get(self, topic):
        """Table of a topic (empty DataFrame when absent)."""
        df = self.tables.get(topic)
        if df is None:
            return pd.DataFrame(columns=['t_bag', 't_header', 'msg_index'])
        return df

    def matching(self, pattern):
        return {t: df for t, df in self.tables.items() if fnmatch.fnmatchcase(t, pattern)}


def load_run(path, topics=None, msg_dir=None, heavy=False, typestore=None, max_cloud_msgs=None):
    """Load a bag or a bag_export.py output dir into a Run."""
    path = Path(path)
    if is_export_dir(path):
        info = json.loads((path / 'export_info.json').read_text())
        tables, types = {}, {}
        for topic, meta in info['topics'].items():
            if topics and not any(fnmatch.fnmatchcase(topic, p) for p in topics):
                continue
            f = path / meta['file']
            if f.exists():
                tables[topic] = pd.read_csv(f, low_memory=False)
                types[topic] = meta.get('type', '')
        return Run(info.get('bag', path.name), tables, types, info.get('start_ns', 0),
                   info.get('duration_s', math.nan), path)
    with Bag(path, msg_dir=msg_dir, typestore=typestore) as bag:
        tables = read_tables(bag, topics, heavy=heavy, max_cloud_msgs=max_cloud_msgs)
        return Run(bag.name, tables, bag.topics, bag.start_ns, bag.duration_s, path)


# --------------------------------------------------------------------------
# Small helpers used by several tools
# --------------------------------------------------------------------------
def last_value(df, col='data', default=None):
    if df is None or len(df) == 0 or col not in df:
        return default
    return df[col].iloc[-1]


def nearest_dt(t_query, t_ref):
    """|t_query - nearest t_ref| for each query time (both sorted arrays)."""
    tq, tr = np.asarray(t_query, float), np.asarray(t_ref, float)
    if tr.size == 0:
        return np.full(tq.shape, np.inf)
    i = np.clip(np.searchsorted(tr, tq), 1, max(tr.size - 1, 1))
    lo = tr[np.clip(i - 1, 0, tr.size - 1)]
    hi = tr[np.clip(i, 0, tr.size - 1)]
    return np.minimum(np.abs(tq - lo), np.abs(tq - hi))


def add_common_args(ap, out_default=None):
    """--out-dir and --msg-dir, shared by all CLI tools."""
    ap.add_argument('--out-dir', default=str(out_default or DEFAULT_FIG_DIR),
                    help='output directory (default: %(default)s)')
    ap.add_argument('--msg-dir', default=None,
                    help=f'robot_msgs/msg directory (default: {DEFAULT_MSG_DIR})')
    return ap


def iter_runs(paths, topics=None, msg_dir=None, heavy=False, max_cloud_msgs=None):
    """Yield a Run for every bag / export dir found under the CLI paths.
    A bag that cannot be read is reported and skipped."""
    store = make_typestore(msg_dir)
    found = find_bags(paths)
    if not found:
        print('no bags found in: ' + ' '.join(map(str, paths)), file=sys.stderr)
    for p in found:
        try:
            yield load_run(p, topics, typestore=store, heavy=heavy, max_cloud_msgs=max_cloud_msgs)
        except Exception as exc:                          # noqa: BLE001
            print(f'{p}: cannot read ({exc}) -- skipped', file=sys.stderr)
