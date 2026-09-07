#!/usr/bin/env python3
"""Publishes the Foxglove overlay live: the TF tree the stack never had, plus
the marker topics that make the 3D panel show the field.

The robot itself needs none of this -- it is purely for visualisation, so the
same layout (wro_overlay_layout.json) works live and on a recorded bag.

Subscribes: /ekf/odom, /scan, /wall_matches,
            /corner_geometry (latched), /inner_geometry (latched)
Publishes:  /tf, /tf_static
            /viz/field         outer box, inner band, corners, wall normals
            /viz/wall_matches  measured wall vs. matched map wall
            /viz/path          driven trajectory
            /viz/robot         body + heading
            /viz/scan_used     only the beams wall_extraction actually uses
"""
import math

import rclpy
from rclpy.node import Node
from rclpy.qos import (QoSProfile, DurabilityPolicy, ReliabilityPolicy,
                       HistoryPolicy, qos_profile_sensor_data)

from geometry_msgs.msg import Point, TransformStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import Marker, MarkerArray
from tf2_ros import TransformBroadcaster, StaticTransformBroadcaster

from robot_msgs.msg import CornerGeometry, WallMatchArray

# Must match ekf.wall_extraction -- the LiDAR is mounted turned around, so
# scan_to_points maps a beam as x = -r*cos(a), y = -r*sin(a): a pure 180 deg yaw.
LIDAR_OFFSET_X = 0.1101
LIDAR_YAW = math.pi
LIDAR_Z = 0.10
BLOCK_ANGLE = math.radians(60.0)
MAX_RANGE = 4.0

PATH_STEP = 0.02        # m of travel between stored path points
PATH_MAX = 6000

C_OUTER = (0.85, 0.87, 0.92, 1.0)
C_INNER = (1.00, 0.45, 0.15, 1.0)
C_CORNER = (0.35, 0.75, 1.00, 1.0)
C_MEAS = (0.20, 0.95, 0.45, 1.0)
C_MAP = (0.30, 0.60, 1.00, 1.0)
C_PATH = (1.00, 0.85, 0.20, 1.0)


def yaw_from_quaternion(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                      1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def pt(x, y, z=0.0):
    p = Point()
    p.x, p.y, p.z = float(x), float(y), float(z)
    return p


def rgba(c):
    return ColorRGBA(r=c[0], g=c[1], b=c[2], a=c[3])


def hnf_segment(nx, ny, d, half):
    """Endpoints of { p : n.p = d }, centred on the point closest to the origin."""
    px, py = nx * d, ny * d
    dx, dy = -ny, nx
    return pt(px - dx * half, py - dy * half), pt(px + dx * half, py + dy * half)


class FoxgloveOverlay(Node):
    def __init__(self):
        super().__init__('foxglove_overlay')

        self.field_hz = self.declare_parameter('field_rate', 2.0).value
        self.robot_hz = self.declare_parameter('robot_rate', 10.0).value
        self.publish_scan_used = self.declare_parameter('publish_scan_used', True).value

        self.outer = None
        self.inner = None
        self.path = []
        self.last_pt = None
        # Stamp of the newest incoming data. Markers are stamped with it, not
        # with the wall clock, so the markers and /tf share one timeline even
        # when a bag is replayed without use_sim_time.
        self.last_stamp = None

        latched = QoSProfile(depth=1, history=HistoryPolicy.KEEP_LAST,
                             reliability=ReliabilityPolicy.RELIABLE,
                             durability=DurabilityPolicy.TRANSIENT_LOCAL)

        self.tf = TransformBroadcaster(self)
        self.tf_static = StaticTransformBroadcaster(self)
        self._publish_static()

        self.create_subscription(Odometry, '/ekf/odom', self.odom_cb, 20)
        self.create_subscription(WallMatchArray, '/wall_matches', self.wm_cb, 10)
        self.create_subscription(CornerGeometry, '/corner_geometry',
                                 self.outer_cb, latched)
        self.create_subscription(CornerGeometry, '/inner_geometry',
                                 self.inner_cb, latched)
        if self.publish_scan_used:
            self.create_subscription(LaserScan, '/scan', self.scan_cb,
                                     qos_profile_sensor_data)
            self.pub_scan = self.create_publisher(LaserScan, '/viz/scan_used', 5)

        self.pub_field = self.create_publisher(MarkerArray, '/viz/field', 1)
        self.pub_wm = self.create_publisher(MarkerArray, '/viz/wall_matches', 5)
        self.pub_path = self.create_publisher(MarkerArray, '/viz/path', 1)
        self.pub_robot = self.create_publisher(MarkerArray, '/viz/robot', 1)

        self.create_timer(1.0 / self.field_hz, self.field_timer)
        self.create_timer(1.0 / self.robot_hz, self.robot_timer)

        self.get_logger().info('foxglove overlay running (tf + /viz/*)')

    # ------------------------------------------------------------------ #

    def _stamp(self):
        if self.last_stamp is not None:
            return self.last_stamp
        return self.get_clock().now().to_msg()

    def _marker(self, ns, mid, typ, frame, **kw):
        m = Marker()
        m.header.stamp = kw['stamp'] if 'stamp' in kw else self._stamp()
        m.header.frame_id = frame
        m.ns, m.id, m.type, m.action = ns, mid, typ, Marker.ADD
        m.pose.orientation.w = 1.0
        sc = kw.get('scale', (0.02, 0.02, 0.02))
        m.scale.x, m.scale.y, m.scale.z = float(sc[0]), float(sc[1]), float(sc[2])
        m.color = rgba(kw.get('color', C_OUTER))
        m.points = kw.get('points', [])
        m.text = kw.get('text', '')
        p = kw.get('at')
        if p is not None:
            m.pose.position.x, m.pose.position.y, m.pose.position.z = p
        return m

    def _publish_static(self):
        out = []
        for child, x, y, z, yaw in (
                ('laser', LIDAR_OFFSET_X, 0.0, LIDAR_Z, LIDAR_YAW),
                ('bno055', 0.0, 0.0, 0.06, 0.0),
                ('esp', 0.0, 0.0, 0.0, 0.0)):
            t = TransformStamped()
            t.header.stamp = self._stamp()
            t.header.frame_id = 'base_link'
            t.child_frame_id = child
            t.transform.translation.x = x
            t.transform.translation.y = y
            t.transform.translation.z = z
            t.transform.rotation.z = math.sin(yaw / 2.0)
            t.transform.rotation.w = math.cos(yaw / 2.0)
            out.append(t)
        self.tf_static.sendTransform(out)

    # ------------------------------------------------------------------ #

    def odom_cb(self, msg):
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        th = yaw_from_quaternion(msg.pose.pose.orientation)
        self.last_stamp = msg.header.stamp

        # The EKF pose is already start-anchored, so map == odom.
        m2o = TransformStamped()
        m2o.header.stamp = msg.header.stamp
        m2o.header.frame_id = 'map'
        m2o.child_frame_id = 'odom'
        m2o.transform.rotation.w = 1.0

        o2b = TransformStamped()
        o2b.header.stamp = msg.header.stamp
        o2b.header.frame_id = 'odom'
        o2b.child_frame_id = 'base_link'
        o2b.transform.translation.x = x
        o2b.transform.translation.y = y
        o2b.transform.rotation.z = math.sin(th / 2.0)
        o2b.transform.rotation.w = math.cos(th / 2.0)
        self.tf.sendTransform([m2o, o2b])

        if self.last_pt is None or math.hypot(x - self.last_pt[0],
                                              y - self.last_pt[1]) >= PATH_STEP:
            self.path.append(pt(x, y, 0.01))
            self.last_pt = (x, y)
            if len(self.path) > PATH_MAX:
                self.path.pop(0)

    def outer_cb(self, msg):
        self.outer = msg
        self.get_logger().info('corner_geometry empfangen')

    def inner_cb(self, msg):
        self.inner = msg
        self.get_logger().info('inner_geometry empfangen -> Innenband im Overlay')

    def scan_cb(self, msg):
        out = LaserScan()
        out.header = msg.header
        out.angle_min = msg.angle_min
        out.angle_max = msg.angle_max
        out.angle_increment = msg.angle_increment
        out.time_increment = msg.time_increment
        out.scan_time = msg.scan_time
        out.range_min = msg.range_min
        out.range_max = msg.range_max
        nan = float('nan')
        keep = []
        for i, r in enumerate(msg.ranges):
            a = msg.angle_min + i * msg.angle_increment
            ok = (r == r and msg.range_min <= r <= msg.range_max
                  and r <= MAX_RANGE and abs(a) > BLOCK_ANGLE)
            keep.append(r if ok else nan)
        out.ranges = keep
        out.intensities = list(msg.intensities)
        self.pub_scan.publish(out)

    def wm_cb(self, msg):
        arr = MarkerArray()
        st = msg.header.stamp
        for i, mt in enumerate(msg.matches):
            am, dm = mt.alpha_meas, mt.d_meas
            ap, dp = mt.alpha_map, mt.d_map
            # alpha_meas/d_meas are already in base_link (lidar_to_base_link),
            # despite header.frame_id saying "laser".
            a, b = hnf_segment(math.cos(am), math.sin(am), dm, 1.1)
            arr.markers.append(self._marker(
                'measured', i, Marker.LINE_STRIP, 'base_link', stamp=st,
                scale=(0.018, 0, 0), color=C_MEAS, points=[a, b]))
            arr.markers.append(self._marker(
                'measured_normal', 20 + i, Marker.ARROW, 'base_link', stamp=st,
                scale=(0.012, 0.026, 0), color=C_MEAS,
                points=[pt(0, 0), pt(math.cos(am) * dm, math.sin(am) * dm)]))
            a2, b2 = hnf_segment(math.cos(ap), math.sin(ap), dp, 1.5)
            arr.markers.append(self._marker(
                'matched_map', 40 + i, Marker.LINE_STRIP, 'map', stamp=st,
                scale=(0.010, 0, 0), color=C_MAP, points=[a2, b2]))
            arr.markers.append(self._marker(
                'match_info', 60 + i, Marker.TEXT_VIEW_FACING, 'base_link', stamp=st,
                scale=(0, 0, 0.07), color=C_MEAS,
                at=(math.cos(am) * dm, math.sin(am) * dm, 0.16),
                text='a=%+.1f d=%+.3f' % (math.degrees(am), dm)))
        self.pub_wm.publish(arr)

    # ------------------------------------------------------------------ #

    def field_timer(self):
        if self.outer is None:
            return
        arr = MarkerArray()
        oc = [(c.x, c.y) for c in self.outer.corners]
        arr.markers.append(self._marker(
            'outer', 0, Marker.LINE_STRIP, 'map', scale=(0.025, 0, 0),
            color=C_OUTER, points=[pt(*c) for c in oc] + [pt(*oc[0])]))
        arr.markers.append(self._marker(
            'outer_corners', 1, Marker.SPHERE_LIST, 'map',
            scale=(0.06, 0.06, 0.06), color=C_CORNER,
            points=[pt(*c) for c in oc]))
        for i, c in enumerate(oc):
            arr.markers.append(self._marker(
                'outer_labels', 10 + i, Marker.TEXT_VIEW_FACING, 'map',
                scale=(0, 0, 0.11), color=C_CORNER, at=(c[0], c[1], 0.14),
                text='C%d (%.2f, %.2f)' % (i, c[0], c[1])))
        for i, w in enumerate(self.outer.walls):
            arr.markers.append(self._marker(
                'outer_normals', 20 + i, Marker.ARROW, 'map',
                scale=(0.012, 0.028, 0), color=C_OUTER,
                points=[pt(w.nx * w.d, w.ny * w.d),
                        pt(w.nx * w.d + w.nx * 0.22, w.ny * w.d + w.ny * 0.22)]))

        if self.inner is not None:
            ic = [(c.x, c.y) for c in self.inner.corners]
            arr.markers.append(self._marker(
                'inner', 40, Marker.LINE_STRIP, 'map', scale=(0.03, 0, 0),
                color=C_INNER, points=[pt(*c) for c in ic] + [pt(*ic[0])]))
            arr.markers.append(self._marker(
                'inner_corners', 41, Marker.SPHERE_LIST, 'map',
                scale=(0.055, 0.055, 0.055), color=C_INNER,
                points=[pt(*c) for c in ic]))
            for i, c in enumerate(ic):
                arr.markers.append(self._marker(
                    'inner_labels', 50 + i, Marker.TEXT_VIEW_FACING, 'map',
                    scale=(0, 0, 0.09), color=C_INNER, at=(c[0], c[1], 0.10),
                    text='I%d (%.3f, %.3f)' % (i, c[0], c[1])))
            for i, w in enumerate(self.inner.walls):
                p1, p2 = ic[i], ic[(i + 1) % 4]
                mx, my = (p1[0] + p2[0]) / 2.0, (p1[1] + p2[1]) / 2.0
                arr.markers.append(self._marker(
                    'inner_normals', 60 + i, Marker.ARROW, 'map',
                    scale=(0.014, 0.032, 0), color=C_INNER,
                    points=[pt(mx, my), pt(mx + w.nx * 0.25, my + w.ny * 0.25)]))
                arr.markers.append(self._marker(
                    'inner_wall_info', 70 + i, Marker.TEXT_VIEW_FACING, 'map',
                    scale=(0, 0, 0.075), color=(1.0, 1.0, 1.0, 1.0),
                    at=(mx + w.nx * 0.34, my + w.ny * 0.34, 0.06),
                    text='W%d  n=(%+.3f,%+.3f)  d=%+.3f' % (i, w.nx, w.ny, w.d)))
        self.pub_field.publish(arr)

        if len(self.path) >= 2:
            p = MarkerArray()
            p.markers.append(self._marker(
                'path', 0, Marker.LINE_STRIP, 'map', scale=(0.014, 0, 0),
                color=C_PATH, points=list(self.path)))
            self.pub_path.publish(p)

    def robot_timer(self):
        arr = MarkerArray()
        body = self._marker('robot', 0, Marker.CUBE, 'base_link',
                            scale=(0.26, 0.16, 0.08),
                            color=(0.25, 0.85, 1.0, 0.75), at=(0.09, 0.0, 0.05))
        arr.markers.append(body)
        arr.markers.append(self._marker(
            'robot', 1, Marker.ARROW, 'base_link', scale=(0.018, 0.045, 0),
            color=(1.0, 0.25, 0.25, 1.0),
            points=[pt(0, 0, 0.05), pt(0.32, 0, 0.05)]))
        self.pub_robot.publish(arr)


def main(args=None):
    rclpy.init(args=args)
    node = FoxgloveOverlay()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
