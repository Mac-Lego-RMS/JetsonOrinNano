"""Only one instance per node.

Twice a scan_processor ran double -- one from start_robot.sh, one started by
hand, plus two ekf_node. Both sent wall hits to the EKF, each against
its own map. In parken_test_14 the second one latched a map rotated by 8 deg
in the middle of the first corner; localisation gone, emergency stop.
You could only see it from duplicated log lines and /wall_matches at
twice the scan rate.

So check before the start whether someone already serves the output topic,
and then do not get going at all.
"""
import time

import rclpy


def ensure_single_instance(topic, node_name, wait_s=2.0):
    """Ends the process if there already is a publisher on `topic`.

    Call after rclpy.init() and BEFORE your own create_publisher. DDS
    discovery needs a moment, so the search goes on for up to `wait_s`
    seconds.
    """
    probe = rclpy.create_node(f'{node_name}_startup_check')
    try:
        end = time.monotonic() + wait_s
        others = []
        while time.monotonic() < end and not others:
            others = probe.get_publishers_info_by_topic(topic)
            if not others:
                time.sleep(0.1)
        if others:
            names = ', '.join(sorted({i.node_namespace.rstrip('/') + '/' + i.node_name
                                      for i in others}))
            probe.get_logger().fatal(
                f'{topic} already has a publisher ({names}) -- is {node_name} '
                f'already running? Two instances disturb each other. Not started.')
            raise SystemExit(1)
    finally:
        probe.destroy_node()
