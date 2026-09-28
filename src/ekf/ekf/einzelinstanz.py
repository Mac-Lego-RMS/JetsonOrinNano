"""Nur eine Instanz je Knoten.

Zweimal lief ein scan_processor doppelt -- einer aus start_robot.sh, einer von
Hand, dazu zwei ekf_node. Beide schickten Wandtreffer an den EKF, jeder gegen
seine eigene Karte. In parken_test_14 latchte der zweite mitten in der ersten
Kurve eine um 8 grad verdrehte Karte nach; Lokalisierung weg, Notstopp.
Sehen konnte man das nur an doppelten Logzeilen und /wall_matches mit der
doppelten Scanrate.

Deshalb vor dem Start nachsehen, ob das Ausgangstopic schon jemand bedient,
und dann gar nicht erst loslegen.
"""
import time

import rclpy


def nur_eine_instanz(topic, knoten, warten=2.0):
    """Beendet den Prozess, wenn auf `topic` schon ein Publisher sitzt.

    Nach rclpy.init() und VOR dem eigenen create_publisher aufrufen. Die
    DDS-Discovery braucht einen Moment, deshalb wird bis `warten` Sekunden
    gesucht.
    """
    pruefer = rclpy.create_node(f'{knoten}_startpruefung')
    try:
        ende = time.monotonic() + warten
        andere = []
        while time.monotonic() < ende and not andere:
            andere = pruefer.get_publishers_info_by_topic(topic)
            if not andere:
                time.sleep(0.1)
        if andere:
            namen = ', '.join(sorted({i.node_namespace.rstrip('/') + '/' + i.node_name
                                      for i in andere}))
            pruefer.get_logger().fatal(
                f'{topic} hat schon einen Publisher ({namen}) -- laeuft {knoten} '
                f'bereits? Zwei Instanzen stoeren sich gegenseitig. Nicht gestartet.')
            raise SystemExit(1)
    finally:
        pruefer.destroy_node()
