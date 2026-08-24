#!/usr/bin/env python3
"""Testbench fuer die ESP-Serial-Bridge.

Klappert die komplette Bridge durch: schickt nacheinander jeden Befehl raus
und loggt alle Antworten/Events, die vom ESP zurueckkommen. Kein Lidar, keine
Kamera -- nur die UART-Bridge testen.

Start (in zwei Terminals):
    ros2 run wall_follower_robot esp_serial_bridge
    ros2 run robot_vision testBench          # oder: python3 testBench.py
"""

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Twist
from sensor_msgs.msg import BatteryState
from std_msgs.msg import (
    Bool, Empty, Float32, Float32MultiArray, Int8, Int32MultiArray, String,
)


class EspTestBench(Node):
    def __init__(self):
        super().__init__('esp_test_bench')

        # --- Publisher: ROS -> Bridge -> ESP ---
        self.pub_cmd_vel = self.create_publisher(Twist, '/cmd_vel', 10)
        self.pub_led = self.create_publisher(Bool, '/led_cmd', 10)
        self.pub_calib = self.create_publisher(Empty, '/calibrate_cmd', 10)
        self.pub_torque = self.create_publisher(Empty, '/torque_cmd', 10)
        self.pub_trim = self.create_publisher(Int8, '/trim_cmd', 10)
        self.pub_pid_set = self.create_publisher(Float32MultiArray, '/pid_set', 10)
        self.pub_pid_get = self.create_publisher(Empty, '/pid_get', 10)
        self.pub_pid_save = self.create_publisher(Empty, '/pid_save', 10)
        self.pub_move = self.create_publisher(Float32, '/move_cmd', 10)
        self.pub_move_abort = self.create_publisher(Empty, '/move_abort', 10)
        self.pub_progress = self.create_publisher(Empty, '/progress_request', 10)
        self.pub_battery = self.create_publisher(Empty, '/battery_request', 10)
        self.pub_emergency = self.create_publisher(Empty, '/emergency_stop', 10)

        # --- Subscriber: ESP -> Bridge -> ROS (nur mitloggen) ---
        self.create_subscription(String, '/esp/esp_data', self._on_text, 10)
        self.create_subscription(Bool, '/button_state', self._on_button, 10)
        self.create_subscription(Float32MultiArray, '/esp/pid_response', self._on_pid, 10)
        self.create_subscription(Bool, '/esp/pid_saved', self._on_pid_saved, 10)
        self.create_subscription(Int32MultiArray, '/esp/move_done', self._on_move_done, 10)
        self.create_subscription(Int32MultiArray, '/esp/progress', self._on_progress, 10)
        self.create_subscription(BatteryState, '/esp/battery', self._on_battery, 10)
        self.create_subscription(Bool, '/esp/battery_warn', self._on_battery_warn, 10)

        # --- Testsequenz: (Beschreibung, Aktion) ---
        self.steps = [
            #('LED an',                        lambda: self.pub_led.publish(Bool(data=True))),
            #('LED aus',                       lambda: self.pub_led.publish(Bool(data=False))),
            ('Motor vorwaerts + Servo links', lambda: self._drive(0.0, 0.5)),
            #('Motor stopp (coast)',           lambda: self._drive(0.0, 0.0)),
            #('CALIBRATE',                     lambda: self.pub_calib.publish(Empty())),
            #('TORQUE',                        lambda: self.pub_torque.publish(Empty())),
            #('TRIM links',                    lambda: self.pub_trim.publish(Int8(data=0))),
            #('TRIM rechts',                   lambda: self.pub_trim.publish(Int8(data=1))),
            #('TRIM speichern',                lambda: self.pub_trim.publish(Int8(data=2))),
            #('PID_SET Kp=4.5',                lambda: self._pid(0, 4.5)),
            #('PID_SET Kd=0.3',                lambda: self._pid(2, 0.3)),
            #('PID_SET maxDuty=700',           lambda: self._pid(4, 700.0)),
            #('PID_SAVE',                      lambda: self.pub_pid_save.publish(Empty())),
            ('PID_GET',                       lambda: self.pub_pid_get.publish(Empty())),
            #('MOVE 360 Grad',                  lambda: self.pub_move.publish(Float32(data=3600.0))),
            #('MOVE -360 Grad',                  lambda: self.pub_move.publish(Float32(data=-360.0))),
            ('PROGRESS abfragen',             lambda: self.pub_progress.publish(Empty())),
            #('MOVE_ABORT',                    lambda: self.pub_move_abort.publish(Empty())),
            ('BATTERY abfragen',              lambda: self.pub_battery.publish(Empty())),
            #('EMERGENCY',                     lambda: self.pub_emergency.publish(Empty())),
        ]
        # Welcher Schritt erwartet welche Antwort von der Bridge?
        # (Fire-and-forget-Befehle stehen hier nicht drin.)
        self.expected_response = {
            'PID_GET': 'PID_RSP',
            'PID_SAVE': 'PID_SAVED',
            'MOVE 90 Grad': 'MOVE_DONE',
            'PROGRESS abfragen': 'PROGRESS_RSP',
            'BATTERY abfragen': 'BATTERY_RSP',
        }
        self._pending = None        # erwartete Antwort des laufenden Schritts
        self._pending_desc = None
        self._response_seen = False

        self.idx = 0
        self.timer = self.create_timer(2.0, self._tick)
        self.get_logger().info('=== ESP Testbench gestartet, laufe Sequenz durch (2 s/Schritt) ===')

    # --- Hilfen zum Senden ---
    def _drive(self, pwm, steer):
        msg = Twist()
        msg.linear.x = float(pwm)
        msg.angular.z = float(steer)
        self.pub_cmd_vel.publish(msg)

    def _pid(self, param_id, value):
        msg = Float32MultiArray()
        msg.data = [float(param_id), float(value)]
        self.pub_pid_set.publish(msg)

    def _tick(self):
        # Zuerst pruefen, ob der vorige Schritt seine erwartete Antwort bekam.
        self._check_pending()

        if self.idx >= len(self.steps):
            self.get_logger().info('=== Sequenz fertig. Lausche weiter auf ESP-Events (Ctrl-C zum Beenden) ===')
            self.timer.cancel()
            return
        desc, action = self.steps[self.idx]
        expects = self.expected_response.get(desc)
        hint = f' (erwarte {expects} von der Bridge)' if expects else ' (Fire-and-forget, keine Antwort)'
        self.get_logger().info(f'[{self.idx + 1}/{len(self.steps)}] -> {desc}{hint}')
        try:
            action()
        except Exception as e:  # noqa: BLE001
            self.get_logger().error(f'Fehler bei "{desc}": {e}')
        self._pending = expects
        self._pending_desc = desc
        self._response_seen = False
        self.idx += 1

    def _check_pending(self):
        """Warnt, wenn auf einen antwortenden Befehl nichts von der Bridge kam."""
        if self._pending is not None and not self._response_seen:
            self.get_logger().warn(
                f'   <= KEINE Antwort von der Bridge auf "{self._pending_desc}" (erwartet: {self._pending})'
            )
        self._pending = None

    def _bridge_says(self, text):
        """Alles, was von der Bridge zurueckkommt, klar sichtbar ausgeben."""
        self._response_seen = True
        self.get_logger().info(f'   <= BRIDGE: {text}')

    # --- Empfangs-Logging (was die Bridge sagt) ---
    def _on_text(self, msg):
        # ASCII-Zeilen des ESP zaehlen nicht als "die erwartete Antwort".
        self.get_logger().info(f'   <= BRIDGE [ASCII]: {msg.data}')

    def _on_button(self, msg):
        self._bridge_says(f'BUTTON gedrueckt={msg.data}')

    def _on_pid(self, msg):
        self._bridge_says(f'PID_RSP Kp={msg.data[0]} Ki={msg.data[1]} Kd={msg.data[2]}')

    def _on_pid_saved(self, msg):
        self._bridge_says(f'PID_SAVED ok={msg.data}')

    def _on_move_done(self, msg):
        move_id, status, pos = msg.data
        names = {0: 'OK', 1: 'TIMEOUT', 2: 'ABORTED'}
        self._bridge_says(f'MOVE_DONE id={move_id} status={names.get(status, status)} pos={pos / 10.0:.1f} Grad')

    def _on_progress(self, msg):
        move_id, active, percent, ist, ziel = msg.data
        self._bridge_says(
            f'PROGRESS id={move_id} aktiv={active} {percent}% ist={ist / 10.0:.1f} ziel={ziel / 10.0:.1f}'
        )

    def _on_battery(self, msg):
        cell = msg.cell_voltage[0] if msg.cell_voltage else float('nan')
        self._bridge_says(f'BATTERY Pack={msg.voltage:.2f}V Zelle={cell:.2f}V')

    def _on_battery_warn(self, msg):
        # Unaufgefordert -- separat, aber trotzdem prominent.
        self.get_logger().warn(f'   <= BRIDGE [BATTERY_WARN]: Unterspannung! {msg.data}')


def main(args=None):
    rclpy.init(args=args)
    node = EspTestBench()
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
