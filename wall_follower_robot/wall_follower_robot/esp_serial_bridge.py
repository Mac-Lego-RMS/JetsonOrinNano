#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import String, Bool, Empty
import serial

class EspSerialBridge(Node):
    def __init__(self):
        super().__init__('esp_serial_bridge')
        self.read_buffer = ""
        
        try:
            self.ser = serial.Serial('/dev/ttyTHS1', 115200, timeout=0)
            self.get_logger().info('Serial Bridge gestartet auf /dev/ttyTHS1 (inklusive LED-Support)')
        except serial.SerialException as e:
            self.get_logger().error(f'Serial Error: {e}')
            exit(1)
            
        self.subscription = self.create_subscription(Twist, '/cmd_vel', self.cmd_vel_callback, 10)
        
        # --- NEU: Subscriber für die LED ---
        self.led_sub = self.create_subscription(Bool, '/led_cmd', self.led_callback, 10)

        # Subscriber für calibrate
        self.calib_sub = self.create_subscription(Empty, '/calibrate_cmd', self.calibrate_callback, 10)
        
        self.esp_pub = self.create_publisher(String, '/esp/esp_data', 10)
        self.timer = self.create_timer(0.02, self.read_from_esp_callback)

    def calibrate_callback(self, msg):
        # --- Kalibrierungsbefehl an ESP32 senden ---
        # Paket: [START_BYTE, CMD_CALIBRATE, DUMMY_BYTE] -> 3 Bytes
        calib_packet = bytearray([0xA5, 0x40, 0x00])
        
        if self.ser.is_open:
            self.ser.write(calib_packet)
            self.ser.flush()
            self.get_logger().info('-> Kalibrierungsbefehl (0x40) gesendet')

    def led_callback(self, msg):
        # --- NEU: LED-Befehl an ESP32 senden ---
        # msg.data ist True (AN) oder False (AUS)
        state = 0x01 if msg.data else 0x00
        
        # Der ESP32 erwartet bei 0x30 genau 1 Byte Daten.
        # Paket: [START_BYTE, CMD_LED, STATE] -> 3 Bytes
        led_packet = bytearray([0xA5, 0x30, state])
        
        if self.ser.is_open:
            self.ser.write(led_packet)
            self.ser.flush()
            self.get_logger().info(f'-> LED Signal gesendet: {"AN" if msg.data else "AUS"}')

    def cmd_vel_callback(self, msg):
        # --- 1. MOTOR (Direktes PWM) ---
        target_pwm = msg.linear.x
        direction = 0x00 if target_pwm <= 0 else 0x01
        
        pwm_val = int(abs(target_pwm)) 
        pwm_val = max(0, min(1023, pwm_val)) 
        
        motor_packet = bytearray([0xA5, 0x10, direction, (pwm_val >> 8) & 0xFF, pwm_val & 0xFF])
        
        # --- 2. SERVO (Symmetrisches Mapping) ---
        # ROS normiert: +1.0 (Links) bis -1.0 (Rechts)
        steering_target = msg.angular.z
        
        # Symmetrische Formel: 
        # Wenn steering_target = -1.0 (Rechts), wird 500 - (-1.0 * 220) = 720
        # Wenn steering_target = 1.0 (Links), wird 500 - (1.0 * 220) = 280
        servo_pos = - int((steering_target * 100))
            
        # Absolute physikalische Grenzen erzwingen (280 bis 720)
        servo_pos = max(-100, min(100, servo_pos))
        
        servo_packet = bytearray([0xA5, 0x20, 1, (servo_pos >> 8) & 0xFF, servo_pos & 0xFF])

        # --- SENDEN ---
        if self.ser.is_open:
            self.get_logger().info(f'-> Motor: {"Vorwärts" if direction == 0x00 else "Rückwärts"}, PWM={pwm_val} | Servo: {steering_target:.2f} -> Pos={servo_pos}')
            self.ser.write(motor_packet)
            self.ser.write(servo_packet)
            self.ser.flush()

    def read_from_esp_callback(self):
        if self.ser.is_open and self.ser.in_waiting > 0:
            try:
                raw_data = self.ser.read(self.ser.in_waiting).decode('utf-8', errors='ignore')
                self.read_buffer += raw_data
                while '\n' in self.read_buffer:
                    line, self.read_buffer = self.read_buffer.split('\n', 1)
                    line = line.strip()
                    if line:
                        out_msg = String()
                        out_msg.data = line
                        self.esp_pub.publish(out_msg)
            except Exception as e:
                self.get_logger().error(f'Error reading from serial: {e}')

def main(args=None):
    rclpy.init(args=args)
    node = EspSerialBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if hasattr(node, 'ser') and node.ser.is_open:
            node.ser.close()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()