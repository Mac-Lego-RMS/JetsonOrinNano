#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import String, Bool, Empty
import serial

class EspSerialBridge(Node):
    def __init__(self):
        super().__init__('esp_serial_bridge')
        
        # Variablen für den Byte-Parser
        self.text_buffer = ""
        self.parse_state = 'WAIT_START'
        self.current_cmd = 0x00
        
        try:
            self.ser = serial.Serial('/dev/ttyTHS1', 115200, timeout=0)
            self.get_logger().info('Serial Bridge gestartet auf /dev/ttyTHS1')
        except serial.SerialException as e:
            self.get_logger().error(f'Serial Error: {e}')
            exit(1)
            
        self.subscription = self.create_subscription(Twist, '/cmd_vel', self.cmd_vel_callback, 10)
        self.led_sub = self.create_subscription(Bool, '/led_cmd', self.led_callback, 10)
        self.calib_sub = self.create_subscription(Empty, '/calibrate_cmd', self.calibrate_callback, 10)
        
        self.esp_pub = self.create_publisher(String, '/esp/esp_data', 10)
        
        self.button_pub = self.create_publisher(Bool, '/button_state', 10)
        
        self.timer = self.create_timer(0.02, self.read_from_esp_callback)

    def calibrate_callback(self, msg):
        calib_packet = bytearray([0xA5, 0x40, 0x00])
        if self.ser.is_open:
            self.ser.write(calib_packet)
            self.ser.flush()
            self.get_logger().info('-> Kalibrierungsbefehl (0x40) gesendet')

    def led_callback(self, msg):
        state = 0x01 if msg.data else 0x00
        led_packet = bytearray([0xA5, 0x30, state])
        if self.ser.is_open:
            self.ser.write(led_packet)
            self.ser.flush()
            self.get_logger().info(f'-> LED Signal gesendet: {"AN" if msg.data else "AUS"}')

    def cmd_vel_callback(self, msg):
        target_pwm = msg.linear.x
        direction = 0x00 if target_pwm <= 0 else 0x01
        
        pwm_val = int(abs(target_pwm)) 
        pwm_val = max(0, min(1023, pwm_val)) 
        
        motor_packet = bytearray([0xA5, 0x10, direction, (pwm_val >> 8) & 0xFF, pwm_val & 0xFF])
        
        steering_target = msg.angular.z
        
        # Symmetrische Formel: 
        # Wenn steering_target = -1.0 (Rechts), wird 500 - (-1.0 * 220) = 720
        # Wenn steering_target = 1.0 (Links), wird 500 - (1.0 * 220) = 280
        servo_pos = - int((steering_target * 100))
            
        # Absolute physikalische Grenzen erzwingen (280 bis 720)
        servo_pos = max(-100, min(100, servo_pos))
        
        servo_packet = bytearray([0xA5, 0x20, 1, (servo_pos >> 8) & 0xFF, servo_pos & 0xFF])

        if self.ser.is_open:
            self.get_logger().info(f'-> Motor: {"Vorwärts" if direction == 0x00 else "Rückwärts"}, PWM={pwm_val} | Servo: {steering_target:.2f} -> Pos={servo_pos}')
            self.ser.write(motor_packet)
            self.ser.write(servo_packet)
            self.ser.flush()

    def read_from_esp_callback(self):
        if self.ser.is_open and self.ser.in_waiting > 0:
            try:
                raw_bytes = self.ser.read(self.ser.in_waiting)
                for byte in raw_bytes:
                    # Diagnose: Zeige aktuellen Zustand und das gelesene Byte als Hex
                    self.get_logger().info(f"Zustand: {self.parse_state} | Byte: {hex(byte)}")
                    
                    if self.parse_state == 'WAIT_START':
                        if byte == 0xA5:
                            self.parse_state = 'WAIT_CMD'
                            self.get_logger().info("-> Wechsel zu WAIT_CMD")
                        else:
                            if byte == ord('\n'):
                                if self.text_buffer.strip():
                                    out_msg = String()
                                    out_msg.data = self.text_buffer.strip()
                                    self.esp_pub.publish(out_msg)
                                self.text_buffer = ""
                            elif byte != ord('\r') and byte < 128:
                                self.text_buffer += chr(byte)
                                
                    elif self.parse_state == 'WAIT_CMD':
                        self.current_cmd = byte
                        if self.current_cmd == 0x70:
                            self.parse_state = 'WAIT_DATA'
                            self.get_logger().info("-> Wechsel zu WAIT_DATA")
                        else:
                            self.parse_state = 'WAIT_START'
                            self.get_logger().warn(f"Falsches Kommando {hex(byte)}, Abbruch zu WAIT_START")
                            
                    elif self.parse_state == 'WAIT_DATA':
                        if self.current_cmd == 0x70:
                            is_pressed = (byte == 0x01)
                            btn_msg = Bool()
                            btn_msg.data = is_pressed
                            self.button_pub.publish(btn_msg)
                            self.get_logger().info(f"=> ERFOLG: /button_state publiziert! (Wert: {is_pressed})")
                        
                        # Nach Verarbeitung immer zurücksetzen
                        self.parse_state = 'WAIT_START'
                        
            except Exception as e:
                self.get_logger().error(f'Python Parse Fehler: {e}')

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