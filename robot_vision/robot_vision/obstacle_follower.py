#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
from std_msgs.msg import String # <--- NEU: Für die YOLO-Befehle
from rclpy.qos import qos_profile_sensor_data
import math

class ObstacleFollower(Node): # <--- Name angepasst
    def __init__(self):
        super().__init__('obstacle_follower')
        self.sub_scan = self.create_subscription(LaserScan, '/ldlidar_node/scan', self.scan_callback, qos_profile_sensor_data)
        
        # --- NEU: Subscriber für YOLO ---
        self.sub_cmd = self.create_subscription(String, '/obstacle_cmd', self.cmd_callback, 10)
        self.current_obstacle_cmd = "CLEAR" # Startet immer mit freier Fahrt
        self.avoid_wall_dist = 0.25 # Ziel-Abstand zur Bande beim Ausweichen (25 cm)
        
        self.pub_cmd_vel = self.create_publisher(Twist, '/cmd_vel', 10)
        
        # --- ZUSTANDSAUTOMAT & RUNDENZÄHLER ---
        self.state = 'STARTUP'  
        self.turn_count = 0        
        self.target_turns = 12     
        self.locked_direction = None 

        # --- KURVEN-COOLDOWN ---
        self.last_turn_time = None
        self.finish_cooldown_sec = 1.0  
        
        # --- STARTUP TIMER ---
        self.start_time = self.get_clock().now()
        self.startup_duration = 0.5  
        
        # --- PARAMETER ---
        self.target_pwm = 1023.0 
        self.turn_pwm = 850.0      
        self.track_width = 0.90    
        self.target_dist = self.track_width / 2.0
        
        self.missing_wall_dist = 1.20 
        self.turn_trigger_dist = 0.9 
        self.finish_stop_dist = 1.80  
        
        # --- PID-PARAMETER ---
        self.kp = 1.8 
        self.kd = 0.0
        self.last_error = 0.0
        
        # --- PHANTOM-KURVEN FILTER ---
        self.turn_start_time = self.get_clock().now() 
        self.min_turn_duration = 0.7  
        
        self.get_logger().info('Obstacle Follower gestartet. Nulle Servo...')

    def cmd_callback(self, msg):
        """Empfängt die Befehle vom yolo_detector Node"""
        self.current_obstacle_cmd = msg.data

    def get_wall_distance(self, ranges, angle_min, angle_increment, target_angle_deg):
        # (Deine geniale Cluster-Logik bleibt 1:1 erhalten!)
        target_rad = math.radians(target_angle_deg)
        window_rad = math.radians(30.0) 
        
        center_idx = int((target_rad - angle_min) / angle_increment)
        window_idx = int((window_rad / 2.0) / angle_increment)
        
        start_idx = max(0, center_idx - window_idx)
        end_idx = min(len(ranges), center_idx + window_idx)
        
        clusters = []
        current_cluster = []
        
        for i in range(start_idx, end_idx):
            dist = ranges[i]
            if math.isinf(dist) or math.isnan(dist) or dist < 0.05:
                if current_cluster:
                    clusters.append(current_cluster)
                    current_cluster = []
                continue
            
            if not current_cluster:
                current_cluster.append(dist)
            else:
                if abs(dist - current_cluster[-1]) < 0.10: 
                    current_cluster.append(dist)
                else:
                    clusters.append(current_cluster)
                    current_cluster = [dist]
        
        if current_cluster: clusters.append(current_cluster)

        valid_clusters = [c for c in clusters if len(c) >= 5]
        if not valid_clusters: return 2.0 
            
        best_cluster = max(valid_clusters, key=len)
        return sum(best_cluster) / len(best_cluster)

    def scan_callback(self, msg):
        # ==========================================
        # STARTUP-PHASE
        # ==========================================
        if self.state == 'STARTUP':
            current_time = self.get_clock().now()
            elapsed_sec = (current_time - self.start_time).nanoseconds / 1e9
            
            cmd = Twist()
            cmd.linear.x, cmd.angular.z = 0.0, 0.0
            self.pub_cmd_vel.publish(cmd)
            
            if elapsed_sec > self.startup_duration:
                self.state = 'FOLLOW_BOTH'
                self.get_logger().warn('>>> HINDERNISRENNEN GESTARTET! <<<')
            return 
        
        # ==========================================
        # SENSOR-DATEN AUSLESEN
        # ==========================================
        dist_right = self.get_wall_distance(msg.ranges, msg.angle_min, msg.angle_increment, 0.0) 
        dist_front = self.get_wall_distance(msg.ranges, msg.angle_min, msg.angle_increment, 90.0)
        dist_left  = self.get_wall_distance(msg.ranges, msg.angle_min, msg.angle_increment, 180.0)
        
        error = 0.0
        steering_output = 0.0
        
        # ==========================================
        # LAUNCH CONTROL
        # ==========================================
        if self.turn_count == 0:
            current_pwm = 350.0  
            active_turn_pwm = 350.0      
        else:
            current_pwm = self.target_pwm 
            active_turn_pwm = self.turn_pwm       

        # ==========================================
        # ZIEL-LINIEN LOGIK
        # ==========================================
        if self.turn_count >= self.target_turns and self.state != 'FINISHED':
            cooldown_active = False
            if self.last_turn_time is not None:
                elapsed_since_turn = (self.get_clock().now() - self.last_turn_time).nanoseconds / 1e9
                if elapsed_since_turn < self.finish_cooldown_sec:
                    cooldown_active = True
            
            if not cooldown_active:
                if dist_left < 1.0 and dist_right < 1.0 and abs(dist_left - dist_right) < 0.30:
                    if dist_front <= self.finish_stop_dist:
                        self.state = 'FINISHED'
                        self.get_logger().warn('ZIEL ERREICHT!')
                    
        # ==========================================
        # STATE MACHINE LOGIK
        # ==========================================
        if self.state == 'FINISHED':
            current_pwm = 0.0
            steering_output = 0.0

        elif self.state == 'FOLLOW_BOTH':
            if dist_front < self.turn_trigger_dist:
                if dist_left < 0.45 and dist_right < 0.45:
                    error = dist_left - dist_right
                elif dist_front < 0.50 or dist_left > 0.70 or dist_right > 0.70:
                    if self.locked_direction is None:
                        self.locked_direction = 'LEFT' if dist_left > dist_right else 'RIGHT'
                    self.state = 'TURN_' + self.locked_direction
                    self.turn_start_time = self.get_clock().now() 
                else:
                    error = dist_left - dist_right
            elif dist_left > self.missing_wall_dist and dist_right < self.missing_wall_dist:
                self.state = 'FOLLOW_RIGHT'
            elif dist_right > self.missing_wall_dist and dist_left < self.missing_wall_dist:
                self.state = 'FOLLOW_LEFT' 
            else:
                error = dist_left - dist_right

        elif self.state == 'FOLLOW_LEFT':
            if dist_front < self.turn_trigger_dist:
                if dist_front < 0.50 or dist_right > 0.70:
                    self.state = 'TURN_' + (self.locked_direction if self.locked_direction else 'RIGHT')
                    self.turn_start_time = self.get_clock().now()
                else:
                    error = (dist_left - self.target_dist) * 2.0
            elif dist_right < self.missing_wall_dist:
                self.state = 'FOLLOW_BOTH'
            else:
                if dist_front < 1.40:
                    error, self.last_error = 0.0, 0.0
                else:
                    error = (dist_left - self.target_dist) * 2.0

        elif self.state == 'FOLLOW_RIGHT':
            if dist_front < self.turn_trigger_dist:
                if dist_front < 0.50 or dist_left > 0.70:
                    self.state = 'TURN_' + (self.locked_direction if self.locked_direction else 'LEFT')
                    self.turn_start_time = self.get_clock().now()
                else:
                    error = (self.target_dist - dist_right) * 2.0
            elif dist_left < self.missing_wall_dist:
                self.state = 'FOLLOW_BOTH'
            else:
                if dist_front < 1.40:
                    error, self.last_error = 0.0, 0.0
                else:
                    error = (self.target_dist - dist_right) * 2.0

        elif self.state == 'TURN_LEFT':
            if dist_front > 1.0:
                elapsed_turn = (self.get_clock().now() - self.turn_start_time).nanoseconds / 1e9
                if elapsed_turn >= self.min_turn_duration:
                    self.state, self.last_turn_time = 'FOLLOW_RIGHT', self.get_clock().now()
                    self.turn_count += 1 
                else:
                    self.state = 'FOLLOW_BOTH'
            else:
                steering_output, current_pwm, self.last_error = 1.0, active_turn_pwm, 0.0

        elif self.state == 'TURN_RIGHT':
            if dist_front > 1.0:
                elapsed_turn = (self.get_clock().now() - self.turn_start_time).nanoseconds / 1e9
                if elapsed_turn >= self.min_turn_duration:
                    self.state, self.last_turn_time = 'FOLLOW_LEFT', self.get_clock().now()
                    self.turn_count += 1
                else:
                    self.state = 'FOLLOW_BOTH'
            else:
                steering_output, current_pwm, self.last_error = -1.0, active_turn_pwm, 0.0

        # ==========================================
        # --- DER NEUE YOLO MASTER-OVERRIDE ---
        # ==========================================
        # Wenn wir normal geradeaus fahren (keine Kurve), darf YOLO eingreifen!
        if self.state in ['FOLLOW_BOTH', 'FOLLOW_LEFT', 'FOLLOW_RIGHT']:
            
            if self.current_obstacle_cmd == "AVOID_RIGHT":
                # Rotes Objekt -> Vorbei auf der rechten Seite -> Kuscheln mit der rechten Wand!
                error = (self.avoid_wall_dist - dist_right) * 2.0
                # self.get_logger().info('+++ OVERRIDE: Weiche ROT aus (Kuscheln Rechts) +++')
                
            elif self.current_obstacle_cmd == "AVOID_LEFT":
                # Grünes Objekt -> Vorbei auf der linken Seite -> Kuscheln mit der linken Wand!
                error = (dist_left - self.avoid_wall_dist) * 2.0
                # self.get_logger().info('+++ OVERRIDE: Weiche GRÜN aus (Kuscheln Links) +++')

            # --- PID BERECHNUNG ---
            derivative = error - self.last_error
            steering_output = (self.kp * error) + (self.kd * derivative)
            self.last_error = error
            
        steering_output = max(-1.0, min(1.0, steering_output))

        # --- BEFEHLE SENDEN ---
        cmd = Twist()
        cmd.linear.x = float(current_pwm) 
        cmd.angular.z = float(steering_output)
        self.pub_cmd_vel.publish(cmd)
        
        # Log-Ausgabe angepasst, damit man das YOLO Kommando sieht
        # self.get_logger().info(f'[{self.state}] CMD: {self.current_obstacle_cmd} | L:{dist_left:.1f} R:{dist_right:.1f} | PWM:{current_pwm}')
        
def main(args=None):
    rclpy.init(args=args)
    node = ObstacleFollower()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        stop_cmd = Twist()
        stop_cmd.linear.x, stop_cmd.angular.z = 0.0, 0.0
        node.pub_cmd_vel.publish(stop_cmd)
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()