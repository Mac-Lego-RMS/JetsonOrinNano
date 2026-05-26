#!/usr/bin/env python3

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_CORETYPE"] = "ARMV8" # Optimierung für Jetson CPUs

from platform import node

import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
import threading
from sensor_msgs.msg import LaserScan, Imu, Image
from geometry_msgs.msg import Twist, Point
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import String, Float64, Bool
from rclpy.qos import qos_profile_sensor_data
import math

# YOLO Imports 
import numpy as np

from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
from robot_vision.steering_lib import SteeringController
from robot_vision.camera_lib import TrackAnalyzer
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
import time
import logging

'''
=============================================================
      📍 DEIN HARDWARE-KOORDINATENSYSTEM (Lidar & Foxglove)
=============================================================

                          +Y (VORNE)
                              ^
                              |  Lidar: 180°
                              |  (Fahrtrichtung)
                              |
                              |
    (LINKS)                   |                   (RECHTS)
      -X  <--------------- [ 🤖 ] --------------->  +X
    Lidar: 270°               |                   Lidar: 90°
                              |
                              |
                              |
                              v
                          -Y (HINTEN)
                       Lidar: 360° / 0°

-------------------------------------------------------------
📐 MATHE-REGELN FÜR DIESEN NODE:
- Willst du die Karotte weiter nach VORNE schieben  -> Y wird größer (+Y)
- Willst du die Karotte weiter nach RECHTS schieben -> X wird größer (+X)
- Willst du die Karotte weiter nach LINKS schieben  -> X wird kleiner (-X)

🦊 FOXGLOVE ANSICHT (Top-Down):
- Oben auf dem Bildschirm  = +Y
- Rechts auf dem Bildschirm = +X

-------------------------------------------------------------
Zone Ids:

20  |   21     |
|   |   |      |
10  |   11     | <-- outer_wall
|   |   |      |
00  |   01     |
    ^
    |
   🤖
============================================================='''

class WallFollower(Node):
    def __init__(self):
        super().__init__('wall_follower')

        # --- MULTITHREADING SETUP ---
        # MutuallyExclusive: Callbacks in dieser Gruppe blockieren sich nur gegenseitig.
        # Wir trennen YOLO und LiDAR physisch voneinander.
        self.sensor_cbg = MutuallyExclusiveCallbackGroup()
        
        # Thread-Lock für Datensicherheit beim Zugriff auf gemeinsame Variablen
        self.data_lock = threading.Lock()
        
        
        
        self.scan_sub = self.create_subscription(
            LaserScan,
            '/ldlidar_node/scan',
            self.scan_callback,
            qos_profile_sensor_data,
            callback_group=self.sensor_cbg  # Läuft in eigenem Thread
        )
        
        # 1. Wir definieren exakt, wie ROS mit den Daten umgehen soll
        imu_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,  # Sensordaten werden meist als "Best Effort" gesendet
            history=HistoryPolicy.KEEP_LAST,            # Wir wollen nur die neuesten
            depth=1                                     # Und zwar exakt EINEN (Warteschlange abschaffen)
        )

        # 2. Wir übergeben das neue Profil anstelle der '10'
        self.sub_imu = self.create_subscription(
            Imu, 
            '/bno055/imu', 
            self.imu_callback, 
            imu_qos,
            callback_group=self.sensor_cbg
        )

        self.button_sub = self.create_subscription(
            Bool,
            '/button_state',
            self.button_callback,
            10
        )

        self.button_start = False

        self.led_pub = self.create_publisher(Bool, '/led_cmd', 10)

        self.button_state = False
        
        # Publisher für Bewegung und RViz [cite: 1, 19]
        self.pub_cmd_vel = self.create_publisher(Twist, '/cmd_vel', 10)
        self.pub_markers = self.create_publisher(MarkerArray, '/wall_follower_markers', 10)

        # Publisher für den Bézier-Pfad in Foxglove
        self.pub_path = self.create_publisher(Path, '/planned_trajectory', 10)
        

        self.yaw_offset = 0.0
        self.current_yaw = 0.0
        self.imu_ready = False
        self.last_raw_yaw = None
        self.start_turn_yaw = None
        self.start_straight_yaw = 0.0
        
        
        # Konfiguration
        self.rviz_frame = 'ldlidar_link'  # Muss in RViz als "Fixed Frame" stehen
        self.get_logger().info('>>> WallFollower Template gestartet. Warte auf LiDAR... <<<')

        # --- STATE MACHINE & PFADPLANUNG ---
        self.state = 'INITIALIZING'    # Startzustand
        self.turn_phase = 'APPROACH'  # Bei Kurven: 'APPROACH', 'TURNING', 'EXIT'

        self.fahrtrichtung = None
        self.saved_intersection_angle = None
        self.saved_curve_radius_m = None

        self.target_turns = 4
        self.turn_count = 0

        self.front_wall = None
        self.left_wall = None
        self.right_wall = None
        
        # Karotten-Parameter Geradeausfahrt
        self.lookahead_dist_straight = 0.20
        self.min_wall_dist = 0.15     

        self.lane_ratio = 0.40

        self.base_entry_distance = None
        self.assumed_lane_width = 0.60

        # --- PID-REGLER PARAMETER ---
        self.kp = 2.0   # Lenkt hart zur Karotte
        self.ki = 0.07   # Integral (oft bei WRO auf 0 gelassen, da schnelle Spurwechsel)
        self.kd = 0.05   # Verhindert das Schlingern (Dämpfung)
        
        self.prev_error = 0.0
        self.integral_error = 0.0

        # Karotten-Parameter Kurve
        self.steering_ctrl = SteeringController(logger=self.get_logger())
        self.lookahead_dist_turn = 0.20
        self.target_point = (0, 0)

        self.waiting_timer = None

        self.TRACK_WIDTH_M = 1.0
        self.ROBOT_WIDTH_M = 0.15
        self.LIDAR_OFFSET_M = 0.08
        self.SAFETY_MARGIN_M = 0.05
        self.MAX_KINEMATIC_RADIUS_M = 1.2
        self.IDEAL_RADIUS_M = 0.28
        self.MIN_TURN_RADIUS_M = 0.20

        self.max_wall_lenght_for_turn = 0.25

        # --- MOTOR PARAMETER (ESP PWM 0 - 1023) ---
        self.base_speed = 450.0  # Normale Geschwindigkeit auf der Geraden
        self.turn_speed = 450.0  # Leicht reduzierter Speed in der Kurve

        self.analyzer = TrackAnalyzer(
            logger=self.get_logger(),
            visualizer_cb=self.visualize_cluster_line
        )

        self.pub_timer = self.create_publisher(Float64, '/robot_timer', 10)
        self.start_time_stamp = None
        self.elapsed_time = 0.0
        self.timer_active = False

        ############ debug ##############
        self.begin = True
        self.counter = 0
        self.test_is_turning = False
        self.curve_radius_m = None
        self.camera_calibration = False

    def send_line(self, marker_array, m_id, p1, p2, color=(1.0, 1.0, 1.0)):
        """Hilfsfunktion zum Erstellen einer Linie für das MarkerArray."""
        marker = Marker()
        marker.header.frame_id = self.rviz_frame
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "walls"
        marker.id = m_id
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        marker.scale.x = 0.03  # Dicke der Linie
        marker.color.r, marker.color.g, marker.color.b = color
        marker.color.a = 1.0
        
        point1 = Point()
        point1.x, point1.y = float(p1[0]), float(p1[1])
        
        point2 = Point()
        point2.x, point2.y = float(p2[0]), float(p2[1])
        
        marker.points = [point1, point2]
        marker_array.markers.append(marker)

    def set_led(self, state: bool):
        # Sendet das Signal an deine bestehende esp_serial_bridge
        msg = Bool()
        msg.data = state
        self.led_pub.publish(msg)

    def imu_callback(self, msg):
        """
        Wandelt Quaternionen in einen UNENDLICHEN, sprungfreien Winkel um.
        (Kein Zurückspringen bei 180° oder 360°!)
        """
        self.imu_ready = True
        q = msg.orientation
        
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        yaw_rad = math.atan2(siny_cosp, cosy_cosp)
        
        # Das ist der rohe Wert mit dem blöden Sprung bei 180 / -180
        raw_yaw = math.degrees(yaw_rad)
        
        # Beim allerersten Datenpaket initialisieren wir einfach
        if self.last_raw_yaw is None:
            self.last_raw_yaw = raw_yaw
            self.current_yaw = raw_yaw
            return

        # Wie weit haben wir uns seit der letzten Millisekunde gedreht?
        delta = raw_yaw - self.last_raw_yaw
        
        # ==========================================
        # DIE MAGIE: Den 360°-Sprung abfangen!
        # ==========================================
        if delta > 180.0:
            delta -= 360.0
        elif delta < -180.0:
            delta += 360.0
            
        # Wir addieren nur das saubere Delta zu unserem unendlichen Winkel
        self.current_yaw += delta
        self.last_raw_yaw = raw_yaw
    
    def button_callback(self, msg):
        if msg.data:  # msg.data ist True, wenn der Button gedrückt wurde
            self.get_logger().info("Hardware-Interrupt empfangen. Trigger ausgelöst.")
            self.button_state = True

    def send_text(self, marker_array, m_id, text, x, y, color=(1.0, 1.0, 1.0), scale=0.15):
        """
        Fügt einen sichtbaren Text-Marker zum MarkerArray hinzu.
        scale: Schriftgröße in Metern (Standard 15cm)
        """
        marker = Marker()
        marker.header.frame_id = "base_link"  # Passe dies an euren Frame an (z.B. "laser")
        marker.header.stamp = self.get_clock().now().to_msg()
        
        marker.ns = "text_labels"
        marker.id = m_id
        marker.type = Marker.TEXT_VIEW_FACING  # Wichtig: Typ 9 für Text
        marker.action = Marker.ADD
        
        # Position
        marker.pose.position.x = float(x)
        marker.pose.position.y = float(y)
        # Leichtes Anheben (z = 0.1) verhindert oft, dass der Text im Grid/Boden verschwindet
        marker.pose.position.z = 0.1 
        
        # Ausrichtung (bei TEXT_VIEW_FACING rotiert der Text automatisch zur Kamera,
        # die Quaternion muss aber trotzdem valide sein)
        marker.pose.orientation.w = 1.0
        
        # SKALIERUNG: Bei Text ist NUR scale.z relevant (Schriftgröße in m)
        marker.scale.z = float(scale) 
        
        # Farbe und Sichtbarkeit
        marker.color.r = float(color[0])
        marker.color.g = float(color[1])
        marker.color.b = float(color[2])
        marker.color.a = 1.0  # Wichtig: 1.0 bedeutet 100% sichtbar (Deckkraft)
        
        # Der eigentliche String
        marker.text = str(text)
        
        marker_array.markers.append(marker)

    def send_sphere(self, marker_array, m_id, x, y, color=(0.0, 1.0, 1.0)):
        """Zeichnet eine leuchtende Kugel (die 'Karotte') in RViz."""
        marker = Marker()
        marker.header.frame_id = self.rviz_frame
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "target"
        marker.id = m_id
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.position.x = float(x)
        marker.pose.position.y = float(y)
        marker.pose.position.z = 0.05 # Leicht über dem Boden
        marker.scale.x = 0.15 # 15 cm Durchmesser
        marker.scale.y = 0.15
        marker.scale.z = 0.15
        marker.color.r, marker.color.g, marker.color.b = color
        marker.color.a = 1.0
        marker_array.markers.append(marker)
    
    def validate_clusters_straight(self, clusters):
        u_profile = [None, None, None]
        if not clusters:
            return u_profile

        # 1. Delta berechnen, wenn wir im Gyro-Modus sind
        # (Wir gehen davon aus, dass self.last_turn_aborted gesetzt wird)
        delta_yaw = 0.0
        delta_yaw = self.current_yaw - self.start_straight_yaw
            
        # Normierung auf -180 bis 180, falls der Sprung über die 180° Marke geht
        while delta_yaw > 180: delta_yaw -= 360
        while delta_yaw < -180: delta_yaw += 360

        right_candidates = []
        left_candidates = []
        front_candidates = []

        for c in clusters:
            if len(c) < 20: continue
            
            # Hier holen wir den lokalen Winkel (relativ zum Roboter)
            local_angle = self.get_cluster_angle(c)
            if local_angle is None: continue

            # ==========================================
            # DER GYRO-SHIFT
            # ==========================================
            # Wir addieren den Delta-Yaw zum lokalen Winkel.
            # Wenn der Roboter 30° nach rechts gedreht ist (Delta = -30),
            # wird eine Wand, die lokal bei +30° liegt, auf 0° korrigiert.
            shifted_angle = local_angle - delta_yaw
            
            # Jetzt nutzen wir den shifted_angle für die Normierung!
            angle_norm = abs(shifted_angle) % 180
            if angle_norm > 90:
                angle_norm = 180 - angle_norm

            # AB HIER BLEIBT ALLES GLEICH
            # Die Logik prüft nun, ob der korrigierte Winkel <= 45° ist.
            if angle_norm <= 45.0:
                # Seitenwand-Logik...
                mean_x_local = sum(p[1] for p in c) / len(c)
                
                if abs(mean_x_local) <= 1.0:
                    # GEOGRAFISCHER SPLIT
                    if mean_x_local > 0:
                        left_candidates.append(c)
                    else:
                        right_candidates.append(c)
                        
            # ==========================================
            # FRONTWÄNDE (> 45°)
            # ==========================================
            else:
                # Da p[2] bei dir Vorne/Hinten ist:
                mean_y = sum(p[2] for p in c) / len(c)
                if mean_y >= 0.15: # Nur Wände VOR dem Roboter
                    front_candidates.append(c)

        # ==========================================
        # 2. SEITENWÄNDE ZUORDNEN & PLAUSIBILITÄT (MIT HNF)
        # ==========================================
        best_right = right_candidates[0] if right_candidates else None
        best_left = left_candidates[0] if left_candidates else None
        
        best_left_hnf = self.cluster_to_hnf(best_left) if (best_left is not None) else None
        best_right_hnf = self.cluster_to_hnf(best_right) if (best_right is not None) else None

        if (best_right is not None) and (best_left is not None):
            self.visualize_cluster_line(best_right, 10, "cyan")
            self.visualize_cluster_line(best_left, 11, "magenta")
            
            if best_left_hnf is not None and best_right_hnf is not None:
                _, _, left_dist = best_left_hnf
                _, _, right_dist = best_right_hnf
                
                # Absolute HNF-Distanzen addieren, um die 40-Grad-Diagonale zu eliminieren!
                track_width = abs(left_dist) + abs(right_dist)
                self.get_logger().info(f"Echte Spurbreite: {track_width:.2f}m, L: {abs(left_dist):.2f}m, R: {abs(right_dist):.2f}m")
                
                if 0.45 <= track_width <= 1.25:
                    u_profile[0] = best_right
                    u_profile[2] = best_left
                else:
                    self.get_logger().warn(f"Spurbreite unplausibel ({track_width:.2f}m). Verwerfe kürzere Wand.")
                    len_r = math.hypot(best_right[-1][1] - best_right[0][1], best_right[-1][2] - best_right[0][2])
                    len_l = math.hypot(best_left[-1][1] - best_left[0][1], best_left[-1][2] - best_left[0][2])
                    
                    if len_r > len_l:
                        u_profile[0] = best_right
                    else:
                        u_profile[2] = best_left
            else:
                self.get_logger().warn("HNF Berechnung für eine der Wände fehlgeschlagen.")
                return [None, None, None]
                    
        elif best_right is not None:
            u_profile[0] = best_right
        elif best_left is not None:
            u_profile[2] = best_left

        # ==========================================
        # 3. FRONTWAND ZUORDNEN (HARDWARE-FIX & ZONEN-LOGIK)
        # ==========================================
        if front_candidates:
            # A) Y-Gruppierung (Tiefe / Abstand nach vorne)
            groups = []
            for c in front_candidates:
                mean_y = sum(p[2] for p in c) / len(c)
                
                # Alles was extrem nah ist, verwerfen
                if mean_y < 0.20:
                    continue 
                
                placed = False
                for g in groups:
                    if abs(g['base_y'] - mean_y) < 0.15:
                        g['clusters'].append(c)
                        placed = True
                        break
                if not placed:
                    groups.append({'base_y': mean_y, 'clusters': [c]})

            # B) Breiten-Filter & Hybrider Zonen-Check (Der Phantom-Wand Fix)
            valid_wall_groups = []
            for g in groups:
                all_x = [p[1] for c in g['clusters'] for p in c]
                total_width = max(all_x) - min(all_x)
                
                min_x = min(all_x)
                max_x = max(all_x)
                mean_y = g['base_y']  # Das ist die Distanz der Wand!
                
                # ----------------------------------------------------
                # DIE ZONEN-LOGIK
                # ----------------------------------------------------
                # 1. Ist die Wand im Fernbereich? (Keine Phantomwände möglich)
                is_far_wall = mean_y > 0.70
                min_allowed_width = 0.40
                
                # 2. Ist die Wand nah, aber blockiert physisch unseren Weg?
                # (Wir definieren einen Fahrkorridor von X = -0.15m bis X = +0.15m)
                is_blocking_path = (min_x < -0.15) and (max_x > 0.15)
                
                # Eine Wand ist gültig, wenn sie:
                # - breit genug ist (kein Rauschen, min 35cm) UND
                # - (entweder weit weg ist ODER unseren direkten Weg blockiert)
                if total_width > min_allowed_width and (is_far_wall or is_blocking_path):
                    if not is_far_wall and is_blocking_path:
                        self.get_logger().warn(f"Nahe aber blockierende Wand gefunden! Abstand: {mean_y:.2f}m")
                    valid_wall_groups.append(g)

            # ==========================================
            # 3. C) Zuweisung (SICHERE VERSION)
            # ==========================================
            if valid_wall_groups:
                # Nur wenn Gruppen den Zonen-Check bestanden haben, 
                # wählen wir die nächste/beste aus.
                valid_wall_groups.sort(key=lambda g: g['base_y'])
                winner_group = valid_wall_groups[0]['clusters']
                winner_group.sort(key=len, reverse=True)
                u_profile[1] = winner_group[0]
            else:
                # Wenn kein Cluster den Check bestanden hat, gibt es 
                # für diesen Durchlauf einfach keine Frontwand.
                u_profile[1] = None
                
                # Optionales Logging für die Fehlersuche:
                if front_candidates:
                    self.get_logger().debug("Front-Kandidaten vorhanden, aber als Phantomwände abgelehnt.")

        return u_profile

    def validate_clusters_turn(self, front_wall, point_data):   # np array upgraded
        '''Überprüft die Cluster in der Kurve. Es muss eine Frontwand geben, aber die Banden können auch fehlen (z.B. bei der ersten Kurve).
        Gibt die Cluster von rechts nach links zurück mit der Reihenfolge: [Rechte Bande, Frontwand, Linke Bande]. Fehlende Banden werden mit None ersetzt.'''

        if front_wall is None:
            self.get_logger().warn("Keine Frontwand gefunden. Kann Kurvenprofil nicht validieren.")
            return [None, None, None]

        clusters = self.get_all_clusters_sorted(point_data)

        def kill_all_clusters_between(front_wall, all_clusters):
            # Finde die Grenzen des Winkelbereichs
            left_angle = min(front_wall, key=lambda p: p[0])[0]
            right_angle = max(front_wall, key=lambda p: p[0])[0]
            
            kept_clusters = []
            
            for cluster in all_clusters:
                # any() prüft, ob IRGENDEIN Punkt im Cluster im kritischen Bereich liegt
                is_in_killzone = any(left_angle < point[0] < right_angle for point in cluster)
                
                # Nur wenn kein Punkt in der Zone liegt, behalten wir den Cluster
                if not is_in_killzone:
                    kept_clusters.append(cluster)
                    
            return kept_clusters


        front_wall_cluster, combined_clusters = self.merge_clusters(clusters, [front_wall])
        front_wall_cluster = front_wall_cluster[0]
        
        clusters = kill_all_clusters_between(front_wall_cluster, clusters)
        clusters.append(front_wall_cluster)


         # Wir fügen die Frontwand wieder hinzu, damit sie in der Sortierung berücksichtigt wird

        if len(clusters) >= 2:
            minimal_cluster_size = 25
            ordered = self.sort_clusters_right_to_left(clusters)
            u_profile = [None, None, None] # 0=Rechts, 1=Front, 2=Links
            if any(c is front_wall_cluster for c in ordered):
                u_profile[1] = front_wall_cluster
                fw_index = next(i for i, c in enumerate(ordered) if c is front_wall_cluster)
            else: 
                self.get_logger().warn("Frontwand nicht in den Clustern gefunden. Kann Kurvenprofil nicht validieren.")
                return [None, None, None]
                
            # Der Cluster LINKS von der Frontwand hat einen KLEINEREN Index
            while u_profile[2] is None and fw_index > 0:
                if len(ordered[fw_index - 1]) > minimal_cluster_size:
                    u_profile[2] = ordered[fw_index - 1]
                else: 
                    ordered.pop(fw_index - 1)
                    fw_index -= 1

            while u_profile[0] is None and fw_index < len(ordered) - 1:
                if len(ordered[fw_index + 1]) > minimal_cluster_size:
                   
                    u_profile[0] = ordered[fw_index + 1]
                    #self.get_logger().info(f"Cluster rechts von der Frontwand gefunden. Größe: {len(ordered[fw_index + 1])} Punkte.")
                    
                else: 
                    ordered.pop(fw_index + 1)

            angles = [self.get_cluster_angle(c) for c in u_profile]
            if angles[1] is None:
                self.get_logger().warn("Fehler bei der Winkelberechnung der Frontwand. Überspringe...")
                return [None, None, None]

            # --- HILFSFUNKTION FÜR WINKEL-DIFFERENZ ---
            # Berechnet den kleinsten Schnittwinkel zwischen zwei Geraden (0° bis 90°)
            def get_angle_diff(a1, a2):
                    diff = abs(a1 - a2) % 180
                    if diff > 90:
                        diff = 180 - diff
                    return diff

                # Differenzen berechnen (0=Rechts, 1=Front, 2=Links)
            if angles[0] is not None:
                diff_0_1 = get_angle_diff(angles[0], angles[1]) # Sollte ~90° sein (orthogonal)
            else: 
                diff_0_1 = None
            if angles[2] is not None:
                diff_1_2 = get_angle_diff(angles[1], angles[2]) # Sollte ~90° sein (orthogonal)
            else:
                diff_1_2 = None
            if angles[0] is not None and angles[2] is not None:
                diff_0_2 = get_angle_diff(angles[0], angles[2]) # Sollte ~0° sein (parallel)
            else:
                diff_0_2 = None


                # --- ÜBERPRÜFUNG ---
                # Toleranz: Wir erlauben bis zu 20° Abweichung von der perfekten Geometrie
                
                # Check A: Sind Rechts und Front orthogonal? (Differenz sollte > 70° sein)
            if diff_0_1 is not None and diff_0_1 < 70:
                self.get_logger().warn(f"Rechts und Front nicht orthogonal! Diff: {diff_0_1:.1f}°")
                u_profile[0] = None


                # Check B: Sind Front und Links orthogonal?
            if diff_1_2 is not None and diff_1_2 < 70:
                self.get_logger().warn(f"Front und Links nicht orthogonal! Diff: {diff_1_2:.1f}°")
                u_profile[2] = None
            
            
            try: 
                wall_dist_left = self.get_closest_point_in_cluster(u_profile[2])[3]
            except: 
                wall_dist_left = None
            try:
                wall_dist_right = self.get_closest_point_in_cluster(u_profile[0])[3]
            except:
                wall_dist_right = None


            if self.fahrtrichtung == "links":
                if wall_dist_right is not None:
                    if wall_dist_right > 1.0:
                        self.get_logger().warn(f"Abstand der rechten Wand {wall_dist_right:.2f}")
                        u_profile[0] = None

                if wall_dist_left is not None:
                    if 0.90 < wall_dist_left < 1.80:
                        self.get_logger().warn(f"Abstand der linken Wand {wall_dist_left:.2f}")
                        u_profile[2] = None

            else: 
                if wall_dist_left is not None:
                    if wall_dist_left > 1.0:
                        self.get_logger().warn(f"Abstand der linken Wand {wall_dist_left:.2f}")
                        u_profile[2] = None

                if wall_dist_right is not None:
                    if 0.90 < wall_dist_right < 1.8:
                        self.get_logger().warn(f"Abstand der rechten Wand {wall_dist_right:.2f}")
                        u_profile[0] = None
            
            u_profile, _ = self.merge_clusters(clusters, u_profile)
            self.visualize_cluster_line(u_profile[0], 0, "cyan")
            self.visualize_cluster_line(u_profile[1], 1, "cyan")
            self.visualize_cluster_line(u_profile[2], 2, "cyan")
            return u_profile

        self.get_logger().warn(f"Kein Cluster außer der Frontwand gefunden.")
        return [None, front_wall_cluster, None]
    
    def sort_clusters_right_to_left(self, clusters): # angepasst auf rechts 90° und links 270°
        """
        Nimmt eine Liste von Clustern und sortiert sie räumlich von rechts nach links.
        System: +X = Rechts, +Y = Vorne
        """
        if not clusters:
            return []

        def get_cluster_bearing(cluster):
            # 1. Schwerpunkt des Clusters berechnen
            # Index 1 = X, Index 2 = Y
            mean_x = sum(p[1] for p in cluster) / len(cluster)
            mean_y = sum(p[2] for p in cluster) / len(cluster)
            
            # 2. Winkel berechnen (0 = Vorne, Negativ = Rechts, Positiv = Links)
            # Das ist exakt die gleiche Logik wie bei unserem Lenk-Servo!
            return math.atan2(-mean_x, mean_y)

        # 3. Aufsteigend sortieren (kleinster/negativster Wert zuerst -> Rechts nach Links)
        sorted_clusters = sorted(clusters, key=get_cluster_bearing, reverse=True)
        
        return sorted_clusters

    def get_all_clusters_sorted(self, point_data):
        if len(point_data) < 2:
            return []

        # 1. Sortieren
        points = point_data[np.argsort(point_data[:, 0])]
        x = points[:, 1]
        y = points[:, 2]

        split_mask = np.zeros(len(points), dtype=bool)

        # ==========================================
        # 2. LÜCKENERKENNUNG (Manhattan-Distanz)
        # ==========================================
        dx = np.diff(x)
        dy = np.diff(y)
        dist_manhattan = np.abs(dx) + np.abs(dy)
        
        # PERFORMANCE-FIX 1: Zurück auf 0.15m (15cm). 
        # Verhindert das Zersplittern von Seitenwänden bei flachen Laser-Winkeln!
        # (Hindernisse sind ohnehin 40cm entfernt und werden weiterhin perfekt abgetrennt).
        split_mask[1:] = dist_manhattan >= 0.15

        # ==========================================
        # 3. AUSREISSER-LOGIK (Vektorisiert)
        # ==========================================
        if len(points) > 2:
            dist_2 = np.abs(x[2:] - x[:-2]) + np.abs(y[2:] - y[:-2])
            fix_2 = (split_mask[2:]) & (dist_2 < 0.15)
            split_mask[2:][fix_2] = False 

        if len(points) > 3:
            dist_3 = np.abs(x[3:] - x[:-3]) + np.abs(y[3:] - y[:-3])
            fix_3 = (split_mask[3:]) & (dist_3 < 0.15)
            split_mask[3:][fix_3] = False 

        # ==========================================
        # 4. ECKEN-ERKENNUNG (Skalarprodukt & Rasieren)
        # ==========================================
        if len(points) > 6:
            vec_a_x = x[3:-3] - x[:-6]
            vec_a_y = y[3:-3] - y[:-6]
            
            vec_b_x = x[6:] - x[3:-3]
            vec_b_y = y[6:] - y[3:-3]

            len_a = np.hypot(vec_a_x, vec_a_y)
            len_b = np.hypot(vec_b_x, vec_b_y)

            valid = (len_a > 0.01) & (len_b > 0.01) & (len_a < 0.30) & (len_b < 0.30)

            dot_product = np.ones(len(points) - 6)
            dot_product[valid] = (
                (vec_a_x[valid] / len_a[valid]) * (vec_b_x[valid] / len_b[valid]) +
                (vec_a_y[valid] / len_a[valid]) * (vec_b_y[valid] / len_b[valid])
            )

            sharp_angles = dot_product < 0.70
            padded_dot = np.pad(dot_product, (1, 1), mode='edge')
            is_local_min = (dot_product <= padded_dot[:-2]) & (dot_product <= padded_dot[2:])

            corner_splits = sharp_angles & is_local_min
            
            # PERFORMANCE-FIX 2: Vektorisiertes Rasieren ohne for-Schleife!
            corner_indices = np.where(corner_splits)[0] + 3 
            if len(corner_indices) > 0:
                idx_prev = np.clip(corner_indices - 1, 0, len(split_mask) - 1)
                idx_next = np.clip(corner_indices + 1, 0, len(split_mask) - 1)
                
                # Alle 3 Punkte auf einen Schlag True setzen (C-Speed)
                split_mask[corner_indices] = True
                split_mask[idx_prev] = True
                split_mask[idx_next] = True

        # ==========================================
        # 5. ARRAY ZERSCHNEIDEN & WRAP-AROUND
        # ==========================================
        split_indices = np.where(split_mask)[0]
        clusters_raw = np.split(points, split_indices)
        
        clusters = [c for c in clusters_raw if len(c) > 2]

        if len(clusters) > 1:
            c_last = clusters[-1]
            c_first = clusters[0]
            
            dist_wrap = abs(c_first[0][1] - c_last[-1][1]) + abs(c_first[0][2] - c_last[-1][2])
            
            # Auch hier wieder zurück auf 15cm
            if dist_wrap < 0.15 and len(c_last) > 3 and len(c_first) > 3:
                v1_x = c_last[-1][1] - c_last[-4][1]
                v1_y = c_last[-1][2] - c_last[-4][2]
                v2_x = c_first[3][1] - c_first[0][1]
                v2_y = c_first[3][2] - c_first[0][2]
                
                l1 = math.hypot(v1_x, v1_y)
                l2 = math.hypot(v2_x, v2_y)
                
                if l1 > 0 and l2 > 0:
                    dot_wrap = (v1_x * v2_x + v1_y * v2_y) / (l1 * l2)
                    if dot_wrap >= 0.85: 
                        clusters[0] = np.vstack((c_last, c_first))
                        clusters.pop()

        # ==========================================
        # 6. NACH PHYSISCHER LÄNGE SORTIEREN (METER)
        # ==========================================
        def get_physical_length(c):
            if len(c) < 2: return 0.0
            # Satz des Pythagoras zwischen dem ersten und dem letzten Punkt
            return math.hypot(c[-1][1] - c[0][1], c[-1][2] - c[0][2])

        clusters.sort(key=get_physical_length, reverse=True)
        return clusters
    
    def get_cluster_angle(self, cluster):
        if cluster is None or len(cluster) < 2:
            return None
        
        # Wandle cluster (Liste) in ein NumPy Array um, falls es noch keins ist
        pts = np.array(cluster) 
        x = pts[:, 1]
        y = pts[:, 2]
        
        # Schwerpunkt und Kovarianz in NumPy (völlig ohne Loops)
        x_mean = np.mean(x)
        y_mean = np.mean(y)
        dx = x - x_mean
        dy = y - y_mean
        
        s_xx = np.sum(dx * dx)
        s_yy = np.sum(dy * dy)
        s_xy = np.sum(dx * dy)
        
        angle_rad = 0.5 * np.arctan2(2.0 * s_xy, s_yy - s_xx)
        return np.degrees(angle_rad)

    def delete_marker(self, marker_array, m_id, ns="walls"):
        """Löscht einen Marker in einem bestimmten Namespace."""
        marker = Marker()
        marker.header.frame_id = self.rviz_frame
        marker.ns = ns  # Jetzt flexibel! Standardmäßig aber "walls"
        marker.id = m_id
        marker.action = Marker.DELETE
        marker_array.markers.append(marker)

    def scan_callback(self, msg):   # np array upgraded
        ranges = np.array(msg.ranges)

        # 1. Maske für gültige Werte erstellen (inf, nan oder außerhalb Reichweite filtern)
        valid_mask = np.isfinite(ranges) & (ranges >= 0.075) & (ranges <= 3.0)

        # 2. Lidar-Winkel für alle Punkte generieren
        angles_rad = msg.angle_min + np.arange(len(ranges)) * msg.angle_increment

        # 3. Nur gültige Werte übernehmen
        valid_ranges = ranges[valid_mask]
        valid_angles_rad = angles_rad[valid_mask]

        # 4. Koordinaten berechnen (Foxglove & Mathe Basis: X = Rechts, Y = Vorne)
        x_ros = valid_ranges * np.cos(valid_angles_rad)
        y_ros = valid_ranges * np.sin(valid_angles_rad)

        # 5. Neues Winkel-System (0-360, Startpunkt ist Hinten)
        angles_deg = np.degrees(valid_angles_rad)
        user_angles_deg = np.mod(angles_deg + 90.0, 360.0)

        # 6. Als N x 4 Array zusammenfügen: (Winkel, X, Y, Distanz)
        point_data = np.column_stack((user_angles_deg, x_ros, y_ros, valid_ranges))
        
        self.last_point_data = point_data

        if self.camera_calibration:
            self.test_turn_main_logic(point_data)
        else:
            self.main_logic(point_data)
            #self.test_turn_main_logic(point_data)
    
    def get_closest_point_in_cluster(self, cluster):
        """
        Gibt den Punkt eines Clusters zurück, der den kürzesten Abstand zum LiDAR hat.
        cluster: Liste von Punkten im Format (angle, x, y, dist)
        """
        if cluster is None or len(cluster) == 0:
            return None

        # Sucht das Element im Cluster, bei dem der Wert an Index 3 (dist) am kleinsten ist
        closest_point = min(cluster, key=lambda p: p[3])
        
        return closest_point

    def merge_clusters(self, all_clusters, validated_clusters): # np array upgraded
        """
        Versucht, benachbarte Cluster zu einem einzigen Cluster zu verschmelzen.
        Prüft den orthogonalen (senkrechten) Abstand und den parallelen (Längs-) Abstand.
        """
        # --- KONFIGURATION DER TOLERANZEN ---
        max_angle_gap = 10.0      # 10 Grad maximale Winkelabweichung
        max_perp_gap = 0.08       # Max 8 cm Abstand WEG von der Wand (verhindert Hindernis-Merge)
        max_parallel_gap = 0.60   # Max 25 cm Lücke ENTLANG der Wand (schließt Löcher)

        valid_ids = [id(v) for v in validated_clusters]
        remaining_clusters = [c for c in all_clusters if id(c) not in valid_ids]
        
        if not remaining_clusters:
            # self.get_logger().info("Keine Cluster zum Mergen gefunden")
            return validated_clusters, []

        def get_angle_diff(a1, a2):
            diff = abs(a1 - a2) % 180
            if diff > 90:
                diff = 180 - diff
            return diff

        combined_clusters = [[] for _ in range(len(validated_clusters))]

        for i, valid_cluster in enumerate(validated_clusters):
            angle = self.get_cluster_angle(valid_cluster)
            if valid_cluster is None or len(valid_cluster) == 0:
                continue

            if angle is None: 
                continue

            # Schwerpunkt des Basis-Clusters berechnen
            bx = np.mean(valid_cluster[:, 1])
            by = np.mean(valid_cluster[:, 2])

            angle_rad = math.radians(angle)
            
            # 1. RICHTUNGSVEKTOR (Parallel zur Wand)
            dir_x = math.sin(angle_rad)
            dir_y = math.cos(angle_rad)

            # 2. NORMALENVEKTOR (Senkrecht zur Wand)
            nx = math.cos(angle_rad)
            ny = -math.sin(angle_rad)

            clusters_to_remove = []

            for other in remaining_clusters:
                other_angle = self.get_cluster_angle(other)
                if other_angle is None: 
                    continue

                if get_angle_diff(angle, other_angle) < max_angle_gap:
                    
                    # 1. ORTHOGONALER OFFSET (Senkrecht zur Wand)
                    ox = np.mean(other[:, 1])
                    oy = np.mean(other[:, 2])
                    offset_perp = abs((ox - bx) * nx + (oy - by) * ny)

                    # 2. PARALLELER GAP (Lücke entlang der Wand)
                    # Wir projizieren alle Punkte beider Cluster auf den Richtungsvektor
                    proj_valid = valid_cluster[:, 1] * dir_x + valid_cluster[:, 2] * dir_y
                    proj_other = other[:, 1] * dir_x + other[:, 2] * dir_y
                    
                    # Ermitteln die Start- und Endpunkte der Cluster auf dieser Linie
                    min_v, max_v = np.min(proj_valid), np.max(proj_valid)
                    min_o, max_o = np.min(proj_other), np.max(proj_other)
                    
                    # Berechnen die Lücke (Ist 0, wenn sich die Cluster überlappen)
                    offset_parallel = max(0, min_o - max_v, min_v - max_o)

                    # 3. MERGE-BEDINGUNG BEIDER ACHSEN
                    if offset_perp < max_perp_gap and offset_parallel < max_parallel_gap:
                        # Referenz-Fix: Das Array direkt in der Hauptliste überschreiben
                        validated_clusters[i] = np.vstack((validated_clusters[i], other))
                        
                        clusters_to_remove.append(other)
                        combined_clusters[i].append(other)

            # Zugeordnete Cluster aus der iterierbaren Liste entfernen
            remove_ids = [id(c) for c in clusters_to_remove]
            remaining_clusters = [c for c in remaining_clusters if id(c) not in remove_ids]

            # Letzter Schritt: Den gemergten Cluster sauber entlang der Wand sortieren
            projections = validated_clusters[i][:, 1] * dir_x + validated_clusters[i][:, 2] * dir_y
            sort_indices = np.argsort(projections)
            validated_clusters[i] = validated_clusters[i][sort_indices]

        return validated_clusters, combined_clusters

    def visualize_target_point(self, x, y, m_id=24, farbe_name="gelb", label="TARGET"):
        """
        Visualisiert die 'Karotte' (Zielpunkt) als Kugel und Text-Label im Foxglove.
        Nutzt das bestehende MarkerArray-System.
        """
        # 1. Farben definieren (analog zu deiner Cluster-Funktion)
        farben = {
            "rot": (1.0, 0.0, 0.0),
            "gruen": (0.0, 1.0, 0.0),
            "blau": (0.0, 0.5, 1.0),
            "cyan": (0.0, 1.0, 1.0),
            "magenta": (1.0, 0.0, 1.0),
            "gelb": (1.0, 1.0, 0.0),
            "orange": (1.0, 0.5, 0.0)
        }
        rgb = farben.get(farbe_name.lower(), (1.0, 1.0, 0.0))

        # 2. MarkerArray initialisieren
        marker_array = MarkerArray()

        # 3. Die Kugel (Sphere) für den Punkt erstellen
        sphere_marker = Marker()
        sphere_marker.header.frame_id = "base_link"
        sphere_marker.header.stamp = self.get_clock().now().to_msg()
        sphere_marker.ns = "target_point"
        sphere_marker.id = m_id
        sphere_marker.type = Marker.SPHERE
        sphere_marker.action = Marker.ADD
        
        # Position setzen
        sphere_marker.pose.position.x = float(x)
        sphere_marker.pose.position.y = float(y)
        sphere_marker.pose.position.z = 0.1  # 10cm über dem Boden
        
        # Größe (Scale)
        sphere_marker.scale.x = 0.15
        sphere_marker.scale.y = 0.15
        sphere_marker.scale.z = 0.15
        
        # Farbe setzen
        sphere_marker.color.r = rgb[0]
        sphere_marker.color.g = rgb[1]
        sphere_marker.color.b = rgb[2]
        sphere_marker.color.a = 1.0
        
        marker_array.markers.append(sphere_marker)

        # 4. Text-Label hinzufügen (Nutzt deine vorhandene send_text Methode!)
        # Wir setzen den Text ein Stück über die Kugel (z+0.2), damit man ihn besser sieht
        self.send_text(marker_array, m_id=m_id + 1, text=label, x=x, y=y, color=rgb)
        
        # Optional: Den Text etwas anheben, falls send_text das nicht kann, 
        # musst du in deiner send_text Methode ggf. das Z-Attribut anpassen.

        # 5. Veröffentlichen auf dem zentralen Marker-Topic
        self.pub_markers.publish(marker_array)

    def get_target_point_straight(self, hnf_innen, hnf_aussen):
        """
        Berechnet den Zielpunkt (Karotte) mithilfe der HNF.
        Ist nur eine Wand vorhanden, wird der Offset direkt von dieser berechnet, 
        ohne eine fehleranfällige virtuelle Wand zu projizieren!
        """
        target_y = self.lookahead_dist_straight
        target_x = 0.0
        
        # Plausibilitäts-Check: Wände, die zu weit weg sind, ignorieren
        if hnf_innen is not None and hnf_innen[2] > 1.0:
            hnf_innen = None
        if hnf_aussen is not None and hnf_aussen[2] > 1.0:
            hnf_aussen = None

        def get_x_at_y(hnf_params, y_val):
            nx, ny, d = hnf_params
            if abs(nx) < 1e-6: return 0.0 
            return (d - ny * y_val) / nx

        # 1. Wo schneiden die erkannten Wände unsere Y-Sichtachse?
        x_innen = get_x_at_y(hnf_innen, target_y) if hnf_innen else None
        x_aussen = get_x_at_y(hnf_aussen, target_y) if hnf_aussen else None

        # ==========================================
        # 2. ZIELPUNKT BERECHNEN (Direktes Offsetting)
        # ==========================================
        # lane_ratio ist der Abstand zur INNENBANDE (z.B. 0.60).
        # Der Offset zur AUSSENBANDE ist logischerweise (1.0 - lane_ratio) (z.B. 0.40).

        if x_innen is not None and x_aussen is not None:
            # Beide Wände da: Wir mitteln die Karotte aus BEIDEN Wänden! 
            # Das halbiert das Sensor-Rauschen und der Roboter fährt wie auf Schienen.
            _, _, dist_innen = hnf_innen
            _, _, dist_aussen = hnf_aussen
            lane_width = abs(dist_innen + dist_aussen)
            if x_innen < 0: # Innenbande ist links
                t_innen = x_innen + self.lane_ratio
                t_aussen = x_aussen - (lane_width * self.lane_ratio)
            else:           # Innenbande ist rechts
                t_innen = x_innen - self.lane_ratio
                t_aussen = x_aussen + (lane_width * self.lane_ratio)
                
            target_x = (t_innen + t_aussen) / 2.0

        elif x_innen is not None:
            # NUR Innenbande da (z.B. bei Ausfahrt aus der Kurve)
            if x_innen < 0:
                target_x = x_innen + (self.assumed_lane_width * self.lane_ratio)
            else:
                target_x = x_innen - (self.assumed_lane_width * self.lane_ratio)

        elif x_aussen is not None:
            # NUR Außenbande da (DAS IST DEINE KURVEN-ANNÄHERUNG!)
            inv_ratio = 1.0 - self.lane_ratio
            if x_aussen < 0: 
                target_x = x_aussen + (self.assumed_lane_width * inv_ratio)
            else:
                target_x = x_aussen - (self.assumed_lane_width * inv_ratio)
                
        else:
            # Notfall (Beide Wände fehlen komplett)
            target_x = 0.0

        # ==========================================
        # 3. KRAFTFELD (MINDESTABSTAND ERZWINGEN)
        # ==========================================
        # Verhindert, dass wir zu nah an EINE der sichtbaren Wände driften
        if x_innen is not None:
            if x_innen < 0 and target_x < x_innen + self.min_wall_dist:
                target_x = x_innen + self.min_wall_dist
            elif x_innen > 0 and target_x > x_innen - self.min_wall_dist:
                target_x = x_innen - self.min_wall_dist
                
        if x_aussen is not None:
            if x_aussen < 0 and target_x < x_aussen + self.min_wall_dist:
                target_x = x_aussen + self.min_wall_dist
            elif x_aussen > 0 and target_x > x_aussen - self.min_wall_dist:
                target_x = x_aussen - self.min_wall_dist

        return (target_x, target_y)

    def track_front_wall(self, point_data, last_front_wall):    # np array upgraded
        if last_front_wall is None or len(point_data) == 0:
            return None

        # 1. Wo war die Wand im letzten Frame? 
        # last_front_wall kann noch eine Liste oder schon ein Array sein
        last_fw_array = np.array(last_front_wall)
        min_angle = np.min(last_fw_array[:, 0])
        max_angle = np.max(last_fw_array[:, 0])

        # 2. Dynamisches Suchfenster (ROI)
        if self.fahrtrichtung == 'links':
            roi_min = min_angle - 30.0
            roi_max = max_angle + 5.0
        else:
            roi_min = min_angle - 5.0
            roi_max = max_angle + 30.0

        # 3. Scheuklappen aufsetzen: NumPy Maske statt for-Schleife
        mask = (point_data[:, 0] >= roi_min) & (point_data[:, 0] <= roi_max)
        roi_points = point_data[mask]

        # 4. Nur diese gefilterten Punkte in Cluster aufteilen
        roi_clusters = self.get_all_clusters_sorted(roi_points)

        # 5. Tracking überprüfen
        if not roi_clusters:
            self.get_logger().warn("ACHTUNG: Getrackte Wand im ROI verloren!")
            return last_front_wall

        return roi_clusters[0]

    def get_closest_measure(self, point_data, target_angle):    # np array upgraded
        if len(point_data) == 0:
            self.get_logger().info(f"Point_Data ist leer!")
            return None

        # Winkel-Differenzen für das gesamte Array auf einmal berechnen
        diffs = (point_data[:, 0] - target_angle + 180.0) % 360.0 - 180.0
        abs_diffs = np.abs(diffs)

        # Index des kleinsten Abstands finden
        closest_idx = np.argmin(abs_diffs)
        
        return point_data[closest_idx]

    # -----------------------
    # --- YOLO - Function ---
    # -----------------------

    def angle_diff(self, a, b):
        """Berechnet den kleinsten Unterschied zwischen zwei Winkeln (Rad)."""
        # Sorgt dafür, dass der Unterschied auch über die 0/360° Grenze korrekt bleibt
        return math.atan2(math.sin(a - b), math.cos(a - b))

    # -----------------------------------------------
    # Neue Koordinatensystem Idee für Kurvenfahrten
    # -----------------------------------------------

    def cluster_to_hnf(self, cluster):
        """
        Berechnet die Hessesche Normalform aus einem Cluster von Messpunkten.
        Erwartetes Cluster-Format: Liste aus Tupeln (winkel, x_coord, y_coord, abstand)
        """
        if cluster is None:
            return None
        
        # 1. Extrahieren der x- und y-Koordinaten in ein NumPy-Array
        # Index 1 ist x_coord, Index 2 ist y_coord
        points = np.array([[p[1], p[2]] for p in cluster])
        
        # Sicherstellen, dass das Cluster nicht leer ist
        if len(points) == 0:
            raise ValueError("Das Cluster ist leer.")
            
        # 2. Schwerpunkt berechnen
        centroid = np.mean(points, axis=0)
        
        # 3. Daten zentrieren
        centered_points = points - centroid
        
        # 4. Singulärwertzerlegung (SVD) für orthogonale Regression
        # Vh enthält die Eigenvektoren der Kovarianzmatrix
        _, _, Vh = np.linalg.svd(centered_points)
        
        # Der Normalenvektor entspricht dem Eigenvektor der geringsten Varianz
        # Bei der SVD in NumPy ist dies die letzte Zeile von Vh
        normal_vector = Vh[-1]
        
        # 5. Abstand d berechnen (Skalarprodukt aus Schwerpunkt und Normalenvektor)
        d = np.dot(centroid, normal_vector)
        
        # 6. Normierung: d muss größer oder gleich 0 sein
        if d < 0:
            normal_vector = -normal_vector
            d = -d
            
        n_x, n_y = normal_vector
        
        return n_x, n_y, d
    
    def extract_wall_lines(self, u_profile):
        if self.fahrtrichtung == "links":
            side_cluster, front_cluster, opposite_cluster = u_profile
        else:
            opposite_cluster, front_cluster, side_cluster = u_profile

        # 1. Frontwand extrahieren
        # Reihenfolge ist wichtig: 'is None' schützt 'len()' vor Fehlern
        if front_cluster is None or len(front_cluster) < 2:
            front_straight = None
        else:
            front_straight = self.cluster_to_hnf(front_cluster)

        # 2. Seitenwand extrahieren
        if side_cluster is not None and len(side_cluster) >= 2:
            # Primäres Ziel: Die innere Kurvenwand
            side_straight = self.cluster_to_hnf(side_cluster)
            
        elif opposite_cluster is not None and len(opposite_cluster) >= 2:
            # Fallback: Gegenüberliegende Wand über die 3m-Distanz spiegeln
            oppo_x, oppo_y, oppo_d = self.cluster_to_hnf(opposite_cluster)
            
            new_nx = -oppo_x
            new_ny = -oppo_y
            new_d = 3.0 - oppo_d
            
            # HNF-Bedingung: d muss immer >= 0 sein
            if new_d < 0:
                self.get_logger().warn("WARNUNG: HNF-Distanz negativ. Spiegelung korrigiert.")
                new_nx = -new_nx
                new_ny = -new_ny
                new_d = abs(new_d)
                
            side_straight = (new_nx, new_ny, new_d)
            
        else:
            # Keine verwertbaren Seitenwände vorhanden
            side_straight = None

        return front_straight, side_straight

    def calculate_target_line(self, side_line_params, front_line_params, desired_lane_ratio):
        if front_line_params is None:
            return None, None

        n_xf, n_yf, d_f = front_line_params
        
        # 1. Basis-Zielgerade
        d_ziel = d_f - desired_lane_ratio
            
        target_line_params = (n_xf, n_yf, d_ziel)
        
        if side_line_params is None:
            return target_line_params, None

        # 3. Radius Limitierung
        delta_d_neu = d_f - d_ziel
        
        r_max_ziel = self.TRACK_WIDTH_M - delta_d_neu - (self.ROBOT_WIDTH_M / 2.0)
        
        max_allowed_radius_m = min(r_max_ziel, self.MAX_KINEMATIC_RADIUS_M)
        
        # Sicherstellen, dass der Radius physikalisch fahrbar bleibt (> 0)
        max_allowed_radius_m = max(max_allowed_radius_m, 0.0)
        
        return target_line_params, max_allowed_radius_m
    
    def get_intersection_point(self, target_line_params):
        """
        Berechnet den Schnittpunkt der y-Achse (Roboter-Trajektorie) mit der Zielgeraden.
        """

        if target_line_params is None:
            return None, None, None
            
        n_x, n_y, d = target_line_params
        
        # Sicherheitsprüfung auf Parallelität (Vermeidung von Division durch Null)
        epsilon = 1e-6
        if abs(n_y) < epsilon:
            # Zielgerade ist parallel zur Fahrtrichtung, kein Schnittpunkt
            return None, None, None
            
        # Schnittpunkt berechnen
        intersection_x_m = 0.0  # Roboter fährt per Definition auf x=0
        intersection_y_m = d / n_y
        
        # Plausibilitätsprüfung: Schnittpunkt muss vor dem Roboter (in Fahrtrichtung) liegen
        if intersection_y_m <= 0.0:
            # Mathematischer Schnittpunkt liegt hinter dem Roboter. 
            # Indiziert Fehler im Lidar-Clustering oder Odometrie-Sprung.
            return None, None, None
        
        # 1. Vorzeichen des Skalarprodukts basierend auf der Kurvenrichtung setzen
        if self.fahrtrichtung == 'links':
            nx_directional = n_x
        else:
            nx_directional = -n_x
            
        # 2. Clipping auf [-1.0, 1.0], aber OHNE den Betrag (abs)
        nx_clipped = max(-1.0, min(1.0, nx_directional))
        
        # 3. Winkel berechnen (liefert jetzt Werte zwischen 0° und 180°)
        turn_angle_deg = math.degrees(math.acos(nx_clipped))
        self.get_logger().info(f"Turn Angle Fehler: {turn_angle_deg:.1f}°")

        return intersection_x_m, intersection_y_m, turn_angle_deg

    def calculate_curve_geometry(self, intersection_y_m, turn_angle_deg, max_allowed_radius_m):
        """
        Berechnet den optimalen Kurvenradius und die Distanz zum Einlenkpunkt auf der y-Achse.
        """
        
        if intersection_y_m is None:
            # Kurve ist geometrisch oder mechanisch unmöglich
            self.get_logger().error("Ungültiger Schnittpunkt: Kein Einlenken möglich.")
            return None, None

        if max_allowed_radius_m is None:
            max_allowed_radius_m = self.MAX_KINEMATIC_RADIUS_M

        if max_allowed_radius_m < self.MIN_TURN_RADIUS_M:
            curve_radius_m = self.MIN_TURN_RADIUS_M # Fehler: Mögliche Kollison mit der Wand, da der Platz nicht ausreicht
            # return None, None 
        else:
            curve_radius_m = max(self.MIN_TURN_RADIUS_M, min(self.IDEAL_RADIUS_M, max_allowed_radius_m))
        # Radius wird zwischen dem mechanischen Minimum und dem Platz-Maximum eingeklemmt
        
        
        # 2. Tangentenlänge (Distanz vom Schnittpunkt zum Einlenkpunkt) berechnen
        alpha_rad = math.radians(abs(turn_angle_deg))
        tangent_length_m = curve_radius_m * math.tan(alpha_rad / 2.0)
        
        lidar_entry_dist_m = intersection_y_m - tangent_length_m
        
        real_axle_dist_m = lidar_entry_dist_m + self.LIDAR_OFFSET_M
        
        if real_axle_dist_m <= 0.0:
            self.get_logger().warn(
                f"NOT-EINLENKEN: Drehpunkt verpasst ({real_axle_dist_m:.2f} m). "
                "Forciere sofortige Kurve."
            )
            # Setze auf 0.0, damit JEDER Trigger sofort feuert
            real_axle_dist_m = 0.0
            
        return curve_radius_m, real_axle_dist_m

    def check_turn_trigger(self, entry_point_distance_m):
        """
        Prüft anhand der aktuell berechneten Distanz, ob das Einlenkmanöver starten muss.
        """
        if entry_point_distance_m is None:
            return False

        # Toleranzwert (z.B. 7 cm), um Verzögerungen in der Hauptschleife abzufangen
        trigger_tolerance_m = 0.05

        if entry_point_distance_m <= trigger_tolerance_m:
            return True
            
        return False

    def execute_turn(self, curve_radius_m):
        """
        Übersetzt den Radius über den SteeringController und publisht die Twist-Message.
        """

        if self.fahrtrichtung == "links":
            is_left_turn = True
        else:
            is_left_turn = False

        cmd = Twist()
        
        # 1. Sicherheitscheck auf ungültige Geometrie
        if curve_radius_m is None or curve_radius_m <= 0.0:
            self.get_logger().error("Ungültiger Kurvenradius.")
            return False
        
        # 3. Abruf des kalibrierten PWM-Signals (-1.0 bis 1.0)
        steering_signal = self.steering_ctrl.get_steering_for_radius(target_radius=curve_radius_m,fahrtrichtung_ist_links=is_left_turn)
        self.get_logger().info(f"Steering-Signal: {steering_signal:.3f}")

        # 4. Twist-Message konstruieren und senden
        cmd.linear.x = float(self.turn_speed)
        cmd.angular.z = float(steering_signal)
        
        self.pub_cmd_vel.publish(cmd)

        self.turn_speed = abs(self.turn_speed)
        
        return True

    def check_turn_completion_fused(self, turn_angle, front_line_params):
        """
        Kombiniert Gyro-Daten mit dem Wandwinkel für maximale Präzision am Kurvenausgang.
        """
        # 1. ROBUSTE GYRO-BERECHNUNG (Wrap-Around-sicher)
        # Differenz berechnen und zwingend auf [-180, 180] normalisieren
        yaw_diff = (self.current_yaw - self.start_turn_yaw + 180) % 360 - 180
        progressed_angle = abs(yaw_diff)
        
        target_angle_abs = abs(turn_angle)
        
        # 2. HARD-EXIT BEI ÜBERSCHWUNG (Verhindert Deadlock)
        # Wenn wir das Ziel (abzüglich 10° Toleranz) physikalisch erreicht haben, 
        # MUSS die Kurve beendet werden, egal was das LiDAR sagt.
        if progressed_angle >= (target_angle_abs - 10.0):
            self.get_logger().info(f"Kurve beendet (Gyro Hard-Exit): {progressed_angle:.1f}° erreicht.")
            return True
            
        # 3. EINTRITTS-FENSTER FÜR SENSOR-FUSION
        # Wir sind noch zu weit weg vom Ziel (> 25° fehlen). Sicher weiterdrehen.
        if progressed_angle < (target_angle_abs - 25.0):
            return False
            
        # 4. PRÄZISE PRÜFUNG VIA LIDAR
        # Wir sind im Fenster (-25° bis -10° vor dem Ziel). Jetzt darf die Wand die Kurve vorzeitig beenden.
        if front_line_params is not None:
            n_x, n_y, d = front_line_params
            
            # WICHTIG: Kein abs() in atan2! Das zerstört die Quadranten-Geometrie des Vektors.
            wall_angle_deg = math.degrees(math.atan2(n_y, n_x))
            
            # Fehler zur perfekten Orthogonalität berechnen (0° oder 180° zur X-Achse)
            wall_error_deg = min(abs(wall_angle_deg % 180), abs(180 - (wall_angle_deg % 180)))
            
            if wall_error_deg < 15.0:
                self.get_logger().info(f"Fused Match! Gyro bei {progressed_angle:.1f}°, Wand perfekt parallel (Error: {wall_error_deg:.1f}°).")
                return True

        return False

    # -----------------------------------------------
    # Testfunktionen
    # -----------------------------------------------

    def clear_all_lines(self):
        """
        Löscht alle Linien-Marker im Namespace 'walls'.
        """
        marker_array = MarkerArray()
        # Wir löschen sicherheitshalber die Marker-IDs 0 bis 50.
        # Falls du mal mehr Wände erwartest, kannst du die Zahl hier erhöhen.
        for i in range(25):
            self.delete_marker(marker_array, m_id=i, ns="walls")
        self.pub_markers.publish(marker_array)

    def visualize_cluster_line(self, cluster, m_id, farbe_name="rot", label="CLUSTER_LINE"):
        """
        Erstellt einen Marker für eine Gerade direkt aus dem ersten und letzten Punkt eines Clusters.
        Das spart enorm CPU-Leistung im Vergleich zur HNF-Visualisierung.
        
        Args:
            cluster: Liste oder Numpy-Array von Punkten (min. 2 Punkte nötig)
            m_id: Eindeutige ID für den Marker
            farbe_name: "rot", "gruen", "blau", "cyan", "magenta", "gelb"
            label: Textbeschriftung an der Linie
        """
        # Sicherheitsabbruch: Brauchen mindestens 2 Punkte für eine Linie
        if cluster is None or len(cluster) < 2:
            return

        # Farb-Mapping (RGB-Werte normiert auf 0.0 - 1.0)
        farben = {
            "rot": (1.0, 0.0, 0.0),
            "gruen": (0.0, 1.0, 0.0),
            "blau": (0.0, 0.5, 1.0),
            "cyan": (0.0, 1.0, 1.0),
            "magenta": (1.0, 0.0, 1.0),
            "gelb": (1.0, 1.0, 0.0)
        }
        
        # Fallback auf Weiß (1.0, 1.0, 1.0), falls Farbe nicht existiert
        rgb = farben.get(farbe_name.lower(), (1.0, 1.0, 1.0))
        
        # 1. Start- und Endpunkt aus dem Cluster extrahieren 
        # (Index 1 = X, Index 2 = Y in deinem Format)
        start_p = (cluster[0][1], cluster[0][2])
        ende_p = (cluster[-1][1], cluster[-1][2])
        
        # 2. Mittelpunkt für das Text-Label berechnen
        mitte_x = (start_p[0] + ende_p[0]) / 2.0
        mitte_y = (start_p[1] + ende_p[1]) / 2.0
        
        # 3. Marker-Array vorbereiten
        marker_array = MarkerArray()
        
        # 4. Linie zeichnen (Nutzt deine interne send_line Methode)
        self.send_line(marker_array, m_id=m_id, p1=start_p, p2=ende_p, color=rgb)
        
        # 5. Text-Label exakt in der Mitte der Linie platzieren
        self.send_text(marker_array, m_id=m_id + 1000, text=label, x=mitte_x, y=mitte_y, color=rgb)
        
        # 6. Veröffentlichen
        self.pub_markers.publish(marker_array)

    def visualize_hnf_line(self, hnf_params, m_id, farbe_name="rot", label="HNF_LINE"):
        """
        Erstellt einen Marker für eine Gerade aus der Hesseschen Normalform (n_x, n_y, d).
        
        Args:
            hnf_params: Tuple (n_x, n_y, d)
            m_id: Eindeutige ID für den Marker
            farbe_name: "rot", "gruen" oder "blau"
            label: Textbeschriftung an der Linie
        """
        if hnf_params is None:
            return

        # Farb-Mapping (RGB-Werte normiert auf 0.0 - 1.0)
        farben = {
            "rot": (1.0, 0.0, 0.0),
            "gruen": (0.0, 1.0, 0.0),
            "blau": (0.0, 0.5, 1.0)
        }
        
        # Fallback auf Weiß, falls die Farbe nicht im Dictionary ist
        rgb = farben.get(farbe_name.lower(), (0.0, 0.0, 0.0))
        
        n_x, n_y, d = hnf_params
        
        # 1. Lotpunkt berechnen (Punkt auf der Linie am nächsten zum Ursprung)
        p_lot_x = n_x * d
        p_lot_y = n_y * d
        
        # 2. Richtungsvektor der Geraden (senkrecht zum Normalenvektor)
        # Da y vorne ist: n=(nx, ny) -> v=(-ny, nx)
        v_x = -n_y
        v_y = n_x
        
        # 3. Zwei Punkte für eine 4 Meter lange Linie (2m in jede Richtung vom Lotpunkt)
        p1 = (p_lot_x + v_x * 3.0, p_lot_y + v_y * 3.0)
        p2 = (p_lot_x - v_x * 3.0, p_lot_y - v_y * 3.0)
        
        # 4. Marker-Array vorbereiten
        marker_array = MarkerArray()
        
        # 5. Linie zeichnen (Nutzt deine interne send_line Methode)
        self.send_line(marker_array, m_id=m_id, p1=p1, p2=p2, color=rgb)
        
        # 6. Text-Label am Lotpunkt (ID versetzt, damit Text und Linie koexistieren)
        self.send_text(marker_array, m_id=m_id + 1000, text=label, x=p_lot_x, y=p_lot_y, color=rgb)
        
        # 7. Veröffentlichen
        self.pub_markers.publish(marker_array)

    def test_extract_wall_lines(self, u_profile):
        '''Testet die Funktion extract_wall_lines() und visualisiert die Ergebnisse in RViz.'''
        front_straight, side_straight = self.extract_wall_lines(u_profile)
        self.get_logger().info(f"Front HNF: {front_straight}")
        self.get_logger().info(f"Side HNF: {side_straight}")
        self.visualize_hnf_line(front_straight, m_id=1, farbe_name="blau", label="")
        self.visualize_hnf_line(side_straight, m_id=0, farbe_name="blau", label="")
        return front_straight, side_straight

    def test_calculate_target_line(self, u_profile, desired_lane_ratio):
        '''Testet die Funktion calculate_target_line() und visualisiert die Ergebnisse in RViz.'''
        front_line_params, side_line_params = self.extract_wall_lines(u_profile)
        target_line_params, max_radius = self.calculate_target_line(side_line_params, front_line_params, desired_lane_ratio)
        radius = f"{max_radius:.2f}" if max_radius is not None else "None"
        #self.get_logger().info(f"Target Line HNF: {target_line_params}, Max Radius: {radius} m")
        self.visualize_hnf_line(target_line_params, m_id=3, farbe_name="gruen", label="")
        return target_line_params, max_radius

    def test_get_intersection_point(self, target_line_params):
        '''Testet die Funktion get_intersection_point() und visualisiert den Schnittpunkt in RViz.'''
        intersection_x, intersection_y, angle = self.get_intersection_point(target_line_params)
        if intersection_x is not None and intersection_y is not None:
            self.get_logger().info(f"Schnittpunkt: (X={intersection_x:.2f}, Y={intersection_y:.2f}), Schnittwinkel: {angle:.2f}°")
            marker_array = MarkerArray()
            self.send_sphere(marker_array, m_id=20, x=intersection_x, y=intersection_y, color=(1.0, 1.0, 0.0)) # Gelb
            self.pub_markers.publish(marker_array)
            return intersection_x, intersection_y, angle
        
        else:
            self.get_logger().warn("Kein gültiger Schnittpunkt gefunden.")
            return None, None, None

    def test_calculate_curve_geometry(self, intersection_y_m, turn_angle_deg, max_allowed_radius_m):
        """Testet die Funktion calculate_curve_geometry() und loggt die Ergebnisse."""
        
        curve_radius_m, entry_point_distance_m = self.calculate_curve_geometry(intersection_y_m, turn_angle_deg, max_allowed_radius_m)
        
        radius_str = f"{curve_radius_m:.2f} m" if curve_radius_m is not None else "None"
        entry_str = f"{entry_point_distance_m:.2f} m" if entry_point_distance_m is not None else "None"
        self.get_logger().info(f"Berechnete Kurvengeometrie: Radius = {radius_str}, Einlenkpunkt-Distanz = {entry_str}")
        
        marker_array = MarkerArray()
    
        if entry_point_distance_m is not None and intersection_y_m is not None:
            # Einlenkpunkt (Start)
            self.send_sphere(marker_array, m_id=21, x=0.0, y=entry_point_distance_m, color=(0.0, 1.0, 0.0))
            
            is_left = (self.fahrtrichtung == 'links')
            
            # GeoGebra Winkel-Sektor zeichnen (Radius 0.4 m)
            self.visualize_geogebra_angle(marker_array, intersection_y_m, turn_angle_deg, is_left_turn=is_left, m_id=20, radius=0.4)
            
            # Text-Label für den Winkel am Rand des Bogens platzieren
            text_x = -0.45 if is_left else 0.45
            self.send_text(marker_array, m_id=961, text=f"{turn_angle_deg:.1f}°", 
                        x=text_x, y=intersection_y_m + 0.2, color=(1.0, 1.0, 0.0))
        else:
            self.delete_marker(marker_array, m_id=900)
            self.delete_marker(marker_array, m_id=960)
            self.delete_marker(marker_array, m_id=961)
            
        if hasattr(self, 'pub_markers'):
            self.pub_markers.publish(marker_array)

        return curve_radius_m, entry_point_distance_m

    def test_check_turn_trigger(self, entry_point_distance_m):
        """Testet die Funktion check_turn_trigger() und loggt, ob der Trigger ausgelöst wird."""
        trigger = self.check_turn_trigger(entry_point_distance_m)
        status = "TRIGGERED" if trigger else "NOT TRIGGERED"
        entry_point_distance_m = f"{entry_point_distance_m:.2f} m" if entry_point_distance_m is not None else "None"
        self.get_logger().info(f"Turn Trigger Check: {status} (Entry Point Distance: {entry_point_distance_m})")
        return trigger

    def test_execute_turn(self, curve_radius_m):
        """Testet die Funktion execute_turn() und loggt das Ergebnis."""
        success = self.execute_turn(curve_radius_m)
        if success:
            self.get_logger().info(f"Turn executed with radius {curve_radius_m:.2f} m")
        else:
            self.get_logger().error("Failed to execute turn.")
        return success

    def test_check_turn_completion_fused(self, target_angle, front_line_params):
        """Testet die Funktion check_turn_completion_fused() und loggt, ob die Kurve als abgeschlossen gilt."""
        self.visualize_hnf_line(front_line_params, m_id=1, farbe_name="rot", label="front_wall")
        completed = self.check_turn_completion_fused(target_angle, front_line_params)
        status = "COMPLETED" if completed else "IN PROGRESS"
        self.get_logger().info(f"Turn Completion Check: {status} (Current Gyro: {self.current_yaw:.2f}°, Target Angle: {target_angle:.2f}°)")
        return completed

    def are_clusters_near(self, cluster_a, cluster_b, threshold=0.02):
        # Mittelpunkt A
        avg_a = np.mean(cluster_a, axis=0) # [x_avg, y_avg]
        # Mittelpunkt B
        avg_b = np.mean(cluster_b, axis=0)
        
        # Euklidische Distanz
        dist = np.linalg.norm(avg_a - avg_b)
        return dist < threshold
    
    def test_turn_main_logic(self, point_data):
        if self.camera_calibration:
            return
 
    def check_undetected_turn(self, front_wall_hnf):
        total_gedreht = abs(self.current_yaw - self.yaw_offset)
        min_total_rotation = self.target_turns * 89.0
        if self.turn_count >= self.target_turns and total_gedreht >= min_total_rotation:
            if front_wall_hnf is not None:
                closest_f = front_wall_hnf[2]
                if closest_f < 1.50:
                        self.state = 'STOPPED'
                
        if abs(self.start_straight_yaw - self.current_yaw) > 75.0:
            self.get_logger().warn(f">>> GYRO KURVE ERKANNT! Zu weit auf der geraden gedreht (Gedreht: {abs(self.start_straight_yaw - self.current_yaw):.1f}°) <<<")
            self.start_straight_yaw = self.current_yaw
            self.turn_count += 1

    def evaluate_steering_straight(self, innenbande_hnf, aussenbande_hnf):
        target_x, target_y = self.get_target_point_straight(innenbande_hnf, aussenbande_hnf)
        self.visualize_target_point(target_x, target_y, farbe_name="orange", label="Target_Point")
        # 2. PID-REGLER BERECHNEN
        # Fehler: X-Abweichung der Karotte. Negatives X = Karotte links = Positiv lenken!
        error = -target_x
        self.get_logger().info(f"SteeringError: {error:.2f}")
        
        # Integral berechnen (mit Anti-Windup, damit der Wert nicht explodiert)
        self.integral_error += error
        self.integral_error = max(-1.0, min(1.0, self.integral_error))
        
        # Derivative berechnen (Veränderung zum letzten Frame)
        derivative = error - self.prev_error
        self.prev_error = error
        
        # Stellgröße (Lenkbefehl) berechnen
        steering_cmd = (self.kp * error) + (self.ki * self.integral_error) + (self.kd * derivative)
        
        # Auf ROS-Grenzen (-1.0 bis 1.0) kappen
        steering_cmd = max(-1.0, min(1.0, steering_cmd))

        return steering_cmd
    
    def execute_init(self):
        self.get_logger().info("Starte den Roboter...")
        
        if not self.imu_ready:
            self.get_logger().info("Warte auf Gyroskop-Bootvorgang...")
            return  # Brich hier ab, mach noch nichts!

        self.set_led(True)
        self.state = 'STARTING'

    def execute_start(self, point_data):
        if self.button_start:
            if not self.button_state:
                return
            self.button_state = False
        else:
            pass

        self.button_state = False  # Flag sofort zurücksetzen
        self.set_led(False)        # LED ausschalten als Bestätigung
        
        self.yaw_offset = self.current_yaw
        self.start_straight_yaw = self.current_yaw
            
        self.get_logger().info("Starte den Roboter... Evaluiere Fahrtrichtung und Kalibriere Gyro.")
        all_clusters = self.get_all_clusters_sorted(point_data)
        validated_clusters = self.validate_clusters_straight(all_clusters)
        merged_validated_clusters, _ = self.merge_clusters(all_clusters, validated_clusters)
        self.right_wall = merged_validated_clusters[2]
        self.front_wall = merged_validated_clusters[1]
        self.left_wall  = merged_validated_clusters[0]
        right_wall_hnf = self.cluster_to_hnf(self.right_wall)
        front_wall_hnf = self.cluster_to_hnf(self.front_wall)
        left_wall_hnf = self.cluster_to_hnf(self.left_wall)

        if self.fahrtrichtung is None:
            # Wir brauchen zwingend beide Seitenwände für den Längenvergleich
            if (self.left_wall is not None and len(self.left_wall) > 0) and (self.right_wall is not None and len(self.right_wall) > 0):
                    # Echte physikalische Länge in Metern berechnen (Satz des Pythagoras)
                    left_len = math.hypot(self.left_wall[0][1] - self.left_wall[-1][1], self.left_wall[0][2] - self.left_wall[-1][2])
                    right_len = math.hypot(self.right_wall[0][1] - self.right_wall[-1][1], self.right_wall[0][2] - self.right_wall[-1][2])
                    
                    self.get_logger().info(f"Scanne Strecke... Länge Links: {left_len:.2f}m, Länge Rechts: {right_len:.2f}m")
                    
                    # Wir brauchen einen deutlichen Unterschied (z.B. 40 cm), um sicher zu sein!
                    if left_len > right_len + 0.30:
                        self.fahrtrichtung = 'rechts' # Rechte Wand ist kürzer = Innenbande = Wir fahren rechts herum!
                        self.get_logger().info(">>> LOCK: FAHRTRICHTUNG RECHTS (Uhrzeigersinn) <<<")
                    elif right_len > left_len + 0.30:
                        self.fahrtrichtung = 'links'  # Linke Wand ist kürzer = Innenbande = Wir fahren links herum!
                        self.get_logger().info(">>> LOCK: FAHRTRICHTUNG LINKS (Gegen den Uhrzeigersinn) <<<")
                    else:    
                        self.get_logger().info("Fehler! Keine Wand ist länger als die")
                        return
            else:
                self.get_logger().info("Fahrtrichtung noch nicht erkannt... Warte auf beide Seitenwände für die Analyse.")
                return
        self.state = 'FOLLOW_LANE'

    def handle_lane_following(self, point_data):
        cmd = Twist()
        all_clusters = self.get_all_clusters_sorted(point_data)

        self.get_logger().info(f"Verfolge die Spur... Aktuelle Yaw: {self.current_yaw:.1f}°, Start-Yaw: {self.start_straight_yaw:.1f}°, Gedreht seit Start: {abs(self.current_yaw - self.start_straight_yaw):.1f}°")

        validated_clusters = self.validate_clusters_straight(all_clusters)
        merged_validated_clusters, _ = self.merge_clusters(all_clusters, validated_clusters)
        self.right_wall = merged_validated_clusters[2]
        self.front_wall = merged_validated_clusters[1]
        self.left_wall  = merged_validated_clusters[0]
        right_wall_hnf = self.cluster_to_hnf(self.right_wall)
        front_wall_hnf = self.cluster_to_hnf(self.front_wall)
        left_wall_hnf = self.cluster_to_hnf(self.left_wall)

        self.check_undetected_turn(front_wall_hnf)

        self.visualize_hnf_line(front_wall_hnf, m_id=1, farbe_name="rot", label="Front HNF")
        self.visualize_hnf_line(left_wall_hnf, m_id=0, farbe_name="blau", label="Links HNF")
        self.visualize_hnf_line(right_wall_hnf, m_id=2, farbe_name="gruen", label="Rechts HNF")

        if self.fahrtrichtung == 'links':
            innenbande = self.left_wall
            innenbande_hnf = left_wall_hnf
            aussenbande = self.right_wall
            aussenbande_hnf = right_wall_hnf
        else:
            innenbande = self.right_wall
            innenbande_hnf = right_wall_hnf
            aussenbande = self.left_wall 
            aussenbande_hnf = left_wall_hnf

        if front_wall_hnf is not None and self.fahrtrichtung is not None:
            _, _, front_dist = front_wall_hnf
            max_y_innen = 0.0
            if innenbande and len(innenbande) > 0:
                max_y_innen = max(p[2] for p in innenbande) 

            if front_dist < 1.20 and max_y_innen < self.max_wall_lenght_for_turn:
                self.state = f"TURN_{self.fahrtrichtung.upper()}"
                self.get_logger().warn(f">>> {self.state} EINGELEITET<<<")
                self.get_logger().info(f"Abstand zur Frontwall: {front_dist:.2f}m")
                return
            else:
                if front_dist < 1.40:
                    self.get_logger().info(f"Warte auf Ecke... (Innenbande ragt noch {max_y_innen:.2f}m nach vorne)")
        
        steering_cmd = self.evaluate_steering_straight(innenbande_hnf, aussenbande_hnf)
        
        # 3. BEFEHLE AN ESP SETZEN
        cmd.linear.x = self.base_speed
        cmd.angular.z = float(steering_cmd)
        self.pub_cmd_vel.publish(cmd)
        self.get_logger().info(f"Lenkung: Speed={self.base_speed:.3f}, Steering={steering_cmd:.3f}")
        self.pub_cmd_vel.publish(cmd)

    def handle_turn_maneuver(self, point_data):
        """
        Kapselt die gesamte Logik für Kurven: Berechnung, Trigger, Ausführung und Abschluss.
        """
        cmd = Twist()
        
        # ---------------------------------------------------------
        # PHASE 1: ANNÄHERUNG (Berechnen & auf Trigger warten)
        # ---------------------------------------------------------
        if self.turn_phase == 'APPROACH':
            # 1. Wände tracken und Geometrie berechnen
            self.front_wall = self.track_front_wall(point_data, self.front_wall)
            front_wall_hnf = self.cluster_to_hnf(self.front_wall)
            self.visualize_hnf_line(front_wall_hnf, m_id=1, farbe_name="rot", label="Front HNF")
            
            validated_clusters = self.validate_clusters_turn(self.front_wall, point_data)
            
            # --- PID FÜR DIE ANNÄHERUNG ---
            right_wall_hnf = self.cluster_to_hnf(validated_clusters[0])
            self.right_wall = validated_clusters[0]
            left_wall_hnf = self.cluster_to_hnf(validated_clusters[2])
            self.left_wall = validated_clusters[2]

            if self.fahrtrichtung == 'links':
                innenbande_hnf = left_wall_hnf
                aussenbande_hnf = right_wall_hnf
                innenbande = self.left_wall
                aussenbande = self.right_wall
            else:
                innenbande_hnf = right_wall_hnf
                aussenbande_hnf = left_wall_hnf
                innenbande = self.right_wall
                aussenbande = self.left_wall

            # Lenkung für Approach berechnen
            self.get_logger().info(f"InnenbandeHNF: {innenbande_hnf}, AussenbandeHNF: {aussenbande_hnf}")
            steering_cmd = self.evaluate_steering_straight(innenbande_hnf, aussenbande_hnf)
            
            # Kurvengeometrie berechnen
            _, _, front_dist = front_wall_hnf
            if innenbande and len(innenbande) > 0:
                max_y_innen = max(p[2] for p in innenbande)
                calculated_lane_width = front_dist - max_y_innen
                target_line_to_wall = abs(calculated_lane_width * 0.40)
            else:
                target_line_to_wall = 0.30
            

            front_wall_params, side_wall_params = self.extract_wall_lines(validated_clusters)
            target_line_params, max_allowed_radius = self.test_calculate_target_line(validated_clusters, target_line_to_wall)
            intersection_x, intersection_y, intersection_angle = self.test_get_intersection_point(target_line_params)
            curve_radius_m, entry_distance_m = self.test_calculate_curve_geometry(intersection_y, intersection_angle, max_allowed_radius)
            
            # 2. Trigger prüfen
            if self.test_check_turn_trigger(entry_distance_m):
                self.get_logger().info(f"Trigger erreicht. Wechsle in EXECUTE-Phase. Abstand zur Wand: {front_wall_hnf[2]:.2f}m, Radius: {curve_radius_m:.2f}m")
                self.saved_intersection_angle = intersection_angle
                self.saved_curve_radius_m = curve_radius_m
                self.start_turn_yaw = self.current_yaw
                self.base_entry_distance = entry_distance_m

                cmd.linear.x = self.turn_speed
                cmd.angular.z = 0.0
                self.pub_cmd_vel.publish(cmd)
                
                self.turn_phase = 'EXECUTE'
                return
            
            # 3. Lenken in der Annäherung
            cmd.linear.x = self.turn_speed
            cmd.angular.z = float(steering_cmd)
            self.pub_cmd_vel.publish(cmd)
                
        # ---------------------------------------------------------
        # PHASE 2: AUSFÜHRUNG (Servo lenkt, Gyro prüft)
        # ---------------------------------------------------------
        elif self.turn_phase == 'EXECUTE':
            # 4. Ausführen und Tracken
            self.execute_turn(self.saved_curve_radius_m)
            self.front_wall = self.track_front_wall(point_data, self.front_wall)
            
            if self.front_wall is not None:
                front_wall_params = self.cluster_to_hnf(self.front_wall)
            else:
                front_wall_params = None

            # =========================================================
            # 5. ABSCHLUSS PRÜFEN (Mit Panic-Exit)
            # =========================================================
            
            # A) Normaler Abschluss (Gyro oder Wand-Parallelität)
            turn_completed = self.test_check_turn_completion_fused(self.saved_intersection_angle, front_wall_params)

            # C) Kurve beenden
            if turn_completed:
                self.get_logger().info("Kurve regulär beendet. Übergebe an Lane-Follower.")
                
                cmd.linear.x = self.base_speed
                cmd.angular.z = 0.0
                self.pub_cmd_vel.publish(cmd)
                    
                self.turn_phase = 'APPROACH'
                
                self.turn_count += 1

                self.state = 'FOLLOW_LANE'
                
                self.prev_error = 0.0
                self.integral_error = 0.0
                
                self.start_straight_yaw = self.current_yaw

    def update_timer(self):
        now = self.get_clock().now()

        # START-BEDINGUNG: Wenn der Roboter den Status STARTING verlässt oder Speed > 0 ist
        if self.state != 'STARTING' and not self.timer_active:
            self.start_time_stamp = now
            self.timer_active = True
            self.get_logger().info("Timer gestartet!")

        # STOP-BEDINGUNG: Wenn der Roboter fertig ist oder gestoppt wurde
        # (Passe 'FINISHED' an deinen Ziel-Zustand an)
        elif self.state == 'FINISHED' and self.timer_active:
            self.timer_active = False
            self.get_logger().info(f"Timer gestoppt! Endzeit: {self.elapsed_time:.2f}s")

        # BERECHNUNG & PUBLISH
        if self.timer_active and self.start_time_stamp is not None:
            diff = now - self.start_time_stamp
            self.elapsed_time = diff.nanoseconds / 1e9
            
            # Nachricht senden
            timer_msg = Float64()
            timer_msg.data = self.elapsed_time
            self.pub_timer.publish(timer_msg)

    def execute_stop(self):
            cmd = Twist()
            cmd.linear.x = 0.0
            cmd.angular.z = 0.0
            self.pub_cmd_vel.publish(cmd)
            self.get_logger().warn(f">>> ZIEL ERREICHT: {self.turn_count} Kurven geschafft! Stoppe den Roboter. <<<")
            self.destroy_node()
            rclpy.shutdown()
            return

    def main_logic(self, point_data):
        # ... (Grundlegende LiDAR-Datenvorbereitung, falls für alle States nötig) ...
        if self.counter == 15:
            self.counter = 0
            self.clear_all_lines()
        self.get_logger().info(f"Aktueller Status: {self.state}, TurnCount: {self.turn_count}, Lane_Ratio: {self.lane_ratio}, EXIT_Ratio: {self.lane_ratio_exit}")

        if self.state == 'INITIALIZING':
            self.execute_init()
        
        elif self.state == 'STARTING':
            self.execute_start(point_data)

        elif self.state == 'FOLLOW_LANE':
            self.handle_lane_following(point_data)
            
        elif self.state in ['TURN_LINKS', 'TURN_RECHTS']:
            self.handle_turn_maneuver(point_data)
            
        elif self.state == 'STOPPED':
            self.execute_stop()

        self.counter += 1
        self.update_timer()
        
def main(args=None):
    rclpy.init(args=args)
    node = WallFollower
    
    # Der MultiThreadedExecutor verwaltet die parallelen Callback-Gruppen
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)
    
    try:
        # Startet alle Threads parallel
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()