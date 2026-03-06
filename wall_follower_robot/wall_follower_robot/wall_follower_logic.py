#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist, Point
from visualization_msgs.msg import Marker, MarkerArray
from rclpy.qos import qos_profile_sensor_data
import math

class WallFollower(Node):
    def __init__(self):
        super().__init__('wall_follower')
        
        # Subscriber für LiDAR-Daten 
        self.sub_scan = self.create_subscription(LaserScan, '/ldlidar_node/scan', self.scan_callback, qos_profile_sensor_data)
        
        # Publisher für Bewegung und RViz [cite: 1, 19]
        self.pub_cmd_vel = self.create_publisher(Twist, '/cmd_vel', 10)
        self.pub_markers = self.create_publisher(MarkerArray, '/wall_follower_markers', 10)
        
        # Konfiguration
        self.rviz_frame = 'ldlidar_link'  # Muss in RViz als "Fixed Frame" stehen
        self.get_logger().info('>>> WallFollower Template gestartet. Warte auf LiDAR... <<<')
        

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


    def get_all_clusters_sorted(self, point_data):
        """
        Sucht ALLE zusammenhängenden Cluster und durchtrennt sie an 90°-Ecken.
        Nutzt die Manhattan-Norm und X/Y-Gradienten-Überwachung.
        point_data: Liste aus (Winkel_deg, x, y, dist)
        """
        if len(point_data) < 2:
            return []

        sorted_points = sorted(point_data, key=lambda p: p[0])

        # --- PARAMETER ---
        # Max Gap etwas höher setzen, da Manhattan-Werte (dx+dy) größer sind als euklidische
        max_gap = 0.40      
        outlier_limit = 2
        # Ab welcher Streckenlänge trauen wir der Ecke? (Filtert Sensor-Rauschen)
        corner_sensitivity = 0.05 
        
        clusters = []
        current_cluster = []
        
        i = 0
        while i < len(sorted_points):
            p_curr = sorted_points[i]
            
            if not current_cluster:
                current_cluster.append(p_curr)
                i += 1
                continue
            
            p_last = current_cluster[-1]
            
            # 1. MANHATTAN-DISTANZ (|dx| + |dy|)
            dx = abs(p_curr[1] - p_last[1])
            dy = abs(p_curr[2] - p_last[2])
            dist_manhattan = dx + dy
            
            if dist_manhattan < max_gap:
                # 2. ECKEN-ERKENNUNG (Richtungswechsel X <-> Y)
                # Wir brauchen mindestens 5 Punkte im Cluster, um einen sauberen Trend zu haben
                if len(current_cluster) >= 5:
                    # Globaler Trend der Wand (vom ersten bis zum letzten Punkt)
                    p_start = current_cluster[0]
                    global_dx = abs(p_last[1] - p_start[1])
                    global_dy = abs(p_last[2] - p_start[2])
                    global_trend = 'X' if global_dx > global_dy else 'Y'
                    
                    # Lokaler Trend an der aktuellen Kurve (über die letzten 3 Punkte)
                    p_recent = current_cluster[-3]
                    local_dx = abs(p_curr[1] - p_recent[1])
                    local_dy = abs(p_curr[2] - p_recent[2])
                    local_trend = 'X' if local_dx > local_dy else 'Y'
                    
                    # Wenn die Wand abknickt (Trend ändert sich) UND es kein Rauschen ist
                    if global_trend != local_trend and max(local_dx, local_dy) > corner_sensitivity:
                        # ECKE ERKANNT! Wir zerschneiden das Cluster genau hier.
                        clusters.append(current_cluster)
                        current_cluster = [p_curr]
                        i += 1
                        continue

                # Wenn keine Ecke, Punkt normal zum Cluster hinzufügen
                current_cluster.append(p_curr)
                i += 1
            else:
                # 3. AUSREISSER-LOGIK (Jetzt auch mit Manhattan-Norm)
                found_connection = False
                for look_ahead in range(1, outlier_limit + 1):
                    if i + look_ahead < len(sorted_points):
                        p_future = sorted_points[i + look_ahead]
                        dist_future = abs(p_future[1] - p_last[1]) + abs(p_future[2] - p_last[2])
                        
                        if dist_future < max_gap:
                            i += look_ahead
                            found_connection = True
                            break
                
                if not found_connection:
                    # Echte Lücke gefunden -> Cluster abspeichern und neu beginnen
                    clusters.append(current_cluster)
                    current_cluster = [p_curr]
                    i += 1
        
        if current_cluster:
            clusters.append(current_cluster)

        # 4. WRAP-AROUND-FIX (Mit Manhattan)
        if len(clusters) > 1:
            first_p = clusters[0][0]
            last_p = clusters[-1][-1]
            dist_wrap = abs(first_p[1] - last_p[1]) + abs(first_p[2] - last_p[2])
            
            # Auch hier prüfen, ob sie über den 180°-Rand hinaus eigentlich die gleiche Wandachse sind
            if dist_wrap < max_gap:
                clusters[0] = clusters[-1] + clusters[0]
                clusters.pop()

        # 5. Sortieren nach Größe, größtes zuerst
        clusters.sort(key=len, reverse=True)
        
        return clusters

    def calculate_wall_angle(self, point_data):
        pass


    def process_my_logic(self, point_data):
        # Alle Cluster holen, sortiert von groß nach klein
        all_clusters = self.get_all_clusters_sorted(point_data)
        
        self.get_logger().info(f"Anzahl gefundener Cluster: {len(all_clusters)}")

        # Marker-Sammelkorb erstellen
        marker_array = MarkerArray()

        # Wir nehmen maximal die größten 3 Cluster (falls der Lidar z.B. nur 2 sieht, stürzt es so nicht ab)
        kandidaten = all_clusters[:3]

        # Feste Farben für Cluster 1, 2 und 3: Rot, Grün, Blau
        colors = [
            (1.0, 0.0, 0.0),  # Größtes Cluster -> Rot
            (0.0, 1.0, 0.0),  # Zweitgrößtes -> Grün
            (0.0, 0.5, 1.0)   # Drittgrößtes -> Blau
        ]

        # Die gefundenen Kandidaten durchgehen und an RViz senden
        for i, cluster in enumerate(kandidaten):
            # Ein Cluster braucht mindestens 2 Punkte für eine Linie
            if len(cluster) >= 2:
                # Start- und Endpunkt (Index 1 = X, Index 2 = Y)
                start_p = (cluster[0][1], cluster[0][2])
                ende_p = (cluster[-1][1], cluster[-1][2])
                
                # Linie zeichnen: ID entspricht dem Index (0, 1, oder 2)
                self.send_line(marker_array, m_id=i, p1=start_p, p2=ende_p, color=colors[i])

        # --- OPTIONALER CLEANUP ---
        # Falls weniger als 3 Cluster gefunden wurden, löschen wir die übrig gebliebenen Linien, 
        # damit in RViz keine "Geisterwände" aus der Vergangenheit stehen bleiben.
        from visualization_msgs.msg import Marker # Nur zur Sicherheit, falls noch nicht oben importiert
        for i in range(len(kandidaten), 3):
            delete_marker = Marker()
            delete_marker.header.frame_id = self.rviz_frame
            delete_marker.ns = "walls"
            delete_marker.id = i
            delete_marker.action = Marker.DELETE
            marker_array.markers.append(delete_marker)

        # Alles veröffentlichen, damit es in RViz2 auftaucht!
        self.pub_markers.publish(marker_array)

    # --- HILFSFUNKTION ZUM LÖSCHEN (damit keine Geisterwände stehen bleiben) ---
    def delete_marker(self, marker_array, m_id):
        marker = Marker()
        marker.header.frame_id = self.rviz_frame
        marker.ns = "walls"
        marker.id = m_id
        marker.action = Marker.DELETE
        marker_array.markers.append(marker)


    def scan_callback(self, msg):
        # Das ist dein Array an Tupeln: (Grad, Distanz, X, Y)
        point_data = []

        for i, dist in enumerate(msg.ranges):
            # Filtere ungültige Werte (inf, nan oder außerhalb der Reichweite) [cite: 6]
            if math.isinf(dist) or math.isnan(dist) or dist < 0.075 or dist > 3.0:
                continue

            # 1. Originalen Lidar-Winkel in Radiant berechnen [cite: 5]
            angle_lidar_rad = msg.angle_min + i * msg.angle_increment
            angle_lidar_deg = math.degrees(angle_lidar_rad)

            # 2. Umrechnung: Vorne = 0, Rechts = positiv (+), Links = negativ (-)
            # Da 90° Lidar = vorne ist: 90 - 90 = 0 (vorne)
            # 80° Lidar (rechts) wird zu 90 - 80 = +10°
            # 100° Lidar (links) wird zu 90 - 100 = -10°
            angle_user_deg = 90.0 - angle_lidar_deg

            # 3. Kartesische Koordinaten für die Logik (X vorne, Y links)
            # WICHTIG: Für die Berechnung von X und Y nutzen wir den originalen Radiant-Wert,
            # damit die Geometrie für RViz und Standard-ROS korrekt bleibt (X = vorne).
            x = dist * math.cos(angle_lidar_rad)
            y = dist * math.sin(angle_lidar_rad)

            # Als Tupel speichern: (Angepasster Winkel, Roh-Distanz, X, Y) [cite: 10]
            point_data.append((angle_user_deg, x, y, dist))
        
        self.process_my_logic(point_data)
        #self.get_logger().info(f"Winkel 0: {self.get_closest_measure(point_data, target_angle=0)} Winkel 105: {self.get_closest_measure(point_data, target_angle=105)} Winkel -105: {self.get_closest_measure(point_data, target_angle=-105)}")  # Beispiel: Finde den Punkt direkt vor dem Roboter (0°)

    def get_closest_measure(self, point_data, target_angle):
        """
        Sucht den Punkt, der dem target_angle am nächsten ist,
        unter Berücksichtigung des Kreisumschlags und negativer Winkel.
        """
        if not point_data:
            return None

        def get_angular_diff(a, b):
            # Berechnet die kleinste Differenz auf einem 360-Grad-Kreis
            # (a - b + 180) % 360 - 180 normiert das Ergebnis auf den Bereich [-180, 180]
            diff = (a - b + 180) % 360 - 180
            return abs(diff)

        # Wir suchen das Tupel, bei dem die Kreis-Differenz am kleinsten ist
        closest_point = min(point_data, key=lambda p: get_angular_diff(p[0], target_angle))
        
        return closest_point

def main(args=None):
    rclpy.init(args=args)
    node = WallFollower()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()