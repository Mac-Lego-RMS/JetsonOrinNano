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

        self.fahrtrichtung = None        

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

    def validate_clusters(self, clusters):
        # Wir nutzen while, da sich die Liste verkleinern kann
        while len(clusters) >= 3:
            
            # 1. Die aktuell größten 3 nehmen und von rechts nach links sortieren
            ordered = self.sort_clusters_right_to_left(clusters[:3])
            
            # 2. Winkel berechnen
            angles = [self.get_cluster_angle(c) for c in ordered]

            if any(angle is None for angle in angles):
                self.get_logger().warn("Fehler bei der Winkelberechnung. Überspringe...")
                return [None, None, None]

            # --- HILFSFUNKTION FÜR WINKEL-DIFFERENZ ---
            # Berechnet den kleinsten Schnittwinkel zwischen zwei Geraden (0° bis 90°)
            def get_angle_diff(a1, a2):
                diff = abs(a1 - a2) % 180
                if diff > 90:
                    diff = 180 - diff
                return diff

            # Differenzen berechnen (0=Rechts, 1=Front, 2=Links)
            diff_0_1 = get_angle_diff(angles[0], angles[1]) # Sollte ~90° sein (orthogonal)
            diff_1_2 = get_angle_diff(angles[1], angles[2]) # Sollte ~90° sein (orthogonal)
            diff_0_2 = get_angle_diff(angles[0], angles[2]) # Sollte ~0° sein (parallel)

            # --- ÜBERPRÜFUNG ---
            # Toleranz: Wir erlauben bis zu 20° Abweichung von der perfekten Geometrie
            
            # Check A: Sind Rechts und Front orthogonal? (Differenz sollte > 70° sein)
            if diff_0_1 < 70:
                self.get_logger().warn(f"Rechts und Front nicht orthogonal! Diff: {diff_0_1:.1f}°")
                # Finde das kleinere der beiden Cluster in 'ordered' und lösche es aus 'clusters'
                if len(ordered[0]) < len(ordered[1]):
                    clusters.remove(ordered[0])
                else:
                    clusters.remove(ordered[1])
                continue # Schleife sofort mit der bereinigten Liste neu starten

            # Check B: Sind Front und Links orthogonal?
            elif diff_1_2 < 70:
                self.get_logger().warn(f"Front und Links nicht orthogonal! Diff: {diff_1_2:.1f}°")
                if len(ordered[1]) < len(ordered[2]):
                    clusters.remove(ordered[1])
                else:
                    clusters.remove(ordered[2])
                continue

            # Check C: Sind Rechts und Links parallel? (Differenz sollte < 20° sein)
            elif diff_0_2 > 10:
                self.get_logger().warn(f"Rechts und Links nicht parallel! Diff: {diff_0_2:.1f}°")
                if len(ordered[0]) < len(ordered[2]):
                    clusters.remove(ordered[0])
                else:
                    clusters.remove(ordered[2])
                continue

            # --- ERFOLG ---
            
            # Gibt exakt zugeordnet zurück: (Rechte Bande, Frontwand, Linke Bande)
            return ordered
        # Wenn die Schleife abbricht (Liste hat weniger als 3 Cluster)
        self.get_logger().warn(f"Kein gültiges U-Profil gefunden. Nur noch {len(clusters)} Cluster übrig.")
        return [None, None, None]
                

               
            
            
        


    def sort_clusters_right_to_left(self, clusters):
        """
        Nimmt eine Liste von Clustern und sortiert sie räumlich von rechts nach links.
        Voraussetzung: Tupel-Format (angle, x, y, dist)
        """
        if not clusters:
            return []

        def get_cluster_bearing(cluster):
            # 1. Schwerpunkt des Clusters berechnen
            # Index 1 = X, Index 2 = Y
            mean_x = sum(p[1] for p in cluster) / len(cluster)
            mean_y = sum(p[2] for p in cluster) / len(cluster)
            
            return math.atan2(mean_y, mean_x)

        sorted_clusters = sorted(clusters, key=get_cluster_bearing, reverse=True)
        
        return sorted_clusters

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
                # 2. NEUE ECKEN-ERKENNUNG (Über echte Vektor-Winkel)
                if len(current_cluster) >= 5:
                    p_start = current_cluster[-5] # Punkt vor der Kurve
                    p_mid = current_cluster[-3]   # Punkt am Scheitel
                    
                    # Vektor 1 (Trend vor der Kurve)
                    dx1 = p_mid[1] - p_start[1]
                    dy1 = p_mid[2] - p_start[2]
                    angle1 = math.atan2(dy1, dx1)
                    
                    # Vektor 2 (Aktuelle Bewegung)
                    dx2 = p_curr[1] - p_mid[1]
                    dy2 = p_curr[2] - p_mid[2]
                    angle2 = math.atan2(dy2, dx2)
                    
                    # Wie stark knickt die Wand physikalisch ab?
                    diff_rad = abs(angle1 - angle2)
                    diff_deg = math.degrees(diff_rad)
                    if diff_deg > 180:
                        diff_deg = 360 - diff_deg
                        
                    # Filter: Hat sich der Punkt auch ausreichend bewegt? (Rauschen ignorieren)
                    dist_moved = math.hypot(dx2, dy2)
                    
                    # Eine echte Parcours-Ecke knickt stark ab (z.B. > 45 Grad)
                    if diff_deg > 45.0 and dist_moved > corner_sensitivity:
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

    
    def get_cluster_angle(self, cluster):
        """
        Berechnet den Durchschnittswinkel aller Punkte im Cluster (Ausgleichsgerade).
        Vorne/Hinten (Seitenbanden) = 0°
        Links/Rechts (Frontwand) = 90° oder -90°
        """

        if cluster is None:
            return None
        
        n = len(cluster)
        if n < 2:
            return None

        

        # 1. Schwerpunkt berechnen
        mean_x = sum(p[1] for p in cluster) / n
        mean_y = sum(p[2] for p in cluster) / n

        # 2. Abweichungen vom Schwerpunkt
        s_xx = 0.0
        s_yy = 0.0
        s_xy = 0.0
        
        for p in cluster:
            dx = p[1] - mean_x
            dy = p[2] - mean_y
            s_xx += dx * dx
            s_yy += dy * dy
            s_xy += dx * dy

        # 3. FIX: Wir tauschen s_xx und s_yy im Nenner! 
        # Dadurch berechnen wir den Winkel relativ zur Y-Achse (Fahrtrichtung) statt zur X-Achse.
        angle_rad = 0.5 * math.atan2(2.0 * s_xy, s_yy - s_xx)
        
        # In Grad umrechnen
        angle_deg = math.degrees(angle_rad)

        return angle_deg



    def process_my_logic(self, point_data):
        # 1. IMMER GANZ OBEN: Den Sammelkorb erstellen, damit er überall in der Funktion existiert!
        marker_array = MarkerArray()

        # 2. Punktwolke in Cluster (Wände) aufteilen
        all_clusters = self.get_all_clusters_sorted(point_data)


        validated_clusters = self.validate_clusters(all_clusters)
        # --- AB HIER IST ALLES SICHER ---
        # 5. Wir haben ein perfektes U-Profil!
        self.get_logger().info(f"Anzahl cluster {len(all_clusters)}")
        kandidaten = self.merge_clusters(all_clusters, validated_clusters)  # Die drei größten Cluster, die wir validiert haben
        #kandidaten = self.validate_clusters(all_clusters)

        # Feste Farben: Rechts=Rot, Front=Grün, Links=Blau
        colors = [
            (1.0, 0.0, 0.0),    # Rot (Original)
            (0.0, 1.0, 0.0),    # Grün (Original)
            (0.0, 0.5, 1.0),    # Azurblau (Original)
            (1.0, 0.5, 0.0),    # Orange
            (0.5, 0.0, 1.0),    # Violett
            (0.0, 1.0, 1.0),    # Cyan
            (1.0, 0.0, 1.0),    # Magenta
            (1.0, 1.0, 0.0),    # Gelb
            (0.5, 0.5, 0.5),    # Grau
            (0.6, 0.3, 0.0)     # Braun
        ]

        # 6. Die drei Wände durchgehen und an RViz senden
        for i, cluster in enumerate(kandidaten):
            # Info in der Konsole ausgeben
            angle = self.get_cluster_angle(cluster)

            if angle is None:
                self.get_logger().warn(f"Wand {i+1}: Winkel konnte nicht berechnet werden.")
                continue # Überspringt diesen Cluster und macht mit dem nächsten weiter
            self.get_logger().info(f"Wand {i+1} (ID {i}): {len(cluster)} Punkte, Winkel: {angle:.2f}°")
            self.get_logger().info(f"Erster Punkt: (X={cluster[0][1]:.2f}, Y={cluster[0][2]:.2f}), Letzter Punkt: (X={cluster[-1][1]:.2f}, Y={cluster[-1][2]:.2f})")

            # Ein Cluster braucht mindestens 2 Punkte für eine Linie
            if len(cluster) >= 2:
                # Start- und Endpunkt (Index 1 = X, Index 2 = Y)
                start_p = (cluster[0][1], cluster[0][2])
                ende_p = (cluster[-1][1], cluster[-1][2])
                
                # Linie zeichnen: ID entspricht dem Index (0, 1, oder 2)
                self.send_line(marker_array, m_id=i, p1=start_p, p2=ende_p, color=colors[i])

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
    
    def get_closest_point_in_cluster(self, cluster):
        """
        Gibt den Punkt eines Clusters zurück, der den kürzesten Abstand zum LiDAR hat.
        cluster: Liste von Punkten im Format (angle, x, y, dist)
        """
        if not cluster:
            return None

        # Sucht das Element im Cluster, bei dem der Wert an Index 3 (dist) am kleinsten ist
        closest_point = min(cluster, key=lambda p: p[3])
        
        return closest_point

    def merge_clusters(self, all_clusters, validated_clusters):
        """
        Versucht, benachbarte Cluster zu einem einzigen Cluster zu verschmelzen.
        Nutzt den Normalenvektor, um nur den senkrechten Abstand (Offset) zu prüfen.
        """
        max_distance_gap = 0.30  # 5 cm maximaler seitlicher Versatz
        max_angle_gap = 5.0      # 5 Grad maximale Winkelabweichung

        remaining_clusters = [c for c in all_clusters if c not in validated_clusters]
        if not remaining_clusters:
            return validated_clusters

        def get_angle_diff(a1, a2):
            diff = abs(a1 - a2) % 180
            if diff > 90:
                diff = 180 - diff
            return diff

        for valid_cluster in validated_clusters:
            angle = self.get_cluster_angle(valid_cluster)
            if valid_cluster is None:
                continue # Überspringe diese leere Wand sofort!

            if angle is None: 
                continue

            bx = sum(p[1] for p in valid_cluster) / len(valid_cluster)
            by = sum(p[2] for p in valid_cluster) / len(valid_cluster)

            angle_rad = math.radians(angle)
            # Normalenvektor für den Abstand
            nx = math.cos(angle_rad)
            ny = math.sin(angle_rad)

            clusters_to_remove = []

            for other in remaining_clusters:
                other_angle = self.get_cluster_angle(other)
                if other_angle is None: 
                    continue

                if get_angle_diff(angle, other_angle) < max_angle_gap:
                    ox = sum(p[1] for p in other) / len(other)
                    oy = sum(p[2] for p in other) / len(other)

                    offset = abs((ox - bx) * nx + (oy - by) * ny)

                    if offset < max_distance_gap:
                        valid_cluster.extend(other)
                        clusters_to_remove.append(other)

            for c in clusters_to_remove:
                remaining_clusters.remove(c)

            # --- DER MAGISCHE FIX ---
            # Wir berechnen den Richtungsvektor der Wand (parallel zur Wand)
            dir_x = -math.sin(angle_rad)
            dir_y = math.cos(angle_rad)
            
            # Wir sortieren die Punkte nach ihrer geometrischen Position ENTLANG der Wand!
            # Dadurch liegen der physikalisch erste und letzte Punkt immer an Index 0 und -1,
            # völlig unabhängig vom LiDAR-Wrap-Around.
            valid_cluster.sort(key=lambda p: p[1] * dir_x + p[2] * dir_y)

        return validated_clusters

            

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