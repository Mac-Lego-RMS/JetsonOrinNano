import numpy as np

class Obstacle:
    def __init__(self, cluster, color):
        self.cluster = cluster
        self.color = color  # "red" oder "green"
        
        # Metrischer Schwerpunkt im lokalen Robotersystem (x=Querachse, y=Fahrtrichtung)
        self.center_x_m = 0.0
        self.center_y_m = 0.0
        
        self.is_valid = False
        self.pass_direction = 0  # 1 = rechts vorbei, -1 = links vorbei
        
        self._process_cluster()
        self._determine_pass_direction()

    def _process_cluster(self):
        """Berechnet den Massepunkt (Schwerpunkt) des Clusters. Muss bei jeder cluster - Aktualisierung mit aktualisiert werden"""
        if not self.cluster or len(self.cluster) < 3:
            self.is_valid = False
            return
            
        # Annahme: cluster ist eine Liste von (x, y) Tupeln
        points = np.array(self.cluster)
        
        # Schwerpunktberechnung (Mittelwert der x- und y-Koordinaten)
        self.center_x_m, self.center_y_m = np.mean(points, axis=0)
        self.is_valid = True

    def _determine_pass_direction(self):
        """Übersetzt die erkannte Farbe in eine mathematische Ausweichrichtung."""
        if self.color == "red":
            self.pass_direction = 1   # Rechts ausweichen
        elif self.color == "green":
            self.pass_direction = -1  # Links ausweichen
        else:
            self.pass_direction = 0