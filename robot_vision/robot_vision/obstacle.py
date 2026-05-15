class Obstacle:
    def __init__(self, color, zone_id=None, prediction=False):
        """
        Repräsentiert ein erkanntes Hindernis.
        
        :param color: "red" oder "green"
        :param zone_id: Optional. Integer für die feste Spielfeld-Position. 
                        Wenn None, ist es eine frühe Sichtung ohne feste Position.
        """
        self.color = color.lower()
        self.zone_id = zone_id
        self.prediction = prediction
        self.pass_direction = self._determine_pass_direction()

    def _determine_pass_direction(self):
        """Übersetzt die erkannte Farbe in eine mathematische Ausweichrichtung."""
        if self.color == "red":
            return 1   # Rechts ausweichen
        elif self.color == "green":
            return -1  # Links ausweichen
        else:
            return 0
            
    @property
    def is_localized(self):
        """Prüft, ob das Hindernis bereits fest auf der Karte verortet wurde."""
        return self.zone_id is not None
        
    def lock_position(self, zone_id):
        """Fixiert die Position des Hindernisses nachträglich im topologischen Speicher."""
        self.zone_id = zone_id

    def __repr__(self):
        loc_str = f"Zone:{self.zone_id}" if self.is_localized else "Sichtung (Unverortet)"
        dir_str = "RECHTS" if self.pass_direction == 1 else "LINKS"
        return f"<Obstacle {loc_str} Farbe:{self.color.upper()} Ausweichen:{dir_str}, Prediction:{self.prediction}>"