class Obstacle:
    def __init__(self, color, zone_id=None, prediction=False):
        """
        Represents a detected obstacle.
        
        :param color: "red" or "green"
        :param zone_id: Optional. Integer for the fixed position on the field. 
                        If None, it is an early sighting without a fixed position.
        """
        self.color = color.lower()
        self.zone_id = zone_id
        self.prediction = prediction
        self.pass_direction = self._determine_pass_direction()

    def _determine_pass_direction(self):
        """Translates the detected colour into a mathematical pass direction."""
        if self.color == "red":
            return 1   # pass on the right
        elif self.color == "green":
            return -1  # pass on the left
        else:
            return 0
            
    @property
    def is_localized(self):
        """Checks whether the obstacle has already been fixed on the map."""
        return self.zone_id is not None
        
    def lock_position(self, zone_id):
        """Fixes the obstacle position afterwards in the topological memory."""
        self.zone_id = zone_id

    def __repr__(self):
        loc_str = f"Zone:{self.zone_id}" if self.is_localized else "Sighting (unlocalised)"
        dir_str = "RIGHT" if self.pass_direction == 1 else "LEFT"
        return f"<Obstacle {loc_str} Colour:{self.color.upper()} Pass:{dir_str}, Prediction:{self.prediction}>"