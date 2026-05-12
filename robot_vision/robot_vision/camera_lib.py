import json
import numpy as np
import os

class TrackAnalyzer:
    def __init__(self, logger=None, visualizer_cb=None, calib_file="camera_calib.json"):
        self.logger = logger
        self.visualizer_cb = visualizer_cb
        
        # Kamera-Kalibrierung laden
        self.camera_coeffs = []
        self._load_camera_calibration(calib_file)

    def _load_camera_calibration(self, filepath):
        """Lädt die Polynom-Koeffizienten aus der JSON-Datei."""
        if os.path.exists(filepath):
            try:
                with open(filepath, 'r') as f:
                    data = json.load(f)
                    self.camera_coeffs = data.get("poly_coeffs", [])
                    self.log_info(f"Kamera-Kalibrierung geladen: {self.camera_coeffs}")
            except Exception as e:
                self.log_warn(f"Fehler beim Laden der Kamera-Config: {e}")
        else:
            self.log_warn(f"Kamera-Config nicht gefunden: {filepath}")

    def get_distance_from_bbox(self, y_max):
        """Rechnet den Pixel-Wert der Bounding Box in Meter um."""
        if not self.camera_coeffs:
            self.log_warn("Keine Kamera-Koeffizienten geladen! Gebe 0.0 zurück.")
            return 0.0
            
        # np.polyval rechnet: coeffs[0]*y^2 + coeffs[1]*y + coeffs[2]
        distance_m = np.polyval(self.camera_coeffs, y_max)
        
        # Sicherheits-Clamp: Keine Distanzen unter 0 Meter erlauben
        return max(0.0, float(distance_m))