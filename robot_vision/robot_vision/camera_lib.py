import json
import numpy as np
import os

class TrackAnalyzer:
    def __init__(self, logger=None, visualizer_cb=None, calib_file="camera_calib.json"):
        # 1. ZUERST die Abhängigkeiten zuweisen
        self.logger = logger
        self.visualizer_cb = visualizer_cb
        self.camera_coeffs = []
        
        # ==========================================
        # DER FIX: Absoluten Pfad zur JSON-Datei erzwingen!
        # ==========================================
        script_dir = os.path.dirname(os.path.abspath(__file__))
        absolute_calib_path = os.path.join(script_dir, calib_file)
        
        # 2. DANN die Kalibrierung mit dem ECHTEN Pfad laden
        self._load_camera_calibration(absolute_calib_path)

    # Diese Methoden MÜSSEN existieren, damit self.log_warn funktioniert
    def log_info(self, msg):
        if self.logger:
            self.logger.info(msg)
        else:
            print(f"[INFO] {msg}")

    def log_warn(self, msg):
        if self.logger:
            self.logger.warn(msg)
        else:
            print(f"[WARN] {msg}")

    def _load_camera_calibration(self, filepath):
        if os.path.exists(filepath):
            try:
                with open(filepath, 'r') as f:
                    data = json.load(f)
                    # HIER IST DER FIX: Suche nach "inverse_coeffs"!
                    self.camera_coeffs = data.get("inverse_coeffs", []) 
                    self.log_info(f"Kamera-Kalibrierung (Inverses Modell) geladen! Koeffizienten: {self.camera_coeffs}")
            except Exception as e:
                self.log_warn(f"Fehler beim Laden der Kamera-Config: {e}")
        else:
            # NEU: Diese Warnung rettet dir beim nächsten Mal das Leben!
            self.log_warn(f"KRITISCH: Kamera-Config nicht gefunden unter: {filepath}")

    def get_distance_from_bbox(self, y_max):
        if not self.camera_coeffs or len(self.camera_coeffs) != 3:
            self.log_warn("Keine Kamera-Koeffizienten geladen! Gebe 0.0 zurück.")
            return 0.0
            
        # 1. Den Kehrwert des abgelesenen Pixels bilden
        u = 1.0 / float(y_max)
        
        # 2. Das Polynom mit dem Kehrwert füttern
        distance_m = np.polyval(self.camera_coeffs, u)
        
        return max(0.0, float(distance_m))