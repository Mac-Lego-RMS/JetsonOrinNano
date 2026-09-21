import json
import os
import numpy as np

class SteeringController:
    def __init__(self, logger=None):
        self.logger = logger
        self.poly_coeffs = np.array([0.0, 1.5, 0.0, 0.0])
        self.load_calibration()

    def log_info(self, msg):
        if self.logger:
            self.logger.info(msg)
        else:
            print(f"[INFO] {msg}")

    def log_error(self, msg):
        if self.logger:
            self.logger.error(msg)
        else:
            print(f"[ERROR] {msg}")

    def load_calibration(self):
        file_path = '/workspace/src/robot_vision/robot_vision/wro_calibration.json'
        
        if os.path.exists(file_path):
            try:
                with open(file_path, 'r') as f:
                    calib_data = json.load(f)
                
                self.poly_coeffs = np.array(calib_data["poly_coeffs"])
                self.log_info("Kalibrierungsdaten aus JSON geladen.")
            except Exception as e:
                self.log_error(f"JSON fehlerhaft. Nutze Standardwerte. Fehler: {e}")
        else:
            self.log_error("Keine Kalibrierungsdatei gefunden! Nutze Standardwerte.")

    def get_steering_for_radius(self, target_radius, fahrtrichtung_ist_links):
        """
        Gibt das benötigte PWM/Servo-Signal u zurück.
        """
        if abs(target_radius) < 0.001:
            kappa_target = 0.0
        else:
            kappa_target = 1.0 / target_radius

        if not fahrtrichtung_ist_links:
            kappa_target = -kappa_target

        u_base = np.polyval(self.poly_coeffs, kappa_target)
        return float(np.clip(u_base, -1.0, 1.0))