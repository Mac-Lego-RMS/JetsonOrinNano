import json
import numpy as np
import os

class TrackAnalyzer:
    def __init__(self, logger=None, visualizer_cb=None, calib_file="camera_calib.json"):
        # 1. FIRST assign the dependencies
        self.logger = logger
        self.visualizer_cb = visualizer_cb
        self.camera_coeffs = []
        
        # ==========================================
        # THE FIX: force an absolute path to the JSON file!
        # ==========================================
        script_dir = os.path.dirname(os.path.abspath(__file__))
        absolute_calib_path = os.path.join(script_dir, calib_file)
        
        # 2. THEN load the calibration with the REAL path
        self._load_camera_calibration(absolute_calib_path)

    # These methods MUST exist so that self.log_warn works
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
                    # HERE IS THE FIX: look for "inverse_coeffs"!
                    self.camera_coeffs = data.get("inverse_coeffs", []) 
                    self.log_info(f"Camera calibration (inverse model) loaded! Coefficients: {self.camera_coeffs}")
            except Exception as e:
                self.log_warn(f"Error loading the camera config: {e}")
        else:
            # NEW: this warning will save your life next time!
            self.log_warn(f"CRITICAL: camera config not found at: {filepath}")

    def get_distance_from_bbox(self, y_max):
        if not self.camera_coeffs or len(self.camera_coeffs) != 3:
            self.log_warn("No camera coefficients loaded! Returning 0.0.")
            return 0.0
            
        # 1. Take the reciprocal of the pixel read off
        u = 1.0 / float(y_max)
        
        # 2. Feed the reciprocal into the polynomial
        distance_m = np.polyval(self.camera_coeffs, u)
        
        return max(0.0, float(distance_m))