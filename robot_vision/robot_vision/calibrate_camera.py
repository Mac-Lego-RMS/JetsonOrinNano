import numpy as np
import json
import os

# 1. Trage hier deine frischen Messwerte ein
y_pixel = np.array([382, 309, 263, 171, 119, 88, 76, 65])
dist_m = np.array([0.2, 0.25, 0.3, 0.5, 0.8, 1.2, 1.6, 2.0])

u = 1.0 / y_pixel

# Jetzt fitten wir das Polynom auf den Kehrwert!
coeffs = np.polyfit(u, dist_m, 2)

calib_data = {
    "messwerte_pixel_y": y_pixel.tolist(),
    "messwerte_dist_m": dist_m.tolist(),
    "inverse_coeffs": coeffs.tolist()  # Neuer Name, zur Sicherheit!
}

script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_dir, 'camera_calib.json')

with open(file_path, 'w') as f:
    json.dump(calib_data, f, indent=4)

print("Perfekt! Die optisch korrekten Inverse-Koeffizienten wurden generiert.")