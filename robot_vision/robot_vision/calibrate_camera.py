import numpy as np
import json
import os

# 1. Trage hier deine frischen Messwerte ein
y_pixel = [450, 400, 360, 310, 270, 240]
dist_m = [0.3, 0.5, 0.8, 1.2, 1.6, 2.0]

# 2. Polynom 2. Grades berechnen
# Liefert ein Array [a, b, c] für a*x^2 + b*x + c
coeffs = np.polyfit(y_pixel, dist_m, 2)

# 3. JSON-Datenstruktur aufbauen
calib_data = {
    "messwerte_pixel_y": y_pixel,
    "messwerte_dist_m": dist_m,
    "poly_coeffs": coeffs.tolist()  # numpy arrays müssen für JSON in Listen gewandelt werden
}

script_dir = os.path.dirname(os.path.abspath(__file__))

# Klebe den Dateinamen an diesen Pfad dran
file_path = os.path.join(script_dir, 'camera_calib.json')

# 4. In Datei speichern
with open(file_path, 'w') as f:
    json.dump(calib_data, f, indent=4)

print("Erfolgreich! camera_calib.json wurde generiert.")
print(f"Koeffizienten: {calib_data['poly_coeffs']}")