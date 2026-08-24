# camera_lidar_fusion

Verheiratet die liegend montierte 360-Grad-Fisheye-Kamera (USB, `/video_source/raw`,
1280x960) mit dem 2D-Lidar. Zwei Nodes:

| Node | Zweck |
| --- | --- |
| `lidar_pixel_mapper` | Farbe je Lidar-Punkt -> CSV, PointCloud2, Debug-Bild |
| `rotation_calibration` | Bildkreis vermessen und Verdrehung der Kamera bestimmen |

## Geometrie

Roboter-/Lidar-Frame nach REP-103: **X vorne, Y links, Z oben**.

Die Kamera liegt auf dem Ruecken, die optische Achse zeigt also nach oben. Bei
270 Grad Oeffnungswinkel reicht das Sichtfeld 45 Grad unter den Horizont -- damit
sieht sie rundum und leicht nach unten. Genau das brauchen wir, um Hindernisse
schon beim Kurveneingang zu erfassen.

Fuer den Nominalfall (Achse exakt senkrecht) faellt der optische Frame mit dem
Roboter-Frame zusammen, die Rotationsmatrix ist die Einheitsmatrix. Die
Kalibrierwinkel beschreiben nur die Abweichung:

* `yaw_deg` -- Drehung um die optische Achse. **Das ist die Verdrehung, um die es
  geht.** Sie haengt nur vom Azimut ab, nicht von der Hoehe des Zielobjekts,
  und ist deshalb geschlossen loesbar.
* `pitch_deg` / `roll_deg` -- Verkippung der Achse aus der Senkrechten.

Projektion: `P_cam = R @ (P_robot - t)`, dann equidistantes Fisheye
`r = f * theta` mit `f = radius_px / (fov/2)`. Bei Bedarf `poly_coeffs` setzen.

Wichtig: die Lidar-Ebene liegt **unter** der Kamera, also ist `theta > 90 Grad`
fuer alle Bodenpunkte. Das ist korrekt und liegt innerhalb der 135 Grad.

### Wo im Bild abgegriffen wird: `sample_mode`

**`horizon` (Default).** Der Lidar-Punkt wird auf Objektivhoehe abgegriffen. Die
Hoehendifferenz zur Kamera ist dann null, `theta` exakt 90 Grad und der
Bildradius konstant `f*pi/2` = 301.3 px -- unabhaengig von der Entfernung. Es
bleibt nur der Azimut, also **eine feste Kreislinie im Bild**.

Das reicht fuer Pylonen, solange das Objektiv zwischen Matte und
Pylonenoberkante sitzt: eine Pylone, die die waagerechte Ebene durch die Linse
durchstoesst, liegt in *jeder* Entfernung auf diesem Ring. Gemessen an echten
Lidar-Daten (2203 Punkte, 0.05 bis 2.96 m): Radiusspanne 0.013 px.

Der Gewinn ist nicht nur Einfachheit -- radial fallen zwei Fehlerquellen
komplett weg: die Entfernungsmessung des Lidars und ein falsches `cam_z`. Uebrig
bleibt allein `yaw`. Genau das, was man fuer weit entfernte Hindernisse braucht.

Wieviel Luft der Ring in der Pylone hat (10-cm-Pylone, Abstand Ring zu Ober- und
Unterkante in px):

| Linsenhoehe | 0.3 m | 1.0 m | 2.0 m |
| --- | --- | --- | --- |
| 2 cm | 50.0 / 12.8 | 15.3 / 3.8 | 7.7 / 1.9 |
| **5 cm** | **31.7 / 31.7** | **9.6 / 9.6** | **4.8 / 4.8** |
| 8 cm | 12.8 / 50.0 | 3.8 / 15.3 | 1.9 / 7.7 |
| 11 cm | daneben | daneben | daneben |

Auf halber Pylonenhoehe ist der Abstand zu beiden Kanten am groessten -- **dort
sollte das Objektiv sitzen**. Ueber der Pylonenoberkante greift der Ring an der
Pylone vorbei und liest die Wand dahinter; dann `height` nehmen.

**`height`.** Abgriff auf fester Hoehe `sample_height_m` ueber der Lidar-Ebene,
Bildradius haengt an der Entfernung. Nur noetig, wenn die Linse nicht zwischen
Matte und Pylonenoberkante sitzt.

### Wenn der Ring zu hoch sitzt: die Brennweite kalibrieren

Der Ring liegt bei `r = f*pi/2`, und `f = radius_px / (fov/2)`. Die FOV war bis
hierher eine **Annahme** (270 Grad aus der Produktbeschreibung), nie gemessen.
Stimmt sie nicht, sitzt der Ring am falschen Radius -- und weil eine zu gross
angenommene FOV `f` zu klein macht, sitzt er dann zu weit **innen**, also zu hoch
im Raum, und schaut ueber die Pylonen hinweg.

| angenommene FOV | Ring bei |
| --- | --- |
| 270 Grad | 301 px |
| 240 Grad | 339 px |
| 220 Grad | 370 px |
| 200 Grad | 407 px |
| 180 Grad | 452 px (= Rand des Bildkreises) |

**Schnellweg -- Ring von Hand setzen.** Der Ring ist im Debug-Bild orange
eingezeichnet. Verschieben, bis er auf Pylonenhoehe liegt:

```bash
ros2 topic pub --once /camera_lidar/calib_cmd std_msgs/msg/String "data: ring"
ros2 param set /camera_rotation_calibration horizon_radius_px 370
ros2 topic pub --once /camera_lidar/calib_cmd std_msgs/msg/String "data: save"
```

**Sauberer Weg -- `radial`.** Eine Pylone bei mehreren Entfernungen samplen
(wichtig: **weit gespreizt**, z.B. 0.2 bis 1.5 m) und dann:

```bash
ros2 param set /camera_rotation_calibration pylon_height_m 0.10
ros2 topic pub --once /camera_lidar/calib_cmd std_msgs/msg/String "data: radial"
```

Der Fusspunkt der Pylone steht auf der Matte, also immer `L` unter dem Objektiv:
`theta_fuss = pi - atan(L/rho)`. Wandert die Pylone von 0.2 auf 2 m, laeuft
dieser Winkel von rund 158 auf 92 Grad -- diese Spreizung macht `f` und `L`
gemeinsam bestimmbar. Die Oberkante liefert dieselbe Gleichung mit `L - H`.

`radial` gibt dir damit auch **die Objektivhoehe ueber der Matte** -- und sagt
direkt, ob der Horizontring ueberhaupt funktionieren kann oder ob die Kamera
tiefer muss.

### Ring nach unten kippen: `sample_depression_deg`

Kippt den Ring um X Grad nach unten; aus der waagerechten Ebene wird ein Kegel.
Der Bildradius bleibt konstant (`f*(pi/2 + X)`), aber die **Abgriffstiefe unter
der Linse waechst mit der Entfernung**: `rho * tan(X)`.

| `sample_depression_deg` | 0.3 m | 1.0 m | 2.0 m |
| --- | --- | --- | --- |
| 0.5 Grad | 0.3 cm | 0.9 cm | 1.7 cm |
| 1.0 Grad | 0.5 cm | 1.7 cm | 3.5 cm |
| 3.0 Grad | 1.6 cm | 5.2 cm | 10.5 cm (unter der Matte) |

Fuer 10-cm-Pylonen sind also nur Bruchteile eines Grades brauchbar. Und der
harte Fall: sitzt die Linse **ueber** der Pylonenoberkante, gibt es gar keinen
Winkel, der nah und fern gleichzeitig trifft -- bei 13 cm Linsenhoehe braucht
0.3 m zwischen 5.7 und 23.4 Grad, 2.0 m aber zwischen 0.9 und 3.7 Grad. Die
Fenster ueberlappen nicht. Dann hilft nur `height`.

Deshalb: Linse in die Pylonenhoehe bringen ist die Loesung, nicht der Kippwinkel.

### Mitteln statt ein Pixel: `sample_band_m`

Statt eines einzelnen Pixels werden `sample_band_count` Stuetzstellen entlang
der **radialen** Linie durch den Punkt gelesen -- die liegt im Fisheye laengs
der Pylone -- und davon der **Median** genommen (nicht der Mittelwert: der
Median haelt stand, wenn ein Ende des Bandes ueber die Pylonenkante rutscht).

Die Bandbreite wird in Metern Pylonenhoehe angegeben und je Punkt aus der
Entfernung in Pixel umgerechnet (`f*atan(band_m/rho)`). Fern schrumpft das Band
also von selbst mit und bleibt automatisch innerhalb der Pylone.

```bash
ros2 param set /lidar_pixel_mapper sample_band_m 0.03   # Default: +-3 cm
ros2 param set /lidar_pixel_mapper sample_band_m 0.0    # aus, ein Pixel
ros2 param set /lidar_pixel_mapper sample_band_count 5
```

**Zur Rechenzeit:** `patch_px` filtert per `medianBlur` das *ganze* Bild und
kostet auf 1280x960 rund 26 ms je Scan -- bei 15 Hz gut 40 Prozent eines Kerns.
Solange das Band aktiv ist, ist das ueberfluessig, deshalb steht `patch_px` auf
1. Nur hochsetzen, wenn du `sample_band_m` auf 0 stellst.

| Schritt | Zeit je Scan (2200 Punkte) |
| --- | --- |
| `medianBlur` 1280x960 | 26.4 ms |
| Bandabtastung, 5 Stuetzstellen | 1.8 ms |
| Klassifikation | 0.7 ms |
| Projektion | 0.4 ms |

```bash
ros2 param set /lidar_pixel_mapper sample_mode horizon
ros2 param set /lidar_pixel_mapper sample_mode height
ros2 param set /lidar_pixel_mapper sample_height_m 0.0   # nur bei height
```

Beide Parameter werden bei jedem Scan neu gelesen, wirken also sofort.

### Die z-Hoehe

`cam_z` (Kamera ueber der Lidar-Ebene) steckt in der Translation und ist damit
voll eingerechnet -- sie bestimmt `theta` und damit den **Bildradius**. Yaw
dagegen bestimmt nur den **Winkel**. Die beiden stehen senkrecht aufeinander und
stoeren sich nicht.

Wie stark z wirkt, haengt an der Entfernung:

| Fehler | 0.2 m | 0.5 m | 1.0 m | 2.0 m |
| --- | --- | --- | --- | --- |
| `cam_z` 1 cm daneben | 8.9 px | 3.8 px | 1.9 px | 1.0 px |
| `cam_z` 2 cm daneben | 17.6 px | 7.6 px | 3.8 px | 1.9 px |
| `yaw` 1 Grad daneben | 6.1 px | 5.6 px | 5.4 px | 5.3 px |

Also: nah zaehlt z, fern verschwindet es -- fuer Hindernisse beim Kurveneingang
(also weit weg) ist yaw das, worauf es ankommt. Ein Zentimeter Messfehler beim
Lineal kostet dich auf 1 m keine 2 Pixel.

Das gilt fuer `sample_mode: height`. Bei `horizon` faellt der Einfluss von
`cam_z` auf den Abgriff komplett weg -- dort wird `cam_z` nur noch gebraucht,
um die Objektivhoehe zu treffen, und der Ring bleibt derselbe.

Bestimmbar ist aus Bildern immer nur der **Hoehenunterschied** zwischen Kamera
und Zielmarke, nie beides getrennt. `height` loest deshalb `cam_z` unter der
Annahme, dass `target_height_m` (Hoehe des Farb-Blob-Schwerpunkts ueber der
Lidar-Ebene) stimmt. Nachmessen mit dem Lineal ist genauer; `height` ist der
Gegencheck.

## Warum ein Referenzscan noetig ist

Am S3 verdecken Kabel und Elektronik einen Teil des Sichtfelds. Dort misst der
Scanner sich selbst -- ein paar Zentimeter -- und das sind damit IMMER die
naechsten Punkte. Eine Suche nach dem naechstgelegenen Objekt findet so nie die
Pylone.

Blindsektoren (``blind``) erwischen den Kern dieser Bereiche, aber nicht den
Rand: dort streift der Strahl den Aufbau und liefert z.B. 0.23 m, also ueber der
Schwelle. Deshalb ist der Referenzscan (``background``) das eigentliche
Werkzeug -- er nimmt die leere Umgebung einmal auf, und danach gilt als Ziel nur
noch, was NAEHER misst als diese Referenz. Kabel, Elektronik, Tischkanten und
Waende fallen damit alle von selbst weg.

## Reihenfolge beim Einrichten

`cam_z` (Hoehe der Kamera ueber der Lidar-Ebene) einmal nachmessen und in die
Kalibrierdatei eintragen -- das ist der einzige Wert, den keine Node erraten kann.

```bash
ros2 run camera_lidar_fusion rotation_calibration

# 0a. Verbaute Lidar-Sektoren ausmessen (zweimal senden: sammeln, auswerten)
ros2 topic pub --once /camera_lidar/calib_cmd std_msgs/msg/String "data: blind"
ros2 topic pub --once /camera_lidar/calib_cmd std_msgs/msg/String "data: blind"

# 0b. Referenzscan der LEEREN Umgebung -- Pylone wegnehmen! (ebenfalls zweimal)
ros2 topic pub --once /camera_lidar/calib_cmd std_msgs/msg/String "data: background"
ros2 topic pub --once /camera_lidar/calib_cmd std_msgs/msg/String "data: background"

# 0c. Farbe der Kalibrierpylone festnageln -- sonst gewinnt der groesste
#     Farbfleck im Raum statt der Pylone.
ros2 param set /camera_rotation_calibration target_label gruen

# 1. Bildkreis automatisch vermessen
ros2 topic pub --once /camera_lidar/calib_cmd std_msgs/msg/String "data: circle"

# 2. Verdrehung messen: EINEN roten/gruenen Klotz hinstellen, sonst nichts im
#    Nahbereich. Pro Position samplen, Klotz rundum versetzen (>= 3 Positionen).
ros2 topic pub --once /camera_lidar/calib_cmd std_msgs/msg/String "data: sample"

# 3. Loesen, pruefen, speichern
ros2 topic pub --once /camera_lidar/calib_cmd std_msgs/msg/String "data: solve"
ros2 topic pub --once /camera_lidar/calib_cmd std_msgs/msg/String "data: verify"
ros2 topic pub --once /camera_lidar/calib_cmd std_msgs/msg/String "data: save"

# 4. Optional: Kamerahoehe gegenpruefen (braucht nahe Samples, < 0.5 m)
ros2 param set /camera_rotation_calibration target_height_m 0.05
ros2 topic pub --once /camera_lidar/calib_cmd std_msgs/msg/String "data: height"
```

Kontrolle in Foxglove: `/camera_lidar/calib_debug`. Das orange X (projiziertes
Lidar-Cluster) muss auf dem Farbring (Kamera-Blob) liegen. Weitere Kommandos:
`list`, `clear`, `reload`, `auto` (sammelt selbststaendig, sobald der Klotz weit
genug versetzt wurde).

Alles laesst sich auch live von Hand nachziehen -- das Debug-Bild folgt sofort:

```bash
ros2 param set /camera_rotation_calibration yaw_deg 12.5
ros2 topic pub --once /camera_lidar/calib_cmd std_msgs/msg/String "data: save"
```

## Farbe je Lidar-Punkt

```bash
ros2 run camera_lidar_fusion lidar_pixel_mapper
ros2 topic pub --once /camera_lidar/capture std_msgs/msg/Empty "{}"
```

Schreibt `/workspace/lidar_color_logs/lidar_pixels_<zeit>.csv` plus die
verwendete Kalibrierung als `_calib.yaml` daneben. Spalten:

```
stamp_sec, idx, angle_deg, range_m, x_m, y_m, z_m,
u_px, v_px, theta_deg, phi_deg, b, g, r, h, s, v, label
```

`label` ist `rot`, `gruen`, `magenta`, `schwarz` oder `unbekannt`.
`csv_mode:=continuous` haengt stattdessen jeden Scan an eine Datei an,
`csv_mode:=off` schaltet die CSV ganz ab.

## Debug-Ansicht in Foxglove

Der Parameter `debug` (Default `true`) ist der Hauptschalter fuer die Anzeige:

```bash
ros2 param set /lidar_pixel_mapper debug true    # an
ros2 param set /lidar_pixel_mapper debug false   # aus, spart CPU im Lauf
```

Ist er an, gehen zwei Topics raus:

* **`/camera_lidar/colored_scan`** -- `PointCloud2` mit RGB: jeder Lidar-Punkt an
  seiner echten x/y-Position, eingefaerbt mit dem Pixel, auf den er projiziert.
  Das ist die Ansicht, an der man in einem Blick sieht, ob die Kalibrierung
  sitzt: stehen die roten Punkte auf dem roten Klotz, stimmt yaw.
  In Foxglove ein 3D-Panel oeffnen, Topic abonnieren, Color-Mode auf `RGB`
  stellen. Der Frame ist der des Lidars (bei `sllidar` = `laser`).
* **`/camera_lidar/debug_image`** -- dasselbe andersherum: das Fisheye-Bild mit
  den eingezeichneten Projektionen, dem Bildkreis und einem Pfeil nach vorne.

Feiner steuerbar mit `publish_cloud`, `publish_debug_image` und `debug_rate_hz`
(Default 5 Hz fuer das Bild; die PointCloud geht mit jedem Scan raus).
`/camera_lidar/summary` zaehlt nur die Labels und laeuft immer.

Die Kalibrier-Node hat denselben Schalter fuer `/camera_lidar/calib_debug`.

## Launch

```bash
ros2 launch camera_lidar_fusion camera_lidar.launch.py
ros2 launch camera_lidar_fusion camera_lidar.launch.py mode:=calib
ros2 launch camera_lidar_fusion camera_lidar.launch.py scan_topic:=/ldlidar_node/scan
```

`scan_topic` steht auf `/scan` (was `sllidar_s3_launch.py` publiziert). Der
aeltere Code in `robot_vision` haengt teils noch auf `/ldlidar_node/scan` --
im Zweifel `ros2 topic list` fragen.

## Kalibrierdatei

Gelesen und geschrieben wird `/workspace/config/fisheye_calib.yaml`
(Parameter `calib_file`). Existiert sie nicht, greift die mitgelieferte Vorgabe
aus `share/camera_lidar_fusion/config/fisheye_calib.yaml`.

## Tests

Das Projektionsmodell laeuft ohne Hardware:

```bash
cd /workspace/src/camera_lidar_fusion && python3 -m pytest test/test_fisheye_model.py -q
```

`test/fake_scan.py` publiziert einen synthetischen 360-Grad-Scan auf `/scan`,
damit sich `lidar_pixel_mapper` auch ohne laufendes Lidar durchtesten laesst.
