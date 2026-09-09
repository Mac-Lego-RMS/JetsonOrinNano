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

### Statt einer Linie eine Zone

Ein einzelner Abgriffsradius ist zerbrechlich: er trifft je nach Entfernung und
Kalibrierfehler mal die Pylone, mal die Wand dahinter, mal den Boden davor. Mit
gesetzter Zone wird stattdessen ein Stueck der radialen Linie abgetastet und
ausgezaehlt, welcher Anteil der Pixel zu welcher Farbe passt; ab
`sample_zone_min_frac` gewinnt eine Farbe.

**Abstimmen, nicht mitteln.** Ein Median ueber ein Segment, das halb auf der
Pylone und halb auf der Wand liegt, ergibt Mischmasch. Der Stimmenanteil bleibt
aussagekraeftig, solange die Pylone einen nennenswerten Teil des Segments
fuellt. Gegenprobe am Aufbau: nimmt man statt der Abstimmung einfach das
gesaettigtste Pixel, findet man in fast jeder Linie irgendwas und erzeugt
Cluster von 30 Grad Breite, wo eine Pylone 5 Grad haette.

**Die Zone ist nicht konstant dick.** Eine Bande fester Hoehe erscheint im
Fisheye kein Kreisband gleicher Dicke -- und das ist der Kern der Sache. Sitzt
das Objektiv auf Hoehe der Bandenoberkante, ist fuer diese Kante die
Hoehendifferenz null, `theta` damit exakt 90 Grad und der Bildradius konstant:
die Oberkante laeuft als gerade Linie. Die Unterkante liegt die Bandenhoehe
tiefer, ihr `theta` naehert sich mit wachsender Entfernung von oben an 90 Grad
an, ihr Radius also von aussen an den der Oberkante:

| Entfernung | Zone | Dicke |
| --- | --- | --- |
| 0.3 m | 392 .. 445 px | 53 px |
| 1.0 m | 398 .. 413 px | 15 px |
| 3.0 m | 399 .. 405 px | 6 px |

Eine in Pixeln festgelegte Breite waere deshalb nah viel zu schmal und fern zu
breit -- fern ragt sie ueber die Bande hinaus und sammelt die helle Matte oder
die Wand mit ein, wodurch Punkte faelschlich als `unbekannt` statt `schwarz`
herauskommen.

Woher die Zonengrenzen kommen, entscheidet sich in dieser Reihenfolge:

1. **`zone_from_band: true`** -- Unterkante live aus dem Bild (siehe unten),
   Oberkante aus der Kalibrierung. Das Beste, was es gibt.
2. **Kalibrierte Kurve** in der Kalibrierdatei (`zone` / `zonefit`).
3. **`sample_zone_low_m` / `sample_zone_high_m`** -- aus zwei Hoehen gerechnet.
   Nur als Notloesung, siehe Modellfehler weiter unten.

```bash
ros2 param set /lidar_pixel_mapper sample_zone_min_frac 0.5   # Stimmenanteil
ros2 param set /lidar_pixel_mapper sample_zone_steps 13       # Stuetzstellen
ros2 param set /lidar_pixel_mapper sample_zone_nutz 1.0       # 0.4 = mittleres Drittel
```

`sample_zone_nutz` tastet nur den mittleren Teil der Zone ab. Sitzen die Grenzen
sauber, ist die Mitte die reinste Stelle -- die Raender tragen Mischpixel bei.
Ganz auf eine Linie zu gehen ist aber riskant: bei 3 m ist die ganze Zone nur
6 px dick, davon 40 Prozent sind zwei Pixel, und dann haengt wieder alles daran,
dass die Zone aufs Pixel genau sitzt.

Der Stimmenanteil muss zur Zonenbreite passen. Am Aufbau gemessen, mit zwei
gruenen und zwei roten Pylonen im Feld:

| `sample_zone_min_frac` | gruene Cluster | rote Cluster |
| --- | --- | --- |
| 0.20 | 7 | 3 |
| 0.40 | 4 | 2 |
| **0.50** | **2** | **2** |

Bei 0.50 blieben genau die vier echten Pylonen uebrig, ohne Fehltreffer.

### Die Zone messen statt rechnen: `zone` und `zonefit`

Die Zone aus `cam_z` und der Brennweite zu RECHNEN funktioniert nicht gut
genug. Der Grund ist ein Modellfehler, der zum Bildrand hin waechst -- genau
dort, wo wir arbeiten: `poly_coeffs` ist leer, das Modell rechnet also streng
equidistant mit `r = f*theta`, und echte Fisheye-Objektive weichen davon am
Rand ab. Am Aufbau waren das **9 bis 12 px**. Wie sich das auswirkt, sieht man
daran, dass eine Rueckrechnung der Bandenoberkante drei verschiedene Hoehen fuer
dieselbe Kante ergab: +10.1 cm bei 0.5 m, +5.5 cm bei 1.0 m, +12.7 cm bei
2.5 m. Mit einer einzigen Hoehe sind nah und fern deshalb nicht gleichzeitig zu
treffen.

Deshalb wird die Zone gemessen. Die Kommandos laufen in `rotation_calibration`:

```bash
# Pylone in einer Entfernung aufstellen, dann:
ros2 topic pub --once /camera_lidar/calib_cmd std_msgs/msg/String "data: zone"
# versetzen, wiederholen -- 5 bis 6 Positionen von 0.3 bis 2.5 m

ros2 topic pub --once /camera_lidar/calib_cmd std_msgs/msg/String "data: zonelist"
ros2 topic pub --once /camera_lidar/calib_cmd std_msgs/msg/String "data: zonedel 3 7"
ros2 topic pub --once /camera_lidar/calib_cmd std_msgs/msg/String "data: zonefit"
ros2 topic pub --once /camera_lidar/calib_cmd std_msgs/msg/String "data: save"
ros2 topic pub --once /camera_lidar/reload std_msgs/msg/Empty "{}"
```

`zone` tastet die radiale Linie durch das Lidar-Cluster ab und misst, ueber
welchen Radiusbereich dort die Pylonenfarbe steht -- also genau den Bereich, den
der Mapper spaeter abtasten soll. `zonefit` legt

    r(rho) = r0 + k / rho

durch die Messungen, getrennt fuer innen und aussen. Diese Form ist nicht
geraten, sie folgt aus der Geometrie: fuer `theta` nahe 90 Grad ist
`atan2(rho, dz)` ungefaehr `pi/2 - dz/rho`, also `r` ungefaehr
`f*pi/2 - f*dz/rho`. Der 1/rho-Anteil traegt die Hoehe der Kante, der konstante
Anteil die Brennweite -- **und mit ihm gleich den Modellfehler**, den man sonst
nirgends loswird. Genau das ist der Gewinn gegenueber dem Rechnen.

Am Aufbau, 9 Samples von 0.27 bis 2.14 m:

```
r_innen  = 399.9  -2.31/rho     RMS 1.0 px
r_aussen = 400.1 +13.33/rho     RMS 1.6 px
```

Das gerechnete Modell setzt den Horizontring auf 412.3 px, der Fit laeuft gegen
399.9 px -- die Differenz von 12.4 px ist der Modellfehler. Und `zone_k_in` von
-2.31 entspricht nur 0.9 cm: das Objektiv sitzt praktisch auf der Hoehe der
Bandenoberkante, deren Bildradius damit konstant ist.

**Weit spreizen.** Der Fit trennt einen konstanten von einem 1/rho-Anteil und
braucht dafuer nah UND fern. Unter Faktor 2.5 Spreizung warnt `zonefit`.

Fuer die Kommandos gilt: erst `background` (sonst findet die Node den eigenen
Aufbau statt der Pylone, siehe unten), und `target_label` auf die Pylonenfarbe
setzen. `target_range_max_m` steht auf 1.5 m -- fuer Samples weiter draussen
hochsetzen.

### Die Bande live finden: `band_detect`

Die verlaesslichste Kante im Bild ist die **Unterkante** der schwarzen Bande,
also der Uebergang zur hellen Matte: dahinter liegt immer dasselbe, egal in
welche Richtung. Die Oberkante taugt dafuer nicht, hinter ihr ist mal weisse
Wand, mal dunkle Couch, mal Holz. An 1362 Kantenpaaren gemessen:

| Kante | RMS des Fits |
| --- | --- |
| Unterkante (gegen die Matte) | 5.3 px |
| Oberkante (gegen den Raum) | 12.8 px |
| zum Vergleich: Pylonenfarbe (`zonefit`) | 1.0 / 1.6 px |

`_bande_finden` laeuft je Azimut von innen nach aussen und nimmt die erste
Stelle, an der es dauerhaft hell wird -- `band_run` Pixel am Stueck. Damit loest
ein einzelner Glanzpunkt auf der Bande die Kante nicht vorzeitig aus. Am Aufbau
werden so **349 von 360 Azimuten** getroffen.

Zwei Ausreisserfilter, beide physikalisch begruendet: die Unterkante muss immer
weiter aussen liegen als die Oberkante (die Bande ist rund 10 cm hoch), und
Nachbarazimute muessen sich aehneln (die Bande springt nicht). Der Filter
schrumpfte den gefundenen Radiusbereich von 368..436 auf 403..436 px.

```bash
ros2 param set /lidar_pixel_mapper band_detect true
ros2 param set /lidar_pixel_mapper zone_from_band true   # Zone daran ausrichten
ros2 param set /lidar_pixel_mapper band_steps 360        # Azimutaufloesung
ros2 param set /lidar_pixel_mapper band_dark_max 60      # so dunkel ist die Bande
ros2 param set /lidar_pixel_mapper band_bright_min 100   # so hell ist die Matte
ros2 param set /lidar_pixel_mapper band_run 4            # helle Pixel am Stueck
ros2 param set /lidar_pixel_mapper band_smooth 9         # Medianfenster
ros2 param set /lidar_pixel_mapper band_max_dev 12.0     # max Abweichung [px]
```

Mit `zone_from_band` kommt die Unterkante der Zone live aus dem Bild, die
Oberkante bleibt konstant. Wo keine Kante gefunden wurde, greift die
Kalibrierkurve -- die Bandensuche kann also nur verbessern, nie verschlechtern.

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

Siehe auch den Abschnitt **Rechenzeit im Betrieb** weiter unten -- dort stehen
die Zahlen fuer Zone, Bandenerkennung und Debug-Bild.

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

# 3b. Abgriffszone messen -- Pylone in 5 bis 6 Entfernungen von 0.3 bis 2.5 m
#     aufstellen und je Position "zone" senden. target_range_max_m vorher
#     hochsetzen, sonst sieht die Node nur bis 1.5 m.
ros2 param set /camera_rotation_calibration target_range_max_m 3.0
ros2 topic pub --once /camera_lidar/calib_cmd std_msgs/msg/String "data: zone"
ros2 topic pub --once /camera_lidar/calib_cmd std_msgs/msg/String "data: zonelist"
ros2 topic pub --once /camera_lidar/calib_cmd std_msgs/msg/String "data: zonefit"
ros2 topic pub --once /camera_lidar/calib_cmd std_msgs/msg/String "data: save"
ros2 topic pub --once /camera_lidar/reload std_msgs/msg/Empty "{}"

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

Welche Farben ueberhaupt gesucht werden, steuert `active_labels`. Der Parameter
wird bei jedem Scan gelesen, laesst sich also im Betrieb umschalten -- anders
als die Schwellen in `color.*`, die beim Start eingefroren werden:

```bash
ros2 param set /lidar_pixel_mapper active_labels "[rot,gruen]"
ros2 param set /lidar_pixel_mapper active_labels "[rot,gruen,magenta]"
```

Magenta produziert in groesserer Entfernung leicht Fehltreffer und stoert nur,
solange die Parkzone nicht gebraucht wird.
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
  seiner echten x/y-Position. Womit er eingefaerbt wird, entscheidet
  `cloud_color_mode`:

  | Modus | Farbe | wofuer |
  | --- | --- | --- |
  | `label` (Default) | kraeftig je Label, Rest dunkelgrau | Pylonen finden |
  | `raw` | die gemessene Pixelfarbe | Kalibrierung und Schwellen pruefen |

  `label` nimmt die Palette `CLOUD_BGR` aus `colors.py`: rot `0xFF0000`, gruen
  `0x00FF00`, magenta `0xFF00FF`, schwarz `0x2D2D2D`, unbekannt `0x555555`.
  Die Werte sind exakt, ein Konsument kann also direkt darauf pruefen statt
  Farbbereiche zu raten.

  Warum das noetig ist: am echten Aufbau gemessen liegen im `raw`-Modus
  praktisch alle Punkte bei R/G/B um 20 bis 25 -- 1028 verschiedene Farbwerte,
  aber allesamt dunkelgrauer Matsch, in dem sich rot und gruen kaum trennen
  lassen. Im `label`-Modus sind es 4 eindeutige Werte.

  `raw` bleibt trotzdem die Ansicht, an der man sieht, ob die Kalibrierung
  sitzt: stehen die roten Punkte auf dem roten Klotz, stimmt yaw.

  In Foxglove ein 3D-Panel oeffnen, Topic abonnieren, Color-Mode auf `RGB`
  stellen. Der Frame ist der des Lidars (bei `sllidar` = `laser`).

  ```bash
  ros2 param set /lidar_pixel_mapper cloud_color_mode raw
  ros2 param set /lidar_pixel_mapper cloud_color_mode label
  ```

  Der Parameter wird bei jedem Scan neu gelesen, wirkt also sofort -- anders
  als die Farbschwellen `color.*`, die nur beim Start eingelesen werden.
* **`/camera_lidar/debug_image`** -- dasselbe andersherum: das Fisheye-Bild mit
  den eingezeichneten Projektionen, dem Bildkreis und einem Pfeil nach vorne.

Feiner steuerbar mit `publish_cloud`, `publish_debug_image` und `debug_rate_hz`
(Default 5 Hz fuer das Bild; die PointCloud geht mit jedem Scan raus).
`/camera_lidar/summary` zaehlt nur die Labels und laeuft immer.

Die Kalibrier-Node hat denselben Schalter fuer `/camera_lidar/calib_debug`.

### Das Debug-Bild: rund plus entzerrt

`/camera_lidar/debug_image` liefert zwei Ansichten uebereinander. Oben das
runde Fisheye mit Bildkreis, Horizontring, Zonengrenzen, den abgetasteten
Segmenten in Label-Farbe und der gefundenen Bandenkante (magenta). Darunter ein
**entzerrter Streifen**: Azimut waagerecht, Bildradius senkrecht.

Der Streifen ist die nuetzlichere Ansicht. Im runden Bild liegt alles
Interessante am aeusseren Rand und ist dort auf wenige Pixel
zusammengedraengt; aufgerollt liegen die Schichten sauber uebereinander -- oben
der Raum, darunter die schwarze Bande, ganz unten die helle Matte. Ob die
Abgriffszone auf der Bande sitzt oder darueber hinweggreift, sieht man dort auf
einen Blick, im runden Bild nicht.

```bash
ros2 param set /lidar_pixel_mapper debug_polar true
ros2 param set /lidar_pixel_mapper debug_polar_height 150
```

Die Kopfzeile nennt den Modus (`zone: Bande live` / `zone: kalibriert` /
`zone: ... gerechnet`), den Stimmenanteil, die Punktzahl und wie viele Azimute
die Bandensuche getroffen hat.

## Rechenzeit im Betrieb

Am Jetson gemessen (Momentanlast ueber /proc, 2100 Punkte je Scan, 15 Hz):

| Konfiguration | CPU |
| --- | --- |
| nur Klassifikation | 28 % eines Kerns |
| + Bandenerkennung (`band_detect`, 15 Hz) | 56 % |
| + Debug-Bild mit Polar-Streifen (5 Hz) | 94 % |
| dasselbe mit `band_steps: 180` | 84 % |

Zwei Dinge sind daran bemerkenswert. Das **Debug-Bild ist der teuerste Posten**
mit 38 Prozent, obwohl es nur mit 5 Hz laeuft -- Zeichnen und Polar-Entzerrung
auf 1280x960 kosten. Im Wettkampflauf also `debug:=false`, das spart die 38
Prozent sofort. Und die **Bandenerkennung kostet 28 Prozent**, weil sie bei
jedem Scan laeuft; `band_steps: 180` statt 360 bringt davon 10 Prozent zurueck,
bei 2 Grad Stuetzstellenabstand immer noch dicht genug fuer eine Bande.

Vorsicht bei der Messmethode: `ps -o pcpu` liefert den Durchschnitt ueber die
gesamte Lebensdauer des Prozesses und taugt fuer einen Vorher/Nachher-Vergleich
nicht. Die Zahlen oben stammen aus der Differenz von `utime + stime` in
`/proc/<pid>/stat` ueber ein festes Intervall.

Beim Optimieren war der groesste Brocken uebrigens nicht das, was man erwartet:
der Medianfilter der Bandenerkennung kostete als Python-Schleife **17.7 von
24 ms**; vektorisiert ueber ein Gleitfenster sind es 1.1 ms. Der V-Kanal
dagegen bleibt bei `cv2.cvtColor` (3.8 ms) -- `img.max(axis=2)` liefert zwar
dasselbe Ergebnis, braucht aber 34.9 ms.

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

Neben Bildkreis, Lage und Blindsektoren stehen dort die vier Koeffizienten der
Abgriffszone:

```yaml
zone_r0_in:  399.9    # r_innen(rho)  = zone_r0_in  + zone_k_in  / rho
zone_k_in:    -2.31
zone_r0_out: 400.1    # r_aussen(rho) = zone_r0_out + zone_k_out / rho
zone_k_out:  +13.33
```

Sind `zone_r0_out` und `zone_k_out` beide 0, gilt die Zone als nicht
kalibriert und der Mapper faellt auf die gerechneten Hoehen zurueck. Die
Startmeldung sagt `GEMESSEN` oder `GERECHNET`, damit man nicht raten muss.

## Tests

Das Projektionsmodell laeuft ohne Hardware:

```bash
cd /workspace/src/camera_lidar_fusion && python3 -m pytest test/test_fisheye_model.py -q
```

`test/fake_scan.py` publiziert einen synthetischen 360-Grad-Scan auf `/scan`,
damit sich `lidar_pixel_mapper` auch ohne laufendes Lidar durchtesten laesst.
