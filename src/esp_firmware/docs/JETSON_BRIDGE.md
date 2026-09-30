# ESP32-S3 Controller — Protokoll-Spezifikation für die Jetson-Bridge

Referenz für die Gegenstelle auf dem Jetson. Quelle der Wahrheit ist
`src/main.cpp` auf dem ESP.

---

## 1. Physikalische Verbindung

| | |
|---|---|
| Schnittstelle | UART, `Serial1` auf dem ESP |
| Baudrate | **115200**, 8N1 |
| ESP RX | GPIO 10 (→ Jetson TX) |
| ESP TX | GPIO 11 (→ Jetson RX) |
| Flusskontrolle | keine |

---

## 2. Rahmenformat

```
+------------+---------+----------------------+
| 0xA5       | CMD     | PAYLOAD (0..17 Byte) |
| Start-Byte | 1 Byte  | Länge ergibt sich    |
|            |         | aus CMD              |
+------------+---------+----------------------+
```

**Es gibt kein Längenfeld und keine Prüfsumme.** Beide Seiten müssen die
Nutzlastlänge je Befehl fest kodiert haben (Tabellen unten).

### Zweite Rahmenform: mit Sendezeitstempel

```
+------------+---------+--------------------+----------------------+
| 0xA6       | CMD     | uint32 t_tx_us     | PAYLOAD (0..12 Byte) |
| Start-Byte | 1 Byte  | Sendezeitpunkt     | wie bei 0xA5         |
+------------+---------+--------------------+----------------------+
```

Ein `0xA6`-Rahmen ist ein `0xA5`-Rahmen mit vier zusätzlichen Byte zwischen
`CMD` und Nutzlast: den unteren 32 Bit der ESP-Uhr in Mikrosekunden. Nutzlast
und deren Länge sind unverändert — der Parser braucht nur einen Zweig mehr.

* **Der ESP versteht `0xA6` im Empfang immer.** Die Bridge darf ihre Befehle
  jederzeit stempeln, ohne vorher etwas einzuschalten.
* **Der ESP *sendet* `0xA6` nur, wenn er dazu aufgefordert wurde** — mit
  `0xB2 STAMP_MODE`. Default ist aus, damit eine Bridge, die `0xA6` nicht kennt,
  weiterläuft.
* Ausnahme: `0xB1 TIME_RSP` geht **nie** als `0xA6` raus. Es trägt seine
  Zeitstempel schon mit voller 64-Bit-Breite in der Nutzlast.

Details zur Bedeutung des Stempels und zum Umrechnen in die Jetson-Uhr:
Abschnitt 5.

### Regeln, die der Parser einhalten muss

1. **Mehrbyte-Zahlen sind Big-Endian** (MSB zuerst). Gilt für `int16` und `int32`.
2. **Fließkommazahlen werden als `int32 × 1000` übertragen.** Kp = 4,25 → `4250`.
   Kein IEEE754 auf der Leitung.
3. **Ein Paket muss in einem einzigen `write()` rausgehen.** Der ESP setzt seinen
   Parser zurück, wenn zwischen zwei Bytes eines Pakets mehr als **100 ms**
   liegen. Byteweises Senden mit Pausen zerlegt das Paket.
4. **Der RX-Strom vom ESP enthält auch ASCII-Klartext.** Bei `CMD_EMERGENCY` und
   `CMD_TRIM` schreibt der ESP zusätzlich lesbare Statuszeilen auf dieselbe
   Leitung. Das ist ungefährlich — ASCII enthält nie `0xA5`, der Sync-Scan
   überspringt es —, aber die Bridge muss diese Bytes tolerieren statt daran zu
   scheitern. Sinnvoll: als Log-Zeile ausgeben.
5. **Unbekannte CMD-Bytes verwirft der ESP** und sucht das nächste `0xA5`. Die
   Bridge sollte es genauso halten.

---

## 3. Jetson → ESP

| CMD | Name | Payload | Beschreibung |
|---|---|---|---|
| `0x10` | MOTOR | 3 B | `dir(1)` + `uint16 speed` |
| `0x20` | SERVO | 3 B | `id(1)` + `int16 lenkung` |
| `0x30` | LED | 1 B | `0`=aus, `≠0`=an |
| `0x40` | CALIBRATE | 0 B | manuelle Kalibrierung starten → Antwort `0x42` |
| `0x41` | CAL | 2 B | `aktion(1)` + `arg(1)` → Antwort `0x42` |
| `0x50` | TORQUE | 0 B | Servo-Last auf USB-Konsole ausgeben |
| `0x60` | TRIM | 1 B | `0`=links, `1`=rechts, `2`=speichern |
| `0x80` | PID_SET | 5 B | `paramId(1)` + `int32 wert×1000` |
| `0x81` | PID_GET | 0 B | → Antwort `0x82` |
| `0x83` | PID_SAVE | 0 B | Parameter ins NVS → Antwort `0x84` |
| `0x90` | MOVE | 5 B | `moveId(1)` + `int32 weite` in 1/10° (relativ) |
| `0x91` | MOVE_ABORT | 0 B | laufende Fahrt abbrechen |
| `0x92` | PROGRESS | 0 B | → Antwort `0x94` |
| `0xA0` | BATTERY | 0 B | → Antwort `0xA1` |
| `0xB0` | TIME_SYNC | 1 B | `seq(1)` → Antwort `0xB1` |
| `0xB2` | STAMP_MODE | 1 B | `0`=aus, `1`=an → Antwort `0xB3` |
| `0xC0` | TELEM_RATE | 2 B | `uint16` Takt in ms, `0`=aus → Antwort `0xC1` |
| `0xFF` | EMERGENCY | 0 B | Nothalt, aktive Bremse |

### `0x10` MOTOR — offene Steuerung

```
A5 10 <dir> <speedHi> <speedLo>
```

* `dir`: `0` = vorwärts, `1` = rückwärts
* `speed`: `0..1023` (10-Bit-PWM)
* **`speed = 0` bedeutet Auslaufen (Coast), nicht Bremsen.** Zum aktiven Bremsen
  gibt es nur `0xFF`.
* Bricht eine laufende Positionsfahrt ab → es kommt ein `0x93` mit Status `0x02`.

> **Heartbeat-Pflicht:** Kommt 5 Sekunden lang kein Befehl, geht der Motor
> selbstständig in Coast. Für Dauerfahrt muss die Bridge `0x10` regelmäßig
> nachschicken (empfohlen: alle 100–500 ms). **Positionsfahrten sind davon
> ausgenommen** — die dürfen länger als 5 s laufen, ohne dass etwas nachkommt.

### `0x20` SERVO — Lenkung

```
A5 20 <id> <pctHi> <pctLo>
```

* `id`: Servo-ID auf dem SCS-Bus (normalerweise `1`)
* `pct`: `int16`, **−100 … +100**. Negativ = rechts, positiv = links, `0` = Mitte.
* Der ESP mappt das auf die kalibrierten Anschläge und begrenzt intern auf
  **80 % des mechanischen Hubs**. `±100` ist also bewusst nicht Vollanschlag.

### `0x40` / `0x41` — manuelle Lenkungs-Kalibrierung

Der SC09 kann sein Drehmoment **nicht** begrenzen: fährt er von selbst gegen
einen Anschlag, drückt er mit vollem Moment weiter, bis Anlenkung oder Getriebe
nachgeben. Das frühere automatische Antasten gibt es deshalb nicht mehr — die
Anschläge werden von Hand angefahren und einzeln bestätigt.

```
A5 40                 Kalibriermodus starten (identisch zu 0x41 mit Aktion 0x00)
A5 41 <aktion> <arg>
```

| Aktion | Name | `arg` | Wirkung |
|---|---|---|---|
| `0x00` | START | – | Modus starten, Antrieb auf Coast, Torque an, aktuelle Stellung halten |
| `0x01` | MINUS | Ticks (`0` = Schrittweite) | ein Schritt Richtung Position `0` |
| `0x02` | PLUS | Ticks (`0` = Schrittweite) | ein Schritt Richtung Position `1023` |
| `0x03` | CENTER | – | aktuelle **Ist**-Stellung als Mitte merken |
| `0x04` | LEFT | – | aktuelle Ist-Stellung als linken Anschlag merken |
| `0x05` | RIGHT | – | aktuelle Ist-Stellung als rechten Anschlag merken |
| `0x06` | SAVE | – | prüfen, ins NVS schreiben, Modus beenden |
| `0x07` | ABORT | – | abbrechen, gespeicherte Werte bleiben |
| `0x08` | FREE | – | Torque **aus**: Lenkung von Hand stellen |
| `0x09` | HOLD | – | Torque an, hält die aktuelle Ist-Stellung |
| `0x0A` | GOTO_CENTER | – | gemerkte Mitte anfahren |
| `0x0B` | SET_STEP | Ticks (1…200) | Schrittweite setzen (Default 10 ≈ 2,9°) |
| `0x0C` | STATUS | – | nur Zustand abfragen, ändert nichts |

**Jede** dieser Aktionen — auch `0x40` — wird mit `0x42` beantwortet.

Ablauf:

1. `0x40` senden.
2. Mit `MINUS`/`PLUS` auf Geradeaus stellen → `CENTER`.
3. Langsam an den linken Anschlag → `LEFT`. **Kurz vor dem harten Anschlag
   stoppen.** Meldet die Antwort, dass die Ist-Position dem Soll nicht mehr
   folgt, drückt der Servo bereits gegen die Mechanik.
4. Zurück und an den rechten Anschlag → `RIGHT`.
5. `SAVE`.

Alternative ohne jede Servokraft: `FREE`, Lenkung von Hand an den Anschlag
schieben, dort `LEFT`/`RIGHT`, danach `HOLD`.

Der ESP prüft vor dem Speichern, dass alle drei Marken gesetzt sind, dass die
Anschläge **mindestens 50 Ticks** auseinanderliegen und dass die Mitte
dazwischen liegt. Sonst kommt `0x42` mit Status `0x02` und **nichts** wird
geschrieben. Erfolgreiches Speichern setzt den Trim-Offset auf `0` zurück — er
bezog sich auf die alte Mitte.

Solange der Modus läuft, **ignoriert der ESP `0x20` SERVO** (mit Log-Zeile auf
der USB-Konsole) — ein Lenkbefehl würde die von Hand angefahrene Stellung sofort
verwerfen.

> Welcher Anschlag „links" ist, entscheidet allein, welchen der Bediener mit
> `0x04` bestätigt. Die Zuordnung in `0x20` rechnet symmetrisch, eine gespiegelt
> montierte Lenkung funktioniert also auch mit vertauschten Rohwerten.

### `0x80` PID_SET

```
A5 80 <paramId> <int32 wert×1000>
```

| paramId | Parameter | Einheit | Default | Beispiel |
|---|---|---|---|---|
| `0` | Kp | Duty pro Count | 4,0 | Kp=4,5 → `4500` |
| `1` | Ki | Duty pro (Count·s) | 0,5 | Ki=0,3 → `300` |
| `2` | Kd | Duty pro (Count/s) | 0,10 | Kd=0,08 → `80` |
| `3` | I-Limit | Duty (Anti-Windup) | 200 | 250 → `250000` |
| `4` | maxDuty | 0..1023 | 700 | 800 → `800000` |
| `5` | Toleranz | 1/10 Grad | 50 (=5,0°) | 1,5° → `15000` |
| `6` | Verweilzeit | ms | 200 | 300 ms → `300000` |
| `7` | Timeout | ms | 10000 | 15 s → `15000000` |
| `8` | Anlauf-Duty | 0..1023 | 0 (aus) | 200 → `200000` |

> **Anlauf-Duty (`8`)** überwindet die Haftreibung. Kurz vor dem Ziel wird die
> Regelabweichung so klein, dass `Kp × Fehler` unter die Losbrechschwelle des
> Getriebemotors fällt — er bleibt stehen und die Fahrt läuft in den Timeout.
> Mit diesem Wert wird außerhalb des Zielfensters nie weniger angelegt.

> **Achtung, häufige Fehlerquelle:** Die `×1000`-Kodierung gilt für *alle*
> Parameter, auch die ganzzahligen. `maxDuty = 700` wird als `700000`
> übertragen, nicht als `700`. Bei `700` käme auf dem ESP `0` an.

`0x80` ändert nur die **laufenden** Werte im RAM. Zum dauerhaften Sichern
anschließend `0x83` senden.

### `0x83` PID_SAVE

```
A5 83
```

Schreibt den **kompletten** aktuellen Parametersatz (alle acht Werte aus der
Tabelle oben) ins NVS und antwortet mit `0x84`. Nach dem nächsten Boot lädt der
ESP diese Werte automatisch.

Typischer Ablauf: alle nötigen `0x80` senden, dann **einmal** `0x83`. Nicht nach
jedem einzelnen Parameter speichern.

> NVS verwirft Schreibvorgänge, bei denen sich der Wert nicht geändert hat.
> Wiederholtes Speichern identischer Werte kostet also keine Flash-Zyklen —
> zyklisches Speichern im Sekundentakt ist trotzdem eine schlechte Idee.

### `0x90` MOVE — Positionsfahrt

```
A5 90 <moveId> <int32 weite in 1/10 Grad>
```

* `moveId`: `1..255`, von der Bridge vergeben. `0` vermeiden (Startwert intern).
* `weite`: **relativ** zur aktuellen Stellung, in 1/10 Grad der Ausgangswelle,
  vorzeichenbehaftet. `900` = „drehe 90° weiter", `-450` = „45° zurück".
* Es gibt **keinen** Nullpunkt und keine Referenzierung — eine Fahrt bezieht sich
  immer auf das Hier und Jetzt und ist damit reset-fest.
* Auf jede Fahrt folgt garantiert **genau ein** `0x93` MOVE_DONE mit derselben ID.

> **Die ×10 gilt nur auf der Leitung.** Für 90° sendet die Bridge `900`.

**Es läuft immer nur eine Fahrt.** Wird `0x90` gesendet, während noch eine aktiv
ist, wird die alte abgelöst und quittiert das mit `0x93` / Status `0x02`, bevor
die neue startet. Die Bridge bekommt also für jede ID eine Antwort, egal in
welcher Reihenfolge sie sendet.

---

## 4. ESP → Jetson

| CMD | Name | Payload | Ausgelöst durch |
|---|---|---|---|
| `0x42` | CAL_RSP | 11 B | `0x40`, `0x41` |
| `0x70` | BUTTON | 1 B | Tastendruck (unaufgefordert) |
| `0x82` | PID_RSP | 12 B | `0x81` |
| `0x84` | PID_SAVED | 1 B | `0x83` |
| `0x93` | MOVE_DONE | 6 B | Ende einer Fahrt (unaufgefordert) |
| `0x94` | PROGRESS_RSP | 11 B | `0x92` |
| `0xA1` | BATTERY_RSP | 6 B | `0xA0` |
| `0xA2` | BATTERY_WARN | 6 B | Unterspannung (unaufgefordert) |
| `0xB1` | TIME_RSP | 17 B | `0xB0` |
| `0xB3` | STAMP_RSP | 1 B | `0xB2` |
| `0xC1` | TELEMETRY | 12 B | Takt aus `0xC0` (unaufgefordert) |

### `0x70` BUTTON
```
A5 70 01
```
`01` = gedrückt. Entprellt mit 200 ms, es gibt kein Loslass-Event.

### `0x42` CAL_RSP
```
A5 42 <aktiv> <flags> <status> <int16 pos> <int16 mitte> <int16 links> <int16 rechts>
```

* `aktiv`: `1` = Kalibriermodus läuft, `0` = nicht (mehr)
* `flags`: Bit 0 = Mitte gesetzt, Bit 1 = Links gesetzt, Bit 2 = Rechts gesetzt,
  Bit 3 = Torque frei
* `pos`: zuletzt kommandierte Rohposition `0..1023`
* `mitte`/`links`/`rechts`: die in diesem Durchgang gesetzte Marke, solange das
  zugehörige Flag `0` ist stattdessen der **gespeicherte** Wert

| status | Bedeutung |
|---|---|
| `0x00` | Aktion ausgeführt |
| `0x01` | gespeichert, Modus beendet |
| `0x02` | abgelehnt — unvollständig oder unplausibel, **nichts geschrieben** |
| `0x03` | Servo antwortet nicht |
| `0x04` | Aktion braucht einen laufenden Kalibriermodus (erst `0x40`) |
| `0x05` | Bereichsende `0`/`1023` erreicht |

### `0x82` PID_RSP
```
A5 82 <int32 Kp×1000> <int32 Ki×1000> <int32 Kd×1000>
```
Liefert nur die drei Regelparameter, nicht Limits/Timeouts.

### `0x84` PID_SAVED
```
A5 84 <status>
```
`0x00` = ins NVS geschrieben, `0x01` = fehlgeschlagen (Partition voll oder
defekt). Kommt nur als Antwort auf `0x83`.

### `0x93` MOVE_DONE
```
A5 93 <moveId> <status> <int32 ist-position in 1/10 Grad>
```

| status | Bedeutung |
|---|---|
| `0x00` | **OK** — Ziel erreicht und für die Verweilzeit gehalten |
| `0x01` | **TIMEOUT** — Ziel in `timeoutMs` nicht erreicht, abgebrochen |
| `0x02` | **ABORTED** — durch `0x91`, `0x10`, `0xFF` oder eine neue Fahrt abgelöst |

Nach *jedem* Ende geht der Motor in **Coast, nicht in Positionshaltung**. Bei
äußerer Last oder Hangabtrieb driftet die Achse danach weg. Wer halten will,
muss zyklisch neu anfahren.

> Das Positionsfeld ist — anders als die Fahrweite in `0x90` — **absolut**:
> Encoder-Stand seit Boot bzw. seit `z` auf der Konsole. Es dient als Telemetrie,
> nicht als Bezug für die nächste Fahrt.

### `0x94` PROGRESS_RSP
```
A5 94 <moveId> <aktiv> <prozent> <int32 ist> <int32 ziel>
```
* `aktiv`: `1` = Fahrt läuft, `0` = idle. Bei `0` beziehen sich die restlichen
  Felder auf die **zuletzt** gefahrene Bewegung.
* `prozent`: `0..100`, berechnet aus zurückgelegtem Weg / Gesamtweg. Bei
  Überschwingen gedeckelt auf 100.
* `ist` / `ziel`: 1/10 Grad, **absolut** (Encoder-Stand seit Boot). `ziel` ist die
  aus der relativen Fahrweite berechnete Endposition.

### `0xA1` BATTERY_RSP / `0xA2` BATTERY_WARN
```
A5 A1 <int32 pack-mV> <int16 zelle-mV>
```
Identische Nutzlast, unterschiedlicher Anlass.

* Der ESP misst **alle 15 s** selbstständig.
* `0xA1` kommt nur auf Anfrage (`0xA0`) und liefert den letzten Messwert.
* `0xA2` kommt **unaufgefordert**, sobald die Zellspannung unter **3,80 V**
  fällt (4S-Pack ⇒ 15,2 V), und wiederholt sich danach **höchstens einmal pro
  Minute**, solange der Zustand anhält.
* Entwarnung erst oberhalb **3,85 V/Zelle** (Hysterese). Es gibt **kein**
  eigenes Entwarn-Paket — das Ausbleiben von `0xA2` ist das Signal.

> Der ESP schaltet bei Unterspannung **nichts ab**. Reagieren muss der Jetson.

### `0xB1` TIME_RSP
```
A5 B1 <seq> <int64 t_rx_us> <int64 t_tx_us>
```

* `seq`: unverändert aus der Anfrage zurück, damit die Bridge Antworten
  zuordnen kann, auch wenn eine Runde verlorengeht.
* `t_rx_us`: ESP-Uhr, als das **letzte Byte der Anfrage** angekommen war.
* `t_tx_us`: ESP-Uhr, wenn das **letzte Byte dieser Antwort** die Leitung
  verlässt. Die Zeit zum Rausschieben der 20 Rahmenbyte (1,74 ms bei 115200)
  ist bereits eingerechnet — der Wert liegt also in der Zukunft, wenn der ESP
  ihn schreibt.

Beide Werte sind volle `int64`-Mikrosekunden seit ESP-Boot, kein Überlauf.
Rechnung siehe Abschnitt 5.

### `0xC1` TELEMETRY
```
A5 C1 <int32 pos in 1/10 grad> <int32 tempo in 1/10 grad/s> <int16 duty> <int16 mA>
```

Der Fahrzustand, den der ESP von sich aus im Takt von `0xC0` schickt.

**Drei der vier Felder sind vorzeichenbehaftet, eines nicht** — das ist die
wahrscheinlichste Stolperstelle beim Parsen:

| Feld | Typ | Vorzeichen |
|---|---|---|
| `pos` | `int32` | **ja** — negativ, wenn die Welle unter dem Startpunkt steht |
| `tempo` | `int32` | **ja** — negativ bei Rückwärtsfahrt |
| `duty` | `int16` | **ja** — negativ bei Rückwärtsfahrt |
| `mA` | `int16` | **nein** — immer ≥ 0 |

* `pos`: Stellung der Ausgangswelle, **absolut** — Encoderstand seit Boot bzw.
  seit `z` auf der Konsole. Derselbe Bezug wie in `0x93`/`0x94`.
* `tempo`: Drehgeschwindigkeit der Ausgangswelle. `1800` = 180 °/s = 30 U/min,
  `-1800` dasselbe rückwärts.
* `duty`: was an der Brücke anliegt, −1023…+1023. `0` bei Coast **und** bei
  Bremse — das ist kein Fehler, es liegt dann wirklich kein Tastverhältnis an.
* `mA`: Motorstrom. Der VNH5019 meldet auf seinem CS-Ausgang **nur den
  Betrag**, nicht die Richtung. Wer die Richtung des Stroms braucht, liest sie
  am Vorzeichen von `duty` ab — mit der Einschränkung, dass beide beim
  Ausrollen und Bremsen nichts Sinnvolles hergeben.

> Welche Drehrichtung "vorwärts" ist, hängt daran, wie Motor und Encoder
> verdrahtet sind — das Protokoll legt es nicht fest. Dafür gibt es in
> `src/main.cpp` den Schalter `DRIVE_INVERT`:
>
> | Symptom | Ursache | Abhilfe |
> |---|---|---|
> | positiver Motorwert fährt rückwärts, `tempo` passt dazu | das Fahrzeug ist als Ganzes andersherum verdrahtet | `DRIVE_INVERT = true` |
> | Motor stimmt, aber `tempo` und `pos` haben das falsche Vorzeichen | nur die Encoderspuren A/B sind vertauscht | Spuren tauschen, `DRIVE_INVERT` **nicht** anfassen |
>
> `DRIVE_INVERT` dreht Motor **und** Encoder zusammen. Nur eines von beiden zu
> drehen wäre ein Eigentor: die Positionsregelung liest dann ein Vorzeichen,
> das nicht zu ihrer Stellgröße passt, und fährt vom Ziel weg statt darauf
> zu, bis der Timeout greift.

> **Position und Tempo entstehen beide im 10-ms-Takt des Motortasks** und sind
> in jedem Paket frisch. `0xC0` ist deshalb auf minimal **10 ms** begrenzt —
> schneller zu senden hiesse, dasselbe Paket zweimal zu schicken.
>
> Diese Grenze steht in `TELEMETRY_MS_MIN`, und zwar **zweimal**: in
> `src/main.cpp` und in `esp_serial_bridge.py`. Bindend ist die in der
> Firmware; wer nur die Python-Seite ändert, bekommt weiter den alten Takt.
>
> **Sendetakt und Messfenster sind zwei verschiedene Dinge.** Das Tempo bildet
> der ESP aus einem *gleitenden* Fenster über die letzten `n` Abtastungen:
>
> ```
> tempo = (count[jetzt] - count[jetzt - n]) / (t[jetzt] - t[jetzt - n])
> ```
>
> Bei jeder Abtastung fällt ein neuer Wert an — die Ausgaberate hängt also
> nicht an `n`. Was an `n` hängt, ist die Auflösung, denn ein einzelner
> Encoderimpuls im Fenster ist der kleinste Sprung, den die Messung machen
> kann:
>
> | `n` | Fenster | ein Impuls | Verzögerung |
> |---|---|---|---|
> | 1 | 10 ms | 14,7 U/min = 88 °/s | 5 ms |
> | 5 | 50 ms | 2,9 U/min = 17,6 °/s | 25 ms |
> | 10 | 100 ms | 1,5 U/min = 8,8 °/s | 50 ms |
>
> bei 408 Impulsen je Umdrehung der Ausgangswelle und rund 30 U/min
> Höchstdrehzahl. **Mit `n = 1` liegt der kleinste Messschritt in derselben
> Grössenordnung wie der Vollausschlag** — das Tempo springt dann zwischen
> wenigen diskreten Stufen hin und her. Das ist kein Fehler, sondern die
> Auflösung des Encoders bei 10 ms.
>
> Eingestellt wird `n` auf der USB-Konsole mit `sw<n>`, z. B. `sw5`. Die
> Verzögerung beträgt exakt ein halbes Fenster — das ist der Vorzug des
> gleitenden Fensters gegenüber einem EMA, dessen Zeitkonstante man nur
> schätzen kann.
>
> Wer beides will, feine Auflösung *und* 10 ms Verzögerung, kommt mit diesem
> Encoder nicht weiter: dafür bräuchte es Zeitstempel einzelner Flanken
> (M/T-Verfahren) statt Impulse je Fenster.

> Der Takt ist weich: der ESP sendet aus seinem `loop()` heraus. Genau dafür
> gibt es den Sendezeitstempel — der Jetson soll den Zeitpunkt **ablesen**, statt
> ihn aus dem Nenn-Takt hochzurechnen.

### `0xC0` TELEM_RATE
```
A5 C0 <uint16 takt in ms>
```
`0` schaltet ab, sonst 20…60000 ms; Werte darunter zieht der ESP still auf 20.
Beantwortet wird der Befehl mit einem sofortigen `0xC1` — das ist gleichzeitig
die Quittung und der erste Messwert.

Der Takt liegt **nicht** im NVS. Nach einem ESP-Reset ist die Telemetrie aus;
die Bridge stellt sie beim Verbinden neu ein.

### `0xB3` STAMP_RSP
```
A5 B3 <modus>
```
`0` = ESP sendet `0xA5`-Rahmen, `1` = ESP sendet `0xA6`-Rahmen mit
Sendezeitstempel. Kommt als Antwort auf `0xB2` und außerdem unaufgefordert,
wenn jemand den Modus auf der USB-Konsole mit `ts0`/`ts1` umstellt.

---

## 5. Zeitsynchronisation

Ziel: zu jedem Paket vom ESP wissen, **wann es losgeschickt wurde** — ausgedrückt
in der Uhr des Jetson, damit sich Ereignisse mit Kamera-, LiDAR- und ROS-Daten
zusammenlegen lassen.

Das zerfällt in zwei unabhängige Teile:

1. **Der Stempel.** Jedes Paket sagt selbst, wann es rausging — das ist der
   `0xA6`-Rahmen aus Abschnitt 2. Der Wert steht in der ESP-Uhr.
2. **Der Uhrenversatz.** Ein Ping-Pong (`0xB0`/`0xB1`) misst, wie weit ESP-Uhr
   und Jetson-Uhr auseinanderliegen. Damit wird aus dem Stempel ein Zeitpunkt
   in der Jetson-Uhr.

### 5.1 Die beiden Uhren

| | ESP | Jetson |
|---|---|---|
| Quelle | `esp_timer_get_time()` | `time.monotonic()` (= `CLOCK_MONOTONIC`) |
| Auflösung | 1 µs | 1 ns (praktisch µs) |
| Nullpunkt | Boot des ESP | Boot des Jetson |
| Überlauf | keiner (`int64`) | keiner |

**Für die Synchronisation muss der Jetson eine monotone Uhr nehmen, nicht
`time.time()`.** Ein NTP-Sprung würde sonst mitten in der Messung den Versatz
verschieben. Der Bezug zur Wanduhr wird erst ganz am Ende hergestellt, mit einem
einmal gemessenen Abstand `CLOCK_REALTIME − CLOCK_MONOTONIC`.

### 5.2 Der Ablauf

```
Jetson                                    ESP
  |                                        |
  |--- A5 B0 <seq> -------------->         |
  |    t1 = letztes Byte raus              |
  |                                     t2 = letztes Byte rein
  |                                        |
  |         <---- A5 B1 seq t2 t3 ---------|
  |    t4 = letztes Byte rein           t3 = letztes Byte raus
```

Alle vier Zeitpunkte meinen **das letzte Byte des jeweiligen Rahmens auf der
Leitung**. Das ist der einzige Bezugspunkt, den beide Seiten sauber treffen
können, und er macht die Rechnung symmetrisch — sonst stünde die
Übertragungsdauer der 20-Byte-Antwort gegen die der 3-Byte-Anfrage und
verfälschte den Versatz um ~0,7 ms.

Wie beide Seiten diesen Punkt treffen:

* **`t1`**: Puffer leerlaufen lassen, **dann** die Uhr nehmen, **dann**
  schreiben — und die Übertragungsdauer des Rahmens (10 Bit je Byte bei 8N1)
  dazurechnen. Also *gerechnet*, nicht gemessen.

  > **Nicht** `write()` und danach `flush()` messen. Das sieht sauberer aus,
  > taugt aber nicht: `flush()` läuft auf `tcdrain()` hinaus, und das kehrt je
  > nach Treiber — auf dem Tegra-UART des Jetson zuverlässig — deutlich später
  > zurück als das letzte Byte die Leitung verlässt. `t1` wird dadurch zu spät,
  > im Extremfall später als `t4`, und der Umlauf rechnerisch **negativ**.

* **`t2`**: der ESP nimmt die Uhr, sobald das Paket vollständig ist.
* **`t3`**: der ESP nimmt die Uhr unmittelbar vor dem Schreiben und **addiert
  die Übertragungsdauer des Rahmens**. Vorher lässt er den Sendepuffer
  leerlaufen, damit die Rechnung stimmt. Dieselbe Methode wie bei `t1`.
* **`t4`**: die Bridge nimmt die Uhr, wenn das letzte Byte des Rahmens gelesen
  ist — also nach dem Zusammensetzen des Pakets, nicht beim Startbyte.

  > **Blockweises Lesen zerstört `t4`.** `ser.read(64)` kehrt erst zurück, wenn
  > 64 Byte beisammen sind oder der Timeout abläuft; alle Pakete in diesem Block
  > bekommen dann denselben Empfangszeitpunkt, nämlich den des letzten. Bei
  > laufender Telemetrie sind das leicht 150 ms Fehler. Richtig ist
  > `read(1)` — das blockiert nur bis zum ersten Byte — und danach
  > `read(in_waiting)` für den Rest ohne weiteres Warten.

### 5.3 Die Rechnung

Die Standard-NTP-Formeln:

```
versatz  = ((t2 - t1) + (t3 - t4)) / 2       # ESP-Uhr minus Jetson-Uhr
umlauf   = (t4 - t1) - (t3 - t2)             # reine Leitungs- + Wartezeit
```

Damit:

```
jetson_zeit = esp_stempel - versatz
```

Der Versatz stimmt nur, wenn Hin- und Rückweg gleich lang sind. Sie sind es im
Mittel, aber nicht in jeder einzelnen Runde — der ESP liest seinen UART aus
`loop()` heraus, `t2` kommt also je nach Schleifendurchlauf verspätet.

**Deshalb: mehrere Runden messen und die mit dem kleinsten `umlauf` nehmen.**
Jede Verzögerung, die die Symmetrie stört, verlängert auch den Umlauf; die
schnellste Runde ist damit automatisch die ehrlichste. Acht bis sechzehn Runden
im Abstand von ~20 ms reichen.

Zwei Fallen bei diesem Filter, beide schon einmal zugeschlagen:

* **Ein negativer Umlauf ist physikalisch unmöglich** und heißt, dass eine der
  vier Zeitmessungen falsch war. Solche Runden müssen **verworfen** werden —
  sonst sucht `min()` sich ausgerechnet die kaputteste als „beste" heraus.
* **Die Schranke muss additiv sein** (`bester + 2 ms`), nicht multiplikativ
  (`bester × 2`). Bei einem negativen Bestwert wird die multiplikative Schranke
  *kleiner* als der Bestwert, nichts kommt durch, und ein Fallback auf „dann
  eben alle" schaufelt den Müll erst recht in die Schätzung.

Bleibt nichts Brauchbares übrig, ist der Versatz **ungültig** — und die Bridge
stempelt ehrlich mit der Lesezeit, statt eine falsche Sendezeit zu erfinden.

### 5.4 Drift

Beide Uhren hängen an eigenen Quarzen, typisch ±20…50 ppm. Gegeneinander sind
das bis zu 100 ppm, also **0,1 ms Abweichung pro Sekunde** — nach zehn Minuten
60 ms.

Zwei Möglichkeiten, das Übliche zuerst:

* **Nachsynchronisieren.** Alle 10 s eine Messrunde. Der Versatz bleibt dann
  unter ~1 ms. Kostet 20 Byte pro Runde, also nichts.
* **Drift schätzen.** Über die letzten Messpunkte eine Gerade
  `versatz(t) = a + b·t` legen (kleinste Quadrate). `b` ist die relative
  Gangabweichung. Damit bleibt der Fehler auch zwischen den Runden klein und der
  Sprung beim Nachsynchronisieren verschwindet. Lohnt sich, wenn Zeitstempel zu
  Bilddaten passen müssen.

Beim Nachsynchronisieren nie hart auf den neuen Wert springen, sondern
einschleifen (`versatz = 0,8·alt + 0,2·neu`) — ein Sprung bringt sonst die
Reihenfolge bereits abgelegter Ereignisse durcheinander.

### 5.5 Der 32-Bit-Stempel im Rahmen

`0xA6` überträgt nur die unteren 32 Bit der ESP-Uhr. Das läuft alle
**71,6 Minuten** über. `0xB1` liefert dagegen den vollen `int64`. Auspacken:

```python
def unwrap(stamp32: int, letzter_voller_wert: int) -> int:
    grob = (letzter_voller_wert & ~0xFFFFFFFF) | stamp32
    for kandidat in (grob - 2**32, grob, grob + 2**32):
        if abs(kandidat - letzter_voller_wert) < 2**31:
            return kandidat
    return grob
```

Solange die Bridge mindestens alle 35 Minuten eine Runde `0xB0` fährt (bei 10 s
Takt garantiert), ist das eindeutig.

### 5.6 Was der Stempel *nicht* sagt

Der Stempel ist der **Sendezeitpunkt des Pakets**, nicht der Zeitpunkt des
Ereignisses. Zwischen beiden liegt bei manchen Paketen etwas:

| Paket | Abstand Ereignis → Senden |
|---|---|
| `0x93` MOVE_DONE | bis zu ein Reglertakt (10 ms) plus ein `loop()`-Durchlauf — der Regler läuft auf Core 0 und reicht das Ergebnis über eine Queue an Core 1 |
| `0x70` BUTTON | die 200 ms Entprellzeit liegen **vor** dem Ereignis, danach ein `loop()`-Durchlauf |
| `0xA1`/`0xA2` Akku | der Messwert ist bis zu 15 s alt (Messraster), der Stempel ist trotzdem taufrisch |
| `0x42`, `0x82`, `0x94` | Antworten, direkt im Anschluss an den Befehl erzeugt — Abstand vernachlässigbar |

Für Latenzmessungen des Links ist der Stempel exakt richtig. Für „wann wurde der
Taster gedrückt" ist er eine obere Schranke.

### 5.7 Fehlerbudget

| Quelle | Größenordnung | Gegenmittel |
|---|---|---|
| `loop()`-Verzögerung beim Lesen auf dem ESP | 0,1…5 ms, stark schwankend | kleinsten Umlauf aus N Runden nehmen |
| `t4` in Python (Scheduling, `select`) | 0,1…1 ms | dito |
| Übertragungsdauer der Rahmen | 0,26 / 1,74 ms | ist beidseitig eingerechnet |
| Quarzdrift | 0,1 ms/s | alle 10 s nachsynchronisieren |
| `tcdrain()` kehrt spät zurück (Tegra) | bis mehrere ms, kann `umlauf` negativ machen | `t1` rechnen statt messen (5.2) |
| Blockweises Lesen für `t4` | bis zum Lese-Timeout, also ~50…150 ms | `read(1)` + `read(in_waiting)` (5.2) |
| Debug-Ausgaben auf der USB-Konsole | bis mehrere ms | beim Messen `dbg0` setzen |

Realistisch sind damit **±1 ms** ohne besonderen Aufwand und **±0,2 ms** mit
Driftschätzung und ruhigem Link. Wer besser braucht, kommt um eine
Hardwareleitung (PPS-Puls vom Jetson auf einen ESP-Interrupt) nicht herum.

### 5.8 Reihenfolge beim Verbindungsaufbau

1. Port öffnen, Lese-Thread starten.
2. 8–16 Runden `0xB0` → erster Versatz.
3. `A5 B2 01` senden, auf `0xB3` warten. Ab jetzt kommen `0xA6`-Rahmen.
4. Im Betrieb alle 10 s eine Runde `0xB0` nachschieben.

Punkt 2 vor Punkt 3: ohne Versatz ist ein Stempel wertlos, und `0xB1` braucht
den Stempelmodus nicht.

### 5.9 Referenzimplementierung

`docs/timesync_jetson.py` enthält die Jetson-Seite als eigenständige Klasse:
Ping-Pong, Minimum-Filter, Driftschätzung, Unwrapping und das Parsen beider
Rahmenformen. Ohne Hardware testbar:

```
python3 docs/timesync_jetson.py --selftest
```

Benutzt wird sie von `src/esp_serial_bridge.py` — siehe Abschnitt 6.

---

## 6. ROS-2-Bridge

`src/esp_serial_bridge.py` ist die fertige Gegenstelle: `EspLink` macht das
Protokoll (ohne ROS-Abhängigkeit, damit ohne Roboter testbar), `EspBridgeNode`
hängt es an ROS. Selbsttest ohne Hardware und ohne ROS:

```
python3 src/esp_serial_bridge.py --selftest
```

**Jede Nachricht mit `header.stamp` trägt den Sendezeitpunkt des ESP**,
umgerechnet in die ROS-Uhr. Die Laufzeit wird dafür in der monotonen Uhr
gemessen und von der ROS-Zeit des Lesens abgezogen — das bleibt auch unter
`use_sim_time` richtig.

### Was hineingeht

| Topic | Typ | Wirkung |
|---|---|---|
| `/cmd_vel` | `geometry_msgs/Twist` | `linear.x` → Motor, `angular.z` → Lenkung |
| `~/motor` | `std_msgs/Int32` | roher Duty, −1023…+1023 |
| `~/steer` | `std_msgs/Float32` | −100…+100 |
| `~/move` | `std_msgs/Float32` | Grad, **relativ** zur jetzigen Stellung |
| `~/led` | `std_msgs/Bool` | |
| `~/trim` | `std_msgs/Int32` | 0 = links, 1 = rechts, 2 = speichern |
| `~/emergency` | `std_msgs/Empty` | Nothalt |
| `~/pid_set` | `std_msgs/Float32MultiArray` | `[id, wert]` oder alle neun Werte |
| `~/cal` | `std_msgs/Int32MultiArray` | `[aktion, arg]` |
| `~/cal_action` | `std_msgs/String` | dasselbe im Klartext: `plus`, `left`, `save` … |

### Was herauskommt

| Topic | Typ | Inhalt |
|---|---|---|
| `~/button` | `std_msgs/Header` | Tastendruck — der Inhalt *ist* der Zeitstempel |
| `~/joint_states` | `sensor_msgs/JointState` | Stellung in rad, aus `0xC1` zusätzlich `velocity` in rad/s |
| `~/speed` | `std_msgs/Float32` | Drehgeschwindigkeit in °/s, **vorzeichenbehaftet** |
| `~/motor_state` | `std_msgs/Float32MultiArray` | `[duty, ampere]` — Duty signed, Strom nicht |
| `~/move_done` | `std_msgs/Int32MultiArray` | `[move_id, status, zehntelgrad]` |
| `~/move_progress` | `std_msgs/Float32` | 0…100 % |
| `~/battery` | `sensor_msgs/BatteryState` | Pack- und Zellspannung |
| `~/battery_low` | `std_msgs/Bool` | Unterspannungswarnung (latched) |
| `~/cal_state` | `std_msgs/Int32MultiArray` | Zustand der Kalibrierung |
| `~/pid` | `std_msgs/Float32MultiArray` | `[kp, ki, kd]` (latched) |
| `~/console` | `std_msgs/String` | ASCII-Zeilen des ESP |
| `/diagnostics` | `diagnostic_msgs/DiagnosticArray` | Uhrenversatz, Drift, Umlauf, Zähler |
| `~/latency_ms` | `std_msgs/Float32` | Laufzeit *dieses* Pakets: abgeschickt → gelesen |
| `~/rtt_ms` | `std_msgs/Float32` | kürzester Umlauf der letzten Abgleichrunde, 1 Hz |
| `~/offset_ms` | `std_msgs/Float64` | Uhrenversatz, 1 Hz |
| `~/drift_ppm` | `std_msgs/Float32` | geschätzter Quarzdrift, 1 Hz |

Die letzten vier tragen dieselben Zahlen wie `/diagnostics`, nur als Zahl statt
als Text — `DiagnosticArray` speichert seine Werte als Strings, und ein String
lässt sich nicht plotten. `~/offset_ms` ist `Float64`, weil der Versatz mehrere
Millionen Millisekunden groß wird; `float32` hätte dort nur noch 1-ms-Schritte
und der Drift verschwände im Rauschen.

### Services

Alles, was eine Quittung hat, ist ein Service statt eines Topics — sonst
erfährt der Aufrufer nie, ob es geklappt hat.

| Service | Typ |
|---|---|
| `~/emergency_stop`, `~/move_abort` | `std_srvs/Trigger` |
| `~/pid_get`, `~/pid_save` | `std_srvs/Trigger` |
| `~/calibrate_start`, `~/calibrate_save` | `std_srvs/Trigger` |
| `~/trim_save`, `~/torque_report`, `~/resync` | `std_srvs/Trigger` |
| `~/set_led`, `~/set_stamp_mode`, `~/servo_torque_free` | `std_srvs/SetBool` |

### Parameter

`port`, `baud`, `servo_id`, `stamp_mode`, `sync_rounds`, `sync_interval`,
`heartbeat_period`, `cmd_vel_timeout`, `battery_period`, `progress_period`,
`telemetry_period`, `max_linear`, `max_angular`.

`telemetry_period` (Default 0,05 s = 20 Hz) ist der Takt, in dem der ESP
Position und Geschwindigkeit von sich aus schickt. Zur Laufzeit änderbar:

```
ros2 param set /esp_serial_bridge telemetry_period 0.1
```

`0` schaltet die Telemetrie ab. Kleiner als 0,02 s nimmt der ESP nicht an —
die Bridge meldet das als Warnung, statt es stillschweigend zu schlucken.

> **`max_linear` und `max_angular` musst du ausmessen.** Der ESP regelt die
> Drehzahl nicht — `/cmd_vel` wird geradeaus auf PWM umgerechnet. Die beiden
> Werte sagen, welche Geschwindigkeit bzw. Drehrate voller Ausschlag bedeutet.

### `/diagnostics` lesen

| Schlüssel | gesund | Alarmzeichen |
|---|---|---|
| `versatz_ms` | beliebig groß, auch mehrere Stunden | `-` (kein Abgleich) |
| `umlauf_ms` | 0,3…3 ms | **negativ** oder > 50 ms |
| `drift_ppm` | −100…+100 | dreistellig |
| `sync_verworfen` | `0` | > 0 (unbrauchbare Messrunden) |
| `zeitstempel` | `an` | `aus` |
| `esp_neustarts` | konstant | steigt im Betrieb |

> **Ein riesiger `versatz_ms` ist normal und kein Fehler.** Die ESP-Uhr zählt
> ab seinem Boot, die Jetson-Uhr ab dessen Boot. Läuft der Jetson seit zwei
> Stunden und der ESP seit fünf Minuten, sind das rund −7 000 000 ms. Genau
> diesen Abstand zu kennen ist der ganze Zweck der Übung.
>
> **Ein negativer `umlauf_ms` dagegen ist immer ein Fehler** — siehe 5.2 und
> 5.3.

### Fertiges Foxglove-Layout

`docs/foxglove_esp_bridge.json` zeigt alle Topics und Services der Bridge.
In Foxglove laden: **Layout → Import from file…**

| Reiter | Inhalt |
|---|---|
| Fahren | Teleop auf `/cmd_vel`, Nothalt, Positionsfahrt, Tempo/Duty, Stellung, Strom |
| Zeit & Link | Latenz, Umlauf, Drift, Versatz, Knopf zum Neuabgleich |
| Akku & Zustand | Pack- und Zellspannung, Unterspannungslampe, Fahrfortschritt, Tastendruck |
| Diagnose & Konsole | `/diagnostics` als Tabelle, ESP-Konsole, `/rosout` |
| Kalibrieren | Lenkung einmessen, Servo freigeben, speichern |
| Service | Nothalt, Fahrt abbrechen, PID lesen/speichern, Zeitstempel, LED, Servolast |
| Befehle | die Roh-Topics `motor`, `steer`, `trim`, `led`, `pid_set`, `cal` |

Zwei Stellen, die je nach Aufbau angepasst werden müssen:

* Das Diagnose-Panel ist auf `hardware_id = /dev/ttyTHS1` eingestellt — das
  ist der Parameter `port`. Bei einem anderen Port das Panel neu auswählen.
* Alle Pfade beginnen mit `/esp_serial_bridge/`. Läuft die Node unter einem
  anderen Namen oder in einem Namespace, in der Datei einmal ersetzen.

### Latenz in Foxglove anzeigen

**Plot-Panel → Pfad eintragen.** Die Pfade sind Topic plus Feldname:

| Was | Pfad |
|---|---|
| Laufzeit je Paket | `/esp_serial_bridge/latency_ms.data` |
| Umlauf (Link-Gesundheit) | `/esp_serial_bridge/rtt_ms.data` |
| Drift | `/esp_serial_bridge/drift_ppm.data` |

X-Achse auf **timestamp** stellen, nicht auf *index* — sonst zeigt der Graph
die Nachrichtennummer statt der Zeit.

`latency_ms` kommt so oft, wie gestempelte Pakete eintreffen — bei 50-ms-Telemetrie
also 20-mal pro Sekunde. Die anderen drei kommen im Sekundentakt.

> **Warum nicht direkt aus `/diagnostics` plotten?** Foxglove hat dafür das
> Diagnostics-Panel, das die Werte als Tabelle zeigt. Plotten kann es sie nicht:
> in `DiagnosticArray` steht jeder Wert als Text, und `"0.601"` ist für das
> Plot-Panel kein Zahlenwert. Deshalb die eigenen Topics oben.

Ein zweiter Weg, der ganz ohne Zusatz-Topics auskommt: das Plot-Panel kann als
X-Achse **receive time** und als Y-Achse `header.stamp` eines gestempelten
Topics zeichnen — zum Beispiel `/esp_serial_bridge/joint_states`. Der Abstand
zwischen beiden *ist* die Latenz. Ablesen lässt sich das aber nur grob, weil
Foxglove die Differenz nicht selbst bildet.

### Zwei Dinge, die die Bridge von selbst tut

* **Heartbeat.** Der ESP lässt den Motor auslaufen, wenn 5 s lang kein Befehl
  kommt. Der letzte Motorbefehl geht deshalb zyklisch neu raus — während einer
  Positionsfahrt bewusst nicht, die darf länger dauern.
* **Watchdog auf `/cmd_vel`.** Bleibt es länger als `cmd_vel_timeout` aus,
  geht der Motor in Coast. Sonst würde eine abgestürzte Steuerung den Roboter
  weiterfahren lassen.

---

## 7. Bekannte Lücken

Bewusst offen gelassen, für die Bridge relevant:

* **Keine Prüfsumme, kein Längenfeld.** Ein verlorenes Byte kaskadiert bis zum
  nächsten `0xA5`. Der 100-ms-Timeout des ESP fängt das ab, die Bridge braucht
  eine entsprechende Absicherung.
* **Keine Quittung für `0x10`, `0x20`, `0x30`, `0x60`, `0x80`.** Fire-and-forget.
  Ob ein PID-Wert ankam, lässt sich nur per `0x81` gegenprüfen.
* **`0x80` allein ist flüchtig.** Ohne anschließendes `0x83` sind die Werte nach
  dem nächsten Reset wieder auf dem gespeicherten Stand.
* **Kein Boot-/Ready-Paket.** Die Bridge merkt einen ESP-Reset nicht direkt. Wer
  das braucht, erkennt es an der Startup-ASCII-Zeile `System Ready. ...` oder
  fragt zyklisch `0x81` ab. Seit der Zeitsynchronisation gibt es einen zweiten,
  eindeutigen Weg: **springt `t_rx_us` in `0xB1` zurück, hat der ESP neu
  gebootet** — die Uhr läuft ab Boot. Dann Versatz und Driftschätzung wegwerfen
  und neu messen, sonst datiert die Bridge alles um die alte Laufzeit falsch.
* **Der Stempel ist der Sendezeitpunkt, nicht der Ereigniszeitpunkt.** Wie weit
  beides auseinanderliegt, steht in Abschnitt 5.6. Für `0x93` MOVE_DONE sind es
  bis zu 10 ms.
* **Die Positions-Telemetrie ist flüchtig.** Fahrbefehle sind relativ und daher
  reset-fest, aber die in `0x93`/`0x94` gemeldeten Absolutwerte beginnen nach
  jedem ESP-Reset wieder bei 0. Wer Wegstrecke über Neustarts hinweg mitzählt,
  muss das auf der Jetson-Seite tun.
* **`0x50` TORQUE antwortet nicht** über UART — das Ergebnis geht nur auf die
  USB-Konsole. (`0x40`/`0x41` antworten seit der manuellen Kalibrierung mit
  `0x42`.)
* **Die Kalibrierung ist ein Handbetrieb.** `0x41` bewegt den Servo pro Paket um
  genau einen Schritt; die Bridge muss die Schritte einzeln absetzen und dem
  Bediener die `0x42`-Rückmeldung zeigen. Es gibt bewusst keine Aktion, die
  selbsttätig bis zum Anschlag fährt.

---

## 8. Mitschnitt auf der ESP-Seite (Bring-up)

Der ESP protokolliert den Jetson-Link auf seiner **USB-Konsole** (separate
Schnittstelle, 115200). Beim Entwickeln der Bridge ist das die schnellste
Antwort auf „kommt überhaupt etwas an?".

```
[RX] MOVE          id=3 ziel=90.5 grad
[TX] MOVE_DONE 03 00 00 00 03 84
[RX] UNBEKANNT cmd=0x55 - verworfen
[RX] ABBRUCH cmd=0x10 MOTOR nach 2/3 Byte (>100 ms Pause) - resync
```

Steuerung über die USB-Konsole:

| Befehl | Wirkung |
|---|---|
| `dbg` | Statistik: Pakete, unbekannte CMDs, Abbrüche, Streubytes, Uhr, Sync-Zähler |
| `dbg0` | Mitschnitt aus |
| `dbg1` | dekodierte Pakete (Default) |
| `dbg2` | zusätzlich alle Rohbytes inkl. Sync-Suche |
| `ts` | Stand der ESP-Uhr und Stempelmodus |
| `ts0` / `ts1` | Sendezeitstempel (`0xA6`-Rahmen) aus / an |
| `tel` | Fahrzustand einmal anzeigen |
| `tel<ms>` | Telemetrietakt setzen, `tel0` = aus |

`ts0`/`ts1` schicken dem Jetson unaufgefordert ein `0xB3` — die Bridge bekommt
die Umstellung also mit, auch wenn sie von der Konsole kam.

Gestempelte Pakete stehen im Mitschnitt mit ihrer Rohzeit:

```
[TX] MOVE_DONE t=45120833 03 00 00 00 03 84
[RX] TIME_SYNC   seq=7
[TX] TIME_RSP 07 00 00 00 00 02 B0 4C 21 00 00 00 00 02 B0 4E 95
```

Dieselbe Kalibrierung lässt sich ohne Jetson direkt auf der USB-Konsole fahren —
nützlich zum Gegenprüfen, wenn die Bridge sich anders verhält als erwartet:

| Befehl | Wirkung |
|---|---|
| `cal` (oder `x`) | starten; bei laufender Kalibrierung Status anzeigen |
| `+` / `-` | ein Schritt, `+50` für einmalig 50 Ticks |
| `caln<t>` | Schrittweite in Ticks |
| `calm` / `call` / `calr` | Mitte / linken / rechten Anschlag merken |
| `calfree` / `calhold` | Torque aus (von Hand stellen) / wieder halten |
| `calgo` | gemerkte Mitte anfahren |
| `calsave` / `calq` | speichern / abbrechen |

Beim Deuten hilft:

* **`ABBRUCH ... nach n/m Byte`** — die Bridge hat ein Paket in mehreren
  `write()`-Aufrufen mit Pause geschickt. Ein Paket muss in einem Rutsch raus.
* **`UNBEKANNT cmd=0x??`** — Sync-Verlust oder falsches CMD-Byte. Wenn das
  Byte plausibel wie Nutzlast aussieht, stimmt vermutlich eine Payload-Länge
  in der Tabelle der Bridge nicht.
* **Viele Streubytes bei `dbg2`, davon ASCII** — normal, das sind die
  Klartextzeilen des ESP (siehe Abschnitt 2, Regel 4).
* **`dbg` zeigt 0 Pakete und 0 Streubytes** — es kommt physisch nichts an.
  Verkabelung (RX/TX gekreuzt?), gemeinsame Masse, Baudrate prüfen.

> `dbg1` schreibt eine Zeile pro Paket. Ein Motor-Heartbeat im 100-ms-Takt
> erzeugt damit 10 Zeilen/s. Für Dauerfahrten `dbg0` setzen.

---

## 9. Referenzwerte

| | |
|---|---|
| Encoder | 408 Counts pro Umdrehung der Ausgangswelle (4× Quadratur) |
| PWM | 20 kHz, 10 Bit (0..1023) |
| Motortreiber | VNH5019 (INA/INB/PWM) |
| Regler-Takt | 100 Hz (10 ms) auf Core 0 |
| Anfahrrampe | max. 25 Duty-Stufen pro 10 ms ⇒ ~410 ms auf Vollgas |
| Akku | 4S, Warnung < 3,80 V/Zelle, Entwarnung > 3,85 V/Zelle |
| Lenkung | SCServo SCS/SCSCL, ID 1, Hub auf 80 % begrenzt |
| ESP-Uhr | `esp_timer`, µs seit Boot, `int64` (kein Überlauf) |
| Rahmenstempel | untere 32 Bit davon ⇒ Überlauf alle 71,6 min |
| Byte auf der Leitung | 10 Bit bei 8N1 ⇒ 86,8 µs bei 115200 |
| Sync-Runde | 3 B hin + 20 B zurück ⇒ 2,0 ms reine Übertragung |
| Drehzahlmessung | alle 100 ms, exponentiell geglättet (α = 0,30) |
| Telemetrie bei 20 Hz | 18 B je Paket ⇒ 360 B/s, ~3 % der Leitung |
| erreichbare Genauigkeit | ±1 ms einfach, ±0,2 ms mit Driftschätzung |

### Gemessen auf dem Jetson (Referenz zum Vergleichen)

Jetson Orin, `/dev/ttyTHS1`, Telemetrie mit 20 Hz, ESP im Leerlauf:

| | gemessen | Alarmschwelle |
|---|---|---|
| `umlauf_ms` | 0,60 | negativ oder > 50 |
| `drift_ppm` | −57 | dreistellig |
| `sync_verworfen` | 0 | > 0 |
| daraus Stempelgenauigkeit | ~±0,3 ms | |

Weicht dein Link davon deutlich ab, stimmt etwas nicht — die üblichen
Ursachen stehen in 5.2 und im Fehlerbudget 5.7.

---

## 10. Prompt für den Jetson-Chat

> **Weitgehend erledigt:** `src/esp_serial_bridge.py` ist die Bridge, Abschnitt 6
> beschreibt sie. Dieser Prompt bleibt als Beschreibung der Anforderungen
> stehen — nützlich, wenn die Bridge einmal neu aufgesetzt oder gegengeprüft
> werden soll.

> Ich habe einen ESP32-S3, der über UART (115200 8N1) einen Fahrmotor, eine
> Servolenkung und die Akkuüberwachung eines Roboters steuert. Das Protokoll ist
> in der beigefügten Spezifikation vollständig beschrieben. Ich brauche auf dem
> Jetson eine Python-Bridge dagegen.
>
> **Was sich gegenüber der bisherigen Bridge geändert hat und deshalb umgeschrieben
> werden muss:**
>
> 1. **Der Empfangspfad wird asynchron.** Bisher kam vom ESP praktisch nur das
>    Button-Event. Jetzt schickt er auch unaufgefordert `0x93` MOVE_DONE und
>    `0xA2` BATTERY_WARN. Ein Request-Response-Modell reicht nicht mehr — es
>    braucht einen dauerhaft lesenden Thread, der Pakete in eine Queue legt und
>    per Callback verteilt.
>
> 2. **Der RX-Parser muss variable Paketlängen können.** Vorher feste 3 Byte,
>    jetzt 1 bis 12 Byte Nutzlast je nach CMD. Die Längentabelle aus der Spec
>    fest verdrahten, unbekannte CMDs verwerfen und zum nächsten `0xA5` resynchen.
>    Wichtig: der Strom enthält zusätzlich ASCII-Statuszeilen vom ESP, die
>    übersprungen (und am besten geloggt) werden müssen.
>
> 3. **Positionsfahrten sind relativ und brauchen ID-Tracking.** `0x90` bekommt eine `moveId`
>    mitgegeben, das Ergebnis kommt irgendwann später als `0x93` mit derselben ID
>    und einem Status (OK / Timeout / Aborted). Bau das als `asyncio.Future` oder
>    Callback pro ID, sodass aufrufender Code auf eine bestimmte Fahrt warten
>    kann. Es kann immer nur eine Fahrt gleichzeitig laufen; eine neue löst die
>    alte ab und die alte quittiert mit Status `0x02`.
>
> 4. **Ein Heartbeat ist neu und zwingend.** Der ESP stoppt den Motor
>    selbstständig, wenn 5 s lang kein Befehl kommt. Für Dauerfahrt muss die
>    Bridge `0x10` MOTOR zyklisch nachschicken (alle 100–500 ms). Positionsfahrten
>    sind ausgenommen, dort darf der Heartbeat pausieren.
>
> 5. **PID-Parameter sind zweistufig.** `0x80` PID_SET ändert nur den laufenden
>    Wert, erst `0x83` PID_SAVE schreibt den ganzen Satz ins NVS und quittiert mit
>    `0x84`. Beim Tuning also viele `0x80` und am Ende genau ein `0x83` — nicht
>    nach jedem Parameter speichern.
>
> 6. **Alle Fließkommawerte gehen als `int32 × 1000` über die Leitung, Big-Endian
>    — auch die ganzzahligen PID-Parameter.** `maxDuty = 700` wird als `700000`
>    kodiert. Das ist die wahrscheinlichste Fehlerquelle, bau dafür Unit-Tests.
>
> 7. **Akkuwarnungen kommen von selbst.** Kein Polling nötig. Der ESP schaltet bei
>    Unterspannung nichts ab — die Reaktion (Fahrt stoppen, zur Ladestation,
>    Alarm) muss auf der Jetson-Seite passieren.
>
> 8. **Es gibt eine Zeitsynchronisation.** Der ESP kann jedes Paket mit seinem
>    Sendezeitpunkt stempeln (`0xA6`-Rahmen, mit `0xB2` einzuschalten), und ein
>    Ping-Pong `0xB0`/`0xB1` liefert den Uhrenversatz zwischen ESP und Jetson.
>    Damit bekommt jedes Ereignis einen Zeitpunkt in der Jetson-Uhr, der sich
>    mit Kamera- und LiDAR-Daten zusammenlegen lässt. Die Rechnung, das
>    Fehlerbudget und eine fertige, ohne Hardware testbare Implementierung
>    stehen in Abschnitt 5 bzw. in `docs/timesync_jetson.py` — **übernimm die
>    von dort, statt die NTP-Formeln neu herzuleiten.** Der RX-Parser muss
>    dafür beide Startbytes können und alle vier Zeitpunkte auf das *letzte*
>    Byte des jeweiligen Rahmens beziehen (`flush()` nach dem Senden!).
>
> Bau die Bridge als Klasse mit sauber getrennten Methoden pro Befehl,
> Typannotationen und einem Kontextmanager fürs Öffnen/Schließen des Ports.
> Serialisierung und Parsing sollen ohne echte Hardware testbar sein.
