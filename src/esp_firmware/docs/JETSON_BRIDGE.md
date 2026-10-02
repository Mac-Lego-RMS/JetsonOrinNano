# ESP32-S3 Controller — Protocol Specification for the Jetson Bridge

Reference for the counterpart on the Jetson. The source of truth is
`src/main.cpp` on the ESP.

---

## 1. Physical connection

| | |
|---|---|
| Interface | UART, `Serial1` on the ESP |
| Baud rate | **115200**, 8N1 |
| ESP RX | GPIO 10 (→ Jetson TX) |
| ESP TX | GPIO 11 (→ Jetson RX) |
| Flow control | none |

---

## 2. Frame format

```
+------------+---------+-----------------------+
| 0xA5       | CMD     | PAYLOAD (0..17 bytes) |
| start byte | 1 byte  | length is implied     |
|            |         | by CMD                |
+------------+---------+-----------------------+
```

**There is no length field and no checksum.** Both sides must have the
payload length per command hard-coded (tables below).

### Second frame form: with send timestamp

```
+------------+---------+--------------------+-----------------------+
| 0xA6       | CMD     | uint32 t_tx_us     | PAYLOAD (0..12 bytes) |
| start byte | 1 byte  | send time          | as for 0xA5           |
+------------+---------+--------------------+-----------------------+
```

A `0xA6` frame is a `0xA5` frame with four extra bytes between `CMD` and the
payload: the lower 32 bits of the ESP clock in microseconds. The payload and
its length are unchanged — the parser needs only one more branch.

* **The ESP always understands `0xA6` on receive.** The bridge may stamp its
  commands at any time, without switching anything on first.
* **The ESP only *sends* `0xA6` when it has been asked to** — with
  `0xB2 STAMP_MODE`. The default is off, so that a bridge that does not know
  `0xA6` keeps working.
* Exception: `0xB1 TIME_RSP` is **never** sent as `0xA6`. It already carries
  its timestamps at full 64-bit width in the payload.

Details on the meaning of the stamp and on converting it to the Jetson clock:
section 5.

### Rules the parser must follow

1. **Multi-byte numbers are big-endian** (MSB first). This applies to `int16` and `int32`.
2. **Floating-point numbers are transmitted as `int32 × 1000`.** Kp = 4.25 → `4250`.
   No IEEE754 on the wire.
3. **A packet must go out in a single `write()`.** The ESP resets its parser
   if more than **100 ms** pass between two bytes of a packet. Sending byte by
   byte with pauses breaks the packet apart.
4. **The RX stream from the ESP also contains plain ASCII text.** On
   `CMD_EMERGENCY` and `CMD_TRIM` the ESP also writes readable status lines on
   the same line. This is harmless — ASCII never contains `0xA5`, and the sync
   scan skips it — but the bridge must tolerate these bytes instead of failing
   on them. Sensible: print them as a log line.
5. **The ESP discards unknown CMD bytes** and searches for the next `0xA5`. The
   bridge should do the same.

---

## 3. Jetson → ESP

| CMD | Name | Payload | Description |
|---|---|---|---|
| `0x10` | MOTOR | 3 B | `dir(1)` + `uint16 speed` |
| `0x20` | SERVO | 3 B | `id(1)` + `int16 steering` |
| `0x30` | LED | 1 B | `0`=off, `≠0`=on |
| `0x40` | CALIBRATE | 0 B | start manual calibration → response `0x42` |
| `0x41` | CAL | 2 B | `action(1)` + `arg(1)` → response `0x42` |
| `0x50` | TORQUE | 0 B | print servo load on the USB console |
| `0x60` | TRIM | 1 B | `0`=left, `1`=right, `2`=save |
| `0x80` | PID_SET | 5 B | `paramId(1)` + `int32 value×1000` |
| `0x81` | PID_GET | 0 B | → response `0x82` |
| `0x83` | PID_SAVE | 0 B | parameters to NVS → response `0x84` |
| `0x90` | MOVE | 5 B | `moveId(1)` + `int32 distance` in 1/10° (relative) |
| `0x91` | MOVE_ABORT | 0 B | abort the current move |
| `0x92` | PROGRESS | 0 B | → response `0x94` |
| `0xA0` | BATTERY | 0 B | → response `0xA1` |
| `0xB0` | TIME_SYNC | 1 B | `seq(1)` → response `0xB1` |
| `0xB2` | STAMP_MODE | 1 B | `0`=off, `1`=on → response `0xB3` |
| `0xC0` | TELEM_RATE | 2 B | `uint16` period in ms, `0`=off → response `0xC1` |
| `0xFF` | EMERGENCY | 0 B | emergency stop, active brake |

### `0x10` MOTOR — open-loop control

```
A5 10 <dir> <speedHi> <speedLo>
```

* `dir`: `0` = forward, `1` = reverse
* `speed`: `0..1023` (10-bit PWM)
* **`speed = 0` means coasting, not braking.** The only way to brake actively
  is `0xFF`.
* Aborts a running position move → a `0x93` with status `0x02` follows.

> **Heartbeat required:** If no command arrives for 5 seconds, the motor goes
> into coast on its own. For continuous driving the bridge must resend `0x10`
> regularly (recommended: every 100–500 ms). **Position moves are exempt from
> this** — they may run for longer than 5 s without anything being sent.

### `0x20` SERVO — steering

```
A5 20 <id> <pctHi> <pctLo>
```

* `id`: servo ID on the SCS bus (normally `1`)
* `pct`: `int16`, **−100 … +100**. Negative = right, positive = left, `0` = centre.
* The ESP maps this onto the calibrated end stops and limits it internally to
  **80 % of the mechanical travel**. So `±100` is deliberately not full lock.

### `0x40` / `0x41` — manual steering calibration

The SC09 **cannot** limit its torque: if it drives into an end stop by itself,
it keeps pushing with full torque until the linkage or the gearbox gives way.
The earlier automatic probing of the end stops has therefore been removed — the
end stops are approached by hand and confirmed one by one.

```
A5 40                 start calibration mode (identical to 0x41 with action 0x00)
A5 41 <action> <arg>
```

| Action | Name | `arg` | Effect |
|---|---|---|---|
| `0x00` | START | – | start the mode, drive to coast, torque on, hold the current position |
| `0x01` | MINUS | ticks (`0` = step size) | one step towards position `0` |
| `0x02` | PLUS | ticks (`0` = step size) | one step towards position `1023` |
| `0x03` | CENTER | – | store the current **actual** position as centre |
| `0x04` | LEFT | – | store the current actual position as left end stop |
| `0x05` | RIGHT | – | store the current actual position as right end stop |
| `0x06` | SAVE | – | check, write to NVS, end the mode |
| `0x07` | ABORT | – | abort, stored values are kept |
| `0x08` | FREE | – | torque **off**: set the steering by hand |
| `0x09` | HOLD | – | torque on, holds the current actual position |
| `0x0A` | GOTO_CENTER | – | move to the stored centre |
| `0x0B` | SET_STEP | ticks (1…200) | set the step size (default 10 ≈ 2.9°) |
| `0x0C` | STATUS | – | only query the state, changes nothing |

**Every** one of these actions — including `0x40` — is answered with `0x42`.

Procedure:

1. Send `0x40`.
2. Use `MINUS`/`PLUS` to set straight ahead → `CENTER`.
3. Slowly towards the left end stop → `LEFT`. **Stop just before the hard end
   stop.** If the response reports that the actual position no longer follows
   the target, the servo is already pushing against the mechanics.
4. Back, and towards the right end stop → `RIGHT`.
5. `SAVE`.

Alternative without any servo force: `FREE`, push the steering to the end stop
by hand, `LEFT`/`RIGHT` there, then `HOLD`.

Before saving, the ESP checks that all three marks are set, that the end stops
are **at least 50 ticks** apart and that the centre lies between them.
Otherwise `0x42` comes back with status `0x02` and **nothing** is written. A
successful save resets the trim offset to `0` — it referred to the old centre.

While the mode is running, **the ESP ignores `0x20` SERVO** (with a log line on
the USB console) — a steering command would immediately discard the position
that was set by hand.

> Which end stop is "left" is decided solely by which one the operator
> confirms with `0x04`. The mapping in `0x20` is calculated symmetrically, so a
> mirror-mounted steering also works with swapped raw values.

### `0x80` PID_SET

```
A5 80 <paramId> <int32 value×1000>
```

| paramId | Parameter | Unit | Default | Example |
|---|---|---|---|---|
| `0` | Kp | duty per count | 4.0 | Kp=4.5 → `4500` |
| `1` | Ki | duty per (count·s) | 0.5 | Ki=0.3 → `300` |
| `2` | Kd | duty per (count/s) | 0.10 | Kd=0.08 → `80` |
| `3` | I limit | duty (anti-windup) | 200 | 250 → `250000` |
| `4` | maxDuty | 0..1023 | 700 | 800 → `800000` |
| `5` | Tolerance | 1/10 degree | 50 (=5.0°) | 1.5° → `15000` |
| `6` | Dwell time | ms | 200 | 300 ms → `300000` |
| `7` | Timeout | ms | 10000 | 15 s → `15000000` |
| `8` | Breakaway duty | 0..1023 | 0 (off) | 200 → `200000` |

> **Breakaway duty (`8`)** overcomes static friction. Shortly before the
> target, the control error becomes so small that `Kp × error` falls below the
> breakaway threshold of the gear motor — it stops and the move runs into the
> timeout. With this value, never less than this is applied outside the target
> window.

> **Caution, common source of errors:** The `×1000` encoding applies to *all*
> parameters, including the integer ones. `maxDuty = 700` is transmitted as
> `700000`, not as `700`. With `700`, the ESP would receive `0`.

`0x80` only changes the **running** values in RAM. To store them permanently,
send `0x83` afterwards.

### `0x83` PID_SAVE

```
A5 83
```

Writes the **complete** current parameter set (all eight values from the table
above) to NVS and responds with `0x84`. After the next boot the ESP loads these
values automatically.

Typical procedure: send all the `0x80` that are needed, then `0x83` **once**.
Do not save after every single parameter.

> NVS discards writes in which the value has not changed. Repeatedly saving
> identical values therefore costs no flash cycles — saving cyclically every
> second is still a bad idea.

### `0x90` MOVE — position move

```
A5 90 <moveId> <int32 distance in 1/10 degree>
```

* `moveId`: `1..255`, assigned by the bridge. Avoid `0` (internal start value).
* `distance`: **relative** to the current position, in 1/10 degree of the
  output shaft, signed. `900` = "turn 90° further", `-450` = "45° back".
* There is **no** zero point and no homing — a move always refers to the here
  and now and is therefore reset-proof.
* Every move is guaranteed to be followed by **exactly one** `0x93` MOVE_DONE
  with the same ID.

> **The ×10 only applies on the wire.** For 90° the bridge sends `900`.

**Only one move runs at a time.** If `0x90` is sent while another move is
still active, the old one is replaced and acknowledges this with `0x93` /
status `0x02` before the new one starts. The bridge therefore gets a response
for every ID, no matter in which order it sends them.

---

## 4. ESP → Jetson

| CMD | Name | Payload | Triggered by |
|---|---|---|---|
| `0x42` | CAL_RSP | 11 B | `0x40`, `0x41` |
| `0x70` | BUTTON | 1 B | button press (unsolicited) |
| `0x82` | PID_RSP | 12 B | `0x81` |
| `0x84` | PID_SAVED | 1 B | `0x83` |
| `0x93` | MOVE_DONE | 6 B | end of a move (unsolicited) |
| `0x94` | PROGRESS_RSP | 11 B | `0x92` |
| `0xA1` | BATTERY_RSP | 6 B | `0xA0` |
| `0xA2` | BATTERY_WARN | 6 B | undervoltage (unsolicited) |
| `0xB1` | TIME_RSP | 17 B | `0xB0` |
| `0xB3` | STAMP_RSP | 1 B | `0xB2` |
| `0xC1` | TELEMETRY | 12 B | period from `0xC0` (unsolicited) |

### `0x70` BUTTON
```
A5 70 01
```
`01` = pressed. Debounced with 200 ms, there is no release event.

### `0x42` CAL_RSP
```
A5 42 <active> <flags> <status> <int16 pos> <int16 centre> <int16 left> <int16 right>
```

* `active`: `1` = calibration mode running, `0` = not (any more)
* `flags`: bit 0 = centre set, bit 1 = left set, bit 2 = right set,
  bit 3 = torque free
* `pos`: last commanded raw position `0..1023`
* `centre`/`left`/`right`: the mark set in this run; as long as the
  corresponding flag is `0`, the **stored** value instead

| status | Meaning |
|---|---|
| `0x00` | action carried out |
| `0x01` | saved, mode ended |
| `0x02` | rejected — incomplete or implausible, **nothing written** |
| `0x03` | servo does not respond |
| `0x04` | action needs a running calibration mode (send `0x40` first) |
| `0x05` | end of range `0`/`1023` reached |

### `0x82` PID_RSP
```
A5 82 <int32 Kp×1000> <int32 Ki×1000> <int32 Kd×1000>
```
Returns only the three control parameters, not limits/timeouts.

### `0x84` PID_SAVED
```
A5 84 <status>
```
`0x00` = written to NVS, `0x01` = failed (partition full or
defective). Only sent as a response to `0x83`.

### `0x93` MOVE_DONE
```
A5 93 <moveId> <status> <int32 actual position in 1/10 degree>
```

| status | Meaning |
|---|---|
| `0x00` | **OK** — target reached and held for the dwell time |
| `0x01` | **TIMEOUT** — target not reached within `timeoutMs`, aborted |
| `0x02` | **ABORTED** — replaced by `0x91`, `0x10`, `0xFF` or a new move |

After *every* end, the motor goes into **coast, not into position hold**. With
an external load or on a slope, the axis then drifts away. Anyone who wants to
hold must move to the position again cyclically.

> Unlike the travel distance in `0x90`, the position field is **absolute**:
> encoder count since boot or since `z` on the console. It serves as
> telemetry, not as the reference for the next move.

### `0x94` PROGRESS_RSP
```
A5 94 <moveId> <active> <percent> <int32 actual> <int32 target>
```
* `active`: `1` = move running, `0` = idle. With `0`, the remaining fields
  refer to the **most recent** move.
* `percent`: `0..100`, calculated from distance covered / total distance.
  Capped at 100 on overshoot.
* `actual` / `target`: 1/10 degree, **absolute** (encoder count since boot).
  `target` is the end position calculated from the relative travel distance.

### `0xA1` BATTERY_RSP / `0xA2` BATTERY_WARN
```
A5 A1 <int32 pack-mV> <int16 cell-mV>
```
Identical payload, different trigger.

* The ESP measures **every 15 s** on its own.
* `0xA1` only comes on request (`0xA0`) and returns the last measured value.
* `0xA2` comes **unsolicited** as soon as the cell voltage falls below
  **3.80 V** (4S pack ⇒ 15.2 V), and then repeats **at most once per
  minute** while the condition persists.
* All-clear only above **3.85 V/cell** (hysteresis). There is **no**
  separate all-clear packet — the absence of `0xA2` is the signal.

> The ESP does **not switch anything off** on undervoltage. The Jetson must react.

### `0xB1` TIME_RSP
```
A5 B1 <seq> <int64 t_rx_us> <int64 t_tx_us>
```

* `seq`: returned unchanged from the request, so that the bridge can match
  responses even if a round gets lost.
* `t_rx_us`: ESP clock when the **last byte of the request** had arrived.
* `t_tx_us`: ESP clock when the **last byte of this response** leaves the
  line. The time to shift out the 20 frame bytes (1.74 ms at 115200) is
  already included — so the value lies in the future when the ESP writes
  it.

Both values are full `int64` microseconds since ESP boot, no overflow.
For the calculation see section 5.

### `0xC1` TELEMETRY
```
A5 C1 <int32 pos in 1/10 deg> <int32 speed in 1/10 deg/s> <int16 duty> <int16 mA>
```

The drive state that the ESP sends on its own at the period set by `0xC0`.

**Three of the four fields are signed, one is not** — this is the most likely
pitfall when parsing:

| Field | Type | Signed |
|---|---|---|
| `pos` | `int32` | **yes** — negative when the shaft is below the starting point |
| `speed` | `int32` | **yes** — negative when reversing |
| `duty` | `int16` | **yes** — negative when reversing |
| `mA` | `int16` | **no** — always ≥ 0 |

* `pos`: position of the output shaft, **absolute** — encoder count since boot
  or since `z` on the console. The same reference as in `0x93`/`0x94`.
* `speed`: rotational speed of the output shaft. `1800` = 180 °/s = 30 rpm,
  `-1800` the same in reverse.
* `duty`: what is applied to the bridge, −1023…+1023. `0` when coasting **and**
  when braking — this is not an error, there really is no duty cycle applied
  then.
* `mA`: motor current. The VNH5019 reports **only the magnitude** on its CS
  output, not the direction. Anyone who needs the direction of the current
  reads it from the sign of `duty` — with the restriction that neither gives
  anything meaningful while coasting or braking.

> Which direction of rotation is "forward" depends on how the motor and the
> encoder are wired — the protocol does not define it. For this there is the
> `DRIVE_INVERT` switch in `src/main.cpp`:
>
> | Symptom | Cause | Remedy |
> |---|---|---|
> | positive motor value drives in reverse, `speed` matches it | the vehicle as a whole is wired the other way round | `DRIVE_INVERT = true` |
> | motor is correct, but `speed` and `pos` have the wrong sign | only the encoder channels A/B are swapped | swap the channels, do **not** touch `DRIVE_INVERT` |
>
> `DRIVE_INVERT` flips the motor **and** the encoder together. Flipping only one
> of the two would be an own goal: the position controller then reads a sign
> that does not match its control output, and drives away from the target
> instead of towards it, until the timeout takes effect.

> **Position and speed are both produced in the 10 ms cycle of the motor task**
> and are fresh in every packet. `0xC0` is therefore limited to a minimum of
> **10 ms** — sending faster would mean sending the same packet twice.
>
> This limit is in `TELEMETRY_MS_MIN`, and **twice** at that: in
> `src/main.cpp` and in `esp_serial_bridge.py`. The firmware value is the
> binding one; anyone who changes only the Python side still gets the old
> period.
>
> **Send period and measurement window are two different things.** The ESP
> computes the speed from a *sliding* window over the last `n` samples:
>
> ```
> speed = (count[now] - count[now - n]) / (t[now] - t[now - n])
> ```
>
> A new value is produced at every sample — so the output rate does not depend
> on `n`. What does depend on `n` is the resolution, because a single encoder
> pulse in the window is the smallest step the measurement can make:
>
> | `n` | Window | One pulse | Delay |
> |---|---|---|---|
> | 1 | 10 ms | 14.7 rpm = 88 °/s | 5 ms |
> | 5 | 50 ms | 2.9 rpm = 17.6 °/s | 25 ms |
> | 10 | 100 ms | 1.5 rpm = 8.8 °/s | 50 ms |
>
> at 408 pulses per revolution of the output shaft and a top speed of about
> 30 rpm. **With `n = 1` the smallest measurement step is of the same order
> of magnitude as full scale** — the speed then jumps back and forth between a
> few discrete levels. This is not an error, but the resolution of the encoder
> at 10 ms.
>
> `n` is set on the USB console with `sw<n>`, e.g. `sw5`. The delay is exactly
> half a window — this is the advantage of the sliding window over an EMA,
> whose time constant can only be estimated.
>
> Anyone who wants both, fine resolution *and* 10 ms delay, cannot get there
> with this encoder: that would need timestamps of individual edges (M/T
> method) instead of pulses per window.

> The period is soft: the ESP sends from within its `loop()`. This is exactly
> what the send timestamp is for — the Jetson should **read** the time instead
> of extrapolating it from the nominal period.

### `0xC0` TELEM_RATE
```
A5 C0 <uint16 period in ms>
```
`0` switches it off, otherwise 20…60000 ms; the ESP silently raises lower
values to 20. The command is answered with an immediate `0xC1` — this is both
the acknowledgement and the first measured value.

The period is **not** stored in NVS. After an ESP reset the telemetry is off;
the bridge sets it again when it connects.

### `0xB3` STAMP_RSP
```
A5 B3 <mode>
```
`0` = ESP sends `0xA5` frames, `1` = ESP sends `0xA6` frames with a send
timestamp. Sent as a response to `0xB2`, and also unsolicited when someone
changes the mode on the USB console with `ts0`/`ts1`.

---

## 5. Time synchronisation

Goal: for every packet from the ESP, know **when it was sent** — expressed in
the Jetson's clock, so that events can be merged with camera, LiDAR and ROS
data.

This splits into two independent parts:

1. **The stamp.** Every packet says itself when it went out — this is the
   `0xA6` frame from section 2. The value is in the ESP clock.
2. **The clock offset.** A ping-pong (`0xB0`/`0xB1`) measures how far apart
   the ESP clock and the Jetson clock are. This turns the stamp into a point
   in time in the Jetson clock.

### 5.1 The two clocks

| | ESP | Jetson |
|---|---|---|
| Source | `esp_timer_get_time()` | `time.monotonic()` (= `CLOCK_MONOTONIC`) |
| Resolution | 1 µs | 1 ns (µs in practice) |
| Zero point | boot of the ESP | boot of the Jetson |
| Overflow | none (`int64`) | none |

**For the synchronisation the Jetson must use a monotonic clock, not
`time.time()`.** Otherwise an NTP jump would shift the offset in the middle of
the measurement. The reference to the wall clock is only established at the
very end, with a distance `CLOCK_REALTIME − CLOCK_MONOTONIC` measured once.

### 5.2 The procedure

```
Jetson                                    ESP
  |                                        |
  |--- A5 B0 <seq> -------------->         |
  |    t1 = last byte out                  |
  |                                     t2 = last byte in
  |                                        |
  |         <---- A5 B1 seq t2 t3 ---------|
  |    t4 = last byte in                t3 = last byte out
```

All four points in time refer to **the last byte of the respective frame on
the line**. This is the only reference point that both sides can hit cleanly,
and it makes the calculation symmetric — otherwise the transmission time of
the 20-byte response would stand against that of the 3-byte request and
distort the offset by ~0.7 ms.

How the two sides hit this point:

* **`t1`**: let the buffer drain, **then** take the clock, **then** write —
  and add the transmission time of the frame (10 bits per byte at 8N1). So it
  is *calculated*, not measured.

  > Do **not** measure `write()` followed by `flush()`. That looks cleaner but
  > does not work: `flush()` ends up in `tcdrain()`, and depending on the
  > driver — reliably on the Jetson's Tegra UART — that returns much later than
  > the last byte leaves the line. `t1` then becomes too late, in the extreme
  > case later than `t4`, and the round trip is mathematically **negative**.

* **`t2`**: the ESP takes the clock as soon as the packet is complete.
* **`t3`**: the ESP takes the clock immediately before writing and **adds the
  transmission time of the frame**. Before that it lets the send buffer drain,
  so that the calculation is correct. The same method as for `t1`.
* **`t4`**: the bridge takes the clock when the last byte of the frame has
  been read — that is, after the packet has been assembled, not at the start
  byte.

  > **Reading in blocks destroys `t4`.** `ser.read(64)` only returns when
  > 64 bytes have accumulated or the timeout expires; all packets in this block
  > then get the same receive time, namely that of the last one. With
  > telemetry running, that is easily an error of 150 ms. The correct way is
  > `read(1)` — this only blocks until the first byte — followed by
  > `read(in_waiting)` for the rest without further waiting.

### 5.3 The calculation

The standard NTP formulas:

```
offset   = ((t2 - t1) + (t3 - t4)) / 2       # ESP clock minus Jetson clock
rtt      = (t4 - t1) - (t3 - t2)             # pure line time + waiting time
```

Then:

```
jetson_time = esp_stamp - offset
```

The offset is only correct if the outbound and return paths are equally long.
They are on average, but not in every single round — the ESP reads its UART
from within `loop()`, so `t2` arrives late by a varying amount depending on the
loop iteration.

**Therefore: measure several rounds and take the one with the smallest `rtt`.**
Every delay that disturbs the symmetry also lengthens the round trip; the
fastest round is therefore automatically the most honest one. Eight to sixteen
rounds at intervals of ~20 ms are enough.

Two traps with this filter, both of which have already struck once:

* **A negative round trip is physically impossible** and means that one of the
  four time measurements was wrong. Such rounds must be **discarded** —
  otherwise `min()` picks precisely the most broken one as the "best".
* **The bound must be additive** (`best + 2 ms`), not multiplicative
  (`best × 2`). With a negative best value, the multiplicative bound becomes
  *smaller* than the best value, nothing gets through, and a fallback to "then
  just take all of them" shovels the garbage into the estimate all the more.

If nothing usable remains, the offset is **invalid** — and the bridge honestly
stamps with the read time instead of inventing a false send time.

### 5.4 Drift

Both clocks run on their own crystals, typically ±20…50 ppm. Against each other
that is up to 100 ppm, i.e. **0.1 ms deviation per second** — 60 ms after ten
minutes.

Two options, the usual one first:

* **Resynchronise.** One measurement round every 10 s. The offset then stays
  below ~1 ms. Costs 20 bytes per round, i.e. nothing.
* **Estimate the drift.** Fit a straight line `offset(t) = a + b·t` through the
  last measurement points (least squares). `b` is the relative rate deviation.
  This keeps the error small between rounds as well, and the jump on
  resynchronisation disappears. Worth it when timestamps have to match image
  data.

When resynchronising, never jump hard to the new value, but blend it in
(`offset = 0.8·old + 0.2·new`) — otherwise a jump mixes up the order of events
that have already been stored.

### 5.5 The 32-bit stamp in the frame

`0xA6` only transmits the lower 32 bits of the ESP clock. This overflows every
**71.6 minutes**. `0xB1`, on the other hand, returns the full `int64`. Unwrapping:

```python
def unwrap(stamp32: int, last_full_value: int) -> int:
    coarse = (last_full_value & ~0xFFFFFFFF) | stamp32
    for candidate in (coarse - 2**32, coarse, coarse + 2**32):
        if abs(candidate - last_full_value) < 2**31:
            return candidate
    return coarse
```

As long as the bridge runs a `0xB0` round at least every 35 minutes
(guaranteed with a 10 s period), this is unambiguous.

### 5.6 What the stamp does *not* tell you

The stamp is the **send time of the packet**, not the time of the event. For
some packets there is a gap between the two:

| Packet | Gap event → send |
|---|---|
| `0x93` MOVE_DONE | up to one controller cycle (10 ms) plus one `loop()` iteration — the controller runs on core 0 and passes the result to core 1 through a queue |
| `0x70` BUTTON | the 200 ms debounce time lies **before** the event, followed by one `loop()` iteration |
| `0xA1`/`0xA2` battery | the measured value is up to 15 s old (measurement grid), the stamp is still brand new |
| `0x42`, `0x82`, `0x94` | responses, generated directly after the command — negligible gap |

For latency measurements of the link the stamp is exactly right. For "when was
the button pressed" it is an upper bound.

### 5.7 Error budget

| Source | Order of magnitude | Countermeasure |
|---|---|---|
| `loop()` delay when reading on the ESP | 0.1…5 ms, highly variable | take the smallest round trip from N rounds |
| `t4` in Python (scheduling, `select`) | 0.1…1 ms | ditto |
| Transmission time of the frames | 0.26 / 1.74 ms | included on both sides |
| Crystal drift | 0.1 ms/s | resynchronise every 10 s |
| `tcdrain()` returns late (Tegra) | up to several ms, can make `rtt` negative | calculate `t1` instead of measuring it (5.2) |
| Reading in blocks for `t4` | up to the read timeout, i.e. ~50…150 ms | `read(1)` + `read(in_waiting)` (5.2) |
| Debug output on the USB console | up to several ms | set `dbg0` while measuring |

Realistically this gives **±1 ms** without special effort and **±0.2 ms** with
drift estimation and a quiet link. Anyone who needs better cannot avoid a
hardware line (PPS pulse from the Jetson to an ESP interrupt).

### 5.8 Order when establishing the connection

1. Open the port, start the read thread.
2. 8–16 rounds of `0xB0` → first offset.
3. Send `A5 B2 01`, wait for `0xB3`. From now on `0xA6` frames arrive.
4. During operation, send one more round of `0xB0` every 10 s.

Step 2 before step 3: without an offset a stamp is worthless, and `0xB1` does
not need the stamp mode.

### 5.9 Reference implementation

`docs/timesync_jetson.py` contains the Jetson side as a standalone class:
ping-pong, minimum filter, drift estimation, unwrapping and parsing of both
frame forms. Testable without hardware:

```
python3 docs/timesync_jetson.py --selftest
```

It is used by `src/esp_serial_bridge.py` — see section 6.

---

## 6. ROS 2 bridge

`src/esp_serial_bridge.py` is the finished counterpart: `EspLink` handles the
protocol (without a ROS dependency, so it can be tested without the robot),
`EspBridgeNode` connects it to ROS. Self-test without hardware and without ROS:

```
python3 src/esp_serial_bridge.py --selftest
```

**Every message with `header.stamp` carries the ESP's send time**, converted
to the ROS clock. For this, the transit time is measured in the monotonic clock
and subtracted from the ROS time of reading — this also stays correct under
`use_sim_time`.

### What goes in

| Topic | Type | Effect |
|---|---|---|
| `/cmd_vel` | `geometry_msgs/Twist` | `linear.x` → motor, `angular.z` → steering |
| `~/motor` | `std_msgs/Int32` | raw duty, −1023…+1023 |
| `~/steer` | `std_msgs/Float32` | −100…+100 |
| `~/move` | `std_msgs/Float32` | degrees, **relative** to the current position |
| `~/led` | `std_msgs/Bool` | |
| `~/trim` | `std_msgs/Int32` | 0 = left, 1 = right, 2 = save |
| `~/emergency` | `std_msgs/Empty` | emergency stop |
| `~/pid_set` | `std_msgs/Float32MultiArray` | `[id, value]` or all nine values |
| `~/cal` | `std_msgs/Int32MultiArray` | `[action, arg]` |
| `~/cal_action` | `std_msgs/String` | the same in plain text: `plus`, `left`, `save` … |

### What comes out

| Topic | Type | Content |
|---|---|---|
| `~/button` | `std_msgs/Header` | button press — the content *is* the timestamp |
| `~/joint_states` | `sensor_msgs/JointState` | position in rad, from `0xC1` additionally `velocity` in rad/s |
| `~/speed` | `std_msgs/Float32` | rotational speed in °/s, **signed** |
| `~/motor_state` | `std_msgs/Float32MultiArray` | `[duty, ampere]` — duty signed, current not |
| `~/move_done` | `std_msgs/Int32MultiArray` | `[move_id, status, tenths_of_degree]` |
| `~/move_progress` | `std_msgs/Float32` | 0…100 % |
| `~/battery` | `sensor_msgs/BatteryState` | pack and cell voltage |
| `~/battery_low` | `std_msgs/Bool` | undervoltage warning (latched) |
| `~/cal_state` | `std_msgs/Int32MultiArray` | state of the calibration |
| `~/pid` | `std_msgs/Float32MultiArray` | `[kp, ki, kd]` (latched) |
| `~/console` | `std_msgs/String` | ASCII lines from the ESP |
| `/diagnostics` | `diagnostic_msgs/DiagnosticArray` | clock offset, drift, round trip, counters |
| `~/latency_ms` | `std_msgs/Float32` | transit time of *this* packet: sent → read |
| `~/rtt_ms` | `std_msgs/Float32` | shortest round trip of the last sync round, 1 Hz |
| `~/offset_ms` | `std_msgs/Float64` | clock offset, 1 Hz |
| `~/drift_ppm` | `std_msgs/Float32` | estimated crystal drift, 1 Hz |

The last four carry the same numbers as `/diagnostics`, only as numbers
instead of text — `DiagnosticArray` stores its values as strings, and a string
cannot be plotted. `~/offset_ms` is `Float64` because the offset grows to
several million milliseconds; `float32` would only have 1 ms steps there and
the drift would disappear in the noise.

### Services

Everything that has an acknowledgement is a service instead of a topic —
otherwise the caller never finds out whether it worked.

| Service | Type |
|---|---|
| `~/emergency_stop`, `~/move_abort` | `std_srvs/Trigger` |
| `~/pid_get`, `~/pid_save` | `std_srvs/Trigger` |
| `~/calibrate_start`, `~/calibrate_save` | `std_srvs/Trigger` |
| `~/trim_save`, `~/torque_report`, `~/resync` | `std_srvs/Trigger` |
| `~/set_led`, `~/set_stamp_mode`, `~/servo_torque_free` | `std_srvs/SetBool` |

### Parameters

`port`, `baud`, `servo_id`, `stamp_mode`, `sync_rounds`, `sync_interval`,
`heartbeat_period`, `cmd_vel_timeout`, `battery_period`, `progress_period`,
`telemetry_period`, `max_linear`, `max_angular`.

`telemetry_period` (default 0.05 s = 20 Hz) is the period at which the ESP
sends position and speed on its own. Can be changed at runtime:

```
ros2 param set /esp_serial_bridge telemetry_period 0.1
```

`0` switches the telemetry off. The ESP does not accept less than 0.02 s —
the bridge reports this as a warning instead of silently swallowing it.

> **You have to measure `max_linear` and `max_angular`.** The ESP does not
> control the speed — `/cmd_vel` is converted directly to PWM. The two values
> say which speed or turn rate full deflection means.

### Reading `/diagnostics`

| Key | Healthy | Warning sign |
|---|---|---|
| `offset_ms` | any size, even several hours | `-` (no sync) |
| `rtt_ms` | 0.3…3 ms | **negative** or > 50 ms |
| `drift_ppm` | −100…+100 | three digits |
| `sync_rejected` | `0` | > 0 (unusable measurement rounds) |
| `timestamps` | `on` | `off` |
| `esp_reboots` | constant | rises during operation |

> **A huge `offset_ms` is normal and not an error.** The ESP clock counts from
> its boot, the Jetson clock from the Jetson's boot. If the Jetson has been
> running for two hours and the ESP for five minutes, that is about
> −7 000 000 ms. Knowing exactly this distance is the whole point of the
> exercise.
>
> **A negative `rtt_ms`, on the other hand, is always an error** — see 5.2 and
> 5.3.

### Ready-made Foxglove layout

`docs/foxglove_esp_bridge.json` shows all topics and services of the bridge.
To load it in Foxglove: **Layout → Import from file…**

| Tab | Content |
|---|---|
| Fahren (Drive) | teleop on `/cmd_vel`, emergency stop, position move, speed/duty, position, current |
| Zeit & Link (Time & link) | latency, round trip, drift, offset, button for resync |
| Akku & Zustand (Battery & state) | pack and cell voltage, undervoltage lamp, move progress, button press |
| Diagnose & Konsole (Diagnostics & console) | `/diagnostics` as a table, ESP console, `/rosout` |
| Kalibrieren (Calibrate) | calibrate the steering, release the servo, save |
| Service | emergency stop, abort move, read/save PID, timestamps, LED, servo load |
| Befehle (Commands) | the raw topics `motor`, `steer`, `trim`, `led`, `pid_set`, `cal` |

Two places that have to be adapted depending on the setup:

* The diagnostics panel is set to `hardware_id = /dev/ttyTHS1` — this is the
  `port` parameter. With a different port, select the panel again.
* All paths start with `/esp_serial_bridge/`. If the node runs under a
  different name or in a namespace, replace it once in the file.

### Showing latency in Foxglove

**Plot panel → enter path.** The paths are topic plus field name:

| What | Path |
|---|---|
| Transit time per packet | `/esp_serial_bridge/latency_ms.data` |
| Round trip (link health) | `/esp_serial_bridge/rtt_ms.data` |
| Drift | `/esp_serial_bridge/drift_ppm.data` |

Set the x-axis to **timestamp**, not to *index* — otherwise the graph shows
the message number instead of the time.

`latency_ms` arrives as often as stamped packets arrive — with 50 ms telemetry,
that is 20 times per second. The other three arrive once per second.

> **Why not plot directly from `/diagnostics`?** Foxglove has the Diagnostics
> panel for that, which shows the values as a table. It cannot plot them:
> in `DiagnosticArray` every value is text, and `"0.601"` is not a numeric
> value for the Plot panel. Hence the separate topics above.

A second way that needs no extra topics at all: the Plot panel can draw
**receive time** as the x-axis and `header.stamp` of a stamped topic as the
y-axis — for example `/esp_serial_bridge/joint_states`. The gap between the two
*is* the latency. It can only be read roughly, though, because Foxglove does
not compute the difference itself.

### Two things the bridge does by itself

* **Heartbeat.** The ESP lets the motor coast if no command arrives for 5 s.
  The last motor command is therefore resent cyclically — deliberately not
  during a position move, which may take longer.
* **Watchdog on `/cmd_vel`.** If it stays silent for longer than
  `cmd_vel_timeout`, the motor goes into coast. Otherwise a crashed controller
  would let the robot keep driving.

---

## 7. Known gaps

Deliberately left open, relevant for the bridge:

* **No checksum, no length field.** A lost byte cascades until the next
  `0xA5`. The ESP's 100 ms timeout catches this; the bridge needs an
  equivalent safeguard.
* **No acknowledgement for `0x10`, `0x20`, `0x30`, `0x60`, `0x80`.** Fire-and-forget.
  Whether a PID value arrived can only be cross-checked with `0x81`.
* **`0x80` on its own is volatile.** Without a subsequent `0x83`, the values
  are back to the stored state after the next reset.
* **No boot/ready packet.** The bridge does not notice an ESP reset directly.
  Anyone who needs this recognises it by the startup ASCII line
  `System Ready. ...` or polls `0x81` cyclically. Since the time
  synchronisation there is a second, unambiguous way: **if `t_rx_us` in `0xB1`
  jumps back, the ESP has rebooted** — the clock runs from boot. Then discard
  the offset and the drift estimate and measure again, otherwise the bridge
  dates everything wrongly by the old uptime.
* **The stamp is the send time, not the event time.** How far apart the two
  are is given in section 5.6. For `0x93` MOVE_DONE it is up to 10 ms.
* **The position telemetry is volatile.** Move commands are relative and
  therefore reset-proof, but the absolute values reported in `0x93`/`0x94`
  start at 0 again after every ESP reset. Anyone who keeps counting distance
  across restarts must do so on the Jetson side.
* **`0x50` TORQUE does not respond** over UART — the result only goes to the
  USB console. (`0x40`/`0x41` respond with `0x42` since the manual
  calibration was introduced.)
* **Calibration is a manual operation.** `0x41` moves the servo by exactly one
  step per packet; the bridge must send the steps one at a time and show the
  operator the `0x42` feedback. There is deliberately no action that drives to
  the end stop by itself.

---

## 8. Logging on the ESP side (bring-up)

The ESP logs the Jetson link on its **USB console** (separate interface,
115200). When developing the bridge, this is the quickest answer to "is
anything arriving at all?".

```
[RX] MOVE          id=3 by +90.5 deg
[TX] MOVE_DONE 03 00 00 00 03 84
[RX] UNKNOWN cmd=0x55 - discarded
[RX] ABORT cmd=0x10 MOTOR after 2/3 bytes (>100 ms gap) - resync
```

Control via the USB console:

| Command | Effect |
|---|---|
| `dbg` | statistics: packets, unknown CMDs, aborts, stray bytes, clock, sync counter |
| `dbg0` | logging off |
| `dbg1` | decoded packets (default) |
| `dbg2` | additionally all raw bytes including the sync search |
| `ts` | state of the ESP clock and stamp mode |
| `ts0` / `ts1` | send timestamp (`0xA6` frames) off / on |
| `tel` | show the drive state once |
| `tel<ms>` | set the telemetry period, `tel0` = off |

`ts0`/`ts1` send the Jetson an unsolicited `0xB3` — so the bridge notices the
change even if it came from the console.

Stamped packets appear in the log with their raw time:

```
[TX] MOVE_DONE t=45120833 03 00 00 00 03 84
[RX] TIME_SYNC   seq=7
[TX] TIME_RSP 07 00 00 00 00 02 B0 4C 21 00 00 00 00 02 B0 4E 95
```

The same calibration can be run directly on the USB console without the
Jetson — useful for cross-checking when the bridge behaves differently than
expected:

| Command | Effect |
|---|---|
| `cal` (or `x`) | start; while calibration is running, show the status |
| `+` / `-` | one step, `+50` for a one-off 50 ticks |
| `caln<t>` | step size in ticks |
| `calm` / `call` / `calr` | store centre / left / right end stop |
| `calfree` / `calhold` | torque off (set by hand) / hold again |
| `calgo` | move to the stored centre |
| `calsave` / `calq` | save / abort |

Help with interpreting:

* **`ABORT ... after n/m bytes`** — the bridge sent a packet in several
  `write()` calls with a pause. A packet must go out in one go.
* **`UNKNOWN cmd=0x??`** — sync loss or wrong CMD byte. If the byte plausibly
  looks like payload, a payload length in the bridge's table is probably
  wrong.
* **Many stray bytes with `dbg2`, some of them ASCII** — normal, these are the
  ESP's plain-text lines (see section 2, rule 4).
* **`dbg` shows 0 packets and 0 stray bytes** — nothing is physically
  arriving. Check the wiring (RX/TX crossed?), common ground and baud rate.

> `dbg1` writes one line per packet. A motor heartbeat at a 100 ms period
> therefore produces 10 lines/s. For continuous driving, set `dbg0`.

---

## 9. Reference values

| | |
|---|---|
| Encoder | 408 counts per revolution of the output shaft (4× quadrature) |
| PWM | 20 kHz, 10 bit (0..1023) |
| Motor driver | VNH5019 (INA/INB/PWM) |
| Controller cycle | 100 Hz (10 ms) on core 0 |
| Acceleration ramp | max. 25 duty steps per 10 ms ⇒ ~410 ms to full throttle |
| Battery | 4S, warning < 3.80 V/cell, all-clear > 3.85 V/cell |
| Steering | SCServo SCS/SCSCL, ID 1, travel limited to 80 % |
| ESP clock | `esp_timer`, µs since boot, `int64` (no overflow) |
| Frame stamp | lower 32 bits of it ⇒ overflow every 71.6 min |
| Byte on the line | 10 bits at 8N1 ⇒ 86.8 µs at 115200 |
| Sync round | 3 B out + 20 B back ⇒ 2.0 ms pure transmission |
| Speed measurement | every 100 ms, exponentially smoothed (α = 0.30) |
| Telemetry at 20 Hz | 18 B per packet ⇒ 360 B/s, ~3 % of the line |
| Achievable accuracy | ±1 ms simple, ±0.2 ms with drift estimation |

### Measured on the Jetson (reference for comparison)

Jetson Orin, `/dev/ttyTHS1`, telemetry at 20 Hz, ESP idle:

| | Measured | Alarm threshold |
|---|---|---|
| `rtt_ms` | 0.60 | negative or > 50 |
| `drift_ppm` | −57 | three digits |
| `sync_rejected` | 0 | > 0 |
| resulting stamp accuracy | ~±0.3 ms | |

If your link differs significantly from this, something is wrong — the usual
causes are in 5.2 and in the error budget 5.7.

---

## 10. Prompt for the Jetson chat

> **Largely done:** `src/esp_serial_bridge.py` is the bridge, section 6
> describes it. This prompt remains as a description of the requirements —
> useful if the bridge ever has to be set up again or cross-checked.

> I have an ESP32-S3 that controls a drive motor, a servo steering and the
> battery monitoring of a robot over UART (115200 8N1). The protocol is fully
> described in the attached specification. I need a Python bridge for it on
> the Jetson.
>
> **What has changed compared with the previous bridge and therefore has to be
> rewritten:**
>
> 1. **The receive path becomes asynchronous.** Until now, practically only the
>    button event came from the ESP. Now it also sends `0x93` MOVE_DONE and
>    `0xA2` BATTERY_WARN unsolicited. A request-response model is no longer
>    enough — it needs a permanently reading thread that puts packets into a
>    queue and dispatches them via callbacks.
>
> 2. **The RX parser must handle variable packet lengths.** Previously a fixed
>    3 bytes, now 1 to 12 bytes of payload depending on the CMD. Hard-wire the
>    length table from the spec, discard unknown CMDs and resync to the next
>    `0xA5`. Important: the stream additionally contains ASCII status lines
>    from the ESP, which must be skipped (and ideally logged).
>
> 3. **Position moves are relative and need ID tracking.** `0x90` is given a `moveId`,
>    the result comes at some later point as `0x93` with the same ID and a
>    status (OK / Timeout / Aborted). Build this as an `asyncio.Future` or a
>    callback per ID, so that calling code can wait for a specific move. Only
>    one move can run at a time; a new one replaces the old one, and the old
>    one acknowledges with status `0x02`.
>
> 4. **A heartbeat is new and mandatory.** The ESP stops the motor on its own
>    if no command arrives for 5 s. For continuous driving the bridge must
>    resend `0x10` MOTOR cyclically (every 100–500 ms). Position moves are
>    exempt; the heartbeat may pause during them.
>
> 5. **PID parameters are two-stage.** `0x80` PID_SET only changes the running
>    value; only `0x83` PID_SAVE writes the whole set to NVS and acknowledges
>    with `0x84`. So when tuning, many `0x80` and exactly one `0x83` at the
>    end — do not save after every parameter.
>
> 6. **All floating-point values go over the wire as `int32 × 1000`, big-endian
>    — including the integer PID parameters.** `maxDuty = 700` is encoded as
>    `700000`. This is the most likely source of errors; write unit tests for it.
>
> 7. **Battery warnings arrive by themselves.** No polling needed. The ESP does
>    not switch anything off on undervoltage — the reaction (stop driving, go
>    to the charging station, alarm) must happen on the Jetson side.
>
> 8. **There is a time synchronisation.** The ESP can stamp every packet with
>    its send time (`0xA6` frame, switched on with `0xB2`), and a ping-pong
>    `0xB0`/`0xB1` provides the clock offset between ESP and Jetson. This gives
>    every event a point in time in the Jetson clock that can be merged with
>    camera and LiDAR data. The calculation, the error budget and a finished
>    implementation that can be tested without hardware are in section 5 and
>    in `docs/timesync_jetson.py` respectively — **take it from there instead
>    of re-deriving the NTP formulas.** For this, the RX parser must handle both
>    start bytes and relate all four points in time to the *last* byte of the
>    respective frame (`flush()` after sending!).
>
> Build the bridge as a class with cleanly separated methods per command,
> type annotations and a context manager for opening/closing the port.
> Serialisation and parsing should be testable without real hardware.
