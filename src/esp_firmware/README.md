# ESP32-S3 firmware

Real-time actuator controller of the vehicle, running on the ESP32-S3 of the
[main board](../../schemes/MainPCB). The Jetson sends drive and steering
commands over UART; the ESP closes the loops that need hard timing.

- **Drive:** VNH5019 motor driver with a quadrature encoder, open-loop duty
  or a PID position controller, current sensing, 5 s watchdog
- **Steering:** Waveshare SC09 servo on a half-duplex bus, calibration of
  center and end stops stored in flash
- **Battery:** 4S pack voltage with an undervoltage warning to the Jetson
- **Start button** and status LED
- **Jetson link:** binary protocol with optional send timestamps for clock
  sync, specified in [`docs/JETSON_BRIDGE.md`](docs/JETSON_BRIDGE.md) — the
  counterpart is the ROS 2 package [`esp_bridge`](../esp_bridge)

Core 1 handles commands, steering and the battery; core 0 runs only the
motor control task every 10 ms.

## Layout

| Path | Content |
|---|---|
| `src/main.cpp` | the firmware |
| `platformio.ini` | board and build settings |
| `lib/SCServo/` | Feetech servo library (not in the PlatformIO registry) |
| `test_sketches/` | standalone Arduino sketches for testing hardware on its own |
| `docs/` | UART protocol specification, Foxglove layout for the bridge |

`COLCON_IGNORE` keeps `colcon build` on the Jetson out of this folder.

## Build and flash

Open this folder in VS Code with the PlatformIO extension, or:

```bash
pio run -t upload        # build and flash over USB
pio device monitor       # debug console, 115200 baud
```

If nothing appears on the console, the USB-CDC connection is not up yet
(the ESP drops output until the host opens the port). The environment
`esp32-s3-uart0` mirrors the console to UART0 (GPIO43 TX / GPIO44 RX) for a
USB-TTL adapter:

```bash
pio run -e esp32-s3-uart0 -t upload
```

## Pins

| Function | GPIO |
|---|---|
| VNH5019 PWM / INA / INB / CS | 41 / 42 / 38 / 8 |
| Encoder A / B | 15 / 16 |
| Servo bus RX / TX | 18 / 17 |
| Jetson UART RX / TX | 10 / 11 |
| LED / button | 13 / 9 |
| Battery divider (100k / 22k) | 1 |

## Debug console

Type `h` on the USB console for the full command list. The most used ones:

| Command | Effect |
|---|---|
| `f<n>` `r<n>` `c` | drive forward / reverse, coast |
| `g<deg>` / `gp<deg>` | rotate the drive shaft by a relative angle (with plot) |
| `kp<f>` `ki<f>` `kd<f>` `pid` `pids` | PID gains, show, save to flash |
| `cal` … `calsave` | manual steering calibration |
| `v` `vc<factor>` | battery voltage, calibrate the divider |
| `dbg1` / `ts` / `tel<ms>` | Jetson link statistics, clock sync, telemetry rate |

## History

The code started as Arduino sketches in the FutureEngineers repository and
came over with its commit history (`git log --follow src/main.cpp`). The very
first controller design (MD10C driver, split into `ActuatorController.h` /
`CommsHandler.h`) was dropped when the firmware moved here; it is still in the
history under `legacy/`.
