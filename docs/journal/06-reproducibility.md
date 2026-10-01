# Reproducibility

<!--
Rubric criterion 5. Mostly judged on the repository itself; this chapter is the
map to it.
6 points: fully reproducible; clear structure; meaningful commits; documented
testing workflow; versioning or release notes.
-->

**Evidence at a glance.** Where this chapter answers each point of the rubric
for criterion 5:

| The rubric asks for | Section | Key evidence |
| --- | --- | --- |
| Fully reproducible | [Building and flashing](#building-and-flashing) | environment recipe, device rules, calibration files, firmware; the three upstream packages pinned or included with their changes |
| Clear project structure | [Repository structure](#repository-structure) | the WRO folder layout; one folder per component |
| CAD, code and wiring included | [Repository structure](#repository-structure) | PCB in KiCad with BOM, chassis CAD, all code; wiring in [chapter 2](03-power-sensors.md) |
| Documented testing workflow | [Testing workflow](#testing-workflow) | unit tests without hardware, reproducible bench sweeps, every field run recorded and evaluated, CI |
| Meaningful commits, versioning | [Versions](#versions) | more than 300 commits since November 2025; tagged releases with notes |

## Repository structure

| Path | Content |
| --- | --- |
| [`src/`](../../src) | all code: the ROS 2 packages, the ESP32 firmware, the start script |
| [`src/esp_firmware/`](../../src/esp_firmware) | firmware of the ESP32-S3 on the main board (PlatformIO) |
| [`schemes/`](../../schemes) | main PCB: KiCad project, schematic PDF, BOM and placement files |
| [`models/`](../../models) | CAD as STEP: full assembly and chassis |
| [`src/Hardware/`](../../src/Hardware) | chassis CAD: Fusion 360 source and STL |
| [`config/`](../../config) | calibration files the nodes load at run time |
| [`setup/`](../../setup) | everything the Jetson needs beyond the repository |
| [`docs/`](..) | this journal, the analysis scripts, the measurement data behind every figure |
| `v-photos/`, `t-photos/`, `video/` | vehicle photos, team photos, driving videos |

<!-- TODO (Clemens): the CAD moves to models/; then update this row, the Hardware paragraph below and the README. -->

The ROS 2 packages in `src/`:

| Package | Role |
| --- | --- |
| `esp_bridge` | serial link to the ESP32; speed controller, steering table, clock sync; `pwm_sweep` for bench tests |
| `ekf` | localisation (EKF and field map), scan processing, race controller, parking and unparking |
| `camera_lidar_fusion` | colour for every LiDAR point; camera calibration tools |
| `robot_msgs` | message definitions |
| `sllidar_ros2`, `bno055`, `ros_deep_learning` | sensor drivers from upstream projects, see below |
| `robot_vision`, `camera_capture` | the first vehicle generation (YOLO) and dataset capture; no longer started |

## Building and flashing

**Software environment.** [`setup/README.md`](../../setup/README.md) takes a
fresh Jetson Orin Nano to a running vehicle in seven steps: JetPack 6.2, clone
with Git LFS and submodules, device rules for `/dev/rplidar` and `/dev/picam`, container
image, workspace build, firmware, autostart. The container image is described by
[`setup/Dockerfile`](../../setup/Dockerfile). The original image had been set up
by hand; the Dockerfile was reconstructed from the running container and lists
the installed versions, but it has not been test-built.

Three packages come from upstream projects. Each is either pinned or included
together with its changes, so the repository builds exactly what the vehicle
runs:

| Package | Source | State |
| --- | --- | --- |
| `ros_deep_learning` | dusty-nv/ros_deep_learning | submodule, unchanged, commit `5229849` |
| `sllidar_ros2` | Slamtec/sllidar_ros2 at `3430009` | included; port `/dev/rplidar`, motor at 900 rpm for 15 Hz |
| `bno055` | flynneva/bno055 at `45e1ff1` | included; parameter `publish_only_imu` |

**Firmware.** A PlatformIO project with the platform release pinned
(pioarduino 54.03.21, Arduino core 3.x) and its libraries listed in
[`platformio.ini`](../../src/esp_firmware/platformio.ini):

```bash
cd src/esp_firmware
pio run -t upload        # over the USB-C port of the main board
```

The serial protocol between the Jetson and the ESP32 is specified in
[`src/esp_firmware/docs/JETSON_BRIDGE.md`](../../src/esp_firmware/docs/JETSON_BRIDGE.md).

**Hardware.** The main PCB is a KiCad project in
[`schemes/MainPCB`](../../schemes/MainPCB); BOM and placement files for
assembly are in its `production/` folder, the Gerber files are generated from the
board when ordering. The CAD is in [`models`](../../models) (STEP) and
[`src/Hardware`](../../src/Hardware) (Fusion 360 source, STL).

## Running the vehicle

At power-on `robot.service` runs [`src/start_robot.sh`](../../src/start_robot.sh).
It starts the container and one `tmux` window per node — LiDAR, IMU, ESP32
bridge, camera, fusion, EKF, scan processor, race controller, Foxglove. The run
itself starts with the start button, as the rules require: one switch powers
the vehicle, one button starts the program.

The challenge is selected at the top of the start script:

| Variable | Values | Effect |
| --- | --- | --- |
| `RACE_MODE` | `open`, `obstacle` | map and pillar detection for the challenge |
| `UNPARK` | `true`, `false` | start from the parking bay |
| `PACE` | `slow`, `medium`, `fast` | speed profile of the race controller |

`tmux attach -t robot_session` shows every node; Foxglove shows the map, the
scan and the camera live. `./start_robot.sh --calib` additionally starts the
camera calibration tool.

## Testing workflow

Every change is checked on three levels.

**Without hardware.** Unit tests run on any computer, without ROS:

```bash
cd src/camera_lidar_fusion && python3 -m pytest test -q   # 56 tests: fisheye model, colours, blind sectors, white point
cd src/ekf/ekf && python3 test_unpark.py                  # unpark geometry; likewise the other test_*.py,
                                                          # some of which take a recorded bag as argument
pip install -r docs/analysis/requirements.txt
cd docs/analysis && python3 -m pytest -q tests            # 14 tests: analysis toolkit on synthetic bags
```

<!-- TODO: src/esp_bridge/test/test_esp_protocol.py imports EspProtocol,
PacketParser and decode_* from esp_serial_bridge, which no longer exist after the
bridge was restructured (now FrameParser, parse_*). Rewrite it against the current
API, then list it here. -->

**On the bench.** Measurements are repeatable by construction: the speed sweep
`ros2 run esp_bridge pwm_sweep` drives identical steps before and after a change,
and hand measurements go to [`docs/data/manual/`](../data/manual) — see
[chapter 2](03-power-sensors.md).

**On the field.** Every run is recorded as a rosbag. The scripts in
[`docs/analysis/`](../analysis) turn the bags into metrics and figures; the
results over all recorded runs are in [chapter 3](04-software.md).

On every push to `main`, GitHub Actions checks the README length and
untranslated text and builds this journal as a PDF.

## Versions

| Version | Commit | State |
| --- | --- | --- |
| [`v1.0`](https://github.com/Mac-Lego-RMS/MaecLEGO-WRO-FE-2026/releases/tag/v1.0) | `40f0dad` | national final, June 2026 |
| `v2.0` | — | international final; tagged with the submission |

Each version is a tagged release with notes on GitHub. For every tag starting
with `v`, the CI attaches the journal PDF to the release. Between the versions
the history has more than 300 commits since November 2025, each one change with
a message that says what changed and why.
