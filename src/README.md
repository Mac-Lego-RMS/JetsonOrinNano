# 🤖 JetsonOrinNano — WRO Future Engineers

> Autonomous self-driving robot for the **World Robot Olympiad (WRO) — Future Engineers** category, built around an **NVIDIA Jetson Orin Nano** and an **ESP32-S3** real-time motor controller.

<p align="center">
  <img src="https://img.shields.io/badge/Competition-WRO%20Future%20Engineers-FF6B00?style=for-the-badge" alt="WRO Future Engineers">
  <img src="https://img.shields.io/badge/ROS%202-Humble-22314E?style=for-the-badge&logo=ros&logoColor=white" alt="ROS 2 Humble">
  <img src="https://img.shields.io/badge/Python-3.10-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python 3.10">
  <img src="https://img.shields.io/badge/C++-ESP32%20Firmware-00599C?style=for-the-badge&logo=cplusplus&logoColor=white" alt="C++">
</p>

<p align="center">
  <img src="https://img.shields.io/badge/NVIDIA-Jetson%20Orin%20Nano-76B900?style=flat-square&logo=nvidia&logoColor=white" alt="Jetson Orin Nano">
  <img src="https://img.shields.io/badge/ESP32--S3-Real--Time%20Control-E7352C?style=flat-square&logo=espressif&logoColor=white" alt="ESP32-S3">
  <img src="https://img.shields.io/badge/Vision-YOLO%20%2B%20TensorRT-00A3E0?style=flat-square" alt="YOLO TensorRT">
  <img src="https://img.shields.io/badge/LiDAR-Wall%20Follower-6E4AFF?style=flat-square" alt="LiDAR">
  <img src="https://img.shields.io/badge/Build-colcon-blue?style=flat-square" alt="colcon">
  <img src="https://img.shields.io/badge/Viz-Foxglove%20%2F%20RViz-FB6E2E?style=flat-square" alt="Foxglove">
</p>

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Hardware Stack](#2-hardware-stack)
3. [Software Architecture](#3-software-architecture)
4. [Repository Structure](#4-repository-structure)
5. [Installation & Setup](#5-installation--setup)
6. [Usage / How to Run](#6-usage--how-to-run)
7. [Features](#7-features)

---

## 1. Project Overview

This repository contains the complete software and firmware stack for our **WRO Future Engineers** autonomous vehicle. The robot must navigate a walled track fully autonomously — completing multiple laps, staying within its lane, and avoiding colored traffic-sign obstacles (red and green pillars) according to the WRO rulebook.

The system splits responsibilities across two compute layers:

- **NVIDIA Jetson Orin Nano** runs the high-level autonomy stack under **ROS 2 Humble** — LiDAR-based wall following, IMU-stabilized heading control, YOLO-based obstacle detection, and trajectory planning.
- **ESP32-S3** acts as a hard real-time actuator controller — driving the propulsion motor and steering servo, handling emergency stops, and reporting the start-button state back to the Jetson over a compact binary serial protocol.

The two boards communicate over a UART serial link, allowing the Jetson to focus on perception and decision-making while the ESP32 guarantees deterministic, low-latency motor and servo control.

**Primary goals**

- Reliable lap completion with consistent lane-keeping via a PID-controlled LiDAR wall follower.
- Robust obstacle avoidance by fusing camera (YOLO) detections with LiDAR distance measurements.
- Smooth, dynamically-computed cornering using kinematically valid turn radii.
- A safe, calibratable, and field-trimmable steering system that survives power cycles.

---

## 2. Hardware Stack

| Component | Role |
| --- | --- |
| **NVIDIA Jetson Orin Nano** | Main compute unit — runs ROS 2 Humble, perception, planning and control |
| **ESP32-S3** | Real-time microcontroller — motor PWM, steering servo, button & LED, safety watchdog |
| **2D LiDAR (LDRobot / `ldlidar_node`)** | 360° distance scanning for wall following and obstacle ranging |
| **Camera (CSI / USB)** | RGB image stream for YOLO obstacle (traffic-sign) detection |
| **BNO055 IMU** | Absolute orientation / yaw for heading stabilization and turn counting |
| **Feetech SC-series serial servo (SCSCL / SC09)** | Steering actuator with position feedback & load (torque) readout |
| **Planetary DC gear motor + PWM driver** | Propulsion, driven via direction + 10-bit PWM (20 kHz) |
| **Start button + status LED** | Hardware-interrupt start trigger and run indicator |
| **Custom Buck-Converter PCB** | Power distribution / voltage regulation for the Jetson, ESP32 and actuators |

> **Note:** The KiCad PCB design files for the custom buck-converter board are not yet committed to this repository. Add the `*.kicad_pro` / `*.kicad_sch` / `*.kicad_pcb` sources under a `pcb/` directory and update the [Repository Structure](#4-repository-structure) section accordingly.

### Jetson ↔ ESP32 Serial Protocol

Communication uses a compact framed binary protocol. Every command frame begins with the start byte `0xA5`:

| Command | Byte | Payload | Direction |
| --- | --- | --- | --- |
| `CMD_MOTOR` | `0x10` | `dir, pwm_hi, pwm_lo` (10-bit PWM) | Jetson → ESP |
| `CMD_SERVO` | `0x20` | `id, pos_hi, pos_lo` (±100 % steering) | Jetson → ESP |
| `CMD_LED` | `0x30` | `state` | Jetson → ESP |
| `CMD_CALIBRATE` | `0x40` | — (runs steering end-stop probe) | Jetson → ESP |
| `CMD_TORQUE` | `0x50` | — (prints servo load) | Jetson → ESP |
| `CMD_TRIM` | `0x60` | `action` (trim L / R / save) | Jetson → ESP |
| `CMD_BUTTON` | `0x70` | `0x01` = pressed | ESP → Jetson |
| `CMD_EMERGENCY` | `0xFF` | — (immediate motor stop) | Jetson → ESP |

The link runs at **115200 baud** on the Jetson side (`/dev/ttyTHS1`). The ESP32 enforces a **5 s watchdog** (`JETSON_TIMEOUT`) that zeroes the motor speed if no valid packet arrives.

---

## 3. Software Architecture

The autonomy stack is a set of **ROS 2 Humble** Python packages running inside a Docker container on the Jetson. Nodes communicate over standard ROS 2 topics; a single launch file brings the whole system up.

```
                ┌──────────────────────────────────────────────────────────┐
                │                    Jetson Orin Nano (ROS 2 Humble)          │
                │                                                            │
   LiDAR  ──────┤  ldlidar_node ──/ldlidar_node/scan──┐                      │
                │                                       ▼                      │
   Camera ──────┤  raw_camera ──/camera/image_raw──► yolo_detector           │
                │                                       │  (YOLO + TensorRT)   │
                │                                       ▼                      │
   IMU   ───────┤  bno055 ──/imu──────────────► wall_follower_logic          │
                │                                  (PID + state machine)       │
                │                                       │ /cmd_vel             │
                │                                       ▼                      │
                │                              esp_serial_bridge ──────────────┼──► UART
                │                                                            │   /dev/ttyTHS1
                │  foxglove_bridge  (live visualization / debugging)          │
                └──────────────────────────────────────────────────────────┘
                                                                                  │
                                                                                  ▼
                                                              ┌─────────────────────────────┐
                                                              │        ESP32-S3 (C++)         │
                                                              │  Core 0: motorControlTask     │
                                                              │  Core 1: serial parse + servo │
                                                              │  ISR: start-button interrupt  │
                                                              └─────────────────────────────┘
```

### Key components

- **`wall_follower_logic`** — the brain. A PID controller (`Kp=2.0, Ki=0.07, Kd=0.05`) drives lane-keeping from LiDAR wall estimates, with an IMU-stabilized heading and a state machine (`INITIALIZING → driving → turn APPROACH/…`) that counts and executes the required turns (target: **12**). It computes a dynamically valid curve radius (ideal `0.28 m`, min `0.20 m`, max kinematic `1.2 m`), estimates lane width on the fly, and publishes `Twist` commands on `/cmd_vel`, plus RViz/Foxglove markers and a planned `Path`.
- **`yolo_detector`** — loads a **YOLO TensorRT `.engine`** model via Ultralytics, detects red/green pillars (confidence > 0.8), and **fuses** each detection's camera bearing with the LiDAR scan to recover real-world distance. It emits avoidance commands (`AVOID_LEFT` / `AVOID_RIGHT` / `CLEAR`) on `/obstacle_cmd` and obstacle markers for visualization.
- **`esp_serial_bridge`** — translates ROS 2 topics (`/cmd_vel`, `/led_cmd`, `/calibrate_cmd`) into the binary serial protocol, and parses inbound ESP frames into `/button_state` and `/esp/esp_data`.
- **ESP32 firmware (`MainCodeESP.ino`)** — dual-core FreeRTOS: a dedicated motor-control task pinned to **Core 0**, the serial protocol parser + servo control on **Core 1**, a **hardware-interrupt** start button, and steering calibration/trim persisted to NVS flash.

### Topic map

| Topic | Type | Publisher → Subscriber |
| --- | --- | --- |
| `/ldlidar_node/scan` | `sensor_msgs/LaserScan` | ldlidar → wall_follower, yolo_detector |
| `/camera/image_raw` | `sensor_msgs/Image` | raw_camera → yolo_detector |
| `/imu` | `sensor_msgs/Imu` | bno055 → wall_follower |
| `/cmd_vel` | `geometry_msgs/Twist` | wall_follower → esp_serial_bridge |
| `/obstacle_cmd` | `std_msgs/String` | yolo_detector → wall_follower |
| `/detected_obstacles` | `visualization_msgs/Marker` | yolo_detector → RViz/Foxglove |
| `/button_state` | `std_msgs/Bool` | esp_serial_bridge → wall_follower |
| `/led_cmd`, `/calibrate_cmd` | `Bool` / `Empty` | wall_follower → esp_serial_bridge |

---

## 4. Repository Structure

```text
JetsonOrinNano/
├── MainCodeESP.ino              # ESP32-S3 firmware (C++ / Arduino, dual-core, binary protocol)
│
├── robot_vision/                # ROS 2 package — perception & launch
│   ├── launch/
│   │   └── bringup.launch.py    # Brings up Foxglove, camera, ESP bridge, IMU, LiDAR
│   ├── robot_vision/
│   │   ├── raw_camera.py        # Camera publisher (/camera/image_raw)
│   │   ├── yolo_detector.py     # YOLO + LiDAR fusion obstacle detection
│   │   ├── obstacle_follower.py # Obstacle-round avoidance logic
│   │   ├── turn_radius_calibration.py  # Dynamic curve-radius calibration
│   │   ├── camera_lib.py        # Camera helpers
│   │   ├── steering_lib.py      # Steering helpers
│   │   ├── calibrate_camera.py  # Intrinsic calibration
│   │   ├── camera_calib.json    # Camera intrinsics
│   │   └── wro_calibration.json # Track / run calibration
│   ├── package.xml
│   └── setup.py
│
├── wall_follower_robot/         # ROS 2 package — control & ESP bridge
│   ├── wall_follower_robot/
│   │   ├── wall_follower_logic.py  # PID wall follower + state machine (core autonomy)
│   │   ├── esp_serial_bridge.py    # ROS 2 ↔ ESP32 serial protocol bridge
│   │   └── yolo_vision_node.py     # Vision node entry point
│   ├── package.xml
│   └── setup.py
│
├── camera_capture/              # ROS 2 package — dataset capture
│   ├── camera_capture/
│   │   └── save_images.py       # Saves frames for YOLO training datasets
│   ├── package.xml
│   └── setup.py
│
└── .gitignore
```

> The third-party sensor drivers — `ldlidar_node`, `bno055`, and `foxglove_bridge` — are installed as ROS 2 dependencies (see [Installation](#5-installation--setup)) and are referenced by `bringup.launch.py` rather than vendored here.

---

## 5. Installation & Setup

### Prerequisites

- NVIDIA Jetson Orin Nano with JetPack and Docker installed
- ROS 2 **Humble** (provided via the Docker container below)
- A YOLO model exported to a TensorRT engine (`best.engine`)
- Arduino IDE or PlatformIO with **ESP32 core 3.x** (for the firmware)

### 5.1 — Jetson: ROS 2 Docker environment

The ROS 2 workspace expects to be mounted at `/workspace` inside the container (the launch file and YOLO loader use absolute `/workspace` paths, e.g. `/workspace/best.engine`).

```bash
# Clone the repository
git clone https://github.com/Mac-Lego-RMS/JetsonOrinNano.git
cd JetsonOrinNano

# Start a ROS 2 Humble container with device access (camera, LiDAR, ESP UART)
# Adjust the image tag to your Jetson ROS 2 base image.
docker run -it --rm \
  --runtime nvidia \
  --network host \
  --privileged \
  -v $(pwd):/workspace \
  -v /dev:/dev \
  --device /dev/ttyTHS1 \
  --name wro_ros \
  <your-ros2-humble-jetson-image>
```


### 5.2 — Build the ROS 2 workspace

Inside the container:

```bash
cd /workspace
source /opt/ros/humble/setup.bash

# Install dependencies (drivers used by bringup.launch.py)
sudo apt update
sudo apt install -y \
  ros-humble-foxglove-bridge \
  ros-humble-bno055 \
  ros-humble-ldlidar-node

pip install ultralytics opencv-python

# Build all packages
colcon build --symlink-install
source install/setup.bash
```

### 5.3 — Flash the ESP32-S3

The firmware depends on the **SCServo** library (Feetech serial servo) and the ESP32 `Preferences` library. Using the Arduino IDE:

```text
1. Install the ESP32 board package (core 3.x) via the Boards Manager.
2. Install the "SCServo" library.
3. Open MainCodeESP.ino and select your ESP32-S3 board.
4. Connect the ESP32-S3 over USB and click Upload.
```

Or with **arduino-cli**:

```bash
arduino-cli compile --fqbn esp32:esp32:esp32s3 MainCodeESP.ino
arduino-cli upload  --fqbn esp32:esp32:esp32s3 -p /dev/ttyUSB0 MainCodeESP.ino
```

---

## 6. Usage / How to Run

### One-shot: full bringup

The simplest path launches the entire sensor + control stack:

```bash
source /opt/ros/humble/setup.bash
source /workspace/install/setup.bash

ros2 launch robot_vision bringup.launch.py
```

This starts, in order: **Foxglove bridge**, **camera**, **ESP serial bridge**, **BNO055 IMU**, and the **LiDAR**.

### Recommended: tmux session (per-node visibility)

For debugging on the track it helps to run each subsystem in its own pane so logs stay readable:

```bash
# Start a named tmux session
tmux new-session -s wro

# Pane 1 — sensor + bridge bringup (LiDAR, IMU, camera, ESP bridge, Foxglove)
ros2 launch robot_vision bringup.launch.py

# Pane 2 (Ctrl-b ") — YOLO obstacle detection
ros2 run robot_vision yolo_detector

# Pane 3 (Ctrl-b ") — the autonomy brain (PID wall follower)
ros2 run wall_follower_robot wall_follower_logic
```

Detach with `Ctrl-b d`, reattach with `tmux attach -t wro`.

### Individual nodes & utilities

```bash
# Calibrate steering end-stops on the ESP (also bound to button/serial 'C')
ros2 topic pub --once /calibrate_cmd std_msgs/msg/Empty "{}"

# Dynamic curve-radius calibration
ros2 run robot_vision turn_radius_calibration

# Capture a training dataset
ros2 run camera_capture save_images
```

### Starting a run

Once the stack is up, press the **physical start button** on the robot. The ESP32 ISR captures the press and forwards a `/button_state` event; `wall_follower_logic` then leaves its `INITIALIZING` state and begins the run.

### Visualization

Open [Foxglove Studio](https://foxglove.dev/) and connect to the robot's `ws://<jetson-ip>:8765` (Foxglove Bridge). Useful topics: `/camera/yolo_debug`, `/detected_obstacles`, `/wall_follower_markers`, `/planned_trajectory`.

---

## 7. Features

- **Dynamic curve radius** — corner trajectories are computed at runtime within kinematic limits (ideal `0.28 m`, minimum `0.20 m`, max `1.2 m`) instead of using fixed turns, producing smoother, faster, and more reliable cornering.
- **PID wall following with IMU stabilization** — a tuned PID controller (`Kp=2.0, Ki=0.07, Kd=0.05`) keeps the robot centered, while BNO055 yaw data stabilizes heading and drives precise turn counting (target 12 turns).
- **Phantom-wall filtering** — LiDAR returns are validated by distance, width, and clustering checks so that noise and spurious "phantom walls" are rejected before they can corrupt the wall estimate or trigger false turns.
- **ESP32 hardware interrupts** — the start button is handled by a true ISR (`IRAM_ATTR`) with debouncing, guaranteeing the press is never missed even under heavy serial load.
- **Dual-core real-time control** — the ESP32-S3 pins the motor-control loop to **Core 0** (FreeRTOS task) while serial parsing and servo control run on **Core 1**, keeping propulsion deterministic.
- **Camera + LiDAR sensor fusion** — YOLO detections (red/green pillars) are matched to LiDAR clusters by bearing to recover accurate obstacle distance, enabling well-timed `AVOID_LEFT` / `AVOID_RIGHT` maneuvers.
- **Self-calibrating, persistent steering** — the firmware probes physical steering end-stops, derives the center, supports live trim (and saves it to NVS flash), and enforces an 80 % throw limit to protect the servo.
- **Safety watchdog & emergency stop** — a 5 s serial watchdog and a dedicated `0xFF` emergency-stop command immediately cut motor power if the link drops.
- **Live telemetry** — Foxglove Bridge streams annotated camera frames, obstacle markers, the planned trajectory, and a run timer for real-time debugging.

---

<p align="center"><sub>Built for the World Robot Olympiad — Future Engineers · Mac-Lego-RMS</sub></p>
