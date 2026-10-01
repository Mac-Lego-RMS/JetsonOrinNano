# Setting up the vehicle from scratch

Everything the Jetson needs beyond this repository. Following these steps on a
fresh Jetson Orin Nano reproduces the software environment the vehicle runs.

| File | Goes to | Purpose |
|---|---|---|
| [`Dockerfile`](Dockerfile) | `docker build` | the ROS 2 environment (image `my_robot_base_yolo2`) |
| [`udev/99-rplidar.rules`](udev/99-rplidar.rules) | `/etc/udev/rules.d/` | fixed name `/dev/rplidar` for the LiDAR behind the CP2102N |
| [`udev/99-picam.rules`](udev/99-picam.rules) | `/etc/udev/rules.d/` | fixed name `/dev/picam` for the camera's image node |
| [`systemd/robot.service`](systemd/robot.service) | `/etc/systemd/system/` | starts the whole stack at power-on |

Calibration and parameter files the nodes read at run time live in the
repository root, where the container sees them as `/workspace/...`:
[`config/fisheye_calib.yaml`](../config/fisheye_calib.yaml) (camera–LiDAR
calibration), [`config/camera_calib.env`](../config/camera_calib.env) (camera
exposure measured on the field) and [`bno055_params.yaml`](../bno055_params.yaml).

## 1. Jetson

Flash JetPack 6.2 (L4T R36.4.3) on the Jetson Orin Nano, then on the host:

```bash
sudo apt install -y git git-lfs tmux docker.io
git lfs install                             # src/ holds CAD files stored with Git LFS
sudo pip3 install -U jetson-stats          # jtop and jtop.service
git clone https://github.com/dusty-nv/jetson-containers
bash jetson-containers/install.sh           # provides `jetson-containers run`
```

## 2. Repository

The start script expects the repository at `/home/macjetson/ros2_ws`
(`WORKSPACE` in [`src/start_robot.sh`](../src/start_robot.sh) — change it there
for another path).

```bash
git clone --recurse-submodules https://github.com/Mac-Lego-RMS/MaecLEGO-WRO-FE-2026 ~/ros2_ws
cd ~/ros2_ws
ln -s src/start_robot.sh start_robot.sh    # the path robot.service calls
```

## 3. Devices

```bash
sudo cp setup/udev/*.rules /etc/udev/rules.d/
sudo udevadm control --reload && sudo udevadm trigger
ls -l /dev/rplidar /dev/picam              # both must exist
```

## 4. Container image

```bash
docker build -t my_robot_base_yolo2 setup/
```

The [Dockerfile](Dockerfile) is reconstructed from the running container and has
not been test-built; the versions in it are the ones installed on the vehicle.

## 5. Workspace

Inside the container (`src/start_robot.sh` starts it; or
`docker exec -it yolo_dev bash`):

```bash
cd /workspace
# ros_deep_learning ships a ROS 1 package.xml; its README says to use the ROS 2 one
cp src/ros_deep_learning/package.ros2.xml src/ros_deep_learning/package.xml
source /opt/ros/humble/setup.bash
colcon build --symlink-install
```

Three packages in `src/` come from upstream projects:

| Package | Source | State |
|---|---|---|
| `ros_deep_learning` | [dusty-nv/ros_deep_learning](https://github.com/dusty-nv/ros_deep_learning) | submodule, unchanged, commit `5229849` |
| `sllidar_ros2` | [Slamtec/sllidar_ros2](https://github.com/Slamtec/sllidar_ros2) at `3430009` | included with two changes: port `/dev/rplidar`, motor at 900 rpm = 15 Hz |
| `bno055` | [flynneva/bno055](https://github.com/flynneva/bno055) at `45e1ff1` | included with one change: parameter `publish_only_imu` |

## 6. ESP32 firmware

Built and flashed with PlatformIO — see [`src/esp_firmware`](../src/esp_firmware).

## 7. Autostart

```bash
sudo cp setup/systemd/robot.service /etc/systemd/system/
sudo systemctl enable --now jtop.service robot.service
tmux attach -t robot_session               # watch the nodes
```

After power-on the stack comes up by itself; the run starts with the start
button on the vehicle.
