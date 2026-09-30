# Environment of the vehicle's ROS 2 stack -- the image src/start_robot.sh
# starts as my_robot_base_yolo2.
#
# RECONSTRUCTED, NOT TEST-BUILT. The original image was set up by hand and
# saved with `docker commit`, so no recipe existed. This file lists what the
# running container contains, as inspected on 2026-09-30:
#
#   Ubuntu 22.04.5, Python 3.10, CUDA 12.6 (compute capability 8.7)
#   ROS 2 Humble from the ROS apt repository
#   jetson-inference / jetson-utils built from source at 45da40a8
#   Python packages the nodes import, at the versions installed
#
# Host: Jetson Orin Nano, JetPack 6.2 (L4T R36.4.3).
#
#   docker build -t my_robot_base_yolo2 setup/
#
# The repository itself is not copied into the image: start_robot.sh mounts
# it at /workspace and the workspace is built there (see setup/README.md).

FROM nvcr.io/nvidia/l4t-jetpack:r36.4.0

ENV DEBIAN_FRONTEND=noninteractive \
    LANG=C.UTF-8 \
    PYTHONUNBUFFERED=1

# --- ROS 2 Humble ------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl gnupg lsb-release ca-certificates \
    && curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
        -o /usr/share/keyrings/ros-archive-keyring.gpg \
    && echo "deb [arch=arm64 signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] \
http://packages.ros.org/ros2/ubuntu jammy main" > /etc/apt/sources.list.d/ros2.list \
    && apt-get update && apt-get install -y --no-install-recommends \
        ros-humble-ros-base \
        ros-humble-cv-bridge \
        ros-humble-diagnostic-updater \
        ros-humble-foxglove-bridge \
        ros-humble-vision-msgs \
        python3-colcon-common-extensions \
        python3-rosdep \
        python3-pip \
        python3-opencv \
        python3-serial \
        python3-yaml \
        v4l-utils \
        git cmake build-essential \
    && rm -rf /var/lib/apt/lists/*

# --- jetson-inference / jetson-utils ------------------------------------------
# Needed by ros_deep_learning, whose video_source node reads the USB camera.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libglew-dev glew-utils libsoup2.4-dev libjson-glib-dev \
        libgstrtspserver-1.0-dev avahi-utils \
        gstreamer1.0-tools gstreamer1.0-libav \
        gstreamer1.0-plugins-base gstreamer1.0-plugins-good \
        gstreamer1.0-plugins-bad gstreamer1.0-plugins-ugly \
    && rm -rf /var/lib/apt/lists/* \
    && git clone --recursive https://github.com/dusty-nv/jetson-inference /opt/jetson-inference \
    && cd /opt/jetson-inference && git checkout 45da40a8 \
    && git submodule update --init --recursive \
    && mkdir build && cd build \
    && cmake -DBUILD_INTERACTIVE=NO .. \
    && make -j"$(nproc)" && make install && ldconfig

# --- Python packages imported by the nodes ------------------------------------
# jetson-stats provides jtop for the /jtop/* CPU and temperature topics.
RUN pip3 install --no-cache-dir \
        numpy==1.26.4 \
        scipy==1.15.3 \
        matplotlib==3.10.8 \
        pyserial==3.5 \
        jetson-stats==7.2.1

WORKDIR /workspace
CMD ["bash"]
