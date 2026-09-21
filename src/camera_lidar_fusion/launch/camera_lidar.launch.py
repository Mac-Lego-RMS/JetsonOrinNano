"""Startet Lidar-Pixel-Mapping bzw. die Rotationskalibrierung.

    ros2 launch camera_lidar_fusion camera_lidar.launch.py
    ros2 launch camera_lidar_fusion camera_lidar.launch.py mode:=calib
    ros2 launch camera_lidar_fusion camera_lidar.launch.py scan_topic:=/ldlidar_node/scan

Lidar und Kamera muessen bereits laufen (siehe start_robot.sh).
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node

CALIB_FILE = '/workspace/config/fisheye_calib.yaml'


def generate_launch_description():
    mode = LaunchConfiguration('mode')
    scan_topic = LaunchConfiguration('scan_topic')
    image_topic = LaunchConfiguration('image_topic')
    calib_file = LaunchConfiguration('calib_file')
    csv_mode = LaunchConfiguration('csv_mode')

    common = [
        {'scan_topic': scan_topic},
        {'image_topic': image_topic},
        {'calib_file': calib_file},
    ]

    return LaunchDescription([
        DeclareLaunchArgument('mode', default_value='map',
                              description='map = Farbe je Lidar-Punkt, calib = Kalibrierung'),
        DeclareLaunchArgument('scan_topic', default_value='/scan'),
        DeclareLaunchArgument('image_topic', default_value='/video_source/raw'),
        DeclareLaunchArgument('calib_file', default_value=CALIB_FILE),
        DeclareLaunchArgument('csv_mode', default_value='trigger',
                              description='trigger | continuous | off'),

        Node(
            package='camera_lidar_fusion',
            executable='lidar_pixel_mapper',
            name='lidar_pixel_mapper',
            output='screen',
            emulate_tty=True,
            parameters=common + [{'csv_mode': csv_mode}],
            condition=IfCondition(PythonExpression(['"', mode, '" == "map"'])),
        ),

        Node(
            package='camera_lidar_fusion',
            executable='rotation_calibration',
            name='camera_rotation_calibration',
            output='screen',
            emulate_tty=True,
            parameters=common,
            condition=IfCondition(PythonExpression(['"', mode, '" == "calib"'])),
        ),
    ])
