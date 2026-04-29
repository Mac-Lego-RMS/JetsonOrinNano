import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node

def generate_launch_description():
    
    # 1. Foxglove Bridge
    foxglove_node = Node(
        package='foxglove_bridge',
        executable='foxglove_bridge',
        name='foxglove_bridge',
        output='screen'
    )

    # 2. Raw Camera
    camera_node = Node(
        package='robot_vision',
        executable='raw_camera',
        name='raw_camera',
        output='screen'
    )

    # 3. ESP Serial Bridge
    esp_bridge_node = Node(
        package='wall_follower_robot',
        executable='esp_serial_bridge',
        name='esp_serial_bridge',
        output='screen'
    )

    # 4. BNO055 IMU
    # Hinweis: In Launch-Dateien sind absolute Pfade für Parameter-Dateien sicherer.
    # Wenn die Datei im Workspace-Root liegt (/workspace), funktioniert dies.
    # Andernfalls nutze os.path.join(get_package_share_directory('paketname'), 'config', 'bno055_params.yaml')
    bno055_node = Node(
        package='bno055',
        executable='bno055',
        name='bno055',
        parameters=[os.path.abspath('bno055_params.yaml')],
        output='screen'
    )

    # 5. LDLidar Launch-Datei einbinden
    # Nimmt an, dass die Datei im 'launch' Verzeichnis des 'ldlidar_node' Pakets liegt
    ldlidar_launch_dir = get_package_share_directory('ldlidar_node')
    ldlidar_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(ldlidar_launch_dir, 'launch', 'ldlidar_auto.launch.py')
        )
    )

    # Rückgabe des Launch-Graphen
    return LaunchDescription([
        foxglove_node,
        camera_node,
        esp_bridge_node,
        bno055_node,
        ldlidar_launch
    ])
