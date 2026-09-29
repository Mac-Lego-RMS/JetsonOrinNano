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
        package='esp_bridge',
        executable='esp_serial_bridge',
        name='esp_serial_bridge',
        output='screen'
    )

    # 4. BNO055 IMU
    # In launch files, absolute paths for parameter files are safer.
    # This works if the file is in the workspace root (/workspace).
    # Otherwise use os.path.join(get_package_share_directory('package_name'), 'config', 'bno055_params.yaml')
    bno055_node = Node(
        package='bno055',
        executable='bno055',
        name='bno055',
        parameters=[os.path.abspath('bno055_params.yaml')],
        output='screen'
    )

    # 5. Include the LDLidar launch file
    # Assumes the file is in the 'launch' directory of the 'ldlidar_node' package
    ldlidar_launch_dir = get_package_share_directory('ldlidar_node')
    ldlidar_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(ldlidar_launch_dir, 'launch', 'ldlidar_auto.launch.py')
        )
    )

    # Return the launch graph
    return LaunchDescription([
        foxglove_node,
        camera_node,
        esp_bridge_node,
        bno055_node,
        ldlidar_launch
    ])
