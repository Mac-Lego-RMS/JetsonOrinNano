from setuptools import find_packages, setup

package_name = 'robot_vision'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='root',
    maintainer_email='root@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'raw_camera = robot_vision.raw_camera:main',
            'yolo_detector = robot_vision.yolo_detector:main',
            'obstacle_follower = robot_vision.obstacle_follower:main',
            'testBench = robot_vision.testBench:main',
            'coord_test = robot_vision.coord_test:main',
            'turn_radius_calibration = robot_vision.turn_radius_calibration:main',
            'steering_lib = robot_vision.steering_lib:main',
            'camera_lib = robot_vision.camera_lib:main',
            'calibrate_camera = robot_vision.calibrate_camera:main',
        ],
    },
)
