from setuptools import find_packages, setup

package_name = 'ekf'

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
    description='Dead-reckoning EKF (gyro + encoder) for the WRO robot',
    license='MIT',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'ekf_node = ekf.ekf_node:main',
            'scan_processor = ekf.scan_processor_node:main',
            'ekf_test = ekf.ekfTest:main',
            'scan_test = ekf.scanNode:main',
            'straight_controller = ekf.straight_controller_node:main',
            'approach_corner = ekf.approach_corner_node:main',
            'round1_controller = ekf.round1_controller_node:main',
            'unpark_test = ekf.unpark_test_node:main',
            'unpark_variants = ekf.unpark_variants_node:main',
            'speed_calib = ekf.speed_calib_node:main',
            'speed_verify = ekf.speed_verify:main',
            'steer_calib = ekf.steer_calib_node:main',
            'foxglove_overlay = ekf.foxglove_overlay_node:main',
        ],
    },
)
