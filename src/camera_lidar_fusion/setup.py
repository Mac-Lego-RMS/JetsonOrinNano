import os
from glob import glob

from setuptools import find_packages, setup

package_name = 'camera_lidar_fusion'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='macjetson',
    maintainer_email='maeclegorms@gmail.com',
    description='Fusion der liegenden 360-Grad-Fisheye-Kamera mit dem 2D-Lidar: '
                'Farbe je Lidar-Punkt (CSV) und Kalibrierung der Kameraverdrehung.',
    license='MIT',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'lidar_pixel_mapper = camera_lidar_fusion.lidar_pixel_mapper:main',
            'rotation_calibration = camera_lidar_fusion.rotation_calibration:main',
        ],
    },
)
