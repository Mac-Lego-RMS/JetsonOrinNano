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
    description='Dead-Reckoning EKF (Gyro + Encoder) fuer den WRO-Roboter',
    license='TODO: License declaration',
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
        ],
    },
)
