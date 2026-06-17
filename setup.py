from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'capytown_esan'

setup(
    name=package_name,
    version='1.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
        (os.path.join('share', package_name, 'maps'), glob('maps/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Grupo CapyTown',
    maintainer_email='grupo@esan.edu.pe',
    description='Paquete ROS2 del proyecto CapyTown - RC-1 La Manzana del Tambo',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'lane_detector = capytown_esan.lane_detector:main',
            'lane_controller = capytown_esan.lane_controller:main',
            'calib_hsv_lab = capytown_esan.calib_hsv_lab:main',
            'teleop_key = capytown_esan.teleop_key:main',
            'calibrate_beff = capytown_esan.calibrate_beff:main',
            'error_odom = capytown_esan.error_odom:main',
            'square = capytown_esan.move_square:main',
        ],
    },
)
