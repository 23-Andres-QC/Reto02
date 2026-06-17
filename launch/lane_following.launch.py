#!/usr/bin/env python3
import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    pkg = get_package_share_directory('capytown_esan')

    hsv = os.path.join(pkg, 'config', 'hsv_params.yaml')
    pid = os.path.join(pkg, 'config', 'pid_params.yaml')

    return LaunchDescription([
        Node(
            package='capytown_esan',
            executable='lane_detector',
            name='lane_detector',
            parameters=[hsv],
            output='screen',
        ),
        Node(
            package='capytown_esan',
            executable='lane_controller',
            name='lane_controller',
            parameters=[pid],
            output='screen',
        ),
    ])
