#!/usr/bin/env python3

import os
from pathlib import Path
import sys

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import (
    LaunchConfiguration,
    PathJoinSubstitution,
    PythonExpression,
)
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    python_executable = str(
        Path(os.environ.get("VIRTUAL_ENV", sys.prefix)) / "bin" / "python"
    )

    semantic_inference_ros_share = Path(
        get_package_share_directory("semantic_inference_ros")
    )
    semantic_inference_share = Path(
        get_package_share_directory("semantic_inference")
    )
    hydra_share = Path(get_package_share_directory("hydra"))

    playback_mode = LaunchConfiguration("playback_mode")
    print_inference_time = LaunchConfiguration("print_inference_time")
    min_separation_s = LaunchConfiguration("min_separation_s")
    labelspace_dir = LaunchConfiguration("labelspace_dir")
    labelspace_name = LaunchConfiguration("labelspace_name")
    camera_info_topic = LaunchConfiguration("camera_info_topic")
    active_object_edges_topic = LaunchConfiguration(
        "active_object_edges_topic"
    )
    odom_frame = LaunchConfiguration("odom_frame")
    sensor_frame = LaunchConfiguration("sensor_frame")
    colormap_path = LaunchConfiguration("colormap_path")
    config_path_vlm = LaunchConfiguration("config_path_vlm")
    pub_vlm_annotations = LaunchConfiguration("pub_vlm_annotations")
    height = LaunchConfiguration("height")
    width = LaunchConfiguration("width")
    vlm_queue_size = LaunchConfiguration("vlm_queue_size")
    python_env = LaunchConfiguration("python_env")

    labelspace_file = PathJoinSubstitution(
        [
            labelspace_dir,
            PythonExpression(
                ["'", labelspace_name, "' + '_label_space.yaml'"]
            ),
        ]
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "playback_mode",
                default_value="false",
                description="Run in playback mode with compressed images",
            ),
            DeclareLaunchArgument(
                "print_inference_time",
                default_value="false",
                description="Print inference time",
            ),
            DeclareLaunchArgument(
                "min_separation_s",
                default_value="0.7",
                description="Enforced separation between input images",
            ),
            DeclareLaunchArgument(
                "labelspace_dir",
                default_value=str(
                    hydra_share / "config" / "label_spaces"
                ),
            ),
            DeclareLaunchArgument(
                "labelspace_name",
                default_value="ade20k_full",
                description="Name of the label space to use",
            ),
            DeclareLaunchArgument(
                "camera_info_topic",
                default_value="/camera/color/camera_info",
                description="Camera-info topic for undistortion",
            ),
            DeclareLaunchArgument(
                "active_object_edges_topic",
                default_value=(
                    "/hydra_ros_node/backend/active_object_edges"
                ),
                description="Topic for active object edges",
            ),
            DeclareLaunchArgument(
                "odom_frame",
                default_value="world",
                description="World frame for VLM node",
            ),
            DeclareLaunchArgument(
                "sensor_frame",
                default_value="camera_color_optical_frame",
                description="Camera frame for VLM node",
            ),
            DeclareLaunchArgument(
                "colormap_path",
                default_value=str(
                    semantic_inference_share
                    / "config"
                    / "distinct_150_colors.csv"
                ),
                description="Visualization colormap",
            ),
            DeclareLaunchArgument(
                "config_path_vlm",
                default_value=str(
                    semantic_inference_ros_share
                    / "config"
                    / "vlm.yaml"
                ),
                description="VLM configuration file",
            ),
            DeclareLaunchArgument(
                "pub_vlm_annotations",
                default_value="true",
                description="Publish VLM annotations",
            ),
            DeclareLaunchArgument(
                "height",
                default_value="720",
                description="Height of the input image",
            ),
            DeclareLaunchArgument(
                "width",
                default_value="1280",
                description="Width of the input image",
            ),
            DeclareLaunchArgument(
                "vlm_queue_size",
                default_value="100",
                description="Queue size for VLM node",
            ),
            DeclareLaunchArgument(
                "python_env",
                default_value=python_executable,
                description="Python interpreter for semantic inference",
            ),
            Node(
                package="semantic_inference_ros",
                executable="vlm_features_node",
                name="vlm_features_node",
                namespace="semantic_inference",
                output="screen",
                prefix=python_env,
                parameters=[
                    {
                        "config_path":
                            config_path_vlm,
                        "labelspace_path":
                            labelspace_file,
                        "model.publish_masks":
                            pub_vlm_annotations,
                        "model.vlm.height":
                            height,
                        "model.vlm.width":
                            width,
                        "recolor.colormap_path":
                            colormap_path,
                        "worker.min_separation_s":
                            min_separation_s,
                        "worker.queue_size":
                            vlm_queue_size,
                        "playback_mode":
                            playback_mode,
                        "world_frame":
                            odom_frame,
                        "camera_frame":
                            sensor_frame,
                        "print_inference_time":
                            print_inference_time,
                    },
                ],
                remappings=[
                    (
                        "/semantic_inference/color/camera_info",
                        camera_info_topic,
                    ),
                    (
                        "/semantic_inference/active_object_edges",
                        active_object_edges_topic,
                    ),
                ],
            ),
        ]
    )
