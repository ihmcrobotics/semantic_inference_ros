#!/usr/bin/env python3
"""Launch the ROS 2 open-set RGB-D segmentation node."""

from pathlib import Path

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
    """Create the open-set segmentation launch description."""
    semantic_inference_ros_share = Path(
        get_package_share_directory("semantic_inference_ros")
    )

    semantic_inference_share = Path(
        get_package_share_directory("semantic_inference")
    )

    config_path = LaunchConfiguration("config_path")
    min_separation_s = LaunchConfiguration("min_separation_s")
    labelspace_name = LaunchConfiguration("labelspace_name")
    colormap_path = LaunchConfiguration("colormap_path")
    segmentation_cuda = LaunchConfiguration("segmentation_cuda")
    labelspace_dir = LaunchConfiguration("labelspace_dir")
    playback_mode = LaunchConfiguration("playback_mode")
    print_inference_time = LaunchConfiguration("print_inference_time")
    camera_info_topic = LaunchConfiguration("camera_info_topic")

    labelspace_filename = PythonExpression(
        [
            "'",
            labelspace_name,
            "_label_space.yaml'",
        ]
    )

    label_grouping_filename = PythonExpression(
        [
            "'",
            labelspace_name,
            ".yaml'",
        ]
    )

    labelspace_file = PathJoinSubstitution(
        [
            labelspace_dir,
            labelspace_filename,
        ]
    )

    label_grouping_file = PathJoinSubstitution(
        [
            str(semantic_inference_share),
            "config",
            "label_groupings",
            label_grouping_filename,
        ]
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "config_path",
                default_value=str(
                    semantic_inference_ros_share
                    / "config"
                    / "openset_segmentation.yaml"
                ),
                description=(
                    "Main configuration file for open-set segmentation"
                ),
            ),
            DeclareLaunchArgument(
                "min_separation_s",
                default_value="0.5",
                description="Minimum time between processed RGB-D frames",
            ),
            DeclareLaunchArgument(
                "labelspace_name",
                default_value="ade20k_full",
                description="Semantic label-space name",
            ),
            DeclareLaunchArgument(
                "colormap_path",
                default_value=str(
                    semantic_inference_share
                    / "config"
                    / "distinct_150_colors.csv"
                ),
                description="Visualization colormap CSV file",
            ),
            DeclareLaunchArgument(
                "segmentation_cuda",
                default_value="true",
                description="Use CUDA for segmentation inference",
            ),
            DeclareLaunchArgument(
                "labelspace_dir",
                default_value=str(
                    semantic_inference_ros_share
                    / "config"
                    / "label_spaces"
                ),
                description=(
                    "Directory containing label-space YAML files"
                ),
            ),
            DeclareLaunchArgument(
                "playback_mode",
                default_value="false",
                description=(
                    "Use compressed image messages during playback"
                ),
            ),
            DeclareLaunchArgument(
                "print_inference_time",
                default_value="false",
                description="Print segmentation inference timing",
            ),
            DeclareLaunchArgument(
                "camera_info_topic",
                default_value="/camera/color/camera_info",
                description="Input RGB camera-info topic",
            ),
            DeclareLaunchArgument("publish_semantic_image", default_value="false"),
            DeclareLaunchArgument("enable_clip", default_value="true",
                                  description="Disable only for segmentation performance tests"),
            Node(
                package="semantic_inference_ros",
                executable="openset_segmentation_node",
                name="semantic_inference",
                namespace="semantic_inference",
                output="screen",
                parameters=[
                    {
                        "config_path":
                            config_path,
                        "labelspace_path":
                            labelspace_file,
                        "label_grouping_path":
                            label_grouping_file,
                        "publish_semantic_image": LaunchConfiguration("publish_semantic_image"),
                        "model.enable_clip": LaunchConfiguration("enable_clip"),
                        "model.cuda":
                            segmentation_cuda,
                        "worker.min_separation_s":
                            min_separation_s,
                        "recolor.colormap_path":
                            colormap_path,
                        "playback_mode":
                            playback_mode,
                        "print_inference_time":
                            print_inference_time,
                    },
                ],
                remappings=[
                    (
                        "/semantic_inference/color/camera_info",
                        camera_info_topic,
                    ),
                ],
            ),
        ]
    )
