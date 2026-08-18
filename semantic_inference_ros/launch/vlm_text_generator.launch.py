#!/usr/bin/env python3

import os
import sys
from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    semantic_inference_ros_share = Path(
        get_package_share_directory("semantic_inference_ros")
    )
    python_executable = str(
        Path(os.environ.get("VIRTUAL_ENV", sys.prefix)) / "bin" / "python"
    )

    verbose = LaunchConfiguration("verbose")
    visual_relationships_encodings_topic = LaunchConfiguration(
        "visual_relationships_encodings_topic"
    )
    config_path = LaunchConfiguration(
        "vlm_labels_from_prompt_config_path"
    )
    height = LaunchConfiguration("height")
    width = LaunchConfiguration("width")
    python_env = LaunchConfiguration("python_env")

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "verbose",
                default_value="true",
                description="Show output via stdout",
            ),
            DeclareLaunchArgument(
                "visual_relationships_encodings_topic",
                default_value=(
                    "/hydra_ros_node/backend/vlm_relationships/"
                    "visual_relationships_encodings"
                ),
            ),
            DeclareLaunchArgument(
                "vlm_labels_from_prompt_config_path",
                default_value=str(
                    semantic_inference_ros_share
                    / "config"
                    / "vlm_prompting.yaml"
                ),
                description="Configuration file for VLM prompting",
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
                "python_env",
                default_value=python_executable,
                description="Python environment for semantic inference",
            ),
            Node(
                package="semantic_inference_ros",
                executable="encode_vlm_labels_node",
                name="encode_vlm_labels_node",
                namespace="semantic_inference",
                output="screen",
                prefix=python_env,
                parameters=[
                    {
                        "config_path":
                            config_path,
                        "vlm.height":
                            height,
                        "vlm.width":
                            width,
                    },
                ],
                remappings=[
                    (
                        "/semantic_inference/"
                        "visual_relationships_encodings",
                        visual_relationships_encodings_topic,
                    ),
                ],
            ),
        ]
    )
