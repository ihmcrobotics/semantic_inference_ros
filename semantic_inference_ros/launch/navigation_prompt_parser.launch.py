#!/usr/bin/env python3

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
    semantic_inference_python_share = Path(
        get_package_share_directory("semantic_inference_python")
    )
    hydra_share = Path(get_package_share_directory("hydra"))

    verbose = LaunchConfiguration("verbose")
    python_env = LaunchConfiguration("python_env")
    config_file = LaunchConfiguration("config_file")
    system_prompts_path = LaunchConfiguration("system_prompts_path")
    examples_path = LaunchConfiguration("examples_path")
    labels_path = LaunchConfiguration("labels_path")

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "verbose",
                default_value="true",
                description="Show output via stdout",
            ),
            DeclareLaunchArgument(
                "python_env",
                default_value=str(
                    semantic_inference_python_share
                    / "ros_semantics_env"
                    / "bin"
                    / "python"
                ),
                description="Python environment for semantic inference",
            ),
            DeclareLaunchArgument(
                "config_file",
                default_value=str(
                    semantic_inference_ros_share
                    / "config"
                    / "navigation_prompt_parser.yaml"
                ),
                description="Path to the configuration file",
            ),
            DeclareLaunchArgument(
                "system_prompts_path",
                default_value=str(
                    semantic_inference_python_share
                    / "config"
                    / "system_prompts"
                    / "gpt_system_prompt.txt"
                ),
                description="Path to the system prompts file",
            ),
            DeclareLaunchArgument(
                "examples_path",
                default_value=str(
                    semantic_inference_python_share
                    / "config"
                    / "system_prompts"
                    / "examples.txt"
                ),
                description="Path to the examples file",
            ),
            DeclareLaunchArgument(
                "labels_path",
                default_value=str(
                    hydra_share
                    / "config"
                    / "label_spaces"
                    / "anymal_label_space.yaml"
                ),
                description="Path to the labels file",
            ),
            Node(
                package="semantic_inference_ros",
                executable="navigation_prompt_service",
                name="navigation_prompt_service",
                namespace="semantic_inference",
                output="screen",
                prefix=python_env,
                parameters=[
                    {
                        "config_path":
                            config_file,
                        "navigation_prompter.system_prompts_path":
                            system_prompts_path,
                        "navigation_prompter.examples_path":
                            examples_path,
                        "navigation_prompter.labels_path":
                            labels_path,
                    },
                ],
            ),
        ]
    )