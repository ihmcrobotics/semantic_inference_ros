#!/usr/bin/env python3

import os
from pathlib import Path
import sys

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    python_executable = str(
        Path(os.environ.get("VIRTUAL_ENV", sys.prefix)) / "bin" / "python"
    )

    semantic_inference_ros_share = Path(
        get_package_share_directory("semantic_inference_ros")
    )
    semantic_inference_python_share = Path(
        get_package_share_directory("semantic_inference_python")
    )
    semantic_inference_share = Path(
        get_package_share_directory("semantic_inference")
    )

    verbose = LaunchConfiguration("verbose")
    python_env = LaunchConfiguration("python_env")
    navigation_topic = LaunchConfiguration("navigation_topic")
    find_paths_service_name = LaunchConfiguration(
        "find_paths_service_name"
    )
    evaluating_objects_topic = LaunchConfiguration(
        "evaluating_objects_topic"
    )
    reset_visualization_service = LaunchConfiguration(
        "reset_visualization_service"
    )
    config_file = LaunchConfiguration("config_file")
    llm_response_parser_prompt_path = LaunchConfiguration(
        "llm_response_parser_prompt_path"
    )
    colormap_path = LaunchConfiguration("colormap_path")
    use_cuda = LaunchConfiguration("use_cuda")

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "verbose",
                default_value="true",
                description="Show output via stdout",
            ),
            DeclareLaunchArgument(
                "python_env",
                default_value=python_executable,
                description="Python interpreter for semantic inference",
            ),
            DeclareLaunchArgument(
                "navigation_topic",
                default_value=(
                    "/hydra_ros_node/navigation/object_search_output"
                ),
                description="Navigation output topic",
            ),
            DeclareLaunchArgument(
                "find_paths_service_name",
                default_value="/hydra_ros_node/navigation/find_paths",
                description="Find-paths service",
            ),
            DeclareLaunchArgument(
                "evaluating_objects_topic",
                default_value=(
                    "/hydra_ros_node/navigation/evaluating_objects"
                ),
                description="Topic for evaluating objects",
            ),
            DeclareLaunchArgument(
                "reset_visualization_service",
                default_value=(
                    "/hydra_dsg_visualizer/"
                    "reset_object_search_coloring"
                ),
                description="Service used to reset visualization",
            ),
            DeclareLaunchArgument(
                "config_file",
                default_value=str(
                    semantic_inference_ros_share
                    / "config"
                    / "vlm_for_navigation.yaml"
                ),
                description="Path to the configuration file",
            ),
            DeclareLaunchArgument(
                "llm_response_parser_prompt_path",
                default_value=str(
                    semantic_inference_python_share
                    / "config"
                    / "system_prompts"
                    / "llm_response_parser_prompt.txt"
                ),
                description="LLM response parser prompt",
            ),
            DeclareLaunchArgument(
                "colormap_path",
                default_value=str(
                    semantic_inference_share
                    / "config"
                    / "alex.csv"
                ),
                description="Visualization colormap",
            ),
            DeclareLaunchArgument(
                "use_cuda",
                default_value="true",
            ),
            Node(
                package="semantic_inference_ros",
                executable="vlm_for_navigation_node",
                name="vlm_for_navigation_node",
                namespace="semantic_inference",
                output="screen",
                prefix=python_env,
                parameters=[
                    {
                        "config_path":
                            config_file,
                        "use_cuda":
                            use_cuda,
                        "vlm_reasoning.verbose":
                            verbose,
                        "vlm_reasoning.llm_response_parser_prompt_path":
                            llm_response_parser_prompt_path,
                        "recolor.colormap_path":
                            colormap_path,
                    },
                ],
                remappings=[
                    (
                        "/semantic_inference/navigation",
                        navigation_topic,
                    ),
                    (
                        "/semantic_inference/find_paths",
                        find_paths_service_name,
                    ),
                    (
                        "/semantic_inference/evaluating_objects",
                        evaluating_objects_topic,
                    ),
                    (
                        "/semantic_inference/reset_visualization",
                        reset_visualization_service,
                    ),
                ],
            ),
        ]
    )
