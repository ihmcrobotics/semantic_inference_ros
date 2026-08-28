#!/usr/bin/env python3
"""Launch fixed-class YOLOE with OpenCLIP embeddings and Alex labels."""

import os
from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import SetEnvironmentVariable
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    """Create the open-vocabulary segmentation launch description."""
    assets_root = Path(
        os.environ.get(
            "SCENE_GRAPH_ASSETS",
            Path.cwd() / "scene_graph_assets",
        )
    ).expanduser()
    semantic_ros_share = Path(
        get_package_share_directory("semantic_inference_ros")
    )
    semantic_share = Path(
        get_package_share_directory("semantic_inference")
    )
    hydra_share = Path(get_package_share_directory("hydra"))

    return LaunchDescription(
        [
            SetEnvironmentVariable(
                "SCENE_GRAPH_ASSETS",
                str(assets_root),
            ),
            Node(
                package="semantic_inference_ros",
                executable="openset_segmentation_node",
                name="semantic_inference",
                namespace="semantic_inference",
                output="screen",
                parameters=[
                    {
                        "config_path": str(
                            semantic_ros_share
                            / "config"
                            / "openset_segmentation.yaml"
                        ),
                        "labelspace_path": str(
                            hydra_share
                            / "config"
                            / "label_spaces"
                            / "alex_label_space.yaml"
                        ),
                        "label_grouping_path": str(
                            semantic_share
                            / "config"
                            / "label_groupings"
                            / "alex.yaml"
                        ),
                        "recolor.colormap_path": str(
                            semantic_share
                            / "config"
                            / "distinct_150_colors.csv"
                        ),
                        "model.cuda": True,
                        "worker.min_separation_s": 0.5,
                        "playback_mode": False,
                        "print_inference_time": False,
                    }
                ],
            ),
        ]
    )
