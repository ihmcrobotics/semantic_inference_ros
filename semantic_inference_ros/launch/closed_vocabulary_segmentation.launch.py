#!/usr/bin/env python3
"""Launch ADE20K EfficientViT closed-vocabulary segmentation."""

import os
from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import ComposableNodeContainer
from launch_ros.descriptions import ComposableNode


def generate_launch_description() -> LaunchDescription:
    """Create the closed-vocabulary segmentation launch description."""
    assets_root = Path(
        os.environ.get(
            "SCENE_GRAPH_ASSETS",
            Path.cwd() / "scene_graph_assets",
        )
    ).expanduser()
    model_directory = (
        assets_root / "models" / "segmentation" / "pretrained" / "ade20k"
    )
    semantic_share = Path(
        get_package_share_directory("semantic_inference")
    )

    segmentation_node = ComposableNode(
        package="semantic_inference_ros",
        plugin="semantic_inference::SegmentationNode",
        name="semantic_inference",
        namespace="semantic_inference",
        parameters=[
            {
                "config_file": str(
                    semantic_share
                    / "config"
                    / "models"
                    / "ade20k-efficientvit_seg_l2.yaml"
                ),
                "label_grouping_file": str(
                    semantic_share
                    / "config"
                    / "label_groupings"
                    / "ade20k_full.yaml"
                ),
                "model_file": str(
                    model_directory
                    / "ade20k-efficientvit_seg_l2.onnx"
                ),
                "engine_file": str(
                    model_directory
                    / "ade20k-efficientvit_seg_l2.trt"
                ),
                "force_rebuild": False,
                "colormap_path": str(
                    semantic_share
                    / "config"
                    / "distinct_150_colors.csv"
                ),
                "open_vocab": False,
                "max_queue_size": 1,
                "image_separation_s": 0.5,
            },
        ],
    )

    return LaunchDescription(
        [
            ComposableNodeContainer(
                name="semantic_inference_container",
                namespace="",
                package="rclcpp_components",
                executable="component_container_mt",
                output="screen",
                composable_node_descriptions=[segmentation_node],
                parameters=[{"image_transport": "raw"}],
            )
        ]
    )
