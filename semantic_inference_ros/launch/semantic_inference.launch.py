#!/usr/bin/env python3

import os
from pathlib import Path
import sys

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition, UnlessCondition
from launch.substitutions import (
    LaunchConfiguration,
    PathJoinSubstitution,
    PythonExpression,
)
from launch_ros.actions import (
    ComposableNodeContainer,
    LoadComposableNodes,
    Node,
)
from launch_ros.descriptions import ComposableNode


def generate_launch_description() -> LaunchDescription:
    semantic_inference_ros_share = Path(
        get_package_share_directory("semantic_inference_ros")
    )
    semantic_inference_python_share = Path(
        get_package_share_directory("semantic_inference_python")
    )
    semantic_inference_share = Path(
        get_package_share_directory("semantic_inference")
    )
    hydra_share = Path(get_package_share_directory("hydra"))

    playback_mode = LaunchConfiguration("playback_mode")
    print_inference_time = LaunchConfiguration("print_inference_time")
    start_manager = LaunchConfiguration("start_manager")
    nodelet_manager = LaunchConfiguration("nodelet_manager")
    debug = LaunchConfiguration("debug")
    launch_prefix = LaunchConfiguration("launch_prefix")
    verbose = LaunchConfiguration("verbose")
    use_prerecorded_semantics = LaunchConfiguration(
        "use_prerecorded_semantics"
    )
    force_rebuild = LaunchConfiguration("force_rebuild")

    rgb_image_transport = LaunchConfiguration("rgb_image_transport")
    max_image_queue_size = LaunchConfiguration(
        "max_image_queue_size"
    )
    min_separation_s = LaunchConfiguration("min_separation_s")
    rotation_type = LaunchConfiguration("rotation_type")
    labelspace_dir = LaunchConfiguration("labelspace_dir")
    labelspace_name = LaunchConfiguration("labelspace_name")
    camera_info_topic = LaunchConfiguration("camera_info_topic")

    active_object_edges_topic = LaunchConfiguration(
        "active_object_edges_topic"
    )
    odom_frame = LaunchConfiguration("odom_frame")
    sensor_frame = LaunchConfiguration("sensor_frame")
    colormap_path = LaunchConfiguration("colormap_path")

    open_vocab = LaunchConfiguration("open_vocab")
    segmentation_cuda = LaunchConfiguration("segmentation_cuda")

    use_vlm = LaunchConfiguration("use_vlm")
    labels_from_prompt = LaunchConfiguration("labels_from_prompt")
    visual_relationships_encodings_topic = LaunchConfiguration(
        "visual_relationships_encodings_topic"
    )
    config_path_vlm = LaunchConfiguration("config_path_vlm")
    vlm_labels_from_prompt_config_path = LaunchConfiguration(
        "vlm_labels_from_prompt_config_path"
    )
    pub_vlm_annotations = LaunchConfiguration("pub_vlm_annotations")
    height = LaunchConfiguration("height")
    width = LaunchConfiguration("width")
    vlm_queue_size = LaunchConfiguration("vlm_queue_size")
    vlm_cuda = LaunchConfiguration("vlm_cuda")

    config_path = LaunchConfiguration("config_path")
    yolo_model_name = LaunchConfiguration("yolo_model_name")
    python_env = LaunchConfiguration("python_env")
    segmentation_model = LaunchConfiguration("segmentation_model")

    use_segmentation_components = PythonExpression(
        [
            "'",
            segmentation_model,
            "' != 'yolosam' and '",
            segmentation_model,
            "' != 'yoloe'",
        ]
    )

    use_open_set_node = PythonExpression(
        [
            "not (",
            use_prerecorded_semantics,
            ") and not (",
            use_segmentation_components,
            ")",
        ]
    )

    use_closed_set_component = PythonExpression(
        [
            "not (",
            use_prerecorded_semantics,
            ") and (",
            use_segmentation_components,
            ")",
        ]
    )

    use_recolor_component = PythonExpression(
        [use_prerecorded_semantics]
    )

    use_clip_node = PythonExpression(
        [
            "(",
            open_vocab,
            ") and (",
            use_segmentation_components,
            ") and not (",
            use_prerecorded_semantics,
            ")",
        ]
    )

    use_dummy_vlm = PythonExpression(
        ["not (", use_vlm, ")"]
    )

    labelspace_filename = PythonExpression(
        [
            "'",
            labelspace_name,
            "_label_space.yaml'",
        ]
    )

    labelspace_file = PathJoinSubstitution(
        [
            labelspace_dir,
            labelspace_filename,
        ]
    )

    label_grouping_filename = PythonExpression(
        [
            "'",
            labelspace_name,
            ".yaml'",
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

    segmentation_config_filename = PythonExpression(
        [
            "'",
            segmentation_model,
            ".yaml'",
        ]
    )

    segmentation_config_file = PathJoinSubstitution(
        [
            str(semantic_inference_share),
            "config",
            "models",
            segmentation_config_filename,
        ]
    )

    model_filename = PythonExpression(
        [
            "'",
            segmentation_model,
            ".onnx'",
        ]
    )

    model_file = PathJoinSubstitution(
        [
            str(semantic_inference_share),
            "models",
            model_filename,
        ]
    )

    engine_filename = PythonExpression(
        [
            "'",
            segmentation_model,
            ".trt'",
        ]
    )

    engine_file = PathJoinSubstitution(
        [
            str(semantic_inference_share),
            "engines",
            engine_filename,
        ]
    )

    declarations = [
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
            "start_manager",
            default_value="true",
            description="Start a separate component container",
        ),
        DeclareLaunchArgument(
            "nodelet_manager",
            default_value="semantic_inference_container",
            description="Name of the ROS 2 component container",
        ),
        DeclareLaunchArgument(
            "debug",
            default_value="false",
            description="Run the component container with GDB",
        ),
        DeclareLaunchArgument(
            "launch_prefix",
            default_value="",
            description="Launch prefix for the component container",
        ),
        DeclareLaunchArgument(
            "verbose",
            default_value="true",
            description="Show node output on stdout",
        ),
        DeclareLaunchArgument(
            "use_prerecorded_semantics",
            default_value="false",
            description="Use prerecorded labels instead of inference",
        ),
        DeclareLaunchArgument(
            "force_rebuild",
            default_value="false",
            description="Force TensorRT to rebuild its engine",
        ),
        DeclareLaunchArgument(
            "rgb_image_transport",
            default_value="raw",
            description="Input image transport type",
        ),
        DeclareLaunchArgument(
            "max_image_queue_size",
            default_value="1",
            description="Maximum number of images to store",
        ),
        DeclareLaunchArgument(
            "min_separation_s",
            default_value="0.7",
            description="Enforced separation between input images",
        ),
        DeclareLaunchArgument(
            "rotation_type",
            default_value="none",
            description="Input camera rotation",
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
            description="Name of the label space",
        ),
        DeclareLaunchArgument(
            "camera_info_topic",
            default_value="/camera/color/camera_info",
        ),
        DeclareLaunchArgument(
            "active_object_edges_topic",
            default_value=(
                "/hydra_ros_node/backend/active_object_edges"
            ),
        ),
        DeclareLaunchArgument(
            "odom_frame",
            default_value="world",
        ),
        DeclareLaunchArgument(
            "sensor_frame",
            default_value="camera_color_optical_frame",
        ),
        DeclareLaunchArgument(
            "colormap_path",
            default_value=str(
                semantic_inference_share
                / "config"
                / "distinct_150_colors.csv"
            ),
        ),
        DeclareLaunchArgument(
            "open_vocab",
            default_value="true",
        ),
        DeclareLaunchArgument(
            "segmentation_cuda",
            default_value="true",
        ),
        DeclareLaunchArgument(
            "use_vlm",
            default_value="true",
        ),
        DeclareLaunchArgument(
            "labels_from_prompt",
            default_value="false",
        ),
        DeclareLaunchArgument(
            "visual_relationships_encodings_topic",
            default_value=(
                "/hydra_ros_node/backend/vlm_relationships/"
                "visual_relationships_encodings"
            ),
        ),
        DeclareLaunchArgument(
            "config_path_vlm",
            default_value=str(
                semantic_inference_ros_share / "config" / "vlm.yaml"
            ),
        ),
        DeclareLaunchArgument(
            "vlm_labels_from_prompt_config_path",
            default_value=str(
                semantic_inference_ros_share
                / "config"
                / "vlm_prompting.yaml"
            ),
        ),
        DeclareLaunchArgument(
            "pub_vlm_annotations",
            default_value="true",
        ),
        DeclareLaunchArgument(
            "height",
            default_value="720",
        ),
        DeclareLaunchArgument(
            "width",
            default_value="1280",
        ),
        DeclareLaunchArgument(
            "vlm_queue_size",
            default_value="100",
        ),
        DeclareLaunchArgument(
            "vlm_cuda",
            default_value="true",
        ),
        DeclareLaunchArgument(
            "config_path",
            default_value=str(
                semantic_inference_ros_share
                / "config"
                / "openset_segmentation.yaml"
            ),
        ),
        DeclareLaunchArgument(
            "semantic_labels_path",
            default_value=label_grouping_file,
        ),
        DeclareLaunchArgument(
            "yolo_model_name",
            default_value="yoloe-11l-seg.pt",
        ),
        DeclareLaunchArgument(
            "python_env",
            default_value=str(
                Path(os.environ["VIRTUAL_ENV"]) / "bin" / "python"
                if "VIRTUAL_ENV" in os.environ
                else Path(sys.executable)
            ),
        ),
        DeclareLaunchArgument(
            "segmentation_model",
            default_value="yoloe",
        ),
    ]

    component_container = ComposableNodeContainer(
        condition=IfCondition(start_manager),
        name=nodelet_manager,
        namespace="",
        package="rclcpp_components",
        executable="component_container_mt",
        output="screen",
        prefix=launch_prefix,
        composable_node_descriptions=[],
        parameters=[
            {
                "image_transport": rgb_image_transport,
            }
        ],
    )

    closed_set_component = LoadComposableNodes(
        condition=IfCondition(use_closed_set_component),
        target_container=nodelet_manager,
        composable_node_descriptions=[
            ComposableNode(
                package="semantic_inference_ros",
                plugin="semantic_inference::SegmentationNode",
                name="semantic_inference",
                namespace="semantic_inference",
                parameters=[
                    segmentation_config_file,
                    label_grouping_file,
                    {
                        "segmenter.model.model_file": model_file,
                        "segmenter.model.engine_file": engine_file,
                        "segmenter.model.force_rebuild": force_rebuild,
                        "output.recolor.colormap_path": colormap_path,
                        "output.open_vocab": open_vocab,
                        "worker.max_queue_size":
                            max_image_queue_size,
                        "worker.image_separation_s":
                            min_separation_s,
                        "image_rotator.rotation": rotation_type,
                    },
                ],
            ),
        ],
    )

    recolor_component = LoadComposableNodes(
        condition=IfCondition(use_recolor_component),
        target_container=nodelet_manager,
        composable_node_descriptions=[
            ComposableNode(
                package="semantic_inference_ros",
                plugin="semantic_inference::RecolorNode",
                name="semantic_inference",
                namespace="semantic_inference",
                parameters=[
                    label_grouping_file,
                    {
                        "worker.max_queue_size":
                            max_image_queue_size,
                        "worker.image_separation_s":
                            min_separation_s,
                    },
                ],
            ),
        ],
    )

    open_set_node = Node(
        condition=IfCondition(use_open_set_node),
        package="semantic_inference_ros",
        executable="openset_segmentation_node",
        name="semantic_inference",
        namespace="semantic_inference",
        output="screen",
        prefix=python_env,
        parameters=[
            {
                "config_path":
                    config_path,
                "labelspace_path":
                    labelspace_file,
                "label_grouping_path":
                    label_grouping_file,
                "model.segmentation.yolo_model_name":
                    yolo_model_name,
                "model.cuda":
                    segmentation_cuda,
                "worker.min_separation_s":
                    min_separation_s,
                "model.segmentation.type":
                    segmentation_model,
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
    )

    clip_features_node = Node(
        condition=IfCondition(use_clip_node),
        package="semantic_inference_ros",
        executable="clip_features_node",
        name="clip_features_node",
        namespace="semantic_inference",
        output="screen",
        prefix=python_env,
        parameters=[
            {
                "config_path":
                    config_path,
                "labelspace_path":
                    labelspace_file,
                "worker.min_separation_s":
                    min_separation_s,
                "recolor.colormap_path":
                    colormap_path,
                "playback_mode":
                    playback_mode,
            },
        ],
        remappings=[
            (
                "/semantic_inference/color/camera_info",
                camera_info_topic,
            ),
        ],
    )

    vlm_features_node = Node(
        condition=IfCondition(use_vlm),
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
                "model.cuda":
                    vlm_cuda,
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
    )

    encode_vlm_labels_node = Node(
        condition=IfCondition(labels_from_prompt),
        package="semantic_inference_ros",
        executable="encode_vlm_labels_node",
        name="encode_vlm_labels_node",
        namespace="semantic_inference",
        output="screen",
        prefix=python_env,
        parameters=[
            {
                "config_path":
                    vlm_labels_from_prompt_config_path,
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
    )

    dummy_vlm_node = Node(
        condition=IfCondition(use_dummy_vlm),
        package="semantic_inference_ros",
        executable="dummy_vlm_node",
        name="dummy_vlm_node",
        namespace="semantic_inference",
        output="screen",
        prefix=python_env,
        parameters=[
            {
                "worker.min_separation_s": min_separation_s,
                "playback_mode": playback_mode,
            },
        ],
    )

    return LaunchDescription(
        declarations
        + [
            component_container,
            closed_set_component,
            recolor_component,
            open_set_node,
            clip_features_node,
            vlm_features_node,
            encode_vlm_labels_node,
            dummy_vlm_node,
        ]
    )
