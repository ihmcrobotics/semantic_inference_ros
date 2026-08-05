#!/usr/bin/env python3

# BSD 3-Clause License
#
# Copyright (c) 2021-2024, Massachusetts Institute of Technology.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2026, IHMC Robotics Lab.
# All rights reserved.
#
"""Generate text using an instruction-following VLM and visual encodings."""

from __future__ import annotations

import json
from enum import Enum
from typing import Any, Dict, List

import numpy as np
import torch
from rclpy.node import Node
from rclpy.qos import QoSProfile
from scipy.spatial.transform import Rotation

from hydra_msgs.msg import (
    LabeledRelationships,
    VisualRelationshipsEncodings,
)


class VLMGeometry(Enum):
    """Geometry representation included in generated prompts."""

    NONE = "none"
    BOUNDING_BOX = "bounding_box"
    CENTER = "center"
    CORNERS = "corners"


class VLMTextGenerator:
    """Generate relationship labels from visual feature encodings."""

    def __init__(
        self,
        node: Node,
        model: Any,
        geometry: VLMGeometry = VLMGeometry.NONE,
        input_topic: str = "visual_relationships_encodings",
        output_topic: str = "labeled_relationships",
        qos_depth: int = 1,
    ) -> None:
        """
        Initialize the ROS 2 publisher and subscriber.

        Args:
            node: ROS 2 node that owns the publisher and subscription.
            model: Model exposing ``generate_caption(features, prompts)`` and
                a ``device`` property.
            geometry: Geometry representation to include in prompts.
            input_topic: Topic containing visual relationship encodings.
            output_topic: Topic used to publish labeled relationships.
            qos_depth: Queue depth for publisher and subscriber.
        """
        if not isinstance(node, Node):
            raise TypeError(
                f"node must be an rclpy.node.Node, got {type(node).__name__}."
            )

        if not hasattr(model, "generate_caption"):
            raise TypeError(
                "model must provide a generate_caption(...) method."
            )

        self._node = node
        self._model = model
        self._geometry = geometry

        qos_profile = QoSProfile(depth=qos_depth)

        self._pub = self._node.create_publisher(
            LabeledRelationships,
            output_topic,
            qos_profile,
        )

        self._sub = self._node.create_subscription(
            VisualRelationshipsEncodings,
            input_topic,
            self._callback,
            qos_profile,
        )

    def _callback(
        self,
        msg: VisualRelationshipsEncodings,
    ) -> None:
        """Generate and publish text labels for visual relationships."""
        self._node.get_logger().info(
            "Received visual relationship encodings."
        )

        labeled_relationships = LabeledRelationships()
        visual_embeddings = []
        prompts = []

        feature_count = len(msg.features.feature)

        if len(msg.object_classes) < 2 * feature_count:
            self._node.get_logger().error(
                "Received fewer object classes than required for "
                f"{feature_count} relationship features."
            )
            return

        if len(msg.features.ids) < 2 * feature_count:
            self._node.get_logger().error(
                "Received fewer object IDs than required for "
                f"{feature_count} relationship features."
            )
            return

        for index, feature in enumerate(msg.features.feature):
            visual_embeddings.append(
                torch.tensor(
                    feature.data,
                    dtype=torch.float32,
                )
                .reshape(feature.rows, feature.cols)
                .to(self._model.device)
            )

            first_index = 2 * index
            second_index = first_index + 1
            first_class = msg.object_classes[first_index]
            second_class = msg.object_classes[second_index]

            prompt = (
                f"{msg.prompt} the {first_class} "
                f"and the {second_class}?"
            )

            if self._geometry == VLMGeometry.CORNERS:
                if len(msg.object_bounding_boxes.boxes) <= second_index:
                    self._node.get_logger().error(
                        "Bounding-box data is missing for a relationship."
                    )
                    return

                first_box = msg.object_bounding_boxes.boxes[first_index]
                second_box = msg.object_bounding_boxes.boxes[second_index]

                geometry_json = {
                    first_class: self._to_corners(
                        self._box_translation(first_box),
                        self._box_quaternion(first_box),
                        self._box_size(first_box),
                    ),
                    second_class: self._to_corners(
                        self._box_translation(second_box),
                        self._box_quaternion(second_box),
                        self._box_size(second_box),
                    ),
                }

                prompt += (
                    " The geometries of the objects are "
                    f"{json.dumps(geometry_json)}."
                )

            elif self._geometry == VLMGeometry.BOUNDING_BOX:
                if (
                    len(msg.object_bounding_boxes.boxes) <= second_index
                    or len(msg.object_poses) <= second_index
                ):
                    self._node.get_logger().error(
                        "Pose or bounding-box data is missing for a relationship."
                    )
                    return

                geometry_json = {
                    first_class: self._bounding_box_geometry(
                        msg.object_poses[first_index],
                        msg.object_bounding_boxes.boxes[first_index],
                    ),
                    second_class: self._bounding_box_geometry(
                        msg.object_poses[second_index],
                        msg.object_bounding_boxes.boxes[second_index],
                    ),
                }

                prompt += (
                    " The geometries of the objects are "
                    f"{json.dumps(geometry_json)}."
                )

            elif self._geometry == VLMGeometry.CENTER:
                if len(msg.object_poses) <= second_index:
                    self._node.get_logger().error(
                        "Object-pose data is missing for a relationship."
                    )
                    return

                first_center = self._pose_position(
                    msg.object_poses[first_index]
                )
                second_center = self._pose_position(
                    msg.object_poses[second_index]
                )

                prompt = (
                    f"{msg.prompt} the {first_class} "
                    f"(spatially located at {first_center} meters) "
                    f"and the {second_class} "
                    f"(spatially located at {second_center} meters)?"
                )

            prompts.append(prompt)
            labeled_relationships.node_ids.append(
                msg.features.ids[first_index]
            )
            labeled_relationships.node_ids.append(
                msg.features.ids[second_index]
            )

        if not visual_embeddings:
            self._node.get_logger().warning(
                "No relationship features were available to process."
            )
            return

        try:
            labeled_relationships.labels = (
                self._model.generate_caption(
                    torch.stack(visual_embeddings),
                    prompts,
                )
            )
        except Exception as exception:
            self._node.get_logger().error(
                "Failed to generate relationship labels: "
                f"{exception}"
            )
            return

        labeled_relationships.header.stamp = (
            self._node.get_clock().now().to_msg()
        )

        self._pub.publish(labeled_relationships)

        self._node.get_logger().info(
            "Published labeled relationships."
        )

    @staticmethod
    def _pose_position(pose) -> List[str]:
        """Convert a pose position to a formatted list."""
        return [
            f"{pose.position.x:.2f}",
            f"{pose.position.y:.2f}",
            f"{pose.position.z:.2f}",
        ]

    @staticmethod
    def _box_translation(box) -> np.ndarray:
        """Extract a box-center translation."""
        return np.array(
            [
                box.center.position.x,
                box.center.position.y,
                box.center.position.z,
            ],
            dtype=np.float64,
        )

    @staticmethod
    def _box_quaternion(box) -> np.ndarray:
        """Extract a box-center quaternion in x, y, z, w order."""
        return np.array(
            [
                box.center.orientation.x,
                box.center.orientation.y,
                box.center.orientation.z,
                box.center.orientation.w,
            ],
            dtype=np.float64,
        )

    @staticmethod
    def _box_size(box) -> np.ndarray:
        """Extract box dimensions."""
        return np.array(
            [
                box.size.x,
                box.size.y,
                box.size.z,
            ],
            dtype=np.float64,
        )

    @classmethod
    def _bounding_box_geometry(
        cls,
        pose,
        box,
    ) -> Dict[str, Any]:
        """Convert a pose and bounding box to a JSON-compatible dictionary."""
        return {
            "object_center": cls._pose_position(pose),
            "bounding_box": {
                "center": [
                    f"{box.center.position.x:.2f}",
                    f"{box.center.position.y:.2f}",
                    f"{box.center.position.z:.2f}",
                ],
                "quaternion": [
                    f"{box.center.orientation.x:.2f}",
                    f"{box.center.orientation.y:.2f}",
                    f"{box.center.orientation.z:.2f}",
                    f"{box.center.orientation.w:.2f}",
                ],
                "axis_size": [
                    f"{box.size.x:.2f}",
                    f"{box.size.y:.2f}",
                    f"{box.size.z:.2f}",
                ],
            },
        }

    @staticmethod
    def _to_corners(
        translation: np.ndarray,
        quaternion: np.ndarray,
        size: np.ndarray,
    ) -> Dict[str, List[float]]:
        """
        Convert a bounding box to eight world-frame corners.

        Args:
            translation: Box-center translation.
            quaternion: Box-center quaternion in x, y, z, w order.
            size: Box dimensions.

        Returns:
            Mapping from corner names to three-dimensional coordinates.
        """
        if translation.shape != (3,):
            raise ValueError(
                f"translation must have shape (3,), got {translation.shape}."
            )

        if quaternion.shape != (4,):
            raise ValueError(
                f"quaternion must have shape (4,), got {quaternion.shape}."
            )

        if size.shape != (3,):
            raise ValueError(
                f"size must have shape (3,), got {size.shape}."
            )

        quaternion_norm = np.linalg.norm(quaternion)
        if quaternion_norm == 0.0:
            raise ValueError("Bounding-box quaternion has zero magnitude.")

        quaternion = quaternion / quaternion_norm

        half_size = size / 2.0

        corners = (
            np.array(
                [
                    [-1, -1, -1],
                    [-1, -1, 1],
                    [-1, 1, -1],
                    [-1, 1, 1],
                    [1, -1, -1],
                    [1, -1, 1],
                    [1, 1, -1],
                    [1, 1, 1],
                ],
                dtype=np.float64,
            )
            * half_size
        )

        rotation_matrix = Rotation.from_quat(
            quaternion
        ).as_matrix()

        transformed_corners = (
            rotation_matrix @ corners.T
        ).T + translation

        return {
            f"corner_{index}": [
                round(float(coordinate), 2)
                for coordinate in transformed_corners[index]
            ]
            for index in range(8)
        }