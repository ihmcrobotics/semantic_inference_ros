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
"""ROS 2 prompt-embedding service utility."""

from __future__ import annotations

from typing import Any

import numpy as np
from rclpy.node import Node

from semantic_inference_msgs.msg import LevelFeatureVectorStamped
from semantic_inference_msgs.srv import EncodeFeature

from semantic_inference_ros.ros_conversions import Conversions


class PromptEncoder:
    """Provide prompt embeddings through a ROS 2 service and publisher."""

    def __init__(
        self,
        node: Node,
        model: Any,
        service_name: str = "embed",
        publisher_topic: str = "clip_embeddings",
    ) -> None:
        """
        Initialize the prompt encoder.

        Args:
            node: ROS 2 node that owns the service and publisher.
            model: Model exposing ``embed_text(...)``.
            service_name: Relative ROS 2 service name.
            publisher_topic: Relative topic used to publish generated embeddings.
        """
        if not isinstance(node, Node):
            raise TypeError(
                f"node must be an rclpy.node.Node, got {type(node).__name__}."
            )

        if not hasattr(model, "embed_text"):
            raise TypeError(
                "model must provide an embed_text(...) method."
            )

        self._node = node
        self._model = model

        self._srv = self._node.create_service(
            EncodeFeature,
            service_name,
            self._callback,
        )

        self._pub = self._node.create_publisher(
            LevelFeatureVectorStamped,
            publisher_topic,
            1,
        )

        self._node.get_logger().info(
            f"Prompt embedding service created at "
            f"'{self._node.resolve_service_name(service_name)}'."
        )

    def _callback(
        self,
        request: EncodeFeature.Request,
        response: EncodeFeature.Response,
    ) -> EncodeFeature.Response:
        """
        Encode a prompt and populate the service response.

        Args:
            request: Service request containing ``prompt`` and ``level``.
            response: Service response to populate.

        Returns:
            Populated ROS 2 service response.
        """
        try:
            embedding = self._model.embed_text(
                request.prompt
            )

            if hasattr(embedding, "detach"):
                embedding = embedding.detach()

            if hasattr(embedding, "cpu"):
                embedding = embedding.cpu()

            if hasattr(embedding, "numpy"):
                embedding = embedding.numpy()

            embedding = np.asarray(
                embedding,
                dtype=np.float32,
            ).squeeze()

            stamp = self._node.get_clock().now().to_msg()

            response.feature.level = request.level
            response.feature.feature.header.stamp = stamp
            response.feature.feature.feature = (
                Conversions.to_feature(embedding)
            )

            published_message = LevelFeatureVectorStamped()
            published_message.level = response.feature.level
            published_message.feature = response.feature.feature

            self._pub.publish(
                published_message
            )

            return response

        except Exception as exception:
            self._node.get_logger().error(
                "Failed to encode prompt "
                f"'{request.prompt}': {exception}"
            )
            raise