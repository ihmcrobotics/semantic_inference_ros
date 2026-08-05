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
"""ROS 2 queue-based image processing worker."""

from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Optional

from rclpy.node import Node
from rclpy.qos import QoSProfile, qos_profile_sensor_data
from sensor_msgs.msg import Image
from std_msgs.msg import Header

from semantic_inference_python import Config
from semantic_inference_ros.ros_conversions import Conversions


ImageCallback = Callable[[Header, Any], None]


@dataclass
class ImageWorkerConfig(Config):
    """Configuration for the image worker."""

    queue_size: int = 1
    min_separation_s: float = 0.0

    @classmethod
    def load(cls, filepath):
        """Load configuration from a file."""
        return Config.load(cls, filepath)

    def validate(self) -> None:
        """Validate configuration values."""
        if self.queue_size <= 0:
            raise ValueError(
                f"queue_size must be greater than zero, got {self.queue_size}."
            )

        if self.min_separation_s < 0.0:
            raise ValueError(
                "min_separation_s must be non-negative, got "
                f"{self.min_separation_s}."
            )


class ImageWorker:
    """Subscribe to images and process them on a background thread."""

    def __init__(
        self,
        node: Node,
        config: ImageWorkerConfig,
        topic: str,
        callback: ImageCallback,
        *,
        qos_profile: QoSProfile = qos_profile_sensor_data,
    ) -> None:
        """
        Initialize the worker.

        Args:
            node: ROS 2 node that owns the subscription.
            config: Worker configuration.
            topic: Image topic to subscribe to.
            callback: Function called as ``callback(header, image_array)``.
            qos_profile: QoS profile used for the image subscription.
        """
        if not isinstance(node, Node):
            raise TypeError(
                f"node must be an rclpy.node.Node, got {type(node).__name__}."
            )

        config.validate()

        self._node = node
        self._config = config
        self._callback = callback

        self._started = False
        self._should_shutdown = threading.Event()
        self._last_stamp_ns: Optional[int] = None
        self._thread: Optional[threading.Thread] = None

        self._queue: queue.Queue[Image] = queue.Queue(
            maxsize=config.queue_size
        )

        self._sub = self._node.create_subscription(
            Image,
            topic,
            self.add_message,
            qos_profile,
        )

        self._node.context.on_shutdown(
            self.stop
        )

        self.start()

    @staticmethod
    def _stamp_to_nanoseconds(stamp) -> int:
        """Convert a builtin_interfaces/Time message to nanoseconds."""
        return (
            int(stamp.sec) * 1_000_000_000
            + int(stamp.nanosec)
        )

    def add_message(
        self,
        msg: Image,
    ) -> None:
        """
        Add a message to the processing queue.

        When the queue is full, the oldest queued image is removed so that
        processing continues with the most recent sensor data.
        """
        if self._should_shutdown.is_set():
            return

        try:
            self._queue.put_nowait(msg)
            return
        except queue.Full:
            pass

        try:
            self._queue.get_nowait()
            self._queue.task_done()
        except queue.Empty:
            pass

        try:
            self._queue.put_nowait(msg)
        except queue.Full:
            self._node.get_logger().warning(
                "Image worker queue remained full; dropping incoming image."
            )

    def start(self) -> None:
        """Start the background worker thread."""
        if self._started:
            return

        self._should_shutdown.clear()

        self._thread = threading.Thread(
            target=self._do_work,
            name="semantic-inference-image-worker",
            daemon=True,
        )
        self._thread.start()
        self._started = True

    def stop(self) -> None:
        """Stop the background worker thread."""
        if not self._started:
            return

        self._should_shutdown.set()

        if (
            self._thread is not None
            and self._thread.is_alive()
            and threading.current_thread() is not self._thread
        ):
            self._thread.join(timeout=2.0)

            if self._thread.is_alive():
                self._node.get_logger().warning(
                    "Image worker thread did not stop within two seconds."
                )

        self._thread = None
        self._started = False

    def spin(self) -> None:
        """
        Wait until the worker exits.

        The ROS 2 node itself should normally be spun with ``rclpy.spin(node)``.
        This method is retained for compatibility with older callers.
        """
        if not self._started:
            return

        while (
            self._thread is not None
            and self._thread.is_alive()
            and not self._should_shutdown.is_set()
            and self._node.context.ok()
        ):
            time.sleep(1.0e-2)

    def _do_work(self) -> None:
        """Process queued image messages."""
        while (
            not self._should_shutdown.is_set()
            and self._node.context.ok()
        ):
            try:
                msg = self._queue.get(
                    timeout=0.1
                )
            except queue.Empty:
                continue

            try:
                current_stamp_ns = self._stamp_to_nanoseconds(
                    msg.header.stamp
                )

                if self._last_stamp_ns is not None:
                    separation_s = (
                        current_stamp_ns
                        - self._last_stamp_ns
                    ) / 1.0e9

                    if (
                        separation_s
                        < self._config.min_separation_s
                    ):
                        continue

                self._last_stamp_ns = current_stamp_ns

                image = Conversions.to_image(
                    msg
                )

                self._callback(
                    msg.header,
                    image,
                )

            except Exception as exception:
                self._node.get_logger().error(
                    "Image worker failed while processing an image: "
                    f"{exception}"
                )
            finally:
                self._queue.task_done()