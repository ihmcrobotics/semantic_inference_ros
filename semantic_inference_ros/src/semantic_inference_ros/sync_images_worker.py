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
"""ROS 2 synchronized, queue-based image processing worker."""

from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Optional, Sequence

import message_filters
from rclpy.node import Node
from rclpy.qos import QoSProfile, qos_profile_sensor_data

from semantic_inference_python import Config
from semantic_inference_ros.ros_conversions import Conversions


SynchronizedImageCallback = Callable[..., None]


@dataclass
class SyncImagesWorkerConfig(Config):
    """Configuration for the synchronized image worker."""

    queue_size: int = 1
    min_separation_s: float = 0.0

    @classmethod
    def load(cls, filepath):
        """Load configuration from a file."""
        return Config.load(cls, filepath)

    def validate(self) -> None:
        """Validate worker configuration."""
        if self.queue_size <= 0:
            raise ValueError(
                f"queue_size must be greater than zero, got {self.queue_size}."
            )

        if self.min_separation_s < 0.0:
            raise ValueError(
                "min_separation_s must be non-negative, got "
                f"{self.min_separation_s}."
            )


class SyncImagesWorker:
    """Synchronize ROS 2 image messages and process them in a worker thread."""

    def __init__(
        self,
        node: Node,
        config: SyncImagesWorkerConfig,
        topics: Sequence[str],
        message_types: Sequence[type],
        callback: SynchronizedImageCallback,
        *,
        qos_profile: QoSProfile = qos_profile_sensor_data,
        approximate_sync: bool = False,
        sync_slop_s: float = 0.05,
    ) -> None:
        """
        Initialize synchronized subscriptions.

        Args:
            node: ROS 2 node that owns the subscriptions.
            config: Worker configuration.
            topics: Input image topic names.
            message_types: ROS message type corresponding to each topic.
            callback: Function called as
                ``callback(header, image_1, image_2, ...)``.
            qos_profile: QoS used for all input subscriptions.
            approximate_sync: Use approximate instead of exact synchronization.
            sync_slop_s: Maximum timestamp difference for approximate sync.
        """
        if not isinstance(node, Node):
            raise TypeError(
                f"node must be an rclpy.node.Node, got {type(node).__name__}."
            )

        if len(topics) == 0:
            raise ValueError("At least one input topic is required.")

        if len(topics) != len(message_types):
            raise ValueError(
                "topics and message_types must have equal lengths: "
                f"{len(topics)} topics and "
                f"{len(message_types)} message types were provided."
            )

        if sync_slop_s < 0.0:
            raise ValueError(
                f"sync_slop_s must be non-negative, got {sync_slop_s}."
            )

        config.validate()

        self._node = node
        self._config = config
        self._callback = callback

        self._started = False
        self._shutdown_event = threading.Event()
        self._last_stamp_ns: Optional[int] = None
        self._thread: Optional[threading.Thread] = None

        self._queue: queue.Queue[list[Any]] = queue.Queue(
            maxsize=config.queue_size
        )

        # Keep these subscribers as members so they are not garbage-collected.
        self._subscribers = [
            message_filters.Subscriber(
                node,
                message_type,
                topic,
                qos_profile=qos_profile,
            )
            for topic, message_type in zip(
                topics,
                message_types,
            )
        ]

        if approximate_sync:
            self._synchronizer = (
                message_filters.ApproximateTimeSynchronizer(
                    self._subscribers,
                    queue_size=config.queue_size,
                    slop=sync_slop_s,
                )
            )
        else:
            self._synchronizer = message_filters.TimeSynchronizer(
                self._subscribers,
                queue_size=config.queue_size,
            )

        self._synchronizer.registerCallback(
            self.add_message
        )

        # Stop the worker when the rclpy context shuts down.
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

    def add_message(self, *messages: Any) -> None:
        """
        Queue one synchronized set of messages.

        If the worker queue is full, discard the oldest synchronized set and
        retain the newest one. This prevents inference latency from growing.
        """
        if self._shutdown_event.is_set():
            return

        synchronized_messages = list(messages)

        try:
            self._queue.put_nowait(
                synchronized_messages
            )
            return
        except queue.Full:
            pass

        try:
            self._queue.get_nowait()
            self._queue.task_done()
        except queue.Empty:
            pass

        try:
            self._queue.put_nowait(
                synchronized_messages
            )
        except queue.Full:
            self._node.get_logger().warning(
                "Synchronized image queue remained full; "
                "dropping the incoming synchronized messages."
            )

    def start(self) -> None:
        """Start the processing thread."""
        if self._started:
            return

        self._shutdown_event.clear()

        self._thread = threading.Thread(
            target=self._do_work,
            name="semantic-inference-sync-images-worker",
            daemon=True,
        )
        self._thread.start()

        self._started = True

    def stop(self) -> None:
        """Stop the processing thread."""
        if not self._started:
            return

        self._shutdown_event.set()

        if (
            self._thread is not None
            and self._thread.is_alive()
            and threading.current_thread() is not self._thread
        ):
            self._thread.join(timeout=2.0)

            if self._thread.is_alive():
                self._node.get_logger().warning(
                    "Synchronized image worker did not stop "
                    "within two seconds."
                )

        self._thread = None
        self._started = False

    def spin(self) -> None:
        """
        Wait for the worker thread to exit.

        Normally, the owning node should be spun using ``rclpy.spin(node)``.
        This method is retained for compatibility with the original API.
        """
        if not self._started:
            return

        while (
            self._thread is not None
            and self._thread.is_alive()
            and not self._shutdown_event.is_set()
            and self._node.context.ok()
        ):
            time.sleep(1.0e-2)

    def _do_work(self) -> None:
        """Process synchronized message sets."""
        while (
            not self._shutdown_event.is_set()
            and self._node.context.ok()
        ):
            try:
                messages = self._queue.get(
                    timeout=0.1
                )
            except queue.Empty:
                continue

            try:
                if not messages:
                    continue

                first_message = messages[0]

                if not hasattr(first_message, "header"):
                    raise AttributeError(
                        "The first synchronized message has no header."
                    )

                current_stamp_ns = self._stamp_to_nanoseconds(
                    first_message.header.stamp
                )

                if self._last_stamp_ns is not None:
                    difference_s = (
                        current_stamp_ns
                        - self._last_stamp_ns
                    ) / 1.0e9

                    if (
                        difference_s
                        < self._config.min_separation_s
                    ):
                        continue

                self._last_stamp_ns = current_stamp_ns

                images = [
                    Conversions.to_image(message)
                    for message in messages
                ]

                self._callback(
                    first_message.header,
                    *images,
                )

            except Exception as exception:
                self._node.get_logger().error(
                    "Synchronized image processing failed: "
                    f"{exception}"
                )
            finally:
                self._queue.task_done()