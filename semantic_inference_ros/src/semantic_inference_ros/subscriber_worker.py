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
"""ROS 2 queue-based generic subscriber worker."""

from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Optional, Type

from rclpy.node import Node
from rclpy.qos import QoSProfile, qos_profile_sensor_data

from semantic_inference_python import Config


SubscriberCallback = Callable[[Any], None]


@dataclass
class SubscriberWorkerConfig(Config):
    """Configuration for the subscriber worker."""

    queue_size: int = 1
    min_separation_s: float = 0.0

    @classmethod
    def load(cls, filepath):
        """Load configuration from a file."""
        return Config.load(cls, filepath)


class SubscriberWorker:
    """Process subscribed ROS 2 messages in a background thread."""

    def __init__(
        self,
        node: Node,
        config: SubscriberWorkerConfig,
        topic: str,
        message_type: Type,
        callback: SubscriberCallback,
        *,
        qos_profile: QoSProfile = qos_profile_sensor_data,
    ) -> None:
        """
        Register the subscriber worker.

        Args:
            node: ROS 2 node that owns the subscription.
            config: Worker configuration.
            topic: Topic to subscribe to.
            message_type: ROS message class for the topic.
            callback: Function called as ``callback(message)``.
            qos_profile: QoS profile used by the subscription.
        """
        if not isinstance(node, Node):
            raise TypeError(
                f"node must be an rclpy.node.Node, got {type(node).__name__}."
            )

        self._node = node
        self._config = config
        self._callback = callback

        self._started = False
        self._should_shutdown = False
        self._last_stamp_ns: Optional[int] = None
        self._thread: Optional[threading.Thread] = None

        self._queue = queue.Queue(
            maxsize=config.queue_size
        )

        self._sub = self._node.create_subscription(
            message_type,
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
        """Convert a ROS 2 timestamp message to nanoseconds."""
        return (
            int(stamp.sec) * 1_000_000_000
            + int(stamp.nanosec)
        )

    def add_message(self, msg: Any) -> None:
        """
        Add a message to the worker queue.

        This preserves the original behavior: when the queue is full, the
        incoming message is dropped.
        """
        if not self._queue.full():
            self._queue.put(
                msg,
                block=False,
            )

    def start(self) -> None:
        """Start the background processing thread."""
        if not self._started:
            self._started = True
            self._should_shutdown = False

            self._thread = threading.Thread(
                target=self._do_work,
                name="semantic-inference-subscriber-worker",
            )
            self._thread.start()

    def stop(self) -> None:
        """Stop the background processing thread."""
        if self._started:
            self._should_shutdown = True

            if (
                self._thread is not None
                and self._thread.is_alive()
                and threading.current_thread() is not self._thread
            ):
                self._thread.join()

        self._started = False
        self._should_shutdown = False

    def spin(self) -> None:
        """Wait until ROS shuts down or the worker exits."""
        if not self._started:
            return

        while (
            self._thread is not None
            and self._thread.is_alive()
            and not self._should_shutdown
            and self._node.context.ok()
        ):
            time.sleep(1.0e-2)

        self.stop()

    def _do_work(self) -> None:
        """Process messages from the queue."""
        while (
            not self._should_shutdown
            and self._node.context.ok()
        ):
            try:
                msg = self._queue.get(
                    timeout=0.1
                )
            except queue.Empty:
                continue

            if self._last_stamp_ns is not None:
                current_stamp_ns = self._stamp_to_nanoseconds(
                    msg.header.stamp
                )

                difference_s = (
                    current_stamp_ns
                    - self._last_stamp_ns
                ) / 1.0e9

                if (
                    difference_s
                    < self._config.min_separation_s
                ):
                    continue

            self._last_stamp_ns = self._stamp_to_nanoseconds(
                msg.header.stamp
            )

            self._callback(msg)