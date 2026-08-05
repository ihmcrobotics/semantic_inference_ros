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
"""Module containing ROS 2 logging shim."""

import logging
from typing import Optional

from rclpy.logging import get_logger
from rclpy.node import Node

from semantic_inference_python import Logger


class RosForwarder(logging.Handler):
    """Forward standard Python logging records to ROS 2 logging."""

    def __init__(
        self,
        node: Optional[Node] = None,
        logger_name: str = "semantic_inference",
    ) -> None:
        """
        Initialize the ROS 2 logging forwarder.

        Args:
            node: Optional ROS 2 node. When provided, its logger is used.
            logger_name: Logger name used when no node is provided.
        """
        super().__init__()

        self._ros_logger = (
            node.get_logger()
            if node is not None
            else get_logger(logger_name)
        )

    def emit(self, record: logging.LogRecord) -> None:
        """Forward one Python logging record to ROS 2."""
        try:
            message = (
                f"{record.name}: {record.getMessage()}"
            )

            if record.levelno >= logging.CRITICAL:
                self._ros_logger.fatal(message)
            elif record.levelno >= logging.ERROR:
                self._ros_logger.error(message)
            elif record.levelno >= logging.WARNING:
                self._ros_logger.warning(message)
            elif record.levelno >= logging.INFO:
                self._ros_logger.info(message)
            else:
                self._ros_logger.debug(message)

        except Exception:
            self.handleError(record)


def setup_ros_log_forwarding(
    node: Optional[Node] = None,
    level: int = logging.INFO,
) -> RosForwarder:
    """
    Forward semantic-inference Python logs to ROS 2.

    Args:
        node: Optional ROS 2 node whose logger should receive messages.
        level: Minimum Python logging level to forward.

    Returns:
        The installed logging handler.
    """
    handler = RosForwarder(
        node=node,
    )
    handler.setLevel(level)

    # Avoid installing duplicate handlers if this function is called more
    # than once.
    for existing_handler in Logger.handlers:
        if isinstance(existing_handler, RosForwarder):
            existing_handler.setLevel(level)
            Logger.setLevel(level)
            return existing_handler

    Logger.addHandler(handler)
    Logger.setLevel(level)

    return handler