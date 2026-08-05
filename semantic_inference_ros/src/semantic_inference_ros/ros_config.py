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
"""ROS 2 configuration parsing infrastructure."""

from __future__ import annotations

from typing import Any, Dict, Type, TypeVar

from rclpy.node import Node

from semantic_inference_python import Config


ConfigType = TypeVar("ConfigType", bound=Config)


def _insert_nested(
    parameters: Dict[str, Any],
    name: str,
    value: Any,
) -> None:
    """
    Insert a ROS 2 parameter into a nested dictionary.

    ROS 2 parameter names normally use dots for nesting:

        model.cuda
        model.segmentation.type
        worker.min_separation_s

    This converts them to:

        {
            "model": {
                "cuda": ...,
                "segmentation": {
                    "type": ...
                }
            },
            "worker": {
                "min_separation_s": ...
            }
        }
    """
    name = name.strip("./")

    if not name:
        return

    parts = [
        part
        for part in name.replace("/", ".").split(".")
        if part
    ]

    current = parameters

    for part in parts[:-1]:
        existing = current.get(part)

        if existing is None:
            current[part] = {}
        elif not isinstance(existing, dict):
            # A scalar parameter already occupies this path. Replace it
            # with a dictionary so that nested parameters can be inserted.
            current[part] = {}

        current = current[part]

    key = parts[-1]
    existing = current.get(key)

    if isinstance(existing, dict) and isinstance(value, dict):
        existing.update(value)
    elif isinstance(existing, list) and isinstance(value, list):
        current[key] = existing + value
    else:
        current[key] = value


def _relative_parameter_name(
    parameter_name: str,
    namespace: str,
) -> str | None:
    """Return a parameter name relative to the requested namespace."""
    normalized_name = parameter_name.strip("./")
    normalized_namespace = namespace.strip("./")

    if not normalized_namespace:
        return normalized_name

    if normalized_name == normalized_namespace:
        return ""

    prefix = normalized_namespace + "."

    if not normalized_name.startswith(prefix):
        return None

    return normalized_name[len(prefix):]


def load_ros2_params(
    node: Node,
    namespace: str = "",
) -> Dict[str, Any]:
    """
    Load declared ROS 2 parameters from a node.

    Args:
        node: ROS 2 node that owns the parameters.
        namespace: Optional parameter prefix. For example, ``"model"`` loads
            only parameters beginning with ``model.``.

    Returns:
        Nested dictionary containing the selected parameter values.

    Important:
        ROS 2 has no global parameter server equivalent to ROS 1. Parameters
        belong to individual nodes and must be declared before they can be
        retrieved.
    """
    if not isinstance(node, Node):
        raise TypeError(
            f"Expected an rclpy.node.Node, received {type(node).__name__}."
        )

    normalized_namespace = namespace.strip("./")

    prefixes = (
        [normalized_namespace]
        if normalized_namespace
        else []
    )

    result = node.list_parameters(
        prefixes=prefixes,
        depth=100,
    )

    parameters: Dict[str, Any] = {}

    for parameter_name in result.names:
        relative_name = _relative_parameter_name(
            parameter_name,
            normalized_namespace,
        )

        if relative_name is None or not relative_name:
            continue

        parameter = node.get_parameter(parameter_name)

        _insert_nested(
            parameters,
            relative_name,
            parameter.value,
        )

    return parameters


def load_from_ros2(
    cls: Type[ConfigType],
    node: Node,
    namespace: str = "",
) -> ConfigType:
    """
    Populate a configuration object from ROS 2 node parameters.

    Args:
        cls: Configuration class derived from ``Config``.
        node: ROS 2 node containing the declared parameters.
        namespace: Optional parameter prefix.

    Returns:
        Populated configuration instance.
    """
    if not isinstance(cls, type) or not issubclass(cls, Config):
        raise TypeError(
            f"{cls!r} is not a semantic_inference_python.Config class."
        )

    instance = cls()
    parameters = load_ros2_params(
        node=node,
        namespace=namespace,
    )

    if parameters:
        instance.update(parameters)

    return instance


# Temporary compatibility alias. This allows older converted files that still
# import ``load_from_ros`` to run while the repository migration is completed.
#
# New ROS 2 code should call ``load_from_ros2`` explicitly.
def load_from_ros(
    cls: Type[ConfigType],
    node: Node,
    namespace: str = "",
) -> ConfigType:
    """Compatibility wrapper around :func:`load_from_ros2`."""
    node.get_logger().warning(
        "load_from_ros() is deprecated in the ROS 2 port; "
        "use load_from_ros2() instead."
    )

    return load_from_ros2(
        cls=cls,
        node=node,
        namespace=namespace,
    )