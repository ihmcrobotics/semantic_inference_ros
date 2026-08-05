#!/usr/bin/env python3

# Copyright (c) 2026, IHMC Robotics Lab.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Prompt parser methods for open-vocabulary and reasoning-enhanced navigation."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import yaml
from rclpy.logging import get_logger

from semantic_inference_python.client import (
    OpenAIClient,
    OpenAIClientConfig,
)
from semantic_inference_python.config import (
    Config,
    config_field,
    register_config,
)
from semantic_inference_python.models import default_device


LOGGER = get_logger("navigation_prompt_service")


@dataclass
class ObjectsPromptPair:
    """Object, subject, and visual-reasoning prompt."""

    object: str = ""
    subject: str = ""
    prompt: str = ""


@dataclass
class NavigationPrompterOutput:
    """Output produced by a navigation prompter."""

    objects: List[str] = field(default_factory=list)
    objects_embeddings: List[np.ndarray] = field(default_factory=list)
    room_embedding: np.ndarray = field(
        default_factory=lambda: np.array([], dtype=np.float32)
    )
    prompt: str = ""
    room: str = ""
    objects_prompt_pairs: List[ObjectsPromptPair] = field(
        default_factory=list
    )
    success: bool = False
    error: str = ""


class HardCodedPrompter:
    """Hard-coded navigation prompter for demonstrations."""

    def __init__(self, config: "HardCodedPrompterConfig") -> None:
        """Initialize the hard-coded navigation prompter."""
        self.config = config

        self.clip_model = self.config.clip_model.create().to(
            default_device(self.config.use_cuda)
        )
        self.clip_model.eval()

    @classmethod
    def construct(cls, **kwargs) -> "HardCodedPrompter":
        """Construct a hard-coded prompter from configuration arguments."""
        config = HardCodedPrompterConfig()
        config.update(kwargs)
        return cls(config)

    def generate(
        self,
        prompt: str,
        room: str,
    ) -> NavigationPrompterOutput:
        """Generate navigation information from a known prompt template."""
        output = NavigationPrompterOutput(
            prompt=prompt
        )

        prompt_lower = prompt.lower()

        prompt_key = next(
            (
                key
                for key in self.config.prompt_objects
                if key.lower() in prompt_lower
            ),
            None,
        )

        if prompt_key is None:
            output.error = "Unsupported prompt"
            return output

        prompt_config = self.config.prompt_objects[
            prompt_key
        ]

        if prompt_config["required_room"]:
            if not room.isdigit():
                output.error = "Room must be an integer"
                return output

            output.room = f"R({room})"
        else:
            output.room = room

        object_names = list(
            prompt_config["objects"]
        )
        output.objects.extend(object_names)

        if object_names:
            embeddings = (
                self.clip_model
                .embed_text(object_names)
                .detach()
                .cpu()
                .numpy()
            )

            output.objects_embeddings = [
                embeddings[index]
                for index in range(len(object_names))
            ]

        room_description = prompt_config.get(
            "room_desc"
        )

        if room_description:
            output.room_embedding = (
                self.clip_model
                .embed_text([room_description])
                .detach()
                .cpu()
                .numpy()[0]
            )

        output.success = True
        return output


@register_config(
    "navigation_prompter",
    name="hard_coded",
    constructor=HardCodedPrompter,
)
@dataclass
class HardCodedPrompterConfig(Config):
    """Configuration for the hard-coded navigation prompter."""

    clip_model: Any = config_field(
        "clip",
        default="open_clip",
    )
    use_cuda: bool = True

    prompt_objects: Dict[str, Dict[str, Any]] = field(
        default_factory=lambda: {
            "clean": {
                "required_room": True,
                "objects": ["chair"],
                "room_desc": None,
            },
            "prepare": {
                "required_room": False,
                "objects": ["monitor"],
                "room_desc": (
                    "a place with monitors or computers"
                ),
            },
        }
    )


class OpenAIPrompter:
    """Generate navigation prompts using an OpenAI-backed parser."""

    def __init__(
        self,
        config: "OpenAIPrompterConfig",
    ) -> None:
        """Initialize the OpenAI navigation prompter."""
        self.config = config

        self.clip_model = self.config.clip_model.create().to(
            default_device(self.config.use_cuda)
        )
        self.clip_model.eval()

        self.system_prompt = ""
        self.labels: List[str] = []

        self._load_system_prompt()
        self._load_labels()
        self._load_examples()

        self.client = OpenAIClient(
            config=self.config.client_config,
            system_prompt=self.system_prompt,
        )

        LOGGER.info(
            "Initialized OpenAI navigation prompter."
        )

    @classmethod
    def construct(
        cls,
        **kwargs,
    ) -> "OpenAIPrompter":
        """Construct an OpenAI prompter from configuration arguments."""
        config = OpenAIPrompterConfig()
        config.update(kwargs)
        return cls(config)

    def _load_system_prompt(self) -> None:
        """Load the main system prompt."""
        path = Path(
            self.config.system_prompts_path
        ).expanduser()

        if not path.is_file():
            if self.config.system_prompts_path:
                LOGGER.warning(
                    f"System prompt file does not exist: '{path}'"
                )
            return

        self.system_prompt = path.read_text(
            encoding="utf-8"
        ).strip()

    def _load_labels(self) -> None:
        """Load allowed object labels."""
        path = Path(
            self.config.labels_path
        ).expanduser()

        if not path.is_file():
            if self.config.labels_path:
                LOGGER.warning(
                    f"Labels file does not exist: '{path}'"
                )
            return

        with path.open(
            "r",
            encoding="utf-8",
        ) as labels_file:
            data = yaml.safe_load(labels_file) or {}

        object_label_ids = set(
            data.get("object_labels", [])
        )

        label_names = data.get(
            "label_names",
            [],
        )

        self.labels = [
            str(label["name"]).lower()
            for label in label_names
            if (
                isinstance(label, dict)
                and label.get("label") in object_label_ids
                and "name" in label
            )
        ]

        if self.labels:
            formatted_labels = ", ".join(
                self.labels
            )
            self.system_prompt += (
                f"\n[{formatted_labels}]\n"
            )

    def _load_examples(self) -> None:
        """Load prompt examples."""
        path = Path(
            self.config.examples_path
        ).expanduser()

        if not path.is_file():
            if self.config.examples_path:
                LOGGER.warning(
                    f"Examples file does not exist: '{path}'"
                )
            return

        examples = path.read_text(
            encoding="utf-8"
        ).strip()

        if examples:
            self.system_prompt += (
                f"\n{examples}\n"
            )

    @staticmethod
    def _to_json(
        response: str,
    ) -> Dict[str, Any]:
        """Convert a Markdown or plain JSON response to a dictionary."""
        cleaned_response = response.strip()

        if cleaned_response.startswith(
            "```json"
        ):
            cleaned_response = cleaned_response[
                len("```json"):
            ].strip()
        elif cleaned_response.startswith("```"):
            cleaned_response = cleaned_response[
                len("```"):
            ].strip()

        if cleaned_response.endswith("```"):
            cleaned_response = cleaned_response[
                :-3
            ].strip()

        parsed = json.loads(
            cleaned_response
        )

        if not isinstance(parsed, dict):
            raise ValueError(
                "Navigation response must be a JSON object."
            )

        return parsed

    def _is_known_label(
        self,
        object_name: str,
    ) -> bool:
        """Return whether an object is part of the configured label space."""
        if not self.labels:
            return True

        return object_name.lower() in self.labels

    @staticmethod
    def _normalize_object_name(
        object_name: str,
    ) -> str:
        """Normalize object names used by the navigation system."""
        normalized = object_name.strip().lower()

        if normalized == "table":
            return "desk"

        return normalized

    def generate(
        self,
        prompt: str,
        room: str,
    ) -> NavigationPrompterOutput:
        """Generate navigation prompts using the configured OpenAI client."""
        output = NavigationPrompterOutput(
            prompt=prompt
        )

        if not (
            room.isdigit()
            or room in {"all", "find"}
        ):
            output.error = (
                "Room must be an integer, 'all', or 'find'"
            )
            return output

        output.room = (
            f"R({room})"
            if room.isdigit()
            else room
        )

        try:
            response, success = (
                self.client.generate_response(
                    f"Task: {prompt}"
                )
            )

            if not success:
                output.error = str(response)
                return output

            response_data = self._to_json(
                response
            )

        except Exception as exception:
            output.error = (
                "Error generating or parsing response: "
                f"{exception}"
            )
            LOGGER.error(output.error)
            return output

        raw_objects = response_data.get(
            "objects"
        )

        if not isinstance(raw_objects, list):
            output.error = (
                "Response does not contain a valid 'objects' list"
            )
            return output

        output.objects = []

        for object_name in raw_objects:
            normalized_name = (
                self._normalize_object_name(
                    str(object_name)
                )
            )

            if (
                normalized_name
                and normalized_name not in output.objects
            ):
                output.objects.append(
                    normalized_name
                )

        interactions = response_data.get(
            "interactions",
            {},
        )

        if isinstance(interactions, dict):
            interaction_values = (
                interactions.values()
            )
        elif isinstance(interactions, list):
            interaction_values = interactions
        else:
            interaction_values = []

        for pair in interaction_values:
            if not isinstance(pair, dict):
                continue

            pair_objects = pair.get(
                "objects",
                [],
            )
            pair_prompt = str(
                pair.get("prompt", "")
            )

            if (
                not isinstance(pair_objects, list)
                or len(pair_objects) < 2
                or not pair_prompt
            ):
                continue

            object_name = self._normalize_object_name(
                str(pair_objects[0])
            )
            subject_name = self._normalize_object_name(
                str(pair_objects[1])
            )

            prompt_lower = pair_prompt.lower()

            if (
                object_name not in prompt_lower
                or subject_name not in prompt_lower
            ):
                continue

            if (
                not self._is_known_label(object_name)
                or not self._is_known_label(subject_name)
            ):
                continue

            output.objects_prompt_pairs.append(
                ObjectsPromptPair(
                    object=object_name,
                    subject=subject_name,
                    prompt=pair_prompt,
                )
            )

            if object_name not in output.objects:
                output.objects.append(
                    object_name
                )

            if subject_name not in output.objects:
                output.objects.append(
                    subject_name
                )

        if output.objects:
            embeddings = (
                self.clip_model
                .embed_text(output.objects)
                .detach()
                .cpu()
                .numpy()
            )

            output.objects_embeddings = [
                embeddings[index]
                for index in range(len(output.objects))
            ]

        output.success = True
        return output


@register_config(
    "navigation_prompter",
    name="openai",
    constructor=OpenAIPrompter,
)
@dataclass
class OpenAIPrompterConfig(Config):
    """Configuration for the OpenAI navigation prompter."""

    client_config: OpenAIClientConfig = field(
        default_factory=OpenAIClientConfig
    )
    system_prompts_path: str = ""
    examples_path: str = ""
    labels_path: str = ""
    clip_model: Any = config_field(
        "clip",
        default="open_clip",
    )
    use_cuda: bool = True