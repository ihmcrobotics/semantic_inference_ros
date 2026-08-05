#!/usr/bin/env python3

# Copyright (c) 2026, IHMC Robotics Lab.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Recoloring utilities for semantic and panoptic segmentation."""

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import torch
from supervision.draw.color import Color, ColorPalette

from semantic_inference_python import Config


RgbaColor = Tuple[int, int, int, int]
IdToRgba = Dict[int, RgbaColor]
NameToId = Dict[str, int]
IdToColorName = Dict[int, str]


def parse_csv_to_mappings(
    csv_file_path: Path,
) -> Tuple[IdToRgba, NameToId, IdToColorName]:
    """
    Parse a colormap CSV file.

    Args:
        csv_file_path: Path to the colormap CSV file.

    Returns:
        Tuple containing:
            - mapping from object ID to RGBA color;
            - mapping from object name to object ID;
            - mapping from object ID to human-readable color name.
    """
    name_to_id: NameToId = {}
    id_to_rgba: IdToRgba = {}
    id_to_color_name: IdToColorName = {}

    with csv_file_path.open(
        mode="r",
        encoding="utf-8",
        newline="",
    ) as csv_file:
        reader = csv.DictReader(csv_file)

        required_columns = {
            "name",
            "id",
            "red",
            "green",
            "blue",
            "alpha",
            "color",
        }

        if reader.fieldnames is None:
            raise ValueError(
                f"Colormap CSV '{csv_file_path}' has no header."
            )

        missing_columns = required_columns.difference(
            reader.fieldnames
        )

        if missing_columns:
            raise ValueError(
                f"Colormap CSV '{csv_file_path}' is missing columns: "
                f"{sorted(missing_columns)}"
            )

        for row in reader:
            name = row["name"]
            object_id = int(row["id"])

            rgba = (
                int(row["red"]),
                int(row["green"]),
                int(row["blue"]),
                int(row["alpha"]),
            )

            color_name = str(row["color"])

            # White is reserved as an ignored/background color.
            if rgba == (255, 255, 255, 255):
                continue

            name_to_id[name] = object_id
            id_to_rgba[object_id] = rgba
            id_to_color_name[object_id] = color_name

    id_to_rgba[-1] = (0, 0, 0, 0)
    name_to_id["invalid"] = -1
    id_to_color_name[-1] = "invalid"

    return (
        id_to_rgba,
        name_to_id,
        id_to_color_name,
    )


@dataclass
class RecolorConfig(Config):
    """Configuration for semantic recoloring."""

    colormap_path: Path = Path("")

    def initialize(self) -> None:
        """Resolve and validate the colormap path."""
        self.colormap_path = Path(
            self.colormap_path
        ).expanduser()

        if not self.colormap_path.is_file():
            raise FileNotFoundError(
                f"Colormap file does not exist: "
                f"'{self.colormap_path}'"
            )


class Recolor:
    """Recolor semantic and panoptic segmentation results."""

    def __init__(
        self,
        config: RecolorConfig,
    ) -> None:
        """Initialize mappings from the configured colormap."""
        self.config = config
        self.config.initialize()

        (
            self.id_to_rgba,
            self.name_to_id,
            self.id_to_color_name,
        ) = parse_csv_to_mappings(
            self.config.colormap_path
        )

    def get_color_from_name(
        self,
        name: str,
    ) -> RgbaColor:
        """
        Get the RGBA color associated with an object name.

        Args:
            name: Object or category name.

        Returns:
            RGBA color. Unknown names return transparent black.
        """
        object_id = self.name_to_id.get(name, -1)

        return self.id_to_rgba.get(
            object_id,
            (0, 0, 0, 0),
        )

    def get_color_name_from_name(
        self,
        name: str,
    ) -> str:
        """
        Get the human-readable color associated with an object name.

        Args:
            name: Object or category name.

        Returns:
            Color name, or ``"unknown"`` when unavailable.
        """
        object_id = self.name_to_id.get(name)

        if object_id is None:
            return "unknown"

        return self.id_to_color_name.get(
            object_id,
            "unknown",
        )

    def get_colorpalette(
        self,
        bgr: bool = True,
    ) -> ColorPalette:
        """
        Construct a Supervision color palette.

        Args:
            bgr: When true, convert stored RGB values to BGR ordering.

        Returns:
            Supervision ``ColorPalette``.
        """
        colors = []

        for red, green, blue, _alpha in self.id_to_rgba.values():
            if bgr:
                colors.append(
                    Color(
                        blue,
                        green,
                        red,
                    )
                )
            else:
                colors.append(
                    Color(
                        red,
                        green,
                        blue,
                    )
                )

        return ColorPalette(colors)

    def recolor_panoptic(
        self,
        panoptic: torch.Tensor,
    ) -> torch.Tensor:
        """
        Recolor a panoptic label image.

        Args:
            panoptic: Panoptic label tensor with shape ``(H, W)`` or
                ``(1, H, W)``.

        Returns:
            RGB image tensor with shape ``(H, W, 3)``.
        """
        if panoptic.ndim == 3:
            if panoptic.shape[0] != 1:
                raise ValueError(
                    "Expected panoptic tensor with shape "
                    "(H, W) or (1, H, W), but received "
                    f"{tuple(panoptic.shape)}."
                )

            panoptic = panoptic.squeeze(0)

        if panoptic.ndim != 2:
            raise ValueError(
                "Expected a two-dimensional panoptic image, "
                f"but received shape {tuple(panoptic.shape)}."
            )

        recolored_image = torch.zeros(
            (
                panoptic.shape[0],
                panoptic.shape[1],
                3,
            ),
            dtype=torch.uint8,
            device=panoptic.device,
        )

        for label in torch.unique(panoptic):
            label_id = int(label.item())
            rgba = self.id_to_rgba.get(label_id)

            if rgba is None:
                continue

            red, green, blue, _alpha = rgba

            color = torch.tensor(
                [red, green, blue],
                dtype=torch.uint8,
                device=panoptic.device,
            )

            recolored_image[panoptic == label] = color

        return recolored_image

    def recolor_image(
        self,
        masks: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Recolor object masks according to their semantic labels.

        Args:
            masks: Boolean mask tensor with shape ``(N, H, W)``.
            labels: Label tensor with shape ``(N,)``.

        Returns:
            RGB image tensor with shape ``(H, W, 3)``.
        """
        if masks.ndim != 3:
            raise ValueError(
                "Expected masks with shape (N, H, W), "
                f"but received {tuple(masks.shape)}."
            )

        if labels.ndim != 1:
            labels = labels.flatten()

        if masks.shape[0] != labels.shape[0]:
            raise ValueError(
                "The number of masks and labels must match: "
                f"{masks.shape[0]} masks and "
                f"{labels.shape[0]} labels were provided."
            )

        masks = masks.to(torch.bool)

        recolored_image = torch.zeros(
            (
                masks.shape[1],
                masks.shape[2],
                3,
            ),
            dtype=torch.uint8,
            device=masks.device,
        )

        for mask, label in zip(
            masks,
            labels,
        ):
            label_id = int(label.item())
            rgba = self.id_to_rgba.get(label_id)

            if rgba is None:
                continue

            red, green, blue, _alpha = rgba

            color = torch.tensor(
                [red, green, blue],
                dtype=torch.uint8,
                device=masks.device,
            )

            recolored_image[mask] = color

        return recolored_image