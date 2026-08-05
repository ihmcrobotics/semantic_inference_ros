# Portions of the following code and their modifications are originally from
# https://github.com/MIT-SPARK/semantic_inference and are licensed under the following
# license:
# -----------------------------------------------------------------------------
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
# -----------------------------------------------------------------------------
#
# Copyright (c) 2026, IHMC Robotics Lab.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Torch module for refining SAM masks."""

from dataclasses import dataclass

import torch
import torchvision
from torch import nn

from semantic_inference_python.config import Config
from semantic_inference_python.models.wrappers import panoptic_image


@dataclass
class SegmentRefinementConfig(Config):
    """Configuration for segment refinement."""

    dilate_masks: bool = False
    kernel_size: int = 3
    kernel_tolerance: float = 1.0e-3
    dilation_passes: int = 1


class SegmentRefinement(nn.Module):
    """Segment refining module."""

    Config = SegmentRefinementConfig

    def __init__(self, config):
        """Initialize the module with the provided config."""
        super().__init__()
        self.config = config

    @classmethod
    def construct(cls, *args, **kwargs):
        """Construct the module from configuration arguments."""
        config = SegmentRefinementConfig(*args, **kwargs)
        return cls(config)

    def forward(self, masks, bboxes, labels, feature_image, confs):
        """
        Refine computed masks.

        Bounding boxes are assumed to each be [min_x, min_y, max_x, max_y].

        Args:
            masks (torch.Tensor): Boolean tensor of N masks with shape (N, R, C).
            bboxes (torch.Tensor): Integer tensor of N boxes with shape (N, 4).
            labels (torch.Tensor): Integer tensor of N labels with shape (N,).
            feature_image (torch.Tensor): Feature image with shape (R, C, D).
            confs (torch.Tensor): Confidence tensor with shape (N,).

        Returns:
            Tuple containing refined masks, boxes, labels, feature image,
            panoptic image, and confidence values.
        """
        masks = masks.to(torch.bool)
        device = masks.device

        num_masks = masks.size(0)
        if num_masks == 0:
            return masks, bboxes, labels, feature_image, None, confs

        # Order masks by area, largest first.
        mask_sizes = torch.sum(masks, dim=(1, 2))
        sorted_idx = torch.argsort(mask_sizes, descending=True)

        dims = masks.size()
        image = torch.zeros(
            (dims[1], dims[2], 1),
            dtype=sorted_idx.dtype,
            device=device,
        )

        for idx in sorted_idx:
            image[masks[idx]] = idx + 1

        refined_masks = torch.zeros_like(masks)
        for idx in sorted_idx:
            non_maximum_suppression_indices = (image == idx + 1)[:, :, 0]
            refined_masks[idx, non_maximum_suppression_indices] = 1

        # Filter out small border artifacts.
        if self.config.dilate_masks:
            kernel_size = self.config.kernel_size
            refined_masks = refined_masks[:, None, :, :].to(torch.float32)
            kernel = torch.ones(
                (1, 1, kernel_size, kernel_size),
                dtype=refined_masks.dtype,
                device=device,
            )

            # Erode masks.
            for _ in range(self.config.dilation_passes):
                refined_masks = nn.functional.conv2d(
                    refined_masks,
                    kernel,
                    padding="same",
                )
                refined_masks[
                    torch.abs(refined_masks - kernel_size**2)
                    > self.config.kernel_tolerance
                ] = 0

            # Dilate masks.
            for _ in range(self.config.dilation_passes):
                refined_masks = nn.functional.conv2d(
                    refined_masks,
                    kernel,
                    padding="same",
                )

            refined_masks = torch.squeeze(refined_masks, 1).to(torch.bool)

        valid = torch.argwhere(
            torch.sum(refined_masks, dim=(1, 2)) > 0
        ).flatten()

        if valid.numel() == 0:
            empty_masks = refined_masks[:0]
            empty_boxes = bboxes[:0]
            empty_labels = labels[:0] if labels is not None else None
            empty_confs = confs[:0] if confs is not None else None

            return (
                empty_masks,
                empty_boxes,
                empty_labels,
                feature_image,
                None,
                empty_confs.cpu().numpy() if empty_confs is not None else None,
            )

        refined_masks = torch.index_select(refined_masks, 0, valid)
        refined_boxes = torchvision.ops.masks_to_boxes(
            refined_masks
        ).to(torch.int32)
        refined_boxes[:, 2:] += 1

        refined_labels = None
        if labels is not None:
            refined_labels = torch.index_select(labels, 0, valid)

        refined_confs = (
            torch.index_select(confs, 0, valid)
            if confs is not None
            else None
        )

        panoptic_labels = (
            refined_labels
            if refined_labels is not None
            else torch.arange(
                refined_masks.shape[0],
                device=refined_masks.device,
            )
            + 1
        )

        return (
            refined_masks,
            refined_boxes,
            refined_labels,
            feature_image,
            panoptic_image(refined_masks, panoptic_labels),
            (
                refined_confs.cpu().numpy()
                if refined_confs is not None
                else None
            ),
        )