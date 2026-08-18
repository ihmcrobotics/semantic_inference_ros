# Copyright (c) 2026, IHMC Robotics Lab.
# All rights reserved.
# Extract the visual encoder from NVIDIA Cosmos-Reason2-2B.

import argparse
import json
from pathlib import Path

import torch
from transformers import Qwen3VLForConditionalGeneration


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract Cosmos-Reason2 visual model from full VLM."
    )

    parser.add_argument(
        "--model_name",
        type=str,
        default="nvidia/Cosmos-Reason2-2B",
        help="Name of the Cosmos-Reason2 model to extract.",
        choices=[
            "nvidia/Cosmos-Reason2-2B",
        ],
    )

    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to save the extracted visual model.",
    )

    return parser.parse_args()


def main(model_name: str, output_path: str) -> None:
    """
    Extract the Qwen3-VL visual encoder used by Cosmos-Reason2-2B.

    :param model_name: Hugging Face model name.
    :param output_path: Directory where extracted weights/configs are saved.
    """

    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Loading {model_name}...")

    model = (
        Qwen3VLForConditionalGeneration.from_pretrained(
            model_name,
            dtype=torch.bfloat16,
        )
        .eval()
    )

    # ------------------------------------------------------------
    # Extract visual encoder
    # ------------------------------------------------------------

    vision = model.model.visual

    print("Visual model:")
    print(vision)

    # ------------------------------------------------------------
    # Save full Cosmos/Qwen configuration
    # ------------------------------------------------------------

    config = model.config.to_dict()

    with (output_path / "config.json").open("w") as f:
        json.dump(config, f, indent=4)

    # ------------------------------------------------------------
    # Save vision-specific configuration
    # ------------------------------------------------------------

    vision_config = model.config.vision_config.to_dict()

    with (output_path / "vision_config.json").open("w") as f:
        json.dump(vision_config, f, indent=4)

    # ------------------------------------------------------------
    # Save complete visual encoder
    #
    # This includes:
    #   - patch embedding
    #   - ViT transformer blocks
    #   - rotary positional components
    #   - merger/projector
    #   - DeepStack visual projectors
    # ------------------------------------------------------------

    torch.save(
        vision.state_dict(),
        output_path / "vision.pt",
    )

    print()
    print("Extraction complete.")
    print(f"Saved to: {output_path}")
    print(f"  {output_path / 'config.json'}")
    print(f"  {output_path / 'vision_config.json'}")
    print(f"  {output_path / 'vision.pt'}")


if __name__ == "__main__":
    args = parse_args()

    main(
        model_name=args.model_name,
        output_path=args.output_path,
    )
