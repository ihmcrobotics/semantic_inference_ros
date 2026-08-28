#!/usr/bin/env python3

# Copyright (c) 2026, IHMC Robotics Lab.
# All rights reserved.

"""Export a prompt-configured YOLOE or YOLO-World model to TensorRT."""

import argparse
import os
from pathlib import Path
import shutil

import torch
import yaml
from ultralytics import YOLO, YOLOE, YOLOWorld


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    default_assets = Path(
        os.environ.get(
            "SCENE_GRAPH_ASSETS",
            Path.cwd() / "scene_graph_assets",
        )
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default="yoloe-11l-seg.pt",
        help="Source YOLOE or YOLO-World PyTorch checkpoint.",
    )
    parser.add_argument(
        "--labels",
        type=Path,
        required=True,
        help="Label-space YAML containing label_names.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            default_assets
            / "models"
            / "segmentation"
            / "yoloe"
            / "alex-yoloe-11l-seg.engine"
        ),
        help="Destination TensorRT engine path.",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument(
        "--half",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Export an FP16 engine (enabled by default).",
    )
    parser.add_argument(
        "--workspace",
        type=float,
        default=None,
        help="Maximum TensorRT workspace in GiB.",
    )
    return parser.parse_args()


def load_class_names(path: Path) -> list[str]:
    """Load object class names from a Hydra label-space YAML file."""
    path = path.expanduser().resolve()
    with path.open("r", encoding="utf-8") as stream:
        contents = yaml.safe_load(stream)

    if isinstance(contents, list):
        names = [str(item) for item in contents]
    elif isinstance(contents, dict):
        object_ids = set(contents.get("object_labels", []))
        entries = contents.get("label_names", [])
        names = [
            str(entry["name"])
            for entry in entries
            if isinstance(entry, dict)
            and "name" in entry
            and (not object_ids or entry.get("label") in object_ids)
        ]
    else:
        raise ValueError(f"Unsupported label-space structure in '{path}'.")

    names = list(dict.fromkeys(name.strip() for name in names if name.strip()))
    if not names:
        raise ValueError(f"No object class names were found in '{path}'.")
    return names


def main() -> None:
    """Create a fixed-class TensorRT engine from YOLOE or YOLO-World."""
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but PyTorch cannot access a GPU.")

    class_names = load_class_names(args.labels)
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    customized_model = output.with_suffix(".pt")

    is_yolo_world = "world" in Path(args.model).name.lower()
    family = "YOLO-World" if is_yolo_world else "YOLOE"
    print(f"Loading {family} checkpoint: {args.model}")
    print(f"Embedding {len(class_names)} object classes from: {args.labels}")
    if is_yolo_world:
        model = YOLOWorld(args.model)
        model.set_classes(class_names)
    else:
        model = YOLOE(args.model)
        model.set_classes(class_names, model.get_text_pe(class_names))
    model.save(customized_model)

    export_args = {
        "format": "engine",
        "device": args.device,
        "half": args.half,
        "imgsz": args.imgsz,
    }
    if args.workspace is not None:
        export_args["workspace"] = args.workspace

    print(f"Exporting TensorRT engine for {args.device}...")
    exported_path = Path(YOLO(customized_model).export(**export_args)).resolve()
    if exported_path != output:
        shutil.move(str(exported_path), output)

    print(f"Customized checkpoint: {customized_model}")
    print(f"TensorRT engine: {output}")
    print("Use the engine path as model.segmentation.yolo_model_name.")


if __name__ == "__main__":
    main()
