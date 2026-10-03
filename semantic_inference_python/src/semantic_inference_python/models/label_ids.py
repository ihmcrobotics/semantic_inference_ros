# Copyright (c) 2026, IHMC Robotics Lab.
# All rights reserved.

"""Translate model-local class indices to the scene's semantic label IDs."""


def model_to_scene_ids(model_names, scene_names):
    """Match names, never assume an exported engine uses scene label ordering."""
    def normalize(name):
        return " ".join(str(name).lower().replace("_", " ").split())

    target = {}
    for label, name in scene_names.items():
        key = normalize(name)
        if key in target:
            raise ValueError(f"Ambiguous scene class name: {name}")
        target[key] = int(label)
    source = enumerate(model_names) if isinstance(model_names, (list, tuple)) else model_names.items()
    mapping = {}
    for label, name in source:
        key = normalize(name)
        if key not in target:
            raise ValueError(f"Model class {label} ({name}) is absent from the scene label space")
        mapping[int(label)] = target[key]
    return mapping
