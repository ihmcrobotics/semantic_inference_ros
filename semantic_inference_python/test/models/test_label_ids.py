# Copyright (c) 2026, IHMC Robotics Lab.
# All rights reserved.

"""Regression checks for compact exported-engine IDs versus sparse scene IDs."""
import unittest
from semantic_inference_python.models.label_ids import model_to_scene_ids


class LabelIDsTest(unittest.TestCase):
    def test_reported_mismatches(self):
        source = {3: "door", 6: "chair", 9: "sofa", 12: "desk", 18: "refrigerator", 48: "monitor"}
        target = {3: "tree", 6: "windowpane", 9: "door", 12: "chair", 16: "sofa",
                  18: "mirror", 20: "desk", 29: "refrigerator", 48: "poster", 64: "monitor"}
        self.assertEqual(model_to_scene_ids(source, target), {3: 9, 6: 12, 9: 16, 12: 20, 18: 29, 48: 64})

    def test_identity_and_normalization(self):
        self.assertEqual(model_to_scene_ids(["Chair", "window_pane"], {0: "chair", 7: "window pane"}), {0: 0, 1: 7})

    def test_unknown_and_ambiguous_names_fail(self):
        with self.assertRaises(ValueError):
            model_to_scene_ids({0: "escalator"}, {0: "water"})
        with self.assertRaises(ValueError):
            model_to_scene_ids({0: "chair"}, {1: "chair", 2: "Chair"})

    def test_wrapper_remaps_before_panoptic_and_downstream_features(self):
        from types import SimpleNamespace
        from unittest.mock import Mock, patch
        import numpy as np
        import torch
        from semantic_inference_python.models import wrappers

        model = wrappers.YOLOESegmentation.__new__(wrappers.YOLOESegmentation)
        torch.nn.Module.__init__(model)
        model.config = SimpleNamespace(confidence=0.4, output_size=4, verbose=False)
        model.scene_label_names = {3: "tree", 9: "door", 12: "chair", 29: "refrigerator", 64: "monitor"}
        source = torch.tensor([3, 6, 18, 48])
        result = SimpleNamespace(names={3: "door", 6: "chair", 18: "refrigerator", 48: "monitor"},
                                 boxes=SimpleNamespace(cls=source, xyxy=torch.zeros(4, 4), conf=torch.ones(4)),
                                 masks=SimpleNamespace(data=torch.ones(4, 4, 4)))
        model.yoloe = SimpleNamespace(predict=Mock(return_value=[result]))
        with patch.object(wrappers.ultralytics.utils.ops, "scale_masks", side_effect=lambda masks, shape: masks), \
             patch.object(wrappers, "panoptic_image", return_value="panoptic") as pack:
            output = model.forward(np.zeros((4, 4, 3), dtype=np.uint8), device="cpu")
        self.assertEqual(output[2].tolist(), [9, 12, 29, 64])
        self.assertEqual(pack.call_args.args[1].tolist(), [9, 12, 29, 64])
        self.assertEqual(output[4], "panoptic")


if __name__ == "__main__":
    unittest.main()
