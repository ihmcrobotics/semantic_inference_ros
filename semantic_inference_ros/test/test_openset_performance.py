"""Regression checks for inference caching and instance-image conversion."""

import numpy as np
import torch
from std_msgs.msg import Header

from semantic_inference_python.models.openset_segmenter import OpensetSegmenter, Results
from semantic_inference_ros.ros_conversions import Conversions


class TextEncoder:
    def __init__(self):
        self.calls = []

    def embed_text(self, names):
        self.calls.append(names)
        return torch.tensor(
            [[len(name), sum(map(ord, name))] for name in names], dtype=torch.float32
        )


def test_class_embeddings_preserve_order_and_invalidate_for_training():
    model = OpensetSegmenter.__new__(OpensetSegmenter)
    torch.nn.Module.__init__(model)
    model.encoder = TextEncoder()
    model._text_embedding_cache = {}
    model.eval()
    with torch.no_grad():
        first = model._encode_class_names(['chair', 'table', 'chair'])
        reordered = model._encode_class_names(['table', 'chair'])
        assert torch.equal(first[[1, 0]], reordered)
        assert torch.equal(first[0], first[2])
        assert len(model.encoder.calls) == 1
        model._encode_class_names(['door'])
        assert model.encoder.calls[-1] == ['door']
    model.train()
    with torch.no_grad():
        model._encode_class_names(['chair'])
    assert not model._text_embedding_cache


def test_instance_conversion_preserves_overlap_ids_and_empty_results():
    masks = torch.tensor([[[1, 0], [1, 0]], [[0, 1], [1, 0]]], dtype=torch.bool)
    result = Results(masks, None, None, torch.ones(2, 4), None, None, None)
    message = Conversions.to_feature_image(Header(), result)
    decoded = Conversions.bridge.imgmsg_to_cv2(message.image)
    assert np.array_equal(decoded, np.array([[1, 2], [2, 0]], dtype=np.uint16))
    result.masks = None
    assert len(Conversions.to_feature_image(Header(), result).image.data) == 0


def test_clip_bypass_keeps_masks_without_fabricating_features():
    from types import SimpleNamespace
    model = OpensetSegmenter.__new__(OpensetSegmenter)
    torch.nn.Module.__init__(model)
    model._canary_param = torch.nn.Parameter(torch.empty(0))
    model.config = SimpleNamespace(
        enable_clip=False, segmentation=SimpleNamespace(verbose=False),
        min_depth=0.05, dense_representation_radius_m=7,
        object_labels=[1], max_batch=1,
    )
    model.dense_encoder = None
    def unused_patches(*args, **kwargs):
        raise AssertionError("CLIP-off must not extract patches")
    model.patch_extractor = unused_patches
    masks = torch.ones(1, 2, 2, dtype=torch.bool)
    panoptic = torch.ones(2, 2, dtype=torch.int32)
    result = model.encode(
        torch.zeros(2, 2, 3), np.ones((2, 2)), masks,
        torch.tensor([[0, 0, 2, 2]]), torch.tensor([1]), None, panoptic,
    )
    assert torch.equal(result.masks, masks)
    assert torch.equal(result.panoptic_image, panoptic)
    assert result.labels.tolist() == [1]
    assert result.features.numel() == 0
    assert result.image_embedding is None
