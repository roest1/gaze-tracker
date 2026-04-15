"""Unit tests for the detect-then-classify pipeline plumbing.

These tests avoid any real model load so they are fast and hermetic.
Integration against the trained classifier is exercised separately by
``notebooks/03_pipeline_demo.ipynb`` and the Phase A evaluate script.
"""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from dashcam_sign_detector.detector.detect import Detection
from dashcam_sign_detector.pipeline.pipeline import (
    DetectionClassificationPipeline,
    PipelineResult,
    crop_bbox,
)

# ---------- crop_bbox ----------------------------------------------------------


def _gradient_image(h: int = 40, w: int = 50) -> np.ndarray:
    """Build a predictable HxWx3 image where every pixel encodes (y, x, 0)."""
    img = np.zeros((h, w, 3), dtype=np.uint8)
    ys, xs = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    img[..., 0] = ys.astype(np.uint8)
    img[..., 1] = xs.astype(np.uint8)
    return img


def test_crop_bbox_interior_region():
    img = _gradient_image(40, 50)
    crop = crop_bbox(img, (5, 10, 25, 30))
    assert crop.shape == (20, 20, 3)
    assert crop[0, 0, 0] == 10  # y at origin of crop
    assert crop[0, 0, 1] == 5   # x at origin of crop


def test_crop_bbox_clamps_out_of_bounds():
    img = _gradient_image(40, 50)
    crop = crop_bbox(img, (-10, -5, 1000, 1000))
    assert crop.shape == (40, 50, 3)
    np.testing.assert_array_equal(crop, img)


def test_crop_bbox_zero_area_returns_empty():
    img = _gradient_image(40, 50)
    assert crop_bbox(img, (10, 10, 10, 20)).size == 0
    assert crop_bbox(img, (10, 10, 20, 10)).size == 0


def test_crop_bbox_entirely_outside_returns_empty():
    img = _gradient_image(40, 50)
    assert crop_bbox(img, (100, 100, 200, 200)).size == 0
    assert crop_bbox(img, (-200, -200, -100, -100)).size == 0


def test_crop_bbox_bad_shape_raises():
    with pytest.raises(ValueError):
        crop_bbox(np.zeros(10, dtype=np.uint8), (0, 0, 5, 5))


# ---------- DetectionClassificationPipeline with injected fakes ---------------


class _FakeDetector:
    def __init__(self, detections: list[Detection]):
        self._detections = detections

    def detect(self, _image):
        return list(self._detections)


class _FakeClassifier:
    """Deterministic fake: label_id = (bbox area) % vocab_len for traceability."""

    def __init__(self, vocab: list[str]):
        self.vocab = vocab
        self.calls: list[int] = []  # number of crops per call

    def predict_batch(self, crops):
        self.calls.append(len(crops))
        preds: list[tuple[int, float]] = []
        for crop in crops:
            assert isinstance(crop, Image.Image)
            w, h = crop.size
            idx = (w * h) % len(self.vocab)
            preds.append((idx, 0.42))
        return preds


def _rgb_image(h: int = 64, w: int = 96) -> np.ndarray:
    rng = np.random.default_rng(seed=0)
    return rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)


def test_pipeline_returns_one_result_per_valid_detection():
    img = _rgb_image()
    dets = [
        Detection(bbox=(0, 0, 10, 10), label="stop sign", score=0.91),
        Detection(bbox=(5, 5, 25, 25), label="traffic light", score=0.73),
    ]
    classifier = _FakeClassifier(vocab=[str(i) for i in range(43)])
    pipe = DetectionClassificationPipeline(detector=_FakeDetector(dets), classifier=classifier)

    results = pipe.run(img)

    assert len(results) == 2
    assert classifier.calls == [2]  # single batched forward pass
    for result, det in zip(results, dets, strict=True):
        assert isinstance(result, PipelineResult)
        assert result.bbox == det.bbox
        assert result.detector_class == det.label
        assert result.detector_score == det.score
        assert result.classifier_confidence == 0.42
        assert result.classifier_class in classifier.vocab


def test_pipeline_drops_detections_that_clamp_to_empty():
    img = _rgb_image()
    dets = [
        Detection(bbox=(0, 0, 20, 20), label="stop sign", score=0.9),   # kept
        Detection(bbox=(200, 200, 250, 250), label="stop sign", score=0.9),  # dropped
        Detection(bbox=(10, 10, 10, 20), label="stop sign", score=0.9),  # dropped (zero w)
    ]
    classifier = _FakeClassifier(vocab=["a", "b", "c"])
    pipe = DetectionClassificationPipeline(detector=_FakeDetector(dets), classifier=classifier)

    results = pipe.run(img)

    assert len(results) == 1
    assert classifier.calls == [1]
    assert results[0].bbox == (0, 0, 20, 20)


def test_pipeline_no_detections_does_not_call_classifier():
    classifier = _FakeClassifier(vocab=["a"])
    pipe = DetectionClassificationPipeline(detector=_FakeDetector([]), classifier=classifier)

    assert pipe.run(_rgb_image()) == []
    assert classifier.calls == [0]


def test_pipeline_accepts_pil_input():
    img = Image.fromarray(_rgb_image(50, 80))
    dets = [Detection(bbox=(5, 5, 30, 30), label="stop sign", score=0.8)]
    classifier = _FakeClassifier(vocab=["x", "y"])
    pipe = DetectionClassificationPipeline(detector=_FakeDetector(dets), classifier=classifier)

    results = pipe.run(img)

    assert len(results) == 1
    assert results[0].bbox == (5, 5, 30, 30)


def test_pipeline_rejects_bgr_shape_mismatch():
    classifier = _FakeClassifier(vocab=["x"])
    pipe = DetectionClassificationPipeline(detector=_FakeDetector([]), classifier=classifier)

    with pytest.raises(ValueError):
        pipe.run(np.zeros((32, 32), dtype=np.uint8))  # missing channel axis


def test_pipeline_propagates_classifier_length_mismatch():
    class _BadClassifier:
        vocab = ["a"]

        def predict_batch(self, crops):
            return []  # wrong length on purpose

    img = _rgb_image()
    dets = [Detection(bbox=(0, 0, 10, 10), label="stop sign", score=0.9)]
    pipe = DetectionClassificationPipeline(
        detector=_FakeDetector(dets), classifier=_BadClassifier()
    )

    with pytest.raises(RuntimeError, match="predictions"):
        pipe.run(img)
