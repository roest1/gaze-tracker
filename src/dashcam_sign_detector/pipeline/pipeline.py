"""Detect-then-classify pipeline: a torchvision detector feeds crops into the
FastAI GTSRB classifier trained in Phase A.

The pipeline works on a single RGB image (``np.ndarray`` or ``PIL.Image``)
and returns a list of :class:`PipelineResult` records. Frames captured from
OpenCV are BGR and must be converted with ``cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)``
before being passed in. The realtime loop (Phase C) owns that conversion.

The classifier step is abstracted behind a lightweight ``CropClassifier``
protocol so tests can inject a fake and so a pure-PyTorch classifier (the
v2 upgrade) can drop in without touching the pipeline.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
from fastai.vision.all import PILImage, load_learner
from PIL import Image

from dashcam_sign_detector.classifier.config import ClassifierConfig
from dashcam_sign_detector.detector.detect import Detection, SignDetector


@dataclass(frozen=True)
class PipelineResult:
    """Per-detection output of the full detect-then-classify pipeline."""

    bbox: tuple[int, int, int, int]
    detector_class: str
    detector_score: float
    classifier_class: str
    classifier_confidence: float


def crop_bbox(image: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    """Crop an HxWx3 image to the given bbox, clamped to image bounds.

    Returns an empty array if the bbox lies entirely outside the image or
    degenerates to zero width/height after clamping.
    """
    if image.ndim < 2:
        raise ValueError(f"Expected at least a 2D image, got shape {image.shape}")
    h, w = image.shape[:2]
    x1, y1, x2, y2 = bbox
    x1 = max(0, min(int(x1), w))
    y1 = max(0, min(int(y1), h))
    x2 = max(0, min(int(x2), w))
    y2 = max(0, min(int(y2), h))
    if x2 <= x1 or y2 <= y1:
        return image[0:0, 0:0]
    return image[y1:y2, x1:x2]


class CropClassifier(Protocol):
    """Minimal interface the pipeline needs from a classifier.

    Implementations return the full vocabulary and a per-crop
    ``(predicted_index, confidence)`` pair for each item in ``crops``.
    """

    vocab: list[str]

    def predict_batch(
        self, crops: list[Image.Image]
    ) -> list[tuple[int, float]]: ...


class FastAIClassifier:
    """CropClassifier backed by a FastAI learner exported with ``learn.export()``.

    Loads the frozen ResNet classifier from ``models/classifier_*.pkl`` and
    runs batched inference via FastAI's ``dls.test_dl`` + ``get_preds``.
    """

    def __init__(self, model_path: Path | str, *, cpu: bool | None = None) -> None:
        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(
                f"No exported classifier at {path}. "
                "Run `python -m dashcam_sign_detector.classifier.train` first."
            )
        load_kwargs = {}
        if cpu is not None:
            load_kwargs["cpu"] = cpu
        self.learn = load_learner(path, **load_kwargs)
        self.vocab: list[str] = list(self.learn.dls.vocab)

    def predict_batch(self, crops: list[Image.Image]) -> list[tuple[int, float]]:
        if not crops:
            return []
        items = [PILImage(img.convert("RGB")) for img in crops]
        dl = self.learn.dls.test_dl(items)
        probs, _ = self.learn.get_preds(dl=dl)
        probs_np = probs.numpy()
        indices = probs_np.argmax(axis=1)
        confidences = probs_np[np.arange(len(probs_np)), indices]
        return [(int(i), float(c)) for i, c in zip(indices, confidences, strict=True)]


class DetectionClassificationPipeline:
    """Detect signs with a torchvision detector, classify each crop with FastAI.

    Instances are reusable across frames; hold on to one and call ``run()``
    in a loop. The detector and classifier are injected so tests can fake
    either side cheaply.
    """

    def __init__(
        self,
        detector: SignDetector,
        classifier: CropClassifier,
    ) -> None:
        self.detector = detector
        self.classifier = classifier

    def run(self, image: np.ndarray | Image.Image) -> list[PipelineResult]:
        array = _to_rgb_array(image)
        detections: list[Detection] = self.detector.detect(array)

        crops: list[Image.Image] = []
        kept: list[Detection] = []
        for det in detections:
            patch = crop_bbox(array, det.bbox)
            if patch.size == 0:
                continue
            crops.append(Image.fromarray(patch))
            kept.append(det)

        predictions = self.classifier.predict_batch(crops)
        if len(predictions) != len(kept):
            raise RuntimeError(
                f"Classifier returned {len(predictions)} predictions for {len(kept)} crops."
            )

        results: list[PipelineResult] = []
        for det, (cls_idx, cls_conf) in zip(kept, predictions, strict=True):
            results.append(
                PipelineResult(
                    bbox=det.bbox,
                    detector_class=det.label,
                    detector_score=det.score,
                    classifier_class=self.classifier.vocab[cls_idx],
                    classifier_confidence=cls_conf,
                )
            )
        return results

    def run_batch(
        self, images: Iterable[np.ndarray | Image.Image]
    ) -> list[list[PipelineResult]]:
        """Convenience: run the pipeline over an iterable of images sequentially."""
        return [self.run(img) for img in images]


def _to_rgb_array(image: np.ndarray | Image.Image) -> np.ndarray:
    if isinstance(image, Image.Image):
        return np.asarray(image.convert("RGB"))
    if not isinstance(image, np.ndarray):
        raise TypeError(f"Expected np.ndarray or PIL.Image, got {type(image).__name__}")
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(
            f"Expected an HxWx3 RGB array, got shape {image.shape}. "
            "OpenCV frames are BGR -- convert with cv2.cvtColor(..., cv2.COLOR_BGR2RGB)."
        )
    return image


def build_default_pipeline(
    *,
    detector_model: str | None = None,
    score_threshold: float = 0.5,
    classifier_path: Path | str | None = None,
    device: str | None = None,
) -> DetectionClassificationPipeline:
    """Factory: build a pipeline from the default config.

    - Detector: torchvision ``fasterrcnn_resnet50_fpn_v2`` unless overridden.
    - Classifier: FastAI learner at ``ClassifierConfig().model_path``.
    """
    cfg = ClassifierConfig()
    detector_kwargs = {"score_threshold": score_threshold}
    if device is not None:
        detector_kwargs["device"] = device
    if detector_model is not None:
        detector = SignDetector(model_name=detector_model, **detector_kwargs)
    else:
        detector = SignDetector(**detector_kwargs)

    classifier_path = Path(classifier_path) if classifier_path else cfg.model_path
    classifier = FastAIClassifier(classifier_path)
    return DetectionClassificationPipeline(detector=detector, classifier=classifier)
