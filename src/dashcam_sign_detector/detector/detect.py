"""Torchvision-based object detector wrapper.

Wraps a COCO-pretrained torchvision detection model and returns a list of
``Detection`` records filtered to the COCO classes that plausibly correspond
to traffic signs (``stop sign``, ``traffic light``). This is the
first stage of the dashcam pipeline; the 43-way GTSRB classification happens
downstream on each crop.

Why torchvision and not YOLOv10: the THU-MIG/yolov10 repo, despite being
described as MIT in some places, ships under AGPL-3.0 (inherited from
Ultralytics). That is incompatible with this repo's MIT license and the
planned Hugging Face Spaces deploy. Torchvision detection models are BSD-3
licensed and ship inside a dependency we already pin.

Supported backbones (see ``list_available_models()``):

- ``fasterrcnn_resnet50_fpn_v2``  -- default; strongest accuracy baseline
- ``fasterrcnn_mobilenet_v3_large_fpn`` -- lighter RPN-based alternative
- ``retinanet_resnet50_fpn_v2``  -- one-stage, good small-object handling
- ``fcos_resnet50_fpn``          -- anchor-free one-stage
- ``ssdlite320_mobilenet_v3_large`` -- fastest CPU option, 320x320 input
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image
from torchvision.models.detection import (
    FasterRCNN_MobileNet_V3_Large_FPN_Weights,
    FasterRCNN_ResNet50_FPN_V2_Weights,
    FCOS_ResNet50_FPN_Weights,
    RetinaNet_ResNet50_FPN_V2_Weights,
    SSDLite320_MobileNet_V3_Large_Weights,
    fasterrcnn_mobilenet_v3_large_fpn,
    fasterrcnn_resnet50_fpn_v2,
    fcos_resnet50_fpn,
    retinanet_resnet50_fpn_v2,
    ssdlite320_mobilenet_v3_large,
)
from torchvision.transforms.functional import to_tensor

DEFAULT_SIGN_COCO_CLASSES: frozenset[str] = frozenset({"stop sign", "traffic light"})


@dataclass(frozen=True)
class Detection:
    """A single post-NMS detection from the object detector."""

    bbox: tuple[int, int, int, int]  # x1, y1, x2, y2 in pixel coordinates
    label: str                        # COCO class name
    score: float                      # model confidence in [0, 1]


_ModelFactory = Callable[..., torch.nn.Module]
_WeightsEnum = type

_MODEL_REGISTRY: dict[str, tuple[_ModelFactory, _WeightsEnum]] = {
    "fasterrcnn_resnet50_fpn_v2": (
        fasterrcnn_resnet50_fpn_v2,
        FasterRCNN_ResNet50_FPN_V2_Weights,
    ),
    "fasterrcnn_mobilenet_v3_large_fpn": (
        fasterrcnn_mobilenet_v3_large_fpn,
        FasterRCNN_MobileNet_V3_Large_FPN_Weights,
    ),
    "retinanet_resnet50_fpn_v2": (
        retinanet_resnet50_fpn_v2,
        RetinaNet_ResNet50_FPN_V2_Weights,
    ),
    "fcos_resnet50_fpn": (
        fcos_resnet50_fpn,
        FCOS_ResNet50_FPN_Weights,
    ),
    "ssdlite320_mobilenet_v3_large": (
        ssdlite320_mobilenet_v3_large,
        SSDLite320_MobileNet_V3_Large_Weights,
    ),
}

DEFAULT_MODEL_NAME = "fasterrcnn_resnet50_fpn_v2"


def list_available_models() -> list[str]:
    return sorted(_MODEL_REGISTRY)


def _resolve_device(device: str | torch.device | None) -> torch.device:
    if device is not None:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _image_to_tensor(image: np.ndarray | Image.Image) -> torch.Tensor:
    """Convert an HxWx3 uint8 RGB image (array or PIL) to a float [0,1] tensor."""
    if isinstance(image, Image.Image):
        pil = image.convert("RGB")
        return to_tensor(pil)
    if not isinstance(image, np.ndarray):
        raise TypeError(f"Expected np.ndarray or PIL.Image, got {type(image).__name__}")
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(
            f"Expected an HxWx3 RGB array, got shape {image.shape}. "
            "If the frame came from OpenCV it is BGR -- convert with cv2.cvtColor."
        )
    return to_tensor(image)


class SignDetector:
    """Thin, license-clean wrapper around a torchvision detection model.

    The detector is always frozen in ``eval()`` mode and wrapped in
    ``torch.inference_mode`` so no autograd state leaks across frames.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        *,
        device: str | torch.device | None = None,
        score_threshold: float = 0.5,
        sign_classes: frozenset[str] | set[str] | None = None,
    ) -> None:
        if model_name not in _MODEL_REGISTRY:
            raise ValueError(
                f"Unknown model {model_name!r}. "
                f"Available: {list_available_models()}"
            )
        factory, weights_enum = _MODEL_REGISTRY[model_name]
        weights = weights_enum.DEFAULT
        self.model_name = model_name
        self.weights = weights
        self.categories: list[str] = list(weights.meta["categories"])
        self.device = _resolve_device(device)
        self.score_threshold = float(score_threshold)
        self.sign_classes = (
            frozenset(sign_classes) if sign_classes is not None else DEFAULT_SIGN_COCO_CLASSES
        )
        unknown = self.sign_classes - set(self.categories)
        if unknown:
            raise ValueError(
                f"sign_classes contains names not in the COCO vocabulary: {sorted(unknown)}"
            )

        self.model = factory(weights=weights)
        self.model.eval()
        self.model.to(self.device)

    @torch.inference_mode()
    def detect(self, image: np.ndarray | Image.Image) -> list[Detection]:
        """Run the detector on a single image and return filtered detections."""
        tensor = _image_to_tensor(image).to(self.device)
        outputs = self.model([tensor])[0]

        boxes = outputs["boxes"].detach().cpu().numpy()
        labels = outputs["labels"].detach().cpu().numpy()
        scores = outputs["scores"].detach().cpu().numpy()

        detections: list[Detection] = []
        for (x1, y1, x2, y2), label_id, score in zip(boxes, labels, scores, strict=True):
            if score < self.score_threshold:
                continue
            if not 0 <= int(label_id) < len(self.categories):
                continue
            name = self.categories[int(label_id)]
            if name not in self.sign_classes:
                continue
            detections.append(
                Detection(
                    bbox=(int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))),
                    label=name,
                    score=float(score),
                )
            )
        return detections
