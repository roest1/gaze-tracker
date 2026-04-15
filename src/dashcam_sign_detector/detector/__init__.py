"""COCO-pretrained object detector stage for the dashcam pipeline."""

from dashcam_sign_detector.detector.detect import (
    DEFAULT_SIGN_COCO_CLASSES,
    Detection,
    SignDetector,
    list_available_models,
)

__all__ = [
    "DEFAULT_SIGN_COCO_CLASSES",
    "Detection",
    "SignDetector",
    "list_available_models",
]
