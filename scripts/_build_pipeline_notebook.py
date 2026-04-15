"""Build notebooks/03_pipeline_demo.ipynb from source cells.

This is an internal authoring helper, not a user entry point. Notebooks are
a pain to hand-edit as JSON, so we emit the canonical version from Python.
Run with ``uv run python scripts/_build_pipeline_notebook.py``.
"""

from __future__ import annotations

from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "notebooks" / "03_pipeline_demo.ipynb"


def md(text: str):
    return new_markdown_cell(text)


def code(source: str):
    return new_code_cell(source.strip("\n"))


cells = [
    md(
        "# Phase B — detect-then-classify pipeline demo\n"
        "\n"
        "Sanity-check the dashcam-sign-detector pipeline end to end:\n"
        "\n"
        "1. The FastAI ResNet classifier (Phase A artifact) is wired correctly via\n"
        "   `FastAIClassifier` and returns sensible predictions on real GTSRB test crops.\n"
        "2. The torchvision detector loads, runs, and filters to the expected COCO classes.\n"
        "3. `DetectionClassificationPipeline` composes the two stages correctly.\n"
        "\n"
        "**Known limitation of the v1 detector:** it is COCO-pretrained and only knows\n"
        "`stop sign` and `traffic light` out of its 91 classes, which covers almost none\n"
        "of the 43 European GTSRB classes. Running the detector directly on GTSRB crops\n"
        "will return zero detections — that is expected, not a bug. The v2 upgrade is to\n"
        "fine-tune the detector on GTSDB (German Traffic Sign Detection Benchmark), which\n"
        "has full-scene bounding boxes. For now we verify the detector on a separate\n"
        "street-scene image and exercise the crop→classify path with a forced bbox."
    ),
    code(
        """
from __future__ import annotations

import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from dashcam_sign_detector.classifier.config import ClassifierConfig
from dashcam_sign_detector.detector.detect import Detection, SignDetector, list_available_models
from dashcam_sign_detector.pipeline import (
    DetectionClassificationPipeline,
    FastAIClassifier,
    build_default_pipeline,
)

cfg = ClassifierConfig()
print("Classifier path:", cfg.model_path)
print("Classifier exists:", cfg.model_path.exists())
print("Available detectors:", list_available_models())
"""
    ),
    md(
        "## Part 1 — classifier wiring check\n"
        "\n"
        "Load the Phase A learner through the `FastAIClassifier` wrapper and batch-predict\n"
        "a handful of GTSRB test crops. The test split directory is organised as\n"
        "`data{N}test/<class_id>/image_*.png`, so the parent directory of each file is the\n"
        "ground-truth label."
    ),
    code(
        """
classifier = FastAIClassifier(cfg.model_path)
print("Vocab size:", len(classifier.vocab))

test_root = cfg.images_dir / cfg.test_split
assert test_root.exists(), f"Processed test split not found at {test_root}"

all_test = list(test_root.rglob("*.png"))
random.seed(42)
sample_paths = random.sample(all_test, k=16)

pil_crops = [Image.open(p).convert("RGB") for p in sample_paths]
true_labels = [p.parent.name for p in sample_paths]

preds = classifier.predict_batch(pil_crops)
pred_labels = [classifier.vocab[idx] for idx, _ in preds]
pred_confs = [conf for _, conf in preds]

correct = sum(1 for t, p in zip(true_labels, pred_labels) if t == p)
print(f"Correct on 16 random test crops: {correct}/16")
"""
    ),
    code(
        """
fig, axes = plt.subplots(4, 4, figsize=(10, 10))
for ax, crop, truth, pred, conf in zip(
    axes.flat, pil_crops, true_labels, pred_labels, pred_confs
):
    ax.imshow(crop)
    colour = "tab:green" if truth == pred else "tab:red"
    ax.set_title(f"true={truth}  pred={pred}\\nconf={conf:.2f}", color=colour, fontsize=9)
    ax.axis("off")
plt.suptitle("FastAI classifier — random GTSRB test crops", fontsize=12)
plt.tight_layout()
plt.show()
"""
    ),
    md(
        "## Part 2 — detector on a real scene\n"
        "\n"
        "Drop a dashcam frame or street-scene JPG at `notebooks/fixtures/scene.jpg` to\n"
        "exercise the detector. If the fixture is missing, the cell falls back to a\n"
        "random-noise frame, which is a useful liveness check (should return zero\n"
        "detections and not crash)."
    ),
    code(
        """
scene_path = Path("fixtures/scene.jpg")
if scene_path.exists():
    scene_rgb = np.asarray(Image.open(scene_path).convert("RGB"))
    source = f"fixture: {scene_path}"
else:
    rng = np.random.default_rng(seed=0)
    scene_rgb = rng.integers(0, 256, size=(480, 640, 3), dtype=np.uint8)
    source = "synthetic random-noise frame (no fixture supplied)"

print("Scene source:", source)
print("Scene shape:", scene_rgb.shape)

detector = SignDetector(model_name="fasterrcnn_resnet50_fpn_v2", score_threshold=0.5)
detections = detector.detect(scene_rgb)
print(f"Detections: {len(detections)}")
for d in detections:
    print(" -", d)
"""
    ),
    code(
        """
fig, ax = plt.subplots(figsize=(10, 7))
ax.imshow(scene_rgb)
for det in detections:
    x1, y1, x2, y2 = det.bbox
    ax.add_patch(
        plt.Rectangle(
            (x1, y1),
            x2 - x1,
            y2 - y1,
            linewidth=2,
            edgecolor="lime",
            facecolor="none",
        )
    )
    ax.text(
        x1,
        y1 - 5,
        f"{det.label} {det.score:.2f}",
        color="black",
        backgroundcolor="lime",
        fontsize=9,
    )
ax.set_title(f"Detector output ({detector.model_name})")
ax.axis("off")
plt.tight_layout()
plt.show()
"""
    ),
    md(
        "## Part 3 — full pipeline with a forced-bbox detector\n"
        "\n"
        "Since the COCO-pretrained detector will not find anything on the pre-cropped\n"
        "GTSRB test images, we compose the pipeline with a tiny in-notebook detector\n"
        "that always returns a full-frame bbox. This exercises the detect→crop→classify\n"
        "plumbing end to end and lets us confirm that `DetectionClassificationPipeline`\n"
        "yields `PipelineResult` records with the classifier's actual label."
    ),
    code(
        """
class FullFrameDetector:
    \"\"\"Test-only detector: emits a single full-frame bbox tagged as 'stop sign'.\"\"\"

    def detect(self, image):
        h, w = image.shape[:2]
        return [Detection(bbox=(0, 0, w, h), label="stop sign", score=0.95)]

pipe = DetectionClassificationPipeline(detector=FullFrameDetector(), classifier=classifier)

demo_paths = random.sample(all_test, k=4)
for p in demo_paths:
    arr = np.asarray(Image.open(p).convert("RGB"))
    results = pipe.run(arr)
    truth = p.parent.name
    for r in results:
        ok = "OK  " if r.classifier_class == truth else "MISS"
        print(
            f"{ok} file={p.name:>16}  true={truth:>3}  "
            f"pred={r.classifier_class:>3}  conf={r.classifier_confidence:.3f}  "
            f"det={r.detector_class}@{r.detector_score:.2f}"
        )
"""
    ),
    md(
        "## Part 4 — realistic pipeline factory\n"
        "\n"
        "`build_default_pipeline()` is the entry point the realtime loop (Phase C) and\n"
        "the Gradio app (Phase E) will use. It wires the torchvision detector in front of\n"
        "the FastAI classifier with sensible defaults. Swap the detector backbone by\n"
        "passing `detector_model=` — use `ssdlite320_mobilenet_v3_large` when CPU latency\n"
        "matters (HF Spaces free tier), or `fasterrcnn_resnet50_fpn_v2` when accuracy\n"
        "matters (local GPU)."
    ),
    code(
        """
default_pipe = build_default_pipeline(
    detector_model="ssdlite320_mobilenet_v3_large",
    score_threshold=0.3,
)
print("Detector:", default_pipe.detector.model_name)
print("Detector device:", default_pipe.detector.device)
print("Classifier vocab size:", len(default_pipe.classifier.vocab))
"""
    ),
    md(
        "## Next steps\n"
        "\n"
        "- **Phase C**: OpenCV real-time loop reading webcam/video/RTSP, feeding each frame\n"
        "  through `build_default_pipeline().run()` and drawing the annotated bboxes back.\n"
        "- **Phase D**: record or source a dashcam clip, run the pipeline, produce the\n"
        "  README hero GIF.\n"
        "- **v2 upgrade**: fine-tune the detector on GTSDB so it actually sees the 43\n"
        "  GTSRB classes instead of just the two COCO ones. Until then, the README must\n"
        "  document the COCO-pretrained limitation honestly."
    ),
]

nb = new_notebook(cells=cells)
nb["metadata"] = {
    "kernelspec": {
        "display_name": "Python 3 (ipykernel)",
        "language": "python",
        "name": "python3",
    },
    "language_info": {"name": "python"},
}

OUT.parent.mkdir(parents=True, exist_ok=True)
with OUT.open("w", encoding="utf-8") as f:
    nbformat.write(nb, f)
print(f"Wrote {OUT}")
