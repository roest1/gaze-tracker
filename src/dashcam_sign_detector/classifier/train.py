"""Transfer-learning schedule for the GTSRB classifier.

Frozen-head-only ``fit_one_cycle`` over a pretrained ResNet backbone. Early
experiments with a second augmented phase and an unfrozen differential-LR
phase regressed validation accuracy on `data0`, so only the frozen phase is
kept. See the git history of this file for the earlier three-phase recipe.
"""

from __future__ import annotations

import argparse

from fastai.vision.all import Learner

from dashcam_sign_detector.classifier.config import ClassifierConfig
from dashcam_sign_detector.classifier.dataset import (
    ClassifierData,
    build_dataloaders,
)
from dashcam_sign_detector.classifier.model import build_learner


def _print_lr_suggestion(learn: Learner) -> None:
    suggestions = learn.lr_find()
    print(f"lr_find suggestion: {suggestions}")


def train(cfg: ClassifierConfig, run_lr_find: bool = False) -> tuple[Learner, ClassifierData]:
    cfg.ensure_dirs()

    print(f"Building dataloaders from {cfg.images_dir}")
    data = build_dataloaders(cfg)
    n_classes = len(data.label_names)
    print(f"Found {n_classes} classes: {data.label_names[:5]}...")

    print(f"Building learner with backbone={cfg.backbone} pretrained=True")
    learn = build_learner(cfg, data.dls, n_classes)

    if run_lr_find:
        _print_lr_suggestion(learn)

    print(f"Training {cfg.train_epochs} epochs (frozen) @ lr={cfg.learning_rate}")
    learn.fit_one_cycle(cfg.train_epochs, lr_max=cfg.learning_rate)

    model_path = cfg.model_path
    model_path.parent.mkdir(parents=True, exist_ok=True)
    learn.export(model_path)
    print(f"Exported trained learner to {model_path}")

    return learn, data


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the GTSRB classifier.")
    parser.add_argument("--data-id", type=int, default=None, help="Pickle id 0..8.")
    parser.add_argument(
        "--backbone",
        choices=["resnet18", "resnet34", "resnet50"],
        default=None,
    )
    parser.add_argument(
        "--lr-find",
        action="store_true",
        help="Run lr_find before training and print the suggestion.",
    )
    args = parser.parse_args()

    cfg = ClassifierConfig()
    if args.data_id is not None:
        cfg.data_id = args.data_id
    if args.backbone is not None:
        cfg.backbone = args.backbone

    train(cfg, run_lr_find=args.lr_find)


if __name__ == "__main__":
    main()
