"""GTSRB data loading for FastAI.

Builds a single unaugmented ``DataLoaders`` from the on-disk PNG tree and a
separate labeled test DataLoader. We originally experimented with selective
rotation augmentation (see the git history of this file), but on `data0` it
degraded val accuracy relative to the frozen-head-only schedule, so the
augmentation machinery was removed.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from fastai.vision.all import (
    CategoryBlock,
    DataBlock,
    DataLoaders,
    GrandparentSplitter,
    ImageBlock,
    Normalize,
    Resize,
    get_image_files,
    imagenet_stats,
)

from dashcam_sign_detector.classifier.config import ClassifierConfig


def _label_from_path(item: Path) -> str:
    return item.parent.name


def _all_labels(images_dir: Path) -> list[str]:
    return sorted({p.parent.name for p in get_image_files(images_dir)})


@dataclass
class ClassifierData:
    dls: DataLoaders
    test_dl: DataLoaders
    label_names: list[str]


def build_dataloaders(cfg: ClassifierConfig) -> ClassifierData:
    """Build train/valid DataLoaders and a held-out labeled test DataLoader."""
    images_dir = cfg.images_dir
    if not images_dir.exists():
        raise FileNotFoundError(
            f"Expected processed images at {images_dir}. "
            "Run `python -m dashcam_sign_detector.classifier.preprocessing` first."
        )

    labels = _all_labels(images_dir)

    block = DataBlock(
        blocks=(ImageBlock, CategoryBlock(vocab=labels)),
        get_items=get_image_files,
        splitter=GrandparentSplitter(
            train_name=cfg.train_split, valid_name=cfg.valid_split
        ),
        get_y=_label_from_path,
        item_tfms=Resize(cfg.image_size),
        batch_tfms=[Normalize.from_stats(*imagenet_stats)],
    )

    dls = block.dataloaders(
        images_dir, bs=cfg.batch_size, num_workers=cfg.num_workers
    )

    test_items = get_image_files(images_dir / cfg.test_split)
    test_dl = dls.test_dl(test_items, with_labels=True)

    return ClassifierData(dls=dls, test_dl=test_dl, label_names=labels)
