"""Path and hyperparameter configuration for the GTSRB classifier.

All paths are resolvable from environment variables so the same code runs in
a local WSL workspace, inside a container, or on a remote box. Nothing is
hardcoded to any user's home directory.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _env_path(var: str, default: Path) -> Path:
    value = os.environ.get(var)
    return Path(value).expanduser().resolve() if value else default


@dataclass
class ClassifierConfig:
    """Filesystem layout and training knobs for the classifier."""

    repo_root: Path = field(default_factory=_repo_root)

    data_root: Path = field(
        default_factory=lambda: _env_path("DSD_DATA_ROOT", _repo_root() / "data")
    )
    models_root: Path = field(
        default_factory=lambda: _env_path("DSD_MODELS_ROOT", _repo_root() / "models")
    )
    reports_root: Path = field(
        default_factory=lambda: _env_path("DSD_REPORTS_ROOT", _repo_root() / "reports")
    )

    data_id: int = int(os.environ.get("DSD_DATA_ID", "0"))
    backbone: str = os.environ.get("DSD_BACKBONE", "resnet34")
    image_size: int = int(os.environ.get("DSD_IMAGE_SIZE", "224"))
    batch_size: int = int(os.environ.get("DSD_BATCH_SIZE", "64"))
    num_workers: int = int(os.environ.get("DSD_NUM_WORKERS", "4"))
    seed: int = int(os.environ.get("DSD_SEED", "42"))

    train_epochs: int = 3
    learning_rate: float = 2.0892962347716093e-03

    @property
    def raw_dir(self) -> Path:
        return self.data_root / "raw" / "gtsrb_4gb"

    @property
    def images_dir(self) -> Path:
        return self.data_root / "processed" / f"data{self.data_id}images"

    @property
    def train_split(self) -> str:
        return f"data{self.data_id}train"

    @property
    def valid_split(self) -> str:
        return f"data{self.data_id}validation"

    @property
    def test_split(self) -> str:
        return f"data{self.data_id}test"

    @property
    def is_grayscale(self) -> bool:
        return self.data_id >= 4

    @property
    def color_channels(self) -> int:
        return 1 if self.is_grayscale else 3

    @property
    def pickle_path(self) -> Path:
        return self.raw_dir / f"data{self.data_id}.pickle"

    @property
    def label_names_csv(self) -> Path:
        return self.raw_dir / "label_names.csv"

    @property
    def model_path(self) -> Path:
        return self.models_root / f"classifier_data{self.data_id}_{self.backbone}.pkl"

    def ensure_dirs(self) -> None:
        for p in (self.data_root, self.models_root, self.reports_root):
            p.mkdir(parents=True, exist_ok=True)
