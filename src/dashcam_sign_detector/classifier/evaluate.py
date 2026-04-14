"""Evaluation artifacts for the trained GTSRB classifier.

Given an exported FastAI learner (``cfg.model_path``) and the test
DataLoader built from ``cfg.images_dir``, produce:

- ``reports/<run_tag>/metrics.json`` with validation + test loss / accuracy
- ``reports/<run_tag>/classification_report.txt`` (sklearn, per-class PRF)
- ``reports/<run_tag>/confusion_matrix.png``
- ``reports/<run_tag>/roc_micro_macro.png``

Label names come from the GTSRB ``label_names.csv`` if it's available, else
from the sorted directory names under the processed images tree.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from fastai.vision.all import ClassificationInterpretation, load_learner
from sklearn.metrics import auc, classification_report, roc_curve
from sklearn.preprocessing import label_binarize

from dashcam_sign_detector.classifier.config import ClassifierConfig
from dashcam_sign_detector.classifier.dataset import build_dataloaders


@dataclass
class EvaluationResults:
    val_loss: float
    val_accuracy: float
    test_loss: float
    test_accuracy: float


def _load_label_names(cfg: ClassifierConfig, fallback: list[str]) -> list[str]:
    csv_path = cfg.label_names_csv
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        if "SignName" in df.columns:
            return df["SignName"].tolist()
    return fallback


def _plot_confusion_matrix(learn, out_path: Path) -> None:
    interp = ClassificationInterpretation.from_learner(learn)
    fig = plt.figure(figsize=(12, 12))
    interp.plot_confusion_matrix(figsize=(12, 12))
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def _plot_roc(y_true: np.ndarray, y_scores: np.ndarray, n_classes: int, out_path: Path) -> None:
    y_true_bin = label_binarize(y_true, classes=list(range(n_classes)))
    fpr: dict = {}
    tpr: dict = {}
    roc_auc: dict = {}
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_scores[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    fpr["micro"], tpr["micro"], _ = roc_curve(y_true_bin.ravel(), y_scores.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= n_classes
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    plt.figure(figsize=(8, 6))
    plt.plot(
        fpr["micro"],
        tpr["micro"],
        label=f'Micro-average ROC (AUC = {roc_auc["micro"]:0.3f})',
        color="deeppink",
        linestyle=":",
        linewidth=4,
    )
    plt.plot(
        fpr["macro"],
        tpr["macro"],
        label=f'Macro-average ROC (AUC = {roc_auc["macro"]:0.3f})',
        color="navy",
        linestyle=":",
        linewidth=4,
    )
    plt.plot([0, 1], [0, 1], "k--", lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("GTSRB classifier — micro/macro ROC")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()


def evaluate(cfg: ClassifierConfig, run_tag: str | None = None) -> EvaluationResults:
    if not cfg.model_path.exists():
        raise FileNotFoundError(
            f"No trained learner at {cfg.model_path}. "
            "Run `python -m dashcam_sign_detector.classifier.train` first."
        )

    tag = run_tag or f"data{cfg.data_id}_{cfg.backbone}"
    report_dir = cfg.reports_root / tag
    report_dir.mkdir(parents=True, exist_ok=True)

    data = build_dataloaders(cfg)
    learn = load_learner(cfg.model_path)
    learn.dls = data.dls

    val_loss, val_acc = learn.validate()
    test_loss, test_acc = learn.validate(dl=data.test_dl)

    print(f"Validation loss={val_loss:.4f} acc={val_acc:.4f}")
    print(f"Test       loss={test_loss:.4f} acc={test_acc:.4f}")

    preds, y_true = learn.get_preds(dl=data.test_dl)
    y_pred = preds.argmax(dim=1).numpy()
    y_true_np = y_true.numpy()
    preds_np = preds.numpy()

    target_names = _load_label_names(cfg, fallback=data.label_names)
    if len(target_names) != len(data.label_names):
        target_names = data.label_names

    report = classification_report(y_true_np, y_pred, target_names=target_names, digits=4)
    (report_dir / "classification_report.txt").write_text(report)

    _plot_confusion_matrix(learn, report_dir / "confusion_matrix.png")
    _plot_roc(y_true_np, preds_np, n_classes=len(data.label_names),
              out_path=report_dir / "roc_micro_macro.png")

    results = EvaluationResults(
        val_loss=float(val_loss),
        val_accuracy=float(val_acc),
        test_loss=float(test_loss),
        test_accuracy=float(test_acc),
    )
    (report_dir / "metrics.json").write_text(json.dumps(results.__dict__, indent=2))
    print(f"Wrote evaluation artifacts to {report_dir}")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the GTSRB classifier.")
    parser.add_argument("--data-id", type=int, default=None)
    parser.add_argument("--backbone", choices=["resnet18", "resnet34", "resnet50"], default=None)
    parser.add_argument("--run-tag", type=str, default=None)
    args = parser.parse_args()

    cfg = ClassifierConfig()
    if args.data_id is not None:
        cfg.data_id = args.data_id
    if args.backbone is not None:
        cfg.backbone = args.backbone

    evaluate(cfg, run_tag=args.run_tag)


if __name__ == "__main__":
    main()
