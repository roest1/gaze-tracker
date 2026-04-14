# classifier

FastAI + PyTorch ResNet classifier fine-tuned on the GTSRB 4GB preprocessed
bundle. Frozen-head transfer learning on ImageNet-pretrained ResNet-34
reaches ~96% test accuracy on `data0` in a single `fit_one_cycle` pass.

## Modules

| file | role |
|---|---|
| `config.py` | `ClassifierConfig` — env-driven paths + hyperparameters |
| `preprocessing.py` | explode GTSRB pickles into `label/image_*.png` trees |
| `dataset.py` | FastAI `DataBlock` builder and held-out test DataLoader |
| `model.py` | backbone resolution (ResNet 18/34/50) + `vision_learner` factory |
| `train.py` | frozen-head `fit_one_cycle` |
| `evaluate.py` | accuracy, classification report, confusion matrix, ROC/AUC |

## End-to-end

```bash
# 1. install deps
uv sync

# 2. download the 4GB bundle (see ../../data/README.md for Kaggle token setup)
uv run python scripts/download_data.py

# 3. explode the pickle into PNGs under data/processed/data0images/
uv run python -m dashcam_sign_detector.classifier.preprocessing --data-id 0

# 4. train
uv run python -m dashcam_sign_detector.classifier.train \
    --data-id 0 --backbone resnet34

# 5. evaluate — drops artifacts under reports/data0_resnet34/
uv run python -m dashcam_sign_detector.classifier.evaluate \
    --data-id 0 --backbone resnet34
```

## Configuration

Everything in `ClassifierConfig` can be overridden via env vars:

| env var | default | effect |
|---|---|---|
| `DSD_DATA_ROOT` | `<repo>/data` | root under which `raw/` and `processed/` live |
| `DSD_MODELS_ROOT` | `<repo>/models` | where `classifier_*.pkl` is saved |
| `DSD_REPORTS_ROOT` | `<repo>/reports` | evaluation artifacts |
| `DSD_DATA_ID` | `0` | pickle variant (0..8) |
| `DSD_BACKBONE` | `resnet34` | `resnet18` / `resnet34` / `resnet50` |
| `DSD_IMAGE_SIZE` | `224` | item transform size |
| `DSD_BATCH_SIZE` | `64` | train/valid batch size |
| `DSD_NUM_WORKERS` | `4` | DataLoader workers |

CLI flags on `train.py` / `evaluate.py` override the matching env vars.

## Training schedule

Single `fit_one_cycle` over a frozen pretrained ResNet-34, 3 epochs at
`lr=2.09e-3` (LR suggested by `lr_find` on the original TSC project and
rechecked here). Re-run with `--lr-find` if you change backbone or
dataset variant; the baked-in LR was tuned for ResNet-34 on `data0`.

The earlier TSC recipe also ran two augmented-frozen epochs followed by
one unfrozen differential-LR epoch. On `data0` both regressed validation
accuracy relative to the frozen-clean baseline, so both were removed
(see git history for the old three-phase form).
