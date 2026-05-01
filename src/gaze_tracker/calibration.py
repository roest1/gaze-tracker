"""Nine-point fixation grid calibration.

Standard textbook eye-tracker calibration: the user fixates each of nine
on-screen points in turn (center, four corners, four edges) while the
tracker captures the 3D gaze feature against the dot's screen position.
The collected (feature, target) pairs feed the 4-parameter linear
GazeModel exactly as before.

Each point runs lock-on (red dot, "look at the dot") then capture (green
dot, frames recorded). No head-pose augmentation: the linear fit assumes
a stable head pose, and any pose-robustness gains come from a downstream
CNN backend, not from inflating the fit's input scatter here.

EAR gating: during each point's lock-on the user's lid aperture stabilizes
on the dot. We sample EAR per eye over the lock-on window and take its
median as the per-point baseline. During the green-dot capture we drop any
frame whose either-eye EAR drifts more than `EAR_TOLERANCE` from baseline —
that catches blinks, half-blinks, and squint transients without forcing
the user to hold an unnatural pose. Without gating, those frames embed
landmark noise (lid-position-dependent iris centroid drift) directly into
the regression.
"""
from __future__ import annotations

import time

import cv2
import numpy as np

from ._display import draw_dot, draw_text, panel_y, screen_size
from .landmarks import FaceMeshTracker
from .mapping import BASIS_CARTESIAN, GazeModel, calibration_path

# 9-point fixation grid: center → 4 corners (TL, TR, BL, BR) → 4 edges (T, B, L, R).
# Center-first lets the user lock on cleanly; corners-before-edges anchors the
# regression early so a mid-calibration abort still yields a usable fit.
GRID_POINTS_NORM: list[tuple[float, float]] = [
    (0.5, 0.5),
    (0.1, 0.1), (0.9, 0.1), (0.1, 0.9), (0.9, 0.9),
    (0.5, 0.1), (0.5, 0.9), (0.1, 0.5), (0.9, 0.5),
]

LOCK_ON_S = 0.8
CAPTURE_S = 1.2
POST_ROLL_S = 3.5
MIN_SAMPLES = 30
EAR_TOLERANCE = 0.08      # absolute EAR units; ~25% of typical open-eye EAR (~0.30)
EAR_BASELINE_MIN_N = 5    # min lock-on frames needed to trust a per-point baseline

# Friendly label per index in GRID_POINTS_NORM. Used by LOOCV warnings to name
# the offending point ("TR" reads better than "(0.9, 0.1)" in the terminal).
GRID_LABELS: tuple[str, ...] = (
    "center",
    "TL", "TR", "BL", "BR",
    "T", "B", "L", "R",
)
LOOCV_OUTLIER_FACTOR = 2.0  # flag a point if its held-out residual > this x median


# --- Helpers ---------------------------------------------------------------


def _sleep_or_abort(seconds: float) -> bool:
    """Wait `seconds`, return True if ESC was pressed."""
    t_end = time.monotonic() + seconds
    while time.monotonic() < t_end:
        if (cv2.waitKey(10) & 0xFF) == 27:
            return True
    return False


def _ear_in_band(
    ear_l: float,
    ear_r: float,
    baseline_l: float,
    baseline_r: float,
    tolerance: float = EAR_TOLERANCE,
) -> bool:
    """True if both eyes' EAR is within `tolerance` of their respective baseline.

    Per-eye gating (rather than averaged) catches asymmetric squints, which
    are the dominant failure mode — a half-blink in one eye corrupts that
    eye's iris centroid even when the other looks fine.
    """
    return (
        abs(ear_l - baseline_l) <= tolerance
        and abs(ear_r - baseline_r) <= tolerance
    )


# Public alias for cross-module reuse (eval, real-time tracking).
ear_in_band = _ear_in_band


# --- LOOCV (post-fit calibration health check) ----------------------------


def _group_samples_by_target(
    samples: list[tuple[tuple[float, float, float], tuple[float, float]]],
) -> dict[tuple[float, float], list[tuple[float, float, float]]]:
    """Group sample features by their target screen position. The capture
    loop reuses the same target tuple object for every sample at one grid
    point, so equality-by-value here corresponds 1:1 to grid points."""
    groups: dict[tuple[float, float], list[tuple[float, float, float]]] = {}
    for feat, target in samples:
        groups.setdefault(target, []).append(feat)
    return groups


def _loocv_residuals(
    samples: list[tuple[tuple[float, float, float], tuple[float, float]]],
    screen_w: int,
    screen_h: int,
) -> list[tuple[tuple[float, float], float]]:
    """Per-point leave-one-out residuals in screen pixels.

    For each unique target in `samples`, fit a fresh GazeModel on samples
    NOT belonging to that target, then predict that target's median feature.
    Returned list is one (target, residual_px) per held-out point.

    A point with high LOOCV residual relative to its peers is one the model
    cannot reproduce from the others — usually because the user blinked or
    glanced away during that point's capture. That sample is structurally
    different from what the rest of the calibration agrees on.
    """
    groups = _group_samples_by_target(samples)
    out: list[tuple[tuple[float, float], float]] = []
    for held_target, held_feats in groups.items():
        train_feats: list[tuple[float, float, float]] = []
        train_targets: list[tuple[float, float]] = []
        for target, feats in groups.items():
            if target == held_target:
                continue
            train_feats.extend(feats)
            train_targets.extend([target] * len(feats))
        train_X = np.asarray(train_feats, dtype=float)
        train_y = np.asarray(train_targets, dtype=float)
        loo_model = GazeModel.fit(
            train_X, train_y, screen_w=screen_w, screen_h=screen_h
        )
        med_feat_arr = np.median(np.asarray(held_feats, dtype=float), axis=0)
        med_feat = (
            float(med_feat_arr[0]), float(med_feat_arr[1]), float(med_feat_arr[2])
        )
        pred = np.array(loo_model.predict(med_feat))
        residual = float(np.linalg.norm(pred - np.array(held_target)))
        out.append((held_target, residual))
    return out


def _label_for_target(
    target_px: tuple[float, float],
    screen_w: int,
    screen_h: int,
) -> str:
    """Map a target's pixel coords back to its GRID_LABELS entry. Returns "?"
    if the target isn't a grid point (only happens for synthetic test data)."""
    nx = target_px[0] / screen_w
    ny = target_px[1] / screen_h
    for i, (gx, gy) in enumerate(GRID_POINTS_NORM):
        if abs(gx - nx) < 0.01 and abs(gy - ny) < 0.01:
            return GRID_LABELS[i]
    return "?"


def loocv_warning(
    residuals: list[tuple[tuple[float, float], float]],
    screen_w: int,
    screen_h: int,
    outlier_factor: float = LOOCV_OUTLIER_FACTOR,
) -> str | None:
    """Build a one-line warning naming any point whose LOOCV residual exceeds
    `outlier_factor` x the median residual across all points. Returns None
    when no point qualifies (or when there are too few points / the median
    is degenerate)."""
    if len(residuals) < 3:
        return None
    rs = [r for _, r in residuals]
    med = float(np.median(rs))
    if med <= 0.0:
        return None
    flagged = [(t, r) for t, r in residuals if r > outlier_factor * med]
    if not flagged:
        return None
    parts = [
        f"{_label_for_target(t, screen_w, screen_h)} ({r:.0f} px, "
        f"{r / med:.1f}x median)"
        for t, r in flagged
    ]
    return (
        f"calibration warning: outlier point(s) — {', '.join(parts)}. "
        "Consider recalibrating."
    )


# --- Phase ----------------------------------------------------------------


def _run_grid_phase(
    canvas: np.ndarray,
    window: str,
    cap: cv2.VideoCapture,
    tracker: FaceMeshTracker,
    screen_w: int,
    screen_h: int,
) -> tuple[
    list[tuple[tuple[float, float, float], tuple[float, float]]],
    bool,
    int,
    list[tuple[float, float, float]],
]:
    samples: list[tuple[tuple[float, float, float], tuple[float, float]]] = []
    head_poses: list[tuple[float, float, float]] = []
    n_points = len(GRID_POINTS_NORM)
    rejected_total = 0

    for idx, (nx, ny) in enumerate(GRID_POINTS_NORM):
        target = (nx * screen_w, ny * screen_h)
        progress = f"point {idx + 1}/{n_points}"
        panel_cy = panel_y(ny, screen_h)

        # Lock-on: red dot + copy. We also poll EAR here so the per-point
        # baseline is taken in the same lid state the user holds while
        # fixating — that's the state we want the capture phase to match.
        canvas[:] = 0
        draw_dot(canvas, target)
        draw_text(
            canvas, progress, panel_cy - 120,
            color=(140, 140, 140), scale=0.7, centered=True,
        )
        draw_text(canvas, "Lock your eyes on the dot.", panel_cy + 120, scale=1.2, centered=True)
        cv2.imshow(window, canvas)

        ear_l_buf: list[float] = []
        ear_r_buf: list[float] = []
        t_end = time.monotonic() + LOCK_ON_S
        while time.monotonic() < t_end:
            if (cv2.waitKey(10) & 0xFF) == 27:
                return samples, True, rejected_total, head_poses
            ok, frame = cap.read()
            if not ok:
                continue
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            feat = tracker.extract(rgb)
            if feat is None:
                continue
            ear_l_buf.append(feat.ear_left)
            ear_r_buf.append(feat.ear_right)

        # If MediaPipe couldn't get enough lock-on reads (e.g. user off-camera
        # at start), skip gating for this point rather than reject everything.
        if len(ear_l_buf) >= EAR_BASELINE_MIN_N and len(ear_r_buf) >= EAR_BASELINE_MIN_N:
            baseline_l = float(np.median(ear_l_buf))
            baseline_r = float(np.median(ear_r_buf))
            gating = True
        else:
            baseline_l = baseline_r = 0.0
            gating = False

        # Capture: green dot only. Drop frames that fail the EAR gate.
        canvas[:] = 0
        draw_dot(canvas, target, capturing=True)
        cv2.imshow(window, canvas)

        rejected = 0
        t_end = time.monotonic() + CAPTURE_S
        while time.monotonic() < t_end:
            if (cv2.waitKey(1) & 0xFF) == 27:
                return samples, True, rejected_total, head_poses
            ok, frame = cap.read()
            if not ok:
                continue
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            feat = tracker.extract(rgb)
            if feat is None:
                continue
            if gating and not _ear_in_band(
                feat.ear_left, feat.ear_right, baseline_l, baseline_r
            ):
                rejected += 1
                continue
            samples.append((feat.gaze, target))
            if feat.head_pose is not None:
                head_poses.append(feat.head_pose)
        rejected_total += rejected

    return samples, False, rejected_total, head_poses


# --- Orchestration ---------------------------------------------------------


def run_calibration(
    camera_index: int = 0, basis: str = BASIS_CARTESIAN
) -> GazeModel | None:
    screen_w, screen_h = screen_size()
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open camera index {camera_index}")

    window = "gaze-tracker: calibration"
    cv2.namedWindow(window, cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty(window, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    canvas = np.zeros((screen_h, screen_w, 3), dtype=np.uint8)

    model: GazeModel | None = None

    try:
        with FaceMeshTracker() as tracker:
            samples, aborted, rejected, head_poses = _run_grid_phase(
                canvas, window, cap, tracker, screen_w, screen_h
            )
            if aborted:
                return None

            if len(samples) < MIN_SAMPLES:
                canvas[:] = 0
                draw_text(
                    canvas,
                    f"Only {len(samples)} samples captured (need {MIN_SAMPLES}). Try again.",
                    screen_h // 2,
                    color=(0, 0, 255),
                    centered=True,
                )
                cv2.imshow(window, canvas)
                cv2.waitKey(2500)
                return None

            features = np.asarray([s[0] for s in samples], dtype=float)
            targets = np.asarray([s[1] for s in samples], dtype=float)
            model = GazeModel.fit(
                features, targets,
                screen_w=screen_w, screen_h=screen_h, basis=basis,
            )
            # Per-axis median of the head pose across all admitted frames.
            # Stored on the model so realtime tracking can refuse predictions
            # when the user's head has rotated out of the calibrated posture.
            if head_poses:
                hp_arr = np.asarray(head_poses, dtype=float)
                model.head_pose_baseline = (
                    float(np.median(hp_arr[:, 0])),
                    float(np.median(hp_arr[:, 1])),
                    float(np.median(hp_arr[:, 2])),
                )
            model.save(calibration_path())

            preds = np.array([model.predict(tuple(f)) for f in features])
            residuals = np.linalg.norm(preds - targets, axis=1)
            rmse = float(np.sqrt(np.mean(residuals**2)))
            median_err = float(np.median(residuals))
            p95 = float(np.percentile(residuals, 95))

            loocv_res = _loocv_residuals(samples, screen_w, screen_h)
            warn = loocv_warning(loocv_res, screen_w, screen_h)

            canvas[:] = 0
            draw_text(
                canvas,
                f"Calibration complete: {len(samples)} samples "
                f"across {len(GRID_POINTS_NORM)} points",
                screen_h // 2 - 80,
                scale=1.0,
                centered=True,
            )
            draw_text(
                canvas,
                f"median {median_err:.0f} px   |   95th pct {p95:.0f} px   |   RMSE {rmse:.0f} px",
                screen_h // 2,
                color=(0, 220, 0),
                centered=True,
            )
            if warn is not None:
                draw_text(
                    canvas,
                    "calibration warning: see terminal",
                    screen_h // 2 + 40,
                    color=(0, 165, 255),  # orange-ish
                    scale=0.75,
                    centered=True,
                )
            draw_text(
                canvas,
                "Press any key or wait...",
                screen_h // 2 + 100,
                color=(140, 140, 140),
                scale=0.7,
                centered=True,
            )
            cv2.imshow(window, canvas)

            print(
                f"[gaze-tracker] fit {len(samples)} samples (basis={basis}, "
                f"EAR-gated: {rejected} frames rejected); "
                f"median={median_err:.1f} px, p95={p95:.1f} px, rmse={rmse:.1f} px"
            )
            if model.head_pose_baseline is not None:
                yaw, pitch, roll = model.head_pose_baseline
                print(
                    f"[gaze-tracker] head pose baseline: "
                    f"yaw={yaw:.1f}deg pitch={pitch:.1f}deg roll={roll:.1f}deg"
                )
            if warn is not None:
                print(f"[gaze-tracker] {warn}")

            t_end = time.monotonic() + POST_ROLL_S
            while time.monotonic() < t_end:
                key = cv2.waitKey(20) & 0xFF
                if key != 255:
                    break
    finally:
        cap.release()
        cv2.destroyWindow(window)

    return model
