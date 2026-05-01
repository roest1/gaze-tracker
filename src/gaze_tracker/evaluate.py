"""Held-out evaluation for the gaze model.

Why this exists: the RMSE the calibration flow prints is *training error* — the
model was fit on the same 9 grid points it's being scored against. With future
changes (polynomial bases, polar coords, head-pose features), you cannot honestly
compare alternative models without a held-out eval. This module is the bench
against which everything else is measured.

Eval point layout: 4x4 stratified-jittered grid (16 points). Cell centers at
[0.20, 0.40, 0.60, 0.80] — every center is >=0.10 from any 9-point calibration
position ({0.1, 0.5, 0.9}), so even after the maximum jitter no eval sample
lands on a calibration point. Rejection sampling stays in as a safety net.
Seed defaults to a hash of the calibration file so the same calibration always
sees the same eval points (model-change diffs aren't contaminated by point
layout variance).

Degrees of visual angle: pixel error -> meters via monitor DPI; meters ->
degrees via arctan(err / face_distance). Defaults assume 96 DPI and 50 cm
viewing distance and warn if the user didn't override DPI (no portable Wayland
DPI query). The face-distance assumption is the load-bearing approximation —
1 deg at 50 cm is ~0.87 cm; at 70 cm it's ~1.22 cm. Caveat called out in the
log so historical comparisons don't silently mix viewing distances.

Per-point measurement: median of EAR-gated 3D gaze features over the capture
window, then a single model.predict() call. With the current linear model this
is equivalent to taking the median of per-frame predictions.
"""
from __future__ import annotations

import csv
import hashlib
import math
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from ._display import draw_dot, draw_text, panel_y, screen_size
from .calibration import (
    EAR_BASELINE_MIN_N,
    GRID_POINTS_NORM,
    ear_in_band,
)
from .landmarks import HEAD_POSE_GATE_DEG, FaceMeshTracker, head_pose_max_dev_deg
from .mapping import ALL_BASES, GazeModel, calibration_path

EVAL_LOCK_ON_S = 0.6
EVAL_CAPTURE_S = 0.8
EVAL_POST_ROLL_S = 4.0
DEFAULT_N_COLS = 4
DEFAULT_N_ROWS = 4
DEFAULT_MARGIN = 0.20  # cell-center inset; keeps centers >=0.10 from cal {0.1, 0.9}
DEFAULT_JITTER = 0.04  # +/- in normalized coords; total spread per axis = 2*jitter
MIN_DIST_TO_CAL = 0.05
INCH_PER_M = 39.3701


def calibration_hash(path: Path) -> str:
    """Stable identifier for a calibration file. First 16 hex chars of sha256."""
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def seed_from_hash(h: str) -> int:
    """First 8 hex chars of the calibration hash, parsed as int.

    Same calibration -> same seed -> same eval points. Lets you compare model
    edits across runs without point-layout variance contaminating the diff.
    """
    return int(h[:8], 16)


def make_eval_points(
    seed: int,
    n_cols: int = DEFAULT_N_COLS,
    n_rows: int = DEFAULT_N_ROWS,
    margin: float = DEFAULT_MARGIN,
    jitter: float = DEFAULT_JITTER,
    calibration_points: list[tuple[float, float]] | None = None,
    min_dist_to_cal: float = MIN_DIST_TO_CAL,
) -> list[tuple[float, float]]:
    """Stratified-jittered eval points in normalized [0, 1] screen coords.

    Cell centers are spaced uniformly inside [margin, 1-margin]. Each sample
    is the cell center plus uniform noise in [-jitter, +jitter] per axis.
    Rejection-sampled (up to 8 retries) to stay clear of any calibration
    grid point — falling back to acceptance if 8 retries can't dodge it.
    """
    if n_cols < 1 or n_rows < 1:
        raise ValueError("n_cols and n_rows must be >= 1")
    if jitter < 0:
        raise ValueError("jitter must be >= 0")
    if margin < 0 or margin >= 0.5:
        raise ValueError("margin must be in [0, 0.5)")
    rng = np.random.default_rng(seed)
    cal = calibration_points or []
    xs = (
        np.linspace(margin, 1.0 - margin, n_cols) if n_cols > 1 else np.array([0.5])
    )
    ys = (
        np.linspace(margin, 1.0 - margin, n_rows) if n_rows > 1 else np.array([0.5])
    )
    out: list[tuple[float, float]] = []
    for y in ys:
        for x in xs:
            cand = (float(x), float(y))
            for _ in range(8):
                jx = float(rng.uniform(-jitter, jitter))
                jy = float(rng.uniform(-jitter, jitter))
                cand = (
                    float(np.clip(x + jx, margin, 1.0 - margin)),
                    float(np.clip(y + jy, margin, 1.0 - margin)),
                )
                if all(
                    math.hypot(cand[0] - cx, cand[1] - cy) > min_dist_to_cal
                    for cx, cy in cal
                ):
                    break
            out.append(cand)
    return out


def pixel_error_to_degrees(
    err_px: float, dpi: float, face_distance_cm: float
) -> float:
    """Convert a screen-pixel error to degrees of visual angle.

    err_meters = (err_px / dpi) * (1 inch / INCH_PER_M)
    degrees    = atan2(err_meters, face_distance_meters)
    """
    if dpi <= 0:
        raise ValueError("dpi must be positive")
    if face_distance_cm <= 0:
        raise ValueError("face_distance_cm must be positive")
    err_m = (err_px / dpi) / INCH_PER_M
    distance_m = face_distance_cm / 100.0
    return math.degrees(math.atan2(err_m, distance_m))


@dataclass(frozen=True)
class EvalReport:
    n_points: int
    n_ear_dropped: int   # frames rejected by the rolling EAR gate (blinks/squints)
    n_pose_dropped: int  # frames rejected by the head-pose gate (out-of-baseline pose)
    median_px: float
    p95_px: float
    rmse_px: float
    median_deg: float
    p95_deg: float
    rmse_deg: float
    monitor_dpi: float
    face_distance_cm: float
    calibration_hash: str
    timestamp_iso: str

    def stdout(self) -> str:
        return (
            f"eval: n={self.n_points} "
            f"dropped(ear={self.n_ear_dropped}, pose={self.n_pose_dropped})  "
            f"median={self.median_px:.0f} px ({self.median_deg:.2f} deg)  |  "
            f"p95={self.p95_px:.0f} px ({self.p95_deg:.2f} deg)  |  "
            f"rmse={self.rmse_px:.0f} px ({self.rmse_deg:.2f} deg)\n"
            f"      cal={self.calibration_hash}  dpi={self.monitor_dpi:.0f}  "
            f"distance={self.face_distance_cm:.0f}cm"
        )


CSV_FIELDNAMES = [
    "timestamp_iso",
    "calibration_hash",
    "n_points",
    "n_ear_dropped",
    "n_pose_dropped",
    "monitor_dpi",
    "face_distance_cm",
    "median_px",
    "p95_px",
    "rmse_px",
    "median_deg",
    "p95_deg",
    "rmse_deg",
]


def eval_log_path() -> Path:
    config_home = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config"))
    return config_home / "gaze-tracker" / "eval_log.csv"


def append_eval_log(report: EvalReport, path: Path | None = None) -> Path:
    """Append one row per eval run. Writes header on first creation only.

    On schema change, the prior log file is rotated to `<name>.bak` and a
    fresh file is created so existing trend rows aren't intermixed with
    rows of a different shape (csv readers would parse them inconsistently).
    """
    p = path or eval_log_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.exists():
        with p.open("r", newline="") as f:
            existing_header = next(csv.reader(f), [])
        if existing_header != CSV_FIELDNAMES:
            backup = p.with_suffix(p.suffix + ".bak")
            p.rename(backup)
            print(
                f"[gaze-tracker] eval_log.csv schema changed; "
                f"rotated previous log to {backup}"
            )
    is_new = not p.exists()
    with p.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
        if is_new:
            w.writeheader()
        w.writerow({k: getattr(report, k) for k in CSV_FIELDNAMES})
    return p


def _capture_eval_point(
    canvas: np.ndarray,
    window: str,
    cap: cv2.VideoCapture,
    tracker: FaceMeshTracker,
    target_px: tuple[float, float],
    target_norm_y: float,
    progress_text: str,
    screen_h: int,
    head_pose_baseline: tuple[float, float, float] | None = None,
) -> tuple[list[tuple[float, float, float]], int, int, bool]:
    """Capture gated gaze features for a single eval point.

    Two gates apply, both matching the realtime tracking loop:
      - EAR gate (per-point baseline from lock-on; rejects blinks/squints).
      - Head-pose gate (if `head_pose_baseline` is provided; rejects frames
        where any axis deviates more than HEAD_POSE_GATE_DEG from baseline).

    Returns (captured_features, ear_dropped, pose_dropped, aborted).
    """
    panel_cy = panel_y(target_norm_y, screen_h)

    # Lock-on: red dot + EAR baseline collection. Same shape as calibration's
    # lock-on so the user's lid state matches the per-point baseline.
    canvas[:] = 0
    draw_dot(canvas, target_px)
    draw_text(
        canvas, progress_text, panel_cy - 120,
        color=(140, 140, 140), scale=0.7, centered=True,
    )
    draw_text(
        canvas, "Lock your eyes on the dot.", panel_cy + 120,
        scale=1.2, centered=True,
    )
    cv2.imshow(window, canvas)

    ear_l_buf: list[float] = []
    ear_r_buf: list[float] = []
    t_end = time.monotonic() + EVAL_LOCK_ON_S
    while time.monotonic() < t_end:
        if (cv2.waitKey(10) & 0xFF) == 27:
            return [], 0, 0, True
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

    if (
        len(ear_l_buf) >= EAR_BASELINE_MIN_N
        and len(ear_r_buf) >= EAR_BASELINE_MIN_N
    ):
        baseline_l = float(np.median(ear_l_buf))
        baseline_r = float(np.median(ear_r_buf))
        gating = True
    else:
        baseline_l = baseline_r = 0.0
        gating = False

    # Capture: green dot, gated frames.
    canvas[:] = 0
    draw_dot(canvas, target_px, capturing=True)
    cv2.imshow(window, canvas)

    captured: list[tuple[float, float, float]] = []
    ear_dropped = 0
    pose_dropped = 0
    t_end = time.monotonic() + EVAL_CAPTURE_S
    while time.monotonic() < t_end:
        if (cv2.waitKey(1) & 0xFF) == 27:
            return captured, ear_dropped, pose_dropped, True
        ok, frame = cap.read()
        if not ok:
            continue
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        feat = tracker.extract(rgb)
        if feat is None:
            continue
        if gating and not ear_in_band(
            feat.ear_left, feat.ear_right, baseline_l, baseline_r
        ):
            ear_dropped += 1
            continue
        if (
            head_pose_baseline is not None
            and feat.head_pose is not None
            and head_pose_max_dev_deg(feat.head_pose, head_pose_baseline)
            > HEAD_POSE_GATE_DEG
        ):
            pose_dropped += 1
            continue
        captured.append(feat.gaze)

    return captured, ear_dropped, pose_dropped, False


def _summarize(
    errors_px: list[float],
    n_ear_dropped: int,
    n_pose_dropped: int,
    monitor_dpi: float,
    face_distance_cm: float,
    cal_hash: str,
) -> EvalReport:
    errs = np.asarray(errors_px, dtype=float)
    median_px = float(np.median(errs))
    p95_px = float(np.percentile(errs, 95))
    rmse_px = float(np.sqrt(np.mean(errs**2)))
    return EvalReport(
        n_points=len(errors_px),
        n_ear_dropped=n_ear_dropped,
        n_pose_dropped=n_pose_dropped,
        median_px=median_px,
        p95_px=p95_px,
        rmse_px=rmse_px,
        median_deg=pixel_error_to_degrees(median_px, monitor_dpi, face_distance_cm),
        p95_deg=pixel_error_to_degrees(p95_px, monitor_dpi, face_distance_cm),
        rmse_deg=pixel_error_to_degrees(rmse_px, monitor_dpi, face_distance_cm),
        monitor_dpi=monitor_dpi,
        face_distance_cm=face_distance_cm,
        calibration_hash=cal_hash,
        timestamp_iso=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )


def run_evaluation(
    camera_index: int = 0,
    monitor_dpi: float = 96.0,
    face_distance_cm: float = 50.0,
    seed: int | None = None,
    n_cols: int = DEFAULT_N_COLS,
    n_rows: int = DEFAULT_N_ROWS,
    log_path: Path | None = None,
    bench: bool = False,
) -> EvalReport | None:
    cal_path = calibration_path()
    if not cal_path.exists():
        raise FileNotFoundError(
            f"No calibration at {cal_path}. Run `gaze-tracker calibrate` first."
        )
    model = GazeModel.load(cal_path)
    cal_hash = calibration_hash(cal_path)
    use_seed = seed if seed is not None else seed_from_hash(cal_hash)
    points_norm = make_eval_points(
        use_seed,
        n_cols=n_cols,
        n_rows=n_rows,
        calibration_points=list(GRID_POINTS_NORM),
    )

    screen_w, screen_h = screen_size()
    if screen_w != model.screen_w or screen_h != model.screen_h:
        # Calibration was taken on a different-size screen. Predictions are in
        # the calibration's coordinate space, so target points scaled to today's
        # screen will mis-compare. Warn but continue — the user may know what
        # they're doing (e.g., evaluating on a similar monitor).
        print(
            f"[gaze-tracker] warning: current screen {screen_w}x{screen_h} "
            f"differs from calibration {model.screen_w}x{model.screen_h}. "
            "Eval errors will conflate gaze error with screen-coord rescale."
        )

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open camera index {camera_index}")

    window = "gaze-tracker: eval"
    cv2.namedWindow(window, cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty(window, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    canvas = np.zeros((screen_h, screen_w, 3), dtype=np.uint8)

    samples_with_targets: list[tuple[tuple[float, float], tuple[float, ...]]] = []
    total_ear_dropped = 0
    total_pose_dropped = 0
    n_points = len(points_norm)
    aborted_flag = False
    if model.head_pose_baseline is None:
        # Pre-#5 calibration; gate is inert. Surface this so the user knows
        # eval pose-noise isn't being filtered.
        print(
            "[gaze-tracker] note: calibration has no head-pose baseline; "
            "head-pose gating disabled for this eval (recalibrate to enable)."
        )

    try:
        with FaceMeshTracker() as tracker:
            for idx, (nx, ny) in enumerate(points_norm):
                target_px = (nx * screen_w, ny * screen_h)
                feats, ear_dropped, pose_dropped, aborted = _capture_eval_point(
                    canvas, window, cap, tracker,
                    target_px=target_px,
                    target_norm_y=ny,
                    progress_text=f"eval point {idx + 1}/{n_points}",
                    screen_h=screen_h,
                    head_pose_baseline=model.head_pose_baseline,
                )
                total_ear_dropped += ear_dropped
                total_pose_dropped += pose_dropped
                if aborted:
                    aborted_flag = True
                    break
                if not feats:
                    # No usable frames for this point (face off-camera, all blinked,
                    # or pose-gated). Skip rather than skew the median with a
                    # synthetic value.
                    continue
                arr = np.asarray(feats, dtype=float)
                med_feat = tuple(np.median(arr, axis=0).tolist())
                samples_with_targets.append((target_px, med_feat))
    finally:
        cap.release()
        cv2.destroyWindow(window)

    if aborted_flag:
        print("eval aborted")
        return None
    if not samples_with_targets:
        print("eval: no usable measurements (face never tracked?)")
        return None

    if bench:
        # Multi-basis A/B on the same captured fixations. Refit each basis on
        # the saved calibration's anchor set so only the model architecture
        # varies. Don't write to eval_log.csv — bench results are exploratory.
        anchor_mask = model.is_anchor
        anchor_feats = model.features[anchor_mask]
        anchor_targets = model.targets[anchor_mask]
        print(
            f"\nbench on {len(samples_with_targets)} captured eval samples "
            f"(dropped: ear={total_ear_dropped}, pose={total_pose_dropped})  "
            f"cal={cal_hash}  dpi={monitor_dpi:.0f}  "
            f"distance={face_distance_cm:.0f}cm\n"
        )
        print(f"{'basis':22s}  {'median':>16s}  {'p95':>16s}  {'rmse':>16s}")
        print(f"{'-' * 22}  {'-' * 16}  {'-' * 16}  {'-' * 16}")
        for basis_name in ALL_BASES:
            test_model = GazeModel.fit(
                anchor_feats, anchor_targets,
                screen_w=model.screen_w, screen_h=model.screen_h,
                basis=basis_name,
            )
            errs = [
                float(np.linalg.norm(np.array(test_model.predict(mf)) - np.array(t)))
                for t, mf in samples_with_targets
            ]
            arr = np.asarray(errs, dtype=float)
            med = float(np.median(arr))
            p95 = float(np.percentile(arr, 95))
            rmse = float(np.sqrt(np.mean(arr**2)))
            d = lambda x: pixel_error_to_degrees(  # noqa: E731 - tight scope
                x, monitor_dpi, face_distance_cm
            )
            print(
                f"{basis_name:22s}  "
                f"{med:>4.0f} px ({d(med):>5.2f} deg)  "
                f"{p95:>4.0f} px ({d(p95):>5.2f} deg)  "
                f"{rmse:>4.0f} px ({d(rmse):>5.2f} deg)"
            )
        return None

    # Single-model: score against the saved model's basis.
    errs = [
        float(np.linalg.norm(np.array(model.predict(mf)) - np.array(t)))
        for t, mf in samples_with_targets
    ]
    report = _summarize(
        errs,
        total_ear_dropped,
        total_pose_dropped,
        monitor_dpi,
        face_distance_cm,
        cal_hash,
    )
    append_eval_log(report, path=log_path)
    print(report.stdout())
    return report
