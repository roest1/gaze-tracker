"""Realtime tracking loop with click-to-refine online learning.

The display trick: rather than building a transparent fullscreen overlay (which
is fussy on Wayland), we show the webcam feed fullscreen and draw the gaze
crosshair at screen-pixel coordinates on top of it. The crosshair then appears
at the same physical monitor location it would in a real overlay, and the
webcam underneath is just visual context.

Click to refine: a left-click inside the window pairs a pre-click gaze feature
with the click coordinate (the user's ground-truth gaze point at that moment),
appends it to the calibration dataset, refits the linear model, and saves the
updated calibration to disk. The "pre-click feature" is the per-axis median of
features captured BEFORE the user's click-saccade — see `features_in_window`
in filter.py for why most-recent-frame is the wrong choice here.
"""
from __future__ import annotations

import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np

from .calibration import EAR_TOLERANCE
from .filter import (
    EARGate,
    MedianSmoother,
    OneEuroFilter2D,
    SaccadeDetector,
    features_in_window,
)
from .landmarks import HEAD_POSE_GATE_DEG, FaceMeshTracker, head_pose_max_dev_deg
from .mapping import GazeModel, calibration_path, error_weight
from .snap import Target, TargetSnap

_FEEDBACK_SECONDS = 1.2
_FEATURE_WINDOW = 5
# ~5s at 30fps. Long enough that a blink (~9 frames) is a robust-median outlier.
_EAR_GATE_WINDOW_FRAMES = 150
# Pre-click window: look back 200ms..50ms before the click. The 50ms cutoff
# excludes the click-saccade itself (eye flying toward the click target);
# the 200ms start covers the prior fixation. >=3 in-window samples needed
# for a robust median over MediaPipe landmark jitter.
_PRE_CLICK_WINDOW_START_S = 0.200
_PRE_CLICK_WINDOW_END_S = 0.050
_PRE_CLICK_MIN_SAMPLES = 3
# 1s @ 60fps / 2s @ 30fps — comfortably exceeds the 200ms window we read.
_PRE_CLICK_BUFFER_MAXLEN = 60


def run_tracking(
    camera_index: int = 0,
    min_cutoff: float = 0.5,
    beta: float = 0.05,
    feature_window: int = _FEATURE_WINDOW,
    targets: list[Target] | None = None,
    snap_attractor_px: float = 140.0,
    snap_lock_px: float = 55.0,
    snap_unlock_px: float = 95.0,
    saccade_px_per_s: float | None = 2500.0,
    click_weight_scale: float = 20.0,
    click_weight_min: float = 5.0,
    click_weight_max: float = 80.0,
) -> None:
    path = calibration_path()
    if not path.exists():
        raise FileNotFoundError(
            f"No calibration at {path}. Run `gaze-tracker calibrate` first."
        )
    model = GazeModel.load(path)

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open camera index {camera_index}")

    window = "gaze-tracker"
    cv2.namedWindow(window, cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty(window, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    smoother = OneEuroFilter2D(min_cutoff=min_cutoff, beta=beta)
    feature_median = MedianSmoother(window=feature_window, dim=3)
    ear_gate = EARGate(window_frames=_EAR_GATE_WINDOW_FRAMES, tolerance=EAR_TOLERANCE)

    snap: TargetSnap | None = None
    if targets:
        snap = TargetSnap(
            attractor_radius=snap_attractor_px,
            lock_radius=snap_lock_px,
            unlock_radius=snap_unlock_px,
        )

    saccade: SaccadeDetector | None = None
    if saccade_px_per_s is not None:
        saccade = SaccadeDetector(threshold_px_per_s=saccade_px_per_s)

    state: dict = {
        # Time-stamped deque of (t_monotonic, smoothed_feature, velocity_at_t).
        # Click-to-refine reads this to take a pre-click median.
        "feature_buffer": deque(maxlen=_PRE_CLICK_BUFFER_MAXLEN),
        "last_prediction": None,    # most-recent displayed (gx, gy)
        "last_locked_target": None, # so EAR-gated frames can keep showing the lock
        "refinements": 0,           # session-only count; cross-session undo is unsafe
        "feedback": None,           # (click_xy, pred_xy, weight, monotonic_timestamp)
        # Generic banner for short status messages: undo, click rejection, etc.
        # Tuple of (monotonic_timestamp, message_text). None when nothing to show.
        "info_banner": None,
    }

    def try_undo() -> None:
        # Session-counter check: refinements only undoes within the running
        # session. After a restart, refinements=0 means undo is a no-op even
        # if disk samples exist — that's intentional, since cross-session
        # undo could pop original calibration samples.
        if state["refinements"] == 0:
            return
        try:
            model.pop_last_sample()
        except ValueError:
            # The most-recent sample is an anchor — happens when the session
            # counter has drifted past the actual model state (e.g., after
            # MAX_REFINEMENTS evictions removed older refinements without
            # decrementing the counter). Reset and surface the state to user.
            state["refinements"] = 0
            state["info_banner"] = (time.monotonic(), "no refinements to undo")
            return
        state["refinements"] -= 1
        _save_safe(model, path)
        state["info_banner"] = (time.monotonic(), "refinement undone")

    def on_mouse(event: int, x: int, y: int, flags: int, userdata: object) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            pred = state["last_prediction"]
            if pred is None:
                return
            t_click = time.monotonic()
            in_window = features_in_window(
                list(state["feature_buffer"]),
                t_click,
                _PRE_CLICK_WINDOW_START_S,
                _PRE_CLICK_WINDOW_END_S,
            )
            # Diagnose rejection inline so the banner can name the cause.
            if len(in_window) < _PRE_CLICK_MIN_SAMPLES:
                state["info_banner"] = (
                    time.monotonic(), "click ignored: pre-click data thin"
                )
                return
            if saccade_px_per_s is not None and any(
                v > saccade_px_per_s for _, v in in_window
            ):
                state["info_banner"] = (
                    time.monotonic(), "click ignored: mid-saccade"
                )
                return
            feat_arr = np.asarray([f for f, _ in in_window], dtype=float)
            feat = tuple(np.median(feat_arr, axis=0).tolist())
            err_px = float(np.hypot(x - pred[0], y - pred[1]))
            weight = error_weight(
                err_px, click_weight_scale, click_weight_min, click_weight_max
            )
            model.add_sample(feat, (float(x), float(y)), weight=weight)
            _save_safe(model, path)
            state["refinements"] += 1
            state["feedback"] = (
                (x, y),
                (float(pred[0]), float(pred[1])),
                weight,
                time.monotonic(),
            )
        elif event == cv2.EVENT_RBUTTONDOWN:
            try_undo()

    cv2.setMouseCallback(window, on_mouse)

    try:
        with FaceMeshTracker() as tracker:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                frame = cv2.flip(frame, 1)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                feat = tracker.extract(rgb)

                canvas = _fit_to_screen(frame, model.screen_w, model.screen_h)
                canvas = (canvas * 0.55).astype(np.uint8)

                locked_target: Target | None = None
                gaze_xy: tuple[float, float] | None = None
                no_face = feat is None

                if feat is not None:
                    pose_gated = (
                        model.head_pose_baseline is not None
                        and feat.head_pose is not None
                        and head_pose_max_dev_deg(
                            feat.head_pose, model.head_pose_baseline
                        ) > HEAD_POSE_GATE_DEG
                    )
                    if not ear_gate(feat.ear_left, feat.ear_right):
                        # EAR-gated (blink/squint). Don't append to the click
                        # buffer (so click-to-refine can't latch a half-blink
                        # frame), don't advance One Euro / saccade. Hold cursor
                        # at the last admitted prediction until eyes re-open.
                        gaze_xy = state["last_prediction"]
                        locked_target = state["last_locked_target"]
                    elif pose_gated:
                        # Head has rotated out of calibration posture. Linear
                        # gaze model accuracy collapses fast off-axis; refuse
                        # to predict and prompt the user to re-center. Banner
                        # is re-set every gated frame so it stays visible until
                        # they move back in-band.
                        gaze_xy = state["last_prediction"]
                        locked_target = state["last_locked_target"]
                        state["info_banner"] = (time.monotonic(), "re-center head")
                    else:
                        smoothed_feature = feature_median(feat.gaze)
                        gx_raw, gy_raw = model.predict(smoothed_feature)
                        # Monotonic, not wall-clock: NTP corrections must not feed dt
                        # to One Euro / saccade detector (would alpha-spike or false-fire).
                        now = time.monotonic()
                        gx, gy = smoother(now, (gx_raw, gy_raw))
                        in_saccade = (
                            saccade(now, (gx, gy)) if saccade is not None else False
                        )
                        velocity = saccade.velocity if saccade is not None else 0.0
                        state["feature_buffer"].append(
                            (now, smoothed_feature, velocity)
                        )
                        # Skip target-snap during saccades so the lock doesn't latch
                        # onto a target the eye is just flying past. The snap object's
                        # hysteresis releases naturally on the first post-saccade frame
                        # if the new gaze is far from the prior lock.
                        if snap is not None and targets and not in_saccade:
                            result = snap((gx, gy), targets)
                            gx, gy = result.xy
                            locked_target = result.locked
                        gaze_xy = (gx, gy)
                        state["last_prediction"] = gaze_xy
                        state["last_locked_target"] = locked_target
                else:
                    # No face. Clear the click buffer — old features here would
                    # be stale (eye position likely moved while face was gone).
                    state["feature_buffer"].clear()

                if targets:
                    _draw_targets(canvas, targets, locked=locked_target)

                if gaze_xy is not None:
                    _draw_gaze(
                        canvas, gaze_xy[0], gaze_xy[1],
                        model.screen_w, model.screen_h,
                        locked=locked_target is not None,
                    )
                elif no_face:
                    _label(canvas, "no face", (0, 0, 255))

                _render_feedback(canvas, state)
                _render_info_banner(canvas, state, model.screen_w)

                hud = (
                    f"ESC to quit  |  L-click: refine  "
                    f"|  R-click or z: undo last  "
                    f"|  refinements: {state['refinements']}"
                )
                _label(canvas, hud, (180, 180, 180), y=canvas.shape[0] - 20)

                cv2.imshow(window, canvas)
                key = cv2.waitKey(1) & 0xFF
                if key == 27:
                    break
                if key == ord("z"):
                    try_undo()
    finally:
        cap.release()
        cv2.destroyAllWindows()


def _save_safe(model: GazeModel, path: Path) -> None:
    try:
        model.save(path)
    except OSError as e:
        # Don't take down the tracker for a disk write blip; surface in HUD.
        print(f"[gaze-tracker] failed to persist refined model: {e}")


def _fit_to_screen(frame: np.ndarray, screen_w: int, screen_h: int) -> np.ndarray:
    h, w = frame.shape[:2]
    scale = min(screen_w / w, screen_h / h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.zeros((screen_h, screen_w, 3), dtype=np.uint8)
    x0 = (screen_w - new_w) // 2
    y0 = (screen_h - new_h) // 2
    canvas[y0 : y0 + new_h, x0 : x0 + new_w] = resized
    return canvas


def _draw_gaze(
    canvas: np.ndarray,
    gx: float,
    gy: float,
    screen_w: int,
    screen_h: int,
    locked: bool = False,
) -> None:
    x = int(np.clip(gx, 0, screen_w - 1))
    y = int(np.clip(gy, 0, screen_h - 1))
    inner = (0, 220, 0) if locked else (0, 0, 255)
    cv2.circle(canvas, (x, y), 42, (0, 0, 0), 3)
    cv2.circle(canvas, (x, y), 42, (255, 255, 255), 1)
    cv2.circle(canvas, (x, y), 7, (0, 0, 0), -1)
    cv2.circle(canvas, (x, y), 5, inner, -1)


def _draw_targets(
    canvas: np.ndarray, targets: list[Target], locked: Target | None
) -> None:
    for t in targets:
        x0, y0, x1, y1 = t.bbox
        is_locked = locked is t
        color = (0, 220, 0) if is_locked else (140, 140, 140)
        thickness = 4 if is_locked else 2
        cv2.rectangle(canvas, (x0, y0), (x1, y1), color, thickness)
        cv2.putText(
            canvas, t.id, (x0 + 8, y0 + 24),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1, cv2.LINE_AA,
        )


def _render_feedback(canvas: np.ndarray, state: dict) -> None:
    fb = state["feedback"]
    if fb is None:
        return
    click_xy, pred_xy, weight, ts = fb
    age = time.monotonic() - ts
    if age >= _FEEDBACK_SECONDS:
        state["feedback"] = None
        return
    cx, cy = int(click_xy[0]), int(click_xy[1])
    px, py = int(pred_xy[0]), int(pred_xy[1])
    # Error line: predicted crosshair -> true click
    cv2.line(canvas, (px, py), (cx, cy), (0, 0, 0), 4)
    cv2.line(canvas, (px, py), (cx, cy), (0, 220, 0), 2)
    # Click anchor
    cv2.circle(canvas, (cx, cy), 18, (0, 0, 0), 3)
    cv2.circle(canvas, (cx, cy), 18, (0, 220, 0), 2)
    err = int(((cx - px) ** 2 + (cy - py) ** 2) ** 0.5)
    label = f"{err}px  w={weight:.1f}"
    cv2.putText(
        canvas, label, (cx + 24, cy - 8),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA,
    )
    cv2.putText(
        canvas, label, (cx + 24, cy - 8),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 0), 1, cv2.LINE_AA,
    )


def _render_info_banner(canvas: np.ndarray, state: dict, screen_w: int) -> None:
    """Render a short status message (undo, click rejection) center-top, then
    fade after _FEEDBACK_SECONDS. Single banner channel for any non-error
    transient feedback the loop wants to surface."""
    banner = state["info_banner"]
    if banner is None:
        return
    ts, msg = banner
    age = time.monotonic() - ts
    if age >= _FEEDBACK_SECONDS:
        state["info_banner"] = None
        return
    (tw, _), _ = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
    x = (screen_w - tw) // 2
    y = 100
    cv2.putText(
        canvas, msg, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 4, cv2.LINE_AA
    )
    cv2.putText(
        canvas, msg, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 220, 220), 2, cv2.LINE_AA
    )


def _label(
    frame: np.ndarray, text: str, color: tuple[int, int, int], y: int = 30
) -> None:
    cv2.putText(
        frame, text, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA
    )
