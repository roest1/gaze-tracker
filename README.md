# gaze-tracker

Laptop webcam gaze-to-screen tracker. MediaPipe Face Mesh extracts iris landmarks, a 9-point fixation calibration fits a 3D-gaze → screen mapping, and a One Euro filter smooths the signal. Click-to-refine online learning lets the model improve in place as you use it.

## Setup

Tested on Fedora 43. Requires a webcam and a display session (X11 or Wayland-with-XWayland).

```bash
# 1. Install uv (one time; ~20 MB)
sudo dnf install uv                 # Fedora
# brew install uv                   # macOS
# curl -LsSf https://astral.sh/uv/install.sh | sh   # any Linux

# 2. Clone and sync
git clone <this-repo> gaze-tracker && cd gaze-tracker
uv sync
```

`uv sync` reads `.python-version` (pinned to 3.12 — mediapipe does not yet support 3.13), auto-downloads that interpreter into `~/.local/share/uv/python/`, creates `.venv/`, and installs runtime + dev dependencies. No system Python is touched.

Verify:

```bash
uv run pytest -q
```

Make sure your user has webcam access. On Fedora this means being in the `video` group (`groups $USER` — desktop users usually already are).

On the first run of `calibrate` or `track`, MediaPipe's Face Landmarker model (~3 MB) is downloaded to `$XDG_CACHE_HOME/gaze-tracker/face_landmarker.task` (usually `~/.cache/gaze-tracker/`). Delete the file to force a re-download.

## Usage

```bash
uv run gaze-tracker calibrate                       # ~25 sec: 9-point fixation grid
uv run gaze-tracker track                           # live gaze with smoothing; fullscreen over webcam
uv run gaze-tracker eval --monitor-dpi <N>          # held-out 16-point eval (px + degrees)
uv run gaze-tracker eval --monitor-dpi <N> --bench  # A/B all regression bases on one capture
```

Press `ESC` to abort any window. The calibration is written to `$XDG_CONFIG_HOME/gaze-tracker/calibration.json` (usually `~/.config/gaze-tracker/`). It contains the fit coefficients, the raw (feature, target) samples with anchor flags, the head-pose baseline, and the basis used. Click-to-refine appends to it; eval runs append a metrics row to a sibling `eval_log.csv`.

`calibrate` walks a 9-point fixation grid: center first, then the four corners (top-left, top-right, bottom-left, bottom-right), then the four edge midpoints (top, bottom, left, right). The dots sit at 10 / 50 / 90 % of the screen — edge-hugging coverage so the regression extrapolates well into the corner zones where target-snap usually lives. At each point a red dot appears for 0.8 s ("Lock your eyes on the dot."), then turns green for 1.2 s while the tracker captures ~30–40 samples. Total ~25 s, ~300 samples, single phase, head held still.

`--basis {cartesian,polar,polynomial-polar}` picks the regression basis. Default is `cartesian` (`[1, vx, vy, vz]`, 4 coefs/axis). Polar and polynomial-polar variants are available but currently lose to cartesian by 40–50% on this feature — see `docs/STATUS.md` for the bench result and why. Use `gaze-tracker eval --bench` to A/B them against your own calibration.

**EAR gating.** During each red-dot lock-on the tracker measures eye-aspect-ratio per eye (vertical lid distance ÷ horizontal corner distance) and takes the median as that point's baseline. During the green-dot capture, any frame whose either-eye EAR drifts more than 0.08 from baseline is dropped — that excludes blinks, half-blinks, and asymmetric squints from the regression data. Without the gate, those frames embed lid-position-dependent iris-centroid noise straight into the linear fit. The terminal log reports how many frames the gate rejected so you can see whether your eyes were behaving.

**Head-pose baseline.** The calibration also records the per-axis median of `(yaw, pitch, roll)` (in degrees) across all admitted frames, derived from MediaPipe's facial transformation matrix. This baseline is consulted during tracking and eval to refuse predictions when the user's current head pose has drifted >15° on any axis from the calibrated posture (linear-model accuracy collapses fast off-axis).

**LOOCV warning.** After fitting, the tracker runs leave-one-point-out cross-validation across the 9 grid points. Any single point whose held-out residual is >2× the median across the others is flagged in the terminal — that's a sign you blinked or glanced away during that point's capture, and a recalibrate is recommended.

After the grid completes, the screen shows median / 95th / RMSE residuals so you know how good the fit is before you start tracking. Recalibrate whenever the model drifts — most commonly after moving the laptop or shifting seating posture significantly.

`track` opens a fullscreen view of the webcam and draws the smoothed gaze crosshair at the predicted screen coordinate. The physical monitor location of the crosshair matches where the calibrated model thinks you are looking.

The cursor freezes silently when you blink (rolling-baseline EAR gate) and freezes with a `re-center head` banner when your head deviates >15° from the calibration posture (head-pose gate). Both are by design — the linear model can't honestly predict in either condition.

**Click to refine.** Left-click where you're actually looking and the tracker takes the per-axis median of gaze features captured 50–200 ms BEFORE the click (skipping the click-saccade itself), pairs that with the click location as ground truth, and refits. A green line flashes from the predicted crosshair to the click so you can see the error you just corrected, and the HUD shows a running count of refinements. The model genuinely improves over time — no need to re-run the full 9-point flow every time head posture drifts.

If the pre-click window is too thin (< 3 frames) or contained a saccade, the click is rejected with a `click ignored: pre-click data thin` or `click ignored: mid-saccade` banner — that protects the fit from being poisoned by mid-saccade features.

**Right-click or `z` to undo.** Pops the most recent session refinement. Anchor-protected: it can't pop original calibration samples. Cross-session undo is intentionally disabled (refinements that survived a save+reload are reclassified as anchors).

The refit is **weighted by the click's prediction error**: a click 500 px from the predicted crosshair has more authority than one that's only 30 px off. Refinements are also **time-decayed** with a 30-min half-life, so a click from 30 minutes ago counts at ~37% authority and one from 2 hours ago at ~2%. The session refinement set is **capped at 30** in a FIFO ring — past that, the oldest auto-evicts.

Tuning flags for `track`:

- `--min-cutoff` (default `0.5`) — lower = smoother when you hold a fixation. Drop toward `0.1` if the crosshair still jitters; raise toward `1.0` if it feels laggy.
- `--beta` (default `0.05`) — higher = more responsive when the eyes move fast. Pair low min-cutoff with higher beta if fixations feel sticky but saccades lag.
- `--saccade-px-per-s` (default `2500`) — velocity threshold above which target-snap is bypassed and click-to-refine rejects.
- `--click-weight-{scale,min,max}` (defaults `20`, `5`, `80`) — lower the scale or raise the min to make clicks bend the fit harder; the max caps how much one freak click can rotate things.
- `--targets-demo` — render a 2×3 grid of demo target boxes and snap gaze to the nearest one (useful for spot-checking dwell behavior).
- `--camera <N>` — override the default webcam index.

## Eval

`gaze-tracker eval --monitor-dpi <N>` runs a 16-point held-out evaluation against the saved calibration and reports median, p95, and RMSE in both pixels and degrees of visual angle. The point layout is a 4×4 stratified-jittered grid at `[0.20, 0.40, 0.60, 0.80]` on each axis (≥0.05 from any calibration grid point by construction). The seed defaults to a hash of the calibration file, so the same calibration always sees the same eval points — apples-to-apples comparisons across model edits.

EAR + head-pose gating are applied during eval capture (matching the realtime loop), so blink frames and pose-drift frames don't contaminate the measurement. Drop counts surface in the output: `dropped(ear=N, pose=M)`.

Each run appends a row to `~/.config/gaze-tracker/eval_log.csv` for trend tracking. If the column schema changes (it has, twice), the existing log is rotated to `.bak` and a fresh one is created.

`--bench` runs one capture, then refits each of the three regression bases (`cartesian`, `polar`, `polynomial-polar`) on the saved calibration's anchor samples and scores them all against the same captured fixations. Side-by-side stdout table; the saved model and eval log are unchanged.

## Approach

- **Landmarks** — MediaPipe Face Mesh (Tasks API) with iris refinement. For each eye, compute the 3D offset from the eye center (mean of four corner landmarks) to the iris center, using MediaPipe's z-bearing landmarks. Averaged across eyes and normalized, this vector is the *iris-displacement-in-eye-socket signal* used as the regression feature. (It's not strictly a gaze direction in world space — see `docs/STATUS.md` for why that distinction matters.)
- **Calibration** — single 9-point fixation grid at 10 / 50 / 90 % of the screen. ~30–40 samples per point, EAR-gated, fitted to a 4-coefficient linear regression per screen axis. Also captures a head-pose baseline used by the realtime gate.
- **Input smoothing** — 5-frame sliding median on the raw 3D gaze feature before regression. Rejects single-frame outliers (MediaPipe glitches, z-axis noise) without adding perceptible lag.
- **Output smoothing** — One Euro Filter (Casiez et al., 2012): adaptive low-pass that attenuates jitter at rest and tracks faithfully during saccades.
- **Online refinement** — left-click in the tracking window takes a pre-click median feature, pairs it with the click coordinate as ground truth, appends it to the calibration dataset (with time decay + a refinement cap), and refits in place. Right-click / `z` undoes the most recent session refinement.

## Status

Best stable single-run eval (cartesian basis, 141 DPI / 50 cm): **median 3.91°, p95 7.18°, rmse 4.56°**. That's at the high end of landmark-based webcam tracking; sub-1° needs IR hardware and ~1–2° needs a CNN feature extractor (planned: ETH-XGaze).

The polar / polynomial-polar bases are 40–50% worse than cartesian — they assume the gaze feature is a true direction vector, which it isn't.

For the full picture (architecture, accuracy, known limitations, roadmap), see [`docs/STATUS.md`](docs/STATUS.md).
