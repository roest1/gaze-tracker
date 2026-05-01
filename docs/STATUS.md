# gaze-tracker — status & roadmap

Snapshot of the project as of the recent realtime-pipeline arc (items #1–#9 plus
the polar A/B). The README covers user-facing usage; this doc covers what's
shipped, where the accuracy ceiling sits, what we're stuck on, and what's next.

## Pipeline (what's shipped)

### Feature extraction (`landmarks.py`)
- MediaPipe Face Landmarker (Tasks API, `face_landmarker.task`).
- Per eye: 3D offset from eye-corner centroid to iris centroid in MediaPipe's
  normalized image coords. Both eyes averaged, then unit-normalized → 3-tuple
  `(vx, vy, vz)` used as the regression feature.
- Per frame also extracts: per-eye EAR (Soukupová & Čech) and 4×4 facial
  transformation matrix → YXZ Euler angles `(yaw, pitch, roll)` in degrees.

### Mapping (`mapping.py`)
- Weighted least-squares fit, four supported regression bases:
  - `cartesian` — `[1, vx, vy, vz]` (4 coefs/axis, default)
  - `polar` — `[1, yaw, pitch]` (3 coefs/axis; `yaw=atan2(vx,vz)`, `pitch=arcsin(vy)`)
  - `polynomial-polar` — `[1, yaw, pitch, yaw², pitch², yaw·pitch]` (6 coefs/axis)
  - (`cartesian-no-z` is a candidate 4th basis — see roadmap.)
- Sample taxonomy: anchors (calibration grid, permanent, weight as fit) vs.
  refinements (click-to-refine, capped at 30 in FIFO ring, time-decayed at
  refit with 30-min half-life). Anchors carry NaN `added_at`; refinements
  carry wall-clock `time.time()` so age survives save+reload.
- JSON schema with backward-compat: pre-anchor / pre-basis calibrations load
  with everything promoted to anchors, basis defaults to cartesian.

### Calibration (`calibration.py`)
- 9-point fixation grid at {0.1, 0.5, 0.9} on each axis. Order: center → 4
  corners → 4 edge midpoints. ~25s, ~300 samples.
- Per-point lock-on (red dot) → capture (green dot). EAR baseline taken from
  lock-on; capture frames whose either-eye EAR drifts >0.08 from baseline are
  dropped (kills blink/squint frames).
- Captures head pose per admitted frame; per-axis median across all frames is
  stored as `model.head_pose_baseline` for the realtime gate.
- LOOCV warning: at end of fit, leave-one-out residual per grid point. Any
  point whose held-out residual is >2× the median across the 9 is flagged in
  the terminal ("calibration warning: outlier point(s) — TR (...) ...") and
  shown as an orange line on the completion screen.

### Realtime tracking (`stream.py`)
- One Euro filter on output (`min_cutoff`, `beta`).
- 5-frame median smoother on the 3D feature before regression (rejects
  single-frame MediaPipe outliers).
- Saccade detector (EMA-smoothed velocity) suppresses target-snap during
  fast eye motion.
- EAR gate: rolling 5s baseline; on out-of-band frame, freezes cursor at
  last admitted prediction and skips One Euro / saccade / refinement-buffer
  updates.
- Head-pose gate: when `model.head_pose_baseline` is set and the current
  frame's pose deviates >15° on any axis, freezes cursor and shows
  `re-center head` banner.
- Target snap with hysteresis (`attractor_radius` / `lock_radius` /
  `unlock_radius`). Snap suppressed during saccades.
- Click-to-refine:
  - 60-frame `(t, feature, velocity)` deque maintained per admitted frame.
  - On left-click: take median of features in `[t_click - 200ms, t_click - 50ms]`.
    Reject with `click ignored: pre-click data thin` if <3 samples in window;
    reject with `click ignored: mid-saccade` if any in-window velocity exceeded
    saccade threshold.
  - Otherwise: add as a refinement (auto-evicts oldest if count > 30).
  - Right-click or `z`: undo last session refinement. Anchor-protected (refuses
    to pop calibration samples even if session counter overcounts after eviction).
- Click feedback: green error line from predicted crosshair to click, label
  `<err>px w=<weight>`, fades after 1.2s. Undo / rejection use a unified
  yellow banner with the same fade timer.

### Held-out evaluation (`evaluate.py`)
- New `gaze-tracker eval` subcommand. 16-point stratified-jittered grid (4×4
  cells at `[0.20, 0.40, 0.60, 0.80]`, ±0.04 jitter; ≥0.05 from any cal point
  by construction). Seed defaults to first 8 hex chars of the calibration's
  sha256, so the same calibration always sees the same eval points.
- EAR + head-pose gating during capture (matches the realtime loop). Per point:
  median of admitted features → single `model.predict` call → pixel error vs.
  the target.
- Reports median, p95, RMSE in pixels and degrees of visual angle. Degrees
  computed from `--monitor-dpi` (default 96 with warning) and `--face-distance-cm`
  (default 50).
- Persistent log: appends one row per run to `~/.config/gaze-tracker/eval_log.csv`.
  On schema mismatch (e.g. column added in #10), the existing file is rotated
  to `.bak` and a fresh one is created.
- `--bench`: captures fixations once, then refits each of `cartesian`, `polar`,
  `polynomial-polar` on the saved calibration's anchor samples and scores the
  same captured fixations against each. Side-by-side stdout table; bench
  results don't write to the log.

## Measured accuracy

Best stable cartesian-basis baseline (recalibrated, head-pose-gated eval, single run):

| metric | px (at 141 DPI, 50 cm) | degrees |
|---|---|---|
| median | 190 | 3.91° |
| p95 | 349 | 7.18° |
| rmse | 222 | 4.56° |

**Eval noise floor**: ~25% variance in median across runs of the same
calibration. For high-stakes A/Bs, run eval 2–3× and use a within-run
comparison (which is exactly what `--bench` does).

### Polar A/B result (cal `4ffa9859…`, single eval, 16 samples)

| basis | median | p95 | rmse |
|---|---|---|---|
| **cartesian** | **4.40°** | **8.66°** | **5.18°** |
| polar | 6.73° (+53%) | 12.66° (+46%) | 7.98° (+54%) |
| polynomial-polar | 6.09° (+38%) | 10.68° (+23%) | 7.18° (+39%) |

**Polar bases lose by 40–50%.** The `(vx, vy, vz)` feature is *not* a true
gaze direction in world space — it's the iris centroid's displacement from
the eye corner centroid in MediaPipe's normalized image coords, then
unit-normalized as a convenience. The polar transformation
`yaw=atan2(vx,vz), pitch=arcsin(vy)` assumes direction-vector semantics that
this feature doesn't have, and amplifies vz noise. Polynomial-polar's extra
parameters partially undo the bad transform but still lose to cartesian's
4 parameters.

**Implication:** sub-4° accuracy needs a different *feature*, not a different
basis. The cartesian basis is correct for this feature.

## Known limitations / what we're stuck on

1. **~4° median ceiling for the current feature.** Industry baseline for
   landmark-based webcam trackers is 2–3°; we're at the high end. ETH-XGaze
   on webcam: ~1–2°. IR hardware: <0.5°.
2. **Feature semantics mismatch.** Iris-displacement-in-eye-socket is not a
   gaze direction. Hand-rolling a true gaze direction needs an eye-sphere
   model + per-user IPD/corneal-curvature calibration — non-trivial. The
   pragmatic path is the CNN swap (ETH-XGaze).
3. **vz is the noisy channel.** MediaPipe's z is inferred from face scale
   and is the noisiest input dimension. Currently part of the cartesian fit
   with non-trivial coefficient (~600 in the latest cal), so it's not pure
   noise — but might be worth dropping. See `cartesian-no-z` experiment.
4. **Calibration is posture-bound.** Each calibration captures one (yaw,
   pitch, roll). Predictions degrade smoothly as posture drifts within the
   15° gate threshold; the gate only catches the worst cases.
5. **Eval run-to-run variance.** ~25% in median. Single-run comparisons are
   unreliable for small effect sizes.
6. **DPI / multi-monitor.** No portable Wayland DPI query. tkinter screen
   size returns the primary display; on multi-monitor setups the cv2
   fullscreen target may be a different display. Workaround: `--monitor-dpi`
   flag, eyeball the right monitor.
7. **Cross-session refinement bookkeeping.** Refinements persist in JSON but
   load as anchors next session (intentional). Cross-session undo disabled
   for safety. JSON grows unbounded across sessions if unevicted refinements
   accumulate; not yet a problem in practice (cap is 30 per session).
8. **OpenCV / Qt warnings on Wayland.** "Cannot find font directory" log spam
   on first window creation. Cosmetic; doesn't affect functionality.

## Roadmap

### Short-term (hours of work each)
- **`cartesian-no-z` basis** as a 4th option in `--bench`. Tests whether vz
  is net signal or net noise. Decides whether to drop it from the default fit.
- **Multi-eval averaging in `--bench`**: run capture 3× back-to-back, report
  median per basis. Reduces eval noise for high-confidence A/Bs.
- **`screeninfo` for DPI**: try the lib for a portable per-monitor DPI query;
  fall back to `--monitor-dpi` if unavailable.
- **Dwell-detection primitive** on top of `snap.py`'s lock state. Emit a
  log event when a target's lock duration crosses a threshold (e.g., 500ms).
  Unblocks the gaze-as-LLM-input prototype.

### Medium-term (the planned arc, multi-session)
- **ETH-XGaze CNN swap** (`xucong-zhang/ETH-XGaze`). Pre-trained ResNet
  taking head-pose-normalized eye crops, outputs gaze direction in
  normalized space. Replaces the iris-displacement feature with a real
  gaze direction. Constraints:
  - CPU-only inference on Riley's machine. ResNet-50 is ~30–60 ms/frame on
    CPU — tight for 30 FPS. Likely need ResNet-18 variant or every-other-frame
    inference.
  - Hard part is the head-pose normalization pipeline: the model expects a
    specific normalized eye/face crop derived from a 6-DoF head pose. Either
    reuse MediaPipe's facial transformation matrix (already extracted for the
    head-pose gate) or port ETH-XGaze's normalization code.
  - Schema bump: feature `(vx, vy, vz)` → `(yaw, pitch)` in head-normalized
    space. New calibration JSON schema.
  - Keep the geometric pipeline as fallback for when the CNN is unavailable.
  - Fallback option if ETH-XGaze isn't enough: FAZE (`NVlabs/few_shot_gaze`).
- **Active recalibration prompt.** Track click-error magnitude over time.
  When the rolling median exceeds a threshold, surface a "consider
  recalibrating" banner. Could schedule a one-time agent to ramp this in.

### Long-term (the actual goal)
- **Gaze as LLM input.** The reason this project exists. Building blocks:
  1. Per-element classification — *have* (target-snap with hysteresis).
  2. Dwell detection — short-term roadmap above.
  3. State classification — "reading" (high-frequency micro-saccades within
     small region) vs. "stuck" (long fixation in one place) vs. "scanning"
     (long-distance saccades). Heuristic at first, learnable later.
  4. LLM client integration — context injection (active window scrape +
     gaze region), prompting, latency. Probably a separate repo.

## Numerical anchors

For px ↔ degrees conversions:

| display | distance | 1° in px |
|---|---|---|
| 138 DPI laptop (1920×1080) | 50 cm | ~46 |
| 141 DPI laptop (1920×1200) | 50 cm | ~47 |
| Generic 96 DPI | 50 cm | ~33 |

Industry accuracy reference points:
- IR hardware (EyeLink, Tobii Pro): <0.5°
- Best webcam research (ETH-XGaze, FAZE, person-specific fine-tune): 1–2°
- Typical landmark-based webcam: 2–3°
- Current state: ~4°
- Dwell-as-input usability bar (per memory): ~2° accuracy + 25 px RMS jitter
