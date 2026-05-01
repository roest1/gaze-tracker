"""One Euro Filter (Casiez et al., 2012).

Adaptive low-pass: heavy smoothing when the signal is stationary, light smoothing
when it moves. Good default for pointer-like signals where EMA would lag.
"""
from __future__ import annotations

import math
from collections.abc import Sequence


class _LowPass:
    def __init__(self) -> None:
        self._y: float | None = None

    def step(self, x: float, alpha: float) -> float:
        if self._y is None:
            self._y = x
        else:
            self._y = alpha * x + (1.0 - alpha) * self._y
        return self._y


def _alpha(dt: float, cutoff: float) -> float:
    tau = 1.0 / (2.0 * math.pi * cutoff)
    return 1.0 / (1.0 + tau / dt)


class OneEuroFilter:
    def __init__(
        self,
        min_cutoff: float = 1.0,
        beta: float = 0.05,
        d_cutoff: float = 1.0,
    ) -> None:
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self._x = _LowPass()
        self._dx = _LowPass()
        self._x_prev: float | None = None
        self._t_prev: float | None = None

    def __call__(self, t: float, x: float) -> float:
        if self._t_prev is None or t <= self._t_prev:
            self._t_prev = t
            self._x_prev = x
            return self._x.step(x, 1.0)

        dt = t - self._t_prev
        dx = (x - self._x_prev) / dt
        dx_hat = self._dx.step(dx, _alpha(dt, self.d_cutoff))

        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        x_hat = self._x.step(x, _alpha(dt, cutoff))

        self._t_prev = t
        self._x_prev = x
        return x_hat


class OneEuroFilter2D:
    def __init__(
        self,
        min_cutoff: float = 1.0,
        beta: float = 0.05,
        d_cutoff: float = 1.0,
    ) -> None:
        self._fx = OneEuroFilter(min_cutoff, beta, d_cutoff)
        self._fy = OneEuroFilter(min_cutoff, beta, d_cutoff)

    def __call__(self, t: float, xy: tuple[float, float]) -> tuple[float, float]:
        return (self._fx(t, xy[0]), self._fy(t, xy[1]))


class MedianSmoother:
    """Sliding-window per-axis median filter over a fixed-length history.

    Used on the raw 3D gaze feature before regression to reject single-frame
    outliers (MediaPipe landmark jitter, especially on the z-axis) without
    introducing the lag an EMA/One-Euro would add at comparable smoothing.
    """

    def __init__(self, window: int = 5, dim: int = 3) -> None:
        from collections import deque

        self._buf: deque[tuple[float, ...]] = deque(maxlen=window)
        self._dim = dim

    def __call__(self, x: tuple[float, ...]) -> tuple[float, ...]:
        if len(x) != self._dim:
            raise ValueError(f"expected {self._dim}-tuple, got {len(x)}")
        self._buf.append(tuple(float(v) for v in x))
        # All buffered tuples have the same length (validated above), so strict=True
        # is safe and is the intent: per-axis collation, not silent truncation.
        cols = list(zip(*self._buf, strict=True))
        return tuple(_median(list(c)) for c in cols)


def _median(xs: list[float]) -> float:
    ys = sorted(xs)
    n = len(ys)
    mid = n // 2
    return ys[mid] if n % 2 else 0.5 * (ys[mid - 1] + ys[mid])


class SaccadeDetector:
    """Velocity-threshold saccade detector on a 2D pointer signal.

    Returns True when the smoothed point velocity exceeds `threshold_px_per_s`
    (eye is in a saccade), False during fixation / smooth pursuit. Used to
    gate target-snap lock acquisition: locks shouldn't fire while the eye is
    flying past a target on the way to another.

    Anchor for picking the threshold: at 50cm viewing distance on a 1080p
    laptop, 1° of visual angle ≈ 42 px. Saccades are typically >100°/s
    (~4200 px/s); fixation drift sits well under 30°/s (~1300 px/s). The
    default 2500 px/s splits the gap with margin.

    Velocity is EMA-smoothed so that a single-frame landmark spike doesn't
    flip the detector into saccade mode for one frame.
    """

    def __init__(
        self,
        threshold_px_per_s: float = 2500.0,
        velocity_smooth: float = 0.35,
    ) -> None:
        if not 0.0 < velocity_smooth <= 1.0:
            raise ValueError("velocity_smooth must be in (0, 1]")
        self.threshold = float(threshold_px_per_s)
        self.alpha = float(velocity_smooth)
        self._prev_xy: tuple[float, float] | None = None
        self._prev_t: float | None = None
        self._smoothed_v: float = 0.0

    def reset(self) -> None:
        self._prev_xy = None
        self._prev_t = None
        self._smoothed_v = 0.0

    @property
    def velocity(self) -> float:
        """Most recent EMA-smoothed velocity in px/s. 0 before first sample."""
        return self._smoothed_v

    def __call__(self, t: float, xy: tuple[float, float]) -> bool:
        if self._prev_xy is None or self._prev_t is None or t <= self._prev_t:
            self._prev_xy = xy
            self._prev_t = t
            return False
        dt = t - self._prev_t
        dx = xy[0] - self._prev_xy[0]
        dy = xy[1] - self._prev_xy[1]
        v = math.sqrt(dx * dx + dy * dy) / dt if dt > 0 else 0.0
        self._smoothed_v = self.alpha * v + (1.0 - self.alpha) * self._smoothed_v
        self._prev_xy = xy
        self._prev_t = t
        return self._smoothed_v > self.threshold


def features_in_window(
    buffer: Sequence[tuple[float, tuple[float, ...], float]],
    t_click: float,
    window_start_s: float,
    window_end_s: float,
) -> list[tuple[tuple[float, ...], float]]:
    """Return (feature, velocity) tuples whose timestamp lies in the
    pre-click window [t_click - window_start_s, t_click - window_end_s].

    Why this primitive exists: the most-recent feature at click time was
    captured DURING the eye's saccade toward the click target — exactly the
    noisy sub-fixation state that should not be used to refit. The pre-click
    window catches the fixation BEFORE the user decided to click, which is
    when the gaze feature most faithfully matches the click coordinate.

    Caller is responsible for diagnosis (sample count, velocity check) and
    median computation — surfaces specific rejection reasons in the UI.
    """
    t_lo = t_click - window_start_s
    t_hi = t_click - window_end_s
    return [(f, v) for t, f, v in buffer if t_lo <= t <= t_hi]


class EARGate:
    """Rolling-baseline EAR gate for the realtime tracking loop.

    Maintains a sliding window of per-eye EAR samples. Once `min_n` baseline
    frames have been seen, rejects frames whose either-eye EAR drifts more
    than `tolerance` from the per-eye rolling median. Catches blinks,
    half-blinks, and squints in the live loop without a per-session baseline
    (calibration-time baselining doesn't survive fatigue / lighting drift).

    Per-eye check (not averaged) catches asymmetric squints — the dominant
    failure mode where one lid drops and corrupts that eye's iris centroid
    while the other looks fine.
    """

    def __init__(
        self,
        window_frames: int = 150,
        min_n: int = 5,
        tolerance: float = 0.08,
    ) -> None:
        if window_frames < 1:
            raise ValueError("window_frames must be >= 1")
        if min_n < 1 or min_n > window_frames:
            raise ValueError("min_n must be in [1, window_frames]")
        if tolerance < 0:
            raise ValueError("tolerance must be >= 0")
        from collections import deque

        self._buf: deque[tuple[float, float]] = deque(maxlen=window_frames)
        self._min_n = min_n
        self._tolerance = tolerance

    @property
    def ready(self) -> bool:
        """True iff baseline has stabilized and the gate is now actively gating."""
        return len(self._buf) >= self._min_n

    def reset(self) -> None:
        self._buf.clear()

    def __call__(self, ear_l: float, ear_r: float) -> bool:
        """Push the new sample, return True iff this frame should be kept.

        Returns True unconditionally until `min_n` samples accumulated — no
        useful baseline before then. Above min_n, returns True iff both eyes'
        EAR is within `tolerance` of their per-eye rolling median.
        """
        self._buf.append((ear_l, ear_r))
        if len(self._buf) < self._min_n:
            return True
        med_l = _median([e[0] for e in self._buf])
        med_r = _median([e[1] for e in self._buf])
        return (
            abs(ear_l - med_l) <= self._tolerance
            and abs(ear_r - med_r) <= self._tolerance
        )
