"""Target-snap / dwell-attractor.

For dwell-as-input on a known UI, the system doesn't need accurate gaze
coordinates — it needs correct nearest-element classification. Given a list
of interactive target bboxes, this module biases the smoothed gaze prediction
toward the nearest target's center when within range:

  - Inside `lock_radius` of a target center  -> output snaps fully to center
  - Inside `attractor_radius` (but outside lock) -> linear blend toward center
  - Outside both -> passthrough

A small hysteresis (`unlock_radius` >= `lock_radius`) prevents flicker between
adjacent targets when the gaze sits near a boundary: once locked on target A,
stay locked until the raw prediction leaves A's unlock radius.
"""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class Target:
    """Axis-aligned target bbox in screen pixels. id is for downstream dispatch."""

    bbox: tuple[int, int, int, int]  # (x0, y0, x1, y1)
    id: str = ""

    @property
    def center(self) -> tuple[float, float]:
        x0, y0, x1, y1 = self.bbox
        return (0.5 * (x0 + x1), 0.5 * (y0 + y1))


@dataclass(frozen=True)
class SnapResult:
    """Output of one snap step.

    `xy` is the (possibly snapped) screen coordinate to display.
    `locked` is the Target this prediction is assigned to, or None if free.
    `weight` is the snap strength in [0, 1]: 0=passthrough, 1=fully snapped.
    """

    xy: tuple[float, float]
    locked: Target | None
    weight: float


class TargetSnap:
    """Stateful snap-to-nearest-target with hysteresis.

    Parameters in screen pixels:
      attractor_radius -- outer radius; beyond this, predictions pass through.
      lock_radius      -- inner radius; once inside, output snaps to center
                          and the target becomes "locked" for hysteresis.
      unlock_radius    -- once locked on T, stay locked until raw prediction
                          leaves this radius around T. Must be >= lock_radius.
    """

    def __init__(
        self,
        attractor_radius: float = 120.0,
        lock_radius: float = 50.0,
        unlock_radius: float = 90.0,
    ) -> None:
        if unlock_radius < lock_radius:
            raise ValueError("unlock_radius must be >= lock_radius")
        if attractor_radius < lock_radius:
            raise ValueError("attractor_radius must be >= lock_radius")
        self.attractor_radius = float(attractor_radius)
        self.lock_radius = float(lock_radius)
        self.unlock_radius = float(unlock_radius)
        self._locked: Target | None = None

    def reset(self) -> None:
        self._locked = None

    def __call__(
        self, xy: tuple[float, float], targets: list[Target]
    ) -> SnapResult:
        if not targets:
            self._locked = None
            return SnapResult(xy=xy, locked=None, weight=0.0)

        # Hysteresis: if we have a lock, hold it as long as we're within unlock.
        if self._locked is not None:
            d = _dist(xy, self._locked.center)
            if d <= self.unlock_radius:
                return SnapResult(xy=self._locked.center, locked=self._locked, weight=1.0)
            self._locked = None

        # Find nearest target by center distance.
        nearest, dist = min(
            ((t, _dist(xy, t.center)) for t in targets), key=lambda p: p[1]
        )

        if dist <= self.lock_radius:
            self._locked = nearest
            return SnapResult(xy=nearest.center, locked=nearest, weight=1.0)

        if dist <= self.attractor_radius:
            # Linear blend: weight = 1 at lock_radius, 0 at attractor_radius.
            span = self.attractor_radius - self.lock_radius
            w = 1.0 - (dist - self.lock_radius) / span if span > 0 else 0.0
            cx, cy = nearest.center
            x = (1.0 - w) * xy[0] + w * cx
            y = (1.0 - w) * xy[1] + w * cy
            return SnapResult(xy=(x, y), locked=None, weight=w)

        return SnapResult(xy=xy, locked=None, weight=0.0)


def _dist(a: tuple[float, float], b: tuple[float, float]) -> float:
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    return math.sqrt(dx * dx + dy * dy)


def grid_targets(
    screen_w: int,
    screen_h: int,
    cols: int = 3,
    rows: int = 3,
    box_w: int = 220,
    box_h: int = 130,
    margin: float = 0.12,
) -> list[Target]:
    """Build a regular grid of demo targets across the screen.

    `margin` is the fraction of screen width/height kept clear at each edge,
    so the grid sits inside the screen rather than running to the bezels.
    """
    if cols < 1 or rows < 1:
        raise ValueError("cols and rows must be >= 1")
    x_lo = int(screen_w * margin)
    x_hi = int(screen_w * (1.0 - margin))
    y_lo = int(screen_h * margin)
    y_hi = int(screen_h * (1.0 - margin))

    def _axis(lo: int, hi: int, n: int) -> list[int]:
        if n == 1:
            return [(lo + hi) // 2]
        step = (hi - lo) / (n - 1)
        return [int(lo + i * step) for i in range(n)]

    cxs = _axis(x_lo, x_hi, cols)
    cys = _axis(y_lo, y_hi, rows)
    out: list[Target] = []
    for r, cy in enumerate(cys):
        for c, cx in enumerate(cxs):
            x0 = cx - box_w // 2
            y0 = cy - box_h // 2
            out.append(
                Target(
                    bbox=(x0, y0, x0 + box_w, y0 + box_h),
                    id=f"r{r}c{c}",
                )
            )
    return out
