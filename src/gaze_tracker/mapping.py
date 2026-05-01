"""Calibration model: 3D gaze direction -> screen pixels.

Weighted linear least-squares fit with a 4-term basis [1, vx, vy, vz] per
axis. The 3D gaze vector is nearly head-pose invariant (see landmarks.py),
so a linear map suffices.

Sample taxonomy: each stored sample is either an **anchor** (the 9-point
calibration grid; permanent, weight applied as-is) or a **refinement** (a
click-to-refine sample; capped at MAX_REFINEMENTS in a FIFO ring buffer,
and time-decayed at refit so a 30-min-old click counts at ~37% authority,
a 2-hour-old click at ~2%). The decay half-life matches a typical session's
posture-stability window — it captures the empirical fact that the click
that helped half an hour ago likely doesn't reflect your current head
position.

Per-sample weights let click-to-refine give corrective clicks much more
authority than a calibration fixation; the decay then walks that authority
back as the click ages out of relevance.

Schema migration: pre-anchor calibrations have no `is_anchor` / `added_at`
fields. They load with all samples promoted to anchors (weight as saved,
no decay) — preserves prior behavior until the next click overwrites.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# Refinement bookkeeping. Anchors ignore both: they're permanent, weight 1.0
# (or whatever calibrator passed), no decay.
MAX_REFINEMENTS = 30
REFINEMENT_HALF_LIFE_S = 1800.0  # 30 min; click 30 min ago counts at e^-1 ≈ 37%

# Available regression bases. The 3D gaze vector is mapped to screen pixels
# via lstsq on a basis expansion of (vx, vy, vz). Each basis trades cost vs.
# expressiveness:
#   cartesian        -- [1, vx, vy, vz]               (4 params/axis; current default)
#   polar            -- [1, yaw, pitch]               (3 params/axis; eye motor space)
#   polynomial-polar -- [1, yaw, pitch, yaw^2,
#                        pitch^2, yaw*pitch]          (6 params/axis; handles tangent
#                                                      stretch at screen edges)
#
# Polar coords are linear in the eye's actual motor degrees of freedom; the
# polynomial expansion absorbs the tan(theta) stretching that fitting a flat
# screen to an angular signal otherwise leaves on the table at the corners.
BASIS_CARTESIAN = "cartesian"
BASIS_POLAR = "polar"
BASIS_POLY_POLAR = "polynomial-polar"
ALL_BASES: tuple[str, ...] = (BASIS_CARTESIAN, BASIS_POLAR, BASIS_POLY_POLAR)


def _basis_dim(basis: str) -> int:
    if basis == BASIS_CARTESIAN:
        return 4
    if basis == BASIS_POLAR:
        return 3
    if basis == BASIS_POLY_POLAR:
        return 6
    raise ValueError(f"unknown basis: {basis!r}")


def _design(features: np.ndarray, basis: str = BASIS_CARTESIAN) -> np.ndarray:
    """Build the regression design matrix for the chosen basis.

    Cartesian: [1, vx, vy, vz]. Direct, requires no axis convention beyond
    MediaPipe's.

    Polar bases convert the unit gaze vector into angular coordinates:
        yaw   = atan2(vx, vz)   — left/right rotation in the camera's XZ plane
        pitch = arcsin(vy)      — up/down (vy clamped to [-1, 1] for safety)
    Then either pass them through linearly (3 params) or polynomially expanded
    (6 params: + yaw^2, pitch^2, yaw*pitch).
    """
    if features.ndim == 1:
        features = features.reshape(1, -1)
    n = features.shape[0]
    ones = np.ones((n, 1))
    if basis == BASIS_CARTESIAN:
        return np.concatenate([ones, features], axis=1)
    vx = features[:, 0]
    vy = features[:, 1]
    vz = features[:, 2]
    yaw = np.arctan2(vx, vz)
    pitch = np.arcsin(np.clip(vy, -1.0, 1.0))
    if basis == BASIS_POLAR:
        return np.column_stack([ones[:, 0], yaw, pitch])
    if basis == BASIS_POLY_POLAR:
        return np.column_stack([
            ones[:, 0], yaw, pitch, yaw * yaw, pitch * pitch, yaw * pitch,
        ])
    raise ValueError(f"unknown basis: {basis!r}")


def _solve(
    features: np.ndarray,
    targets: np.ndarray,
    weights: np.ndarray,
    basis: str = BASIS_CARTESIAN,
) -> tuple[np.ndarray, np.ndarray]:
    X = _design(features, basis)
    sw = np.sqrt(np.asarray(weights, dtype=float))
    Xw = X * sw[:, None]
    yx = targets[:, 0] * sw
    yy = targets[:, 1] * sw
    coef_x, *_ = np.linalg.lstsq(Xw, yx, rcond=None)
    coef_y, *_ = np.linalg.lstsq(Xw, yy, rcond=None)
    return coef_x, coef_y


def error_weight(
    err_px: float, scale_px: float, min_w: float, max_w: float
) -> float:
    """Map prediction-error magnitude to a refit sample weight.

    Linear in `err_px`, clipped to `[min_w, max_w]`. The clamp matters: an
    unbounded weight from a freak 5000px click could rotate the fit by 30%
    in one update and cascade into oscillation on the next click.
    """
    return float(np.clip(err_px / scale_px, min_w, max_w))


def _effective_weights(
    weights: np.ndarray,
    is_anchor: np.ndarray,
    added_at: np.ndarray,
    half_life_s: float,
    now: float,
) -> np.ndarray:
    """Apply time decay to refinement weights. Anchors pass through unchanged.

    Anchors have NaN added_at (no meaningful "age"); refinements have a
    wall-clock timestamp. Decay is `weight * exp(-age_s / half_life_s)`
    with age clamped to >=0 so a clock skew can't amplify a refinement.
    """
    eff = weights.astype(float).copy()
    refinement_mask = ~is_anchor
    if refinement_mask.any():
        ages = now - added_at[refinement_mask]
        ages = np.maximum(ages, 0.0)
        decay = np.exp(-ages / half_life_s)
        eff[refinement_mask] = weights[refinement_mask] * decay
    return eff


@dataclass
class GazeModel:
    coef_x: np.ndarray
    coef_y: np.ndarray
    screen_w: int
    screen_h: int
    features: np.ndarray = field(default_factory=lambda: np.empty((0, 3)))
    targets: np.ndarray = field(default_factory=lambda: np.empty((0, 2)))
    weights: np.ndarray = field(default_factory=lambda: np.empty((0,)))
    # True for the original 9-point grid (and any sample explicitly added with
    # is_anchor=True). False for click-to-refine samples — those are subject to
    # eviction past MAX_REFINEMENTS and to time decay at refit.
    is_anchor: np.ndarray = field(
        default_factory=lambda: np.empty((0,), dtype=bool)
    )
    # Wall-clock seconds (time.time()) when a refinement was added. NaN for
    # anchors. Wall clock not monotonic so the value survives save+reload.
    added_at: np.ndarray = field(default_factory=lambda: np.empty((0,)))
    # Median (yaw, pitch, roll) in degrees of the user's head during calibration.
    # None for pre-#5 calibrations (gating then disabled). Realtime tracking
    # compares the live pose against this baseline to refuse predictions when
    # the head has rotated out of the calibrated posture.
    head_pose_baseline: tuple[float, float, float] | None = None
    # Regression basis used to expand (vx, vy, vz) for the lstsq. Backward
    # compat: pre-basis calibrations load as BASIS_CARTESIAN.
    basis: str = BASIS_CARTESIAN

    @classmethod
    def fit(
        cls,
        features: np.ndarray,
        targets: np.ndarray,
        screen_w: int,
        screen_h: int,
        basis: str = BASIS_CARTESIAN,
    ) -> GazeModel:
        features = np.asarray(features, dtype=float)
        targets = np.asarray(targets, dtype=float)
        weights = np.ones(features.shape[0], dtype=float)
        is_anchor = np.ones(features.shape[0], dtype=bool)
        added_at = np.full(features.shape[0], np.nan)
        coef_x, coef_y = _solve(features, targets, weights, basis)
        return cls(
            coef_x=coef_x,
            coef_y=coef_y,
            screen_w=screen_w,
            screen_h=screen_h,
            features=features,
            targets=targets,
            weights=weights,
            is_anchor=is_anchor,
            added_at=added_at,
            basis=basis,
        )

    def predict(self, feature: tuple[float, float, float]) -> tuple[float, float]:
        X = _design(np.asarray([feature], dtype=float), self.basis)
        return ((X @ self.coef_x).item(), (X @ self.coef_y).item())

    def add_sample(
        self,
        feature: tuple[float, float, float],
        target: tuple[float, float],
        weight: float = 1.0,
        is_anchor: bool = False,
        added_at: float | None = None,
    ) -> None:
        """Append a (feature, target, weight) sample and refit in place.

        Refinements (default) get a wall-clock `added_at` and may evict the
        oldest existing refinement if the cap is hit. Anchors are permanent
        and ignore the cap.

        `weight` controls how much the new sample bends the fit relative to
        existing samples (calibration fixations are weight 1.0). Click-to-
        refine derives `weight` from prediction error via `error_weight`.
        """
        if added_at is None:
            added_at = time.time() if not is_anchor else float("nan")
        self.features = np.vstack(
            [self.features, np.asarray(feature, dtype=float).reshape(1, -1)]
        )
        self.targets = np.vstack(
            [self.targets, np.asarray(target, dtype=float).reshape(1, -1)]
        )
        self.weights = np.concatenate(
            [self.weights, np.asarray([weight], dtype=float)]
        )
        self.is_anchor = np.concatenate(
            [self.is_anchor, np.asarray([is_anchor], dtype=bool)]
        )
        self.added_at = np.concatenate(
            [self.added_at, np.asarray([added_at], dtype=float)]
        )
        if not is_anchor:
            self._evict_excess_refinements()
        self._refit()

    def _evict_excess_refinements(self) -> None:
        """FIFO-evict the oldest refinements when the count exceeds MAX_REFINEMENTS.

        Eviction is based on `added_at` ascending (oldest first). Anchors are
        never touched even if their slot indices fall in the eviction range.
        """
        refinement_mask = ~self.is_anchor
        n_refinements = int(refinement_mask.sum())
        excess = n_refinements - MAX_REFINEMENTS
        if excess <= 0:
            return
        refinement_indices = np.where(refinement_mask)[0]
        oldest_order = np.argsort(self.added_at[refinement_indices])
        indices_to_evict = refinement_indices[oldest_order[:excess]]
        keep_mask = np.ones(self.features.shape[0], dtype=bool)
        keep_mask[indices_to_evict] = False
        self.features = self.features[keep_mask]
        self.targets = self.targets[keep_mask]
        self.weights = self.weights[keep_mask]
        self.is_anchor = self.is_anchor[keep_mask]
        self.added_at = self.added_at[keep_mask]

    def pop_last_sample(
        self,
    ) -> tuple[tuple[float, float, float], tuple[float, float], float]:
        """Remove and return the most recent (feature, target, weight); refit.

        Raises ValueError if the model has no samples to pop OR if the most
        recent sample is an anchor — protects calibration grid samples from
        being silently undone when the session refinement counter has drifted
        past actual model state (which happens after eviction).
        """
        if self.features.shape[0] == 0:
            raise ValueError("no samples to pop")
        if bool(self.is_anchor[-1]):
            raise ValueError("last sample is an anchor; refusing to pop")
        feat = (
            float(self.features[-1, 0]),
            float(self.features[-1, 1]),
            float(self.features[-1, 2]),
        )
        targ = (float(self.targets[-1, 0]), float(self.targets[-1, 1]))
        weight = float(self.weights[-1])
        self.features = self.features[:-1]
        self.targets = self.targets[:-1]
        self.weights = self.weights[:-1]
        self.is_anchor = self.is_anchor[:-1]
        self.added_at = self.added_at[:-1]
        if self.features.shape[0] > 0:
            self._refit()
        return feat, targ, weight

    def _refit(self) -> None:
        """Recompute coefficients with current effective weights."""
        if self.features.shape[0] == 0:
            return
        eff = _effective_weights(
            self.weights, self.is_anchor, self.added_at,
            REFINEMENT_HALF_LIFE_S, time.time(),
        )
        self.coef_x, self.coef_y = _solve(
            self.features, self.targets, eff, self.basis
        )

    def to_json(self) -> str:
        # JSON doesn't represent NaN, so anchor added_at NaNs serialize as null.
        added_at_serialized = [
            None if not np.isfinite(x) else float(x) for x in self.added_at
        ]
        return json.dumps(
            {
                "coef_x": self.coef_x.tolist(),
                "coef_y": self.coef_y.tolist(),
                "screen_w": self.screen_w,
                "screen_h": self.screen_h,
                "features": self.features.tolist(),
                "targets": self.targets.tolist(),
                "weights": self.weights.tolist(),
                "is_anchor": self.is_anchor.tolist(),
                "added_at": added_at_serialized,
                "head_pose_baseline": (
                    list(self.head_pose_baseline)
                    if self.head_pose_baseline is not None
                    else None
                ),
                "basis": self.basis,
            }
        )

    @classmethod
    def from_json(cls, s: str) -> GazeModel:
        d = json.loads(s)
        coef_x = np.asarray(d["coef_x"], dtype=float)
        coef_y = np.asarray(d["coef_y"], dtype=float)
        # Pre-basis calibrations have no "basis" key -> default cartesian (the
        # only basis that existed before #10).
        basis = d.get("basis", BASIS_CARTESIAN)
        expected_dim = _basis_dim(basis)
        if coef_x.shape != (expected_dim,) or coef_y.shape != (expected_dim,):
            raise ValueError(
                f"Calibration coefs have shape {coef_x.shape}, "
                f"expected ({expected_dim},) for basis {basis!r}. "
                "Re-run `gaze-tracker calibrate` to overwrite."
            )
        features = np.asarray(d.get("features", []), dtype=float).reshape(-1, 3)
        targets = np.asarray(d.get("targets", []), dtype=float).reshape(-1, 2)
        # Pre-weighting calibrations have no "weights" key — synthesize ones so
        # they load and continue to refit identically until a click overwrites.
        raw_w = d.get("weights")
        if raw_w is None:
            weights = np.ones(features.shape[0], dtype=float)
        else:
            weights = np.asarray(raw_w, dtype=float)
        # Pre-anchor calibrations have no is_anchor / added_at — promote everything
        # to anchors. Pre-existing refinements (if any) lose their "refinement"
        # status across this load; that's acceptable, since we don't trust their
        # original added_at values to reflect the current session anyway.
        raw_anchor = d.get("is_anchor")
        if raw_anchor is None:
            is_anchor = np.ones(features.shape[0], dtype=bool)
        else:
            is_anchor = np.asarray(raw_anchor, dtype=bool)
        raw_added_at = d.get("added_at")
        if raw_added_at is None:
            added_at = np.full(features.shape[0], np.nan)
        else:
            added_at = np.asarray(
                [np.nan if x is None else x for x in raw_added_at], dtype=float
            )
        raw_hp = d.get("head_pose_baseline")
        head_pose_baseline = (
            (float(raw_hp[0]), float(raw_hp[1]), float(raw_hp[2]))
            if raw_hp is not None
            else None
        )
        return cls(
            coef_x=coef_x,
            coef_y=coef_y,
            screen_w=int(d["screen_w"]),
            screen_h=int(d["screen_h"]),
            features=features,
            targets=targets,
            weights=weights,
            is_anchor=is_anchor,
            added_at=added_at,
            head_pose_baseline=head_pose_baseline,
            basis=basis,
        )

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_json())

    @classmethod
    def load(cls, path: Path) -> GazeModel:
        return cls.from_json(path.read_text())


def calibration_path() -> Path:
    config_home = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config"))
    return config_home / "gaze-tracker" / "calibration.json"
