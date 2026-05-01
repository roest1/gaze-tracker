import numpy as np

from gaze_tracker.calibration import (
    EAR_TOLERANCE,
    GRID_LABELS,
    GRID_POINTS_NORM,
    _ear_in_band,
    _group_samples_by_target,
    _label_for_target,
    _loocv_residuals,
    loocv_warning,
)


def test_grid_has_nine_unique_points():
    assert len(GRID_POINTS_NORM) == 9
    assert len(set(GRID_POINTS_NORM)) == 9


def test_grid_within_unit_square():
    for x, y in GRID_POINTS_NORM:
        assert 0.0 <= x <= 1.0
        assert 0.0 <= y <= 1.0


def test_grid_starts_with_center():
    assert GRID_POINTS_NORM[0] == (0.5, 0.5)


def test_grid_corners_before_edges():
    # Indices 1–4 must all be corner points: both coords in {0.1, 0.9}.
    for x, y in GRID_POINTS_NORM[1:5]:
        assert x in (0.1, 0.9)
        assert y in (0.1, 0.9)
    # Indices 5–8 must all be edge midpoints: exactly one coord == 0.5.
    for x, y in GRID_POINTS_NORM[5:9]:
        assert (x == 0.5) ^ (y == 0.5)


def test_grid_uses_edge_hugging_layout():
    xs = {x for x, _ in GRID_POINTS_NORM}
    ys = {y for _, y in GRID_POINTS_NORM}
    assert xs == {0.1, 0.5, 0.9}
    assert ys == {0.1, 0.5, 0.9}


def test_ear_in_band_accepts_baseline_match():
    # Both eyes at exactly the per-eye baseline → trivially in band.
    assert _ear_in_band(0.30, 0.28, 0.30, 0.28)


def test_ear_in_band_accepts_within_tolerance():
    # Small drift below the absolute tolerance is fine.
    assert _ear_in_band(0.30 + EAR_TOLERANCE - 0.01, 0.28, 0.30, 0.28)
    assert _ear_in_band(0.30, 0.28 - EAR_TOLERANCE + 0.01, 0.30, 0.28)


def test_ear_in_band_rejects_one_eye_blink():
    # The asymmetric case the gate exists for: one eye at baseline, the other
    # half-closed. Must reject — half-blink corrupts that eye's iris centroid
    # and would silently bias the regression.
    assert not _ear_in_band(0.30, 0.05, 0.30, 0.28)
    assert not _ear_in_band(0.05, 0.28, 0.30, 0.28)


def test_ear_in_band_rejects_both_eyes_squint():
    # Both eyes drop together — full blink / heavy squint. Reject.
    assert not _ear_in_band(0.10, 0.09, 0.30, 0.28)


def test_ear_in_band_tolerance_override():
    # A wider tolerance accepts what the default rejects.
    assert _ear_in_band(0.30, 0.15, 0.30, 0.28, tolerance=0.20)
    assert not _ear_in_band(0.30, 0.15, 0.30, 0.28, tolerance=0.05)


# --- LOOCV ---------------------------------------------------------------


def _make_linear_grid_samples(
    samples_per_point: int = 4,
    screen_w: int = 1920,
    screen_h: int = 1080,
) -> list[tuple[tuple[float, float, float], tuple[float, float]]]:
    """Build samples that exactly satisfy a known affine map from feature
    -> target. Each grid point gets `samples_per_point` features that all
    map to the same target via a fixed (Ax, Ay, bx, by) — LOOCV residuals
    on this data should be ~0 within numeric noise."""
    Ax = np.array([1500.0, -100.0, 50.0])
    Ay = np.array([60.0, 1100.0, -30.0])
    bx, by = 960.0, 540.0

    samples: list[tuple[tuple[float, float, float], tuple[float, float]]] = []
    for nx, ny in GRID_POINTS_NORM:
        target = (nx * screen_w, ny * screen_h)
        # Pick the representative feature for this point by 2x2-solving for
        # (f0, f1) with f2 fixed at 0 — gives one feature that exactly
        # satisfies the affine map. Reuse it `samples_per_point` times so
        # all features at this point imply the same target (which is what
        # LOOCV's median-feature step expects).
        M = np.array([[Ax[0], Ax[1]], [Ay[0], Ay[1]]])
        rhs = np.array([target[0] - bx, target[1] - by])
        f0, f1 = np.linalg.solve(M, rhs)
        feat = (float(f0), float(f1), 0.0)
        for _ in range(samples_per_point):
            samples.append((feat, target))
    return samples


def test_group_samples_by_target_collects_per_point():
    samples = _make_linear_grid_samples(samples_per_point=4)
    groups = _group_samples_by_target(samples)
    assert len(groups) == len(GRID_POINTS_NORM)
    for feats in groups.values():
        assert len(feats) == 4


def test_loocv_residuals_near_zero_for_perfect_linear_data():
    samples = _make_linear_grid_samples(samples_per_point=4)
    res = _loocv_residuals(samples, 1920, 1080)
    assert len(res) == len(GRID_POINTS_NORM)
    for target, r in res:
        # Should be exact within numerical tolerance — the held-out point's
        # target is reproducible from any 8 of 9 grid points under the affine.
        assert r < 1e-3, f"target {target} had unexpected residual {r}"


def test_loocv_residuals_flags_corrupted_point():
    # Take the perfectly-linear set and corrupt one point's features so the
    # model trained on the OTHER 8 mispredicts that point's target.
    samples = _make_linear_grid_samples(samples_per_point=4)
    # Find the last point's samples and rewrite their feature.
    target_to_corrupt = samples[-1][1]
    for i, (feat, t) in enumerate(samples):
        if t == target_to_corrupt:
            samples[i] = ((feat[0] + 1.0, feat[1] - 1.0, feat[2] + 0.5), t)

    res = _loocv_residuals(samples, 1920, 1080)
    corrupted_r = next(r for t, r in res if t == target_to_corrupt)
    other_rs = [r for t, r in res if t != target_to_corrupt]
    # The corrupted point's residual must dominate the others.
    assert corrupted_r > max(other_rs) * 10


def test_label_for_target_resolves_corner():
    # GRID_POINTS_NORM[1] is (0.1, 0.1) labeled "TL".
    assert _label_for_target((0.1 * 1920, 0.1 * 1080), 1920, 1080) == "TL"


def test_label_for_target_resolves_center():
    assert _label_for_target((0.5 * 1920, 0.5 * 1080), 1920, 1080) == "center"


def test_label_for_target_unknown_returns_question_mark():
    # A point that doesn't sit on the calibration grid.
    assert _label_for_target((100.0, 100.0), 1920, 1080) == "?"


def test_loocv_warning_flags_outlier():
    residuals = [
        ((0.1 * 1920, 0.1 * 1080), 50.0),
        ((0.9 * 1920, 0.1 * 1080), 60.0),
        ((0.1 * 1920, 0.9 * 1080), 55.0),
        ((0.9 * 1920, 0.9 * 1080), 200.0),  # 4x median -> flagged
        ((0.5 * 1920, 0.5 * 1080), 45.0),
    ]
    msg = loocv_warning(residuals, 1920, 1080, outlier_factor=2.0)
    assert msg is not None
    assert "BR" in msg          # the corrupted point's grid label
    assert "200" in msg         # px count present
    assert "median" in msg      # the multiplier annotation is there


def test_loocv_warning_returns_none_when_uniform():
    residuals = [
        ((0.1 * 1920, 0.1 * 1080), 50.0),
        ((0.9 * 1920, 0.1 * 1080), 55.0),
        ((0.1 * 1920, 0.9 * 1080), 60.0),
        ((0.9 * 1920, 0.9 * 1080), 52.0),
    ]
    assert loocv_warning(residuals, 1920, 1080, outlier_factor=2.0) is None


def test_loocv_warning_returns_none_when_too_few_points():
    residuals = [
        ((0.1 * 1920, 0.1 * 1080), 50.0),
        ((0.9 * 1920, 0.1 * 1080), 200.0),
    ]
    assert loocv_warning(residuals, 1920, 1080) is None


def test_loocv_warning_returns_none_when_median_is_zero():
    # Degenerate: every point predicts perfectly. No useful "outlier" notion.
    residuals = [
        ((0.1 * 1920, 0.1 * 1080), 0.0),
        ((0.9 * 1920, 0.1 * 1080), 0.0),
        ((0.1 * 1920, 0.9 * 1080), 0.0),
    ]
    assert loocv_warning(residuals, 1920, 1080) is None


def test_grid_labels_match_grid_points_length():
    assert len(GRID_LABELS) == len(GRID_POINTS_NORM)
