import json
import math
import time

import numpy as np
import pytest

from gaze_tracker.mapping import (
    ALL_BASES,
    BASIS_CARTESIAN,
    BASIS_POLAR,
    BASIS_POLY_POLAR,
    MAX_REFINEMENTS,
    REFINEMENT_HALF_LIFE_S,
    GazeModel,
    _basis_dim,
    _design,
    _effective_weights,
    error_weight,
)


def test_fit_recovers_affine_map_exactly():
    # With a linear design matrix and no noise, lstsq should recover the affine
    # map screen = A @ gaze + b exactly.
    rng = np.random.default_rng(0)
    feats = rng.uniform(-0.2, 0.2, (30, 3))
    A_x = np.array([1800.0, 50.0, -300.0])
    A_y = np.array([40.0, 1000.0, 200.0])
    b_x, b_y = 960.0, 540.0
    targets = np.stack(
        [feats @ A_x + b_x, feats @ A_y + b_y],
        axis=1,
    )
    model = GazeModel.fit(feats, targets, screen_w=1920, screen_h=1080)
    for i in range(30):
        px, py = model.predict(tuple(feats[i]))
        assert abs(px - targets[i, 0]) < 1e-6
        assert abs(py - targets[i, 1]) < 1e-6


def test_fit_generalizes_to_held_out_points():
    # Noise-free affine generation; train on 9 points (as in the calibration
    # grid), evaluate on 20 held-out points. Should match exactly since the
    # model class contains the true function.
    rng = np.random.default_rng(1)
    A_x = np.array([1500.0, -100.0, 0.0])
    A_y = np.array([0.0, 900.0, 0.0])
    b_x, b_y = 960.0, 540.0
    train_feats = rng.uniform(-0.2, 0.2, (9, 3))
    train_targets = np.stack(
        [train_feats @ A_x + b_x, train_feats @ A_y + b_y], axis=1
    )
    model = GazeModel.fit(train_feats, train_targets, screen_w=1920, screen_h=1080)

    test_feats = rng.uniform(-0.2, 0.2, (20, 3))
    for f in test_feats:
        px, py = model.predict(tuple(f))
        assert abs(px - (f @ A_x + b_x)) < 1e-6
        assert abs(py - (f @ A_y + b_y)) < 1e-6


def test_json_roundtrip_preserves_samples():
    rng = np.random.default_rng(2)
    feats = rng.uniform(-0.2, 0.2, (9, 3))
    targets = rng.uniform(0, 1920, (9, 2))
    model = GazeModel.fit(feats, targets, screen_w=1920, screen_h=1080)
    model2 = GazeModel.from_json(model.to_json())
    np.testing.assert_allclose(model.coef_x, model2.coef_x)
    np.testing.assert_allclose(model.coef_y, model2.coef_y)
    np.testing.assert_allclose(model.features, model2.features)
    np.testing.assert_allclose(model.targets, model2.targets)
    assert model2.screen_w == 1920
    assert model2.screen_h == 1080


def test_add_sample_shrinks_local_error():
    # Simulate a misfit model: true map is nonlinear, we fit a linear base on
    # noisy samples far from the new test point. Adding a truth sample at that
    # point should reduce error there.
    rng = np.random.default_rng(3)
    base_feats = rng.uniform(-0.1, 0.1, (9, 3))
    A_x = np.array([1500.0, -200.0, 50.0])
    A_y = np.array([100.0, 900.0, -10.0])
    bx, by = 960.0, 540.0

    def true_screen(f: np.ndarray) -> np.ndarray:
        # Slight nonlinearity that the linear model cannot capture perfectly.
        return np.array([f @ A_x + bx + 200.0 * f[0] * f[1], f @ A_y + by])

    base_targets = np.stack([true_screen(f) for f in base_feats], axis=0)
    model = GazeModel.fit(base_feats, base_targets, screen_w=1920, screen_h=1080)

    probe = np.array([0.15, 0.12, -0.93])  # outside the base cluster
    truth = true_screen(probe)
    before = np.array(model.predict(tuple(probe)))
    err_before = float(np.linalg.norm(before - truth))

    model.add_sample(tuple(probe), tuple(truth))
    after = np.array(model.predict(tuple(probe)))
    err_after = float(np.linalg.norm(after - truth))

    assert err_after < err_before


def test_add_sample_grows_dataset():
    rng = np.random.default_rng(4)
    feats = rng.uniform(-0.1, 0.1, (9, 3))
    targets = rng.uniform(0, 1920, (9, 2))
    model = GazeModel.fit(feats, targets, screen_w=1920, screen_h=1080)
    assert model.features.shape == (9, 3)
    model.add_sample((0.1, -0.05, -0.95), (800.0, 400.0))
    assert model.features.shape == (10, 3)
    assert model.targets.shape == (10, 2)
    assert model.weights.shape == (10,)
    assert model.weights[-1] == 1.0


def test_error_weight_scales_and_clips():
    # Linear region: err / scale = weight, when within bounds.
    assert error_weight(40.0, 20.0, 1.0, 80.0) == 2.0
    assert error_weight(200.0, 20.0, 1.0, 80.0) == 10.0
    # Floor: tiny error pinned to min.
    assert error_weight(2.0, 20.0, 5.0, 80.0) == 5.0
    # Ceiling: huge error clipped to max so a freak click can't blow up the fit.
    assert error_weight(5000.0, 20.0, 5.0, 80.0) == 80.0
    # Boundary cases land exactly on the bounds.
    assert error_weight(0.0, 20.0, 1.0, 80.0) == 1.0


def test_weighted_add_sample_pulls_fit_harder_than_unweighted():
    # Build a clean linear calibration on 30 samples. A single click far from
    # the fit added at weight 1 nudges the prediction; the same click added at
    # weight 30 should pull the prediction much closer to the click coord.
    rng = np.random.default_rng(7)
    feats = rng.uniform(-0.15, 0.15, (30, 3))
    A_x = np.array([1500.0, -100.0, 50.0])
    A_y = np.array([60.0, 1100.0, -30.0])
    bx, by = 960.0, 540.0
    targets = np.stack([feats @ A_x + bx, feats @ A_y + by], axis=1)

    base = GazeModel.fit(feats, targets, screen_w=1920, screen_h=1080)
    probe_feat = (0.2, 0.18, -0.92)
    click_target = (1700.0, 950.0)
    pred_before = np.array(base.predict(probe_feat))
    err_before = float(np.linalg.norm(pred_before - np.array(click_target)))

    # Two parallel forks of the same model: light vs heavy click.
    light = GazeModel.fit(feats, targets, screen_w=1920, screen_h=1080)
    light.add_sample(probe_feat, click_target, weight=1.0)
    pred_light = np.array(light.predict(probe_feat))
    err_light = float(np.linalg.norm(pred_light - np.array(click_target)))

    heavy = GazeModel.fit(feats, targets, screen_w=1920, screen_h=1080)
    heavy.add_sample(probe_feat, click_target, weight=30.0)
    pred_heavy = np.array(heavy.predict(probe_feat))
    err_heavy = float(np.linalg.norm(pred_heavy - np.array(click_target)))

    # Both reduce error vs the base; heavy reduces it strictly more.
    assert err_light < err_before
    assert err_heavy < err_light


def test_json_roundtrip_preserves_weights():
    rng = np.random.default_rng(8)
    feats = rng.uniform(-0.1, 0.1, (9, 3))
    targets = rng.uniform(0, 1920, (9, 2))
    model = GazeModel.fit(feats, targets, screen_w=1920, screen_h=1080)
    model.add_sample((0.05, -0.02, -0.95), (1000.0, 500.0), weight=42.0)
    model2 = GazeModel.from_json(model.to_json())
    np.testing.assert_allclose(model.weights, model2.weights)
    assert model2.weights[-1] == 42.0
    np.testing.assert_allclose(model.coef_x, model2.coef_x)
    np.testing.assert_allclose(model.coef_y, model2.coef_y)


def test_pop_last_sample_returns_popped_data():
    rng = np.random.default_rng(10)
    feats = rng.uniform(-0.1, 0.1, (9, 3))
    targets = rng.uniform(0, 1920, (9, 2))
    model = GazeModel.fit(feats, targets, screen_w=1920, screen_h=1080)
    feat, targ, w = (0.05, -0.02, -0.95), (1000.0, 500.0), 7.5
    model.add_sample(feat, targ, weight=w)

    popped = model.pop_last_sample()
    assert popped[0] == feat
    assert popped[1] == targ
    assert popped[2] == w
    assert model.features.shape == (9, 3)
    assert model.targets.shape == (9, 2)
    assert model.weights.shape == (9,)


def test_pop_last_sample_restores_pre_add_fit():
    # Add a sample then pop it. The coefficients should match the pre-add
    # values within numerical precision — the round-trip is a no-op.
    rng = np.random.default_rng(11)
    feats = rng.uniform(-0.1, 0.1, (9, 3))
    targets = rng.uniform(0, 1920, (9, 2))
    model = GazeModel.fit(feats, targets, screen_w=1920, screen_h=1080)
    coef_x_before = model.coef_x.copy()
    coef_y_before = model.coef_y.copy()

    model.add_sample((0.05, 0.02, -0.93), (1100.0, 600.0), weight=12.0)
    model.pop_last_sample()

    np.testing.assert_allclose(model.coef_x, coef_x_before)
    np.testing.assert_allclose(model.coef_y, coef_y_before)


def test_pop_last_sample_raises_on_empty():
    # Hand-build an empty model without going through fit() so we can
    # exercise the empty-state guard directly.
    empty = GazeModel(
        coef_x=np.zeros(4),
        coef_y=np.zeros(4),
        screen_w=1920,
        screen_h=1080,
    )
    with pytest.raises(ValueError, match="no samples"):
        empty.pop_last_sample()


# --- Anchors / refinements bookkeeping (#4) ------------------------------


def test_fit_marks_all_samples_as_anchors():
    feats = np.zeros((5, 3))
    targets = np.zeros((5, 2))
    model = GazeModel.fit(feats, targets, screen_w=1920, screen_h=1080)
    assert model.is_anchor.shape == (5,)
    assert model.is_anchor.all()
    assert np.isnan(model.added_at).all()


def test_add_refinement_default_is_not_anchor_and_records_time():
    feats = np.zeros((1, 3))
    targets = np.zeros((1, 2))
    model = GazeModel.fit(feats, targets, screen_w=1920, screen_h=1080)
    before = time.time()
    model.add_sample((0.1, 0.0, -0.9), (100.0, 100.0))
    after = time.time()
    assert bool(model.is_anchor[-1]) is False
    assert before <= model.added_at[-1] <= after


def test_add_anchor_explicit_keeps_is_anchor_true():
    feats = np.zeros((1, 3))
    targets = np.zeros((1, 2))
    model = GazeModel.fit(feats, targets, screen_w=1920, screen_h=1080)
    model.add_sample((0.1, 0.0, -0.9), (100.0, 100.0), is_anchor=True)
    assert bool(model.is_anchor[-1]) is True
    assert np.isnan(model.added_at[-1])


def test_refinement_evicts_oldest_when_over_cap():
    feats = np.zeros((1, 3))
    targets = np.zeros((1, 2))
    model = GazeModel.fit(feats, targets, screen_w=1920, screen_h=1080)
    # Add MAX + 1 refinements with strictly-increasing timestamps so eviction
    # order is deterministic. The oldest (added_at=1000) should be evicted.
    for i in range(MAX_REFINEMENTS + 1):
        model.add_sample(
            (i * 0.001, 0.0, -0.9),
            (i * 10.0, 0.0),
            added_at=1000.0 + i,
        )
    refinements = model.added_at[~model.is_anchor]
    assert len(refinements) == MAX_REFINEMENTS
    assert refinements.min() == 1001.0  # oldest survivor; 1000.0 was evicted


def test_anchor_addition_does_not_trigger_eviction():
    # Even with a count well above MAX_REFINEMENTS, anchors are never evicted.
    feats = np.zeros((1, 3))
    targets = np.zeros((1, 2))
    model = GazeModel.fit(feats, targets, screen_w=1920, screen_h=1080)
    for _ in range(MAX_REFINEMENTS * 2):
        model.add_sample((0.1, 0.0, -0.9), (100.0, 100.0), is_anchor=True)
    assert int(model.is_anchor.sum()) == MAX_REFINEMENTS * 2 + 1


def test_pop_last_sample_refuses_anchor():
    feats = np.zeros((3, 3))
    targets = np.zeros((3, 2))
    model = GazeModel.fit(feats, targets, screen_w=1920, screen_h=1080)
    # All 3 are anchors after fit().
    with pytest.raises(ValueError, match="anchor"):
        model.pop_last_sample()


def test_pop_last_sample_after_eviction_still_pops_refinement():
    # End-to-end: add MAX+1 refinements, then pop_last_sample should pop the
    # most recent refinement (not raise on anchor) — protects undo against
    # silent counter drift after eviction.
    feats = np.zeros((1, 3))
    targets = np.zeros((1, 2))
    model = GazeModel.fit(feats, targets, screen_w=1920, screen_h=1080)
    for i in range(MAX_REFINEMENTS + 1):
        model.add_sample((0.0, 0.0, -0.9), (10.0 * i, 0.0), added_at=1000.0 + i)
    feat, targ, w = model.pop_last_sample()
    # The most recent had added_at=1000+MAX_REFINEMENTS, target_x=10*MAX_REFINEMENTS
    assert targ == (10.0 * MAX_REFINEMENTS, 0.0)


# --- Time decay (#4) ------------------------------------------------------


def test_effective_weights_pass_anchors_through():
    weights = np.array([5.0, 3.0])
    is_anchor = np.array([True, True])
    added_at = np.array([np.nan, np.nan])
    eff = _effective_weights(weights, is_anchor, added_at, REFINEMENT_HALF_LIFE_S, now=2000.0)
    np.testing.assert_array_equal(eff, weights)


def test_effective_weights_decay_at_half_life():
    # Refinement at exactly one half-life of age -> e^-1 ~= 0.3679.
    weights = np.array([10.0])
    is_anchor = np.array([False])
    now = 5000.0
    added_at = np.array([now - REFINEMENT_HALF_LIFE_S])
    eff = _effective_weights(weights, is_anchor, added_at, REFINEMENT_HALF_LIFE_S, now=now)
    assert abs(eff[0] - 10.0 * np.exp(-1.0)) < 1e-9


def test_effective_weights_no_decay_for_zero_age():
    weights = np.array([7.0])
    is_anchor = np.array([False])
    now = 5000.0
    added_at = np.array([now])
    eff = _effective_weights(weights, is_anchor, added_at, REFINEMENT_HALF_LIFE_S, now=now)
    assert abs(eff[0] - 7.0) < 1e-9


def test_effective_weights_clamp_negative_age_to_zero():
    # Future-stamped refinement (clock skew). Should not amplify weight.
    weights = np.array([4.0])
    is_anchor = np.array([False])
    now = 5000.0
    added_at = np.array([now + 10000.0])  # in the future
    eff = _effective_weights(weights, is_anchor, added_at, REFINEMENT_HALF_LIFE_S, now=now)
    assert eff[0] == 4.0  # clamped age=0 -> decay factor 1.0


def test_effective_weights_mixed_anchors_and_refinements():
    weights = np.array([1.0, 1.0, 1.0])
    is_anchor = np.array([True, False, False])
    now = 2000.0
    added_at = np.array([np.nan, now - REFINEMENT_HALF_LIFE_S, now])
    eff = _effective_weights(weights, is_anchor, added_at, REFINEMENT_HALF_LIFE_S, now=now)
    assert eff[0] == 1.0                # anchor
    assert abs(eff[1] - np.exp(-1.0)) < 1e-9  # refinement at half-life
    assert eff[2] == 1.0                # fresh refinement


# --- Schema migration (#4) ------------------------------------------------


def test_load_pre_anchor_save_promotes_all_to_anchors():
    # Synthesize an old-schema blob (no is_anchor / added_at). Loading must
    # treat every sample as an anchor — preserves prior fit behavior.
    rng = np.random.default_rng(20)
    feats = rng.uniform(-0.1, 0.1, (5, 3))
    targets = rng.uniform(0, 1920, (5, 2))
    model = GazeModel.fit(feats, targets, screen_w=1920, screen_h=1080)
    blob = json.loads(model.to_json())
    blob.pop("is_anchor", None)
    blob.pop("added_at", None)
    revived = GazeModel.from_json(json.dumps(blob))
    assert revived.is_anchor.shape == (5,)
    assert revived.is_anchor.all()
    assert np.isnan(revived.added_at).all()
    # Coefficients should still match (no decay applied to anchors).
    np.testing.assert_allclose(revived.coef_x, model.coef_x, atol=1e-9)
    np.testing.assert_allclose(revived.coef_y, model.coef_y, atol=1e-9)


def test_save_load_preserves_anchors_and_added_at():
    rng = np.random.default_rng(21)
    feats = rng.uniform(-0.1, 0.1, (2, 3))
    targets = rng.uniform(0, 1920, (2, 2))
    model = GazeModel.fit(feats, targets, screen_w=1920, screen_h=1080)
    model.add_sample((0.05, 0.0, -0.9), (500.0, 300.0), added_at=1234567.0)
    revived = GazeModel.from_json(model.to_json())
    np.testing.assert_array_equal(revived.is_anchor, model.is_anchor)
    # Anchors -> NaN; refinement -> the explicit timestamp we set.
    assert np.isnan(revived.added_at[0])
    assert np.isnan(revived.added_at[1])
    assert revived.added_at[2] == 1234567.0


# --- Head pose baseline (#5) ---------------------------------------------


def test_head_pose_baseline_defaults_to_none():
    feats = np.zeros((2, 3))
    targets = np.zeros((2, 2))
    model = GazeModel.fit(feats, targets, screen_w=1920, screen_h=1080)
    assert model.head_pose_baseline is None


def test_head_pose_baseline_serializes_round_trip():
    feats = np.zeros((2, 3))
    targets = np.zeros((2, 2))
    model = GazeModel.fit(feats, targets, screen_w=1920, screen_h=1080)
    model.head_pose_baseline = (1.5, -2.0, 0.3)
    revived = GazeModel.from_json(model.to_json())
    assert revived.head_pose_baseline == (1.5, -2.0, 0.3)


def test_head_pose_baseline_loads_as_none_when_missing():
    feats = np.zeros((2, 3))
    targets = np.zeros((2, 2))
    model = GazeModel.fit(feats, targets, screen_w=1920, screen_h=1080)
    blob = json.loads(model.to_json())
    blob.pop("head_pose_baseline", None)
    revived = GazeModel.from_json(json.dumps(blob))
    assert revived.head_pose_baseline is None


def test_head_pose_baseline_explicit_null_loads_as_none():
    feats = np.zeros((2, 3))
    targets = np.zeros((2, 2))
    model = GazeModel.fit(feats, targets, screen_w=1920, screen_h=1080)
    blob = json.loads(model.to_json())
    blob["head_pose_baseline"] = None
    revived = GazeModel.from_json(json.dumps(blob))
    assert revived.head_pose_baseline is None


# --- Regression bases (#10 polar A/B) -------------------------------------


def test_basis_dim_known_values():
    assert _basis_dim(BASIS_CARTESIAN) == 4
    assert _basis_dim(BASIS_POLAR) == 3
    assert _basis_dim(BASIS_POLY_POLAR) == 6


def test_basis_dim_unknown_raises():
    with pytest.raises(ValueError):
        _basis_dim("nonsense")


def test_design_cartesian_is_intercept_then_features():
    feats = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
    X = _design(feats, BASIS_CARTESIAN)
    assert X.shape == (2, 4)
    np.testing.assert_array_equal(X[:, 0], [1.0, 1.0])
    np.testing.assert_array_equal(X[:, 1:], feats)


def test_design_polar_returns_yaw_pitch():
    # For (vx, vy, vz) = (sin(yaw)cos(pitch), sin(pitch), cos(yaw)cos(pitch))
    # we should recover the (yaw, pitch) we started from.
    yaw_in = 0.4
    pitch_in = -0.3
    feats = np.array([[
        math.sin(yaw_in) * math.cos(pitch_in),
        math.sin(pitch_in),
        math.cos(yaw_in) * math.cos(pitch_in),
    ]])
    X = _design(feats, BASIS_POLAR)
    assert X.shape == (1, 3)
    assert X[0, 0] == 1.0
    assert abs(X[0, 1] - yaw_in) < 1e-9
    assert abs(X[0, 2] - pitch_in) < 1e-9


def test_design_polynomial_polar_includes_quadratic_terms():
    yaw_in = 0.2
    pitch_in = 0.1
    feats = np.array([[
        math.sin(yaw_in) * math.cos(pitch_in),
        math.sin(pitch_in),
        math.cos(yaw_in) * math.cos(pitch_in),
    ]])
    X = _design(feats, BASIS_POLY_POLAR)
    assert X.shape == (1, 6)
    # [1, yaw, pitch, yaw^2, pitch^2, yaw*pitch]
    assert X[0, 0] == 1.0
    assert abs(X[0, 1] - yaw_in) < 1e-9
    assert abs(X[0, 2] - pitch_in) < 1e-9
    assert abs(X[0, 3] - yaw_in * yaw_in) < 1e-9
    assert abs(X[0, 4] - pitch_in * pitch_in) < 1e-9
    assert abs(X[0, 5] - yaw_in * pitch_in) < 1e-9


def test_design_polar_clamps_vy_for_arcsin_safety():
    # vy slightly out of [-1, 1] from float drift. Must not crash arcsin.
    feats = np.array([[0.0, 1.0 + 1e-12, 0.0]])
    X = _design(feats, BASIS_POLAR)
    assert abs(X[0, 2] - math.pi / 2) < 1e-3


def test_design_unknown_basis_raises():
    with pytest.raises(ValueError):
        _design(np.zeros((1, 3)), "no-such-basis")


def test_polar_fit_recovers_known_linear_yaw_pitch_map():
    # Generate synthetic features from known (yaw, pitch), targets linear in
    # (yaw, pitch). Polar lstsq should recover the coefficients exactly.
    rng = np.random.default_rng(50)
    yaws = rng.uniform(-0.5, 0.5, 30)
    pitches = rng.uniform(-0.5, 0.5, 30)
    feats = np.column_stack([
        np.sin(yaws) * np.cos(pitches),
        np.sin(pitches),
        np.cos(yaws) * np.cos(pitches),
    ])
    target_x = 100.0 + 500.0 * yaws + 50.0 * pitches
    target_y = 200.0 + 30.0 * yaws + 800.0 * pitches
    targets = np.column_stack([target_x, target_y])
    model = GazeModel.fit(
        feats, targets, screen_w=1920, screen_h=1080, basis=BASIS_POLAR,
    )
    for i in range(30):
        px, py = model.predict(tuple(feats[i]))
        assert abs(px - target_x[i]) < 1e-6
        assert abs(py - target_y[i]) < 1e-6


def test_polynomial_polar_fit_recovers_known_quadratic_map():
    # Same setup with a quadratic-in-(yaw,pitch) ground truth — only the
    # polynomial-polar basis class contains this function.
    rng = np.random.default_rng(51)
    yaws = rng.uniform(-0.4, 0.4, 30)
    pitches = rng.uniform(-0.4, 0.4, 30)
    feats = np.column_stack([
        np.sin(yaws) * np.cos(pitches),
        np.sin(pitches),
        np.cos(yaws) * np.cos(pitches),
    ])
    target_x = (
        100.0 + 500.0 * yaws + 50.0 * pitches
        + 10.0 * yaws ** 2 + 5.0 * pitches ** 2 + 20.0 * yaws * pitches
    )
    target_y = (
        200.0 + 30.0 * yaws + 800.0 * pitches
        + 7.0 * yaws ** 2 + 3.0 * pitches ** 2 + 15.0 * yaws * pitches
    )
    targets = np.column_stack([target_x, target_y])
    model = GazeModel.fit(
        feats, targets, screen_w=1920, screen_h=1080, basis=BASIS_POLY_POLAR,
    )
    for i in range(30):
        px, py = model.predict(tuple(feats[i]))
        assert abs(px - target_x[i]) < 1e-3
        assert abs(py - target_y[i]) < 1e-3


def test_fit_default_basis_is_cartesian():
    feats = np.zeros((4, 3))
    targets = np.zeros((4, 2))
    model = GazeModel.fit(feats, targets, screen_w=1920, screen_h=1080)
    assert model.basis == BASIS_CARTESIAN
    assert model.coef_x.shape == (4,)


def test_save_load_preserves_basis():
    feats = np.zeros((10, 3))
    targets = np.zeros((10, 2))
    model = GazeModel.fit(
        feats, targets, screen_w=1920, screen_h=1080, basis=BASIS_POLAR,
    )
    revived = GazeModel.from_json(model.to_json())
    assert revived.basis == BASIS_POLAR
    assert revived.coef_x.shape == (3,)


def test_load_pre_basis_save_defaults_to_cartesian():
    feats = np.zeros((5, 3))
    targets = np.zeros((5, 2))
    model = GazeModel.fit(feats, targets, screen_w=1920, screen_h=1080)
    blob = json.loads(model.to_json())
    blob.pop("basis", None)
    revived = GazeModel.from_json(json.dumps(blob))
    assert revived.basis == BASIS_CARTESIAN


def test_from_json_rejects_coef_basis_mismatch():
    # If someone hand-edits the file to claim "polar" but coef_x is 4-dim,
    # loading should fail loudly rather than silently mis-predict forever.
    feats = np.zeros((4, 3))
    targets = np.zeros((4, 2))
    model = GazeModel.fit(feats, targets, screen_w=1920, screen_h=1080)  # cartesian
    blob = json.loads(model.to_json())
    blob["basis"] = BASIS_POLAR  # liar
    with pytest.raises(ValueError, match="polar"):
        GazeModel.from_json(json.dumps(blob))


def test_all_bases_round_trip():
    # Smoke test: every advertised basis must be fit-able, save-able, load-able.
    rng = np.random.default_rng(60)
    feats = rng.uniform(-0.2, 0.2, (15, 3))
    targets = rng.uniform(0, 1920, (15, 2))
    for basis in ALL_BASES:
        model = GazeModel.fit(
            feats, targets, screen_w=1920, screen_h=1080, basis=basis,
        )
        revived = GazeModel.from_json(model.to_json())
        assert revived.basis == basis
        np.testing.assert_allclose(revived.coef_x, model.coef_x)
        np.testing.assert_allclose(revived.coef_y, model.coef_y)


def test_from_json_synthesizes_weights_for_pre_weight_calibrations():
    # A calibration saved before per-sample weights existed has no "weights"
    # key. Loading must default to ones so behavior is identical until the
    # next click overwrites the model.
    rng = np.random.default_rng(9)
    feats = rng.uniform(-0.1, 0.1, (5, 3))
    targets = rng.uniform(0, 1920, (5, 2))
    model = GazeModel.fit(feats, targets, screen_w=1920, screen_h=1080)
    blob = json.loads(model.to_json())
    blob.pop("weights")
    revived = GazeModel.from_json(json.dumps(blob))
    np.testing.assert_allclose(revived.weights, np.ones(5))
    np.testing.assert_allclose(revived.coef_x, model.coef_x)
    np.testing.assert_allclose(revived.coef_y, model.coef_y)
