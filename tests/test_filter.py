import random

import pytest

from gaze_tracker.filter import (
    EARGate,
    MedianSmoother,
    OneEuroFilter,
    OneEuroFilter2D,
    SaccadeDetector,
    features_in_window,
)


def test_constant_signal_converges_to_value():
    f = OneEuroFilter(min_cutoff=1.0, beta=0.0)
    t = 0.0
    y = 0.0
    for _ in range(200):
        t += 1 / 30.0
        y = f(t, 5.0)
    assert abs(y - 5.0) < 1e-6


def test_stationary_signal_attenuates_noise():
    random.seed(0)
    f = OneEuroFilter(min_cutoff=1.0, beta=0.0)
    raw = [5.0 + random.gauss(0, 0.3) for _ in range(300)]
    t = 0.0
    filtered = []
    for x in raw:
        t += 1 / 60.0
        filtered.append(f(t, x))

    def var(xs: list[float]) -> float:
        m = sum(xs) / len(xs)
        return sum((x - m) ** 2 for x in xs) / len(xs)

    assert var(filtered[-100:]) < 0.25 * var(raw[-100:])


def test_high_velocity_tracks_closely_with_beta():
    # Linear ramp signal; beta>0 should let the filter keep up near the true value.
    f = OneEuroFilter(min_cutoff=1.0, beta=1.0)
    t = 0.0
    latest = 0.0
    for i in range(100):
        t += 1 / 60.0
        latest = f(t, float(i))
    assert abs(latest - 99.0) < 5.0


def test_2d_filter_passes_first_sample_through():
    f = OneEuroFilter2D()
    out = f(0.0, (1.0, 2.0))
    assert out == (1.0, 2.0)


def test_median_smoother_rejects_single_frame_outlier():
    m = MedianSmoother(window=5, dim=3)
    stable = (0.1, 0.2, -0.9)
    for _ in range(4):
        m(stable)
    out_spike = m((5.0, 5.0, 5.0))  # outlier, but only 1 of 5 in window
    # median of {0.1, 0.1, 0.1, 0.1, 5.0} per axis = 0.1 (unchanged)
    assert abs(out_spike[0] - 0.1) < 1e-9
    assert abs(out_spike[1] - 0.2) < 1e-9
    assert abs(out_spike[2] - (-0.9)) < 1e-9


def test_median_smoother_tracks_steady_shift():
    m = MedianSmoother(window=5, dim=3)
    # Push enough of the new value that it becomes the median
    for _ in range(3):
        m((0.0, 0.0, 0.0))
    for _ in range(3):
        out = m((1.0, 2.0, 3.0))
    assert out == (1.0, 2.0, 3.0)


def test_saccade_detector_first_sample_is_fixation():
    s = SaccadeDetector()
    assert s(0.0, (500.0, 500.0)) is False


def test_saccade_detector_stationary_signal_stays_fixated():
    s = SaccadeDetector(threshold_px_per_s=2500.0)
    t = 0.0
    for _ in range(60):
        t += 1 / 60.0
        assert s(t, (500.0, 500.0)) is False


def test_saccade_detector_slow_drift_stays_fixated():
    # 200 px/s drift — well below saccade range. Should never trip.
    s = SaccadeDetector(threshold_px_per_s=2500.0)
    t = 0.0
    x = 500.0
    for _ in range(120):
        t += 1 / 60.0
        x += 200.0 / 60.0
        assert s(t, (x, 500.0)) is False


def test_saccade_detector_fires_on_fast_motion():
    # 6000 px/s sustained — clearly a saccade.
    s = SaccadeDetector(threshold_px_per_s=2500.0, velocity_smooth=0.5)
    t = 0.0
    x = 0.0
    fired = False
    for _ in range(10):
        t += 1 / 60.0
        x += 6000.0 / 60.0
        if s(t, (x, 0.0)):
            fired = True
            break
    assert fired


def test_saccade_detector_returns_to_fixation_after_motion_stops():
    # Saccade, then sit still — detector should drop back to False once velocity decays.
    s = SaccadeDetector(threshold_px_per_s=2500.0, velocity_smooth=0.5)
    t = 0.0
    x = 0.0
    for _ in range(8):
        t += 1 / 60.0
        x += 6000.0 / 60.0
        s(t, (x, 0.0))
    # Now hold still; velocity should decay back below threshold.
    in_saccade_now = True
    for _ in range(20):
        t += 1 / 60.0
        in_saccade_now = s(t, (x, 0.0))
        if not in_saccade_now:
            break
    assert in_saccade_now is False


def test_saccade_detector_ignores_single_frame_spike():
    # One large displacement followed by stillness should not latch saccade.
    # The EMA on velocity damps single-frame spikes.
    s = SaccadeDetector(threshold_px_per_s=2500.0, velocity_smooth=0.2)
    t = 0.0
    s(t, (0.0, 0.0))
    # one big spike (3000 px/s for one frame)
    t += 1 / 60.0
    s(t, (50.0, 0.0))
    # Even if the spike trips it, holding still should clear it within a few frames.
    cleared = False
    for _ in range(15):
        t += 1 / 60.0
        if not s(t, (50.0, 0.0)):
            cleared = True
            break
    assert cleared
    # The exposed velocity attribute should remain comfortably below threshold
    # — not necessarily near zero (we may have broken out of the loop on the
    # first quiet frame), just nowhere near saccade range.
    assert s.velocity < 2500.0


def test_saccade_detector_reset_clears_state():
    s = SaccadeDetector(threshold_px_per_s=2500.0)
    t = 0.0
    x = 0.0
    for _ in range(8):
        t += 1 / 60.0
        x += 6000.0 / 60.0
        s(t, (x, 0.0))
    s.reset()
    assert s.velocity == 0.0
    # First call after reset must be treated as a fresh start (fixation).
    assert s(t + 1.0, (x, 0.0)) is False


def test_saccade_detector_invalid_smooth_raises():
    with pytest.raises(ValueError):
        SaccadeDetector(velocity_smooth=0.0)
    with pytest.raises(ValueError):
        SaccadeDetector(velocity_smooth=1.5)


# --- EARGate --------------------------------------------------------------


def test_ear_gate_admits_first_frames_below_min_n():
    g = EARGate(window_frames=20, min_n=5, tolerance=0.08)
    # First 4 frames have no useful baseline; admit unconditionally even at
    # absurd EAR values. The 5th frame trips into "ready" mode.
    for _ in range(4):
        assert g(0.01, 0.01) is True
    assert g.ready is False
    g(0.30, 0.28)  # 5th, gate becomes ready
    assert g.ready is True


def test_ear_gate_admits_steady_state():
    g = EARGate(window_frames=20, min_n=5, tolerance=0.08)
    for _ in range(40):
        assert g(0.30, 0.28) is True


def test_ear_gate_rejects_full_blink():
    g = EARGate(window_frames=20, min_n=5, tolerance=0.08)
    # Build steady baseline at open-eye EAR.
    for _ in range(15):
        g(0.30, 0.28)
    # Blink frame: both eyes drop. Must reject.
    assert g(0.05, 0.05) is False


def test_ear_gate_rejects_one_eye_blink():
    g = EARGate(window_frames=20, min_n=5, tolerance=0.08)
    for _ in range(15):
        g(0.30, 0.28)
    # Asymmetric squint: left half-closed, right open. Must reject — the
    # whole reason gating is per-eye and not averaged.
    assert g(0.10, 0.28) is False
    assert g(0.30, 0.10) is False


def test_ear_gate_admits_within_tolerance():
    g = EARGate(window_frames=20, min_n=5, tolerance=0.08)
    for _ in range(15):
        g(0.30, 0.28)
    # Drift inside tolerance band on both eyes — accept.
    assert g(0.30 + 0.07, 0.28 - 0.07) is True


def test_ear_gate_window_rolls_old_samples_evict():
    g = EARGate(window_frames=10, min_n=5, tolerance=0.08)
    # Fill window with one regime.
    for _ in range(10):
        g(0.30, 0.28)
    # Push a new regime through the window. After 10 new samples the old
    # samples are gone and the new baseline should be the new regime.
    for _ in range(10):
        g(0.20, 0.18)
    # A frame at the new baseline must be admitted; one at the old baseline
    # must now be rejected.
    assert g(0.20, 0.18) is True
    assert g(0.30, 0.28) is False


def test_ear_gate_reset_clears_state():
    g = EARGate(window_frames=20, min_n=5, tolerance=0.08)
    for _ in range(15):
        g(0.30, 0.28)
    assert g.ready is True
    g.reset()
    assert g.ready is False
    # First post-reset call admits unconditionally (back below min_n).
    assert g(0.05, 0.05) is True


def test_ear_gate_invalid_args_raise():
    with pytest.raises(ValueError):
        EARGate(window_frames=0)
    with pytest.raises(ValueError):
        EARGate(window_frames=10, min_n=0)
    with pytest.raises(ValueError):
        EARGate(window_frames=10, min_n=20)  # min_n > window
    with pytest.raises(ValueError):
        EARGate(tolerance=-0.01)


# --- features_in_window ---------------------------------------------------


def test_features_in_window_keeps_only_in_range_samples():
    # Window is [t_click - 200ms, t_click - 50ms] = [0.800, 0.950]. Keep only
    # samples whose timestamp is within those bounds inclusive.
    t_click = 1.000
    buf = [
        (0.700, (1.0, 1.0, 1.0), 0.0),     # 300ms ago — too old
        (0.820, (2.0, 2.0, 2.0), 0.0),     # 180ms ago — keep
        (0.900, (3.0, 3.0, 3.0), 7.5),     # 100ms ago — keep
        (0.970, (10.0, 10.0, 10.0), 0.0),  # 30ms ago — too recent (saccade window)
        (1.005, (100.0, 0.0, 0.0), 0.0),   # post-click — drop
    ]
    out = features_in_window(buf, t_click, 0.200, 0.050)
    assert out == [((2.0, 2.0, 2.0), 0.0), ((3.0, 3.0, 3.0), 7.5)]


def test_features_in_window_inclusive_bounds():
    # Samples landing exactly on the window edges should be included.
    t_click = 1.000
    buf = [
        (0.800, (1.0,), 0.0),  # exactly t-200ms
        (0.950, (2.0,), 0.0),  # exactly t-50ms
    ]
    out = features_in_window(buf, t_click, 0.200, 0.050)
    assert len(out) == 2


def test_features_in_window_empty_buffer():
    assert features_in_window([], 1.0, 0.2, 0.05) == []


def test_features_in_window_no_overlap_returns_empty():
    t_click = 1.000
    # All samples too old
    buf = [(0.500, (1.0,), 0.0), (0.600, (2.0,), 0.0)]
    assert features_in_window(buf, t_click, 0.2, 0.05) == []
