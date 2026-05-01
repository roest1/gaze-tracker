import math

import numpy as np
import pytest

from gaze_tracker.landmarks import (
    _LEFT_EYE,
    _RIGHT_EYE,
    _ear,
    head_pose_max_dev_deg,
    head_pose_yxz_degrees,
)


class _LM:
    """Minimal stand-in for a MediaPipe normalized landmark."""

    __slots__ = ("x", "y")

    def __init__(self, x: float, y: float) -> None:
        self.x = x
        self.y = y


def _make_lm(eye_pixels: dict[int, tuple[float, float]], frame_w: int, frame_h: int):
    """Build an indexable landmark list with normalized coords from pixel coords."""
    max_idx = max(eye_pixels) + 1
    lm = [_LM(0.0, 0.0) for _ in range(max_idx)]
    for i, (px, py) in eye_pixels.items():
        lm[i] = _LM(px / frame_w, py / frame_h)
    return lm


def test_ear_typical_open_eye():
    # Roughly anatomical: 30px lid opening over 100px corner-to-corner = 0.30.
    outer, inner, top, bot = _LEFT_EYE
    lm = _make_lm(
        {outer: (100, 250), inner: (200, 250), top: (150, 235), bot: (150, 265)},
        frame_w=640, frame_h=480,
    )
    assert abs(_ear(lm, _LEFT_EYE, 640, 480) - 0.30) < 1e-6


def test_ear_drops_when_lids_close():
    outer, inner, top, bot = _LEFT_EYE
    closed = _make_lm(
        {outer: (100, 250), inner: (200, 250), top: (150, 247), bot: (150, 253)},
        frame_w=640, frame_h=480,
    )
    # 6px / 100px = 0.06, well into "blink" territory.
    assert _ear(lm := closed, _LEFT_EYE, 640, 480) < 0.10
    del lm


def test_ear_independent_of_eye_indexed():
    # Same geometry plugged into the right eye's indices should match the left.
    outer_l, inner_l, top_l, bot_l = _LEFT_EYE
    outer_r, inner_r, top_r, bot_r = _RIGHT_EYE
    base = {outer_l: (100, 250), inner_l: (200, 250), top_l: (150, 235), bot_l: (150, 265)}
    base.update({outer_r: (100, 250), inner_r: (200, 250), top_r: (150, 235), bot_r: (150, 265)})
    lm = _make_lm(base, frame_w=640, frame_h=480)
    assert _ear(lm, _LEFT_EYE, 640, 480) == _ear(lm, _RIGHT_EYE, 640, 480)


def test_ear_pixel_units_correct_for_non_square_frame():
    # If we computed in normalized coords the answer would differ between
    # 640x480 and 1280x720 frames for the same pixel geometry. The pixel
    # conversion inside _ear must cancel this — same physical lid opening,
    # same EAR.
    outer, inner, top, bot = _LEFT_EYE
    lm_a = _make_lm(
        {outer: (100, 250), inner: (200, 250), top: (150, 235), bot: (150, 265)},
        frame_w=640, frame_h=480,
    )
    lm_b = _make_lm(
        {outer: (100, 250), inner: (200, 250), top: (150, 235), bot: (150, 265)},
        frame_w=1280, frame_h=720,
    )
    assert abs(_ear(lm_a, _LEFT_EYE, 640, 480) - _ear(lm_b, _LEFT_EYE, 1280, 720)) < 1e-6


def test_ear_zero_horiz_returns_zero():
    # Degenerate: outer and inner corners at the same x. Avoid divide-by-zero;
    # return 0 so the gate naturally rejects the frame.
    outer, inner, top, bot = _LEFT_EYE
    lm = _make_lm(
        {outer: (150, 250), inner: (150, 250), top: (150, 235), bot: (150, 265)},
        frame_w=640, frame_h=480,
    )
    assert _ear(lm, _LEFT_EYE, 640, 480) == 0.0


# --- head_pose_yxz_degrees ------------------------------------------------


def _ry(deg: float) -> np.ndarray:
    """Rotation matrix about Y (yaw)."""
    r = math.radians(deg)
    return np.array(
        [
            [math.cos(r), 0, math.sin(r)],
            [0, 1, 0],
            [-math.sin(r), 0, math.cos(r)],
        ]
    )


def _rx(deg: float) -> np.ndarray:
    """Rotation matrix about X (pitch)."""
    r = math.radians(deg)
    return np.array(
        [
            [1, 0, 0],
            [0, math.cos(r), -math.sin(r)],
            [0, math.sin(r), math.cos(r)],
        ]
    )


def _rz(deg: float) -> np.ndarray:
    """Rotation matrix about Z (roll)."""
    r = math.radians(deg)
    return np.array(
        [
            [math.cos(r), -math.sin(r), 0],
            [math.sin(r), math.cos(r), 0],
            [0, 0, 1],
        ]
    )


def test_head_pose_identity_is_zero():
    yaw, pitch, roll = head_pose_yxz_degrees(np.eye(3))
    assert abs(yaw) < 1e-9
    assert abs(pitch) < 1e-9
    assert abs(roll) < 1e-9


def test_head_pose_pure_yaw_recovered():
    yaw, pitch, roll = head_pose_yxz_degrees(_ry(30.0))
    assert abs(yaw - 30.0) < 1e-9
    assert abs(pitch) < 1e-9
    assert abs(roll) < 1e-9


def test_head_pose_pure_pitch_recovered():
    yaw, pitch, roll = head_pose_yxz_degrees(_rx(20.0))
    assert abs(yaw) < 1e-9
    assert abs(pitch - 20.0) < 1e-9
    assert abs(roll) < 1e-9


def test_head_pose_pure_roll_recovered():
    yaw, pitch, roll = head_pose_yxz_degrees(_rz(15.0))
    assert abs(yaw) < 1e-9
    assert abs(pitch) < 1e-9
    assert abs(roll - 15.0) < 1e-9


def test_head_pose_composite_yxz_round_trip():
    # Build a known YXZ rotation and decompose; should round-trip.
    R = _ry(12.0) @ _rx(-7.0) @ _rz(4.0)
    yaw, pitch, roll = head_pose_yxz_degrees(R)
    assert abs(yaw - 12.0) < 1e-6
    assert abs(pitch - (-7.0)) < 1e-6
    assert abs(roll - 4.0) < 1e-6


def test_head_pose_accepts_4x4_transformation_matrix():
    # MediaPipe surfaces 4x4 matrices; decomposition should ignore translation.
    R = _ry(25.0)
    M = np.eye(4)
    M[:3, :3] = R
    M[:3, 3] = [10.0, -20.0, 30.0]  # arbitrary translation, must be ignored
    yaw, pitch, roll = head_pose_yxz_degrees(M)
    assert abs(yaw - 25.0) < 1e-9
    assert abs(pitch) < 1e-9
    assert abs(roll) < 1e-9


def test_head_pose_clamps_asin_arg_for_numerical_drift():
    # A rotation matrix with a tiny float overflow on R[1,2] would otherwise
    # crash asin. The clamp keeps it within domain.
    R = _rx(90.0)
    R[1, 2] = -1.0 - 1e-12  # nudge below -1 by float noise
    yaw, pitch, roll = head_pose_yxz_degrees(R)  # must not raise
    assert abs(pitch - 90.0) < 1e-3


def test_head_pose_rejects_wrong_shape():
    with pytest.raises(ValueError):
        head_pose_yxz_degrees(np.zeros((2, 2)))


# --- head_pose_max_dev_deg ------------------------------------------------


def test_head_pose_max_dev_zero_when_identical():
    assert head_pose_max_dev_deg((1.0, -2.0, 3.0), (1.0, -2.0, 3.0)) == 0.0


def test_head_pose_max_dev_picks_largest_axis_diff():
    # Differences: 4, 9, 1 -> max 9
    assert head_pose_max_dev_deg((10.0, 1.0, 4.0), (6.0, -8.0, 5.0)) == 9.0


def test_head_pose_max_dev_uses_absolute_value():
    # Whether current is above or below baseline, the magnitude is what gates.
    assert head_pose_max_dev_deg((-5.0, 0.0, 0.0), (10.0, 0.0, 0.0)) == 15.0
    assert head_pose_max_dev_deg((10.0, 0.0, 0.0), (-5.0, 0.0, 0.0)) == 15.0
