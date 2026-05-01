"""MediaPipe Face Landmarker wrapper that extracts a 3D gaze direction feature.

For each eye we compute the 3D offset from the eye center (mean of four corner
landmarks) to the iris center, using MediaPipe's z-bearing landmarks. That
offset *is* the gaze direction in camera coordinates: when the head moves but
gaze stays on a fixed world point, both the eye center and the iris translate
or rotate together, and the 3D difference vector between them stays (approximately)
constant. This makes the feature robust to head pose without ever computing
Euler angles or inverting the head transformation explicitly.

We average the two eyes' vectors, normalize, and return a 3-tuple (vx, vy, vz).
MediaPipe's axes: x and y are normalized image coords (y increases downward);
z is depth with smaller values closer to the camera. The image has been flipped
horizontally upstream so the signal matches what the user perceives.

First run downloads the face_landmarker.task model (~3 MB) to
`$XDG_CACHE_HOME/gaze-tracker/face_landmarker.task`.
"""
from __future__ import annotations

import math
import os
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

# Indices in the MediaPipe Face Landmarker topology (478-point model, includes iris).
_LEFT_EYE = (33, 133, 159, 145)   # outer, inner, top, bottom
_RIGHT_EYE = (362, 263, 386, 374)
_LEFT_IRIS = 468
_RIGHT_IRIS = 473

_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
)


def _model_path() -> Path:
    cache_home = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    return cache_home / "gaze-tracker" / "face_landmarker.task"


def _ensure_model() -> Path:
    p = _model_path()
    if not p.exists():
        p.parent.mkdir(parents=True, exist_ok=True)
        print(f"[gaze-tracker] downloading Face Landmarker model to {p}")
        urllib.request.urlretrieve(_MODEL_URL, p)
    return p


@dataclass(frozen=True)
class GazeFeatures:
    """Normalized 3D gaze direction (camera frame, both eyes averaged) plus
    per-eye EAR. EAR is exposed so calibration can gate out frames captured
    with anomalous lid aperture (squints, blinks).

    `head_pose` is YXZ Euler angles (yaw, pitch, roll) in degrees, derived
    from MediaPipe's facial transformation matrix. None when MediaPipe didn't
    return a matrix (older versions, or face not yet locked). Used by the
    realtime loop to refuse predictions when the head has drifted from the
    posture the calibration was taken in."""

    gaze: tuple[float, float, float]
    ear_left: float
    ear_right: float
    head_pose: tuple[float, float, float] | None = None


def head_pose_yxz_degrees(matrix: np.ndarray) -> tuple[float, float, float]:
    """Decompose a rotation matrix to YXZ Euler angles in degrees.

    Accepts a 3x3 rotation matrix or a 4x4 transformation matrix (uses the
    upper-left 3x3 in the latter case). Convention:
        R = R_y(yaw) @ R_x(pitch) @ R_z(roll)
    where:
        yaw   = rotation about the Y (vertical) axis    — head turning L/R
        pitch = rotation about the X (horizontal) axis  — head nodding U/D
        roll  = rotation about the Z (forward) axis     — head tilting sideways

    The asin argument is clamped to [-1, 1] so MediaPipe's tiny float drift
    doesn't blow up the decomposition with a domain error. Gimbal lock at
    pitch ≈ ±90° is unhandled — that's "head looking straight up/down" and
    isn't a realistic webcam pose.
    """
    R = np.asarray(matrix, dtype=float)
    if R.shape == (4, 4):
        R = R[:3, :3]
    if R.shape != (3, 3):
        raise ValueError(f"expected 3x3 or 4x4 matrix, got shape {R.shape}")
    sin_pitch = max(-1.0, min(1.0, float(-R[1, 2])))
    pitch_rad = math.asin(sin_pitch)
    yaw_rad = math.atan2(float(R[0, 2]), float(R[2, 2]))
    roll_rad = math.atan2(float(R[1, 0]), float(R[1, 1]))
    return (
        math.degrees(yaw_rad),
        math.degrees(pitch_rad),
        math.degrees(roll_rad),
    )


def head_pose_max_dev_deg(
    current: tuple[float, float, float],
    baseline: tuple[float, float, float],
) -> float:
    """Largest absolute axis-wise deviation in degrees. Used to gate the
    realtime tracking loop: above some threshold the model's calibration
    no longer applies, so predictions are misleading."""
    return max(abs(c - b) for c, b in zip(current, baseline, strict=True))


# Default deviation threshold for the head-pose gate. Shared across the
# realtime tracking loop and the eval flow so both reject the same frames.
HEAD_POSE_GATE_DEG = 15.0


class FaceMeshTracker:
    def __init__(self) -> None:
        options = mp_vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(_ensure_model())),
            running_mode=mp_vision.RunningMode.IMAGE,
            num_faces=1,
            output_facial_transformation_matrixes=True,
        )
        self._landmarker = mp_vision.FaceLandmarker.create_from_options(options)

    def close(self) -> None:
        self._landmarker.close()

    def __enter__(self) -> FaceMeshTracker:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def extract(self, rgb_frame: np.ndarray) -> GazeFeatures | None:
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        result = self._landmarker.detect(mp_image)
        if not result.face_landmarks:
            return None
        lm = result.face_landmarks[0]

        lx, ly, lz = _gaze_vec(lm, _LEFT_IRIS, _LEFT_EYE)
        rx, ry, rz = _gaze_vec(lm, _RIGHT_IRIS, _RIGHT_EYE)
        mx, my, mz = 0.5 * (lx + rx), 0.5 * (ly + ry), 0.5 * (lz + rz)
        norm = math.sqrt(mx * mx + my * my + mz * mz)
        if norm < 1e-6:
            return None

        h, w = rgb_frame.shape[:2]
        ear_l = _ear(lm, _LEFT_EYE, w, h)
        ear_r = _ear(lm, _RIGHT_EYE, w, h)

        # MediaPipe versions / configs that don't surface the matrix degrade
        # to head_pose=None — the realtime gate then short-circuits to "no
        # gating" and behavior matches pre-#5.
        head_pose: tuple[float, float, float] | None = None
        matrices = getattr(result, "facial_transformation_matrixes", None)
        if matrices:
            head_pose = head_pose_yxz_degrees(np.asarray(matrices[0]))

        return GazeFeatures(
            gaze=(mx / norm, my / norm, mz / norm),
            ear_left=ear_l,
            ear_right=ear_r,
            head_pose=head_pose,
        )


def _gaze_vec(
    lm,
    iris_idx: int,
    eye_corners: tuple[int, int, int, int],
) -> tuple[float, float, float]:
    ix, iy, iz = lm[iris_idx].x, lm[iris_idx].y, lm[iris_idx].z
    xs = [lm[i].x for i in eye_corners]
    ys = [lm[i].y for i in eye_corners]
    zs = [lm[i].z for i in eye_corners]
    cx = sum(xs) / len(xs)
    cy = sum(ys) / len(ys)
    cz = sum(zs) / len(zs)
    return (ix - cx, iy - cy, iz - cz)


def _ear(
    lm,
    eye_indices: tuple[int, int, int, int],
    frame_w: int,
    frame_h: int,
) -> float:
    """Soukupová & Čech eye-aspect-ratio. Convert MediaPipe normalized image
    coords back to pixel space first, otherwise non-square frames distort the
    ratio (lm.x scales by frame_w, lm.y by frame_h independently)."""
    outer, inner, top, bot = eye_indices
    vert = abs(lm[top].y - lm[bot].y) * frame_h
    horiz = abs(lm[outer].x - lm[inner].x) * frame_w
    if horiz < 1e-6:
        return 0.0
    return vert / horiz
