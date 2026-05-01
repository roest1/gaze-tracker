import csv
import math

import pytest

from gaze_tracker.calibration import GRID_POINTS_NORM
from gaze_tracker.evaluate import (
    CSV_FIELDNAMES,
    DEFAULT_N_COLS,
    DEFAULT_N_ROWS,
    EvalReport,
    append_eval_log,
    calibration_hash,
    make_eval_points,
    pixel_error_to_degrees,
    seed_from_hash,
)

# --- make_eval_points -----------------------------------------------------


def test_make_eval_points_default_count_is_n_cols_times_n_rows():
    pts = make_eval_points(seed=0)
    assert len(pts) == DEFAULT_N_COLS * DEFAULT_N_ROWS


def test_make_eval_points_reproducible_with_same_seed():
    a = make_eval_points(seed=42)
    b = make_eval_points(seed=42)
    assert a == b


def test_make_eval_points_differs_across_seeds():
    a = make_eval_points(seed=1)
    b = make_eval_points(seed=2)
    assert a != b


def test_make_eval_points_inside_unit_square_with_margin():
    pts = make_eval_points(seed=0, margin=0.20)
    for x, y in pts:
        assert 0.20 <= x <= 0.80
        assert 0.20 <= y <= 0.80


def test_make_eval_points_avoids_calibration_grid_under_default_layout():
    # Default 4x4 cell centers ([0.20, 0.40, 0.60, 0.80]) are by construction
    # >= 0.10 from any cal grid point ({0.1, 0.5, 0.9}). With default jitter
    # +/-0.04, no eval point should land within 0.05 of any cal point.
    pts = make_eval_points(seed=7, calibration_points=list(GRID_POINTS_NORM))
    for x, y in pts:
        for cx, cy in GRID_POINTS_NORM:
            dist = math.hypot(x - cx, y - cy)
            assert dist > 0.05, (
                f"eval point ({x:.3f}, {y:.3f}) too close to cal ({cx}, {cy})"
            )


def test_make_eval_points_avoids_calibration_grid_across_many_seeds():
    # Confirm the geometric guarantee holds for arbitrary seeds, not just one.
    for seed in range(50):
        pts = make_eval_points(seed=seed, calibration_points=list(GRID_POINTS_NORM))
        for x, y in pts:
            for cx, cy in GRID_POINTS_NORM:
                assert math.hypot(x - cx, y - cy) > 0.05


def test_make_eval_points_falls_back_when_rejection_impossible():
    # If we ask for a single point at a position that overlaps a cal point and
    # jitter is zero, the function should still return one point (fallback after
    # 8 retries) rather than hang or error.
    pts = make_eval_points(
        seed=0, n_cols=1, n_rows=1, jitter=0.0,
        calibration_points=[(0.5, 0.5)],
    )
    assert len(pts) == 1


def test_make_eval_points_rejects_invalid_dims():
    with pytest.raises(ValueError):
        make_eval_points(seed=0, n_cols=0)
    with pytest.raises(ValueError):
        make_eval_points(seed=0, n_rows=0)


def test_make_eval_points_rejects_invalid_jitter_or_margin():
    with pytest.raises(ValueError):
        make_eval_points(seed=0, jitter=-0.01)
    with pytest.raises(ValueError):
        make_eval_points(seed=0, margin=-0.01)
    with pytest.raises(ValueError):
        make_eval_points(seed=0, margin=0.5)


# --- pixel_error_to_degrees ------------------------------------------------


def test_pixel_error_to_degrees_zero_error_is_zero_degrees():
    assert pixel_error_to_degrees(0.0, dpi=96.0, face_distance_cm=50.0) == 0.0


def test_pixel_error_to_degrees_known_value():
    # 138 DPI, 50 cm distance: 1 px = (1/138) inch = 0.00725 inch = 0.0001841 m.
    # arctan(0.0001841 / 0.50) = arctan(0.000368) rad = 0.0211 deg.
    deg = pixel_error_to_degrees(1.0, dpi=138.0, face_distance_cm=50.0)
    assert abs(deg - 0.0211) < 1e-3

    # Sanity at 138 DPI, 50 cm: 1 deg ≈ 47 px (tan(1deg) * 0.5 m / 0.0254 * 138).
    # So 47 px -> ~1 deg.
    deg_one = pixel_error_to_degrees(47.0, dpi=138.0, face_distance_cm=50.0)
    assert 0.95 < deg_one < 1.05


def test_pixel_error_to_degrees_invalid_inputs_raise():
    with pytest.raises(ValueError):
        pixel_error_to_degrees(10.0, dpi=0.0, face_distance_cm=50.0)
    with pytest.raises(ValueError):
        pixel_error_to_degrees(10.0, dpi=-1.0, face_distance_cm=50.0)
    with pytest.raises(ValueError):
        pixel_error_to_degrees(10.0, dpi=96.0, face_distance_cm=0.0)


def test_pixel_error_to_degrees_doubles_with_double_distance_proxy():
    # At small angles, doubling face distance ~halves the angle for the same px.
    # (Only approximate — small-angle regime.)
    a = pixel_error_to_degrees(50.0, dpi=138.0, face_distance_cm=50.0)
    b = pixel_error_to_degrees(50.0, dpi=138.0, face_distance_cm=100.0)
    assert 0.45 < (b / a) < 0.55


# --- calibration_hash + seed_from_hash ------------------------------------


def test_calibration_hash_stable_for_same_content(tmp_path):
    p = tmp_path / "cal.json"
    p.write_text('{"some": "calibration"}')
    h1 = calibration_hash(p)
    h2 = calibration_hash(p)
    assert h1 == h2
    assert len(h1) == 16  # 16 hex chars


def test_calibration_hash_changes_with_content(tmp_path):
    a = tmp_path / "a.json"
    b = tmp_path / "b.json"
    a.write_text('{"x": 1}')
    b.write_text('{"x": 2}')
    assert calibration_hash(a) != calibration_hash(b)


def test_seed_from_hash_deterministic():
    # 8 hex chars -> int. Same input -> same int.
    assert seed_from_hash("deadbeef00000000") == 0xdeadbeef
    assert seed_from_hash("deadbeefcafebabe") == seed_from_hash("deadbeefcafebabe")


# --- EvalReport + CSV log -------------------------------------------------


def _sample_report(**overrides):
    base = dict(
        n_points=15,
        n_ear_dropped=3,
        n_pose_dropped=2,
        median_px=80.0,
        p95_px=180.0,
        rmse_px=110.0,
        median_deg=1.7,
        p95_deg=3.8,
        rmse_deg=2.3,
        monitor_dpi=138.0,
        face_distance_cm=50.0,
        calibration_hash="abcdef1234567890",
        timestamp_iso="2026-04-26T18:00:00+00:00",
    )
    base.update(overrides)
    return EvalReport(**base)


def test_eval_report_stdout_includes_both_drop_counts():
    s = _sample_report().stdout()
    assert "n=15" in s
    # Both gates surfaced separately in the output.
    assert "ear=3" in s
    assert "pose=2" in s
    assert "abcdef1234567890" in s
    assert "138" in s


def test_append_eval_log_creates_file_with_header(tmp_path):
    p = tmp_path / "eval_log.csv"
    append_eval_log(_sample_report(), path=p)
    assert p.exists()
    rows = list(csv.DictReader(p.open()))
    assert len(rows) == 1
    assert set(rows[0].keys()) == set(CSV_FIELDNAMES)
    assert rows[0]["n_points"] == "15"
    assert rows[0]["n_ear_dropped"] == "3"
    assert rows[0]["n_pose_dropped"] == "2"
    assert rows[0]["calibration_hash"] == "abcdef1234567890"


def test_append_eval_log_appends_without_duplicating_header(tmp_path):
    p = tmp_path / "eval_log.csv"
    append_eval_log(_sample_report(n_points=10), path=p)
    append_eval_log(_sample_report(n_points=12), path=p)
    rows = list(csv.DictReader(p.open()))
    assert len(rows) == 2
    assert rows[0]["n_points"] == "10"
    assert rows[1]["n_points"] == "12"
    # The DictReader treats the first non-header line as data; if a header
    # was duplicated it'd appear as a row with literal "n_points" in n_points.
    for r in rows:
        assert r["n_points"] != "n_points"


def test_append_eval_log_creates_parent_dirs(tmp_path):
    p = tmp_path / "deeper" / "subdir" / "eval_log.csv"
    append_eval_log(_sample_report(), path=p)
    assert p.exists()


def test_append_eval_log_rotates_on_schema_change(tmp_path):
    # Pre-pose-gate eval_log files have a different column shape. Loading them
    # and appending a row with the new schema would mix incompatible rows in
    # one CSV. Confirm the old file is rotated to .bak and a fresh log is made.
    p = tmp_path / "eval_log.csv"
    old_headers = [
        "timestamp_iso", "calibration_hash", "n_points", "n_dropped",
        "monitor_dpi", "face_distance_cm",
        "median_px", "p95_px", "rmse_px",
        "median_deg", "p95_deg", "rmse_deg",
    ]
    p.write_text(
        ",".join(old_headers) + "\n"
        "2026-01-01T00:00:00+00:00,deadbeef,15,5,138,50,100,200,150,2.0,4.0,3.0\n"
    )
    append_eval_log(_sample_report(), path=p)
    backup = p.with_suffix(p.suffix + ".bak")
    assert backup.exists()
    rows = list(csv.DictReader(p.open()))
    assert len(rows) == 1
    assert "n_pose_dropped" in rows[0]
    assert "n_dropped" not in rows[0]


def test_append_eval_log_does_not_rotate_on_matching_schema(tmp_path):
    p = tmp_path / "eval_log.csv"
    append_eval_log(_sample_report(n_points=10), path=p)
    append_eval_log(_sample_report(n_points=12), path=p)
    # No .bak should have been created — schemas matched both times.
    assert not p.with_suffix(p.suffix + ".bak").exists()
