import pytest

from gaze_tracker.snap import Target, TargetSnap, grid_targets


def _t(cx: float, cy: float, w: int = 100, h: int = 100, id: str = "") -> Target:
    return Target(bbox=(int(cx - w / 2), int(cy - h / 2), int(cx + w / 2), int(cy + h / 2)), id=id)


def test_passthrough_with_no_targets():
    snap = TargetSnap()
    out = snap((500.0, 500.0), [])
    assert out.xy == (500.0, 500.0)
    assert out.locked is None
    assert out.weight == 0.0


def test_passes_through_outside_attractor():
    snap = TargetSnap(attractor_radius=120, lock_radius=50)
    target = _t(500, 500, id="a")
    # Far outside attractor radius
    out = snap((1000.0, 500.0), [target])
    assert out.xy == (1000.0, 500.0)
    assert out.locked is None
    assert out.weight == 0.0


def test_locks_inside_lock_radius():
    snap = TargetSnap(attractor_radius=120, lock_radius=50)
    target = _t(500, 500, id="a")
    out = snap((520.0, 500.0), [target])  # 20px from center
    assert out.xy == target.center
    assert out.locked is target
    assert out.weight == 1.0


def test_blends_in_attractor_zone():
    snap = TargetSnap(attractor_radius=120, lock_radius=50)
    target = _t(500, 500, id="a")
    # 85px from center: midway between lock (50) and attractor (120) → w ≈ 0.5
    out = snap((585.0, 500.0), [target])
    assert out.locked is None
    assert 0.4 < out.weight < 0.6
    # Output should be between raw and target center on the x-axis
    assert 500.0 < out.xy[0] < 585.0
    assert out.xy[1] == pytest.approx(500.0)


def test_blend_weight_decreases_with_distance():
    snap_a = TargetSnap(attractor_radius=120, lock_radius=50)
    snap_b = TargetSnap(attractor_radius=120, lock_radius=50)
    target = _t(500, 500, id="a")
    near = snap_a((560.0, 500.0), [target])  # 60px out
    far = snap_b((110.0 + 500.0, 500.0), [target])  # 110px out
    assert near.weight > far.weight


def test_picks_nearest_of_multiple_targets():
    snap = TargetSnap(attractor_radius=300, lock_radius=50)
    a = _t(200, 500, id="a")
    b = _t(800, 500, id="b")
    out = snap((780.0, 500.0), [a, b])
    assert out.locked is b
    assert out.xy == b.center


def test_hysteresis_holds_lock_in_unlock_band():
    # lock at 40, unlock at 80 — once locked, stay locked between 40 and 80.
    snap = TargetSnap(attractor_radius=200, lock_radius=40, unlock_radius=80)
    a = _t(500, 500, id="a")
    snap((510.0, 500.0), [a])  # acquire lock
    out = snap((570.0, 500.0), [a])  # 70px out — past lock_radius but within unlock
    assert out.locked is a
    assert out.xy == a.center
    assert out.weight == 1.0


def test_hysteresis_releases_outside_unlock():
    snap = TargetSnap(attractor_radius=200, lock_radius=40, unlock_radius=80)
    a = _t(500, 500, id="a")
    snap((510.0, 500.0), [a])  # acquire lock
    out = snap((620.0, 500.0), [a])  # 120px out — outside unlock_radius
    assert out.locked is None


def test_hysteresis_prevents_neighbor_flicker():
    # Two targets 200px apart. Acquire lock on A. A small wobble toward B but
    # within A's unlock radius should NOT flip the lock to B.
    snap = TargetSnap(attractor_radius=200, lock_radius=40, unlock_radius=80)
    a = _t(400, 500, id="a")
    b = _t(600, 500, id="b")
    snap((400.0, 500.0), [a, b])  # lock A
    out = snap((460.0, 500.0), [a, b])  # 60px from A (within A's unlock), 140px from B
    assert out.locked is a


def test_invalid_radii_raise():
    with pytest.raises(ValueError):
        TargetSnap(attractor_radius=100, lock_radius=50, unlock_radius=40)
    with pytest.raises(ValueError):
        TargetSnap(attractor_radius=30, lock_radius=50)


def test_grid_targets_count_and_layout():
    targets = grid_targets(screen_w=1920, screen_h=1080, cols=3, rows=3)
    assert len(targets) == 9
    # All centers must be inside the screen
    for t in targets:
        cx, cy = t.center
        assert 0 < cx < 1920
        assert 0 < cy < 1080
    # Distinct ids
    assert len({t.id for t in targets}) == 9


def test_grid_targets_single_cell():
    targets = grid_targets(screen_w=1920, screen_h=1080, cols=1, rows=1)
    assert len(targets) == 1
    cx, cy = targets[0].center
    assert cx == pytest.approx(1920 / 2, abs=1)
    assert cy == pytest.approx(1080 / 2, abs=1)


def test_reset_clears_lock():
    snap = TargetSnap(attractor_radius=200, lock_radius=40, unlock_radius=80)
    a = _t(500, 500, id="a")
    snap((510.0, 500.0), [a])
    assert snap._locked is a  # lock acquired
    snap.reset()
    assert snap._locked is None
