import numpy as np
import cv2
from wall_tracker import WallTracker


def _reference():
    rng = np.random.default_rng(2026)
    return cv2.GaussianBlur(rng.integers(0, 255, (360, 480, 3), dtype=np.uint8), (3, 3), 0)


def test_fast_preview_rejects_large_jump_without_mutating_anchor():
    base = _reference()
    tracker = WallTracker(base, validate_motion=True)
    moved = cv2.warpAffine(base, np.float32([[1, 0, 115], [0, 1, 0]]), (480, 360))
    assert tracker.estimate(moved) is None
    assert np.allclose(tracker.last_transform, np.eye(3))


def test_fast_preview_keeps_normal_small_camera_motion():
    base = _reference()
    tracker = WallTracker(base, validate_motion=True)
    moved = cv2.warpAffine(base, np.float32([[1, 0, 7], [0, 1, 5]]), (480, 360))
    assert tracker.estimate(moved) is not None


def test_legacy_does_not_enable_new_guard():
    assert not WallTracker(_reference()).validate_motion
