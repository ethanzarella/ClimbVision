import cv2
import numpy as np
from reference_wall_tracker import ReferenceWallTracker
from wall_tracker import transform_holds


def scene():
    rng = np.random.default_rng(33)
    img = np.full((400, 300, 3), 185, np.uint8)
    for i in range(500):
        x, y = rng.integers(3, 297), rng.integers(3, 397)
        cv2.circle(img, (int(x), int(y)), int(rng.integers(1, 4)),
                   tuple(int(v) for v in rng.integers(0, 170, size=3)), -1)
    return img


def test_translation_with_foreground_occlusion():
    first = scene()
    holds = [{'hold_id': 'H13', 'x': 125, 'y': 120, 'w': 30, 'h': 20}]
    tracker = ReferenceWallTracker(first, holds=holds, min_inliers=12)
    frame = cv2.warpAffine(first, np.float32([[1, 0, 12], [0, 1, 8]]), (300, 400),
                           borderMode=cv2.BORDER_REFLECT)
    cv2.rectangle(frame, (95, 95), (170, 230), (0, 0, 0), -1)
    H = tracker.estimate(frame, 1)
    assert H is not None
    box = transform_holds(holds, H, frame.shape)[0]
    assert abs(box['x'] - 137) <= 4 and abs(box['y'] - 128) <= 4
    assert abs(box['w'] - 30) <= 4


def test_bad_frame_is_not_accepted():
    tracker = ReferenceWallTracker(scene())
    H = tracker.estimate(np.zeros((400, 300, 3), np.uint8), 1)
    assert H is not None  # Short, explicitly uncertain recovery window.
    assert tracker.status == 'uncertain'
    assert tracker.failure_reason


def test_keyframe_recovers_cumulative_motion():
    first = scene()
    tracker = ReferenceWallTracker(first, keyframe_interval=5)
    for i in range(1, 13):
        moved = cv2.warpAffine(first, np.float32([[1, 0, i*2], [0, 1, i]]),
                               (300, 400), borderMode=cv2.BORDER_REFLECT)
        assert tracker.estimate(moved, i) is not None
    assert tracker.last_key_index >= 5


def test_recovery_expires_and_never_counts_as_valid():
    tracker = ReferenceWallTracker(scene(), recovery_frames=2)
    blank = np.zeros((400, 300, 3), np.uint8)
    assert tracker.estimate(blank, 1) is not None
    assert tracker.status == 'uncertain'
    assert tracker.estimate(blank, 2) is not None
    assert tracker.estimate(blank, 3) is None
    assert tracker.status == 'lost'


def test_clustered_correspondences_rejected():
    tracker = ReferenceWallTracker(scene())
    rng = np.random.default_rng(1)
    src = rng.uniform([20, 20], [30, 35], size=(50, 2)).astype(np.float32).reshape(-1, 1, 2)
    dst = src + np.array([3, 4], dtype=np.float32)
    assert tracker._fit_points(src, dst) is None


def test_diagnostics_report_spatial_evidence():
    first = scene()
    tracker = ReferenceWallTracker(first)
    moved = cv2.warpAffine(first, np.float32([[1, 0, 6], [0, 1, 3]]),
                           (300, 400), borderMode=cv2.BORDER_REFLECT)
    assert tracker.estimate(moved, 1) is not None
    assert tracker.status == 'valid'
    assert tracker.spatial_coverage > .055
    assert tracker.inlier_ratio > .43
    assert tracker.reprojection_error < 2.8
