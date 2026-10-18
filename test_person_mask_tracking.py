import cv2
import numpy as np
from reference_wall_tracker import ReferenceWallTracker


def texture():
    rng = np.random.default_rng(87)
    frame = np.full((420, 320, 3), 170, dtype=np.uint8)
    for _ in range(1300):
        x, y = rng.integers(2, 318), rng.integers(2, 418)
        color = tuple(int(v) for v in rng.integers(30, 200, size=3))
        cv2.circle(frame, (int(x), int(y)), 2, color, -1)
    return frame


def test_person_rectangle_does_not_supply_orb_features():
    img = texture()
    box = (70, 80, 210, 330)
    t = ReferenceWallTracker(img, person_bounds=box)
    assert np.count_nonzero(t.wall_mask[80:331, 70:211]) == 0
    assert all(not (70 <= p.pt[0] <= 210 and 80 <= p.pt[1] <= 330) for p in t.ref_kp)


def test_person_mask_keeps_wall_registration_valid():
    first = texture()
    t = ReferenceWallTracker(first, person_bounds=(100, 150, 180, 320))
    next_frame = cv2.warpAffine(first, np.float32([[1,0,8], [0,1,5]]), (320,420))
    cv2.rectangle(next_frame, (116, 155), (195, 315), (20, 25, 25), -1)
    H = t.estimate(next_frame, 1, person_bounds=(116, 155, 195, 315))
    assert H is not None and t.status == 'valid'
    actual = cv2.perspectiveTransform(np.float32([[[70,80]]]), H).reshape(2)
    assert np.linalg.norm(actual - [78,85]) < 5
