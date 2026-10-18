import unittest
import cv2
import numpy as np
from wall_tracker import WallTracker, transform_holds


def test_project_boxes_translate():
    H=np.array([[1,0,25],[0,1,12],[0,0,1]],float)
    out=transform_holds([dict(x=20,y=30,w=40,h=20,hold_id='H01')],H,(300,400,3))
    assert len(out)==1 and (out[0]['x'],out[0]['y'])==(45,42)
    assert out[0]['hold_id']=='H01'


def test_hide_when_alignment_lost():
    assert transform_holds([dict(x=10,y=10,w=15,h=15)],None,(200,200,3))==[]


def test_orb_estimates_wall_shift():
    rng=np.random.default_rng(42)
    frame=rng.integers(0,255,(480,640,3),dtype=np.uint8)
    # Blur to create repeatable textured background with corners.
    frame=cv2.GaussianBlur(frame,(5,5),0)
    moved=cv2.warpAffine(frame,np.float32([[1,0,16],[0,1,9]]),(640,480))
    tracker=WallTracker(frame)
    assert tracker.estimate(moved) is not None
    out=transform_holds([dict(x=150,y=150,w=50,h=50)],tracker.last_transform,moved.shape)
    assert abs(out[0]['x']-166)<=3 and abs(out[0]['y']-159)<=3


class TestTrackingDiagnosticsCompatibility(unittest.TestCase):
    def test_legacy_tracker_exposes_export_diagnostics(self):
        import numpy as np
        import cv2
        from wall_tracker import WallTracker
        rng = np.random.default_rng(41)
        frame = rng.integers(0, 256, (256, 384, 3), dtype=np.uint8)
        for validate_motion in (True, False):
            tracker = WallTracker(frame, validate_motion=validate_motion)
            self.assertEqual(tracker.source, 'original_reference')
            self.assertEqual(tracker.confidence, 0.0)
            result = tracker.estimate(frame)
            self.assertIsNotNone(result)
            self.assertGreaterEqual(tracker.confidence, 0.0)
            self.assertLessEqual(tracker.confidence, 1.0)
            self.assertTrue(tracker.source)

    def test_failed_registration_resets_confidence(self):
        import numpy as np
        from wall_tracker import WallTracker
        rng = np.random.default_rng(42)
        frame = rng.integers(0, 256, (256, 384, 3), dtype=np.uint8)
        tracker = WallTracker(frame)
        self.assertIsNotNone(tracker.estimate(frame))
        self.assertIsNone(tracker.estimate(np.zeros_like(frame)))
        self.assertEqual(tracker.confidence, 0.0)
