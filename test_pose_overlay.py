import unittest
from types import SimpleNamespace
from pose_overlay import PoseOverlay

class TestPoseOverlay(unittest.TestCase):
    def point(self,x,y,confidence=0.9):
        return SimpleNamespace(x=x,y=y,visibility=confidence,presence=confidence)

    def test_continuous_between_samples(self):
        o=PoseOverlay()
        o.update([self.point(.2,.3)],0.0)
        self.assertIn(0,o.visible(.1))

    def test_old_landmark_expires(self):
        o=PoseOverlay()
        o.update([self.point(.2,.3)],0.0)
        self.assertNotIn(0,o.visible(.5))

    def test_low_confidence_does_not_replace_good_position(self):
        o=PoseOverlay()
        o.update([self.point(.2,.3)],0.0)
        o.update([self.point(.8,.8,.1)],.1)
        self.assertAlmostEqual(o.visible(.1)[0][0],.2)

    def test_smooths_jumps(self):
        o=PoseOverlay(smoothing=.5)
        o.update([self.point(.2,.3)],0)
        o.update([self.point(.8,.3)],.1)
        self.assertLess(o.visible(.1)[0][0],.8)
        self.assertGreater(o.visible(.1)[0][0],.2)

if __name__ == '__main__': unittest.main()
