import unittest
from analysis import crop_bounds, remap_landmarks, valid_pose

class Fake:
    def __init__(self, x, y, visibility=1., presence=1.):
        self.x=x; self.y=y; self.visibility=visibility; self.presence=presence

class RoiTests(unittest.TestCase):
    def test_crop(self):
        self.assertEqual(crop_bounds(1000, 1000, (.2,.1,.8,.75)), (200,100,800,750))
    def test_bad_crop(self):
        with self.assertRaises(ValueError):
            crop_bounds(1000,1000,(.8,.2,.2,.9))
    def test_remap(self):
        p = remap_landmarks([Fake(.5,.5)], (200,100,800,700), 1000,1000)[0]
        self.assertAlmostEqual(p.x,.5)
        self.assertAlmostEqual(p.y,.4)
    def test_false_flat_pose(self):
        landmarks=[Fake(.5,.95) for _ in range(33)]
        self.assertFalse(valid_pose(landmarks))
    def test_plausible_pose(self):
        landmarks=[Fake(.5,.5) for _ in range(33)]
        for i,x,y in [(11,.4,.3),(12,.6,.3),(23,.43,.6),(24,.57,.6)]:
            landmarks[i] = Fake(x,y)
        self.assertTrue(valid_pose(landmarks))

if __name__ == '__main__': unittest.main()
