import unittest
from roi_controls import region_from_controls

class RegionControlTests(unittest.TestCase):
    def test_default_in_bounds(self):
        l,t,r,b = region_from_controls(75,75,50,5)
        self.assertTrue(0 <= l < r <= 1)
        self.assertTrue(0 <= t < b <= 1)
    def test_larger_height_extends_region(self):
        a = region_from_controls(70,30,50,50)
        b = region_from_controls(70,70,50,50)
        self.assertLess(b[1], a[1]); self.assertGreater(b[3], a[3])
    def test_larger_width_extends_region(self):
        a = region_from_controls(30,70,50,50)
        b = region_from_controls(70,70,50,50)
        self.assertLess(b[0], a[0]); self.assertGreater(b[2], a[2])
    def test_move_down(self):
        a = region_from_controls(70,70,50,0)
        b = region_from_controls(70,70,50,100)
        self.assertGreater(b[1],a[1])
    def test_full_frame(self):
        self.assertEqual(region_from_controls(100,100,99,99),(0.,0.,1.,1.))

if __name__ == '__main__': unittest.main()
