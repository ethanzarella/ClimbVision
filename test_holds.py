import unittest
import numpy as np
import cv2
from holds import hex_to_hsv, segment_color, detect_holds, proximity_rows

class HoldTests(unittest.TestCase):
    def test_red_wrap(self):
        img = np.zeros((130, 200, 3), dtype=np.uint8)
        img[20:55, 30:65] = (0, 0, 255)
        mask = segment_color(img, '#ff1515')
        self.assertGreater(cv2.countNonZero(mask), 900)
        self.assertEqual(int(mask[100, 100]), 0)

    def test_detect_one_hold(self):
        img = np.zeros((160, 160, 3), dtype=np.uint8)
        cv2.circle(img, (60, 60), 13, (0, 0, 255), -1)
        found = detect_holds(img, (0, 0, 160, 160), '#ff0000', min_area=50)
        self.assertEqual(len(found), 1)
        self.assertAlmostEqual(found[0]['cx'], 60, delta=1)

    def test_near_is_unconfirmed(self):
        h = [{'hold_id': 'H01', 'cx': 30., 'cy': 40.}]
        rows = proximity_rows(2.0, {15: (.3, .4)}, h, 100, 100)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]['status'], 'near (unconfirmed)')
        self.assertEqual(proximity_rows(2.0, {15: (.9, .9)}, h, 100, 100), [])

    def test_invalid_color(self):
        with self.assertRaises(ValueError):
            hex_to_hsv('red')

if __name__ == '__main__':
    unittest.main()

class SampleColorTests(unittest.TestCase):
    def test_sample_rgb_hex(self):
        from holds import sample_color
        img = np.full((11, 11, 3), (140, 80, 110), dtype=np.uint8)
        self.assertEqual(sample_color(img, 5, 5), '#6e508c')
