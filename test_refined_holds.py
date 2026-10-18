import unittest
import cv2
import numpy as np
from holds import detect_holds

class RefinedHoldTests(unittest.TestCase):
    def test_rejects_long_strip_and_edge_blob(self):
        img = np.zeros((240, 360, 3), np.uint8)
        cv2.circle(img, (110, 90), 17, (0, 255, 255), -1)
        cv2.rectangle(img, (25, 170), (325, 173), (0, 255, 255), -1)
        cv2.rectangle(img, (0, 50), (10, 90), (0, 255, 255), -1)
        items = detect_holds(img, (0, 0, 360, 240), '#ffff00', min_area=50)
        self.assertEqual(len(items), 1)
        self.assertAlmostEqual(items[0]['cx'], 110, delta=3)

    def test_advanced_filter_can_be_disabled(self):
        img = np.zeros((100, 120, 3), np.uint8)
        cv2.rectangle(img, (0, 30), (15, 43), (255, 0, 0), -1)
        self.assertEqual(detect_holds(img, (0, 0, 120, 100), '#0000ff', min_area=30), [])
        self.assertEqual(len(detect_holds(img, (0, 0, 120, 100), '#0000ff', min_area=30, exclude_boundary=False)), 1)
