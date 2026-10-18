import unittest
import numpy as np
from holds import sample_color_robust, hex_to_hsv
from roboflow_detector import matching_color_fraction

class RouteColorTest(unittest.TestCase):
    def test_background_does_not_dominate_trained_box(self):
        img = np.full((100, 100, 3), (170, 200, 210), np.uint8)  # beige BGR
        img[30:70, 25:75] = (0, 70, 235)  # orange-red
        self.assertGreater(matching_color_fraction(img, '#f05020', 12, 65), .65)
        self.assertLess(matching_color_fraction(img, '#2040ef', 12, 65), .1)

    def test_single_noise_spot_does_not_match(self):
        img = np.full((100, 100, 3), (170, 200, 210), np.uint8)
        img[49:52, 49:52] = (0, 70, 235)
        self.assertEqual(matching_color_fraction(img, '#f05020', 12, 65), 0)

    def test_robust_sample_shaded_red_remains_red(self):
        img=np.full((80,80,3),(160,195,205),np.uint8)
        img[26:54,26:54]=(20,40,175)
        img[32:40,32:40]=(16,60,230)
        col=sample_color_robust(img,36,36,radius=17)
        hue=hex_to_hsv(col)[0]
        self.assertTrue(hue < 15 or hue > 170, col)

if __name__=='__main__': unittest.main()
