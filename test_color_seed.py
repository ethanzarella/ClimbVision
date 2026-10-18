import unittest
import cv2
import numpy as np
from holds import sample_color_robust, hex_to_hsv

class SeedTest(unittest.TestCase):
    def test_seed_stays_red_with_large_tan_background(self):
        img = np.zeros((100,100,3), dtype=np.uint8)
        img[:]=(130,180,215) # tan wall
        img[43:58,43:58]=(20,55,230) # orange-red plastic
        sample=sample_color_robust(img,50,50,radius=22)
        expected=int(cv2.cvtColor(np.uint8([[[20,55,230]]]), cv2.COLOR_BGR2HSV)[0,0,0])
        measured=hex_to_hsv(sample)[0]
        self.assertLessEqual(min(abs(expected-measured),180-abs(expected-measured)),4)

    def test_hue_wrap(self):
        img=np.zeros((90,90,3),np.uint8)
        img[:]=(210,190,160)
        img[35:55,35:55]=(5,5,245)
        hue=hex_to_hsv(sample_color_robust(img,45,45,radius=15))[0]
        self.assertLessEqual(min(hue,180-hue),4)
