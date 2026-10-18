import unittest
from pose_backend import COCO_TO_MP, Landmark

import math
class TestPoseMapping(unittest.TestCase):
    def test_anatomical_mapping(self):
        self.assertEqual(COCO_TO_MP[9],15)
        self.assertEqual(COCO_TO_MP[10],16)
        self.assertEqual(COCO_TO_MP[15],27)
        self.assertNotIn(31, COCO_TO_MP.values())
        self.assertNotIn(32, COCO_TO_MP.values())
    def test_unavailable_toes_not_fabricated(self):
        points=[Landmark() for _ in range(33)]
        points[27]=Landmark(.4,.8,.9,.9)
        points[28]=Landmark(.6,.8,.9,.9)
        self.assertEqual(points[31].visibility,0)
        self.assertEqual(points[32].visibility,0)
        self.assertEqual(points[27].visibility,.9)
if __name__=='__main__': unittest.main()
