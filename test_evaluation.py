import unittest
import pandas as pd
from evaluation import iou, evaluate_boxes, read_ground_truth
class EvaluationTests(unittest.TestCase):
    def test_iou(self):
        self.assertEqual(iou(dict(x=0,y=0,w=10,h=10),dict(x=0,y=0,w=10,h=10)),1)
        self.assertEqual(iou(dict(x=0,y=0,w=10,h=10),dict(x=40,y=40,w=10,h=10)),0)
    def test_match_one_to_one(self):
        p=pd.DataFrame([dict(x=0,y=0,w=10,h=10)]*2)
        g=pd.DataFrame([dict(x=0,y=0,w=10,h=10)])
        r=evaluate_boxes(p,g)
        self.assertEqual((r['tp'],r['fp'],r['fn']),(1,1,0))
    def test_reject_bad_csv(self):
        with self.assertRaises(ValueError): read_ground_truth(b'x,y,w,h\n0,0,-1,5\n')
if __name__=='__main__': unittest.main()
