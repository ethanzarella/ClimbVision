import unittest
import numpy as np
from hybrid_tracker import identify, bridge_anchors, optical_displacement

class HybridTests(unittest.TestCase):
    def test_ids(self):
        self.assertEqual(list(identify([{'hold_id':'H04','x':3}])), ['H04'])
    def test_anchors_and_bridge(self):
        a = {'hold_id':'H04', 'x':10, 'y':10,'w':12,'h':8}
        b = {'hold_id':'H04', 'x':30, 'y':14,'w':12,'h':8}
        result = bridge_anchors(0,4,a,b,{1:(3,1),2:(3,1),3:(3,1),4:(3,1)})
        self.assertEqual(result[0]['x'], 10)
        self.assertEqual(result[4]['x'], 30)
        self.assertTrue(all(result[t]['x'] < result[t+1]['x'] for t in range(4)))
    def test_missing_flow(self):
        a={'x':1,'y':1,'w':10,'h':10}
        self.assertEqual(bridge_anchors(0,5,a,a,{}), {})
    def test_optical_flow_translation(self):
        import cv2
        a=np.zeros((150,150),np.uint8)
        rng=np.random.default_rng(5)
        a[45:95,45:95]=rng.integers(0,255,(50,50),dtype=np.uint8)
        m=np.float32([[1,0,4],[0,1,3]])
        b=cv2.warpAffine(a,m,(150,150))
        d=optical_displacement(a,b,{'x':45,'y':45,'w':50,'h':50})
        self.assertIsNotNone(d)
        self.assertAlmostEqual(d[0],4,delta=1)
        self.assertAlmostEqual(d[1],3,delta=1)

if __name__=='__main__': unittest.main()
