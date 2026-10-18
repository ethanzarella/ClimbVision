import unittest
from roi_picker import valid_region

class DragRegionTests(unittest.TestCase):
    def test_valid(self):
        self.assertEqual(valid_region([.1,.2,.8,.9]),(.1,.2,.8,.9))
    def test_reject_outside(self):
        self.assertEqual(valid_region([-.1,0,.8,.9], (0,0,1,1)),(0,0,1,1))
    def test_reject_tiny(self):
        self.assertEqual(valid_region([.1,.1,.12,.5], (0,0,1,1)),(0,0,1,1))
    def test_reject_string(self):
        self.assertEqual(valid_region('0.1,0.2,0.8,0.9',(0,0,1,1)),(0,0,1,1))

if __name__=='__main__': unittest.main()
