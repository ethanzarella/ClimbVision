import unittest
import numpy as np
from holds import normalize_reviewed_holds, suggested_box, remove_hold_near, reindex_holds

class ReviewedTests(unittest.TestCase):
    def test_scale(self):
        boxes=normalize_reviewed_holds([dict(hold_id='H01',x=100,y=50,w=40,h=30)],1920,1080)
        self.assertEqual((boxes[0]['x'],boxes[0]['y'],boxes[0]['w'],boxes[0]['h']), (50,25,20,15))
    def test_add_box(self):
        frame=np.zeros((200,200,3),dtype=np.uint8)
        frame[50:80,60:100] = (180,50,110)
        box=suggested_box(frame,75,65,'#6e32b4',radius=30)
        self.assertTrue(box['x']<=75<=box['x']+box['w'])
        self.assertTrue(box['y']<=65<=box['y']+box['h'])
    def test_remove(self):
        boxes=reindex_holds([dict(x=10,y=20,w=30,h=40),dict(x=80,y=80,w=10,h=10)])
        remain,removed=remove_hold_near(boxes,20,30)
        self.assertTrue(removed)
        self.assertEqual(len(remain),1)
        self.assertEqual(remain[0]['x'],80)
    def test_empty(self):
        self.assertEqual(normalize_reviewed_holds([],1920,1080), [])
if __name__ == '__main__': unittest.main()
