import unittest
from hybrid_tracker import reject_inconsistent_anchors

def hold(x,y,w=20,h=20):return dict(x=x,y=y,w=w,h=h)
class Guard(unittest.TestCase):
 def test_single_jumping_hold_rejected(self):
  before={f'H{i}':hold(i*40,50) for i in range(5)}
  after={f'H{i}':hold(i*40+3,51) for i in range(5)}
  after['H4']=hold(500,400)
  clean=reject_inconsistent_anchors({0:before,15:after})
  self.assertNotIn('H4',clean[15]);self.assertIn('H3',clean[15])
 def test_growth_rejected(self):
  clean=reject_inconsistent_anchors({0:{'H1':hold(40,40)},10:{'H1':hold(40,40,100,100)}})
  self.assertNotIn('H1',clean[10])
 def test_consistent_pan_allowed(self):
  a={f'H{i}':hold(i*40,50) for i in range(5)}
  b={f'H{i}':hold(i*40+40,80) for i in range(5)}
  self.assertEqual(len(reject_inconsistent_anchors({0:a,15:b})[15]),5)
if __name__=='__main__': unittest.main()
