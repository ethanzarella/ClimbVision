import unittest
from hold_rect_picker import validate_action

class HoldRectangleTests(unittest.TestCase):
    def test_add_rectangle(self):
        op = {'action':'add','box':{'x':20,'y':30,'w':80,'h':19}}
        self.assertEqual(validate_action(op, 200, 100), op)
    def test_off_frame_rejected(self):
        self.assertIsNone(validate_action({'action':'add','box':{'x':195,'y':30,'w':80,'h':19}},200,100))
    def test_tiny_drag_rejected(self):
        self.assertIsNone(validate_action({'action':'add','box':{'x':20,'y':30,'w':2,'h':19}},200,100))
    def test_remove(self):
        self.assertEqual(validate_action({'action':'remove','x':10,'y':5},100,100),{'action':'remove','x':10,'y':5})
    def test_bad_action(self):
        self.assertIsNone(validate_action({'action':'all'},100,100))

if __name__=='__main__':unittest.main()
