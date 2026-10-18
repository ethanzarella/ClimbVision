import unittest
from pathlib import Path
from manual_holds import manual_hold_box

class TestManualWorkflow(unittest.TestCase):
    def test_center(self):
        self.assertEqual(manual_hold_box(100, 100, 50, 50, 10), {'x':40,'y':40,'w':21,'h':21})
    def test_edge(self):
        self.assertEqual(manual_hold_box(100, 100, 0, 0, 10), {'x':0,'y':0,'w':11,'h':11})
    def test_small_frame(self):
        box = manual_hold_box(10, 9, 5, 5, 99)
        self.assertEqual((box['w'], box['h']), (10,9))
    def test_no_auto_detection_ui(self):
        text=Path(__file__).with_name('app.py').read_text()
        for forbidden in ('Generate trained hold suggestions', 'Generate color-based suggestions', 'Click a route hold in the first video frame to sample its color', 'Current capability: watch and analyze each attempt separately.'):
            self.assertNotIn(forbidden, text)
        self.assertIn('draw_hold_rectangle(original_frame', text)
