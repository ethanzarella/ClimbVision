import unittest
import numpy as np
from holds import annotate_holds

class TestHoldAnnotationFloat(unittest.TestCase):
    def test_float_boxes_draw(self):
        frame = np.zeros((100, 160, 3), dtype=np.uint8)
        annotate_holds(frame, [{'x': 10.5, 'y': 20.3, 'w': 30.4, 'h': 18.8, 'hold_id': 'H04'}])
        self.assertGreater(int(frame.sum()), 0)

    def test_bad_and_outside_boxes_skip(self):
        frame = np.zeros((100, 160, 3), dtype=np.uint8)
        annotate_holds(frame, [
            {'x': float('nan'), 'y': 2, 'w': 10, 'h': 10, 'hold_id': 'bad'},
            {'x': 1000.5, 'y': 1000.5, 'w': 5.2, 'h': 5.2, 'hold_id': 'off'},
            {'x': 10, 'y': 10, 'w': -3, 'h': 3, 'hold_id': 'neg'},
            {'x': -5.4, 'y': 40.1, 'w': 22.3, 'h': 18.8, 'hold_id': 'partial'},
        ])
        self.assertGreater(int(frame.sum()), 0)

if __name__ == '__main__': unittest.main()
