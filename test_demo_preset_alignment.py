import json
import unittest
from pathlib import Path

class DemoPresetsTest(unittest.TestCase):
    def test_same_output_sizes_and_presets(self):
        root=Path(__file__).parent / 'demo_videos'
        for filename in ('attempt_a_holds.json', 'attempt_b_holds.json'):
            data=json.loads((root / filename).read_text())
            self.assertEqual((data['frame_width'],data['frame_height']),(720,1280))
            self.assertEqual(len(data['holds']),10)
            self.assertTrue(all(0<=h['x']<720 and 0<=h['y']<1280 and h['w']>0 and h['h']>0 for h in data['holds']))
