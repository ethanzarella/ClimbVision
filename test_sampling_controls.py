import ast
import unittest
from pathlib import Path

class IndependentFPSRegression(unittest.TestCase):
    def test_sam_sampling_uses_independent_fps(self):
        tree = ast.parse(Path('analysis.py').read_text())
        calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == 'sam_track_video']
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].args[2].id, 'sam_fps')
    def test_video_writer_uses_original_fps(self):
        tree = ast.parse(Path('analysis.py').read_text())
        calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == 'VideoWriter']
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].args[2].id, 'fps')
    def test_slider_is_independent(self):
        src = Path('app.py').read_text()
        self.assertIn("st.slider('Pose analysis FPS'", src)
        self.assertIn("st.slider('SAM diagnostic FPS'", src)
        self.assertIn("hold_settings['sam_fps']", src)

if __name__ == '__main__':
    unittest.main()
