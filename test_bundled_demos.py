from pathlib import Path
import ast
import unittest

class BundledDemoTests(unittest.TestCase):
    def test_app_syntax(self):
        root = Path(__file__).parent
        ast.parse((root / 'app.py').read_text())
    def test_both_examples_exist(self):
        root = Path(__file__).parent / 'demo_videos'
        for filename in ('attempt_a.mp4', 'attempt_b.mp4'):
            file = root / filename
            self.assertTrue(file.is_file(), filename)
            self.assertGreater(file.stat().st_size, 10000)
    def test_no_hardcoded_deployment_url(self):
        content = (Path(__file__).parent / 'CLOUD_DEPLOYMENT.md').read_text()
        self.assertIn('actual URL', content)

if __name__ == '__main__':
    unittest.main()
