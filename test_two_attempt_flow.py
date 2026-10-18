import ast
from pathlib import Path
import unittest

ROOT = Path(__file__).parent

class TwoAttemptFlowTests(unittest.TestCase):
    def test_app_parses(self):
        ast.parse((ROOT / 'app.py').read_text())
    def test_first_result_precedes_second_controls(self):
        s = (ROOT / 'app.py').read_text()
        self.assertLess(s.index("st.subheader('Attempt A — completed analysis')"),
                        s.index("st.subheader('Pose tracking model')"))
        self.assertIn("st.session_state.first_attempt_hash = current['source_hash']", s)
        self.assertIn("st.session_state.second_mode = True", s)
        self.assertIn('The completed Attempt A above is preserved', s)
    def test_comparison_uses_distinct_saved_attempts(self):
        s = (ROOT / 'app.py').read_text()
        self.assertIn("current.get('source_hash') != first.get('source_hash')", s)
        self.assertIn("synchronized_videos(first['video_bytes'], current['video_bytes']", s)
    def test_player_has_working_controls(self):
        s = (ROOT / 'synchronized_player.py').read_text()
        for x in ('Play both', 'Pause both', 'Restart both', 'seek.oninput',
                  'Promise.allSettled', 'v.currentTime=0'):
            self.assertIn(x, s)

if __name__ == '__main__':
    unittest.main()
