import tempfile
import unittest
from pathlib import Path
import pandas as pd
from result_cache import fingerprint, save_result, load_result, clear_cache

class CacheTests(unittest.TestCase):
    def result(self):
        result = {'video_bytes': b'video', 'info': {'fps': 30}, 'converted': True,
                  'conversion_warning': None, 'hold_enabled': True,
                  'source_hash': 'test', 'source_name': 'a.mp4', 'model_name': 'Lite',
                  'sample_fps': 5, 'processing_seconds_limit': 10}
        for name in ('df', 'metrics', 'events', 'holds_df', 'near_df', 'wall_df'):
            result[name] = pd.DataFrame({'x': [1, 2], 'label': ['a', 'b']})
        return result

    def test_roundtrip(self):
        with tempfile.TemporaryDirectory() as folder:
            key = fingerprint(b'a', 'Lite', 5, [0,0,1,1], None, 10)
            save_result(folder, key, self.result())
            item = load_result(folder, key)
            self.assertEqual(item['video_bytes'], b'video')
            self.assertEqual(item['info']['fps'], 30)
            self.assertEqual(item['df'].iloc[1]['label'], 'b')
            clear_cache(folder)
            self.assertIsNone(load_result(folder, key))

    def test_cache_keys_change_with_inputs(self):
        base = fingerprint(b'a', 'Lite', 5, [0,0,1,1], None, 10)
        self.assertNotEqual(base, fingerprint(b'b', 'Lite', 5, [0,0,1,1], None, 10))
        self.assertNotEqual(base, fingerprint(b'a', 'Lite', 10, [0,0,1,1], None, 10))
        self.assertNotEqual(base, fingerprint(b'a', 'Lite', 5, [0,0,1,1], {'reviewed_holds': []}, 10))

    def test_incomplete_is_miss(self):
        with tempfile.TemporaryDirectory() as folder:
            self.assertIsNone(load_result(folder, 'bad'))

if __name__ == '__main__':
    unittest.main()
