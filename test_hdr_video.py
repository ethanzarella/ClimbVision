import unittest
from hdr_video import is_hdr, probe_video, prepare_upload

class HDRVideoTests(unittest.TestCase):
    def test_hlg_detected(self):
        self.assertTrue(is_hdr({'color_transfer': 'arib-std-b67'}))
    def test_pq_detected(self):
        self.assertTrue(is_hdr({'color_transfer': 'smpte2084'}))
    def test_sdr_not_treated_as_hdr(self):
        self.assertFalse(is_hdr({'color_transfer': 'bt709'}))
    def test_missing_metadata_not_assumed_hdr(self):
        self.assertFalse(is_hdr({}))

if __name__ == '__main__':
    unittest.main()
