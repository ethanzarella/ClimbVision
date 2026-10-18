import unittest
import numpy as np
import cv2
import tempfile
from pathlib import Path
from sam_tracker import mask_to_hold, extract_sampled_frames

class SAMAdapterTests(unittest.TestCase):
    def test_mask_bbox_and_id(self):
        mask = np.zeros((80,100),dtype=bool)
        mask[20:45,30:50] = True
        result = mask_to_hold(mask, {'hold_id':'H04'}, (100,80))
        self.assertEqual((result['x'],result['y'],result['w'],result['h']),(30,20,20,25))
        self.assertEqual(result['hold_id'],'H04')
    def test_empty_returns_none(self):
        self.assertIsNone(mask_to_hold(np.zeros((10,10),bool), {'hold_id':'H01'}, (10,10)))
    def test_sampled_indices(self):
        with tempfile.TemporaryDirectory() as d:
            video = str(Path(d)/'x.mp4')
            writer = cv2.VideoWriter(video,cv2.VideoWriter_fourcc(*'mp4v'),10,(80,60))
            self.assertTrue(writer.isOpened())
            for i in range(20):
                writer.write(np.full((60,80,3),i,dtype=np.uint8))
            writer.release()
            folder = Path(d)/'frames'; folder.mkdir()
            indices = extract_sampled_frames(video,folder,(80,60),5)
            self.assertEqual(indices,list(range(0,20,2)))
            self.assertEqual(len(list(folder.glob('*.jpg'))),10)

if __name__ == '__main__': unittest.main()
