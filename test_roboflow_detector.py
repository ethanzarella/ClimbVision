import unittest
import numpy as np
import cv2
from roboflow_detector import normalize_predictions, filter_predictions_by_color, detect_with_roboflow

class TestRoboflowDetector(unittest.TestCase):
    def test_coordinate_conversion_and_tag_exclusion(self):
        data = {'image': {'width': 100, 'height': 50}, 'predictions': [
            {'x': 50,'y': 25,'width': 20,'height': 10,'confidence':.9,'class':'Hold'},
            {'x': 25,'y': 10,'width': 10,'height': 10,'confidence':.99,'class':'Tag'}]}
        output=normalize_predictions(data,(100,200,3))
        self.assertEqual(len(output),1)
        self.assertEqual(tuple(output[0][k] for k in ('x','y','w','h')),(80,40,40,20))
    def test_roi_and_confidence(self):
        data={'predictions':[{'x':20,'y':20,'width':12,'height':12,'confidence':.6,'class':'Hold'},
                             {'x':80,'y':80,'width':12,'height':12,'confidence':.2,'class':'Hold'}]}
        self.assertEqual(len(normalize_predictions(data,(100,100,3),.4,(0,0,40,40))),1)
    def test_invalid_boxes_are_ignored(self):
        data={'predictions':[{'x':float('nan'),'y':5,'width':20,'height':20,'confidence':.9,'class':'Hold'}]}
        self.assertEqual(normalize_predictions(data,(100,100,3)),[])
    def test_color_filter_only_within_boxes(self):
        frame = np.zeros((100,100,3),np.uint8)
        frame[10:30,10:30]=(0,255,255) # yellow
        frame[40:60,40:60]=(255,0,0)   # blue
        preds=[{'x':10,'y':10,'w':20,'h':20},{'x':40,'y':40,'w':20,'h':20}]
        filtered=filter_predictions_by_color(frame,preds,'#ffff00')
        self.assertEqual(len(filtered),1)
        self.assertEqual(filtered[0]['x'],10)
    def test_missing_api_key_fails_before_import(self):
        with self.assertRaisesRegex(ValueError,'API key'):
            detect_with_roboflow(np.zeros((10,10,3),np.uint8),'')

if __name__=='__main__': unittest.main()
