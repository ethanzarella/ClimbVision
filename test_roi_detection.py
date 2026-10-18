import unittest
from unittest.mock import patch
import numpy as np
from roboflow_detector import crop_detection_frame, restore_prediction_coordinates, detect_with_roboflow

class RegionDetectionTests(unittest.TestCase):
    def test_crop_and_offset(self):
        img = np.zeros((100,200,3),dtype=np.uint8)
        img[25:75,40:110] = (12,34,56)
        crop, pos = crop_detection_frame(img,(40,25,110,75))
        self.assertEqual(crop.shape,(50,70,3))
        self.assertEqual(pos,(40,25))
        self.assertEqual(tuple(crop[0,0]),(12,34,56))

    def test_restore(self):
        p = restore_prediction_coordinates([{'x':3,'y':4,'w':10,'h':12,'confidence':.8}],(40,25))
        self.assertEqual((p[0]['x'],p[0]['y'],p[0]['w'],p[0]['h']),(43,29,10,12))

    def test_invalid_roi(self):
        with self.assertRaises(ValueError):
            crop_detection_frame(np.zeros((20,20,3),dtype=np.uint8),(1,1,3,3))

    def test_inference_sees_crop_only(self):
        import sys, types
        received=[]
        class Client:
            def __init__(self, **kwargs): pass
            def infer(self, image, model_id):
                received.append(image.copy())
                return {'image':{'width':60,'height':40},'predictions':[{'class':'hold','confidence':.9,'x':30,'y':20,'width':20,'height':10}]}
        sdk=types.SimpleNamespace(InferenceHTTPClient=Client)
        with patch.dict(sys.modules, {'inference_sdk':sdk}):
            results=detect_with_roboflow(np.zeros((100,150,3),dtype=np.uint8),'key',roi=(20,30,80,70))
        self.assertEqual(received[0].shape,(40,60,3))
        self.assertEqual((results[0]['x'],results[0]['y'],results[0]['w'],results[0]['h']),(40,45,20,10))

if __name__=='__main__': unittest.main()
