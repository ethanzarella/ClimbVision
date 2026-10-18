import unittest
import numpy as np
import pandas as pd
from analysis import detect_motion_events, measure

PARTS = ('left_hand', 'right_hand', 'left_foot', 'right_foot')
def sample(t, x, detected=True):
    row = {'time_s': t, 'detected': detected, 'hip_x': .5, 'hip_y': .4}
    for p in PARTS:
        row[p+'_x'] = x if p == 'left_hand' else .5
        row[p+'_y'] = .4
    return row

class EventsTest(unittest.TestCase):
    def test_coordinates(self):
        m=measure({15:(.2,.3),31:(.7,.8)})
        self.assertAlmostEqual(m['left_hand_x'],.2)
        self.assertAlmostEqual(m['left_foot_y'],.8)
        self.assertTrue(np.isnan(m['right_hand_x']))
    def test_still_generates_pause(self):
        data=pd.DataFrame([sample(i*.2,.5) for i in range(12)])
        metrics, events=detect_motion_events(data)
        self.assertTrue((events.event=='Low-motion interval').any())
        self.assertTrue(metrics['left_hand_speed'].iloc[1:].eq(0).all())
    def test_motion(self):
        data=pd.DataFrame([sample(i*.2,.2 + .05*i) for i in range(12)])
        _, events=detect_motion_events(data)
        self.assertTrue(((events.event=='Hand movement') & (events.body_part=='left_hand')).any())
    def test_gap_not_bridged(self):
        data=pd.DataFrame([sample(i*.2, .1 if i < 5 else .9, detected=(i!=5)) for i in range(11)])
        metrics, _=detect_motion_events(data)
        self.assertTrue(np.isnan(metrics.at[5,'left_hand_speed']))
        self.assertTrue(np.isnan(metrics.at[6,'left_hand_speed']))
    def test_no_false_contact(self):
        data=pd.DataFrame([sample(i*.2,.5) for i in range(8)])
        _, events=detect_motion_events(data)
        self.assertFalse(events.event.str.contains('contact|hold', case=False).any())

if __name__=='__main__': unittest.main()
