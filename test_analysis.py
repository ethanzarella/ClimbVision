from analysis import angle, measure, make_feedback
import pandas as pd
import math

def test_right_angle():
    assert abs(angle((1,0),(0,0),(0,1))-90)<1e-9

def test_degenerate():
    assert math.isnan(angle((0,0),(0,0),(1,1)))

def test_missing_points():
    assert math.isnan(measure({})['hip_x'])

def test_low_coverage():
    df = pd.DataFrame([{'detected':False}] * 5)
    assert 'unreliable' in make_feedback(df)[0]
