from pathlib import Path

def test_color_picker_has_only_one_still_frame_widget():
    source = Path(__file__).with_name('app.py').read_text()
    segment = source.split("st.markdown('**Click a route hold in the first video frame to sample its color**')")[1].split('hue_tolerance = st.slider')[0]
    assert segment.count('streamlit_image_coordinates(') == 1
    assert 'st.image(' not in segment
    assert 'cv2.drawMarker' in segment
