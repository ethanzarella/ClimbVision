"""Browser drag/resize ROI picker with normalized coordinates."""
from pathlib import Path
import base64
import cv2


def valid_region(value, fallback=(0.125, 0.0125, 0.875, 0.7625)):
    """Accept normalized [left,top,right,bottom]; reject malformed browser values."""
    if not isinstance(value, (tuple, list)) or len(value) != 4:
        return tuple(fallback)
    try:
        l, t, r, b = (float(v) for v in value)
    except (ValueError, TypeError):
        return tuple(fallback)
    if not (0 <= l < r <= 1 and 0 <= t < b <= 1 and r-l >= .06 and b-t >= .06):
        return tuple(fallback)
    return (l, t, r, b)


def draggable_region(frame_bgr, initial_region, key):
    """Draw/move/resize crop directly over the frame. Return normalized crop."""
    h, w = frame_bgr.shape[:2]
    max_width = 960
    if w > max_width:
        frame_bgr = cv2.resize(frame_bgr, (max_width, max(1, round(h * max_width / w))))
    success, encoded = cv2.imencode('.jpg', frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 88])
    if not success:
        return tuple(initial_region)
    image = 'data:image/jpeg;base64,' + base64.b64encode(encoded.tobytes()).decode('ascii')
    import streamlit.components.v1 as components
    component = components.declare_component('climbvision_roi_drag', path=str(Path(__file__).parent / 'roi_picker_ui'))
    result = component(image=image, region=list(initial_region), key=key, default=list(initial_region))
    return valid_region(result, initial_region)
