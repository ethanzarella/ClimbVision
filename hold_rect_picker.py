"""Browser drag-to-draw hold annotation with full-resolution image coordinates."""
from pathlib import Path
import base64
import cv2


def validate_action(value, width, height):
    if not isinstance(value, dict):
        return None
    action = value.get('action')
    if action == 'add':
        b = value.get('box')
        if not isinstance(b, dict):
            return None
        try:
            x, y, w, h = (int(b[k]) for k in ('x', 'y', 'w', 'h'))
        except (KeyError, ValueError, TypeError):
            return None
        if x < 0 or y < 0 or w < 4 or h < 4 or x + w > width or y + h > height:
            return None
        return {'action': 'add', 'box': {'x': x, 'y': y, 'w': w, 'h': h}}
    if action == 'remove':
        try:
            x, y = int(value['x']), int(value['y'])
        except (KeyError, ValueError, TypeError):
            return None
        if not (0 <= x < width and 0 <= y < height):
            return None
        return {'action': 'remove', 'x': x, 'y': y}
    return None


def draw_hold_rectangle(frame_bgr, holds, mode, key):
    height, width = frame_bgr.shape[:2]
    max_width = 950
    displayed = frame_bgr
    if width > max_width:
        displayed = cv2.resize(frame_bgr, (max_width, max(1, round(height * max_width / width))))
    ok, jpg = cv2.imencode('.jpg', displayed, [cv2.IMWRITE_JPEG_QUALITY, 89])
    if not ok:
        return None
    source = 'data:image/jpeg;base64,' + base64.b64encode(jpg.tobytes()).decode('ascii')
    import streamlit.components.v1 as components
    picker = components.declare_component('climbvision_hold_rect', path=str(Path(__file__).parent / 'hold_rect_ui'))
    raw = picker(image=source, holds=holds, originalWidth=width, originalHeight=height,
                 mode=mode, key=key, default=None)
    return validate_action(raw, width, height)
