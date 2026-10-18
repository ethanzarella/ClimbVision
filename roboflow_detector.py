"""Optional Roboflow hosted trained hold detector; no API calls without a user action.

Only the user-selected first video frame is sent to the remote inference endpoint.
Predictions are proposals that must be reviewed before analysis.
"""
import cv2
import numpy as np
from holds import hex_to_hsv

BOULDERVISION_MODEL = 'climbing-route-1-hold-detection/6'


def normalize_predictions(response, frame_shape, threshold=0.35, roi=None):
    """Convert Roboflow center-based boxes to clipped pixel rectangles.

    Roboflow results can be returned at a resized inference resolution.
    """
    height, width = frame_shape[:2]
    image = response.get('image') or {}
    source_w = float(image.get('width') or width)
    source_h = float(image.get('height') or height)
    if source_w <= 0 or source_h <= 0:
        raise ValueError('Invalid Roboflow response image dimensions')
    sx, sy = width / source_w, height / source_h
    results = []
    for p in response.get('predictions', []):
        label = str(p.get('class', '')).strip().lower()
        if label not in ('hold', 'holds', 'climbing hold'):
            continue  # Model can also identify route tags.
        confidence = float(p.get('confidence', 0))
        if not np.isfinite(confidence) or confidence < threshold:
            continue
        try:
            x, y, bw, bh = (float(p[k]) for k in ('x', 'y', 'width', 'height'))
        except (ValueError, TypeError, KeyError):
            continue
        if not all(np.isfinite(v) for v in (x, y, bw, bh)) or bw <= 0 or bh <= 0:
            continue
        x1 = max(0, min(width, int(round((x - bw/2) * sx))))
        x2 = max(0, min(width, int(round((x + bw/2) * sx))))
        y1 = max(0, min(height, int(round((y - bh/2) * sy))))
        y2 = max(0, min(height, int(round((y + bh/2) * sy))))
        if x2 - x1 < 3 or y2 - y1 < 3:
            continue
        if roi is not None:
            rx1, ry1, rx2, ry2 = roi
            mx, my = (x1+x2)/2, (y1+y2)/2
            if not (rx1 <= mx <= rx2 and ry1 <= my <= ry2):
                continue
        results.append(dict(x=x1, y=y1, w=x2-x1, h=y2-y1,
                            confidence=confidence, model_class=label))
    return sorted(results, key=lambda h: (h['y'], h['x']))


def matching_color_fraction(bgr_crop, target_hex, hue_tolerance=20, min_saturation=35):
    """Fraction of sufficiently saturated hold-colored pixels within cropped detection."""
    if bgr_crop.size == 0:
        return 0.0
    hsv = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2HSV)
    center_hue = hex_to_hsv(target_hex)[0]
    hue_dist = np.abs(hsv[..., 0].astype(np.int16) - center_hue)
    hue_dist = np.minimum(hue_dist, 180-hue_dist)
    plausible = (hsv[..., 1] >= min_saturation) & (hsv[..., 2] >= 40)
    matched = plausible & (hue_dist <= hue_tolerance)
    # The detection box can contain a substantial amount of beige wall. Score
    # the route-color proportion among plausible colored pixels rather than
    # among all pixels, or large boxes always fail. Require a minimum number
    # of actual colored pixels so a few noisy points cannot pass.
    count = int(plausible.sum())
    if count < max(9, int(matched.size * .035)):
        return 0.0
    return float(matched.sum() / count)


def filter_predictions_by_color(frame, proposals, target_hex, min_match=0.14,
                                hue_tolerance=20, min_saturation=35):
    """Apply route-color checking ONLY to detected hold boxes, not the full wall."""
    keep = []
    for p in proposals:
        x, y, w, h = (int(p[k]) for k in ('x','y','w','h'))
        # Slightly inset each detected box to avoid color contamination from the wall.
        inset_x, inset_y = int(w*.08), int(h*.08)
        patch = frame[y+inset_y:y+h-inset_y, x+inset_x:x+w-inset_x]
        fraction = matching_color_fraction(patch, target_hex, hue_tolerance, min_saturation)
        if fraction >= min_match:
            keep.append({**p, 'color_match': round(fraction, 3)})
    return keep


def crop_detection_frame(frame_bgr, roi):
    """Crop actual inference input to ROI; return image and offset for full-frame mapping."""
    height, width = frame_bgr.shape[:2]
    if roi is None:
        return frame_bgr, (0, 0)
    x1, y1, x2, y2 = map(int, roi)
    x1, x2 = max(0, min(width, x1)), max(0, min(width, x2))
    y1, y2 = max(0, min(height, y1)), max(0, min(height, y2))
    if x2 - x1 < 8 or y2 - y1 < 8:
        raise ValueError('The selected tracking region is too small for hold detection.')
    return frame_bgr[y1:y2, x1:x2], (x1, y1)


def restore_prediction_coordinates(predictions, offset):
    """Map cropped-image proposals back to the full image for manual review."""
    ox, oy = offset
    return [{**p, 'x': p['x'] + ox, 'y': p['y'] + oy} for p in predictions]


def detect_with_roboflow(frame_bgr, api_key, model_id=BOULDERVISION_MODEL,
                         threshold=0.35, roi=None):
    """Use Roboflow hosted inference, requiring a key; fail loudly if unavailable."""
    if not api_key or not api_key.strip():
        raise ValueError('Enter a Roboflow API key before running trained detection.')
    try:
        from inference_sdk import InferenceHTTPClient
    except ImportError as exc:
        raise RuntimeError('Install the optional detector dependency: python -m pip install inference-sdk') from exc
    client = InferenceHTTPClient(api_url='https://serverless.roboflow.com', api_key=api_key.strip())
    # Send ONLY the user-selected region to the hosted model (not the full frame).
    cropped, offset = crop_detection_frame(frame_bgr, roi)
    response = client.infer(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB), model_id=model_id)
    predictions = normalize_predictions(response, cropped.shape, threshold)
    return restore_prediction_coordinates(predictions, offset)
