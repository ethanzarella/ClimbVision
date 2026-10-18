"""Lightweight optical-flow bridging between sparse SAM 2 anchor detections.

Hold coordinates are estimates, not confirmed contact or segmentations.
"""
from collections import defaultdict
import cv2
import numpy as np


def identify(holds):
    return {str(h.get('hold_id', h.get('id', h.get('label', f'H{i+1:02d}')))): dict(h) for i, h in enumerate(holds)}


def optical_displacement(prev_gray, gray, hold):
    """Median Lucas-Kanade flow within and around a hold, or None when unreliable."""
    h, w = prev_gray.shape
    cx = float(hold['x']) + float(hold['w']) / 2
    cy = float(hold['y']) + float(hold['h']) / 2
    radius = max(12, int(max(hold['w'], hold['h']) * 0.85))
    x1, x2 = max(0, int(cx-radius)), min(w, int(cx+radius))
    y1, y2 = max(0, int(cy-radius)), min(h, int(cy+radius))
    if x2-x1 < 6 or y2-y1 < 6:
        return None
    mask = np.zeros_like(prev_gray, np.uint8)
    mask[y1:y2, x1:x2] = 255
    pts = cv2.goodFeaturesToTrack(prev_gray, maxCorners=45, qualityLevel=0.015,
                                 minDistance=4, mask=mask)
    if pts is None or len(pts) < 3:
        return None
    new, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, pts, None,
                                              winSize=(21, 21), maxLevel=3)
    if new is None or status is None:
        return None
    ok = status.ravel() == 1
    if int(ok.sum()) < 3:
        return None
    displacements = (new[ok] - pts[ok]).reshape(-1, 2)
    median = np.median(displacements, axis=0)
    err = np.linalg.norm(displacements-median, axis=1)
    if np.median(err) > 5.5 or np.linalg.norm(median) > max(w, h)*0.12:
        return None
    return tuple(float(v) for v in median)


def bridge_anchors(start_frame, end_frame, start_hold, end_hold, increments):
    """Correct integrated flow drift gradually to meet the next SAM anchor exactly.

    increments maps frame index -> (dx,dy), None denotes untrusted optical flow.
    Returns positions for indices [start_frame, end_frame] or none if unreliable.
    """
    if end_frame <= start_frame:
        raise ValueError('Anchor order must be increasing')
    n = end_frame-start_frame
    misses = sum(increments.get(t) is None for t in range(start_frame+1, end_frame+1))
    if misses > max(2, int(0.3*n)):
        return {}
    shifts = np.zeros((n+1, 2), float)
    for i, t in enumerate(range(start_frame+1, end_frame+1), 1):
        increment = increments.get(t)
        shifts[i] = shifts[i-1] + (np.asarray(increment) if increment is not None else 0)
    a = np.array([start_hold['x'], start_hold['y']], float)
    b = np.array([end_hold['x'], end_hold['y']], float)
    residual = (b-a)-shifts[-1]
    output = {}
    for i, frame_idx in enumerate(range(start_frame, end_frame+1)):
        t = i / n
        xy = a+shifts[i]+residual*t
        bw = (1-t)*start_hold['w']+t*end_hold['w']
        bh = (1-t)*start_hold['h']+t*end_hold['h']
        obj = dict(start_hold)
        obj.update(x=float(xy[0]), y=float(xy[1]), w=float(bw), h=float(bh),
                   cx=float(xy[0]+bw/2), cy=float(xy[1]+bh/2),
                   tracking_source='sam_anchor' if i in (0, n) else 'optical_flow_bridge')
        output[frame_idx] = obj
    return output


def reject_inconsistent_anchors(anchors, max_area_ratio=2.5, max_residual_px=35):
    """Reject per-hold SAM anchor jumps inconsistent with surrounding stationary holds.

    Uses the median displacement of other visible holds as a scene-motion reference.
    If too few reference holds exist, use conservative area/size screening only.
    Returns a new anchor mapping; it does not mutate raw segmentation results.
    """
    import math
    cleaned = {t: {k: dict(v) for k, v in data.items()} for t, data in anchors.items()}
    times = sorted(cleaned)
    for prev_t, now_t in zip(times[:-1], times[1:]):
        before, after = cleaned[prev_t], cleaned[now_t]
        common = before.keys() & after.keys()
        shifts = {}
        for k in common:
            a,b = before[k],after[k]
            aw,ah,bw,bh = [float(v) for v in (a['w'],a['h'],b['w'],b['h'])]
            if min(aw,ah,bw,bh) < 1:
                continue
            ac = np.array((float(a['x'])+aw/2,float(a['y'])+ah/2))
            bc = np.array((float(b['x'])+bw/2,float(b['y'])+bh/2))
            shifts[k] = bc-ac
        for k in common:
            if k not in shifts:
                after.pop(k,None)
                continue
            a,b=before[k],after[k]
            ar = float(b['w'])*float(b['h'])/(float(a['w'])*float(a['h']))
            if not math.isfinite(ar) or not (1/max_area_ratio <= ar <= max_area_ratio):
                after.pop(k,None)
                continue
            others = [v for name,v in shifts.items() if name != k]
            if len(others) >= 3:
                expected = np.median(others,axis=0)
                if np.linalg.norm(shifts[k]-expected) > max_residual_px:
                    after.pop(k,None)
    return cleaned


def generate_hybrid_tracks(video_path, size, sam_frames, initial_holds, max_frames):
    """Read native frames once to estimate local flow, constrain each segment by SAM anchors.

    Never invent labels missing at both anchors. No boxes when a segment fails validation.
    """
    if not sam_frames:
        return {}
    anchors = reject_inconsistent_anchors({int(f): identify(hs) for f, hs in sam_frames.items()})
    ids = identify(initial_holds)
    frames = sorted(anchors)
    if not frames:
        return {}
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError('Cannot read input for hybrid tracking')
    increments = defaultdict(dict)
    prev = None
    estimates = dict(ids)
    try:
        for idx in range(min(max_frames, frames[-1]+1)):
            ok, image = cap.read()
            if not ok:
                break
            image = cv2.resize(image, size)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            if prev is not None:
                for name, hold in list(estimates.items()):
                    delta = optical_displacement(prev, gray, hold)
                    increments[name][idx] = delta
                    if delta is not None:
                        nxt = dict(hold)
                        nxt['x'] += delta[0]; nxt['y'] += delta[1]
                        estimates[name] = nxt
            if idx in anchors:
                estimates.update(anchors[idx])
            prev = gray
    finally:
        cap.release()
    result = defaultdict(list)
    for frame_a, frame_b in zip(frames[:-1], frames[1:]):
        for name in anchors[frame_a].keys() & anchors[frame_b].keys():
            segment = bridge_anchors(frame_a, frame_b, anchors[frame_a][name],
                                     anchors[frame_b][name], increments[name])
            for frame_idx, obj in segment.items():
                if frame_idx < frame_b:
                    result[frame_idx].append(obj)
    for name, obj in anchors[frames[-1]].items():
        result[frames[-1]].append({**obj, 'tracking_source':'sam_anchor'})
    if len(frames) == 1:
        return {frames[0]:result[frames[0]]}
    return dict(result)
