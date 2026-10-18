"""Video-based climbing pose tracking and descriptive motion metrics."""
from __future__ import annotations
import math
import time
from pathlib import Path
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from holds import background_from_video, detect_holds, annotate_holds, proximity_rows
from wall_tracker import WallTracker, transform_holds
from reference_wall_tracker import ReferenceWallTracker
from sam_tracker import sam_track_video
from hybrid_tracker import generate_hybrid_tracks
from sam_wall_geometry import anchor_agreement
from pose_overlay import PoseOverlay
from pose_backend import create_backend

MODEL_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
MODEL_URLS = {
    'Lite (fast)': MODEL_URL,
    'Full (recommended)': 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task',
    'Heavy (slower)': 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task',
}
# MediaPipe pose landmarks: shoulders 11/12, elbows 13/14, wrists 15/16,
# hips 23/24, knees 25/26, ankles 27/28.
EDGES = [(11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
         (11, 23), (12, 24), (23, 24), (23, 25), (25, 27),
         (24, 26), (26, 28)]
REQUIRED = [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]
# MediaPipe left/right are the climber's anatomical sides (not the viewer's).
EXTREMITIES = {'left_hand': 15, 'right_hand': 16, 'left_foot': 31, 'right_foot': 32}



def angle(a, b, c):
    """Angle ABC in degrees from 2D points; NaN for degenerate vectors."""
    u, v = np.asarray(a) - np.asarray(b), np.asarray(c) - np.asarray(b)
    norm = np.linalg.norm(u) * np.linalg.norm(v)
    if norm < 1e-9:
        return float('nan')
    return float(np.degrees(np.arccos(np.clip(np.dot(u, v) / norm, -1, 1))))


def measure(points):
    """points: dict index -> (x,y) normalized coordinates, only visible points."""
    def joint(i, j, k):
        return angle(points[i], points[j], points[k]) if all(t in points for t in (i, j, k)) else float('nan')
    hips = (np.array(points[23]) + points[24]) / 2 if 23 in points and 24 in points else [np.nan, np.nan]
    result = {f'{label}_{axis}': float(points[idx][j]) if idx in points else float('nan')
              for label, idx in EXTREMITIES.items() for j, axis in enumerate(('x', 'y'))}
    result.update(dict(hip_x=float(hips[0]), hip_y=float(hips[1]),
                left_elbow=joint(11, 13, 15), right_elbow=joint(12, 14, 16),
                left_knee=joint(23, 25, 27), right_knee=joint(24, 26, 28)))
    return result


def draw_pose(frame, landmarks, visibility_threshold=0.55):
    h, w = frame.shape[:2]
    visible = {i: (p.x, p.y) for i, p in enumerate(landmarks)
               if p.visibility >= visibility_threshold and p.presence >= visibility_threshold}
    for a, b in EDGES:
        if a in visible and b in visible:
            pa = (round(visible[a][0] * w), round(visible[a][1] * h))
            pb = (round(visible[b][0] * w), round(visible[b][1] * h))
            cv2.line(frame, pa, pb, (30, 240, 180), 2)
    for i in REQUIRED + list(EXTREMITIES.values()):
        if i in visible:
            p = (round(visible[i][0] * w), round(visible[i][1] * h))
            cv2.circle(frame, p, 4, (20, 80, 255), -1)
    return visible


def valid_pose(landmarks, threshold=0.55):
    """Conservative geometric sanity checks, not a guarantee of correct identity."""
    pts = {i: p for i, p in enumerate(landmarks)
           if p.visibility >= threshold and p.presence >= threshold
           and 0 <= p.x <= 1 and 0 <= p.y <= 1}
    if not all(i in pts for i in (11, 12, 23, 24)):
        return False
    xs = [pts[i].x for i in (11, 12, 23, 24)]
    ys = [pts[i].y for i in (11, 12, 23, 24)]
    # The visible torso should have meaningful extent and a plausible aspect.
    torso = np.linalg.norm([(pts[11].x + pts[12].x - pts[23].x - pts[24].x)/2,
                            (pts[11].y + pts[12].y - pts[23].y - pts[24].y)/2])
    shoulder_width = np.linalg.norm([pts[11].x - pts[12].x, pts[11].y - pts[12].y])
    hip_width = np.linalg.norm([pts[23].x - pts[24].x, pts[23].y - pts[24].y])
    return (torso >= .055 and shoulder_width >= .025 and hip_width >= .015
            and max(xs)-min(xs) < .85 and max(ys)-min(ys) < .85)


def crop_bounds(width, height, roi):
    """Convert normalized (left, top, right, bottom) into pixel coordinates."""
    left, top, right, bottom = roi
    if not (0 <= left < right <= 1 and 0 <= top < bottom <= 1):
        raise ValueError('Invalid crop region: left < right and top < bottom required.')
    x1, y1 = int(left * width), int(top * height)
    x2, y2 = int(right * width), int(bottom * height)
    if x2 - x1 < 64 or y2 - y1 < 64:
        raise ValueError('Crop region too small. Expand the selection.')
    return x1, y1, x2, y2


def remap_landmarks(landmarks, bounds, width, height):
    """Return full-frame-normalized landmarks, preserving visibility/presence."""
    x1, y1, x2, y2 = bounds
    class Point:
        def __init__(self, p):
            self.x = (x1 + p.x * (x2-x1)) / width
            self.y = (y1 + p.y * (y2-y1)) / height
            self.visibility = p.visibility
            self.presence = p.presence
    return [Point(p) for p in landmarks]


def analyze_video(input_path: str, output_path: str, model_path: str,
                  sample_fps: float = 5, max_seconds: float = 90,
                  roi=(0.15, 0.03, 0.90, 0.78), hold_settings=None, progress_callback=None,
                  pose_backend_name="MediaPipe Full", yolo_device="cpu"):
    """Returns (frame-level dataframe, metadata). Video is a preview, not a diagnosis."""
    def progress(stage, fraction, detail=''):
        if progress_callback is not None:
            progress_callback(stage, max(0., min(1., float(fraction))), detail)
    progress('Preparing', 0.01, 'Opening video')
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise ValueError('Cannot open video. Try converting it to MP4 (H.264).')
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not np.isfinite(fps) or fps <= 0:
        cap.release()
        raise ValueError('Invalid video frame rate.')
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if w <= 0 or h <= 0:
        cap.release()
        raise ValueError('Invalid video dimensions.')
    # Cap output size to keep the first prototype responsive.
    scale = min(1.0, 960 / max(w, h))
    size = (max(2, int(w * scale) // 2 * 2), max(2, int(h * scale) // 2 * 2))
    bounds = crop_bounds(size[0], size[1], roi)
    holds = []
    if hold_settings is not None:
        if 'reviewed_holds' in hold_settings:
            holds = [dict(item) for item in hold_settings['reviewed_holds']]
        else:
            bg = background_from_video(cap, size)
            holds = detect_holds(bg, bounds, **hold_settings)
    tracker_mode = hold_settings.get('tracker', '') if hold_settings else ''
    tracking_enabled = bool(holds) and bool(hold_settings) and hold_settings.get('track_wall', False)
    wall_tracker = None
    wall_tracking_rows = []
    sam_frames = {}
    sam_indices = []
    sam_fps = None
    if holds and hold_settings and hold_settings.get('tracker') == 'SAM 2.1 (AI masks)' and hold_settings.get('sam_diagnostics', False):
        sam_fps = float(hold_settings.get('sam_fps', 2))
        progress('SAM 2 hold tracking', .04, f'Tracking {len(holds)} holds; this may take several minutes on M3')
        sam_frames, sam_indices = sam_track_video(input_path, size, sam_fps, holds,
            hold_settings['sam_model'], hold_settings['sam_checkpoint'],
            max_seconds=max_seconds, device=hold_settings.get('sam_device', 'mps'),
            progress_callback=lambda done, total: progress('SAM 2 hold tracking', .04 + .67 * done / max(total,1), f'{done}/{total} AI reference frames'))
        tracking_enabled = False
    max_frames = min(int(max_seconds * fps), int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or int(max_seconds * fps))
    progress('Optical-flow tracking', .72, 'Estimating intermediate hold positions')
    # SAM masks may incorporate an occluding hand. They are observations, not geometry.
    # All reviewed holds instead share a single wall-to-camera homography.
    hybrid_frames = {}
    target_fps = min(sample_fps, fps)
    step = max(1, round(fps / target_fps))
    actual_fps = fps / step
    writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, size)
    if not writer.isOpened():
        cap.release()
        raise RuntimeError('Could not create annotated video.')
    rows, index, processed, near_rows = [], 0, 0, []
    overlay = PoseOverlay()
    latest_person_bounds = None
    progress('Pose analysis and video rendering', .78, 'Rendering at original frame rate')
    try:
        with create_backend(pose_backend_name, model_path, yolo_device) as detector:
            while index < max_frames:
                ok, frame = cap.read()
                if not ok:
                    break
                analyze_sample = index % step == 0
                frame = cv2.resize(frame, size)
                # Predict pose before estimating wall movement so the moving
                # climber cannot supply registration features.
                pose = None
                if analyze_sample:
                    bx1, by1, bx2, by2 = bounds
                    pose_rgb = cv2.cvtColor(frame[by1:by2, bx1:bx2], cv2.COLOR_BGR2RGB)
                    pose = detector.predict(pose_rgb, int(round(index / fps * 1000)))
                    if pose is not None and valid_pose(pose):
                        pts = remap_landmarks(pose, bounds, *size)
                        visible = [(p.x * size[0], p.y * size[1]) for p in pts
                                   if p.visibility >= .35 and p.presence >= .35 and
                                   np.isfinite(p.x) and np.isfinite(p.y)]
                        if len(visible) >= 8:
                            xs, ys = zip(*visible)
                            pad_x, pad_y = size[0] * .045, size[1] * .035
                            latest_person_bounds = (max(0, min(xs)-pad_x), max(0, min(ys)-pad_y),
                                                    min(size[0]-1, max(xs)+pad_x),
                                                    min(size[1]-1, max(ys)+pad_y))
                sam_geometry_mode = tracker_mode == 'SAM 2.1 (AI masks)' and bool(holds)
                if sam_geometry_mode and wall_tracker is None:
                    wall_tracker = ReferenceWallTracker(frame, holds=holds, keyframe_interval=max(10, round(fps * 2)), wall_bounds=bounds,
                                                        person_bounds=latest_person_bounds)
                if sam_geometry_mode:
                    H = wall_tracker.estimate(frame, index=index, person_bounds=latest_person_bounds)
                    current_holds = transform_holds(holds, H, frame.shape)
                    # A SAM mask does not update positions or size: it can include hands.
                    # Rejecting uncertain alignment is preferable to false hold contacts.
                    agreement = anchor_agreement(current_holds, sam_frames.get(index, []))
                    wall_tracking_rows.append({'time_s': round(index/fps, 3),
                        **agreement, 'aligned': H is not None, 'matches': wall_tracker.good_matches,
                        'inliers': wall_tracker.inliers, 'visible_holds': len(current_holds),
                        'failure_reason': wall_tracker.failure_reason,
                        'tracking_status': wall_tracker.status,
                        'reference_source': wall_tracker.source,
                        'registration_confidence': round(wall_tracker.confidence, 3),
                        'inlier_ratio': round(wall_tracker.inlier_ratio, 3),
                        'reprojection_error_px': round(wall_tracker.reprojection_error, 3),
                        'spatial_coverage': round(wall_tracker.spatial_coverage, 3),
                        'frame_displacement_px': round(wall_tracker.displacement, 2),
                        'reference_frame_id': wall_tracker.last_key_index if wall_tracker.source == 'keyframe' else 0,
                        'sam_anchor_observations': len(sam_frames.get(index, []))})
                if tracking_enabled and wall_tracker is None:
                    wall_tracker = WallTracker(frame, validate_motion=(tracker_mode == 'Fast Preview (OpenCV, no SAM)'))
                if tracking_enabled:
                    H = wall_tracker.estimate(frame)
                    current_holds = transform_holds(holds, H, frame.shape)
                    agreement = anchor_agreement(current_holds, sam_frames.get(index, []))
                    wall_tracking_rows.append({'time_s': round(index/fps, 3),
                        **agreement, 'aligned': H is not None, 'matches': wall_tracker.good_matches,
                        'inliers': wall_tracker.inliers, 'reference_source': wall_tracker.source,
                        'registration_confidence': round(wall_tracker.confidence, 3),
                        'visible_holds': len(current_holds),
                        'failure_reason': wall_tracker.failure_reason})
                elif not sam_frames and not (hold_settings and hold_settings.get('tracker') == 'SAM 2.1 (AI masks)'):
                    current_holds = holds
                x1, y1, x2, y2 = bounds
                # This is a fixed image-space pose ROI, so draw it on EVERY output frame.
                cv2.rectangle(frame, (x1, y1), (x2, y2), (225, 175, 20), 2)
                if not analyze_sample:
                    # Draw a confidence-filtered, briefly retained visualization.
                    # Raw measurements are never filled from visualization state.
                    overlay.draw(frame, index/fps)
                    if current_holds:
                        annotate_holds(frame, current_holds)
                    writer.write(frame)
                    index += 1
                    continue
                record = dict(time_s=round(index / fps, 3), detected=False)
                if pose is not None and valid_pose(pose):
                    full_landmarks = remap_landmarks(pose, bounds, *size)
                    overlay.update(full_landmarks, index/fps)
                    points = {i: (p.x, p.y) for i, p in enumerate(full_landmarks)
                              if p.visibility >= .55 and p.presence >= .55}
                    overlay.draw(frame, index/fps)
                    record.update(measure(points))
                    if not sam_geometry_mode or wall_tracker.status == 'valid':
                        near_rows.extend(proximity_rows(record['time_s'], points, current_holds, *size))
                    record['detected'] = all(j in points for j in (11, 12, 23, 24))
                else:
                    record.update(measure({}))
                    overlay.draw(frame, index/fps)
                if current_holds:
                    annotate_holds(frame, current_holds)
                if (tracking_enabled or sam_geometry_mode) and not current_holds:
                    cv2.putText(frame, 'WALL ALIGNMENT LOST - HOLDS HIDDEN', (12, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, .63, (0, 0, 255), 2)
                rows.append(record)
                writer.write(frame)
                processed += 1
                index += 1
                if index % max(1, round(fps * 2)) == 0:
                    progress('Pose analysis and video rendering', .78 + .18*index/max(1,max_frames), f'{index}/{max_frames} output frames')
    finally:
        cap.release()
        writer.release()
    if not rows:
        raise ValueError('No frames could be processed.')
    progress('Finalizing measurements', .97, 'Video rendering completed')
    return pd.DataFrame(rows), {'frames_analyzed': processed, 'sample_rate_fps': round(actual_fps, 2), 'export_fps': round(fps, 2), 'sam_sample_rate_fps': round(fps / max(1, round(fps / min(sam_fps, fps))), 2) if sam_fps else None, 'sam_anchor_frames': len(sam_indices), 'hold_tracking_mode': 'reference-keyframe wall tracking (SAM diagnostics optional)' if tracker_mode == 'SAM 2.1 (AI masks)' else (tracker_mode or 'none'),
                                'seconds_analyzed': round((processed-1) / actual_fps, 2), 'roi': roi, 'holds': holds, 'near_rows': near_rows, 'wall_tracking': wall_tracking_rows}


def make_feedback(df):
    """Transparent, cautious rule-based observations, NOT learned coaching."""
    good = df[df['detected']].copy()
    if len(good) < 5:
        return ['Pose tracking was unreliable. Try filming from a fixed camera with your full body visible.']
    tips = []
    hip = good[['hip_x', 'hip_y']].dropna()
    if len(hip) >= 5:
        # Frame-normalized displacement: camera must be static for interpretation.
        delta = np.linalg.norm(np.diff(hip.to_numpy(), axis=0), axis=1)
        tips.append(f'Median hip movement between sampled frames: {np.median(delta):.3f} frame widths/heights (normalized). Compare this across attempts filmed from the same angle; larger values are not necessarily worse.')
    for label, column in [('Left knee', 'left_knee'), ('Right knee', 'right_knee'),
                          ('Left elbow', 'left_elbow'), ('Right elbow', 'right_elbow')]:
        values = good[column].dropna()
        if len(values) >= 5:
            tips.append(f'{label} angle range: {values.min():.0f}–{values.max():.0f}°. Review sharp changes in the video; camera angle and landmark errors may distort the estimate.')
    if len(good) / len(df) < 0.75:
        tips.append('Tracking coverage is limited. Film with better lighting and less body/hold occlusion.')
    tips.append('Technique review: identify moments of deliberate foot placement, hip repositioning, and pauses manually. The prototype does not yet determine whether a movement is technically good or bad.')
    return tips


# Version 0.3: events use visible projected landmarks only. They are NOT inferred
# hold contacts, route decisions, force estimates, or coaching judgments.
def detect_motion_events(df: pd.DataFrame, move_threshold: float = 0.09,
                         pause_threshold: float = 0.018, min_duration: float = 0.6):
    """Return (per-frame metrics, event table). All thresholds in normalized pixels/s.

    Smoothed coordinates avoid noisy instantaneous spikes. Tracking gaps stay gaps;
    no interpolation across lost detections is allowed. Events require consecutive
    valid samples and a minimum duration so a single glitch cannot create one.
    """
    data = df.copy().sort_values('time_s').reset_index(drop=True)
    labels = list(EXTREMITIES)
    for label in labels + ['hip']:
        cols = [f'{label}_x', f'{label}_y']
        for col in cols:
            if col not in data:
                data[col] = np.nan
        valid = data['detected'].fillna(False).astype(bool) & data[cols].notna().all(axis=1)
        # Trailing median only within consecutive valid runs; no bridging gaps.
        runs = (valid != valid.shift(fill_value=False)).cumsum()
        smooth = data[cols].where(valid).groupby(runs).transform(lambda x: x.rolling(3, min_periods=1).median())
        dt = data['time_s'].diff()
        adjacent = valid & valid.shift(fill_value=False) & dt.gt(0)
        displacement = np.sqrt(smooth[cols[0]].diff()**2 + smooth[cols[1]].diff()**2)
        speed = (displacement / dt).where(adjacent)
        data[f'{label}_speed'] = speed
    events = []

    def extract(label, kind, condition, min_len):
        # Only contiguous TRUE samples count; each event includes the preceding
        # sample because velocity is measured between two observations.
        condition = condition.fillna(False)
        blocks = (condition != condition.shift(fill_value=False)).cumsum()
        for _, indices in data[condition].groupby(blocks[condition]).groups.items():
            indices = list(indices)
            first, last = indices[0], indices[-1]
            before = max(first - 1, 0)
            duration = float(data.at[last, 'time_s'] - data.at[before, 'time_s'])
            if duration >= min_len:
                events.append({'start_s': round(float(data.at[before, 'time_s']), 2),
                               'end_s': round(float(data.at[last, 'time_s']), 2),
                               'event': kind, 'body_part': label,
                               'duration_s': round(duration, 2)})

    for label in labels:
        speed = data[f'{label}_speed']
        extract(label, 'Hand movement' if 'hand' in label else 'Foot movement',
                speed.gt(move_threshold), 0.18)
    # 'Low-motion interval', not a rest, requires four extremities and hips tracked.
    speed_cols = [f'{label}_speed' for label in labels + ['hip']]
    valid_all = data[speed_cols].notna().all(axis=1)
    stable = valid_all & data[speed_cols].max(axis=1).lt(pause_threshold)
    extract('whole_body', 'Low-motion interval', stable, min_duration)
    return data, pd.DataFrame(events, columns=['start_s', 'end_s', 'event', 'body_part', 'duration_s']).sort_values('start_s').reset_index(drop=True)
