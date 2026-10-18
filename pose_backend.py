"""Interchangeable pose estimators with MediaPipe-shaped landmark output.

YOLO standard COCO keypoints do not contain heel/toe positions. Missing
landmarks carry zero confidence and are never fabricated for metrics.
"""
from dataclasses import dataclass
import time
import numpy as np

YOLO_NAMES = ('YOLO26 Small Pose (17 keypoints)',)
# COCO17 to MediaPipe33 anatomical landmark indices.
COCO_TO_MP = {0:0, 1:2, 2:5, 3:7, 4:8, 5:11, 6:12, 7:13, 8:14,
              9:15, 10:16, 11:23, 12:24, 13:25, 14:26, 15:27, 16:28}

@dataclass
class Landmark:
    x: float = 0.0
    y: float = 0.0
    visibility: float = 0.0
    presence: float = 0.0


class MediaPipeBackend:
    def __init__(self, model_path):
        import mediapipe as mp
        self.mp = mp
        options = mp.tasks.vision.PoseLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=model_path),
            running_mode=mp.tasks.vision.RunningMode.VIDEO, num_poses=1,
            min_pose_detection_confidence=.5,
            min_pose_presence_confidence=.5,
            min_tracking_confidence=.5)
        self.detector = mp.tasks.vision.PoseLandmarker.create_from_options(options)
    def __enter__(self):
        return self
    def __exit__(self, *args):
        self.detector.close()
    def predict(self, rgb, timestamp_ms):
        mp = self.mp
        img = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
        result = self.detector.detect_for_video(img, timestamp_ms)
        return result.pose_landmarks[0] if result.pose_landmarks else None


class YoloPoseBackend:
    def __init__(self, model_path='yolo26s-pose.pt', device='cpu', confidence=.25):
        try:
            from ultralytics import YOLO
        except ImportError as error:
            raise RuntimeError('YOLO backend needs ultralytics. Run: pip install ultralytics') from error
        self.model = YOLO(model_path)
        self.device = device
        self.confidence = confidence
    def __enter__(self):
        return self
    def __exit__(self, *args):
        return False
    def predict(self, rgb, timestamp_ms):
        import cv2
        # Ultralytics expects BGR numpy frames; our common interface uses RGB.
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        result = self.model.predict(source=bgr, device=self.device, verbose=False,
                                    conf=self.confidence, imgsz=640)[0]
        if result.keypoints is None or result.keypoints.xy is None or len(result.keypoints.xy) == 0:
            return None
        boxes = result.boxes.conf.cpu().numpy() if result.boxes is not None else np.ones(len(result.keypoints.xy))
        person = int(np.argmax(boxes))
        coords = result.keypoints.xyn[person].cpu().numpy()
        scores = result.keypoints.conf
        scores = scores[person].cpu().numpy() if scores is not None else np.ones(17)
        points = [Landmark() for _ in range(33)]
        for coco, mp_idx in COCO_TO_MP.items():
            x, y = map(float, coords[coco])
            score = float(scores[coco])
            if np.isfinite(x) and np.isfinite(y) and np.isfinite(score) and 0 <= x <= 1 and 0 <= y <= 1:
                points[mp_idx] = Landmark(x, y, score, score)
        return points


def create_backend(backend_name, model_path=None, yolo_device='cpu'):
    if backend_name in YOLO_NAMES:
        return YoloPoseBackend(device=yolo_device)
    return MediaPipeBackend(model_path)


def benchmark_models(video_path, roi, mediapipe_model_path, sample_fps=5, max_seconds=20,
                     yolo_device='cpu'):
    """Side-by-side diagnostic benchmark; NOT a ground-truth accuracy evaluation.

    Both models process identical decoded frames and crops; inference time excludes
    video decoding, models' first-load cost, and video writing.
    """
    import cv2
    from analysis import crop_bounds, remap_landmarks, valid_pose
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError('Could not open benchmark video')
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        cap.release()
        raise ValueError('Invalid video FPS')
    step = max(1, round(fps / sample_fps))
    frames, index = [], 0
    limit = round(fps * max_seconds)
    try:
        while index < limit:
            ok, frame = cap.read()
            if not ok:
                break
            if index % step == 0:
                h, w = frame.shape[:2]
                x1,y1,x2,y2 = crop_bounds(w, h, roi)
                rgb = cv2.cvtColor(frame[y1:y2,x1:x2], cv2.COLOR_BGR2RGB)
                frames.append((index/fps, rgb, (x1,y1,x2,y2), w, h))
            index += 1
    finally:
        cap.release()
    if not frames:
        raise ValueError('No frames for benchmark')
    import pandas as pd
    output = []
    for name in ['MediaPipe Full', YOLO_NAMES[0]]:
        path = mediapipe_model_path if name == 'MediaPipe Full' else None
        with create_backend(name, path, yolo_device) as backend:
            for seconds, rgb, bounds, width, height in frames:
                start = time.perf_counter()
                pose = backend.predict(rgb, int(seconds*1000))
                elapsed = (time.perf_counter()-start)*1000
                valid = pose is not None and valid_pose(pose)
                points = remap_landmarks(pose,bounds,width,height) if valid else []
                record = {'backend':name, 'time_s':seconds, 'inference_ms':elapsed,
                          'torso_detected':bool(valid)}
                for part, idx in [('left_wrist',15),('right_wrist',16),('left_ankle',27),('right_ankle',28),('left_toe',31),('right_toe',32)]:
                    p = points[idx] if points else None
                    ok = bool(p and min(p.visibility,p.presence) >= .55)
                    record[f'{part}_visible'] = ok
                    record[f'{part}_x'] = p.x if ok else np.nan
                    record[f'{part}_y'] = p.y if ok else np.nan
                output.append(record)
    details = pd.DataFrame(output)
    metrics = []
    for name, group in details.groupby('backend',sort=False):
        entry = {'backend':name, 'frames':len(group),
                 'torso_coverage_pct':round(100*group.torso_detected.mean(),1),
                 'mean_inference_ms':round(group.inference_ms.mean(),1),
                 'median_inference_ms':round(group.inference_ms.median(),1)}
        for part in ['left_wrist','right_wrist','left_ankle','right_ankle','left_toe','right_toe']:
            entry[f'{part}_coverage_pct'] = round(100*group[f'{part}_visible'].mean(),1)
            xy=group[[f'{part}_x',f'{part}_y']].to_numpy(float)
            diff=np.linalg.norm(np.diff(xy,axis=0),axis=1)
            dt=np.diff(group.time_s.to_numpy(float))
            speeds=diff/dt
            entry[f'{part}_median_speed_norm_per_s']=round(float(np.median(speeds[np.isfinite(speeds)])),4) if np.any(np.isfinite(speeds)) else np.nan
        metrics.append(entry)
    return pd.DataFrame(metrics), details
