"""Estimate background/wall motion and transfer reference-frame hold boxes.

This tracks wall geometry, NOT recognition or physical hand/foot contact.
The estimator reports failure rather than making unbounded predictions.
"""
import cv2
import numpy as np


class WallTracker:
    def __init__(self, reference, min_inliers=12, validate_motion=False):
        self.reference = reference.copy()
        self.reference_gray = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
        self.orb = cv2.ORB_create(nfeatures=2200, fastThreshold=12)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        self.ref_kp, self.ref_desc = self.orb.detectAndCompute(self.reference_gray, None)
        self.min_inliers = min_inliers
        self.validate_motion = validate_motion
        self.height, self.width = reference.shape[:2]
        self.failure_reason = None
        self.last_transform = np.eye(3, dtype=np.float64)
        self.last_good = True
        self.good_matches = 0
        self.inliers = 0
        # Shared diagnostics interface used by the analysis exporter.
        self.source = 'original_reference'
        self.confidence = 0.0

    def estimate(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        kp, desc = self.orb.detectAndCompute(gray, None)
        self.good_matches = self.inliers = 0
        self.confidence = 0.0
        self.failure_reason = None
        if self.ref_desc is None or desc is None or len(self.ref_desc) < 8 or len(desc) < 8:
            self.last_good = False
            self.failure_reason = "insufficient features or invalid transform"
            return None
        pairs = self.matcher.knnMatch(self.ref_desc, desc, k=2)
        good = [a for a, b in pairs if a.distance < 0.75 * b.distance]
        self.good_matches = len(good)
        if len(good) < self.min_inliers:
            self.last_good = False
            self.failure_reason = "insufficient features or invalid transform"
            return None
        source = np.float32([self.ref_kp[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        target = np.float32([kp[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        H, mask = cv2.findHomography(source, target, cv2.RANSAC, 3.0)
        self.inliers = int(mask.sum()) if mask is not None else 0
        if H is None or self.inliers < self.min_inliers or not np.isfinite(H).all():
            self.last_good = False
            self.failure_reason = "insufficient features or invalid transform"
            return None
        H = H / H[2, 2]
        # Reject pathological fits and folds. Perspective effects still require inspection.
        A = H[:2, :2]
        scale = abs(float(np.linalg.det(A))) ** 0.5
        if not (0.35 <= scale <= 2.8) or np.linalg.det(A) <= 0:
            self.last_good = False
            self.failure_reason = "insufficient features or invalid transform"
            return None
        if self.validate_motion:
            if self.good_matches and self.inliers / self.good_matches < 0.35:
                self.last_good = False
                self.failure_reason = 'low geometric inlier ratio'
                return None
            # Compare the new wall projection against the last accepted one.
            # Outliers must NOT update last_transform.
            corners = np.float32([[[0, 0]], [[self.width, 0]],
                                  [[self.width, self.height]], [[0, self.height]]])
            old = cv2.perspectiveTransform(corners, self.last_transform).reshape(-1, 2)
            new = cv2.perspectiveTransform(corners, H).reshape(-1, 2)
            displacement = np.linalg.norm(new - old, axis=1)
            max_jump = max(18., 0.08 * float(np.hypot(self.width, self.height)))
            if not np.isfinite(new).all() or np.median(displacement) > max_jump:
                self.last_good = False
                self.failure_reason = 'implausible frame-to-frame wall movement'
                return None
            old_span = np.linalg.norm(old[1] - old[0])
            new_span = np.linalg.norm(new[1] - new[0])
            ratio = new_span / max(1e-6, old_span)
            if not (0.86 <= ratio <= 1.16):
                self.last_good = False
                self.failure_reason = 'implausible wall scale change'
                return None
        self.last_transform = H
        self.last_good = True
        self.confidence = float(np.clip(self.inliers / max(1, self.good_matches), 0.0, 1.0))
        return H


def transform_holds(holds, H, frame_shape):
    """Transform 4 reference-box corners; output axis-aligned boxes in current frame."""
    if H is None:
        return []
    height, width = frame_shape[:2]
    output = []
    for hold in holds:
        x, y, w, h = (float(hold[k]) for k in ('x', 'y', 'w', 'h'))
        corners = np.float32([[x, y], [x+w, y], [x+w, y+h], [x, y+h]]).reshape(-1, 1, 2)
        projected = cv2.perspectiveTransform(corners, H).reshape(-1, 2)
        if not np.isfinite(projected).all():
            continue
        left, top = np.maximum(0, np.floor(projected.min(axis=0))).astype(int)
        right, bottom = np.minimum([width-1, height-1], np.ceil(projected.max(axis=0))).astype(int)
        if right <= left or bottom <= top:
            continue
        output.append({**hold, 'x': int(left), 'y': int(top), 'w': int(right-left),
                       'h': int(bottom-top), 'cx': float((left+right)/2),
                       'cy': float((top+bottom)/2)})
    return output
