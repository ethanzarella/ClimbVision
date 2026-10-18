"""Confidence-aware, geometry-first wall registration for moving-camera videos.

All transformations map reviewed FIRST-frame holds into the current frame.
Neither object segmentation nor a person's keypoints may independently move holds.
"""
import cv2
import numpy as np


class ReferenceWallTracker:
    def __init__(self, first_frame, holds=(), min_inliers=12, keyframe_interval=30,
                 wall_bounds=None, recovery_frames=5, person_bounds=None):
        self.orb = cv2.ORB_create(nfeatures=2600, fastThreshold=10)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        self.height, self.width = first_frame.shape[:2]
        self.first_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
        self.wall_mask = np.zeros_like(self.first_gray)
        if wall_bounds is None:
            self.wall_mask[:] = 255
        else:
            x1, y1, x2, y2 = [int(v) for v in wall_bounds]
            cv2.rectangle(self.wall_mask, (max(0, x1), max(0, y1)),
                          (min(self.width-1, x2), min(self.height-1, y2)), 255, -1)
        self._exclude_person(self.wall_mask, person_bounds)
        # Manual holds are legitimate fixed wall geometry; do not exclude them.
        self.ref_kp, self.ref_desc = self.orb.detectAndCompute(self.first_gray, self.wall_mask)
        self.key_kp, self.key_desc = self.ref_kp, self.ref_desc
        self.key_to_reference = np.eye(3, dtype=np.float64)
        self.last_transform = np.eye(3, dtype=np.float64)
        self.last_key_index = 0
        self.last_frame_index = 0
        self.keyframe_interval = max(5, int(keyframe_interval))
        self.min_inliers = int(min_inliers)
        self.recovery_frames = max(0, int(recovery_frames))
        self.prev_gray = self.first_gray.copy()
        self.prev_points = self._seed_points(self.first_gray, self.wall_mask)
        self.inliers = self.good_matches = 0
        self.inlier_ratio = self.reprojection_error = self.spatial_coverage = 0.
        self.displacement = 0.
        self.failure_reason = None
        self.source = 'initial'
        self.status = 'valid'
        self.confidence = 1.
        self.lost_frames = 0

    @staticmethod
    def _exclude_person(mask, bounds):
        """Exclude detected person region from wall registration features."""
        if bounds is None:
            return mask
        x1, y1, x2, y2 = (int(v) for v in bounds)
        if x2 > x1 and y2 > y1:
            cv2.rectangle(mask, (max(0, x1), max(0, y1)),
                          (min(mask.shape[1]-1, x2), min(mask.shape[0]-1, y2)), 0, -1)
        return mask

    def _seed_points(self, gray, mask):
        p = cv2.goodFeaturesToTrack(gray, 700, .012, 6, mask=mask, blockSize=5)
        return p.astype(np.float32) if p is not None else None

    def _coverage(self, pts):
        if len(pts) < 4:
            return 0.
        p = pts.reshape(-1, 2)
        # Occupied grid cells measure spatial distribution, not just bounding extent.
        ix = np.clip((p[:, 0] / self.width * 4).astype(int), 0, 3)
        iy = np.clip((p[:, 1] / self.height * 4).astype(int), 0, 3)
        occupied = len(set(zip(ix.tolist(), iy.tolist())))
        spread = (np.ptp(p[:, 0]) / self.width) * (np.ptp(p[:, 1]) / self.height)
        return float(min(occupied / 16., spread * 2.))

    def _fit_points(self, src, dst):
        if len(src) < self.min_inliers:
            return None
        h, mask = cv2.findHomography(src, dst, cv2.RANSAC, 3.)
        if h is None or mask is None or not np.isfinite(h).all() or abs(h[2, 2]) < 1e-8:
            return None
        mask = mask.ravel().astype(bool)
        count, total = int(mask.sum()), len(mask)
        ratio = count / total
        if count < self.min_inliers or ratio < .43:
            return None
        projected = cv2.perspectiveTransform(src[mask].reshape(-1, 1, 2), h).reshape(-1, 2)
        err = float(np.median(np.linalg.norm(projected - dst[mask].reshape(-1, 2), axis=1)))
        coverage = self._coverage(dst[mask])
        if coverage < .055 or err > 2.8:
            return None
        return h / h[2, 2], count, total, ratio, err, coverage

    def _match(self, source_kp, source_desc, dest_kp, dest_desc):
        if source_desc is None or dest_desc is None or len(source_desc) < 12 or len(dest_desc) < 12:
            return None
        pairs = self.matcher.knnMatch(source_desc, dest_desc, k=2)
        good = [a for pair in pairs if len(pair) == 2 for a, b in [pair]
                if a.distance < .74 * b.distance]
        if len(good) < self.min_inliers:
            return None
        src = np.float32([source_kp[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst = np.float32([dest_kp[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        return self._fit_points(src, dst)

    def _flow(self, gray, current_mask):
        if self.prev_points is None or len(self.prev_points) < self.min_inliers:
            return None
        next_pts, valid, _ = cv2.calcOpticalFlowPyrLK(self.prev_gray, gray, self.prev_points,
                                                     None, winSize=(23, 23), maxLevel=3)
        if next_pts is None or valid is None:
            return None
        back, back_valid, _ = cv2.calcOpticalFlowPyrLK(gray, self.prev_gray, next_pts,
                                                       None, winSize=(23, 23), maxLevel=3)
        if back is None or back_valid is None:
            return None
        good = (valid.ravel() > 0) & (back_valid.ravel() > 0)
        good &= np.linalg.norm(self.prev_points.reshape(-1, 2) - back.reshape(-1, 2), axis=1) < 1.5
        x = np.clip(next_pts[:, 0, 0].astype(int), 0, self.width - 1)
        y = np.clip(next_pts[:, 0, 1].astype(int), 0, self.height - 1)
        good &= current_mask[y, x] > 0
        if int(good.sum()) < self.min_inliers:
            return None
        return self._fit_points(self.prev_points[good], next_pts[good])

    def _geometry_ok(self, H):
        if not np.isfinite(H).all() or abs(np.linalg.det(H)) < 1e-9:
            return False
        corners = np.float32([[[0., 0.]], [[self.width, 0.]],
                              [[self.width, self.height]], [[0., self.height]]])
        new = cv2.perspectiveTransform(corners, H).reshape(4, 2)
        old = cv2.perspectiveTransform(corners, self.last_transform).reshape(4, 2)
        if not np.isfinite(new).all():
            return False
        sx = np.linalg.norm(new[1] - new[0]) / self.width
        sy = np.linalg.norm(new[3] - new[0]) / self.height
        if not (.45 < sx < 2.2 and .45 < sy < 2.2):
            return False
        jump = float(np.median(np.linalg.norm(new - old, axis=1)))
        limit = max(22., np.hypot(self.width, self.height) *
                    min(.30, .065 + .012 * self.lost_frames))
        return (jump <= limit and
                cv2.contourArea(new.astype(np.float32)) > .1 * self.width * self.height and
                cv2.isContourConvex(new.astype(np.float32)))

    def _current_mask(self, gray, person_bounds=None):
        # The wall mask is transformed into the current view. It is not a person mask.
        mask = cv2.warpPerspective(self.wall_mask, self.last_transform,
                                   (self.width, self.height), flags=cv2.INTER_NEAREST)
        # Dynamic occlusion guard: compare adjacent frames after compensating for
        # the LAST accepted camera motion. Large residual regions can be a moving
        # person. This is deliberately conservative to retain wall features.
        if self.last_frame_index > 0 and self.displacement < 2.5:
            diff = cv2.absdiff(gray, self.prev_gray)
            _, moving = cv2.threshold(diff, 58, 255, cv2.THRESH_BINARY)
            moving = cv2.morphologyEx(moving, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
            if np.count_nonzero(moving) < .22 * moving.size:
                moving = cv2.dilate(moving, np.ones((11, 11), np.uint8))
                mask[moving > 0] = 0
        return self._exclude_person(mask, person_bounds)

    def estimate(self, frame, index=0, person_bounds=None):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        current_mask = self._current_mask(gray, person_bounds)
        kp, desc = self.orb.detectAndCompute(gray, current_mask)
        candidates = []
        for label, skp, sdesc, conv in (
            ('reference', self.ref_kp, self.ref_desc, np.eye(3)),
            ('keyframe', self.key_kp, self.key_desc, self.key_to_reference)):
            fit = self._match(skp, sdesc, kp, desc)
            if fit is None:
                continue
            h, n, total, ratio, err, cover = fit
            H = h @ conv
            if abs(H[2, 2]) < 1e-8:
                continue
            H /= H[2, 2]
            if self._geometry_ok(H):
                candidates.append((n * ratio * cover, label, H.copy(), n, total, ratio, err, cover))
        flow = self._flow(gray, current_mask)
        if flow is not None:
            h, n, total, ratio, err, cover = flow
            H = h @ self.last_transform
            if abs(H[2, 2]) > 1e-8:
                H /= H[2, 2]
                if self._geometry_ok(H):
                    candidates.append((n * ratio * cover, 'optical_flow', H.copy(), n, total, ratio, err, cover))
        anchors = [c for c in candidates if c[1] in ('reference', 'keyframe')]
        if anchors:
            best_anchor = max(anchors, key=lambda c: c[0])
            # An independently registered wall reference takes precedence over
            # optical flow when the two disagree at representative wall points.
            sample = np.float32([[[self.width*.2, self.height*.2]],
                                 [[self.width*.8, self.height*.2]],
                                 [[self.width*.2, self.height*.8]],
                                 [[self.width*.8, self.height*.8]]])
            ref_pts = cv2.perspectiveTransform(sample, best_anchor[2]).reshape(-1, 2)
            candidates = [c for c in candidates if c[1] != 'optical_flow' or
                          np.median(np.linalg.norm(
                              cv2.perspectiveTransform(sample, c[2]).reshape(-1, 2) - ref_pts,
                              axis=1)) <= max(6., .012 * np.hypot(self.width, self.height))]
        if not candidates:
            self.lost_frames += 1
            self.inliers = self.good_matches = 0
            self.inlier_ratio = self.reprojection_error = self.spatial_coverage = 0.
            self.failure_reason = 'no spatially validated wall registration'
            self.confidence = 0.
            self.status = 'uncertain' if self.lost_frames <= self.recovery_frames else 'lost'
            self.source = 'recovery_hold' if self.status == 'uncertain' else 'lost'
            # Do not advance optical-flow origin on failure: next frame recovers
            # by reference/keyframe matching, not chained unverified estimates.
            return self.last_transform.copy() if self.status == 'uncertain' else None
        _, label, H, n, total, ratio, err, cover = max(candidates, key=lambda c: c[0])
        self.source = label
        self.inliers, self.good_matches = n, total
        self.inlier_ratio, self.reprojection_error, self.spatial_coverage = ratio, err, cover
        self.confidence = float(np.clip(ratio * min(1., cover / .25) * max(0., 1 - err / 4.), 0, 1))
        self.failure_reason = None
        self.status = 'valid'
        corners = np.float32([[[0., 0.]], [[self.width, 0.]], [[self.width, self.height]], [[0., self.height]]])
        self.displacement = float(np.median(np.linalg.norm(
            cv2.perspectiveTransform(corners, H).reshape(-1, 2) -
            cv2.perspectiveTransform(corners, self.last_transform).reshape(-1, 2), axis=1)))
        self.last_transform = H.copy()
        self.last_frame_index = index
        self.lost_frames = 0
        self.prev_gray = gray.copy()
        self.prev_points = self._seed_points(gray, current_mask)
        if index - self.last_key_index >= self.keyframe_interval and kp is not None and len(kp) >= 30:
            self.key_to_reference = np.linalg.inv(H)
            self.key_kp, self.key_desc = kp, desc
            self.last_key_index = index
        return H
