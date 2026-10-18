"""Display-only temporal stabilization. Never substitute these points for measurements."""
import math
import cv2

EDGES = [(11,12),(11,13),(13,15),(12,14),(14,16),(11,23),(12,24),(23,24),(23,25),(25,27),(24,26),(26,28),(27,31),(28,32)]
TORSO = {11,12,23,24}

class PoseOverlay:
    def __init__(self, smoothing=0.48, max_age_s=0.32, min_confidence=0.38):
        self.smoothing = smoothing
        self.max_age_s = max_age_s
        self.min_confidence = min_confidence
        self.state = {}

    def update(self, landmarks, seconds):
        """Accepts full-frame-normalized MediaPipe landmarks or None on detection failure."""
        if landmarks is None:
            return
        for idx, p in enumerate(landmarks):
            confidence = min(float(p.visibility), float(p.presence))
            if (not all(math.isfinite(x) for x in (p.x,p.y,confidence)) or
                not (-0.03 <= p.x <= 1.03 and -0.03 <= p.y <= 1.03) or
                confidence < self.min_confidence):
                continue
            previous = self.state.get(idx)
            if previous is not None and seconds - previous[2] <= self.max_age_s:
                # High confidence observations react faster; uncertain ones smooth more.
                alpha = max(self.smoothing, min(0.85, confidence * 0.8))
                x = alpha * p.x + (1-alpha) * previous[0]
                y = alpha * p.y + (1-alpha) * previous[1]
            else:
                x,y = float(p.x),float(p.y)
            self.state[idx] = (x,y,seconds,confidence)

    def visible(self, seconds):
        visible = {}
        for idx,(x,y,t,confidence) in self.state.items():
            age = seconds-t
            # Lost wrists/feet should not float indefinitely.
            max_age = self.max_age_s if idx in TORSO else self.max_age_s * 0.7
            if 0 <= age <= max_age:
                visible[idx] = (x,y,confidence)
        return visible

    def draw(self, frame, seconds):
        h,w=frame.shape[:2]
        visible=self.visible(seconds)
        for a,b in EDGES:
            if a in visible and b in visible:
                pa=tuple(round(v*scale) for v,scale in zip(visible[a][:2],(w,h)))
                pb=tuple(round(v*scale) for v,scale in zip(visible[b][:2],(w,h)))
                cv2.line(frame,pa,pb,(30,240,180),2,cv2.LINE_AA)
        for i in set(x for edge in EDGES for x in edge):
            if i in visible:
                p=tuple(round(v*scale) for v,scale in zip(visible[i][:2],(w,h)))
                cv2.circle(frame,p,3,(20,80,255),-1,cv2.LINE_AA)
        return visible
