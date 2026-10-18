"""SAM observations are quality signals, never authority over wall-attached hold geometry."""
import math


def box_iou(a, b):
    ax, ay = float(a['x']), float(a['y'])
    bx, by = float(b['x']), float(b['y'])
    ar, ab = ax + float(a['w']), ay + float(a['h'])
    br, bb = bx + float(b['w']), by + float(b['h'])
    w, h = max(0.0, min(ar, br) - max(ax, bx)), max(0.0, min(ab, bb) - max(ay, by))
    intersection = w * h
    total = float(a['w']) * float(a['h']) + float(b['w']) * float(b['h']) - intersection
    return intersection / total if total > 0 and math.isfinite(total) else 0.0


def anchor_agreement(wall_holds, sam_holds, threshold=0.35):
    """Summarize independent observations; never modify reviewed/projected boxes."""
    wall = {str(h.get('hold_id', '')): h for h in wall_holds}
    matched = 0
    rejected = 0
    for h in sam_holds:
        baseline = wall.get(str(h.get('hold_id', '')))
        if baseline is None or box_iou(baseline, h) < threshold:
            rejected += 1
        else:
            matched += 1
    return {'sam_agreed': matched, 'sam_disagreed': rejected}
