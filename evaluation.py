"""Transparent first-frame bounding-box benchmarking (not trajectory evaluation)."""
import io
import math
import pandas as pd

COLUMNS = ('x', 'y', 'w', 'h')

def read_ground_truth(raw):
    try:
        df = pd.read_csv(io.BytesIO(raw) if isinstance(raw, bytes) else raw)
    except Exception as e:
        raise ValueError(f'Could not read CSV: {e}') from e
    if any(k not in df for k in COLUMNS):
        raise ValueError('Required CSV columns: x, y, w, h')
    if df.empty:
        raise ValueError('Ground truth must contain at least one hold')
    for k in COLUMNS:
        df[k] = pd.to_numeric(df[k], errors='coerce')
    if not df[list(COLUMNS)].apply(lambda col: col.map(math.isfinite)).all().all():
        raise ValueError('Coordinates must be finite numbers')
    if (df[['w','h']] <= 0).any().any():
        raise ValueError('Hold widths and heights must be positive')
    return df

def ground_truth_template(boxes):
    if boxes.empty:
        return pd.DataFrame(columns=list(COLUMNS))
    return boxes.loc[:, list(COLUMNS)].copy()

def iou(a, b):
    ax,ay,aw,ah=(float(a[k]) for k in COLUMNS)
    bx,by,bw,bh=(float(b[k]) for k in COLUMNS)
    if min(aw,ah,bw,bh)<=0:
        return 0.0
    dx=max(0.0,min(ax+aw,bx+bw)-max(ax,bx))
    dy=max(0.0,min(ay+ah,by+bh)-max(ay,by))
    intersection=dx*dy
    union=aw*ah+bw*bh-intersection
    return intersection/union if union>0 else 0.0

def evaluate_boxes(predictions, truth, threshold=0.5):
    if not (0 < threshold <= 1):
        raise ValueError('IoU threshold must be in (0,1]')
    pairs = [(iou(p,t), i,j) for i,(_,p) in enumerate(predictions.iterrows())
             for j,(_,t) in enumerate(truth.iterrows())]
    used_p=set(); used_t=set()
    for score,i,j in sorted(pairs,reverse=True):
        if score < threshold: break
        if i not in used_p and j not in used_t:
            used_p.add(i); used_t.add(j)
    tp=len(used_p); fp=len(predictions)-tp; fn=len(truth)-tp
    precision=tp/(tp+fp) if tp+fp else 0.
    recall=tp/(tp+fn) if tp+fn else 0.
    return {'tp':tp,'fp':fp,'fn':fn,'precision':precision,'recall':recall,
            'f1':2*precision*recall/(precision+recall) if precision+recall else 0.}
