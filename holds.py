"""Experimental static, same-color climbing-hold segmentation (not contact detection)."""
import cv2
import numpy as np


def hex_to_hsv(color):
    """HTML #RRGGBB -> OpenCV HSV tuple."""
    if not isinstance(color, str) or len(color) != 7 or color[0] != '#':
        raise ValueError('Expected a #RRGGBB route color')
    rgb = np.array([[[int(color[i:i+2], 16) for i in (1, 3, 5)]]], dtype=np.uint8)
    return tuple(int(v) for v in cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)[0, 0])


def sample_color(frame, x, y, radius=3):
    """Pick a robust BGR patch median and convert to HTML RGB hex."""
    h, w = frame.shape[:2]
    if not (0 <= x < w and 0 <= y < h):
        raise ValueError('Sample point outside image')
    patch = frame[max(0, y-radius):min(h, y+radius+1), max(0, x-radius):min(w, x+radius+1)]
    b, g, r = np.median(patch.reshape(-1, 3), axis=0).astype(np.uint8)
    return f'#{r:02x}{g:02x}{b:02x}'


def sample_color_robust(frame, x, y, radius=9):
    """Select the color around the CLICK, not the most common color nearby.

    A wide patch can contain the tan climbing wall and adjacent holds. Estimate
    the seed hue from a tight center neighborhood, then include only pixels
    with a similar hue in a moderately sized patch. Return a representative
    saturated HSV sample, excluding glare and very dark shadows.
    """
    height, width = frame.shape[:2]
    if not (0 <= x < width and 0 <= y < height):
        raise ValueError('Sample point outside image')
    radius = max(3, int(radius))
    hsv_full = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    seed_radius = max(2, min(5, radius // 3))
    seed = hsv_full[max(0,y-seed_radius):min(height,y+seed_radius+1),
                    max(0,x-seed_radius):min(width,x+seed_radius+1)].reshape(-1, 3)
    reliable = seed[(seed[:,1] >= 55) & (seed[:,2] >= 45) & (seed[:,2] <= 250)]
    if len(reliable) < 3:
        return sample_color(frame, x, y, radius=seed_radius)
    # Circular hue median around a locally representative center value.
    hues = reliable[:,0].astype(np.int16)
    dist = np.abs(hues[:, None] - hues[None, :])
    dist = np.minimum(dist, 180-dist)
    seed_hue = int(hues[np.argmin(np.sum(dist, axis=1))])
    patch = hsv_full[max(0,y-radius):min(height,y+radius+1),
                     max(0,x-radius):min(width,x+radius+1)].reshape(-1,3)
    delta = np.abs(patch[:,0].astype(np.int16)-seed_hue)
    delta = np.minimum(delta,180-delta)
    selected = patch[(delta <= 9) & (patch[:,1] >= 55) &
                     (patch[:,2] >= 45) & (patch[:,2] <= 250)]
    if len(selected) < 3:
        selected = reliable
    best_hue = int(np.median(((selected[:,0].astype(np.int16)-seed_hue+90)%180)-90))
    h = (seed_hue + best_hue) % 180
    # The target is a COLOR FAMILY, not the literal RGB value at the clicked
    # (possibly shaded) pixel. Representative upper quantiles describe the
    # painted plastic more faithfully without changing its estimated hue.
    s = int(np.percentile(selected[:,1], 70))
    v = int(np.percentile(selected[:,2], 75))
    rgb = cv2.cvtColor(np.uint8([[[h,s,v]]]), cv2.COLOR_HSV2RGB)[0,0]
    return '#%02x%02x%02x' % tuple(int(t) for t in rgb)


def segment_color(frame, color, hue_tolerance=12, min_saturation=65):
    """Segment brightly colored holds; HSV hue wraps around red."""
    h, s, v = hex_to_hsv(color)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    dh = np.abs(hsv[:, :, 0].astype(np.int16) - h)
    hue_dist = np.minimum(dh, 180 - dh)
    sat_ok = hsv[:, :, 1] >= min_saturation
    # Excludes shadows and most bright white text/LED highlights.
    mask = ((hue_dist <= hue_tolerance) & sat_ok & (hsv[:, :, 2] >= 45)).astype(np.uint8) * 255
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


def detect_holds(background, roi, route_color, hue_tolerance=12,
                 min_saturation=65, min_area=70, max_area_fraction=.025,
                 min_fill_ratio=.38, exclude_boundary=True, min_thickness=2.1):
    """Conservative, geometry-filtered HSV proposals; NOT semantic hold recognition.

    Removes narrow strips, weak/fragmented regions, low-purity boxes and most
    incomplete edge proposals. Intentionally trades recall for fewer false boxes.
    """
    height, width = background.shape[:2]
    x1, y1, x2, y2 = map(int, roi)
    if x1 >= x2 or y1 >= y2:
        return []
    crop = background[y1:y2, x1:x2]
    mask = segment_color(crop, route_color, hue_tolerance, min_saturation)
    # Thin tape / seams connect unrelated holds. Suppress narrow colored regions
    # before grouping components (kernel is in processed-image pixels).
    if min_thickness > 0:
        distance = cv2.distanceTransform(mask, cv2.DIST_L2, 3)
        core = np.uint8(distance >= min_thickness) * 255
        # Expand surviving cores only within the original color mask.
        restored = cv2.dilate(core, np.ones((5, 5), np.uint8))
        mask = cv2.bitwise_and(mask, restored)
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)
    holds = []
    roi_area = max(1, (x2-x1)*(y2-y1))
    roi_h, roi_w = mask.shape
    for component in range(1, count):
        x, y, bw, bh, area = [int(z) for z in stats[component]]
        if not min_area <= area <= max_area_fraction*roi_area:
            continue
        if bw < 6 or bh < 6:
            continue
        aspect = max(bw, bh) / max(1, min(bw, bh))
        if aspect > 4.5:
            continue
        fill_ratio = area / float(bw*bh)
        if fill_ratio < min_fill_ratio:
            continue
        if exclude_boundary and (x <= 1 or y <= 1 or x+bw >= roi_w-1 or y+bh >= roi_h-1):
            continue
        # Reject very thin silhouettes even when diagonal bbox looks compact.
        part = np.uint8(labels[y:y+bh, x:x+bw] == component) * 255
        if cv2.distanceTransform(part, cv2.DIST_L2, 3).max() < 2.5:
            continue
        cx, cy = centroids[component]
        holds.append({'x': x+x1, 'y': y+y1, 'w': bw, 'h': bh,
                      'cx': float(cx+x1), 'cy': float(cy+y1), 'area': area,
                      'fill_ratio': round(fill_ratio, 3)})
    holds.sort(key=lambda hold: (hold['cy'], hold['cx']))
    for i, hold in enumerate(holds, 1):
        hold['hold_id'] = f'H{i:02d}'
    return holds


def background_from_video(cap, output_size, sample_count=15):
    """Temporal median reduces influence of a moving climber on a static wall.

    For prolonged stationary climbers, their clothing may still appear in median.
    """
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    samples = []
    if n > 1:
        for idx in np.linspace(0, n-1, min(sample_count, n), dtype=int):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ok, frame = cap.read()
            if ok:
                samples.append(cv2.resize(frame, output_size))
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    if not samples:
        raise ValueError('Could not sample frames to detect holds')
    return np.median(np.stack(samples), axis=0).astype(np.uint8)


def annotate_holds(frame, holds):
    """Draw hold boxes safely after SAM/optical-flow floating-point updates."""
    height, width = frame.shape[:2]
    for hold in holds:
        try:
            values = np.asarray([hold[k] for k in ('x', 'y', 'w', 'h')], dtype=float)
        except (KeyError, TypeError, ValueError):
            continue
        if not np.all(np.isfinite(values)):
            continue
        x, y, w, h = (float(v) for v in values)
        if w <= 0 or h <= 0:
            continue
        x1, y1 = max(0, int(round(x))), max(0, int(round(y)))
        x2, y2 = min(width - 1, int(round(x + w))), min(height - 1, int(round(y + h)))
        if x1 >= x2 or y1 >= y2:
            continue
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 80, 220), 2)
        cv2.putText(frame, str(hold.get('hold_id', '?')), (x1, max(12, y1-4)),
                    cv2.FONT_HERSHEY_SIMPLEX, .40, (255, 80, 220), 1, cv2.LINE_AA)


def proximity_rows(time_s, points, holds, width, height, max_distance=0.04):
    """Possible 2D proximity, NOT proven physical contact."""
    landmarks = {'left_hand': 15, 'right_hand': 16, 'left_foot': 31, 'right_foot': 32}
    result = []
    for name, idx in landmarks.items():
        if idx not in points or not holds:
            continue
        px, py = points[idx]
        distances = [(np.hypot(px - hold['cx']/width, py-hold['cy']/height), hold)
                     for hold in holds]
        distance, closest = min(distances, key=lambda pair: pair[0])
        if distance <= max_distance:
            result.append({'time_s': round(time_s, 3), 'limb': name,
                           'hold_id': closest['hold_id'], 'distance_norm': round(float(distance), 4),
                           'status': 'near (unconfirmed)'})
    return result


def normalize_reviewed_holds(holds, width, height):
    """Convert original-video pixel boxes to analysis-size boxes (max 960 px)."""
    scale = min(1., 960 / max(width, height))
    out_w = max(2, int(width * scale) // 2 * 2)
    out_h = max(2, int(height * scale) // 2 * 2)
    out = []
    for item in holds:
        x = max(0, min(out_w - 1, round(item['x'] * out_w / width)))
        y = max(0, min(out_h - 1, round(item['y'] * out_h / height)))
        w = max(1, min(out_w - x, round(item['w'] * out_w / width)))
        h = max(1, min(out_h - y, round(item['h'] * out_h / height)))
        out.append(dict(hold_id=item['hold_id'], x=x, y=y, w=w, h=h,
                        cx=x + w / 2, cy=y + h / 2, area=w*h))
    return out


def suggested_box(frame, x, y, color, radius=25, tolerance=25):
    """Find a color-connected region near a click, falling back to a sized box.

    All coordinates in the original frame; this is a proposal, not model inference.
    """
    height, width = frame.shape[:2]
    x, y = int(x), int(y)
    if not (0 <= x < width and 0 <= y < height):
        raise ValueError('Click outside frame')
    radius = max(6, int(radius))
    region = frame[max(0, y-2*radius):min(height, y+2*radius+1),
                   max(0, x-2*radius):min(width, x+2*radius+1)]
    ox, oy = max(0, x-2*radius), max(0, y-2*radius)
    hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
    seed = cv2.cvtColor(frame[y:y+1, x:x+1], cv2.COLOR_BGR2HSV)[0,0]
    hue_dist = abs(hsv[:,:,0].astype(np.int16)-int(seed[0]))
    dist = np.minimum(hue_dist, 180-hue_dist)
    mask = ((dist <= tolerance) & (hsv[:,:,1] >= 25) & (hsv[:,:,2] >= 35)).astype(np.uint8)
    components, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    selected = int(labels[y-oy, x-ox])
    if selected and stats[selected, cv2.CC_STAT_AREA] >= 10:
        xx, yy, ww, hh, _ = [int(v) for v in stats[selected]]
        if ww > 3 and hh > 3:
            return {'x': ox+xx, 'y': oy+yy, 'w': ww, 'h': hh}
    return {'x': max(0,x-radius), 'y': max(0,y-radius),
            'w': min(width,max(0,x+radius))-max(0,x-radius),
            'h': min(height,max(0,y+radius))-max(0,y-radius)}


def remove_hold_near(holds, x, y):
    """Remove the hold containing a click, else the closest center within 40px."""
    if not holds:
        return list(holds), False
    inside = [i for i, box in enumerate(holds)
              if box['x'] <= x <= box['x']+box['w'] and box['y'] <= y <= box['y']+box['h']]
    if inside:
        index = min(inside, key=lambda i: holds[i]['w']*holds[i]['h'])
    else:
        distances = [float(np.hypot(x-(b['x']+b['w']/2), y-(b['y']+b['h']/2))) for b in holds]
        index = int(np.argmin(distances))
        if distances[index] > 40:
            return list(holds), False
    return [dict(b) for i,b in enumerate(holds) if i != index], True


def reindex_holds(holds):
    return [dict(b, hold_id=f'H{i:02d}') for i,b in enumerate(holds,1)]
