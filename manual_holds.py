"""Manual, deterministic first-frame hold boxes; no inferred colors or model calls."""


def manual_hold_box(width, height, center_x, center_y, radius=30):
    width, height = int(width), int(height)
    if width <= 0 or height <= 0:
        raise ValueError('Image dimensions must be positive')
    x = max(0, min(width - 1, int(center_x)))
    y = max(0, min(height - 1, int(center_y)))
    radius = max(1, int(radius))
    left = max(0, x - radius)
    top = max(0, y - radius)
    right = min(width, x + radius + 1)
    bottom = min(height, y + radius + 1)
    return {'x': left, 'y': top, 'w': right - left, 'h': bottom - top}
