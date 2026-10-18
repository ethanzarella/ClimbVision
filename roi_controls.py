"""Human-readable ROI size and position controls, independent of Streamlit."""

def region_from_controls(width_pct, height_pct, horizontal_pct, vertical_pct):
    """Return left,top,right,bottom fractions; positions mean 0=left/top, 100=right/bottom.

    Region size always increases with width/height controls, even near an edge.
    """
    width = max(10., min(100., float(width_pct))) / 100.
    height = max(10., min(100., float(height_pct))) / 100.
    horizontal = max(0., min(100., float(horizontal_pct))) / 100.
    vertical = max(0., min(100., float(vertical_pct))) / 100.
    left = (1. - width) * horizontal
    top = (1. - height) * vertical
    return (left, top, left + width, top + height)
