"""Optional SAM 2.1 prompted multi-object video tracker.

No silent fallback: missing packages/checkpoints or inference failures must surface.
Tracks sampled video frames at the same resolution/timestamps as ClimbVision output.
"""
from pathlib import Path
import tempfile
import cv2
import numpy as np

CONFIGS = {
    'Tiny (M3 recommended)': ('configs/sam2.1/sam2.1_hiera_t.yaml', 'sam2.1_hiera_tiny.pt'),
    'Small': ('configs/sam2.1/sam2.1_hiera_s.yaml', 'sam2.1_hiera_small.pt'),
}


def mask_to_hold(mask, original, frame_size, min_pixels=12):
    """Convert binary segmentation to a clipped bbox; reject empty/tiny masks."""
    width, height = frame_size
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 2:
        raise ValueError('Expected a 2D segmentation mask')
    if mask.shape != (height, width):
        mask = cv2.resize(mask.astype('uint8'), (width, height), interpolation=cv2.INTER_NEAREST) > 0
    yy, xx = np.nonzero(mask)
    if len(xx) < min_pixels:
        return None
    x1, x2, y1, y2 = int(xx.min()), int(xx.max()) + 1, int(yy.min()), int(yy.max()) + 1
    if x2 <= x1 or y2 <= y1:
        return None
    return {**original, 'x': x1, 'y': y1, 'w': x2-x1, 'h': y2-y1,
            'cx': (x1+x2)/2, 'cy': (y1+y2)/2, 'area': len(xx)}


def extract_sampled_frames(input_path, out_dir, frame_size, sample_fps, max_seconds=90):
    """Export sequential JPEGs and return original frame indices at analysis cadence."""
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise ValueError('Cannot open video')
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not np.isfinite(fps) or fps <= 0:
            raise ValueError('Invalid video FPS')
        step = max(1, round(fps / min(sample_fps, fps)))
        limit = min(int(max_seconds * fps), int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or int(max_seconds * fps))
        indices = []
        idx = 0
        while idx < limit:
            ok, frame = cap.read()
            if not ok:
                break
            if idx % step == 0:
                frame = cv2.resize(frame, frame_size)
                path = Path(out_dir) / f'{len(indices):05d}.jpg'
                if not cv2.imwrite(str(path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95]):
                    raise RuntimeError(f'Could not save {path}')
                indices.append(idx)
            idx += 1
        if not indices:
            raise ValueError('Video contains no usable frames')
        return indices
    finally:
        cap.release()


def sam_track_video(input_path, frame_size, sample_fps, holds, model_name,
                    checkpoint_path, max_seconds=90, device='mps', max_holds=16, progress_callback=None):
    """Return mapping original frame-index -> per-object SAM masks/bboxes.

    SAM2 is invoked lazily, so the classical tracker still works without Torch.
    The checkpoint is supplied by the user; no hidden downloads.
    """
    if not holds:
        return {}, []
    if len(holds) > max_holds:
        raise ValueError(f'SAM 2 is capped at {max_holds} holds per run on a laptop. Reduce the reviewed list.')
    ckpt = Path(checkpoint_path).expanduser()
    if not ckpt.is_file():
        raise FileNotFoundError(f'SAM 2 checkpoint not found: {ckpt}. See README instructions.')
    if model_name not in CONFIGS:
        raise ValueError('Unknown SAM 2 model')
    try:
        import torch
        from sam2.build_sam import build_sam2_video_predictor
    except ImportError as exc:
        raise RuntimeError('SAM 2 / PyTorch not installed in this environment. Follow the SAM setup in README.') from exc
    if device == 'mps' and not torch.backends.mps.is_available():
        raise RuntimeError('Apple MPS is unavailable. Check native arm64 Python and PyTorch installation.')
    cfg, expected_name = CONFIGS[model_name]
    if ckpt.name != expected_name:
        raise ValueError(f'{model_name} requires {expected_name}; selected {ckpt.name}')
    with tempfile.TemporaryDirectory(prefix='climbvision_sam_') as folder:
        indices = extract_sampled_frames(input_path, folder, frame_size, sample_fps, max_seconds)
        predictor = build_sam2_video_predictor(cfg, str(ckpt), device=device, vos_optimized=False)
        with torch.inference_mode():
            state = predictor.init_state(folder, offload_video_to_cpu=False,
                                         offload_state_to_cpu=False)
            for number, hold in enumerate(holds, 1):
                x, y, w, h = [float(hold[k]) for k in ('x','y','w','h')]
                box = np.array([x, y, x+w, y+h], dtype=np.float32)
                predictor.add_new_points_or_box(state, frame_idx=0, obj_id=number, box=box)
            tracked = {}
            for sequence_index, ids, logits in predictor.propagate_in_video(state):
                if sequence_index >= len(indices):
                    continue
                per_frame = []
                for n, obj_id in enumerate(ids):
                    original = holds[int(obj_id)-1]
                    mask = (logits[n] > 0.0).squeeze().detach().to('cpu').numpy()
                    updated = mask_to_hold(mask, original, frame_size)
                    if updated:
                        per_frame.append(updated)
                tracked[indices[sequence_index]] = per_frame
                if progress_callback is not None and (sequence_index % 2 == 0 or sequence_index == len(indices)-1):
                    progress_callback(sequence_index+1, len(indices))
            predictor.reset_state(state)
            return tracked, indices
