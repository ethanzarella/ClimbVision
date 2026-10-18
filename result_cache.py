"""Atomic disk cache for completed ClimbVision analyses.

Only local JSON, CSV, and MP4 are read. Cache keys include input bytes and every
analysis-relevant setting. Incomplete runs are never exposed as hits.
"""
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import time
import pandas as pd

CACHE_SCHEMA = '0.11.3-pose-masked-wall-v1'
FRAME_KEYS = ('df', 'metrics', 'events', 'holds_df', 'near_df', 'wall_df')


def fingerprint(video_bytes, model_name, sample_fps, roi, hold_settings, preview_seconds):
    model_file = None
    if hold_settings and hold_settings.get('sam_checkpoint'):
        cp = Path(hold_settings['sam_checkpoint']).expanduser()
        if cp.is_file():
            stat = cp.stat()
            model_file = {'size': stat.st_size, 'modified_ns': stat.st_mtime_ns}
    config = {'schema': CACHE_SCHEMA, 'sam_checkpoint_file': model_file,
              'video_sha256': hashlib.sha256(video_bytes).hexdigest(),
              'model': model_name, 'pose_fps': sample_fps,
              'roi': roi, 'holds': hold_settings, 'seconds': preview_seconds}
    encoded = json.dumps(config, sort_keys=True, separators=(',', ':'), default=str).encode()
    return hashlib.sha256(encoded).hexdigest()


def cache_root():
    return Path.home() / '.cache' / 'climbvision' / 'analyses'


def load_result(root, key):
    folder = Path(root) / key
    try:
        manifest = json.loads((folder / 'manifest.json').read_text())
        if manifest.get('schema') != CACHE_SCHEMA or manifest.get('key') != key:
            return None
        result = manifest['result']
        result['video_bytes'] = (folder / 'video.mp4').read_bytes()
        for name in FRAME_KEYS:
            result[name] = pd.read_json(folder / (name + '.json'), orient='table')
        return result
    except (OSError, ValueError, KeyError, TypeError):
        return None


def save_result(root, key, result):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    target = root / key
    temp = Path(tempfile.mkdtemp(prefix='.partial-', dir=root))
    try:
        (temp / 'video.mp4').write_bytes(result['video_bytes'])
        for name in FRAME_KEYS:
            result[name].to_json(temp / (name + '.json'), orient='table', index=False)
        metadata = {k: v for k, v in result.items() if k not in (*FRAME_KEYS, 'video_bytes')}
        (temp / 'manifest.json').write_text(json.dumps({
            'schema': CACHE_SCHEMA, 'key': key, 'created_at': time.time(),
            'result': metadata}, default=_json_safe))
        if target.exists():
            shutil.rmtree(target)
        os.replace(temp, target)
    finally:
        if temp.exists():
            shutil.rmtree(temp)


def _json_safe(obj):
    if hasattr(obj, 'item'):
        return obj.item()
    if hasattr(obj, 'tolist'):
        return obj.tolist()
    raise TypeError(f'Cannot serialize {type(obj).__name__}')


def clear_cache(root):
    root = Path(root).expanduser()
    if root.exists():
        for folder in root.iterdir():
            if folder.is_dir() and (folder.name.startswith('.partial-') or len(folder.name) == 64):
                shutil.rmtree(folder)
