"""Color-managed upload preprocessing for HLG/PQ HDR video.

Convert once, before preview or analysis, so all visual consumers share the
same BT.709 SDR pixels. Source bytes remain unchanged.
"""
import json
import shutil
import subprocess
from pathlib import Path
import tempfile

HDR_TRANSFERS = {'arib-std-b67', 'smpte2084'}
PIPELINE_VERSION = 'hdr-to-sdr-v1'


def find_tool(name):
    found = shutil.which(name)
    if found:
        return found
    for prefix in ('/opt/homebrew/opt/ffmpeg-full/bin', '/opt/homebrew/bin', '/usr/local/bin'):
        path = Path(prefix) / name
        if path.is_file():
            return str(path)
    raise RuntimeError(f'{name} was not found. Install FFmpeg with ffprobe (brew install ffmpeg-full).')


def probe_video(path, ffprobe=None):
    exe = ffprobe or find_tool('ffprobe')
    call = subprocess.run([exe, '-v', 'error', '-select_streams', 'v:0',
        '-show_entries', 'stream=codec_name,color_transfer,color_space,color_primaries,pix_fmt',
        '-of', 'json', str(path)], capture_output=True, text=True, timeout=25, check=True)
    streams = json.loads(call.stdout).get('streams', [])
    if not streams:
        raise RuntimeError('No video stream found in uploaded file.')
    return streams[0]


def is_hdr(stream):
    return stream.get('color_transfer', '').lower() in HDR_TRANSFERS


def convert_hdr(path, output_path, ffmpeg=None):
    exe = ffmpeg or find_tool('ffmpeg')
    capability = subprocess.run([exe, '-filters'], capture_output=True, text=True, timeout=15)
    if ' zscale ' not in capability.stdout:
        full = Path('/opt/homebrew/opt/ffmpeg-full/bin/ffmpeg')
        if ffmpeg is None and full.is_file():
            exe = str(full)
            capability = subprocess.run([exe, '-filters'], capture_output=True, text=True, timeout=15)
        if ' zscale ' not in capability.stdout:
            raise RuntimeError('FFmpeg does not support zscale. Install with brew install ffmpeg-full.')
    filters = ('zscale=transfer=linear:npl=100,format=gbrpf32le,'
               'tonemap=tonemap=hable:desat=0,'
               'zscale=primaries=bt709:transfer=bt709:matrix=bt709:range=limited,'
               'format=yuv420p')
    command = [exe, '-hide_banner', '-loglevel', 'error', '-nostdin', '-y',
               '-i', str(path), '-map', '0:v:0', '-map', '0:a:0?',
               '-vf', filters, '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '19',
               '-pix_fmt', 'yuv420p', '-color_primaries', 'bt709', '-color_trc', 'bt709',
               '-colorspace', 'bt709', '-c:a', 'aac', '-b:a', '160k',
               '-movflags', '+faststart', str(output_path)]
    result = subprocess.run(command, capture_output=True, text=True, timeout=900)
    if result.returncode or not Path(output_path).is_file() or Path(output_path).stat().st_size == 0:
        message = (result.stderr or 'No converted output produced').strip()
        raise RuntimeError('HDR conversion failed. Ensure FFmpeg has the zscale filter '
                           '(brew install ffmpeg-full). ' + message[-1200:])
    return str(output_path)


def prepare_upload(video_bytes, suffix):
    """Return (working_video_bytes, color_metadata), preserving original input."""
    with tempfile.TemporaryDirectory(prefix='climbvision_hdr_') as directory:
        inp = Path(directory) / ('original' + (suffix or '.mov'))
        inp.write_bytes(video_bytes)
        meta = probe_video(inp)
        if not is_hdr(meta):
            return video_bytes, {'hdr': False, 'converted': False, 'video_stream': meta}
        out = Path(directory) / 'working_sdr.mp4'
        convert_hdr(inp, out)
        result_meta = probe_video(out)
        if is_hdr(result_meta) or result_meta.get('color_transfer') != 'bt709':
            raise RuntimeError('HDR conversion did not yield BT.709 SDR; refusing to use inconsistent colors.')
        return out.read_bytes(), {'hdr': True, 'converted': True,
                                  'video_stream': meta, 'output_stream': result_meta}
