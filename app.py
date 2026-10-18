from pathlib import Path
import platform
import tempfile
import cv2
import numpy as np
import urllib.request
import subprocess
import hashlib
import streamlit as st
from analysis import MODEL_URLS, analyze_video, make_feedback, detect_motion_events
from holds import remove_hold_near, reindex_holds, normalize_reviewed_holds
from hold_rect_picker import draw_hold_rectangle
from evaluation import evaluate_boxes, ground_truth_template, read_ground_truth
from result_cache import fingerprint, cache_root, load_result, save_result, clear_cache
from hdr_video import prepare_upload, PIPELINE_VERSION
from pose_backend import YOLO_NAMES, benchmark_models
from roi_controls import region_from_controls
from roi_picker import draggable_region

st.set_page_config(page_title='ClimbVision', page_icon='🧗', layout='wide')
st.title('ClimbVision')
st.caption('Climbing movement analysis • two same-route attempts • experimental prototype v0.11.3')

if 'analysis_result' not in st.session_state:
    st.session_state.analysis_result = None
if 'comparison_results' not in st.session_state:
    st.session_state.comparison_results = {}
if 'compare_open' not in st.session_state:
    st.session_state.compare_open = False
if 'second_mode' not in st.session_state:
    st.session_state.second_mode = False
if 'first_attempt_hash' not in st.session_state:
    st.session_state.first_attempt_hash = None

def browser_compatible_video(source: str, destination: str):
    """Convert OpenCV mp4v output to browser-friendly H.264 + yuv420p.

    Returns (video bytes, was_converted, warning). Keeps a usable download fallback.
    """
    raw = Path(source).read_bytes()
    try:
        import imageio_ffmpeg
        exe = imageio_ffmpeg.get_ffmpeg_exe()
        command = [exe, '-hide_banner', '-loglevel', 'error', '-y', '-i', source,
                   '-an', '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '23',
                   '-pix_fmt', 'yuv420p', '-movflags', '+faststart', destination]
        subprocess.run(command, capture_output=True, text=True, check=True, timeout=180)
        if Path(destination).stat().st_size == 0:
            raise RuntimeError('H.264 conversion returned an empty file')
        return Path(destination).read_bytes(), True, None
    except (ImportError, subprocess.SubprocessError, OSError, RuntimeError) as error:
        return raw, False, str(error)


@st.cache_resource(show_spinner='Downloading pose model (first use of each model only)...')
def get_model(model_name):
    filename = {'Lite (fast)': 'pose_landmarker_lite.task',
                'Full (recommended)': 'pose_landmarker_full.task',
                'Heavy (slower)': 'pose_landmarker_heavy.task'}[model_name]
    path = Path.home() / '.cache' / 'climbvision' / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists() or path.stat().st_size == 0:
        urllib.request.urlretrieve(MODEL_URLS[model_name], path)
    return str(path)

# When the user asks to analyze another attempt, keep the first completed
# result ABOVE the second video's controls. The first result is not invalidated.
if st.session_state.second_mode and st.session_state.first_attempt_hash:
    first = st.session_state.comparison_results.get(st.session_state.first_attempt_hash)
    if first is not None:
        st.subheader('Attempt A — completed analysis')
        st.caption(f"{first['source_name']} · {first['model_name']} · {first['sample_fps']} pose FPS")
        first_df = first['df']
        first_info = first['info']
        f1, f2, f3 = st.columns(3)
        f1.metric('Frames analyzed', first_info.get('frames_analyzed', 0))
        f2.metric('Tracking coverage', f"{first_df['detected'].mean():.0%}" if not first_df.empty else '—')
        f3.metric('Analyzed duration', f"{first_info.get('seconds_analyzed', 0):.1f}s")
        st.video(first['video_bytes'], format='video/mp4')
        with st.expander('Attempt A — saved data and downloads', expanded=False):
            st.caption('Original analysis remains available while you analyze the second attempt.')
            st.download_button('Download Attempt A annotated video', first['video_bytes'],
                               'attempt_a_annotated.mp4', 'video/mp4', key='a_video_download')
            for title, field in (('Pose measurements', 'df'), ('Movement measurements', 'metrics'),
                                 ('Movement events', 'events'), ('Route holds', 'holds_df'),
                                 ('Hold proximity', 'near_df'), ('Wall tracking', 'wall_df')):
                dataset = first.get(field)
                if dataset is not None and not dataset.empty:
                    with st.expander(title):
                        st.dataframe(dataset, hide_index=True, width='stretch')
                        st.download_button('Download ' + title + ' CSV', dataset.to_csv(index=False).encode(),
                                           file_name='attempt_a_' + field + '.csv', mime='text/csv',
                                           key='a_csv_' + field)
        st.divider()
        st.subheader('Attempt B — analyze another video')
        st.caption('The completed Attempt A above is preserved. Choose the second video and analyze it here.')

st.subheader('Pose tracking model')
model_name = st.selectbox('Choose model', list(MODEL_URLS) + list(YOLO_NAMES), index=1,
    help='Full/Heavy generally trade speed for quality. Neither guarantees accuracy on climbing poses. Compare the annotated outputs.')

# Built-in browser-ready SDR samples. Both show attempts on the same route.
# The originals are not distributed; these compact copies are intended for demonstration.
from io import BytesIO

DEMO_DIR = Path(__file__).parent / 'demo_videos'
SOURCE_OPTIONS = ['Demo A — attempt one', 'Demo B — attempt two', 'Upload your own video']
if 'pending_source_choice' in st.session_state:
    st.session_state['video_source_choice'] = st.session_state.pop('pending_source_choice')
source_choice = st.radio('Choose a climbing video', SOURCE_OPTIONS, horizontal=True, key='video_source_choice',
                         help='Both demo videos are attempts on the same route. No upload required.')
selected_sample = {'Demo A — attempt one': 'attempt_a.mp4',
                   'Demo B — attempt two': 'attempt_b.mp4'}.get(source_choice)
if selected_sample:
    sample_path = DEMO_DIR / selected_sample
    if sample_path.is_file():
        class DemoUpload(BytesIO):
            def __init__(self, path):
                super().__init__(path.read_bytes())
                self.name = path.name
            def getvalue(self):
                return super().getvalue()
        uploaded = DemoUpload(sample_path)
    else:
        st.error('The bundled demo video is missing. Choose Upload your own video instead.')
        uploaded = None
else:
    uploaded = st.file_uploader('Upload a climbing video', type=['mp4', 'mov', 'avi'])

@st.cache_data(show_spinner=False, max_entries=3)
def color_manage_upload(content, suffix, pipeline_version):
    return prepare_upload(content, suffix)

working_bytes = None
color_metadata = None
if uploaded:
    upload_bytes_original = uploaded.getvalue()
    try:
        with st.spinner('Checking video color metadata and converting HDR to SDR if needed (first upload only)...'):
            working_bytes, color_metadata = color_manage_upload(
                upload_bytes_original, Path(uploaded.name).suffix.lower(), PIPELINE_VERSION)
    except Exception as error:
        st.error(f'Unable to prepare the video consistently for color analysis: {error}')
        st.info('HDR uploads need FFmpeg with the zscale filter. On Mac: brew install ffmpeg-full. '
                'A failed conversion is not silently replaced with washed-out frames.')
        st.stop()
    if color_metadata['converted']:
        st.success('HDR (HLG/PQ) detected and converted to BT.709 SDR for consistent preview, color sampling, and analysis.')
        with st.expander('Show color-managed input video', expanded=False):
            st.video(working_bytes, format='video/mp4')
    else:
        st.caption('SDR video detected; original pixels retained for analysis.')

# MPS is an Apple Silicon GPU backend and is unavailable on Linux cloud hosts.
# The setting remains selectable for local testing, but CPU is the cloud default.
is_apple_silicon = platform.system() == 'Darwin' and platform.machine().lower() == 'arm64'
yolo_devices = ['mps', 'cpu'] if is_apple_silicon else ['cpu']
yolo_device = st.selectbox(
    'YOLO inference device', yolo_devices, index=0,
    help='Apple Silicon Macs can run YOLO with MPS. Linux cloud deployments run on CPU.'
) if model_name in YOLO_NAMES else 'cpu'
sample_fps = st.slider('Pose analysis FPS', 2, 20, 10, help='How often the selected pose estimator measures body pose. Does not control SAM 2 inference or export FPS.')
if uploaded:
    with tempfile.TemporaryDirectory() as preview_dir:
        preview_path = str(Path(preview_dir) / ('preview.mp4' if color_metadata['converted'] else 'preview' + Path(uploaded.name).suffix.lower()))
        Path(preview_path).write_bytes(working_bytes)
        cap = cv2.VideoCapture(preview_path)
        ok, preview = cap.read()
        cap.release()
    if ok:
        st.subheader('Climber tracking region')
        st.caption('Select the area where the climber moves. Keep the floor and equipment outside the selection. The crop is applied BEFORE pose detection.')
        upload_roi_id = hashlib.sha256(working_bytes).hexdigest()[:16]
        roi_state_key = f'roi_drag_selection_{upload_roi_id}'
        if roi_state_key not in st.session_state:
            st.session_state[roi_state_key] = region_from_controls(75, 75, 50, 5)
        roi = draggable_region(preview, st.session_state[roi_state_key], key=f"roi_drag_{upload_roi_id}_{st.session_state.get(f'roi_drag_epoch_{upload_roi_id}', 0)}")
        st.session_state[roi_state_key] = roi
        h, w = preview.shape[:2]
        original_frame = preview.copy()
        if st.button('Reset tracking region', key=f'roi_reset_{upload_roi_id}'):
            st.session_state[roi_state_key] = region_from_controls(75, 75, 50, 5)
            # Changing the component key creates a fresh rectangle.
            epoch_key = f'roi_drag_epoch_{upload_roi_id}'
            st.session_state[epoch_key] = st.session_state.get(epoch_key, 0) + 1
            st.rerun()
        st.subheader('Manual route hold selection')
        detect_route = st.checkbox('Annotate route holds', value=True,
            help='Select only the climbing holds in this route. No automatic recognition or color matching is used.')
        hold_settings = None
        if detect_route:
            upload_id = hashlib.sha256(uploaded.getvalue()).hexdigest()[:16]
            edit_key = f'hold_review_{upload_id}'
            if edit_key not in st.session_state:
                # Bundled route presets are per recording; never reuse pixel boxes
                # across two videos without alignment.
                preset_file = {'attempt_a.mp4': 'attempt_a_holds.json',
                               'attempt_b.mp4': 'attempt_b_holds.json'}.get(selected_sample)
                if preset_file:
                    preset = __import__('json').loads((DEMO_DIR / preset_file).read_text())
                    if (preset['frame_width'], preset['frame_height']) == (w, h):
                        st.session_state[edit_key] = reindex_holds(preset['holds'])
                    else:
                        st.session_state[edit_key] = []
                else:
                    st.session_state[edit_key] = []
            edit_mode = st.radio('Editing action', ['Draw a hold', 'Remove a hold'], horizontal=True,
                                 key=f'edit_mode_{upload_id}')
            revision_key = f'hold_revision_{upload_id}'
            if revision_key not in st.session_state:
                st.session_state[revision_key] = 0
            operation = draw_hold_rectangle(original_frame, st.session_state[edit_key],
                mode='add' if edit_mode == 'Draw a hold' else 'remove',
                key=f'hold_rect_{upload_id}_{st.session_state[revision_key]}')
            if operation is not None:
                if operation['action'] == 'add':
                    box = operation['box']
                    st.session_state[edit_key] = reindex_holds(st.session_state[edit_key] + [box])
                elif operation['action'] == 'remove':
                    ex, ey = operation['x'], operation['y']
                    new_list, removed = remove_hold_near(st.session_state[edit_key], ex, ey)
                    if removed:
                        st.session_state[edit_key] = reindex_holds(new_list)
                    else:
                        st.warning('Click inside an existing box to remove it.')
                st.session_state[revision_key] += 1
                st.rerun()
            c1,c2 = st.columns(2)
            c1.metric('Reviewed hold boxes', len(st.session_state[edit_key]))
            if c2.button('Clear all boxes', key=f'resetreview_{upload_id}'):
                st.session_state[edit_key] = []
                st.session_state[revision_key] += 1
                st.rerun()
            import json
            st.download_button('Download reviewed route JSON',
                json.dumps({'frame_width':w,'frame_height':h,'holds':st.session_state[edit_key]},indent=2),
                file_name='climbvision_reviewed_route.json',mime='application/json', on_click='ignore',
                key=f'routejson_{upload_id}')
            saved_route = st.file_uploader('Load previously reviewed route JSON', type=['json'],
                                            key=f'loadroute_{upload_id}')
            if saved_route is not None and st.button('Load this route',key=f'importreview_{upload_id}'):
                try:
                    payload=json.loads(saved_route.getvalue())
                    if (int(payload['frame_width']),int(payload['frame_height'])) != (w,h):
                        st.error('Saved route dimensions differ from this video. Use a matching video.')
                    else:
                        valid=[dict(x=int(b['x']),y=int(b['y']),w=int(b['w']),h=int(b['h'])) for b in payload['holds']]
                        st.session_state[edit_key]=reindex_holds(valid)
                        st.session_state[revision_key]+=1
                        st.rerun()
                except (ValueError,KeyError,TypeError) as exc:
                    st.error(f'Invalid route file: {exc}')
            st.markdown('### Camera-motion compensation')
            tracker = st.radio('Hold tracking algorithm',
                ['Fast Preview (OpenCV, no SAM)', 'SAM 2.1 (AI masks)', 'Stationary camera (fixed holds)', 'OpenCV wall alignment (old)'], horizontal=True,
                help='Use Fast Preview for quick iterations; SAM 2 is significantly slower on M3.')
            track_wall = tracker in ('Fast Preview (OpenCV, no SAM)', 'OpenCV wall alignment (old)')
            hold_settings = {'reviewed_holds': normalize_reviewed_holds(
                st.session_state[edit_key], w, h), 'track_wall': track_wall,
                'tracker': tracker}
            if tracker == 'SAM 2.1 (AI masks)':
                st.caption('Fast wall tracking is the default: reviewed holds follow validated camera geometry. SAM 2 is optional diagnostic inference, not required to move the hold boxes.')
                hold_settings['sam_diagnostics'] = st.checkbox('Run additional SAM 2 mask diagnostics (slow)', value=False, help='Leave OFF for fast hold tracking. Turn ON only to compare SAM masks against wall-locked holds.')
                if hold_settings['sam_diagnostics']:
                    hold_settings['sam_model'] = st.selectbox('SAM 2.1 checkpoint size',
                        ['Tiny (M3 recommended)', 'Small'])
                    hold_settings['sam_checkpoint'] = st.text_input('SAM 2 checkpoint path',
                        str(Path.home() / 'Downloads' / 'sam2' / 'checkpoints' / 'sam2.1_hiera_tiny.pt'),
                        help='Download the corresponding model from the official SAM 2 repository; see README.')
                    hold_settings['sam_device'] = st.selectbox('SAM inference device', ['mps', 'cpu'], index=0)
                    hold_settings['sam_fps'] = st.slider('SAM diagnostic FPS', 1, 3, 1,
                        help='Diagnostic masks only; they never move reviewed hold boxes.')
                else:
                    st.caption('Fast mode: SAM 2 is skipped. No SAM checkpoint or inference is required.')
            elif tracker == 'Stationary camera (fixed holds)':
                st.caption('Stationary mode: holds stay at their reviewed image coordinates. Use only for a camera that does not move.')
            elif tracker == 'Fast Preview (OpenCV, no SAM)':
                st.caption('Fast Preview validates wall transformations and hides holds when alignment is unreliable; this mode does not use SAM 2.')
            else:
                st.caption('Legacy OpenCV wall alignment has fewer motion checks and is preserved as an experimental baseline.')
            if not st.session_state[edit_key]:
                st.warning('No reviewed holds selected. Add route holds before analyzing if you want proximity outputs.')
        else:
            hold_settings = None
    else:
        st.error('Could not preview video. Try MP4 H.264.')
        st.stop()
else:
    roi = region_from_controls(75, 75, 50, 5)
if uploaded:
    preview_seconds = st.slider('Analyze first N seconds (quick tests)', 10, 90, 10, step=10, help='Limit processing time while testing. Increase for full evaluation.')
    st.caption('Speed tip: Stationary Camera or Fast Preview + Lite pose model + 5 FPS + a short clip avoids SAM 2 entirely. Export still uses source FPS.')
if uploaded:
    use_cache = st.checkbox('Reuse saved analysis when settings are unchanged', value=True, help='Completed runs are stored on this Mac. Identical videos and tracking settings reuse results, including the exported video.')
    with st.expander('Manage analysis cache'):
        st.caption(f'Location: {cache_root()} · Cached videos may contain private footage. Entries persist across app restarts.')
        if st.button('Delete all saved ClimbVision analyses'):
            clear_cache(cache_root())
            st.success('Saved analyses cleared. Current on-screen results remain until the next run.')
if uploaded and st.button('Analyze climb', type='primary'):
    # Clear only the ACTIVE attempt, never the saved first attempt.
    st.session_state.analysis_result = None
    status_box = st.status('Preparing analysis...', expanded=True)
    progress_bar = st.progress(0, text='Preparing video...')
    stage_line = st.empty()
    def update_progress(stage, fraction, detail=''):
        progress_bar.progress(min(100, max(0, int(fraction*100))), text=f'{stage}: {detail}')
        stage_line.caption(f'Current stage: {stage} — {detail}')
    with status_box:
        try:
            with tempfile.TemporaryDirectory() as workdir:
                src = str(Path(workdir) / ('input.mp4' if color_metadata['converted'] else 'input' + Path(uploaded.name).suffix.lower()))
                dst = str(Path(workdir) / 'annotated.mp4')
                compatible = str(Path(workdir) / 'annotated_h264.mp4')
                upload_bytes = uploaded.getvalue()
                key = fingerprint(working_bytes, model_name + ':' + yolo_device, sample_fps, roi, hold_settings, preview_seconds)
                cached = load_result(cache_root(), key) if use_cache else None
                if cached is not None:
                    cached['cache_hit'] = True
                    st.session_state.analysis_result = cached
                    update_progress('Cached analysis', 1., 'Loaded existing results; SAM 2 and MediaPipe were skipped')
                    status_box.update(label='Loaded from disk cache — no AI inference', state='complete', expanded=False)
                    st.rerun()
                Path(src).write_bytes(working_bytes)
                df, info = analyze_video(src, dst, get_model(model_name) if model_name in MODEL_URLS else None, sample_fps, roi=roi, hold_settings=hold_settings, max_seconds=preview_seconds, progress_callback=update_progress, pose_backend_name=model_name, yolo_device=yolo_device)
                update_progress('Movement metrics', .98, 'Calculating events')
                metrics, events = detect_motion_events(df)
                update_progress('Browser video conversion', .99, 'Encoding H.264 for playback')
                video_bytes, converted, conversion_warning = browser_compatible_video(dst, compatible)
                import pandas as pd
                st.session_state.analysis_result = {
                    'video_bytes': video_bytes,
                    'converted': converted,
                    'conversion_warning': conversion_warning,
                    'df': df,
                    'info': info,
                    'metrics': metrics,
                    'events': events,
                    'holds_df': pd.DataFrame(info['holds']),
                    'near_df': pd.DataFrame(info['near_rows']),
                    'wall_df': pd.DataFrame(info.get('wall_tracking', [])),
                    'hold_enabled': hold_settings is not None,
                    'source_hash': hashlib.sha256(upload_bytes).hexdigest(),
                    'source_name': uploaded.name,
                    'color_management': color_metadata,
                    'model_name': model_name,
                    'sample_fps': sample_fps,
                    'processing_seconds_limit': preview_seconds,
                    'cache_hit': False,
                }
                if use_cache:
                    save_result(cache_root(), key, st.session_state.analysis_result)
                update_progress('Complete', 1., 'Results are ready below')
                status_box.update(label='Analysis completed', state='complete', expanded=False)
        except Exception as exc:
            status_box.update(label='Analysis failed', state='error', expanded=True)
            st.error(f'Could not analyze video: {exc}')

# Save each successful analysis by source so analyzing a second video does not overwrite the first.
if st.session_state.analysis_result is not None:
    completed = st.session_state.analysis_result
    if completed.get('source_hash'):
        st.session_state.comparison_results[completed['source_hash']] = completed

# Render from session state, not inside the Analyze button branch. All Streamlit
# widget interactions (including downloads) can rerun the script safely.
result = st.session_state.analysis_result
if result is not None:
    if uploaded is None or hashlib.sha256(uploaded.getvalue()).hexdigest() != result['source_hash']:
        st.info('The uploaded video changed. Click Analyze climb to produce results for the new video.')
    else:
        df, info = result['df'], result['info']
        metrics, events = result['metrics'], result['events']
        holds_df, near_df = result['holds_df'], result['near_df']
        wall_df = result['wall_df']
        st.subheader('Tracking results')
        st.caption('Source: ' + ('Disk cache — inference skipped' if result.get('cache_hit') else 'Fresh analysis'))
        st.caption(f"{result['source_name']} · {result['model_name']} · {result['sample_fps']} pose FPS")
        a, b, c = st.columns(3)
        a.metric('Frames analyzed', info['frames_analyzed'])
        b.metric('Tracking coverage', f"{df['detected'].mean():.0%}")
        c.metric('Clip analyzed', f"{info['seconds_analyzed']:.1f}s")
        with st.expander('Processing details', expanded=False):
            st.caption(f"Video export: {info.get('export_fps', 'unknown')} FPS · Pose sampling: {info.get('sample_rate_fps', 'unknown')} FPS · SAM sampling: {info.get('sam_sample_rate_fps', 'off')} FPS ({info.get('sam_anchor_frames', 0)} anchor frames) · Hold tracker: {info.get('hold_tracking_mode', 'unknown')}")
            st.caption('A detected pose does not guarantee each joint is correctly placed.')
        st.subheader('Annotated video — watch here')
        st.video(result['video_bytes'], format='video/mp4')
        if not result['converted']:
            st.warning('H.264 conversion was unavailable, so this is the original MP4V file. Some browsers cannot play it inline. Download the file or install the dependencies from requirements.txt. Conversion issue: ' + str(result['conversion_warning']))
        st.download_button('Download annotated video', data=result['video_bytes'],
                           file_name='climbvision_v0_10_7_annotated.mp4', mime='video/mp4',
                           on_click='ignore', key='download_annotated')
        if df['detected'].mean() < 0.3:
            st.warning('Very low valid pose coverage. Expand or reposition the crop and try again. Do not interpret the motion measurements as technique analysis.')
        with st.expander('Route holds and tracking details', expanded=False):
            if result['hold_enabled']:
                st.subheader('Reviewed route holds')
                st.caption('These boxes were manually annotated on the first frame; no automatic hold recognition was used.')
                st.metric('Manually selected holds', len(holds_df))
                if not holds_df.empty:
                    st.dataframe(holds_df, hide_index=True, width='stretch')
                    st.download_button('Download manually selected holds CSV', holds_df.to_csv(index=False).encode(),
                                       'climbvision_holds.csv', 'text/csv', on_click='ignore')
                if not wall_df.empty:
                    st.subheader('Wall tracking reliability')
                    st.metric('Frames with confirmed wall alignment', f"{wall_df['aligned'].mean():.1%}")
                    st.dataframe(wall_df, width='stretch', hide_index=True)
                    st.download_button('Download wall tracking diagnostics', wall_df.to_csv(index=False).encode(),
                                       'climbvision_wall_tracking.csv', 'text/csv', on_click='ignore')
                    st.caption('This is alignment availability, not verified hold-detection accuracy. Holds are hidden when alignment fails.')
                st.subheader('Manual hold annotation evaluation — reference frame')
                st.caption('Create ground-truth bounding boxes for the FIRST video frame only. Coordinates refer to the resized analysis frame (maximum dimension 960 px), not the original upload. The evaluation uses one-to-one IoU matching at 0.5; this is NOT video tracking accuracy.')
                if not holds_df.empty:
                    template = ground_truth_template(holds_df)
                    st.download_button('Download ground-truth starter CSV', template.to_csv(index=False).encode(), 'climbvision_ground_truth_starter.csv', 'text/csv', key='gt_template', on_click='ignore')
                    st.caption('Edit the CSV to describe all true route holds, including any missed detections, then upload it. Do not simply accept proposed boxes as ground truth.')
                    gt_upload = st.file_uploader('Upload manually corrected ground-truth CSV (x, y, w, h)', type=['csv'], key='groundtruth_upload')
                    if gt_upload is not None:
                        try:
                            truth = read_ground_truth(gt_upload.getvalue())
                            ev = evaluate_boxes(holds_df, truth)
                            a1, a2, a3 = st.columns(3)
                            a1.metric('Precision @ IoU 0.5', f"{ev['precision']:.1%}")
                            a2.metric('Recall @ IoU 0.5', f"{ev['recall']:.1%}")
                            a3.metric('F1 @ IoU 0.5', f"{ev['f1']:.1%}")
                            st.caption(f"Matched {ev['tp']} holds · {ev['fp']} false positives · {ev['fn']} missed holds. This compares the reviewed/suggested initial boxes only.")
                        except (ValueError, KeyError) as error:
                            st.error(f'Invalid ground-truth CSV: {error}')
                st.subheader('Possible hand/foot proximity to holds')
                st.caption('Near means close in the 2D image only. It does NOT confirm gripping, stepping, loading or physical contact.')
                if near_df.empty:
                    st.info('No close landmark/hold pairs passed the distance threshold.')
                else:
                    st.dataframe(near_df, hide_index=True, width='stretch')
                    st.download_button('Download proximity CSV', near_df.to_csv(index=False).encode(),
                                       'climbvision_proximity.csv', 'text/csv', on_click='ignore')
        with st.expander('Movement events and timeline', expanded=False):
            st.subheader('Climbing movement events (experimental)')
            st.caption('Hand/foot movements are projected landmark motion, not confirmed hold changes. Low-motion intervals are not necessarily rests. Missing or unreliable tracking produces missing events.')
            if events.empty:
                st.info('No events passed the conservative filters. This can happen with slow motion, short events, or partial tracking.')
            else:
                st.dataframe(events, width='stretch', hide_index=True)
            st.download_button('Download event timeline (CSV)', events.to_csv(index=False).encode(),
                               'climbvision_events.csv', 'text/csv', on_click='ignore')
        with st.expander('Movement charts and observations', expanded=False):
            st.subheader('Hand and foot speeds')
            st.caption('Normalized image dimensions per second (camera-dependent). Gaps indicate missing data.')
            st.line_chart(metrics.set_index('time_s')[[f'{p}_speed' for p in ('left_hand','right_hand','left_foot','right_foot')]])
            st.subheader('Hip trajectory')
            st.line_chart(df.set_index('time_s')[['hip_x', 'hip_y']])
            st.subheader('Joint angles (2D)')
            st.line_chart(df.set_index('time_s')[['left_elbow', 'right_elbow', 'left_knee', 'right_knee']])
            st.subheader('Observations and review prompts')
            for tip in make_feedback(df):
                st.write('• ' + tip)
        st.download_button('Download measurements (CSV)', metrics.to_csv(index=False).encode(),
                           'climbvision_metrics.csv', 'text/csv', on_click='ignore')


# Transition to a new analysis *below* saved results, without erasing them.
completed_results = st.session_state.comparison_results
if completed_results and not st.session_state.second_mode:
    current = st.session_state.analysis_result
    if current is not None and current.get('source_hash') in completed_results:
        st.divider()
        if st.button('Analyze another video', key='analyze_second_video'):
            st.session_state.first_attempt_hash = current['source_hash']
            st.session_state.second_mode = True
            st.session_state.analysis_result = None
            choice = st.session_state.get('video_source_choice', SOURCE_OPTIONS[0])
            if choice == SOURCE_OPTIONS[0]:
                st.session_state['pending_source_choice'] = SOURCE_OPTIONS[1]
            elif choice == SOURCE_OPTIONS[1]:
                st.session_state['pending_source_choice'] = SOURCE_OPTIONS[0]
            st.rerun()

if st.session_state.second_mode and st.session_state.first_attempt_hash:
    first = completed_results.get(st.session_state.first_attempt_hash)
    current = st.session_state.analysis_result
    if first is not None and current is not None and current.get('source_hash') != first.get('source_hash'):
        st.divider()
        st.subheader('Compare the two analyzed attempts')
        st.caption('Both annotated videos are shown together. Shared Play starts both at the beginning; it does not align climbing movements by hold events.')
        from synchronized_player import synchronized_videos
        synchronized_videos(first['video_bytes'], current['video_bytes'],
                            first['source_name'], current['source_name'])
        left, right = st.columns(2)
        for col, name, entry in ((left, 'Attempt A', first), (right, 'Attempt B', current)):
            with col:
                coverage = entry['df']['detected'].mean() if not entry['df'].empty else 0
                st.metric(name + ' pose detection coverage', f'{coverage:.0%}')
                st.caption(f"{entry['source_name']} · {entry['model_name']} · {entry['info'].get('seconds_analyzed', 0):.1f}s")
                st.download_button('Download ' + name + ' movement CSV',
                    entry['metrics'].to_csv(index=False).encode(),
                    file_name=name.lower().replace(' ', '_') + '_movement.csv',
                    mime='text/csv', key='comparison_csv_' + name.lower().replace(' ', '_'))
        st.caption('Detection coverage is not pose accuracy. Movement-event synchronization is not yet implemented.')
    elif first is not None and current is None:
        st.info('Attempt A is saved above. Choose another video, then click Analyze climb to add Attempt B below it.')


if uploaded and working_bytes is not None and 'roi' in globals():
    with st.expander('Compare pose backends (benchmark)', expanded=False):
        st.caption('Runs MediaPipe Full and YOLO26 Small on the SAME cropped frames. Reports detection coverage, joint availability, jitter proxy (median joint speed) and inference runtime. This is not ground-truth pose accuracy. YOLO has no toe landmarks.')
        bench_seconds = st.slider('Benchmark first N seconds', 3, 30, 10, key='pose_benchmark_seconds')
        bench_fps = st.slider('Benchmark sampled FPS', 2, 15, 5, key='pose_benchmark_fps')
        if st.button('Run pose backend comparison', key='run_pose_comparison'):
            try:
                with st.spinner('Comparing models on identical frames...'):
                    with tempfile.TemporaryDirectory() as workdir:
                        ext = '.mp4' if color_metadata['converted'] else Path(uploaded.name).suffix.lower()
                        source = str(Path(workdir) / ('comparison_input' + ext))
                        Path(source).write_bytes(working_bytes)
                        summary, details = benchmark_models(source, roi, get_model('Full (recommended)'), bench_fps, bench_seconds, yolo_device)
                        st.session_state.pose_benchmark_result = (summary, details)
            except Exception as error:
                st.error(f'Pose benchmark failed: {error}')
        if 'pose_benchmark_result' in st.session_state:
            summary, details = st.session_state.pose_benchmark_result
            st.dataframe(summary, use_container_width=True)
            st.download_button('Download pose comparison summary CSV',summary.to_csv(index=False).encode(), 'climbvision_pose_comparison.csv','text/csv')
            st.download_button('Download per-frame pose comparison CSV',details.to_csv(index=False).encode(), 'climbvision_pose_comparison_frames.csv','text/csv')
