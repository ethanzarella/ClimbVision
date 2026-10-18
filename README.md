# ClimbVision

ClimbVision is a Python/Streamlit application for analyzing movement in climbing videos. It combines 2D pose estimation with manually annotated route holds and camera-motion compensation to produce annotated video and movement data. It also supports comparing two attempts and benchmarking pose models on the same frames.

## What it does

- Tracks a climber using MediaPipe Pose Landmarker (Lite, Full, or Heavy) or YOLO26 Small Pose.
- Lets the user select a tracking region and draw bounding boxes around holds on the first frame.
- Tracks the wall with OpenCV feature matching and geometric checks. Unreliable hold positions are flagged or hidden rather than treated as confirmed.
- Exports joint angles, hip trajectory, hand/foot movement estimates, hold proximity, event candidates, and wall-registration diagnostics.
- Produces annotated video and downloadable CSV files.
- Retains completed results for two different videos in one session and plays them side by side with shared playback controls. Playback is synchronized by time, not by movement or hold-contact events.
- Benchmarks MediaPipe Full and YOLO26 Small on the same decoded frames and crop region.

Route holds are **annotated manually**. The current workflow does not automatically recognize climbing holds or verify that a hand or foot is in physical contact with one.

## Run locally

Python 3.11 is recommended. You'll also need FFmpeg and `ffprobe` available in your PATH; on macOS they can be installed with Homebrew (`brew install ffmpeg`). HDR video conversion requires an FFmpeg build with the `zscale` filter.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

On other systems, use a supported Python executable and activate the virtual environment accordingly. The app opens in your browser. MediaPipe task files and YOLO weights are downloaded when a model is first selected, so the initial run requires an internet connection.

Two example climbing attempts are included in `demo_videos/` and can be selected directly in the app. Each has a corresponding hold annotation file that the app loads as a starting point. You can also choose **Upload your own video** and draw or import hold boxes for a different route.

| Bundled file | Purpose |
| --- | --- |
| `demo_videos/attempt_a.mp4` | First sample attempt |
| `demo_videos/attempt_b.mp4` | Second sample attempt |
| `demo_videos/attempt_a_holds.json` | Reviewed hold boxes for attempt A |
| `demo_videos/attempt_b_holds.json` | Reviewed hold boxes for attempt B |

The folder also contains `annotated_a.mp4`, `annotated_b.mp4`, and `climbvision_preview.mp4`, which are pre-rendered demonstration clips. The two annotated attempts are examples, not the output of a fresh analysis run.

## Analysis workflow

1. Choose a video and a pose model, then select the region containing the climber.
2. Draw boxes around route holds. Hold coordinates can be saved and reloaded as JSON for the same video resolution.
3. Select a wall-tracking mode. **Fast Preview** uses checked OpenCV registration; **Stationary camera** keeps the initial boxes fixed; the legacy OpenCV mode is a baseline. The SAM 2 option uses geometry-first wall tracking and can run optional mask diagnostics when SAM is installed separately.
4. Analyze the clip. Review the annotated video, coverage information, estimated joint angles, proximity measurements, and wall-tracking status.
5. Export results, or select **Analyze another video** for side-by-side playback and separate metrics.

The app caches completed analyses locally under `~/.cache/climbvision/analyses/`. Cached results can include copies of the annotated video and can be cleared from the app.

## Pose model comparison

The built-in benchmark runs both models on matching frames and the same crop. It saves a summary and per-frame CSV, including inference latency and the availability of selected landmarks.

One 150-frame comparison produced the following preliminary measurements:

| Measurement | MediaPipe Full | YOLO26 Small |
| --- | ---: | ---: |
| Mean inference latency | 21.9 ms | 71.2 ms |
| Left-wrist coverage | 77.3% | 80.0% |
| Right-wrist coverage | 66.7% | 96.7% |

These are observations from a single clip, not model accuracy scores. The benchmark CSV does not record the hardware or inference device, so the timing should not be treated as a reproducible cross-hardware comparison. Pose coverage indicates whether a landmark passed the detection/visibility checks; it does not show how close the estimate was to the real joint position. Hand-labeled landmarks and additional climbing videos are needed to evaluate accuracy and robustness.

YOLO's 17 COCO keypoints are mapped into the application's MediaPipe-style output. COCO does not include separate heel/toe landmarks, so those fields remain unavailable rather than being estimated. YOLO can use Apple's MPS backend on compatible Macs; cloud Linux deployments without supported GPU acceleration use CPU.

## Project structure

| File | Purpose |
| --- | --- |
| `app.py` | Streamlit interface, analysis flow, and exports |
| `analysis.py` | Frame processing, pose measurements, and movement events |
| `pose_backend.py` | MediaPipe/YOLO backends and frame-matched benchmarking |
| `reference_wall_tracker.py`, `wall_tracker.py` | Wall registration and motion checks |
| `holds.py`, `hold_rect_picker.py` | Hold data, proximity calculations, and annotation UI |
| `hdr_video.py`, `result_cache.py` | Video preprocessing and local result caching |
| `sam_tracker.py`, `hybrid_tracker.py` | Experimental tracking and mask-diagnostic code |
| `test_*.py` | Unit and regression tests |

## Testing

From the repository directory, with dependencies installed:

```bash
python -m pytest -q
```

Install `pytest` separately if needed. The included demo recordings are used by the bundled-demo regression tests.

## Limitations

This is an experimental 2D analysis tool, not a validated climbing coach. Camera motion, occlusion, perspective changes, and pose-estimation failures affect the measurements. Hold proximity is measured in image coordinates and is not proof of contact. Joint speed and angle outputs depend on which landmarks pass confidence filters, and the two-attempt viewer does not align individual climbing moves. SAM 2 mask diagnostics require an external installation and compatible checkpoints.

The application uses third-party models and libraries with their own licenses. Check those terms before redistributing model weights or deploying a commercial version.
