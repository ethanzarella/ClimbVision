# ClimbVision browser-demo deployment (v0.10.3)

This release bundles two SDR MP4 demonstration clips (same climbing route) and keeps the existing Streamlit analysis pipeline. The recordings have been downscaled from original HLG HDR to SDR for browser playback.

## Local test

```bash
cd ~/Downloads/"climbvision 27"
source ~/Downloads/"climbvision 12"/.venv/bin/activate
python -m streamlit run app.py
```

If you extracted under another name, replace the folder path accordingly.

## Streamlit Community Cloud

1. **Obtain permission from all identifiable people in the demo videos** before publishing or committing them to a public repository. If permission is unavailable, use only footage of yourself or private deployment.
2. Push the contents of this `climbvision` folder to a GitHub repository. To keep the main portfolio separate, a new repository dedicated to the app is recommended. Include the `demo_videos/` folder and `roi_picker_ui/` files.
3. Open https://share.streamlit.io/ , connect GitHub, select the repository/branch, and set the app entry point to `app.py`. Choose Python 3.11 if offered.
4. The deploy installs `requirements.txt` and the system packages listed in `packages.txt`. The **built-in SDR videos do not require HDR conversion**. Custom HDR files will require an FFmpeg build supporting zscale; the base apt FFmpeg varies by platform.
5. After a successful deployment, copy the app's actual URL into your GitHub Pages site as a `Launch Interactive Demo` link. The app URL does **not** exist until you deploy; do not put a guessed link on the website.
6. Test in a private/incognito browser: built-in sample A/B, video previews, draggable ROI, manual holds, and the shortest live MediaPipe Full run. Community Cloud has limited CPU/RAM; YOLO/SAM availability/performance is not guaranteed. `ultralytics` is optional and is not installed automatically to keep deployment lighter.

## What the app demonstrates

- Two attempts on the same route, available without uploading a file.
- Side-by-side visual playback (independent video players; not frame synchronized).
- Existing individual-attempt body tracking, selectable route holds, and descriptive metrics.
- A research direction, **not yet implemented**: matched hold-contact events, time-aligned poses, and automated side-by-side movement-difference reports.

## Privacy and reproducibility

- Both built-in files are derived, compressed versions of recordings supplied by the project owner; preserve the original recordings elsewhere.
- Do not claim automatic judging or validated technique scores.
- Benchmark results must identify sample sizes, device, inference model, video segment, and definition of coverage.
