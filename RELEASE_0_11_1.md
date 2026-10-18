# ClimbVision v0.11.1 — Robust Wall Registration

Changes from v0.11.0:

- First-frame, keyframe and bidirectional optical-flow candidate registrations.
- Restricts registration features to the selected tracking region (which must cover the visible wall).
- Rejects clustered RANSAC matches, poor inlier ratios, high reprojection errors and implausible wall motion.
- Holds uncertain positions for at most five frames, marked as uncertain, and excludes them from hand/hold proximity measurements. After five missed registrations, hide holds.
- Expanded wall-tracking diagnostics CSV: registration source, confidence, inlier ratio, reprojection error, spatial coverage, frame displacement, and tracking status.
- Invalidate old cached analyses so the revised code actually runs.
- SAM diagnostics remain optional and off by default. No SAM mask may move confirmed hold boxes.

**Caveats:** The selected tracking region is also used as a wall feature mask; make it sufficiently broad to include fixed wall texture. The current dynamic suppression is conservative and is not a full person segmentation mask. Neither a synthetic feature-match test nor a valid-registration count establishes hold-location accuracy. Validate boxes visually, especially during occlusion and viewpoint change.

Run with the same existing Python environment and compare the downloaded wall diagnostics against v0.11.0.
