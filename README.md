# ClimbVision research portfolio website

A static, responsive website for showcasing the ClimbVision computer vision prototype to prospective research supervisors. No Python environment or GPU is required for a visitor.

## Preview locally

From the directory containing `index.html`:

```bash
python3 -m http.server 8000
```

Open http://localhost:8000 in your browser.

## Publish with GitHub Pages

1. Create a public GitHub repository (for example, `climbvision-site`).
2. Upload **the contents of this folder**, keeping `index.html` in the repository root and `assets/` beside it.
3. In repository Settings > Pages, select Deploy from a branch, select `main` and `/ (root)`, then Save.
4. Once GitHub finishes publishing, open the URL shown on the Pages settings screen. Add **that verified URL** to your academic CV.

Alternatively deploy the same folder on Netlify or Vercel as a static site.

## Before sharing with a professor

- Verify the embedded video plays on mobile and desktop.
- Replace the sample video if you prefer a shorter, more representative climbing sequence.
- Add actual benchmark results only after completing experiments; this website intentionally has no fabricated measurements.
- Publish the ClimbVision software in a separate, cleaned-up GitHub repository before advertising it as a maintained open-source project. The source ZIP here is an interim snapshot of v0.9.1.
- Ensure everyone identifiable in the climbing footage has consented to public distribution. If not, replace the video before publishing.
- Optionally remove the source ZIP from `assets/` and the corresponding buttons until the source is ready for release.

## File structure

- `index.html` — content and sections
- `styles.css` — responsive design
- `script.js` — minimal progressive enhancement
- `assets/climbvision-demo.mp4` — real user-provided annotated demo footage
- `assets/demo-poster.jpg` — still from the demo
- `assets/climbvision-source-v0.9.1.zip` — source snapshot

## Scope

This is a public **project showcase**, not a deployment of the compute-intensive Streamlit analysis pipeline. It does not perform live video inference or promise validated coaching accuracy.
