// Pause decorative hero playback when the visitor requests reduced motion.
if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) {
  const heroVideo = document.querySelector('.hero-media video');
  if (heroVideo) { heroVideo.removeAttribute('autoplay'); heroVideo.pause(); }
}
