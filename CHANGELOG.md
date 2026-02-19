

```md
# Changelog

All notable changes to this project are documented here.

---

## v0.4.0 – MediaPipe Tasks & Dynamic Backgrounds

### Added
- Migrated face detection to MediaPipe Tasks API (BlazeFace)
- Added semantic segmentation using MediaPipe ImageSegmenter
- Implemented dynamic background replacement based on stable emotion
- Added emotion-based audio feedback on background transitions
- Added temporal smoothing for emotion probabilities
- Improved HUD with FPS and emotion bars

### Fixed
- Broadcasting errors in segmentation mask handling
- Background polarity inversion (foreground/background mask mismatch)
- MediaPipe running mode misuse (IMAGE vs VIDEO mode)
- Environment and dependency mismatches

---

## v0.3.0 – Emotion Pipeline Stabilization

### Added
- Emotion probability smoothing
- Emotion history buffer for stability
- Background transition debounce logic

### Fixed
- Emotion flickering issues
- Thread safety in emotion analyzer

---

## v0.2.0 – UI & Performance

### Added
- FPS counter
- HUD overlay for emotion probabilities
- Visual face bounding box

### Improved
- Frame processing performance
- Threaded emotion inference

---

## v0.1.0 – Initial Prototype

### Added
- Webcam capture
- Haar cascade face detection
- Basic emotion detection using DeepFace
