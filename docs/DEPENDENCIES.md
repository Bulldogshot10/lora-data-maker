# Dependency Policy

Status: Phase 0 locked

The default installation must be small enough for Termux.

## Required runtime

- Python 3.12+
- Pillow
- requests

## Optional runtime

tagger: ONNX Runtime for local WD14-compatible tagging.

face: NumPy and OpenCV headless for the built-in YuNet face detector on regular Linux/Python.

dev: pytest and future development tooling.

## Rules

1. No AI/training framework may be imported by the core package.
2. Optional packages must be imported lazily inside the feature that needs them.
3. A missing optional dependency must produce an actionable error instead of a package-wide import failure.
4. No dependency should be added only for cosmetic CLI behavior.
5. Termux-native packages are preferred when they avoid fragile source builds.
6. Version constraints must be conservative and tested on the supported Python version before release.

## Important Termux constraint

Do not make opencv-python a hard dependency. Termux users should prefer the native Termux OpenCV package. The face extra is intended for regular Linux/Python environments; Termux can keep using the core package and install its native cv2 package separately.
