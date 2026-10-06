# Dependency Policy

Status: Phase 0 locked

The default installation must be small enough for Termux.

## Required runtime

- Python 3.12+
- Pillow
- requests

## Optional runtime

opencv: image processing and face detection backends that need OpenCV.

tagger: ONNX Runtime for local WD14-compatible tagging.

gallery: gallery-dl integration for supported external sources.

dev: pytest and future development tooling.

## Rules

1. No AI/training framework may be imported by the core package.
2. Optional packages must be imported lazily inside the feature that needs them.
3. A missing optional dependency must produce an actionable error instead of a package-wide import failure.
4. No dependency should be added only for cosmetic CLI behavior.
5. Termux-native packages are preferred when they avoid fragile source builds.
6. Version constraints must be conservative and tested on the supported Python version before release.

## Important Termux constraint

Do not make opencv-python a hard dependency. Termux users may prefer the native Termux package for OpenCV. The package specification keeps OpenCV optional so installation strategy can be chosen per platform.
