# v1.0.0

Initial stable release of lora-data-maker.

## Included

- Termux-first project initialization and manifest state tracking.
- Google Images and direct URL scraping.
- Image validation, exact and perceptual deduplication, and quality filtering.
- Pillow-based image processing with optional face-aware processing interface.
- WD14-compatible ONNX tagging with pinned model revision and checksum verification.
- Tag normalization, blacklist filtering, deduplication, and canonical subject injection.
- Deterministic caption generation with idempotent trigger handling.
- Full pipeline orchestration with stage ranges, skipping, resume behavior, and build summaries.
- Incremental raw/manifest reconciliation and manifest-aware scrape filtering.
- Kohya-style dataset export and ZIP packaging with manifest inclusion.
- Termux installation, CLI, configuration, dataset layout, and troubleshooting documentation.
- Apache License 2.0 licensing, third-party notices, and responsible-use guidance.

## Benchmark status

Phase 14 benchmarking is intentionally deferred to direct testing on the target
Android/Termux hardware.

## Known platform scope

Primary target: aarch64 Android running current Termux Python 3.12+.
Secondary target: regular Linux Python environments.

Optional native dependencies such as ONNX Runtime should be validated on the
target Termux installation before treating them as device-wide requirements.
