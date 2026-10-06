# v1.3.0

## Tagger presets

- Added the `sweetspot` WD14-compatible preset.
- Added the `potato` mobile QUINT8 SwinV2 preset.
- Pinned model revisions and SHA-256 verification through preset metadata.
- Updated configuration examples to use the current presets instead of the legacy EVA02 model.

## Scraper hardening

- Hardened Chromium/Selenium startup for Termux by removing the forced custom browser profile.
- Added bounded navigation, retry handling, and Google-to-Bing fallback behavior.
- Preserved candidate ordering across Chromium providers.

## Pipeline and paths

- Extended `build` orchestration through export and pack.
- Added support for absolute configured raw/work/output paths in manifest records.

## Historical v1.0.0 release

`v1.0.0` established the initial stable pipeline and is retained below for history.

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
