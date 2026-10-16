# v2.5.1 - 2026-10-16

## VLM captioning and WebUI

- Added optional local llama.cpp and ONNX VLM captioning plus Gemini and OpenAI-compatible API runtimes.
- Added per-user API-key storage outside dataset projects, with owner-only file permissions and a WebUI clear action.
- Expanded the WebUI with provider model discovery, safer upload limits, request-origin checks, and local-only access assumptions.

## Browser scraping

- Added authenticated Google Images scraping by attaching to an existing local Chromium/CDP session; verification challenges are not bypassed.
- Added Bing publication-date filtering by year or custom range in the CLI and WebUI.
- Hardened local CDP metadata discovery by rejecting redirects and bounding the response size.

## Public-repository readiness

- Added a repository-level security policy and documented credential, WebUI, custom API endpoint, and Chromium profile boundaries.
- Expanded the third-party dependency/model notice and corrected README guidance for credential persistence and manual CI workflows.

# v2.1.0

## WebUI

- Added the local WebUI dashboard shell, project explorer, project switching, image upload, pipeline controls, progress tracking, output download, and responsive mobile layouts.
- Added manual image removal from Recent Images for dataset cherry-picking, including cleanup of derived artifacts.
- Added scraper cancellation with preserved downloads completed before cancellation.
- Added the WebUI favicon and packaged static icon assets.

## Quality and runtime

- Fixed YOLOX-Nano preprocessing, output decoding, and model-contract validation for the pinned runtime.
- Added optional YOLOX person detection and YuNet face detection with pinned model revisions, SHA-256 verification, and local model caching.
- Added JoyTag ONNX support and mobile-friendly tagger presets with concurrency-safe model downloads.
- Improved quality-stage error handling, scraper resilience, manifest behavior, and tagger backend/preset consistency.

## Playwright image search

- Added separate `playwright-bing` and `playwright-google` backends using Termux-native Chromium through CDP.
- Pinned `playwright-termux==2.0.0` for deterministic Termux compatibility.
- Added bounded direct-CDP navigation to avoid the adapter's lifecycle waiter race.
- Added original-image URL extraction, Google verification detection, retry handling, and parser/config regression coverage.

## Scraper resilience

- Failed individual image downloads are skipped and the scraper continues with later candidates.
- Partial scrape completion now exits successfully when at least one new image was downloaded; all-failed runs still fail.
- Chromium mode replaces its Google provider with a DuckDuckGo-first, Bing-fallback chain.
- Bing candidates remain usable when the Bing browser stage fails after another provider already supplied candidates.
- Added regression tests for failed-download continuation and multi-provider fallback.

## Face crop

- Added an optional OpenCV YuNet face detector for CLI processing.
- Pinned the YuNet 2023mar model revision and SHA-256 checksum.
- Added a separate face runtime dependency tier so core Termux installs stay lightweight.
- Added CI coverage for the real detector runtime.

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
