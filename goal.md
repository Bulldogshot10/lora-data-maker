# Project Goal and Handoff

Last reviewed: 2026-10-25. The current README describes v2.5.1; `docs/ROADMAP.md` has older phase/version framing, so reconcile both when updating status.

## Goal
Deliver a Termux-first and Linux-compatible dataset preparation toolkit for LoRA training: scrape, validate, deduplicate, quality-filter, crop/resize, tag or caption, export, and package datasets. Model training and image generation are outside this project's scope.

## Feature inventory

### Implemented / present in current project documentation and source
- CLI and WebUI, configuration loading, manifest-backed per-image state, resumable stages, incremental processing, and raw/manifest reconciliation.
- Scrapers for Google Images through local Chromium/CDP authentication, direct URLs, and documented Bing date filtering.
- Image validation, format/orientation handling, exact SHA-256 deduplication, perceptual near-duplicate detection, and quality filtering with rejection metadata.
- Processing/cropping/resizing, optional face-aware processing, standardized outputs, and preservation of raw source images.
- WD14-compatible ONNX tagging with documented `sweetspot` and `potato` presets, pinned artifacts, and SHA-256 verification.
- Deterministic tag cleaning, subject injection, trigger handling, and caption writing.
- Local/API VLM captioning paths, build orchestration through packaging, plain/Kohya-style exports, and ZIP packaging with a manifest.
- Unit/integration coverage and offline end-to-end acceptance tests exist in `tests/`; Termux-specific manual validation is documented.

### Partial / needs verification
- Full authenticated Google scraping and interactive Chromium/CDP behavior must be tested on the target Termux/X11 device; mocks or CI smoke tests alone do not prove the live login flow.
- Termux/device performance benchmarks were explicitly deferred in `docs/ROADMAP.md`; record actual device, model, timings, and memory use when run.
- Reconcile the older phase roadmap with current v2.5.1 behavior so it does not mislabel newer features as missing.

### Not implemented / deferred
- Direct-device benchmark phase is explicitly deferred until target Android/Termux hardware testing.
- Training LoRA weights and generating images are non-goals for this repository.

## Handoff notes for the next agent
1. Read `docs/ARCHITECTURE.md`, `docs/CLI.md`, `docs/CONFIG.md`, `docs/INSTALL_TERMUX.md`, and `docs/TROUBLESHOOTING.md` for the relevant change.
2. Preserve raw-image immutability, deterministic manifest transitions, resumability, and audit reasons.
3. Exercise both core-smoke and WebUI-smoke workflows when relevant; inspect the real failing step instead of guessing from the final CI badge.
4. Never log credentials/cookies or commit login state. Keep secrets in the intended local credential mechanism.
5. Update this file and `docs/ROADMAP.md` together when planned phases or shipped capabilities change.

## Validation
Relevant checks are defined in `.github/workflows/core-smoke.yml`, `.github/workflows/web-smoke.yml`, and `tests/`. No fresh CI pass or live-device run is claimed here.
