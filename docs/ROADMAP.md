# Roadmap

Status review: 2026-10-27. This roadmap describes the current `main` implementation rather than the original v1 phase boundary. Implemented means present in source and covered where noted; live browser/network behavior still requires target-device verification.

Phase 0 — architecture and dependency lock
- Define scope and non-goals.
- Lock the raw → work → output lifecycle.
- Lock dependency tiers and optional-backend rules.
- Define project config, subject/trigger semantics, and manifest requirements.

Phase 1 — core framework
- CLI entrypoint, config loading, paths, logging, errors, utilities.
- Add init/info commands.

Phase 2 — dataset/project model
- Dataset, image record, processing result, tag record, and manifest models.
- Durable per-image state and resume support.

Phase 3 — scraper engine ✓ implemented
- Scraper interface and manager.
- Google Images backend, direct URL backend, retries, timeout, delay, archive, resume.
- Project manifest records successful downloads and scrape summary.

Phase 4 — image validation ✓ implemented
- Decode validation, format handling, dimensions, file size, orientation, metadata.
- Added standalone `validate` CLI command and manifest status updates.

Phase 5 — deduplication and quality ✓ implemented
- SHA-256 exact deduplication.
- Perceptual-hash near-duplicate detection.
- Resolution, aspect-ratio, and blur/quality filters with auditable rejection reasons.
- Added standalone `clean` stage with persistent manifest metadata.

Phase 6 — image processing ✓ implemented
- Resize, crop, aspect-ratio handling.
- Standardized RGB JPEG output under `work/processed/`.
- EXIF orientation handling without modifying `raw/`.
- Optional face-aware processing behind a detector interface.

Phase 7 — tagger engine ✓ implemented
- Tagger interface and lazy optional runtime loading.
- WD14-compatible ONNX backend using the 448x448 model contract.
- Pinned Hugging Face model revision with SHA-256 verification.
- Model cache and batch inference.
- Standalone `tag` stage writing tags and confidence/rating metadata to the manifest.

Phase 8 — tag cleaning and subject injection ✓ implemented
- Normalize tags, blacklist unwanted tags, deduplicate tags deterministically.
- Inject canonical `subject` exactly once at the front of each tagged image's tag list.
- Persist tag-cleaning metadata and expose standalone `clean-tags` stage.

Phase 9 — caption and trigger system ✓ implemented
- Deterministic UTF-8 caption writer under `work/captions/`.
- Derive trigger from subject by default, with explicit trigger support.
- Idempotent trigger injection and resumable captioned state.

Phase 10 — build orchestration ✓ implemented
- End-to-end `build` command from scrape through pack.
- Contiguous stage ranges plus repeated explicit `--skip` controls.
- State-aware resume through existing stage eligibility rules.
- Progress output and persisted per-stage before/after summaries in the manifest.

Phase 11 — export and packaging ✓ implemented
- Plain matching image + `.txt` export.
- Kohya-style repeat/subject directory output.
- ZIP packaging containing the exported dataset and `manifest.json`.

Phase 12 — incremental processing ✓ implemented
- Add-only scrape runs with manifest-aware source URL filtering.
- State-gated processing skips records already advanced to later stages.
- Safe raw/manifest reconciliation with missing-file audit metadata.

Phase 13 — tests ✓ implemented
- Unit and integration coverage across the pipeline stages.
- Offline end-to-end acceptance fixture.
- Failure, retry, state-regression, and incremental resume coverage.
- Termux-specific validation matrix documented for manual device execution.

Phase 14 — benchmark ⏭ skipped for now
- Deferred to direct device testing on the target Android/Termux hardware.

Phase 15 — documentation ✓ implemented
- Termux installation guide.
- Full CLI reference for all current commands.
- Configuration and dataset-layout references.
- Tag rules and trigger behavior documentation.
- Troubleshooting guide.

Phase 16 — v1.0 release ✓ implemented
- Freeze the supported feature set at v1.0.0.
- Run clean-install and offline end-to-end acceptance tests.
- Publish release notes, Apache 2.0 license, third-party NOTICE, responsible-use policy, and known platform constraints.

Phase 17 — v1.3 tagger presets ✓ implemented
- Sweetspot and potato WD14-compatible presets.
- Pinned model artifacts with SHA-256 verification.
- Mobile-friendly tagging path for constrained devices.

## Current implementation follow-up (v2.x)

- **WebUI configuration and project workflow: implemented in source.** Local workspace project creation/switching, upload and curation, settings persistence, asynchronous stage jobs, progress, scrape cancellation, gallery preview, ZIP download, and loopback-only request checks.
- **Caption runtime selection: implemented in source.** Deterministic tag captions; local VLM runtimes; Gemini and OpenAI-compatible API model discovery; API-key persistence separate from dataset config; VLM mode bypasses taggers.
- **Authenticated browser scraping: implemented in source.** Persistent Google/Bing Playwright browser profiles, original-source-first URL recovery with marked thumbnail fallback, date filters, and generic authenticated page image capture using a separate profile. Live provider markup can change; target Termux/X11 verification remains a separate task.
- **Optional WireGuard proxy: implemented in source.** The WebUI can store a WireGuard client config and route Playwright Google/Bing scraping through a temporary local `wireproxy` instance. Other scraper backends do not use this option. Provider access, routing behavior, and live stability need device verification.
- **Tagging presets: implemented in source.** `sweetspot`, `potato`, and `joytag`, with pinned artifacts/checksum checks and optional runtime dependencies.
- **Quality gates: implemented in source.** Exact/perceptual deduplication, image quality heuristics, optional person detection, and PP-OCRv3 text filtering. The text detector is English-focused.
- **Validation and benchmark: needs device verification.** Execute core/WebUI tests against the current commit and exercise Playwright browser scraping, optional WireGuard routing, local VLM, and mobile tagger runtime on the supported Termux device.
- **Open maintenance work.** Keep README, architecture, roadmap, config/CLI/install/troubleshooting docs and `goal.md` synchronized whenever WebUI settings, scraper capability, or caption runtime behavior changes. Add regression coverage for browser-side CSS and configuration-validation issues.
