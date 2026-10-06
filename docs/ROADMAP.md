# Roadmap

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

Phase 7 — tagger engine
- Tagger interface and lazy optional runtime loading.
- WD14-compatible ONNX backend.
- Model cache and batch inference.

Phase 8 — tag cleaning and subject injection
- Normalize tags, blacklist unwanted tags, deduplicate tags.
- Treat subject as canonical dataset identity.

Phase 9 — caption and trigger system
- Deterministic caption writer.
- Derive trigger from subject by default.
- Idempotent trigger injection.

Phase 10 — build orchestration
- End-to-end build command.
- Stage skipping, resume, progress, and stage summaries.

Phase 11 — export and packaging
- Plain image+txt export.
- Kohya-compatible dataset output.
- ZIP packaging and manifest inclusion.

Phase 12 — incremental processing
- Add-only scrape runs.
- Skip already processed/tagged images.
- Reconcile manifest state safely.

Phase 13 — tests
- Unit and integration coverage.
- Pipeline fixtures.
- Failure/retry/resume tests.
- Termux-specific validation.

Phase 14 — benchmark
- Measure time, RAM, disk usage, scraper throughput, and tagger throughput.
- Establish practical defaults for Android/Termux.

Phase 15 — documentation
- Termux install guide.
- Full CLI reference.
- Dataset layout, config, tag rules, and troubleshooting.

Phase 16 — v1.0 release
- Freeze the supported feature set.
- Run clean-install and end-to-end acceptance tests.
- Publish release notes and known platform constraints.