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
- End-to-end `build` command from scrape through caption.
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

Phase 13 — tests
- Unit and integration coverage.
- Pipeline fixtures.
- Failure/retry/resume tests.
- Termux-specific validation.

Phase 14 — benchmark ⏭ skipped for now
- Deferred to direct device testing on the target Android/Termux hardware.

Phase 15 — documentation ✓ implemented
- Termux installation guide.
- Full CLI reference for all current commands.
- Configuration and dataset-layout references.
- Tag rules and trigger behavior documentation.
- Troubleshooting guide.

Phase 16 — v1.0 release
- Freeze the supported feature set.
- Run clean-install and end-to-end acceptance tests.
- Publish release notes and known platform constraints.