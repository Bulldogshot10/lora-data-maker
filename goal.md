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

## Session checkpoint / handoff (2026-10-25)

### Current task and scope
Fix Playwright-Termux scraping on the current `main` branch, specifically the user-reported Google/X11 run where `Page.navigate` times out and the scraper then scrolls with zero candidates. Do not create or switch branches. Keep changes limited to scraper/browser behavior, regression tests, and directly related documentation.

### Progress and changed files
- `src/lora_data_maker/scrapers/playwright.py`: expanded Playwright-Termux usage for persistent Google/Bing Chromium profiles, locator-based dynamic result waits and lazy-load polling, request/response monitoring, and blocking common analytics URLs. Google parsing prefers source URLs (`ou` / `imgres?imgurl=`) and explicit source URL attributes rather than arbitrary thumbnail `img/src`. Added navigation-ack timeout recovery by checking live browser state, Google challenge/consent/blocked-page detection, and live DOM source-URL extraction.
- `src/lora_data_maker/scrapers/existing_chromium.py`: supports passing an explicit headless choice through the environment-based Google browser configuration.
- `src/lora_data_maker/scrapers/manager.py`: applies Google-only dimension/blur quality gates and removes quality-rejected URLs from the archive so those URLs can be reconsidered.
- `src/lora_data_maker/scrapers/archive.py`: added `discard(url)` for removing a rejected source URL from the persistent archive.
- `tests/test_scraper.py`: added/updated regression cases for Google original URLs, quality rejection, persistent Bing profile, network control, dynamic locator waits, CDP navigation timeout recovery, blocked-page handling, `imgres` extraction, and live DOM URL extraction.
- Related docs updated: `README.md`, `docs/INSTALL_TERMUX.md`, `docs/TROUBLESHOOTING.md`, and `RELEASE_NOTES.md` to describe the new scraping behavior.

### Feature inventory
**Implemented in source, needs runtime verification**
- Playwright-Termux network request/response monitoring and common analytics URL blocking.
- Persistent provider-specific browser profiles and configurable headless mode.
- Dynamic result-card waits and scroll/lazy-load polling via adapter locator APIs.
- Google source-URL extraction plus pre-ingestion quality screening.
- Recovery from a delayed `Page.navigate` acknowledgement when the live browser has navigated, and a manual-wait path for recognized Google consent/verification/blocked pages.

**Partial / needs verification**
- Current Google/X11 failure is not yet confirmed fixed on the user's actual Android/Termux device. Google changes result markup frequently; test the live DOM path and inspect the returned page URL/title/body text if zero candidates persist.
- The blur threshold (100) and minimum dimensions (384x384) are heuristics and may need calibration against a representative image set.
- The persistent authenticated browser and WireGuard-specific browser profile behavior require device-side confirmation.

**Not implemented / not claimed**
- No successful live scrape or device test is claimed.
- No full test-suite or CI pass is claimed for these latest commits.
- No claim that Google verification can be bypassed; the intended behavior is to wait for a person to complete it.

### Bugs and errors observed
User's exact runtime log included:
`Playwright google navigation failed: Timeout waiting for CDP command: Page.navigate`
followed by `scroll=1/30 candidates=0` through `scroll=4/30 candidates=0`, and finally `Google image search returned no parseable original image URLs`.
The new code attempts to check the actual page state after that specific CDP acknowledgement timeout, but this remains a hypothesis-driven fix until reproduced/verified on device.

### Validation
- Inspected the root `AGENTS.md`, current `goal.md`, relevant scraper sources, adapter APIs, tests, and troubleshooting/install documentation.
- Confirmed the newly added regression-test definitions are present on `main`.
- GitHub combined-status lookup for the checked commit returned no status entries; no fresh CI pass or live-device run was available/verified in this session.
- Tests not yet run locally: `python -m compileall -q src tests`; focused scraper tests; full `pytest -q -m "not real_caption_runtime"`; live WebUI scrape on Termux/X11 with and without WireGuard.

### Next steps
1. Read root `AGENTS.md` and `goal.md` before continuing. The required root `agent.md` was queried at the repository root but GitHub returned 404; locate it if it exists under a different path, otherwise proceed only using available root and docs guidance.
2. Pull latest `main` into a local checkout and run `python -m compileall -q src tests`.
3. Run focused tests in `tests/test_scraper.py` for navigation timeout recovery, blocked verification pages, original URL extraction, network callbacks, and quality filtering. Resolve actual test failures before saying the behavior is fixed.
4. Run the core smoke test suite and inspect its real logs/status.
5. Re-test on Termux/X11. If it still returns zero candidates, capture the current URL, document title, a safely redacted short visible body excerpt, the relevant DOM `a[href]` and `data-*` attribute names (not session cookies/tokens), and network error summary to adapt the parser to the actual page structure.
