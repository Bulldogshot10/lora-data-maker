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


## Session checkpoint / handoff (2026-10-25, Google Images URL-mode fix)

### Current task and scope
On the user-selected `main` branch, fix the reported Playwright Google/X11 timeout in which `Page.navigate` acknowledgement times out, the browser records substantial network activity, and `_goto()` ultimately reports that it never reached a Google search/consent page. Keep the change limited to Google Images URL compatibility, regression coverage, troubleshooting guidance, and this handoff. No branch was created or switched.

### Progress and changed files
- `src/lora_data_maker/scrapers/playwright.py`: Google Images URL generation and its default startup URL now use `udm=2`; navigation readiness accepts both `udm=2` and legacy `tbm=isch` so redirects/session URLs remain compatible.
- `src/lora_data_maker/scrapers/existing_chromium.py`: the default Google startup URL now uses `udm=2` too.
- `tests/test_scraper.py`: URL-builder and default-profile assertions updated to `udm=2`; browser launch URL expectations updated; the navigation-acknowledgement regression test exercises both modern and legacy Google Images URLs.
- `docs/TROUBLESHOOTING.md`: documented the new URL mode and the false-timeout failure mode this compatibility handles.
- `goal.md`: checkpoint added; prior project goal and unresolved device-validation notes retained.

### Feature inventory
**Implemented in source, pending test execution**
- Google Images navigation recognizes current `udm=2` and legacy `tbm=isch` URL modes.
- Google Images URL builder requests `udm=2`.
- Regression coverage exists for both URL forms when the `Page.navigate` CDP acknowledgement times out.

**Needs verification**
- Python compile check and focused scraper tests have not been executed in this environment.
- Live Google scraping on the user's Termux/X11 device has not been verified; this fix targets the URL-mode mismatch indicated by the current detector, not an independently reproduced device session.
- No CI pass or successful live scrape is claimed.

### Bugs and errors
The user reported `scrape: browser network summary requests=145 responses=144` followed by `Playwright google search failed after 4 attempts: Playwright google navigation timed out after 20s (Google did not reach a Google search/consent page; last URL=...)`. The URL suffix was truncated in the provided log, so the precise final URL could not be confirmed. A root cause consistent with the source and current Google URL conventions was that `_goto()` accepted only `tbm=isch` even though Google Images uses `udm=2`; the patch handles both.

### Validation
- Read root `AGENTS.md` and `goal.md`; checked the root listing and confirmed the required `agent.md` file is absent from `main` (GitHub returned 404).
- Inspected the Google URL builder, navigation-readiness logic, source-URL extraction, regression-test definitions, and troubleshooting guidance.
- A local `git clone` attempt could not proceed because this environment could not resolve `github.com`; therefore local pytest/compile checks were not run.
- No live-device test or successful CI run was available in this session.

### Next steps
1. In a checkout of `main`, run `python -m compileall -q src tests`.
2. Run `pytest -q tests/test_scraper.py -k "playwright_google_builds_direct_google_images_url or recovers_when_page_navigate_ack_times_out"` and then the broader scraper test file.
3. Re-run the Google scraper on Termux/X11. If the timeout persists, capture the complete final URL from the error and inspect the current document title/body marker plus relevant source-link attributes; do not infer that the live-device bug is fixed from mock tests alone.


## Documentation follow-up (2026-10-25)

The user restored the required root `agent.md` after the previous checkpoint. It is now present on `main` (blob SHA: `943c50dca0ac2eb15f8b57825aa7568be94d9ee8`) and has been read. The earlier checkpoint's note that `agent.md` was missing records the situation at that earlier inspection; that documentation blocker is resolved as of this follow-up. No source-code or runtime status is changed by this documentation update.


## Session checkpoint / handoff (2026-10-25, Google thumbnail fallback)

### Scope
On the existing main branch, fix the reported case where Playwright Google exhausts retries with no parseable original image URLs. Preserve original-source URLs as the first choice, try click-based browser automation on rendered Google result cards, then download card thumbnail URLs only when source extraction and browser-assisted recovery both fail. Do not create or switch branches.

### Changes
- src/lora_data_maker/scrapers/playwright.py: add browser-assisted result-card clicks and repeated source URL parsing; add live DOM thumbnail extraction with a result-card-scoped HTML fallback; mark thumbnail results with google_thumbnail_fallback and original_url_unavailable metadata; improve the final no-candidate diagnostic.
- src/lora_data_maker/scrapers/manager.py: bypass the original-source dimensions/blur quality gate only for results explicitly marked as thumbnail fallback so the requested thumbnail can actually be retained.
- tests/test_scraper.py: add regressions for browser click recovery, live DOM thumbnail extraction, HTML thumbnail fallback, end-to-end candidate metadata, and quality-gate bypass for marked fallback thumbnails.
- README.md, docs/TROUBLESHOOTING.md, and RELEASE_NOTES.md: document the new source-first recovery order and thumbnail fallback behavior.

### Feature status
Implemented in source: original URL parsing remains first; on no original candidates the scraper attempts browser interaction; if that does not recover original URLs, it extracts thumbnail URLs and marks them explicitly; manager retains those marked thumbnails without applying the source-image quality gate.

Needs verification: compileall and pytest have not been run in this environment; live Termux/X11 Google scraping has not been tested. The fallback depends on current Google result-card markup and should be validated against real page DOM.

### Validation
- Read AGENTS.md, agent.md, goal.md, relevant scraper/manager/downloader source, existing tests, README, troubleshooting docs, and release notes.
- Added regression tests but did not execute them in this environment; no CI or device success is claimed.
- This fallback applies when original URL parsing yields no candidates; it does not switch to thumbnails merely because an already-parsed original image URL later fails to download.

### Next steps
1. In Termux, run python -m compileall -q src tests.
2. Run python -m pytest -q tests/test_scraper.py -k "browser_click_recovery or thumbnail_fallback or does_not_fallback_to_rendered_thumbnails".
3. Run python -m pytest -q tests/test_scraper.py and inspect any actual failing traceback before adjusting code.
4. Repeat the original Google/X11 query. Confirm logs show browser-assisted clicks, then a marked thumbnail fallback if original URLs still cannot be recovered; confirm the fallback image is downloaded and appears in the raw dataset.


## Session checkpoint / handoff (2026-10-25, Playwright evaluate execution fix)

### User-reported failure after previous patch
The user retried the updated scraper and still saw no results:
`Playwright google search failed after 4 attempts: Playwright google image search returned no parseable original or fallback thumbnail image URLs`.
They report the scraper continues scrolling and produces no candidate.

### Root-cause finding
Inspection of the pinned `playwright-termux` Page.evaluate implementation shows it wraps source strings beginning with `() =>` as a function expression but does not invoke that function. The previous code passed arrow-function strings for live source extraction, click recovery, and thumbnail DOM scanning. Those expressions could therefore return no serializable array / no truthy click result, leaving fallback dependent on the narrow HTML card selectors. This explains why the prior fallback could fail despite being called.

### Changes in this follow-up
- src/lora_data_maker/scrapers/playwright.py: converted live source extraction, browser-click recovery, and thumbnail scanning scripts to immediately invoked arrow expressions (`(() => {...})()`). Browser click recovery also probes eligible image elements when known card classes are absent. Thumbnail extraction now scans all live `img` nodes, checks known Google thumbnail hosts and named/rendered images, filters obvious branding, and prints explicit counts/failure diagnostics. The rendered-HTML fallback also accepts known thumbnail hosts or named images with explicit dimensions.
- tests/test_scraper.py: added assertions that DOM and click scripts are immediate invocations and a regression for a thumbnail in unknown result-card markup.
- goal.md: recorded the user's observed failure and the concrete adapter-level cause; runtime validation remains pending.

### Feature status
**Implemented in source, needs verification:** live source extraction, browser click recovery, and thumbnail DOM scanning now execute as JavaScript expressions instead of returning arrow-function objects; generic thumbnail fallback is less dependent on Google CSS class names.

**Needs verification:** compileall, focused pytest, full scraper tests, and live Termux/X11 Google scraping have not run in this environment. The user's current device session is the authoritative check for whether the original case is fixed.

### Next steps
1. Pull latest main on Termux.
2. Run `python -m compileall -q src tests`.
3. Run `python -m pytest -q tests/test_scraper.py -k "browser_click_recovery or thumbnail_fallback or live_dom_when_outer_html_has_none"`.
4. Re-run the original query. Logs should now show whether browser click recovery runs and either how many live DOM/HTML thumbnails were extracted or an explicit "found no usable image URLs" diagnostic. If still empty, inspect a safe dump of current image src hostnames/alt/dimensions and result-link href patterns from the live DOM, excluding cookies/tokens.
