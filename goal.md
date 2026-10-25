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


## Session checkpoint / handoff (2026-10-25, cross-provider Playwright audit)

### User scope
Audit both Playwright Google and Playwright Bing because repeated narrow fixes did not resolve scraping. Identify source/adapter conflicts and repeatable bugs, not just add more selectors/retries. Target branch: main; no new branch.

### Audit findings and fixes in this commit
- Google fallback order was unsafe: browser automation clicked a result before caching thumbnail URLs. If the click navigated to a preview/detail URL, the later fallback inspected the wrong DOM. The search loop now caches thumbnails before clicking, tries to recover the original URL, and only selects the cached thumbnails if original recovery fails.
- Google original URL parsing now reads imgurl/mediaurl from the current /imgres URL. Browser-click recovery restores the search grid after navigation away so later scrolls do not remain on a detail page.
- Google browser recovery previously caught ScraperCancelledError through its ScraperError handler. Cancellation is now re-raised.
- Google and Bing dynamic result counts now have a DOM-based fallback when provider CSS classes stop matching.
- Bing parsing accepts structured image payloads when the iusc class changes, and selects turl as a marked fallback when murl is unavailable or points to a Bing cache URL.
- Added regression tests for Google /imgres extraction, thumbnail caching before clicks, cancellation propagation, Bing class changes, Bing turl fallback, and selector-independent DOM counts.
- docs/TROUBLESHOOTING.md updated with the current recovery order.

### Feature status
**Implemented in source, needs verification:** Google source-first -> browser recovery -> cached thumbnail flow; Google search-page restoration after result clicks; cancellation propagation; selector-independent DOM counts; Bing structured payload parsing and turl fallback.

**Needs verification:** Live Termux/X11 runs of both Google and Bing. Core smoke for audit commit `ebad4504a22fbeccaf7b36a5b1844da7ddad3fee` completed: compileall passed; pytest reported 372 passed, 2 failed, 2 skipped, 1 deselected. All failures were the same two pre-existing clean/quality tests below; no scraper test failed.

### CI baseline failure discovered
- Core smoke for ab688ce6a2ed1663623fdf41eb40c94f4c59c7f8 failed at Run tests: 366 passed, 2 failed, 2 skipped, 1 deselected. Compileall passed.
- The same two tests failed on the immediately preceding commit c2072848874d453d7c18588ea876eae892137891, so these failures predate this audit patch:
  - tests/test_clean.py::test_clean_exact_duplicate_and_quality_filter: expected exact_duplicates=1, but CleanSummary reported exact_duplicates=0 and quality_rejected=4.
  - tests/test_clean.py::test_clean_quality_rejection_happens_before_perceptual_dedup: expected the sharp fixture to be validated, but it was rejected.
- These are outside Playwright scope and remain a separate red-CI blocker. Do not claim the whole suite is green.

### Next steps
1. Check the core smoke run for this commit and inspect its exact failure summary.
2. Fix any new scraper-specific regression from its traceback; run tests/test_scraper.py focused and full.
3. Re-run Playwright Google on Termux/X11. Logs should show pre-click thumbnail caching, browser click attempts, and either recovered originals or selected cached thumbnails.
4. Run Playwright Bing with a normal murl and missing/invalid murl payloads; verify turl fallback downloads and metadata marks the fallback.
5. Diagnose the two already-failing tests in tests/test_clean.py separately after scraper validation.


### Validation update (2026-10-25)
- GitHub Actions core smoke for `ebad4504a22fbeccaf7b36a5b1844da7ddad3fee`: `python -m compileall -q src tests` passed; pytest reported 372 passed, 2 failed, 2 skipped, 1 deselected.
- Only failures: `tests/test_clean.py::test_clean_exact_duplicate_and_quality_filter` and `tests/test_clean.py::test_clean_quality_rejection_happens_before_perceptual_dedup`. Both also failed in earlier commits `c2072848874d453d7c18588ea876eae892137891` and `ab688ce6a2ed1663623fdf41eb40c94f4c59c7f8`; they are not caused by the current Playwright audit.
- Added a regression test for click navigation to Google `/imgres?imgurl=`, recovering the original source from the current URL, and restoring the search query. Core smoke for follow-up commit `46cbb92db80829057645165b11450ca8ee447640` completed: compileall passed; pytest reported 373 passed, 2 failed, 2 skipped, 1 deselected. The new regression passed; the same two pre-existing clean/quality failures remain.
- No live Termux/X11 run has been performed; do not claim the user's device failure is fixed until tested there.


## Session checkpoint / handoff (2026-10-25, clean quality-test threshold correction)

### User request
Resolve the remaining red CI tests after the Playwright Google/Bing audit. Target the existing `main` branch. Do not create or switch branches.

### Root cause
The CI log for run [40192945319](https://github.com/Bulldogshot10/lora-data-maker/actions/runs/40192945319) showed the two failures were both in `tests/test_clean.py` and happened before duplicate detection: test fixtures passed `blur_threshold=1000`. The current `laplacian_variance` implementation downsamples images to 256px and removes the one-pixel border before computing variance; with Pillow 12.3.0, the synthetic sharp fixture scores about 70.8 while the blurred fixture scores about 0.48. Therefore 1000 rejected every fixture, so exact dedup never saw eligible records and the sharp image in the order-of-operations test was also rejected.

### Change made
- `tests/test_clean.py`: corrected the two fixture thresholds from 1000 to 10. This keeps the sharp synthetic fixture valid while rejecting its blurred counterpart, matching what the tests are trying to verify.
- No production quality or deduplication behavior was changed to mask a bad test expectation.
- This is a regression-test calibration correction; it does not claim that the live Playwright Google/Bing device workflow has been validated.

### Validation status
- Inspected the exact failing CI log and reproduced the sharp/blurred Laplacian variance values with Pillow 12.3.0 in this environment.
- The failing tests and full core-smoke suite must be rerun against the resulting commit; results are pending at the time of this checkpoint.
- Compile/test runs are still the authoritative gate for this change. Live Termux/X11 Google and Bing behavior remains a separate device-verification item.

### Next steps
1. Confirm the commit triggers core-smoke and read its actual job log.
2. If the focused clean tests and full `pytest -q -m "not real_caption_runtime"` pass, record the exact counts and current commit below.
3. Keep the live Termux/X11 Google/Bing validation marked as needing verification until a device run is observed.


### Validation update (2026-10-25, clean test fix confirmed)

- Commit tested: `c568b54947a5f15ea865d08d2a65f15d9d0c2aaf` (`Fix clean quality test blur thresholds`).
- GitHub Actions core smoke run [40193793585](https://github.com/Bulldogshot10/lora-data-maker/actions/runs/40193793585) completed with **success**.
- `python -m compileall -q src tests`: passed.
- `pytest -q -m "not real_caption_runtime"`: **375 passed, 2 skipped, 1 deselected**, 1 warning. All CLI, documentation, and init/inspect smoke steps also passed.
- The two previous failures in `tests/test_clean.py` are resolved by calibrating the synthetic fixture threshold from 1000 to 10; exact deduplication and quality rejection now run on the intended eligible fixtures.
- Remaining verification boundary: CI's mocked/unit coverage is green, but live Google and Bing scraping on the user's Termux/X11 device is still **needs verification**. Do not represent the device-side scraper workflow as proven until it has been run there.
- Current state: no known failing test remains in the non-`real_caption_runtime` core suite. The two skips and one deselection are reported by the configured test command and are not failures.


## Session checkpoint / handoff (2026-10-25, Google original-image recovery and date filters)

### Scope
On the existing `main` branch, fix the Google Images scraper's tendency to choose thumbnail URLs when original image URLs are present after clicking a result, audit the original-vs-thumbnail extraction path, and enable date filtering for Playwright Google in the CLI, config, and WebUI. No branch was created or switched.

### Findings from source audit and online research
- The original URL parser matched only quoted `"ou"`/`'ou'` keys. Current Google Images payloads can be JavaScript object literals using unquoted `ou:`, so this was a concrete extraction gap.
- Click-based recovery waited a fixed 600 ms, then ran the same source parser. The live DOM path read source links and explicit data attributes but did not inspect a large image displayed in the clicked viewer. The recovery flow now polls briefly and inspects explicit source attributes and large non-grid viewer images before falling back to thumbnails.
- Google officially documents `after:` and `before:` operators for date-limiting indexed pages; Google Search Community also points to these operators for image search. The Google Images-specific legacy `tbs=cdr` date-range behavior varies across layouts, so this implementation uses operators inside the Google Images query.
- Google date range semantics are **indexed result-page dates**, not the image's guaranteed EXIF capture date.

### Changes committed
- `src/lora_data_maker/scrapers/playwright.py`: accepts unquoted `ou:` fields; after result clicks, polls for preview hydration and inspects explicit source attributes plus large non-grid viewer images; prevents generic `googleusercontent.com` cache/preview URLs discovered via viewer image src from being promoted to an original URL; accepts custom dates for Google and appends `after:`/`before:` operators.
- `src/lora_data_maker/scrapers/manager.py`: allows date ranges for Playwright Google and forwards them to the scraper; other backends still reject unsupported date filtering.
- `src/lora_data_maker/cli.py`: help text reflects Google and Bing support.
- `src/lora_data_maker/web.py`: date controls support Playwright Google and Bing with provider-specific explanatory text; API validation permits Google.
- `tests/test_scraper.py`: adds coverage for unquoted `ou`, viewer image extraction, rejecting Google-hosted viewer cache URLs as originals, Google query date operators, and manager date forwarding; adjusts old expectations that Google date filtering was unsupported.
- `tests/test_web_scrape.py`: adds a WebUI settings test for saving a Google custom date range.
- `README.md`, `docs/CLI.md`, `docs/CONFIG.md`, and `docs/TROUBLESHOOTING.md`: document the recovery and date-filter behavior.

### Validation update (2026-10-25)
- Latest core smoke run [40196435697](https://github.com/Bulldogshot10/lora-data-maker/actions/runs/40196435697) completed with **success** on commit `caa2b599c1847ea0a316b061b0dfc89cc8bc6a36`.
- `python -m compileall -q src tests`: passed.
- `pytest -q -m "not real_caption_runtime"`: **381 passed, 2 skipped, 1 deselected**, 1 warning. CLI, documentation, and init/inspect smoke checks also passed.
- Two earlier checks were corrected from actual CI traces: the date-query test now decodes URL query parameters before assertion, and the WebUI's unsupported-backend validation message retains its established compatibility wording.
- Live Google DOM in the user's Termux/X11 session is still **needs verification**. CI mocks cover the targeted extraction/parser/date-wiring paths but cannot prove that the user's current Google viewer exposes a full-resolution source URL.
- Google date filtering is wired through the CLI, config, manager, and WebUI using documented `after:`/`before:` query operators. These operators filter indexed result-page dates, not guaranteed original photo capture dates, and still need a live Google Images check.

### Next steps
1. Pull latest `main` on Termux.
2. Test a Google query and confirm logs show `ou` extraction or live preview DOM recovery before thumbnail fallback.
3. Test a custom Google date range and confirm the browser query contains the encoded `after:` and `before:` operators and results are filtered as expected.
4. If the session still only obtains thumbnails, capture safe DOM diagnostics after a tile click (source-attribute presence, preview image source host/dimensions, viewer link href patterns, current URL and title). Do not include cookies, tokens, or logged-in session details.


## Session checkpoint / handoff (2026-10-25, root-cause Google payload parser follow-up)

### User report
The earlier Google original-image recovery changes and green CI still did not change the user's live Termux:X11 behavior: `playwright-google` continued saving thumbnails only. The prior click/viewer-based recovery was insufficient; don't close this issue based on mock tests alone.

### Root cause found through online source research
Google Images can embed original image URLs inside `AF_initDataCallback` script payloads as positional arrays shaped like `["https://source.example/image.jpg", height, width]`, not necessarily `ou: "..." fields and not in result-image `img.src`. The scraper previously searched named `ou` keys, links with `imgurl`, explicit data attributes, and the clicked viewer. It could miss original URLs already present in the result HTML, then deterministically fall back to thumbnails. Background reference: [Google Images scraping research (July 26, 2026)](https://status.scrapebadger.com/blog/how-to-scrape-google-images-with-python-in-2026), especially its AF_initDataCallback and URL-dimension tuple sections.

### Changes committed in `73aa5069b07fc0d3e8a59c25d675440dcd0f7802`
- `src/lora_data_maker/scrapers/playwright.py`: parse URL + two dimension tuples from embedded Google payloads after normalizing JSON-escaped slashes, without relying on brittle nested array indices; reject Google-owned thumbnail/cache URLs from this tuple path; log the tuple-source count and the total original/source candidates before thumbnail fallback.
- `tests/test_scraper.py`: regression test mirrors an AF_initDataCallback payload containing plain and escaped original URLs, encrypted-tbn thumbnails, googleusercontent previews, and Google UI assets. The test asserts only the two external source URLs survive.
- `docs/TROUBLESHOOTING.md`: documents the payload tuple extraction path ahead of viewer clicking/fallback.
- This retains earlier date-filter support for Playwright Google.

### Validation update (2026-10-25)
- GitHub Actions [core smoke run 40197339993](https://github.com/Bulldogshot10/lora-data-maker/actions/runs/40197339993) completed with **success** on `73aa5069b07fc0d3e8a59c25d675440dcd0f7802`.
- `python -m compileall -q src tests`: passed.
- `pytest -q -m "not real_caption_runtime"`: **382 passed, 2 skipped, 1 deselected**, 1 warning. CLI, documentation, init/inspect, and all other core-smoke steps passed.
- The new regression test passed, which shows the semantic tuple parser now extracts the original source URLs from the tested AF_initDataCallback shape and rejects the tested Google thumbnail/cache hosts.
- **Live Termux:X11 status: needs verification.** This environment cannot exercise the user's visible Chromium session. CI proves the source parser on a representative fixture; it does not prove Google returns that payload in the user's current session.
- Google date filtering remains wired through CLI/config/manager/WebUI with `after:` and `before:` query operators; live behavior remains subject to on-device confirmation and reflects indexed result-page dates, not guaranteed photo capture dates.

### Next steps
1. Pull latest `main` on Termux.
2. Re-run the original query and look for `Google original URL payload tuples yielded N source candidate(s)` before the fallback log. Confirm the resulting downloads are from original source hosts rather than Google thumbnail/cache hosts.
3. Test a Google custom date range separately.
4. If tuple count is zero, capture a bounded safe diagnostic: `AF_initDataCallback` occurrence count, dimensional URL tuple count before host filtering, and a few source hostnames/dimension values. Do not dump full page content, cookies, tokens, or logged-in session state.
5. If original URLs are extracted but saved image dimensions are still thumbnail-sized, trace the original URL and downloaded image dimensions through downloader/manager independently.


## Session checkpoint / handoff (2026-10-25, authenticated Playwright website capture)

### User request and scope
On the existing main branch, add a Playwright start mode that opens a visible browser without scraping so the user can sign into a website, then scrape images from authenticated pages with sequential scrolling and optional CSS-selector scoping, reusing the existing dataset crop/resize pipeline. Do not create/switch branches or open a PR.

### Changes in this commit
- Added src/lora_data_maker/scrapers/site.py: dedicated persistent website profile and CDP port, manual login-browser startup helper, HTTP(S) page URL validation, rendered-image DOM extraction, lazy-load scrolling, optional CSS selector, per-page metadata, and in-memory CDP cookie capture for authenticated image downloads.
- Added the playwright-site backend to the built-in scraper registry and manager. Browser cookies are copied into the per-run requests cookie jar only while downloading; no cookie data is written to files, config, manifest, logs, or the repository.
- Added CLI command lora-data-maker playwright-browser --url URL and --selector CSS for scrape playwright-site URL.
- Added tests/test_playwright_site.py and updated README/CLI/Termux instructions.
- Crop/resize remains the existing processing stage and uses project configuration; the website scrape writes files into raw/.

### Feature status
Implemented in source, needs verification: dedicated persistent browser profile; no-scrape browser launch; generic HTTP(S) page scraper; lazy-scroll DOM capture; CSS-selector scoping; in-memory browser-cookie transfer for protected image downloads.

Needs verification: compileall, focused tests, full core test suite, live Termux:X11 login and protected-image download, and confirmation that the target site exposes usable image elements/original URLs. The code has not yet been run on the user's device in this session.

Not implemented: automatic interactive natural-language agent actions on arbitrary websites; current controls are URL + CSS selector, and image cropping uses the existing dataset processing configuration.

### Validation
- Read root AGENTS.md, agent.md, goal.md, relevant scraper/CLI/adapter source, tests, README, CLI docs, and Termux install docs before implementation.
- Automated tests were added but have not been executed locally in this connector-only environment. No CI or live-device pass is claimed.

### Next steps
1. Pull the updated main branch and run python -m compileall -q src tests.
2. Run python -m pytest -q tests/test_playwright_site.py tests/test_scraper.py.
3. Test lora-data-maker playwright-browser --url https://example.com/login in Termux:X11, complete login, then run lora-data-maker scrape playwright-site https://example.com/gallery --count 20.
4. Check that protected image downloads succeed with session cookies and that validate/process apply the configured crop/resize. If image elements are not selected, inspect the page and retry with an accurate CSS --selector.

### Review follow-up (2026-10-25, site-scroll/error handling)
- Manual source review found that the first version jumped directly to the bottom of long pages. Changed the website scraper to scroll one viewport at a time so intermediate lazy-loaded images have a chance to load before the next capture pass.
- Generic page-evaluation/CDP exceptions are now converted to ScraperError and run through the configured retry/error path rather than escaping as an unhandled CLI exception.
- Added unit coverage for viewport-sized scrolling and for transferring browser cookies into the in-memory HTTP cookie jar without writing project files.
- Validation remains pending: GitHub Actions status currently returns no checks for the feature commit, and local git clone failed because this environment cannot resolve github.com (Could not resolve host: github.com). No test pass is claimed. After pulling the new main, run python -m compileall -q src tests and python -m pytest -q tests/test_playwright_site.py tests/test_scraper.py.
