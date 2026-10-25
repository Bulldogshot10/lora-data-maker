# Troubleshooting

## config.toml not found

Run inside the project or pass the project path:

~~~bash
lora-data-maker info ~/datasets/julia
~~~

The CLI searches parent directories for the project config.

## manifest.json not found

Initialize the project:

~~~bash
lora-data-maker init ~/datasets/julia --name julia --subject julia
~~~

## Scrape returns zero new images

Check the query and backend:

~~~bash
lora-data-maker info
lora-data-maker scrape playwright-bing "julia" --count 20
~~~

For direct scraping, provide a text file with one URL per line.

Zero new downloads during an incremental run can simply mean every returned source URL is already known.

## Playwright Google source images and quality filtering

The `playwright-google` backend opens Google Images with the current `udm=2` URL mode and accepts the legacy `tbm=isch` mode if Google redirects an existing browser session to it. This matters because newer Google pages can finish loading and generate normal network traffic while a detector that only recognizes `tbm=isch` incorrectly waits for navigation to time out.

The backend first extracts original source URLs (`ou`, source links, and source URL attributes), including the unquoted `ou:` form used in JavaScript object literals. When parsing finds none, it invokes Playwright browser clicks, waits for the viewer to hydrate, and checks its live DOM for explicit source attributes or a large displayed image outside the result grid. If original URLs still cannot be recovered, it scans rendered Google image elements for thumbnail URLs and uses those as a marked last-resort fallback. Thumbnail fallbacks bypass the original-source quality gate because they may be smaller or softer than the original.

Google date filters work with `lora-data-maker scrape playwright-google "portrait" --year 2019` or `--date-from 2019-04-01 --date-to 2019-09-30`. The Google backend sends the dates as `after:`/`before:` query operators. These filter indexed result-page dates, not EXIF metadata or necessarily the original photo's capture date. Playwright Bing keeps its native date-range filter.

Before a Playwright Google image enters the dataset, the scraper checks that both dimensions are at least 384 pixels and that its Laplacian-variance blur score is at least 100 (measured after downsampling to 256 pixels). This is a heuristic quality gate: very soft-focus images can be rejected, and sharp-looking images with large simple edges can still pass. The Google backend checks up to 10 candidates per requested image to compensate for rejected or inaccessible sources. Other backends keep their existing selection behavior.

## Scraper source is blocked or returns no candidates

Chromium mode tries DuckDuckGo Images first and Bing Images second. The Playwright backends use persistent Termux Chromium profiles and locator-based waits for dynamic result grids, with DOM-based result-count fallbacks when provider card classes change. Google extraction caches the visible thumbnails before browser-assisted clicks so a preview navigation cannot erase fallback candidates. Bing accepts structured image metadata even when the iusc class changes and uses turl only when murl is unavailable or points to a Bing cache URL. Google consent/verification pages pause the active scrape and wait for manual completion in the visible Chromium window, then resume the same session; use Cancel to stop waiting. Browser-side HTTP errors are logged, and common analytics endpoints are blocked without blocking page scripts or verification resources. A provider may still reject automated traffic or return no candidates; lower request volume or switch provider when that happens.

## Validation rejects images

Inspect the manifest state and validation configuration:

~~~toml
[validation]
min_width = 1
min_height = 1
min_file_size = 1
~~~

Common causes are corrupt image data, dimensions below configured minimums, or files that are too small.

## Clean rejects or deduplicates too much

Review:

~~~toml
[quality]
min_width = 512
min_height = 512
blur_threshold = 0.0
max_aspect_ratio = 0.0

[dedup]
exact = true
perceptual = true
perceptual_distance = 6
~~~

Lower the minimum dimensions or reduce perceptual aggressiveness when needed.

## Processing fails

Check processing settings:

~~~toml
[processing]
target_width = 512
target_height = 512
resize_mode = "cover"
jpeg_quality = 95
face_crop = false
~~~

When face_crop is enabled, a compatible face detector must be provided by the processing backend.

## Tagger runtime import error

Install the optional tagger dependencies:

~~~bash
python -m pip install ".[tagger]"
~~~

Then check the basic CLI:

~~~bash
lora-data-maker tag --help
~~~

The non-tagger commands do not require ONNX Runtime.

## Tagger model download fails

Retry after checking network access:

~~~bash
lora-data-maker tag
~~~

The configured model revision is cached and its SHA-256 is verified.

## Tags contain unwanted entries

Edit tag_rules.remove and rerun:

~~~bash
lora-data-maker clean-tags
~~~

## Caption trigger is wrong

Check:

~~~toml
[project]
subject = "julia"
trigger = "julia-v1"

[caption]
trigger_from_subject = true
~~~

With trigger_from_subject enabled, the effective trigger is julia.
With it disabled, the manifest trigger is used.

## Export produces nothing

Export consumes captioned or exported records. Check:

~~~bash
lora-data-maker info
lora-data-maker caption
lora-data-maker export --repeats 10
~~~

Each source record needs both a processed image and a caption path.

## Pack cannot find the dataset

Export first:

~~~bash
lora-data-maker export --repeats 10
lora-data-maker pack
~~~

Or provide the dataset directory explicitly:

~~~bash
lora-data-maker pack --dataset-dir output/10_julia
~~~

## Incremental run does not change anything

Inspect state:

~~~bash
lora-data-maker info
lora-data-maker reconcile
~~~

Completed records are intentionally skipped. A newly added raw file will enter the pipeline from discovered.

## Missing raw file

Run reconcile:

~~~bash
lora-data-maker reconcile
~~~

The manifest record remains for audit and gets missing-file metadata.

## Build stopped at a stage

Run that stage directly:

~~~bash
lora-data-maker <stage>
~~~

Then resume:

~~~bash
lora-data-maker build --from-stage <stage>
~~~

Build stores stage results and the failed stage in manifest.source.
