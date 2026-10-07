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
lora-data-maker scrape chromium "julia" --count 20
~~~

For direct scraping, provide a text file with one URL per line.

Zero new downloads during an incremental run can simply mean every returned source URL is already known.

## Scraper source is blocked or returns no candidates

Chromium mode tries DuckDuckGo Images first and Bing Images second. A provider can still return a temporary block, challenge, or empty result set. The scraper treats a failed provider as a source-level fallback condition when another provider is available. For repeated failures, retry later, reduce request volume, or use a prepared direct URL list.

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
