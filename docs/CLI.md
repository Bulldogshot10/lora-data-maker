# CLI reference

The executable is:

~~~text
lora-data-maker
~~~

When a project path is supplied, or when the command is run from a child directory, the CLI searches upward for config.toml.

## Global options

~~~text
--version
-v, --verbose
--help
~~~

## init

Create a project. With `--force`, the config is replaced and `manifest.json` is recreated, resetting recorded dataset state.

~~~bash
lora-data-maker init [path] [--name NAME] [--subject SUBJECT] [--trigger TRIGGER] [--force]
~~~

subject is the canonical dataset identity. trigger defaults to subject.

## info

Inspect project identity, paths, image counts, and manifest state.

~~~bash
lora-data-maker info [path]
~~~

## scrape

Download images through a built-in backend.

~~~bash
lora-data-maker scrape {google,duckduckgo,chromium,playwright-bing,playwright-google,direct} [query] [--count COUNT] [--year YEAR | --date-from YYYY-MM-DD --date-to YYYY-MM-DD] [--path PATH]
~~~

Google uses query text with the legacy HTTP parser. DuckDuckGo uses query text and its image JSON endpoint. Chromium mode uses DuckDuckGo Images first, then local headless Chromium for Bing Images. The Playwright backends use the Termux-native Chromium CDP adapter directly for Bing or Google Images. Direct uses a text file containing source URLs. `--year` and `--date-from/--date-to` apply only to `playwright-bing`; omit both to search any date.

Scrape is add-only. Known manifest source URLs and archived URLs are skipped.

## reconcile

Reconcile files currently present in raw/ with the manifest.

~~~bash
lora-data-maker reconcile [--path PATH]
~~~

New files become discovered records. Existing records keep their current state. Missing raw files are marked in metadata and remain in the manifest.

## validate

Decode and validate raw images.

~~~bash
lora-data-maker validate [--path PATH]
~~~

Validation records dimensions, file size, MIME type, image format, EXIF orientation, and metadata.

Records already advanced beyond downloaded are skipped, preventing state regression.

## clean

Run exact deduplication, perceptual deduplication, and quality filtering.

~~~bash
lora-data-maker clean [--path PATH]
~~~

Duplicate and rejected records remain auditable in the manifest.

## process

Create standardized training images.

~~~bash
lora-data-maker process [--path PATH]
~~~

Output goes under work/processed/. Supported resize modes are cover, fit, and stretch.

## tag

Run the configured ONNX image tagger. Built-in presets are `sweetspot`, `potato`, and `joytag`.

~~~bash
lora-data-maker tag [--path PATH] [--preset PRESET]
~~~

Examples:

~~~bash
lora-data-maker tag --preset sweetspot
lora-data-maker tag --preset potato
lora-data-maker tag --preset joytag
~~~

The model is cached under the user cache and tagger metadata is stored in the manifest.

## clean-tags

Normalize and filter generated tags, then inject the canonical subject.

~~~bash
lora-data-maker clean-tags [--path PATH]
~~~

The subject is inserted exactly once at the front.

## caption

Write caption files from cleaned image records.

~~~bash
lora-data-maker caption [--path PATH]
~~~

The default caption.backend = "deterministic" builds captions from cleaned tags. With caption.backend = "vlm", the stage reads each processed image directly and sends it to the local llama-cpp-python VLM runtime. The VLM returns four short fields (pose, clothing, expression, background), which become the ordered caption body after the canonical trigger. WD14 and JoyTag are skipped completely. The four validated VLM fields are written directly after the canonical trigger; caption.max_tokens is not applied in VLM mode. VLM inference or formatting errors are recorded as failed images and never fall back to booru tags.

## build

Run the full preparation pipeline from scrape through pack.

~~~bash
lora-data-maker build [--path PATH] [--from-stage STAGE] [--to-stage STAGE] [--skip STAGE]
~~~

Examples:

~~~bash
lora-data-maker build
lora-data-maker build --from-stage process
lora-data-maker build --from-stage tag --to-stage caption
lora-data-maker build --skip scrape --skip tag
~~~

The skip option may be repeated. Build stops on the first failed stage and records its summary and error details in the manifest.

## export

Export captioned records as matching image and TXT pairs.

~~~bash
lora-data-maker export [--path PATH] [--repeats N] [--folder-name NAME]
~~~

Default directory:

~~~text
output/<repeats>_<subject>/
~~~

## pack

Create a ZIP containing the exported dataset and manifest.json.

~~~bash
lora-data-maker pack [--path PATH] [--archive FILE] [--dataset-dir DIR]
~~~

Without dataset-dir, pack uses the last exported directory recorded in the manifest.

## Exit codes

0 means the requested operation completed successfully or the requested state was already satisfied.
1 means the requested operation could not complete.
