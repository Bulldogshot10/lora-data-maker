# lora-data-maker

Termux-first toolkit for building clean, train-ready image datasets for LoRA training.

## Scope

The project handles dataset preparation only:

scrape → validate → deduplicate → quality filter → crop/resize → tag → filter tags → inject subject/trigger → write captions → export/pack.

Model training and image generation are outside the scope of this project.

## Current state

v1.3.0 adds two mobile-friendly WD14 tagger presets:

- **sweetspot**: SmilingWolf WD ViT Tagger v3, a compact general-purpose WD14-compatible ONNX tagger.
- **potato**: QUINT8 mobile SwinV2 tagger for the smallest built-in footprint.

The default preset is **sweetspot**. The CLI supports the complete dataset-preparation pipeline through export and packaging, incremental processing, resumable stages, manifest tracking, and Termux/Linux operation.

## Design goals

- Runs cleanly on Termux and normal Linux/Python environments.
- Keeps the core dependency set small.
- Keeps raw images immutable.
- Makes every processing stage resumable and inspectable.
- Treats dataset identity (subject) and trigger as first-class metadata.
- Makes taggers and image-processing backends replaceable.

## Tagger presets

The built-in taggers are:

- **sweetspot**: SmilingWolf WD ViT Tagger v3, a compact general-purpose WD14-compatible ONNX tagger.
- **potato**: QUINT8 mobile SwinV2 tagger for the smallest built-in footprint.
- **joytag**: JoyTag ViT-B/16 ONNX tagger for tag-style image descriptions across illustrated and photographic images.

Run the default tagger:

~~~bash
lora-data-maker tag
~~~

Select sweetspot explicitly:

~~~bash
lora-data-maker tag --preset sweetspot
~~~

Select the smallest built-in tagger:

~~~bash
lora-data-maker tag --preset potato
~~~

Select JoyTag:

~~~bash
lora-data-maker tag --preset joytag
~~~

The preset can also be stored in a project's `config.toml`:

~~~toml
[tagger]
backend = "wd14"
preset = "sweetspot"
threshold = 0.35
character_threshold = 0.85
use_mcut = false
batch_size = 1
download_timeout = 60
~~~

For JoyTag:

~~~toml
[tagger]
backend = "joytag"
preset = "joytag"
threshold = 0.4
batch_size = 1
download_timeout = 60
~~~

**sweetspot** downloads the pinned WD ViT Tagger v3 ONNX model (about 379 MB) plus its tag mapping.

**potato** downloads the pinned QUINT8 mobile SwinV2 ONNX model (about 176 MB) plus its `tags.json` mapping. The mobile model accepts RGB uint8 images and performs its own preprocessing inside ONNX.

**joytag** downloads the pinned JoyTag `model.onnx` (about 366 MB) plus `top_tags.txt`. JoyTag uses 448×448 input, 5813 labels, and a default threshold of 0.4. The official model is Apache-2.0 licensed and is trained for a mix of illustrated and photographic images.

Tagger models are cached under:

~~~text
~/.cache/lora-data-maker/taggers/wd14/
~/.cache/lora-data-maker/taggers/joytag/
~~~

## Quick start

Install from the repository:

~~~bash
pkg update
pkg install python git

git clone https://github.com/Bulldogshot10/lora-data-maker.git
cd lora-data-maker
python -m pip install .
~~~

For Chromium/Selenium scraping support:

~~~bash
pkg install chromium
python -m pip install ".[chromium]"
~~~

For Playwright image-search support on Termux:

~~~bash
pkg install x11-repo chromium
python -m pip install ".[playwright]"
~~~

The Playwright backend uses the native Termux Chromium package through the
Termux-specific Playwright/CDP adapter. It does not run `playwright install`
or download a glibc browser bundle.

For tagging support:

~~~bash
python -m pip install ".[tagger]"
~~~

For face cropping on regular Linux/Python environments:

~~~bash
python -m pip install ".[face]"
~~~

Initialize a dataset project:

~~~bash
lora-data-maker init ~/datasets/[name] --name [name] --subject [name]
cd ~/datasets/[name]
~~~

Run the full preparation pipeline:

~~~bash
lora-data-maker build
~~~

Or run stages manually:

~~~bash
lora-data-maker reconcile
lora-data-maker validate
lora-data-maker clean
lora-data-maker process
lora-data-maker tag --preset sweetspot
lora-data-maker clean-tags
lora-data-maker caption
lora-data-maker export
lora-data-maker pack
~~~

## Face cropping

Set `processing.face_crop = true` to enable the built-in YuNet face detector during processing. The small YuNet 2023mar ONNX model is downloaded once into the project's configured `cache/face/` directory and checksum-verified before use.

On Termux, install native OpenCV with:

~~~bash
pkg install x11-repo opencv-python
~~~

On regular Linux/Python environments, install the optional face extra with `python -m pip install ".[face]"`.

## Pipeline

The normal preparation flow is:

~~~text
scrape
  ↓
validate
  ↓
clean
  ↓
process
  ↓
tag
  ↓
clean-tags
  ↓
caption
  ↓
export
  ↓
pack
~~~

The pipeline is manifest-aware and can resume from later stages. For example:

~~~bash
lora-data-maker build --from-stage process
lora-data-maker build --from-stage tag --to-stage caption
lora-data-maker build --skip scrape --skip tag
~~~

## Scraping

Built-in scraper backends are:

~~~text
google
duckduckgo
chromium
playwright-bing
playwright-google
direct
~~~

`chromium` is the existing Selenium backend. The Playwright backends use
Termux-native Chromium and extract original image URLs from the rendered
Google or Bing Images pages. The Playwright navigation layer uses direct CDP navigation
because the Termux adapter's stock navigation waiter can race lifecycle events.

Example:

~~~bash
lora-data-maker scrape duckduckgo "julia subject" --count 200
~~~

The scraper is add-only and uses the manifest to avoid re-downloading known source URLs.

## Dataset layout

A project created with `init` contains:

~~~text
project/
├── config.toml
├── manifest.json
├── raw/
├── work/
│   ├── processed/
│   └── captions/
├── output/
└── cache/
~~~

Raw files stay immutable. Processing stages write derived files under `work/`, while `output/` contains the final exported dataset.

## Documentation

More detailed docs live in `docs/`:

~~~text
INSTALL_TERMUX.md
CLI.md
CONFIG.md
DATASET.md
TROUBLESHOOTING.md
ARCHITECTURE.md
ROADMAP.md
RESPONSIBLE_USE.md
~~~

## Development

Install the development dependencies:

~~~bash
python -m pip install ".[dev]"
pytest -q
~~~

The project CI smoke-tests Python 3.12, 3.13, and 3.14, including package installation, compilation, tests, CLI entry points, documentation files, and project initialization.

## License

Licensed under the Apache License 2.0. See `LICENSE` and `NOTICE`.

## WebUI

The project includes an optional localhost WebUI. It is designed for local dataset work from a phone or desktop browser. The scraper is intentionally outside the WebUI pipeline.

Install the WebUI and tagger extras:

    pip install -e ".[web,tagger]"

From an initialized dataset project:

    lora-data-maker web --path .

Then open http://127.0.0.1:8787 in the browser.

The WebUI supports image upload, project identity/trigger settings, resize settings, WD14 or JoyTag selection, validation, staging, tagging, tag cleaning, caption generation, Kohya export, ZIP packaging, gallery preview, and ZIP download.

The one-click Prepare Dataset flow runs:

    validate -> stage -> tag -> clean tags -> caption -> export -> ZIP

The image quality/dedup cleaning stage remains an explicit optional button because a manually curated/cherry-picked image set may intentionally skip that stage.

