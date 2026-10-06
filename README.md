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

**sweetspot** downloads the pinned WD ViT Tagger v3 ONNX model (about 379 MB) plus its tag mapping.

**potato** downloads the pinned QUINT8 mobile SwinV2 ONNX model (about 176 MB) plus its `tags.json` mapping. The mobile model accepts RGB uint8 images and performs its own preprocessing inside ONNX.

Tagger models are cached under:

~~~text
~/.cache/lora-data-maker/taggers/wd14/
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

For Chromium scraping support:

~~~bash
pkg install chromium
python -m pip install ".[chromium]"
~~~

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
direct
~~~

Chromium renders Google Images and can fall back to Bing Images when Google presents a consent or verification page.

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

The project CI smoke-tests Python 3.12 and 3.13, including package installation, compilation, tests, CLI entry points, documentation files, and project initialization.

## License

Licensed under the Apache License 2.0. See `LICENSE` and `NOTICE`.
