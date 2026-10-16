# lora-data-maker

Termux-first toolkit for building clean, train-ready image datasets for LoRA training.

## Scope

The project handles dataset preparation only:

scrape → validate → deduplicate → quality filter → crop/resize → deterministic tags OR local VLM captions → write captions → export/pack.

Model training and image generation are outside the scope of this project.

## Current state

v2.1.0 includes two mobile-friendly WD14 tagger presets plus a JoyTag ONNX backend:

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

## VLM captions

The caption stage supports local `llama-cpp-python` plus API runtimes for Google Gemini and OpenAI-compatible multimodal endpoints. The existing deterministic tag-based captioner remains the default.

For local GGUF captioning:

~~~bash
python -m pip install ".[caption-vlm]"
~~~

Configure a local runtime with:

~~~toml
[caption]
backend = "vlm"

[caption.vlm]
runtime = "llama_cpp"
model_path = "/path/to/model.gguf"
mmproj_path = "/path/to/mmproj.gguf"
chat_format = "mtmd"
max_tokens = 64
temperature = 0.1
top_p = 0.95
top_k = 40
n_ctx = 4096
n_threads = 4
n_gpu_layers = 0
seed = 42
~~~

In the WebUI, choose the Gemini or OpenAI-compatible runtime, paste the API key into the password field, and use **Set / Save / Reuse**. The key is saved in a per-user credential file outside dataset projects and loaded into the server process environment. It is never written to `config.toml`. The file is plaintext protected by filesystem permissions, not encrypted; see [SECURITY.md](SECURITY.md) before using API credentials. After the key is set, the WebUI discovers available models and fills the model selector.

Gemini model discovery calls the provider's model-list endpoint and keeps models that advertise `generateContent` support. The WebUI therefore does not require a model-name textbox. For OpenAI-compatible runtimes, the WebUI reads the standard `/models` response from the configured base URL and exposes the returned model IDs in the selector. OpenAI-compatible endpoints use `/chat/completions` for image-to-text captioning.

For an OpenAI-compatible custom provider, only the API base URL may need to be changed in Settings. The model is always selected from the discovered dropdown. Providers that do not expose a compatible /models endpoint may need model configuration outside the current WebUI discovery flow.

The selected model is stored in the dataset config so the next caption job uses the same model. API keys entered in the WebUI are saved per provider in a per-user credential file outside dataset folders (`~/.config/lora-data-maker/api-keys.json`, or under `$XDG_CONFIG_HOME` when set). The file and its containing directory are restricted to the current OS user; the keys are plaintext at rest, not encrypted. Keys are never written into a dataset `config.toml`, are restored when the WebUI server starts, and stay available when switching dataset projects. Use **clear** to remove the selected provider key.

The WebUI caches provider model discovery for ten minutes. The model selector shows known Gemini choices immediately while refreshing the provider's model list in the background.

Every VLM runtime receives the same fixed caption constraint. The model must return exactly four fields in order: `pose`, `clothing`, `expression`, and `background`. Each field is limited to five words. Pose includes one camera-view term when visible, background includes lighting, and hair color, eye color, face, age, and gender are excluded. The parser rejects malformed output.

After parsing, the code always injects the resolved dataset trigger as the first caption token and removes duplicate trigger mentions from the four generated fields. The final order is trigger → pose → clothing → expression → background. WD14/JoyTag tags are not merged into VLM prose and `caption.max_tokens` does not limit VLM output. API or formatting failures mark that image failed instead of falling back to deterministic tags.


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
python -m pip install ".[quality]"
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
or download a glibc browser bundle. For authenticated Google searches, the
backend can attach to an already-running Chromium using `CHROME_USER_DATA_DIR`
and the local CDP port (9222 by default); see [Termux installation notes](docs/INSTALL_TERMUX.md).
A CDP endpoint grants powerful control over the browser and its signed-in accounts.
Use a dedicated profile, bind debugging to `127.0.0.1`, and never expose the port
through a LAN address, tunnel, or reverse proxy. Read [SECURITY.md](SECURITY.md).

For tagging support:

~~~bash
python -m pip install ".[tagger]"
~~~

For local VLM captioning:

~~~bash
python -m pip install ".[caption-vlm]"
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
  ├────────────→ tag → clean-tags ───→ deterministic caption → trigger → .txt
  │
  └────────────→ local VLM → strict fields → trigger → .txt
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

Examples:

~~~bash
lora-data-maker scrape duckduckgo "julia subject" --count 200
lora-data-maker scrape playwright-bing "julia portrait" --year 2019 --count 100
lora-data-maker scrape playwright-bing "julia portrait" --date-from 2019-04-01 --date-to 2019-09-30 --count 100
~~~

Bing date filters are also available in the WebUI as **any date**, **specific year**, or **custom range**. The date filter controls appear when `playwright-bing` is selected. The scraper is add-only and uses the manifest to avoid re-downloading known source URLs. A running WebUI scrape can be cancelled, and downloads completed before cancellation are preserved.

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
SECURITY.md
~~~

## Development

Install the development dependencies:

~~~bash
python -m pip install ".[dev]"
pytest -q
~~~

The core and WebUI smoke workflows are currently manual (`workflow_dispatch`) to avoid automatic GitHub Actions runs. Run the test commands locally before publishing a release; a successful prior workflow run is not evidence that the current branch passes.

## License

Licensed under the Apache License 2.0. See `LICENSE` and `NOTICE`.

## WebUI

The project includes an optional localhost WebUI. It is designed for local dataset work from a phone or desktop browser. The scraper is intentionally outside the WebUI pipeline.

Install the WebUI and tagger extras:

    pip install -e ".[web,tagger,quality,caption-vlm]"

Start the WebUI from a dataset workspace:

    mkdir -p ~/datasets
    lora-data-maker web --path ~/datasets

Then open http://127.0.0.1:8787 in the browser. The WebUI can create a new dataset project from the workspace, open existing projects, and switch between them without using the CLI init command.

The WebUI supports image upload, project identity/trigger settings, resize settings, WD14 or JoyTag selection for deterministic captions, local VLM model/mmproj paths and chat format, the final caption token budget, validation, staging, caption generation, Kohya export, ZIP packaging, gallery preview, manual image removal for cherry-picking, scraper cancellation, and ZIP download. VLM mode skips WD14/JoyTag and generates captions as trigger → pose → clothing → expression → background.

The one-click Prepare Dataset flow runs:

    validate -> clean / dedup -> stage -> caption -> export -> ZIP (VLM mode)
    validate -> clean / dedup -> stage -> tag -> clean tags -> caption -> export -> ZIP (deterministic mode)

The clean stage includes the default PP-OCRv3 text gate. Install the quality extra before using the full pipeline: python -m pip install ".[quality]". The bundled detector is the upstream English PP-OCRv3 model, so non-English text can be less reliable.

The image quality/dedup cleaning stage remains an explicit optional button because a manually curated/cherry-picked image set may intentionally skip that stage.

