# Architecture Contract

Status: current v1.3 architecture

## Purpose

lora-data-maker prepares image datasets for downstream LoRA training systems. It does not train models and does not perform Stable Diffusion inference.

## Pipeline

scrape
  ↓
validate
  ↓
deduplicate
  ↓
quality filter
  ↓
process / crop / resize
  ├──→ tag → tag filtering → deterministic caption → trigger → .txt
  │
  └──→ local/API VLM → SD 1.5 comma-separated caption → trigger → .txt
  ↓
export
  ↓
pack

Stages must be independently runnable and resumable. build is an orchestration command over these stages.

## Data lifecycle

Raw source data is immutable.

raw/
work/
output/

A stage writes a new result or updates explicit stage metadata. The pipeline must never silently destroy the original downloaded image.

## Dataset identity

subject is the canonical identity for a dataset.

Example:

subject = julia
trigger = julia

By default the trigger is derived from the subject. Caption generation prepends the resolved trigger once, after tag normalization and filtering.

Example final caption:

julia, upper body portrait, standing, white shirt, slight smile, indoor studio

Trigger injection must be idempotent. Re-running the caption stage must not produce repeated trigger tokens.

## Manifest and state

Each project has a root-level manifest.json.

The manifest is versioned, JSON-serializable, atomically replaced on save, and contains:

- project name
- subject
- trigger
- source metadata
- configuration snapshot (captured when the project is initialized)
- per-image records
- per-image status
- rejection reason where applicable
- hashes and file metadata when available
- timestamps

Current image states:

discovered → downloaded → validated → processed → tagged → captioned → exported

Images can also enter duplicate or rejected states with an auditable reason. The clean stage records exact SHA-256 hashes, perceptual hashes, duplicate provenance, blur scores, aspect ratios, deterministic quality scores, person counts, subject visibility metrics, and quality rejection reasons.

The manifest is the durable state used for resume and audit. Incremental stages preserve forward progress: raw reconciliation adds newly discovered files, scraper source URLs already present in the manifest are skipped, and state-gated stages only consume records eligible for their next transition. Existing advanced states are never regressed by a normal validation scan. The build orchestrator runs the stages in pipeline order through export and pack, supports stage ranges and explicit skips, stops on the first failed stage, and persists per-stage before/after status counts plus build timestamps and outcome metadata under `manifest.source`.

## Incremental processing boundary

Incremental processing is add-only. New raw images are reconciled into the manifest as `discovered` records, while existing records retain their current state. Validation only processes `discovered` and `downloaded` records, preventing a completed record from being moved backward to `validated`. Scrape receives the manifest's known source URLs and skips them before download. Missing raw files are recorded as reconciliation metadata; manifest records are retained for audit and are never silently deleted.

## Build boundary

The build orchestrator is a thin coordinator over the standalone stage commands. The default range is `scrape` through `pack`. `--from-stage` and `--to-stage` select a contiguous range, while repeated `--skip` values remove individual stages from that range. Stage implementations remain responsible for state-aware resume behavior, so already completed records are not reprocessed by earlier state-gated stages. A build stops on the first non-zero stage result and persists the completed stage summaries.

## Tagger boundary

Taggers implement a common interface and are optional dependencies.

The built-in backends are WD14-compatible ONNX taggers exposed as pinned presets. Sweetspot uses the standard 448x448 WD ViT model contract. Potato uses the mobile SwinV2 ONNX model with embedded preprocessing that accepts RGB uint8 image tensors. Model files are cached by repository and revision, and the pinned model SHA-256 is verified before use.

Tagger output is treated as visual metadata. General tags, character tags, confidence scores, and rating scores are persisted in the image manifest. The tag-cleaning stage normalizes tags, removes configured blacklist entries, deduplicates while preserving order, and injects the canonical `subject` exactly once at the front. The tagger is not responsible for dataset identity or caption-file trigger handling.

## Caption boundary

Caption generation branches by backend. With `caption.backend = "deterministic"`, the final caption is built from normalized, deduplicated tags on tagged records. With `caption.backend = "vlm"`, local and API VLM runtimes use the same SD 1.5-focused prompt and produce one concise, comma-separated caption line covering framing, pose/action, clothing/accessories, expression/gaze, subject count when clear, and background/lighting. Stable identity traits are left to the dataset trigger for character/person LoRAs. WD14 and JoyTag are completely bypassed in VLM mode. The caption builder accepts the new flat format and legacy five-field responses, normalizes items, removes duplicate trigger mentions, and prepends the resolved trigger exactly once. The deterministic `caption.max_tokens` budget is not applied to VLM captions. VLM inference failures or empty captions mark the image failed instead of falling back to deterministic tags.

Captions are written as UTF-8 `.txt` files under `work/captions/`, one file per image record. The resolved trigger is derived from `subject` when `caption.trigger_from_subject = true`; otherwise the manifest trigger is used. Trigger injection is idempotent. Caption backend/model/fallback metadata is retained in the manifest for audit and resume.

## Export boundary



The export stage consumes `captioned` records and writes matched image + caption files into `output/<repeats>_<subject>/` by default. Each pair shares the same image-record basename, so a file such as `abc123.jpg` is paired with `abc123.txt`. The directory naming convention follows the legacy Kohya-style repeat/identifier layout; current sd-scripts also supports image directories with same-basename `.txt` captions and explicit repeat counts. See the [sd-scripts project](https://github.com/kohya-ss/sd-scripts) for downstream training details. Export copies from `work/` and never rewrites `raw/` or the processed/caption source files.



Export is idempotent. Re-running the stage overwrites the deterministic destination pair and leaves already-exported records in the `exported` state. Export failures are recorded in per-image metadata without marking the source record as exported.



## Packaging boundary



The pack stage creates a ZIP containing the exported dataset tree plus the project-level `manifest.json`. It deliberately excludes `raw/`, `work/`, caches, and other project directories unless they are inside the explicitly selected exported dataset. The archive path defaults to `output/<project-name>.zip` and can be overridden on the CLI. Pack metadata is recorded in `manifest.source`.



## Processing boundary

Image processing is implemented as a Pillow-only core stage that reads validated images from `raw/` and writes standardized RGB JPEGs under `work/processed/`. EXIF orientation is applied during processing without modifying the source file. Resize modes are `cover`, `fit`, and `stretch`.

Face detection is optional. The processor consumes a small `FaceDetector` protocol and can perform padded square face crops when a detector is supplied. The base pipeline works without any face detector dependency.

The quality gate runs during the clean stage. When person_check is enabled it uses a pinned YOLOX-Nano ONNX model through ONNX Runtime to count people and estimate subject visibility. The gate can also run a pinned PP-OCRv3 ONNX text detector through OpenCV to reject images containing visible text. These measurements are combined with Pillow-based sharpness, contrast, and exposure heuristics. Raw images remain untouched and every rejection reason is persisted in the manifest.

## Scraper boundary

Scrapers implement a common interface and produce download records plus source metadata.

Google Images is its own backend. External downloaders such as gallery-dl may be integrated as optional backends for sources they support, but the Google backend must not depend on them.

## Local WebUI and credential boundary

The optional WebUI is intended for a single local user and defaults to loopback. It has no user authentication. Middleware validates the Host header and loopback peer and rejects browser requests that declare a cross-origin Origin or Fetch Metadata context. These checks reduce common browser-based cross-origin attacks; they do not turn the WebUI into a network-facing multi-user service. Keep it bound to localhost and do not expose it through a tunnel, reverse proxy, or port forwarding.

API provider keys are persisted separately from dataset configuration in a per-user JSON file. They are plaintext protected by owner-only file permissions, not encrypted, and are loaded into the running WebUI process environment. Authenticated Google scraping may attach to an existing Chromium session via local CDP. The CDP endpoint is a privileged browser-control interface, so the profile should be dedicated to scraping and its debug port must stay on loopback. See the repository's [Security Policy](../SECURITY.md) for operational guidance and known limits of outbound URL checks.

## Non-goals

The following are out of scope for v1:

- PyTorch
- Diffusers
- PEFT
- Accelerate
- Gradio UI
- LoRA training
- Stable Diffusion inference
- networked captioning services
- GPU-specific inference stacks

## Dependency tiers

### Core

- Python standard library
- Pillow
- requests

### Optional

- OpenCV for image/face/text processing backends
- ONNX Runtime and NumPy for taggers and the optional person-quality gate
- gallery-dl for optional external download backends
- pytest for development
- llama-cpp-python for optional local VLM captioning

Optional features must never make the basic CLI import fail. The `tagger` and `quality` extras provide NumPy and ONNX Runtime without adding them to the core dependency set.

## Compatibility target

Primary target: current Termux Python on aarch64 Android.

Secondary target: regular Linux environments.

Platform-specific packages must be validated on-device before being treated as required dependencies.
