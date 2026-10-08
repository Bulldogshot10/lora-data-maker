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
  ↓
tag
  ↓
tag filtering
  ↓
subject / trigger injection
  ↓
caption write
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

julia, blonde hair, blue eyes, portrait

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

Caption generation consumes records after tag cleaning. The tagger and VLM captioner both read the same processed image independently; tagger output is never passed into the VLM prompt.

With `caption.backend = "deterministic"`, caption text is built from normalized, deduplicated tags. With `caption.backend = "vlm"`, the local llama-cpp-python runtime generates a raw caption from the processed image and configured prompt, then the caption stage normalizes the text and injects the resolved trigger. VLM inference failures fall back to the deterministic tag caption for that image.

Captions are written as UTF-8 `.txt` files under `work/captions/`, one file per image record. The resolved trigger is derived from `subject` when `caption.trigger_from_subject = true`; otherwise the manifest trigger is used. Trigger injection is idempotent. Caption backend/model/fallback metadata is retained in the manifest for audit and resume.

## Export boundary



The export stage consumes `captioned` records and writes matched image + caption files into `output/<repeats>_<subject>/` by default. Each pair shares the same image-record basename, so a file such as `abc123.jpg` is paired with `abc123.txt`. The directory naming convention follows the legacy Kohya-style repeat/identifier layout; current sd-scripts also supports image directories with same-basename `.txt` captions and explicit repeat counts. citeturn825645search0turn825645search2 Export copies from `work/` and never rewrites `raw/` or the processed/caption source files.



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

## Non-goals

The following are out of scope for v1:

- PyTorch
- Diffusers
- Transformers
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
