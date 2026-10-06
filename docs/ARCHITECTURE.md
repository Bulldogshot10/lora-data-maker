# Architecture Contract

Status: Phase 0 locked

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

By default the trigger is derived from the subject. Caption generation prepends the subject once, after tag normalization and filtering.

Example final caption:

julia, blonde hair, blue eyes, portrait

Trigger injection must be idempotent. Re-running the caption stage must not produce repeated subject tokens.

## Tagger boundary

Taggers implement a common interface and are optional dependencies.

The first planned backend is a WD14-compatible ONNX tagger. The core package must remain usable without an installed tagger runtime or model.

Tagger output is treated as visual metadata. The tagger is not responsible for dataset identity.

## Processing boundary

Image processing backends implement small interfaces where platform-specific or optional dependencies are involved.

Face detection is optional. The base pipeline must work without a face detector.

## Scraper boundary

Scrapers implement a common interface and produce download records plus source metadata.

Google Images is its own backend. External downloaders such as gallery-dl may be integrated as optional backends for sources they support, but the Google backend must not depend on them.

## Manifest

Every project maintains a manifest containing at least:

- project/dataset name
- subject
- trigger
- source query/backend
- processing configuration
- tagger configuration
- per-image status
- rejection reason where applicable
- timestamps

The manifest is the durable state used for resume and audit.

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
- local LLM captioning
- GPU-specific inference stacks

## Dependency tiers

### Core

- Python standard library
- Pillow
- requests

### Optional

- OpenCV for image/face processing backends
- ONNX Runtime for taggers
- gallery-dl for optional external download backends
- pytest for development

Optional features must never make the basic CLI import fail.

## Compatibility target

Primary target: current Termux Python on aarch64 Android.

Secondary target: regular Linux environments.

Platform-specific packages must be validated on-device before being treated as required dependencies.
