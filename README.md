# lora-data-maker

Termux-first toolkit for building clean, train-ready image datasets for LoRA training.

## Scope

The project handles dataset preparation only:

scrape → validate → deduplicate → quality filter → crop/resize → tag → filter tags → inject subject/trigger → write captions → export/pack.

Model training and image generation are intentionally outside this project.

## Current state

v1.3.0 adds two mobile-friendly WD14 tagger presets: sweetspot uses SmilingWolf's WD ViT Tagger v3, while potato uses the QUINT8 mobile SwinV2 variant. for current image-search environments where Google no longer exposes static image-result markup. The CLI supports the full dataset-preparation pipeline through export and packaging, incremental processing, and a complete project documentation set for Termux installation, CLI usage, configuration, dataset layout, troubleshooting, licensing, and responsible use.

## Design goals

- Runs cleanly on Termux and normal Linux/Python environments.
- Keeps the core dependency set small.
- Keeps raw images immutable.
- Makes every processing stage resumable and inspectable.
- Treats dataset identity (subject) and trigger as first-class metadata.
- Makes taggers and image-processing backends replaceable.

## Tagger presets

The default tagger preset is sweetspot. Select it explicitly with:

```bash
lora-data-maker tag --preset sweetspot
```

For the smallest built-in tagger:

```bash
lora-data-maker tag --preset potato
```

The preset can also be stored in config.toml under [tagger]:

```toml
[tagger]
preset = "sweetspot"
threshold = 0.35
character_threshold = 0.85
```

sweetspot downloads the 379 MB WD ViT Tagger v3 ONNX model. potato downloads the 176 MB QUINT8 mobile SwinV2 model and its tags.json mapping. The mobile model accepts RGB uint8 images and performs its own preprocessing inside ONNX.

## Quick start

Initialize a project:

```bash