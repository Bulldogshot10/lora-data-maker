# lora-data-maker

Termux-first toolkit for building clean, train-ready image datasets for LoRA training.

## Scope

The project handles dataset preparation only:

scrape → validate → deduplicate → quality filter → crop/resize → tag → filter tags → inject subject/trigger → write captions → export/pack.

Model training and image generation are intentionally outside this project.

## Current state

v1.3.0 adds a configurable sequential scraper fallback chain. The CLI supports the full dataset-preparation pipeline through export and packaging, incremental processing, and a complete project documentation set for Termux installation, CLI usage, configuration, dataset layout, troubleshooting, licensing, and responsible use.

Fallbacks are sequential, never parallel. The chain stops as soon as the requested download count is reached. URLs already known, archived, or seen earlier in the same run are excluded across all engines, and exact duplicate image content is discarded.

## Design goals

- Runs cleanly on Termux and normal Linux/Python environments.
- Keeps the core dependency set small.
- Keeps raw images immutable.
- Makes every processing stage resumable and inspectable.
- Treats dataset identity (subject) and trigger as first-class metadata.
- Makes taggers and image-processing backends replaceable.

## Quick start

Initialize a project:

```bash