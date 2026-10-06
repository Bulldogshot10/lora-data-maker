# lora-data-maker

Termux-first toolkit for building clean, train-ready image datasets for LoRA training.

## Scope

The project handles dataset preparation only:

scrape → validate → deduplicate → quality filter → crop/resize → tag → filter tags → inject subject/trigger → write captions → export/pack.

Model training and image generation are intentionally outside this project.

## Current state

Phase 6 is implemented. The CLI can initialize projects, inspect manifests, scrape/download images, validate image files, deduplicate exact and near-duplicate images, apply quality filters, and process validated images into standardized work outputs.

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
lora-data-maker init ./my-dataset --name my-dataset --subject my-dataset
cd my-dataset
```

Scrape Google Images:

```bash
lora-data-maker scrape google "my subject" --count 300
```

Or download a prepared URL list:

```bash
lora-data-maker scrape direct urls.txt --count 300
```

Validate the downloaded images:

```bash
lora-data-maker validate
```

Deduplicate and quality-filter the validated set:

```bash
lora-data-maker clean
```

Process the cleaned images into train-ready work files:

```bash
lora-data-maker process
```

Downloaded images stay in `raw/`. The clean and process stages never delete or rewrite them. Processed outputs are written under `work/processed/` as RGB JPEGs, with orientation, resize, crop, and output metadata recorded in `manifest.json`.

Inspect project state:

```bash
lora-data-maker info
```

Persistent download URLs are stored in `cache/downloads.txt`, and successful downloads plus stage metadata are recorded in `manifest.json`.

The scraper does not attempt CAPTCHA, login, or provider-verification bypasses.

## Planned core commands

```text
lora-data-maker init
lora-data-maker scrape
lora-data-maker clean
lora-data-maker process
lora-data-maker tag
lora-data-maker caption
lora-data-maker build
lora-data-maker export
lora-data-maker pack
lora-data-maker info
```

Upcoming stages build on the Phase 6 processing stage.
