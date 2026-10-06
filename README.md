# lora-data-maker

Termux-first toolkit for building clean, train-ready image datasets for LoRA training.

## Scope

The project handles dataset preparation only:

scrape → validate → deduplicate → quality filter → crop/resize → tag → filter tags → inject subject/trigger → write captions → export/pack.

Model training and image generation are intentionally outside this project.

## Current state

Phase 3 is implemented. The core CLI can initialize projects, inspect their manifest, scrape Google Images, download from a plain URL list, retry failed requests, resume through a persistent download archive, and record successful downloads in the project manifest.

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

Inspect project state:

```bash
lora-data-maker info
```

Downloaded images go to `raw/`. Persistent download URLs are stored in `cache/downloads.txt`, and successful downloads are recorded in `manifest.json`.

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

Upcoming stages build on the Phase 3 scraper engine.
