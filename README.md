# lora-data-maker

Termux-first toolkit for building clean, train-ready image datasets for LoRA training.

## Scope

The project handles dataset preparation only:

scrape → validate → deduplicate → quality filter → crop/resize → tag → filter tags → inject subject/trigger → write captions → export/pack.

Model training and image generation are intentionally outside this project.

## Design goals

- Runs cleanly on Termux and normal Linux/Python environments.
- Keeps the core dependency set small.
- Keeps raw images immutable.
- Makes every processing stage resumable and inspectable.
- Treats dataset identity (subject) and trigger as first-class metadata.
- Makes taggers and image-processing backends replaceable.

## Planned core commands

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

This repository is currently in Phase 0: architecture and dependency contract only.
