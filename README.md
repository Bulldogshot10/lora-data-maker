# lora-data-maker

Termux-first toolkit for building clean, train-ready image datasets for LoRA training.

## Scope

The project handles dataset preparation only:

scrape → validate → deduplicate → quality filter → crop/resize → tag → filter tags → inject subject/trigger → write captions → export/pack.

Model training and image generation are intentionally outside this project.

## Current state

Phase 16 / v1.0.0 feature freeze is complete. The CLI supports the full dataset-preparation pipeline through export and packaging, incremental processing, and a complete project documentation set for Termux installation, CLI usage, configuration, dataset layout, troubleshooting, licensing, and responsible use.

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

Run the configured WD14 tagger on processed images:

```bash
lora-data-maker tag
```

Normalize/filter tags and inject the canonical dataset subject:

```bash
lora-data-maker clean-tags
```

Write deterministic caption files from the cleaned tags:

```bash
lora-data-maker caption
```

Run the full preparation pipeline:

```bash
lora-data-maker build
```

Resume from a later stage or skip individual stages:

```bash
lora-data-maker build --from-stage process
lora-data-maker build --skip scrape --skip tag
```

Export the captioned dataset as matching image + `.txt` pairs:

```bash
lora-data-maker export --repeats 10
```

Package the exported dataset together with `manifest.json`:

```bash
lora-data-maker pack
```

The first tag run downloads the pinned WD14 model into `~/.cache/lora-data-maker/taggers/wd14/`. The core package does not import ONNX Runtime until the tagger is actually used.

Downloaded images stay in `raw/`. Scrape is add-only, and manifest-aware URL filtering prevents already-known source URLs from being downloaded again. The reconcile stage only adds missing raw-file records and marks missing raw files in metadata without deleting manifest records or regressing completed states. The clean, process, tag-clean, and caption stages never delete or rewrite source images. Processed outputs are written under `work/processed/` as RGB JPEGs, while captions are written under `work/captions/`. Tags, tagger scores, canonical subject metadata, and caption metadata are recorded in `manifest.json`.

Reconcile raw files into the manifest without regressing completed image states:

```bash
lora-data-maker reconcile
```

Inspect project state:

```bash
lora-data-maker info
```

Persistent download URLs are stored in `cache/downloads.txt`, and successful downloads plus stage metadata are recorded in `manifest.json`.

The scraper does not attempt CAPTCHA, login, or provider-verification bypasses.

## Core commands

```text
lora-data-maker init
lora-data-maker scrape
lora-data-maker clean
lora-data-maker process
lora-data-maker tag
lora-data-maker clean-tags
lora-data-maker caption
lora-data-maker build
lora-data-maker export
lora-data-maker pack
lora-data-maker info
```

## License and responsible use

lora-data-maker is licensed under the Apache License 2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE) for licensing and third-party attribution information. Responsible-use guidance is in [docs/RESPONSIBLE_USE.md](docs/RESPONSIBLE_USE.md). The project is intended for educational and research-oriented dataset preparation. Users are responsible for respecting consent, identity, privacy, copyright, source-site rules, and applicable law.

## Documentation

~~~text
docs/INSTALL_TERMUX.md
docs/CLI.md
docs/CONFIG.md
docs/DATASET.md
docs/TROUBLESHOOTING.md
docs/ARCHITECTURE.md
docs/ROADMAP.md
~~~

Start with INSTALL_TERMUX.md for Android/Termux setup, then CLI.md and CONFIG.md for daily use.
