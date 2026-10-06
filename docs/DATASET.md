# Dataset layout

A project has four working areas:

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

## raw

raw/ stores downloaded or user-supplied source images.

Raw files are treated as immutable source data. Normal pipeline stages do not rewrite them.

## manifest.json

The manifest is the durable state store.

It records project identity, config snapshot, source metadata, image status, hashes, dimensions, tagger metadata, tags, caption paths, export paths, rejection reasons, and stage summaries.

Manifest writes are atomic.

## work/processed

Processed images are written as:

~~~text
work/processed/<image_id>.jpg
~~~

The base processor emits standardized RGB JPEGs.

## work/captions

Caption files are written as:

~~~text
work/captions/<image_id>.txt
~~~

Each file contains one deterministic UTF-8 caption.

## output

The export stage creates a directory such as:

~~~text
output/
└── 10_julia/
    ├── abc123.jpg
    ├── abc123.txt
    ├── def456.jpg
    └── def456.txt
~~~

Each image has a same-basename TXT caption.

## cache

Project cache contains scraper state such as cache/downloads.txt.

The WD14 model cache is separate:

~~~text
~/.cache/lora-data-maker/taggers/wd14/
~~~

## State flow

~~~text
discovered
   ↓
downloaded
   ↓
validated
   ↓
processed
   ↓
tagged
   ↓
captioned
   ↓
exported
~~~

Duplicate and rejected are terminal audit states for the current pipeline.

## Incremental flow

New files are added as discovered records. Existing records retain their current state. Later stages consume only records eligible for that stage.

## Training hand-off

Create the dataset:

~~~bash
lora-data-maker export --repeats 10
~~~

Then inspect output/10_subject/.

Create a portable bundle:

~~~bash
lora-data-maker pack
~~~

The ZIP contains the exported dataset and manifest.json. A full `build` performs export and pack automatically.
