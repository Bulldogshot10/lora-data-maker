# Configuration reference

The project configuration lives in config.toml. The complete generated example is also maintained in config.example.toml.

## project

~~~toml
[project]
name = "my-dataset"
subject = "my-dataset"
trigger = "my-dataset"
~~~

subject is the canonical dataset identity.
trigger is the explicit trigger token and defaults to subject during project initialization.

## paths

~~~toml
[paths]
raw = "raw"
work = "work"
output = "output"
cache = "cache"
~~~

Relative paths are resolved from the project root. Absolute paths are accepted.

## scraper

~~~toml
[scraper]
backend = "duckduckgo"
query = "my-dataset"
count = 300
timeout = 20
retries = 3
delay = 1.0
max_scrolls = 30
~~~

Built-in backends are google, duckduckgo, chromium, playwright-bing, playwright-google, and direct.

The existing `chromium` backend uses Selenium, DuckDuckGo first, and Bing Images as its browser fallback. The Playwright backends use Termux-native Chromium through the `playwright-termux` CDP adapter.

For authenticated Google Playwright scraping, set these Termux environment
variables before running `playwright-google`:

~~~bash
export DISPLAY=:1
export CHROME_USER_DATA_DIR="$HOME/.local/share/lora-data-maker/google-profile"
export CHROME_DEBUG_PORT=9222
~~~

`CHROME_USER_DATA_DIR` enables attachment to the existing logged-in Chromium
profile. `CHROME_DEBUG_PORT` defaults to 9222. Chromium must already be
running with that profile and local remote debugging enabled.

## validation

~~~toml
[validation]
min_width = 1
min_height = 1
max_width = 0
max_height = 0
min_file_size = 1
max_file_size = 0
~~~

A zero max value disables the corresponding upper bound.

## quality

~~~toml
[quality]
min_width = 512
min_height = 512
blur_threshold = 0.0
max_aspect_ratio = 0.0
~~~

A zero blur threshold disables blur rejection.
A zero max aspect ratio disables that upper bound.

## processing

~~~toml
[processing]
target_width = 512
target_height = 512
resize_mode = "cover"
jpeg_quality = 95
face_crop = false
face_padding = 1.75
~~~

When face_crop is true, the CLI uses the built-in YuNet face detector. The detector model is cached under the configured project cache directory in cache/face/ and verified against its pinned SHA-256 checksum.

cover fills the target and crops overflow.
fit keeps the whole image and pads the remaining area.
stretch resizes directly to the target dimensions.

## dedup

~~~toml
[dedup]
exact = true
perceptual = true
perceptual_distance = 6
~~~

exact enables SHA-256 duplicate detection.
perceptual enables pHash near-duplicate detection.
perceptual_distance controls the maximum Hamming distance for near duplicates.

## tagger

~~~toml
[tagger]
backend = "wd14"
preset = "sweetspot"
threshold = 0.35
character_threshold = 0.85
use_mcut = false
batch_size = 1
download_timeout = 60
~~~

Built-in presets pin the model repository, revision, filenames, and model SHA-256. Use sweetspot for the default WD ViT tagger or potato for the smallest mobile SwinV2 tagger.

For chromium, the optional Selenium dependency is required. The backend expects
chromium-browser and chromedriver on PATH by default. Selenium manages the temporary browser profile.

## tag_rules

~~~toml
[tag_rules]
remove = [
  "watermark",
  "signature",
  "logo",
  "text",
]
~~~

Rules are normalized before matching. Underscores become spaces, whitespace is collapsed, and tags are lowercased.

## caption

~~~toml
[caption]
separator = ", "
subject_position = "first"
trigger_from_subject = true
~~~

separator joins caption tags.
subject_position currently supports first.
When trigger_from_subject is true, the effective trigger comes from project.subject. Otherwise project.trigger is used.

## export

~~~toml
[export]
repeats = 1
folder_name = ""
~~~

With an empty folder_name, the output directory is repeats_subject, for example 10_julia.
A non-empty folder_name overrides the generated directory name.
