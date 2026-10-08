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
person_check = false
person_threshold = 0.35
max_persons = 1
min_subject_area = 0.12
min_subject_visibility = 0.35
text_check = true
text_confidence = 0.5
text_min_area_ratio = 0.0004
~~~

A zero blur threshold disables blur rejection.
A zero max aspect ratio disables that upper bound.
When text_check is true, the clean stage runs the pinned PP-OCRv3 text detector and rejects images with detected text regions. text_confidence controls the minimum detector confidence and text_min_area_ratio ignores extremely small detected regions.
When person_check is true, the quality gate runs YOLOX-Nano through ONNX Runtime and rejects images with no detected person, more than max_persons people, or a primary subject that is too small/poorly visible. The detector is cached under cache/person/ and verified against its pinned SHA-256 checksum.

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

For JoyTag:

~~~toml
[tagger]
backend = "joytag"
preset = "joytag"
threshold = 0.4
batch_size = 1
download_timeout = 60
~~~

Built-in presets pin the model repository, revision, filenames, and model SHA-256. Use sweetspot for the default WD ViT tagger, potato for the smallest mobile SwinV2 tagger, or joytag for the JoyTag ViT-B/16 ONNX tagger. JoyTag uses a 0.4 default threshold when no threshold is configured.

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
backend = "deterministic"
separator = ", "
subject_position = "first"
trigger_from_subject = true
max_tokens = 77

[caption.vlm]
model_path = ""
mmproj_path = ""
chat_format = "mtmd"
max_tokens = 128
temperature = 0.1
top_p = 0.95
top_k = 40
n_ctx = 4096
n_threads = 4
n_gpu_layers = 0
seed = 42
~~~

backend = "deterministic" keeps the existing tag-based captioner. Set backend = "vlm" to caption each processed image with the local llama-cpp-python runtime.
model_path points to the local VLM GGUF and mmproj_path points to its multimodal projector GGUF. Both must be set together.
chat_format supports mtmd, qwen2.5-vl, and gemma4.
The VLM receives only the processed image and the configured prompt. Its prompt requests exactly four short fields: pose, clothing, expression, and background with lighting. Hair color, eye color, face, age, and gender are intentionally excluded from the VLM output.
The caption parser strips the field labels and merges the four phrases with cleaned tagger output into one final caption file.
The final caption has one shared caption.max_tokens budget of 1–77 tokens, default 77. Truncation preserves the configured separator, and the caption stage rejects a budget too small to preserve the resolved trigger.
VLM inference or formatting failures fall back to the deterministic tag caption for that image and are recorded in manifest metadata.

Install the optional runtime with:

~~~bash
python -m pip install ".[caption-vlm]"
~~~

The WebUI exposes the caption backend, VLM chat format, model GGUF path, mmproj GGUF path, and final 1–77 token budget under Settings. Paths entered there are stored in the project config; relative paths are resolved from the dataset root when the caption stage starts.

## export

~~~toml
[export]
repeats = 1
folder_name = ""
~~~

With an empty folder_name, the output directory is repeats_subject, for example 10_julia.
A non-empty folder_name overrides the generated directory name.
