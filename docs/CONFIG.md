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
runtime = "llama_cpp"
model_path = ""
mmproj_path = ""
chat_format = "mtmd"
api_key_env = ""
api_model = ""
api_base_url = ""
api_timeout = 120.0
max_tokens = 64
temperature = 0.1
top_p = 0.95
top_k = 40
n_ctx = 4096
n_threads = 4
n_gpu_layers = 0
seed = 42
~~~

backend = "deterministic" keeps the existing tag-based captioner. Set runtime to "llama_cpp" for local GGUF inference. Set runtime to "gemini" for the Google Gemini REST API. Set runtime to "openai_compatible" for a vision-capable OpenAI-compatible Chat Completions endpoint.

For API runtimes, api_key_env names the environment variable holding the API key. The secret itself is never stored in config.toml. api_model selects the model. api_base_url defaults to https://generativelanguage.googleapis.com/v1beta for Gemini and https://api.openai.com/v1 for OpenAI-compatible mode. The built-in Gemini example uses gemini-3.5-flash-lite. api_timeout controls the HTTP request timeout in seconds.

Every VLM runtime uses the same strict caption prompt: exactly four fields in order, pose, clothing, expression, background; maximum five words per field; pose includes a camera view term when visible; background includes lighting; hair color, eye color, face, age, and gender are excluded. The parser rejects malformed responses.

The final VLM caption always injects the resolved dataset trigger first, removes duplicate trigger mentions from generated fields, and produces trigger → pose → clothing → expression → background. WD14/JoyTag tags are not merged into VLM prose, and the deterministic caption.max_tokens budget does not apply to VLM output.

Install the local runtime with:

~~~bash
python -m pip install ".[caption-vlm]"
~~~

The API runtimes need no extra Python package beyond the project's existing requests dependency.
## export

~~~toml
[export]
repeats = 1
folder_name = ""
~~~

With an empty folder_name, the output directory is repeats_subject, for example 10_julia.
A non-empty folder_name overrides the generated directory name.
