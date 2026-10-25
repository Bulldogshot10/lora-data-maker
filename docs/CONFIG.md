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
year = 0
date_from = ""
date_to = ""
safe_search = false
~~~

`safe_search = false` disables SafeSearch for the Google and Playwright Google/Bing backends, which allows explicit image results to appear. Set it to `true` to filter them. The WebUI exposes the same option as the **enable SafeSearch filter** checkbox; the default is off.

Built-in backends are google, duckduckgo, chromium, playwright-bing, playwright-google, and direct.

The existing `chromium` backend uses Selenium, DuckDuckGo first, and Bing Images as its browser fallback. The Playwright backends use Termux-native Chromium through the `playwright-termux` CDP adapter. The optional `year` or `date_from` + `date_to` fields support both `playwright-bing` and `playwright-google`; use `year = 0` and blank date strings for any date. Google appends `after:YYYY-MM-DD before:YYYY-MM-DD` to its query. Google's operators filter indexed/search-result page dates, not the image's original capture date.

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
enabled = true
target_width = 512
target_height = 512
resize_mode = "cover"
jpeg_quality = 95
face_crop = false
face_padding = 1.75
~~~

Set `enabled = false` to bypass resize/crop processing. The pipeline skips the process stage; VLM captioning, tagging, and export use the original files under `raw/`, with no resized copies or synthetic borders. Validation and clean/quality filtering still run. Keep `enabled = true` for the standard processed-image path.

When face_crop is true, the CLI uses the built-in YuNet face detector. The detector model is cached under the configured project cache directory in cache/face/ and verified against its pinned SHA-256 checksum.

cover fills the target, preserves aspect ratio, and crops overflow.
fit keeps the whole image and preserves aspect ratio without padding or synthetic border pixels; the output fits inside the target dimensions and may be smaller on one axis.
stretch resizes directly to the exact target dimensions and can distort aspect ratio.

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

For API runtimes, api_key_env names the environment variable holding the API key. The secret itself is never stored in config.toml. The WebUI stores provider keys separately from dataset configuration in `~/.config/lora-data-maker/api-keys.json` (or `$XDG_CONFIG_HOME/lora-data-maker/api-keys.json` when `XDG_CONFIG_HOME` is set). Set `LORA_DATA_MAKER_API_KEY_FILE` to override the path. The credential file is plaintext protected by owner-only filesystem permissions (file `0600`; the app-created configuration directory is `0700`); it is not encrypted. An existing caller-supplied parent directory is not forcibly chmodded. Keys load automatically on WebUI startup and are shared across dataset projects, and are present in the running WebUI process environment. The WebUI's **clear** action removes that provider's saved key. See the repository's [Security Policy](../SECURITY.md) for the local-only threat model and authenticated Chromium/CDP warnings. api_model selects the model; Gemini accepts either the bare model ID or the provider's models/ prefix. api_base_url defaults to https://generativelanguage.googleapis.com/v1beta for Gemini and https://api.openai.com/v1 for OpenAI-compatible mode. The built-in Gemini example uses gemini-3.5-flash-lite. api_timeout controls the HTTP request timeout in seconds. Transient API responses (429, 500, 502, 503, and 504) are retried up to two times before the image is marked failed.

The WebUI caches provider model discovery for ten minutes. The model selector shows known Gemini choices immediately while refreshing the provider's model list in the background.

Local and API VLM runtimes share one SD 1.5-focused caption prompt. The VLM returns one concise, single-line caption with comma-separated tags or short visual phrases, covering framing, pose/action, clothing/accessories, expression/gaze, subject count when clear, and background/lighting. It aims for roughly 25-45 words and fewer than 70 CLIP tokens to leave room within SD 1.5's 77-token text-encoder window. The prompt avoids stable identity traits such as hair/eye color, hairstyle, facial identity, age, and gender so a character/person LoRA can learn those through its dataset trigger. The pipeline prepends the resolved trigger and accepts both this flat caption format and legacy five-field VLM output.

API requests use the shared built-in prompt and do not expose local sampling/runtime settings such as temperature, top-p, top-k, seed, context size, or chat format. The API adapter sends the image, the fixed prompt, and the selected API model.

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
