# llama.cpp embedded runtime

The project can run a local multimodal GGUF model directly inside the Python process through `llama-cpp-python`. The tagger and captioner do not connect to `llama-server` and do not require an HTTP endpoint.

`llama-cpp-python` builds and installs llama.cpp alongside its Python bindings. Its current multimodal support exposes `MTMDChatHandler`, which loads an external `mmproj` and feeds image content into the same in-process Llama context. citeturn596160search0turn270397search0

Install the optional dependency:

~~~bash
python -m pip install ".[llama_cpp]"
~~~

For Termux, the upstream llama.cpp project supports building its native runtime directly on Android/Termux. The Python binding is a separate packaging layer, so the most portable GUI architecture is to keep the runtime behind a small Python adapter. citeturn974045search0turn596160search5

Configure a local model and multimodal projector:

~~~toml
[tagger]
backend = "llama_cpp"

[tagger.llama_cpp]
model_path = "/data/data/com.termux/files/home/models/vision-Q4_K_M.gguf"
mmproj_path = "/data/data/com.termux/files/home/models/mmproj.gguf"
n_ctx = 4096
n_threads = 4
n_gpu_layers = -1
verbose = false
temperature = 0.0
max_tokens = 256
max_tags = 64
seed = 0

[caption]
backend = "llama_cpp"

[caption.llama_cpp]
model_path = "/data/data/com.termux/files/home/models/vision-Q4_K_M.gguf"
mmproj_path = "/data/data/com.termux/files/home/models/mmproj.gguf"
n_ctx = 4096
n_threads = 4
n_gpu_layers = -1
verbose = false
temperature = 0.2
max_tokens = 128
seed = 0
~~~

`n_gpu_layers = -1` asks llama.cpp to offload all supported layers. Set it to `0` for CPU-only execution. The correct value depends on the native backend compiled into the installed llama.cpp build.

The runtime is intentionally reusable:

~~~text
GUI service
    ↓
LlamaCppRuntime (one loaded model)
    ├── VLM tagger
    └── VLM captioner
~~~

This means the future GUI can load one GGUF/mmproj pair and reuse the same in-memory context instead of loading the model again for every image.

Basic CLI usage:

~~~bash
lora-data-maker tag
lora-data-maker clean-tags
lora-data-maker caption
~~~

The tagger returns structured JSON internally and maps it into the existing `TagResult`. The later `clean-tags` stage remains responsible for dataset subject/trigger injection.

The captioner receives the processed image and existing tag hints, then enforces the dataset trigger as the first caption fragment after generation.

No `llama-server`, localhost port, REST client, or Python server extra is needed for this backend. The actual native runtime is loaded in-process by `llama-cpp-python`. citeturn596160search0turn270397search0
