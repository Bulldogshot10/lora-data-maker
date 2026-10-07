# llama.cpp backends

The project can use a local llama-server as a multimodal VLM backend for both tagging and captioning. The Python side talks to the OpenAI-compatible /v1/chat/completions endpoint, so the llama.cpp binary stays outside the Python dependency set.

Current llama.cpp server builds expose multimodal chat input through image_url content parts. A vision model needs its supported multimodal projector enabled when the server is started.

Example server:

~~~bash
llama-server \
  -m /path/to/model.gguf \
  --mmproj /path/to/mmproj.gguf \
  --host 127.0.0.1 \
  --port 8080
~~~

Then configure the project:

~~~toml
[tagger]
backend = "llama_cpp"

[tagger.llama_cpp]
base_url = "http://127.0.0.1:8080/v1"
model = "local-model"
temperature = 0.0
max_tokens = 256

[caption]
backend = "llama_cpp"

[caption.llama_cpp]
base_url = "http://127.0.0.1:8080/v1"
model = "local-model"
temperature = 0.2
max_tokens = 128
~~~

Run the stages normally:

~~~bash
lora-data-maker tag
lora-data-maker clean-tags
lora-data-maker caption
~~~

CLI overrides are also available:

~~~bash
lora-data-maker tag --backend llama_cpp
lora-data-maker caption --backend llama_cpp
~~~

The tagger asks for JSON and maps tags, scores, ratings, and characters into the existing TagResult. The later clean-tags stage still owns subject and trigger injection.

The captioner receives the image plus existing tag hints and produces a concise comma-separated caption. The dataset trigger is enforced as the first caption fragment so the LLM cannot silently drop the training identity.

No llama.cpp Python wheel is required. Install or build llama-server separately and point the config at its local /v1 endpoint.
