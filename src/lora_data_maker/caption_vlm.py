from __future__ import annotations

import base64
import mimetypes
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from .exceptions import LoraDataMakerError

DEFAULT_VLM_PROMPT = (
    "describe this photo for a stable diffusion character lora training caption.\n"
    "use exactly these 5 fields, one per line.\n"
    "use short phrases only, with a maximum of 5 words per field.\n\n"
    "pose: describe body pose and shot type, e.g. close-up portrait, medium shot standing, cowboy shot, full-body shot standing, wide shot seated.\n"
    "choose shot type from visible framing: extreme close-up, close-up, medium close-up, medium shot, medium-long shot, cowboy shot, full-body shot, wide shot.\n"
    "angle camera: describe the visible camera angle or view\n"
    "prefer camera terms: straight-on, from above, from below, high-angle shot, low-angle shot, overhead shot.\n"
    "clothing: describe visible clothing\n"
    "expression: describe visible facial expression\n"
    "background: describe setting and lighting\n\n"
    "do not mention hair color, eye color, face, age, or gender.\n"
    "no extra text. be factual, concise, and specific.\n"
    "prioritize concrete visual details over generic descriptions.\n"
    "do not guess hidden or ambiguous details.\n"
    "do not mention image quality, the image itself, ai, or this task."
)


class Captioner(Protocol):
    def caption(self, image_path: Path | str) -> str:
        """Generate one caption for a processed image."""

    def close(self) -> None:
        """Release any runtime resources."""


class VlmCaptionerError(LoraDataMakerError):
    """Raised when a VLM captioner cannot initialize or infer."""


@dataclass(frozen=True)
class VlmConfig:
    model_path: str = ""
    mmproj_path: str = ""
    runtime: str = "llama_cpp"
    chat_format: str = "mtmd"
    prompt: str = DEFAULT_VLM_PROMPT
    max_tokens: int = 64
    temperature: float = 0.1
    api_key_env: str = ""
    api_model: str = ""
    api_base_url: str = ""
    api_timeout: float = 120.0
    top_p: float = 0.95
    top_k: int = 40
    n_ctx: int = 4096
    n_threads: int = 4
    n_gpu_layers: int = 0
    seed: int = 42

    def __post_init__(self) -> None:
        if self.runtime not in {"llama_cpp", "onnx", "gemini", "openai_compatible"}:
            raise ValueError(
                "caption.vlm.runtime must be one of 'llama_cpp', 'onnx', 'gemini', or 'openai_compatible'"
            )
        if self.chat_format not in {"mtmd", "qwen2.5-vl", "gemma4", "qwen3.5", "smolvlm"}:
            raise ValueError(
                "caption.vlm.chat_format must be one of "
                "'mtmd', 'qwen2.5-vl', 'gemma4', 'qwen3.5', or 'smolvlm'"
            )

        has_model = bool(self.model_path.strip())
        has_mmproj = bool(self.mmproj_path.strip())
        if self.runtime == "llama_cpp" and has_model != has_mmproj:
            raise ValueError(
                "caption.vlm.model_path and caption.vlm.mmproj_path "
                "must be configured together for the llama_cpp runtime"
            )

        if not self.prompt.strip():
            raise ValueError("caption.vlm.prompt must not be empty")
        if self.max_tokens < 1:
            raise ValueError("caption.vlm.max_tokens must be greater than zero")
        if not 0.0 <= self.temperature:
            raise ValueError("caption.vlm.temperature must be >= 0")
        if not 0.0 < self.top_p <= 1.0:
            raise ValueError("caption.vlm.top_p must be in (0, 1]")
        if self.top_k < 0:
            raise ValueError("caption.vlm.top_k must be >= 0")
        if self.n_ctx < 256:
            raise ValueError("caption.vlm.n_ctx must be >= 256")
        if self.n_threads < 1:
            raise ValueError("caption.vlm.n_threads must be greater than zero")
        if self.api_timeout <= 0.0:
            raise ValueError("caption.vlm.api_timeout must be greater than zero")


def _data_uri(path: Path) -> str:
    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise VlmCaptionerError(
            f"could not read processed image {path}: {exc}"
        ) from exc

    mime_type = mimetypes.guess_type(path.name)[0] or "image/jpeg"
    encoded = base64.b64encode(payload).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def _load_chat_handler(
    *,
    chat_format: str,
    mmproj_path: str,
    use_gpu: bool,
) -> Any:
    try:
        from llama_cpp import llama_chat_format
    except ImportError as exc:
        raise VlmCaptionerError(
            "VLM captions require llama-cpp-python; install the optional "
            "caption-vlm dependency"
        ) from exc

    handler_map = {
        "mtmd": (
            getattr(llama_chat_format, "MTMDChatHandler", None),
            "MTMDChatHandler",
        ),
        "qwen3.5": (
            getattr(llama_chat_format, "MTMDChatHandler", None),
            "MTMDChatHandler",
        ),
        "smolvlm": (
            getattr(llama_chat_format, "MTMDChatHandler", None),
            "MTMDChatHandler",
        ),
        "qwen2.5-vl": (
            getattr(llama_chat_format, "Qwen25VLChatHandler", None),
            "Qwen25VLChatHandler",
        ),
        "gemma4": (
            getattr(llama_chat_format, "Gemma4ChatHandler", None),
            "Gemma4ChatHandler",
        ),
    }
    handler_class, handler_name = handler_map[chat_format]
    if handler_class is None:
        raise VlmCaptionerError(
            f"installed llama-cpp-python does not provide the "
            f"{handler_name} required by chat format {chat_format!r}"
        )

    try:
        kwargs: dict[str, Any] = {
            "clip_model_path": mmproj_path,
            "verbose": False,
        }
        if chat_format in {"mtmd", "qwen3.5", "smolvlm", "gemma4"}:
            kwargs["use_gpu"] = use_gpu
        return handler_class(**kwargs)
    except Exception as exc:
        raise VlmCaptionerError(
            f"could not initialize {chat_format} multimodal projector: {exc}"
        ) from exc


class LlamaCppVlmCaptioner:
    """In-process local VLM caption runtime backed by llama-cpp-python."""

    def __init__(self, config: VlmConfig) -> None:
        self.config = config
        model_path = Path(config.model_path).expanduser().resolve()
        mmproj_path = Path(config.mmproj_path).expanduser().resolve()

        if not model_path.is_file():
            raise VlmCaptionerError(f"VLM model not found: {model_path}")
        if not mmproj_path.is_file():
            raise VlmCaptionerError(
                f"VLM multimodal projector not found: {mmproj_path}"
            )

        try:
            import llama_cpp
        except ImportError as exc:
            raise VlmCaptionerError(
                "VLM captions require llama-cpp-python; install "
                "lora-data-maker[caption-vlm]"
            ) from exc

        handler = _load_chat_handler(
            chat_format=config.chat_format,
            mmproj_path=str(mmproj_path),
            use_gpu=config.n_gpu_layers != 0,
        )

        try:
            self._llm = llama_cpp.Llama(
                model_path=str(model_path),
                chat_handler=handler,
                n_ctx=config.n_ctx,
                n_threads=config.n_threads,
                n_gpu_layers=config.n_gpu_layers,
                seed=config.seed,
                use_mmap=True,
                verbose=False,
            )
        except Exception as exc:
            raise VlmCaptionerError(
                f"could not load VLM model {model_path}: {exc}"
            ) from exc

        self.model_path = model_path
        self.mmproj_path = mmproj_path

    def caption(self, image_path: Path | str) -> str:
        path = Path(image_path).expanduser().resolve()
        image_url = _data_uri(path)

        prompt = self.config.prompt
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": image_url},
                    },
                    {
                        "type": "text",
                        "text": prompt,
                    },
                ],
            }
        ]

        def generate(request_messages: list[dict[str, Any]], **overrides: Any) -> dict[str, Any]:
            params: dict[str, Any] = {
                "messages": request_messages,
                "max_tokens": self.config.max_tokens,
                "temperature": self.config.temperature,
                "top_p": self.config.top_p,
                "top_k": self.config.top_k,
                "seed": self.config.seed,
                "stream": False,
            }
            params.update(overrides)
            return self._llm.create_chat_completion(**params)

        try:
            response = generate(messages)
        except Exception as exc:
            raise VlmCaptionerError(
                f"VLM inference failed for {path.name}: {exc}"
            ) from exc

        try:
            choice = response["choices"][0]
            message = choice["message"]
            raw_content = message.get("content")
            finish_reason = choice.get("finish_reason")
            usage = response.get("usage")
            completion_tokens = (
                usage.get("completion_tokens")
                if isinstance(usage, dict)
                else None
            )
        except (KeyError, IndexError, TypeError, AttributeError) as exc:
            raise VlmCaptionerError(
                f"VLM returned an invalid chat completion for {path.name}"
            ) from exc

        if not isinstance(raw_content, str) or not raw_content.strip():
            raw_preview = repr(raw_content)
            if len(raw_preview) > 300:
                raw_preview = raw_preview[:297] + "..."
            raise VlmCaptionerError(
                f"VLM returned an empty caption for {path.name}; "
                f"finish_reason={finish_reason!r}; "
                f"completion_tokens={completion_tokens!r}; "
                f"raw_content={raw_preview}"
            )

        return raw_content

    def close(self) -> None:
        llm = getattr(self, "_llm", None)
        if llm is None:
            return
        try:
            close = getattr(llm, "close", None)
            if callable(close):
                close()
        finally:
            self._llm = None  # type: ignore[assignment]
