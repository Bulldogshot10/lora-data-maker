from __future__ import annotations

import base64
import mimetypes
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from PIL import Image, UnidentifiedImageError

from .exceptions import LoraDataMakerError
from .security import MAX_IMAGE_PIXELS, MAX_IMAGE_UPLOAD_BYTES

DEFAULT_VLM_PROMPT = (
    "Create concise, accurate training captions for human character LoRAs on Stable Diffusion 1.5 (SD 1.5).\n"
    "Describe the visible, image-specific attributes that should remain independently controllable during generation. Inspect the whole image and report only details supported by visible evidence.\n"
    "\n"
    "CAPTION PRIORITY — describe the most useful details first:\n"
    "1. Framing and composition: extreme close-up, close-up portrait, head-and-shoulders, bust shot, upper body, cowboy shot, full body, wide shot, subject placement, foreground and visible cropping.\n"
    "2. Camera viewpoint: front view, side profile, three-quarter view, eye-level, low angle, high angle, overhead view, camera perspective and shot distance.\n"
    "3. Pose and physical positioning: standing, sitting, kneeling, leaning, walking, torso orientation, weight distribution, arm position, hand gestures, leg position, crossed legs, and visible actions.\n"
    "4. Clothing and accessories: garment type, visible colors, patterns, material when clear, layers, fit, jewelry, eyewear, bags, and other visible accessories.\n"
    "5. Expression and gaze: smiling, neutral expression, frowning, eyes closed, looking at viewer, looking away, and gaze direction when clearly visible.\n"
    "6. Objects and interaction: visible props, handheld objects, furniture, and how the subject interacts with nearby objects.\n"
    "7. Environment: indoor or outdoor setting, identifiable location, background objects, foreground objects, and relevant environmental context.\n"
    "8. Lighting and photographic appearance: natural window light, direct flash, soft light, hard shadows, backlighting, overcast daylight, artificial lighting, depth of field, and visible photographic or illustration medium when clear.\n"
    "\n"
    "CHARACTER IDENTITY POLICY:\n"
    "The dataset trigger represents the character's stable identity. Leave stable facial identity, facial structure, eye color, permanent hair color, stable skin tone, body proportions, and other defining identity traits for the trigger to learn.\n"
    "Do not infer age, ethnicity, personality, relationships, intentions, or personal history.\n"
    "Do not describe hair color or hairstyle, eye color, or other stable identity traits; describe hair styling, makeup, clothing, and accessories only when they are visibly varied and intended to remain independently controllable. Omit signature traits that appear to be permanent identity features.\n"
    "Do not output the character name, dataset trigger, or subject class noun such as person, woman, or man; the dataset pipeline adds it automatically.\n"
    "Do not guess hidden, occluded, ambiguous, or off-frame details.\n"
    "\n"
    "CAPTION QUALITY:\n"
    "Use concrete visual vocabulary and familiar short tags or phrases that Stable Diffusion 1.5's CLIP text encoder can understand.\n"
    "Prefer approximately 6-12 informative items and 25-45 words total, ideally below roughly 70 CLIP tokens.\n"
    "Prioritize details that distinguish this image from other images of the same character. Avoid redundant synonyms, repeated descriptions, generic praise, subjective judgments, quality-rating tags, negative prompts, unsupported camera specifications, and long narrative sentences.\n"
    "Do not caption watermarks, signatures, logos, or incidental text.\n"
    "\n"
    "OUTPUT FORMAT:\n"
    "Return exactly one concise, single-line caption using comma-separated tags or short visual phrases.\n"
    "Output only the comma-separated caption on one line. No explanations, headings, field labels, bullets, numbering, quotations, markdown, or introductory text.\n"
    "\n"
    "Example: upper body portrait, three-quarter view, leaning against a wall, right hand on hip, left arm raised, white halter top, black mini skirt, slight smile, looking away, indoor studio, soft side lighting\n"
    "Example: full body, seated on a chair, side profile, legs crossed, red wool coat, black boots, holding a handbag, city street, overcast daylight\n"
    "Example: close-up portrait, eye-level, looking at viewer, subtle smile, dark jacket, shallow depth of field, warm window light\n")



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
        file_size = path.stat().st_size
        if file_size < 1 or file_size > MAX_IMAGE_UPLOAD_BYTES:
            raise VlmCaptionerError("image exceeds the 25 MiB VLM input limit")
        with Image.open(path) as image:
            width, height = image.size
            if width < 1 or height < 1 or width * height > MAX_IMAGE_PIXELS:
                raise VlmCaptionerError(
                    f"unsafe image dimensions {width}x{height}; maximum is {MAX_IMAGE_PIXELS} pixels"
                )
            image_format = (image.format or "").upper()
        payload = path.read_bytes()
    except VlmCaptionerError:
        raise
    except (OSError, UnidentifiedImageError) as exc:
        raise VlmCaptionerError(
            f"could not read a valid image {path}: {exc}"
        ) from exc

    mime_by_format = {
        "JPEG": "image/jpeg",
        "PNG": "image/png",
        "WEBP": "image/webp",
        "BMP": "image/bmp",
        "GIF": "image/gif",
        "TIFF": "image/tiff",
        "AVIF": "image/avif",
    }
    mime_type = mime_by_format.get(image_format) or mimetypes.guess_type(path.name)[0] or "image/jpeg"
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

        caption_content = ""
        if isinstance(raw_content, str):
            # Reasoning-only completions are not usable captions.
            caption_content = re.sub(
                r"<think\b[^>]*>.*?</think\s*>",
                "",
                raw_content,
                flags=re.IGNORECASE | re.DOTALL,
            ).strip()

        if not caption_content:
            raw_preview = repr(raw_content)
            if len(raw_preview) > 300:
                raw_preview = raw_preview[:297] + "..."
            raise VlmCaptionerError(
                f"VLM returned an empty caption for {path.name}; "
                f"finish_reason={finish_reason!r}; "
                f"completion_tokens={completion_tokens!r}; "
                f"raw_content={raw_preview}"
            )

        return caption_content

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
