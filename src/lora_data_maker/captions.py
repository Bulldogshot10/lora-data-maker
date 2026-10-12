from __future__ import annotations

import json
import os
import re
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Iterable

from .caption_api import GeminiVlmCaptioner, OpenAICompatibleVlmCaptioner
from .caption_vlm import (
    DEFAULT_VLM_PROMPT,
    Captioner,
    LlamaCppVlmCaptioner,
    VlmConfig,
    VlmCaptionerError,
)
from .paths import manifest_path
from .taggers.filters import normalize_tag

if TYPE_CHECKING:
    from .models import ProjectManifest


@dataclass(frozen=True)
class CaptionSummary:
    scanned: int = 0
    captioned: int = 0
    failed: int = 0
    skipped: int = 0
    fallback: int = 0


@dataclass(frozen=True)
class CaptionConfig:
    separator: str = ", "
    subject_position: str = "first"
    trigger_from_subject: bool = True
    backend: str = "deterministic"
    max_tokens: int = 77
    vlm: VlmConfig = field(default_factory=VlmConfig)

    def __post_init__(self) -> None:
        if self.subject_position != "first":
            raise ValueError(
                "caption.subject_position must be 'first'"
            )
        if self.backend not in {"deterministic", "vlm"}:
            raise ValueError(
                "caption.backend must be 'deterministic' or 'vlm'"
            )
        if not 1 <= self.max_tokens <= 77:
            raise ValueError("caption.max_tokens must be between 1 and 77")
        if self.backend == "vlm":
            if self.vlm.runtime == "llama_cpp" and not self.vlm.model_path.strip():
                raise ValueError("caption.vlm.model_path is required for the llama_cpp runtime")
            if self.vlm.runtime == "llama_cpp" and not self.vlm.mmproj_path.strip():
                raise ValueError(
                    "caption.vlm.mmproj_path is required for the llama_cpp runtime"
                )
            if self.vlm.runtime in {"gemini", "openai_compatible"}:
                if not self.vlm.api_key_env.strip():
                    raise ValueError(
                        "caption.vlm.api_key_env must not be empty for API runtimes"
                    )
                if not self.vlm.api_model.strip():
                    raise ValueError(
                        "caption.vlm.api_model must not be empty for API runtimes"
                    )
        if not isinstance(self.separator, str):
            raise ValueError("caption.separator must be a string")
        if not isinstance(self.trigger_from_subject, bool):
            raise ValueError(
                "caption.trigger_from_subject must be a boolean"
            )


def caption_config_from_dict(
    data: dict[str, object],
    *,
    root: Path | str | None = None,
) -> CaptionConfig:
    """Parse a project [caption] table into a validated runtime config."""
    separator = data.get("separator", ", ")
    subject_position = data.get("subject_position", "first")
    trigger_from_subject = data.get("trigger_from_subject", True)
    backend = str(data.get("backend", "deterministic")).strip().lower()
    max_tokens = data.get("max_tokens", 77)
    if isinstance(max_tokens, bool) or not isinstance(max_tokens, int):
        raise ValueError("config field caption.max_tokens must be an integer")

    if not isinstance(separator, str):
        raise ValueError("config field caption.separator must be a string")
    if not isinstance(subject_position, str):
        raise ValueError(
            "config field caption.subject_position must be a string"
        )
    if not isinstance(trigger_from_subject, bool):
        raise ValueError(
            "config field caption.trigger_from_subject must be a boolean"
        )

    vlm_data = data.get("vlm", {})
    if not isinstance(vlm_data, dict):
        raise ValueError("config section [caption.vlm] must be a table")

    def string_field(name: str, default: str) -> str:
        value = vlm_data.get(name, default)
        if not isinstance(value, str):
            raise ValueError(
                f"config field caption.vlm.{name} must be a string"
            )
        return value

    def int_field(name: str, default: int) -> int:
        value = vlm_data.get(name, default)
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(
                f"config field caption.vlm.{name} must be an integer"
            )
        return value

    def float_field(name: str, default: float) -> float:
        value = vlm_data.get(name, default)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(
                f"config field caption.vlm.{name} must be a number"
            )
        return float(value)

    vlm_runtime = string_field("runtime", "llama_cpp").strip().lower()
    model_path = string_field("model_path", "")
    mmproj_path = string_field("mmproj_path", "")
    if root is not None:
        root_path = Path(root).expanduser().resolve()
        model_candidate = Path(model_path).expanduser() if model_path else None
        is_hf_repo_id = (
            vlm_runtime == "onnx"
            and bool(
                re.fullmatch(
                    r"[A-Za-z0-9._-]+/[A-Za-z0-9._-]+",
                    model_path.strip(),
                )
            )
            and model_candidate is not None
            and not (root_path / model_candidate).is_dir()
        )
        if model_path and not Path(model_path).expanduser().is_absolute() and not is_hf_repo_id:
            model_path = str((root_path / model_path).resolve())
        if mmproj_path and not Path(mmproj_path).expanduser().is_absolute():
            mmproj_path = str((root_path / mmproj_path).resolve())

    vlm_chat_format = string_field("chat_format", "mtmd").strip().lower()
    if vlm_runtime == "onnx":
        vlm_chat_format = "smolvlm"

    default_api_key_env = (
        "GEMINI_API_KEY" if vlm_runtime == "gemini" else "OPENAI_API_KEY"
    )
    default_api_model = (
        "gemini-3.5-flash-lite" if vlm_runtime == "gemini" else ""
    )
    default_api_base_url = (
        "https://generativelanguage.googleapis.com/v1beta"
        if vlm_runtime == "gemini"
        else "https://api.openai.com/v1"
    )
    api_key_env = string_field("api_key_env", default_api_key_env).strip()
    api_model = string_field("api_model", default_api_model).strip()
    api_base_url = string_field("api_base_url", default_api_base_url).strip()
    api_timeout = float_field("api_timeout", 120.0)

    if vlm_runtime in {"gemini", "openai_compatible"}:
        if not api_key_env:
            api_key_env = default_api_key_env
        if vlm_runtime == "gemini" and not api_model:
            api_model = default_api_model
        if not api_base_url:
            api_base_url = default_api_base_url

    if vlm_runtime == "gemini":
        api_model = api_model.removeprefix("models/")

    vlm = VlmConfig(
        model_path=model_path,
        mmproj_path=mmproj_path,
        runtime=vlm_runtime,
        chat_format=vlm_chat_format,
        prompt=string_field("prompt", DEFAULT_VLM_PROMPT),
        max_tokens=int_field("max_tokens", 64),
        temperature=float_field("temperature", 0.1),
        top_p=float_field("top_p", 0.95),
        top_k=int_field("top_k", 40),
        n_ctx=int_field("n_ctx", 4096),
        n_threads=int_field("n_threads", 4),
        n_gpu_layers=int_field("n_gpu_layers", 0),
        seed=int_field("seed", 42),
        api_key_env=api_key_env,
        api_model=api_model,
        api_base_url=api_base_url,
        api_timeout=api_timeout,
    )

    return CaptionConfig(
        separator=separator,
        subject_position=subject_position,
        trigger_from_subject=trigger_from_subject,
        backend=backend,
        max_tokens=max_tokens,
        vlm=vlm,
    )


def resolve_trigger(
    *,
    subject: str,
    trigger: str,
    trigger_from_subject: bool,
) -> str:
    """Resolve the caption trigger from dataset identity settings."""
    selected = subject if trigger_from_subject else trigger
    return normalize_tag(selected)


def build_caption(
    tags: Iterable[str],
    *,
    subject: str,
    trigger: str,
    config: CaptionConfig | None = None,
) -> str:
    """Build one deterministic caption with an idempotent leading trigger."""
    settings = config or CaptionConfig()

    normalized_tags: list[str] = []
    seen: set[str] = set()
    for raw_tag in tags:
        tag = normalize_tag(raw_tag)
        if not tag or tag in seen:
            continue
        seen.add(tag)
        normalized_tags.append(tag)

    resolved_trigger = resolve_trigger(
        subject=subject,
        trigger=trigger,
        trigger_from_subject=settings.trigger_from_subject,
    )

    if resolved_trigger:
        normalized_tags = [
            tag for tag in normalized_tags if tag != resolved_trigger
        ]
        normalized_tags.insert(0, resolved_trigger)

    return truncate_caption(
        settings.separator.join(normalized_tags),
        settings.max_tokens,
        separator=settings.separator,
    )


_LEADING_NOISE = re.compile(
    r"^(?:caption|description|answer)\s*:\s*",
    re.IGNORECASE,
)
_LEADING_PREFIXES = (
    "the image shows ",
    "this image shows ",
    "the photo shows ",
    "this photo shows ",
    "the picture shows ",
    "this picture shows ",
    "this is a photo of ",
    "this is an image of ",
    "a photo of ",
    "an image of ",
)


def clean_vlm_caption(text: str) -> str:
    """Normalize raw VLM output into one compact training-caption line."""
    value = str(text).replace("\r\n", "\n").replace("\r", "\n").strip()
    value = _LEADING_NOISE.sub("", value)
    value = value.replace("\n", " ")
    value = value.strip().strip(chr(96)).strip()
    value = re.sub(r"^[-*•]\s+", "", value)
    value = re.sub(r"\s+", " ", value).strip()

    lowered = value.lower()
    for prefix in _LEADING_PREFIXES:
        if lowered.startswith(prefix):
            value = value[len(prefix):].lstrip()
            break

    return value.strip(chr(96)).strip()


_VLM_FIELDS = ("pose", "angle camera", "clothing", "expression", "background")
_VLM_FIELD_RE = re.compile(
    r"(?<![A-Za-z0-9_])(pose|angle[ _-]+camera|camera[ _-]+angle|clothing|expression|background)\s*:\s*",
    re.IGNORECASE,
)

_VLM_UNKNOWN_FIELD_LINE_RE = re.compile(
    r"^\s*[A-Za-z][A-Za-z0-9 _-]{0,30}\s*:\s*.*$"
)



def _strip_vlm_wrappers(text: str) -> str:
    """Remove common model wrappers without touching caption content."""
    value = str(text).replace("\r\n", "\n").replace("\r", "\n").strip()
    value = re.sub(r"^\s*\x60\x60\x60(?:json|text|plaintext)?\s*", "", value, flags=re.IGNORECASE)
    value = re.sub(r"\s*\x60\x60\x60\s*$", "", value)
    value = re.sub(r"(?m)^[ \t]*(?:[-*•]|\d+[.)])[ \t]+", "", value)
    return value.strip()


def _parse_vlm_json(value: str) -> list[str] | None:
    """Accept structured JSON as a tolerant API-side fallback."""
    candidate = value.strip()
    if not candidate.startswith("{") or not candidate.endswith("}"):
        return None
    try:
        data = json.loads(candidate)
    except (TypeError, ValueError):
        return None
    if not isinstance(data, dict):
        return None

    fields = []
    for field_name in _VLM_FIELDS:
        raw_value = data.get(field_name)
        if field_name == "angle camera" and raw_value is None:
            raw_value = data.get("camera angle", data.get("angle_camera"))
        if not isinstance(raw_value, str):
            return None
        phrase = clean_vlm_caption(raw_value)
        if not phrase:
            raise VlmCaptionerError(f"VLM field {field_name} is empty")
        fields.append(phrase)
    return fields


def parse_vlm_fields(raw_caption: str) -> list[str]:
    """Extract the five caption fields from tolerant, structured VLM output."""
    value = _strip_vlm_wrappers(raw_caption)
    if not value:
        raise VlmCaptionerError("VLM caption is empty")

    json_fields = _parse_vlm_json(value)
    if json_fields is not None:
        return json_fields

    matches = list(_VLM_FIELD_RE.finditer(value))
    if not matches:
        raise VlmCaptionerError(
            "VLM caption does not contain the expected pose, angle camera, clothing, expression, background fields"
        )

    by_name = {}
    for index, match in enumerate(matches):
        name = re.sub(r"[ _-]+", " ", match.group(1).lower()).strip()
        if name == "camera angle":
            name = "angle camera"
        if name in by_name:
            continue
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(value)
        fragment = value[start:end].strip()
        fragment = re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", fragment)
        fragment_lines = []
        for line in fragment.split("\n"):
            if _VLM_UNKNOWN_FIELD_LINE_RE.match(line.strip()):
                continue
            fragment_lines.append(line)
        fragment = " ".join(fragment_lines).strip(" ,;|")
        phrase = clean_vlm_caption(fragment)
        if phrase:
            by_name[name] = phrase

    missing = [field_name for field_name in _VLM_FIELDS if not by_name.get(field_name)]
    if missing:
        raise VlmCaptionerError(
            "VLM caption is missing required fields: " + ", ".join(missing)
        )

    return [by_name[field_name] for field_name in _VLM_FIELDS]

def parse_vlm_caption(raw_caption: str) -> str:
    """Parse the VLM response into one ordered caption fragment."""
    return ", ".join(parse_vlm_fields(raw_caption))


def reset_vlm_captions_for_deterministic(
    manifest: "ProjectManifest",
    *,
    root: Path | str,
) -> int:
    """Requeue VLM captions before entering the deterministic tag pipeline."""
    root_path = Path(root).expanduser().resolve()
    reset = 0

    for image in manifest.images.values():
        if image.status != "captioned":
            continue
        if str(image.metadata.get("caption_backend", "")).strip().lower() != "vlm":
            continue

        if image.caption_path:
            stale_caption = (root_path / image.caption_path).resolve()
            try:
                stale_caption.relative_to(root_path)
            except ValueError:
                stale_caption = None
            if stale_caption is not None and stale_caption.is_file():
                try:
                    stale_caption.unlink()
                except OSError:
                    pass

        image.caption_path = None
        image.metadata["caption_stage"] = "invalidated_backend_switch"
        image.metadata["caption_previous_backend"] = "vlm"
        image.set_status("processed")
        reset += 1

    if reset:
        manifest.touch()
    return reset


def _normalize_api_vlm_fields(fields: list[str]) -> list[str]:
    """Turn the five API fields into four consistent, readable caption phrases."""
    if len(fields) != 5:
        return fields

    pose, camera_angle, clothing, expression, background = (
        re.sub(r"\s+", " ", str(value)).strip(" ,.;:")
        for value in fields
    )

    # Fold the camera angle into the pose phrase so the API caption has four fields.
    if camera_angle and camera_angle.casefold() not in pose.casefold():
        pose = " ".join(part for part in (pose, camera_angle) if part)

    # Normalize clothing into a natural phrase without duplicating existing wording.
    if clothing and not re.match(
        r"^(?:wearing\b|dressed\s+in\b|clad\s+in\b|in\s+(?:a|an|the)\b)",
        clothing,
        re.IGNORECASE,
    ):
        if re.match(r"^(?:a|an|the)\s+", clothing, re.IGNORECASE):
            clothing = "wearing " + clothing
        else:
            last_word = clothing.split()[-1].strip(".,;:").lower()
            plural_items = {
                "jeans", "pants", "shorts", "leggings", "tights", "trousers",
                "clothes", "clothing", "overalls", "glasses", "sunglasses",
                "socks", "shoes", "boots", "sneakers", "sandals", "gloves",
                "earrings",
            }
            article = "" if last_word in plural_items else (
                "an " if clothing.split()[0][:1].lower() in {"a", "e", "i", "o", "u"} else "a "
            )
            clothing = "wearing " + article + clothing

    if expression and not re.search(r"\bexpression\b", expression, re.IGNORECASE):
        expression += " expression"

    # Make the common "indoor/outdoor + lighting" shape read naturally while
    # leaving specific location phrases such as "indoor studio" intact.
    if background and not re.match(r"^(?:indoors|outdoors)\s+(?:with|in)\b", background, re.IGNORECASE):
        match = re.match(r"^(indoor|indoors|outdoor|outdoors)\s+(.+)$", background, re.IGNORECASE)
        if match:
            setting, detail = match.groups()
            if re.search(r"\b(?:light|lighting|illumination|sunlight|shadows?)\b", detail, re.IGNORECASE):
                location = "indoors" if setting.lower().startswith("indoor") else "outdoors"
                background = f"{location} with {detail}"

    return [pose, clothing, expression, background]


def build_api_vlm_caption(
    raw_caption: str,
    *,
    subject: str,
    trigger: str,
    config: CaptionConfig,
) -> str:
    """Normalize an API VLM caption and inject the trigger, even if its format is unstructured."""
    try:
        fields = parse_vlm_fields(raw_caption)
    except VlmCaptionerError as parse_error:
        # Keep non-empty API output so it can be audited instead of failing the image.
        fallback = clean_vlm_caption(_strip_vlm_wrappers(raw_caption))
        if not fallback:
            raise parse_error
        fields = [fallback]
    else:
        fields = _normalize_api_vlm_fields(fields)

    resolved_trigger = resolve_trigger(
        subject=subject,
        trigger=trigger,
        trigger_from_subject=config.trigger_from_subject,
    )

    cleaned_fields: list[str] = []
    for field_value in fields:
        value = field_value
        if resolved_trigger and value:
            pattern = re.compile(
                rf"(?<![A-Za-z0-9_]){re.escape(resolved_trigger)}(?![A-Za-z0-9_])",
                re.IGNORECASE,
            )
            value = pattern.sub("", value)
            value = re.sub(r"\s+", " ", value).strip(" ,;:")
        if value:
            cleaned_fields.append(value)

    parts: list[str] = []
    if resolved_trigger:
        parts.append(resolved_trigger)
    parts.extend(cleaned_fields)
    return config.separator.join(parts)


def build_vlm_caption(
    raw_caption: str,
    *,
    subject: str,
    trigger: str,
    config: CaptionConfig,
) -> str:
    """Build a VLM caption from strict fields."""
    fields = parse_vlm_fields(raw_caption)
    # Keep the local training caption focused on pose, clothing, expression, and background.
    fields = [fields[0], *fields[2:]]
    resolved_trigger = resolve_trigger(
        subject=subject,
        trigger=trigger,
        trigger_from_subject=config.trigger_from_subject,
    )

    cleaned_fields: list[str] = []
    for field_value in fields:
        value = field_value
        if resolved_trigger and value:
            pattern = re.compile(
                rf"(?<![A-Za-z0-9_]){re.escape(resolved_trigger)}(?![A-Za-z0-9_])",
                re.IGNORECASE,
            )
            value = pattern.sub("", value)
            value = re.sub(r"\s+", " ", value).strip(" ,;:")
        if value:
            cleaned_fields.append(value)

    parts: list[str] = []
    if resolved_trigger:
        parts.append(resolved_trigger)
    parts.extend(cleaned_fields)

    return config.separator.join(parts)


def estimate_caption_tokens(text: str) -> int:
    """Estimate deterministic caption token usage without a tokenizer dependency."""
    value = str(text).strip()
    if not value:
        return 0

    token_count = 2  # BOS + EOS used by the SD1.x CLIP text encoder.
    for piece in re.findall(r"[A-Za-z0-9_]+|[^A-Za-z0-9_\s]", value):
        if piece.isalnum() or "_" in piece:
            token_count += max(1, (len(piece) + 3) // 4)
        else:
            token_count += 1
    return token_count


def truncate_caption(
    text: str,
    max_tokens: int,
    *,
    separator: str = ", ",
) -> str:
    """Trim a caption to the configured conservative token estimate."""
    if max_tokens < 1:
        raise ValueError("caption max tokens must be greater than zero")
    if not isinstance(separator, str):
        raise ValueError("caption separator must be a string")

    value = re.sub(r"\s+", " ", str(text)).strip(" ,.;:")
    if not value or estimate_caption_tokens(value) <= max_tokens:
        return value

    if separator:
        parts = [
            part.strip(" ,.;:")
            for part in re.split(re.escape(separator), value)
        ]
    else:
        parts = [value]
    parts = [part for part in parts if part]
    selected: list[str] = []

    for part in parts:
        candidate = separator.join((*selected, part))
        if estimate_caption_tokens(candidate) <= max_tokens:
            selected.append(part)
            continue
        break

    if selected:
        result = separator.join(selected).strip(" ,.;:")
        if result:
            return result

    words = re.findall(r"\S+", value)
    selected_words: list[str] = []
    for word in words:
        candidate = " ".join((*selected_words, word))
        if estimate_caption_tokens(candidate) <= max_tokens:
            selected_words.append(word)
        else:
            break
    return " ".join(selected_words).strip(" ,.;:")


def write_caption(path: Path | str, caption: str) -> Path:
    """Atomically write a UTF-8 caption file with a single trailing newline."""
    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)

    payload = caption.rstrip("\r\n") + "\n"
    fd, temp_name = tempfile.mkstemp(
        prefix=f".{target.name}.",
        suffix=".tmp",
        dir=target.parent,
        text=True,
    )

    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, target)
    except Exception:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass
        raise

    return target


def caption_manifest(
    manifest: "ProjectManifest",
    *,
    root: Path | str,
    output_dir: Path | str,
    config: CaptionConfig | None = None,
    captioner: Captioner | None = None,
    progress: Callable[[int, int, str], None] | None = None,
    use_processed_images: bool = True,
) -> CaptionSummary:
    """Write deterministic or local-VLM captions for eligible records."""
    root_path = Path(root).expanduser().resolve()
    caption_root = Path(output_dir).expanduser().resolve()
    settings = config or CaptionConfig()

    if settings.backend == "deterministic":
        reset_vlm_captions_for_deterministic(manifest, root=root_path)

    eligible_statuses = (
        (
            {"processed", "tagged", "captioned"}
            if use_processed_images
            else {"validated", "processed", "tagged", "captioned"}
        )
        if settings.backend == "vlm"
        else {"tagged", "captioned"}
    )
    candidates = sorted(
        (
            image
            for image in manifest.images.values()
            if image.status in eligible_statuses
        ),
        key=lambda image: image.path,
    )

    runtime = captioner
    owns_runtime = False
    if settings.backend == "vlm" and runtime is None:
        if progress is not None:
            progress(
                0,
                len(candidates),
                f"loading VLM runtime: {settings.vlm.runtime}",
            )
        if settings.vlm.runtime == "onnx":
            from .caption_onnx import SmolVlmOnnxCaptioner

            runtime = SmolVlmOnnxCaptioner(settings.vlm)
        elif settings.vlm.runtime == "gemini":
            runtime = GeminiVlmCaptioner(settings.vlm)
        elif settings.vlm.runtime == "openai_compatible":
            runtime = OpenAICompatibleVlmCaptioner(settings.vlm)
        else:
            runtime = LlamaCppVlmCaptioner(settings.vlm)
        owns_runtime = True
        if progress is not None:
            progress(
                0,
                len(candidates),
                f"VLM runtime ready: {settings.vlm.runtime}",
            )

    hard_failures = 0
    captioned_count = 0
    failed_count = 0
    try:
        resolved_trigger = resolve_trigger(
            subject=manifest.subject,
            trigger=manifest.trigger,
            trigger_from_subject=settings.trigger_from_subject,
        )
        if settings.backend != "vlm":
            trigger_tokens = estimate_caption_tokens(resolved_trigger)
            if resolved_trigger and trigger_tokens > settings.max_tokens:
                raise ValueError(
                    "caption.max_tokens is too small to preserve the resolved trigger"
                )

        total_candidates = len(candidates)
        for index, image in enumerate(candidates, start=1):
            caption = ""
            original_status = image.status
            if progress is not None:
                progress(
                    index - 1,
                    total_candidates,
                    f"captioning {index}/{total_candidates}: {image.path}",
                )
            if settings.backend == "vlm":
                source_image: Path | None = None
                source_kind = "raw"

                if use_processed_images and image.processed_path:
                    processed_candidate = (
                        root_path / str(image.processed_path)
                    ).resolve()
                    try:
                        processed_candidate.relative_to(root_path)
                    except ValueError:
                        processed_candidate = None
                    if processed_candidate is not None and processed_candidate.is_file():
                        source_image = processed_candidate
                        source_kind = "processed"

                if source_image is None:
                    raw_candidate = (root_path / image.path).resolve()
                    try:
                        raw_candidate.relative_to(root_path)
                    except ValueError:
                        raw_candidate = None
                    if raw_candidate is not None and raw_candidate.is_file():
                        source_image = raw_candidate
                        source_kind = "raw"

                if source_image is None:
                    failed_count += 1
                    image.metadata["caption_stage"] = "failed"
                    image.metadata["caption_vlm_error"] = (
                        "neither a usable processed image nor the raw source image was found"
                    )
                    if image.caption_path:
                        stale_caption = (root_path / image.caption_path).resolve()
                        try:
                            stale_caption.relative_to(root_path)
                        except ValueError:
                            stale_caption = None
                        if stale_caption is not None and stale_caption.is_file():
                            try:
                                stale_caption.unlink()
                            except OSError:
                                pass
                    image.caption_path = None
                    retry_status = (
                        original_status
                        if original_status in {"validated", "processed", "tagged"}
                        else ("tagged" if image.tags else "validated")
                    )
                    image.set_status(retry_status)
                    continue

                try:
                    assert runtime is not None
                    raw_caption = runtime.caption(source_image)
                    if settings.vlm.runtime in {"gemini", "openai_compatible"}:
                        try:
                            parse_vlm_fields(raw_caption)
                        except VlmCaptionerError as format_error:
                            image.metadata["caption_vlm_format_warning"] = str(format_error)[:500]
                            image.metadata["caption_vlm_raw_output"] = str(raw_caption)[:4000]
                        else:
                            image.metadata.pop("caption_vlm_format_warning", None)
                            image.metadata.pop("caption_vlm_raw_output", None)
                        caption = build_api_vlm_caption(
                            raw_caption,
                            subject=manifest.subject,
                            trigger=manifest.trigger,
                            config=settings,
                        )
                    else:
                        image.metadata.pop("caption_vlm_format_warning", None)
                        caption = build_vlm_caption(
                            raw_caption,
                            subject=manifest.subject,
                            trigger=manifest.trigger,
                            config=settings,
                        )
                    if not caption:
                        raise VlmCaptionerError(
                            f"VLM returned an empty caption for {image.path}"
                        )
                except Exception as exc:
                    failed_count += 1
                    image.metadata["caption_stage"] = "failed"
                    image.metadata["caption_vlm_error"] = str(exc)[:500]
                    image.metadata["caption_backend"] = "vlm"
                    if image.caption_path:
                        stale_caption = (root_path / image.caption_path).resolve()
                        try:
                            stale_caption.relative_to(root_path)
                        except ValueError:
                            stale_caption = None
                        if stale_caption is not None and stale_caption.is_file():
                            try:
                                stale_caption.unlink()
                            except OSError:
                                pass
                    image.caption_path = None
                    if source_kind == "processed":
                        retry_status = "processed"
                    elif original_status in {"validated", "processed", "tagged"}:
                        retry_status = original_status
                    else:
                        retry_status = "tagged" if image.tags else "validated"
                    image.set_status(retry_status)
                    if progress is not None:
                        progress(
                            index,
                            total_candidates,
                            f"caption failed {index}/{total_candidates}: {image.path} ({str(exc)[:240]})",
                        )
                    continue

            if settings.backend != "vlm":
                caption = build_caption(
                    image.tags,
                    subject=manifest.subject,
                    trigger=manifest.trigger,
                    config=settings,
                )

            try:
                target = caption_root / f"{image.image_id}.txt"
                write_caption(target, caption)
            except OSError:
                hard_failures += 1
                image.metadata["caption_stage"] = "failed"
                raise

            image.caption_path = manifest_path(root_path, target)
            image.metadata["caption_stage"] = "captioned"
            image.metadata["caption_backend"] = settings.backend
            image.metadata["caption_separator"] = settings.separator
            image.metadata["caption_subject"] = manifest.subject
            image.metadata["caption_trigger"] = resolved_trigger
            image.metadata["caption_trigger_from_subject"] = (
                settings.trigger_from_subject
            )
            image.metadata["caption_tag_count"] = (
                0 if settings.backend == "vlm" else len(image.tags)
            )
            if settings.backend == "vlm":
                image.metadata.pop("caption_token_budget", None)
                image.metadata["caption_vlm_model"] = (
                    settings.vlm.api_model
                    if settings.vlm.runtime in {"gemini", "openai_compatible"}
                    else (
                        str(Path(settings.vlm.model_path).expanduser())
                        if settings.vlm.model_path
                        else ""
                    )
                )
                image.metadata["caption_vlm_runtime"] = settings.vlm.runtime
                image.metadata["caption_vlm_image_source"] = source_kind
                image.metadata["caption_vlm_chat_format"] = (
                    settings.vlm.chat_format
                )
                if settings.vlm.runtime in {"gemini", "openai_compatible"}:
                    image.metadata["caption_vlm_api_base_url"] = (
                        settings.vlm.api_base_url
                    )
                else:
                    image.metadata.pop("caption_vlm_api_base_url", None)
                image.metadata.pop("caption_vlm_error", None)
                image.metadata.pop("caption_fallback_backend", None)
            else:
                image.metadata["caption_estimated_tokens"] = (
                    estimate_caption_tokens(caption)
                )
                image.metadata["caption_token_budget"] = settings.max_tokens
                for key in (
                    "caption_vlm_error",
                    "caption_fallback_backend",
                    "caption_vlm_model",
                    "caption_vlm_runtime",
                    "caption_vlm_chat_format",
                    "caption_vlm_image_source",
                ):
                    image.metadata.pop(key, None)

            image.metadata["caption_fallback"] = False
            image.set_status("captioned")
            captioned_count += 1
            if progress is not None:
                progress(
                    index,
                    total_candidates,
                    f"captioned {index}/{total_candidates}: {image.path}",
                )
    finally:
        if owns_runtime and runtime is not None:
            runtime.close()

    total_failures = failed_count + hard_failures
    skipped = sum(
        1
        for image in manifest.images.values()
        if image.status not in eligible_statuses and image.status != "captioned"
    )
    return CaptionSummary(
        scanned=len(candidates),
        captioned=captioned_count,
        failed=total_failures,
        skipped=skipped,
        fallback=0,
    )
