from __future__ import annotations

import os
import re
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Iterable

from .caption_vlm import (
    DEFAULT_VLM_PROMPT,
    Captioner,
    LlamaCppVlmCaptioner,
    VlmConfig,
    VlmCaptionerError,
)
from .clip_tokenizer import (
    ClipTokenizerError,
    count_clip_tokens,
    load_clip_tokenizer,
    pack_clip_caption,
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
            if not self.vlm.model_path.strip() or not self.vlm.mmproj_path.strip():
                raise ValueError(
                    "caption.vlm.model_path and caption.vlm.mmproj_path "
                    "are required for VLM captions"
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

    model_path = string_field("model_path", "")
    mmproj_path = string_field("mmproj_path", "")
    if root is not None:
        root_path = Path(root).expanduser().resolve()
        if model_path and not Path(model_path).expanduser().is_absolute():
            model_path = str((root_path / model_path).resolve())
        if mmproj_path and not Path(mmproj_path).expanduser().is_absolute():
            mmproj_path = str((root_path / mmproj_path).resolve())

    vlm = VlmConfig(
        model_path=model_path,
        mmproj_path=mmproj_path,
        chat_format=string_field("chat_format", "mtmd"),
        prompt=string_field("prompt", DEFAULT_VLM_PROMPT),
        max_tokens=int_field("max_tokens", 128),
        temperature=float_field("temperature", 0.1),
        top_p=float_field("top_p", 0.95),
        top_k=int_field("top_k", 40),
        n_ctx=int_field("n_ctx", 4096),
        n_threads=int_field("n_threads", 4),
        n_gpu_layers=int_field("n_gpu_layers", 0),
        seed=int_field("seed", 42),
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


_VLM_FIELDS = ("pose", "clothing", "expression", "background")
_VLM_FIELD_RE = re.compile(
    r"^(pose|clothing|expression|background)\s*:\s*(.*?)\s*$",
    re.IGNORECASE,
)


def parse_vlm_fields(raw_caption: str) -> list[str]:
    """Parse the strict four-field VLM response into ordered caption fields."""
    value = str(raw_caption).replace("\r\n", "\n").replace("\r", "\n").strip()
    lines = [line.strip() for line in value.split("\n") if line.strip()]
    if not lines:
        raise VlmCaptionerError("VLM caption is empty")

    structured = any(
        re.match(
            r"^(?:pose|clothing|expression|background)\s*:",
            line,
            re.IGNORECASE,
        )
        for line in lines
    )
    if not structured:
        cleaned = clean_vlm_caption(value)
        return [cleaned] if cleaned else []

    if len(lines) != len(_VLM_FIELDS):
        raise VlmCaptionerError(
            "VLM caption must contain exactly four non-empty fields"
        )

    fields: list[str] = []
    for expected, line in zip(_VLM_FIELDS, lines, strict=True):
        match = _VLM_FIELD_RE.fullmatch(line)
        if match is None or match.group(1).lower() != expected:
            raise VlmCaptionerError(
                "VLM caption fields must be pose, clothing, expression, background in order"
            )
        phrase = clean_vlm_caption(match.group(2))
        if not phrase:
            raise VlmCaptionerError(f"VLM field {expected} is empty")
        if len(re.findall(r"\S+", phrase)) > 5:
            raise VlmCaptionerError(
                f"VLM field {expected} exceeds the 5-word limit"
            )
        fields.append(phrase)

    return fields


def parse_vlm_caption(raw_caption: str) -> str:
    """Parse the VLM response into one ordered caption fragment."""
    return ", ".join(parse_vlm_fields(raw_caption))


def build_vlm_caption(
    raw_caption: str,
    *,
    subject: str,
    trigger: str,
    config: CaptionConfig,
    tokenizer: object | None = None,
) -> str:
    """Build a VLM-only caption bounded by the exact SD1.x CLIP tokenizer."""
    fields = parse_vlm_fields(raw_caption)
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
            value = re.sub(r"\s+", " ", value).strip(" ,.;:")
        if value:
            cleaned_fields.append(value)

    parts: list[str] = []
    if resolved_trigger:
        parts.append(resolved_trigger)
    parts.extend(cleaned_fields)

    return pack_clip_caption(
        parts,
        config.max_tokens,
        separator=config.separator,
        tokenizer=tokenizer,
    )

def estimate_caption_tokens(text: str) -> int:
    """Estimate SD1.5 caption token usage without adding a tokenizer dependency."""
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
) -> CaptionSummary:
    """Write deterministic or local-VLM captions for eligible records."""
    root_path = Path(root).expanduser().resolve()
    caption_root = Path(output_dir).expanduser().resolve()
    settings = config or CaptionConfig()

    eligible_statuses = (
        {"processed", "captioned"}
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
    clip_tokenizer = None
    if settings.backend == "vlm":
        try:
            clip_tokenizer = load_clip_tokenizer()
        except ClipTokenizerError:
            raise
        if runtime is None:
            runtime = LlamaCppVlmCaptioner(settings.vlm)
            owns_runtime = True

    hard_failures = 0
    captioned_count = 0
    failed_count = 0
    try:
        resolved_trigger = resolve_trigger(
            subject=manifest.subject,
            trigger=manifest.trigger,
            trigger_from_subject=settings.trigger_from_subject,
        )
        if settings.backend == "vlm":
            trigger_tokens = count_clip_tokens(
                resolved_trigger,
                tokenizer=clip_tokenizer,
            ) if resolved_trigger else 0
        else:
            trigger_tokens = estimate_caption_tokens(resolved_trigger)

        if resolved_trigger and trigger_tokens > settings.max_tokens:
            raise ValueError(
                "caption.max_tokens is too small to preserve the resolved trigger"
            )

        for image in candidates:
            caption = ""
            if settings.backend == "vlm":
                processed_path = (
                    root_path / str(image.processed_path)
                    if image.processed_path
                    else None
                )
                if processed_path is None or not processed_path.is_file():
                    failed_count += 1
                    image.metadata["caption_stage"] = "failed"
                    image.metadata["caption_vlm_error"] = (
                        "processed image was not found"
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
                    image.set_status("processed")
                    continue

                try:
                    assert runtime is not None
                    raw_caption = runtime.caption(processed_path)
                    caption = build_vlm_caption(
                        raw_caption,
                        subject=manifest.subject,
                        trigger=manifest.trigger,
                        config=settings,
                        tokenizer=clip_tokenizer,
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
                    image.metadata["caption_tokenizer"] = (
                        "openai/clip-vit-large-patch14"
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
                    image.set_status("processed")
                    continue

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
            image.metadata["caption_token_budget"] = settings.max_tokens
            if settings.backend == "vlm":
                image.metadata["caption_clip_tokens"] = count_clip_tokens(
                    caption,
                    tokenizer=clip_tokenizer,
                )
                image.metadata["caption_tokenizer"] = (
                    "openai/clip-vit-large-patch14"
                )
                image.metadata["caption_vlm_model"] = (
                    str(Path(settings.vlm.model_path).expanduser())
                    if settings.vlm.model_path
                    else ""
                )
                image.metadata["caption_vlm_chat_format"] = (
                    settings.vlm.chat_format
                )
                image.metadata.pop("caption_vlm_error", None)
                image.metadata.pop("caption_fallback_backend", None)
            else:
                caption = build_caption(
                    image.tags,
                    subject=manifest.subject,
                    trigger=manifest.trigger,
                    config=settings,
                )
                image.metadata["caption_estimated_tokens"] = (
                    estimate_caption_tokens(caption)
                )
                for key in (
                    "caption_vlm_error",
                    "caption_fallback_backend",
                    "caption_vlm_model",
                    "caption_vlm_chat_format",
                    "caption_clip_tokens",
                    "caption_tokenizer",
                ):
                    image.metadata.pop(key, None)

            image.metadata["caption_fallback"] = False
            image.set_status("captioned")
            captioned_count += 1
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
