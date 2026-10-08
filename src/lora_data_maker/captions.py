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

    return settings.separator.join(normalized_tags)


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


def build_vlm_caption(
    raw_caption: str,
    *,
    subject: str,
    trigger: str,
    config: CaptionConfig,
) -> str:
    """Clean VLM text and inject the canonical trigger exactly once."""
    value = clean_vlm_caption(raw_caption)
    resolved_trigger = resolve_trigger(
        subject=subject,
        trigger=trigger,
        trigger_from_subject=config.trigger_from_subject,
    )

    if resolved_trigger:
        pattern = re.compile(
            rf"\b{re.escape(resolved_trigger)}\b",
            re.IGNORECASE,
        )
        value = pattern.sub("", value).strip()
        value = re.sub(r"\s*([,.;:])\s*", r"\1 ", value)
        value = re.sub(r"\s+", " ", value).strip(" ,.;:")

    if not resolved_trigger:
        return value
    if not value:
        return resolved_trigger
    return config.separator.join((resolved_trigger, value))


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
    """Write deterministic or local-VLM captions for tagged/captioned records."""
    root_path = Path(root).expanduser().resolve()
    caption_root = Path(output_dir).expanduser().resolve()
    settings = config or CaptionConfig()

    candidates = sorted(
        (
            image
            for image in manifest.images.values()
            if image.status in {"tagged", "captioned"}
        ),
        key=lambda image: image.path,
    )

    runtime = captioner
    owns_runtime = False
    if settings.backend == "vlm" and runtime is None:
        runtime = LlamaCppVlmCaptioner(settings.vlm)
        owns_runtime = True

    fallback_count = 0
    hard_failures = 0
    try:
        for image in candidates:
            deterministic_caption = build_caption(
                image.tags,
                subject=manifest.subject,
                trigger=manifest.trigger,
                config=settings,
            )
            caption = deterministic_caption
            used_fallback = False

            if settings.backend == "vlm":
                processed_path = (
                    root_path / str(image.processed_path)
                    if image.processed_path
                    else None
                )
                if processed_path is None:
                    used_fallback = True
                    image.metadata["caption_vlm_error"] = (
                        "processed image path is missing"
                    )
                else:
                    try:
                        assert runtime is not None
                        raw_caption = runtime.caption(processed_path)
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
                        used_fallback = True
                        image.metadata["caption_vlm_error"] = str(exc)[:500]

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
            image.metadata["caption_trigger"] = resolve_trigger(
                subject=manifest.subject,
                trigger=manifest.trigger,
                trigger_from_subject=settings.trigger_from_subject,
            )
            image.metadata["caption_trigger_from_subject"] = (
                settings.trigger_from_subject
            )
            image.metadata["caption_tag_count"] = len(image.tags)
            image.metadata["caption_fallback"] = used_fallback
            if used_fallback:
                fallback_count += 1
                image.metadata["caption_fallback_backend"] = "deterministic"
            else:
                image.metadata.pop("caption_vlm_error", None)
                image.metadata.pop("caption_fallback_backend", None)

            if settings.backend == "vlm":
                image.metadata["caption_vlm_model"] = (
                    str(Path(settings.vlm.model_path).expanduser())
                    if settings.vlm.model_path
                    else ""
                )
                image.metadata["caption_vlm_chat_format"] = (
                    settings.vlm.chat_format
                )
            else:
                for key in (
                    "caption_vlm_error",
                    "caption_fallback_backend",
                    "caption_vlm_model",
                    "caption_vlm_chat_format",
                ):
                    image.metadata.pop(key, None)

            image.set_status("captioned")
    finally:
        if owns_runtime and runtime is not None:
            runtime.close()

    return CaptionSummary(
        scanned=len(candidates),
        captioned=len(candidates) - hard_failures,
        failed=hard_failures,
        skipped=sum(
            1
            for image in manifest.images.values()
            if image.status not in {"tagged", "captioned"}
        ),
        fallback=fallback_count,
    )
