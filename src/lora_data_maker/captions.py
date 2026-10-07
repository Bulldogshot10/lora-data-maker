from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable

from .paths import manifest_path
from .taggers.filters import normalize_tag

if TYPE_CHECKING:
    from .captioners.base import Captioner
    from .models import ProjectManifest


@dataclass(frozen=True)
class CaptionSummary:
    scanned: int = 0
    captioned: int = 0
    failed: int = 0
    skipped: int = 0


@dataclass(frozen=True)
class CaptionConfig:
    separator: str = ", "
    subject_position: str = "first"
    trigger_from_subject: bool = True

    def __post_init__(self) -> None:
        if self.subject_position != "first":
            raise ValueError(
                "caption.subject_position must be 'first'"
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


def build_captioner(config: dict[str, Any]) -> "Captioner | None":
    """Build the configured non-deterministic captioner backend."""
    backend = str(config.get("backend", "deterministic")).strip().lower()

    if backend == "deterministic":
        return None

    if backend == "llama_cpp":
        llama_config = config.get("llama_cpp", {})
        if not isinstance(llama_config, dict):
            raise ValueError("config section [caption.llama_cpp] must be a table")

        from .captioners.llama_cpp import LlamaCppCaptioner

        seed_value = llama_config.get("seed")
        return LlamaCppCaptioner(
            base_url=str(
                llama_config.get(
                    "base_url",
                    "http://127.0.0.1:8080/v1",
                )
            ),
            model=str(llama_config.get("model", "local-model")),
            timeout=float(llama_config.get("timeout", 120.0)),
            temperature=float(llama_config.get("temperature", 0.2)),
            max_tokens=int(llama_config.get("max_tokens", 128)),
            system_prompt=llama_config.get("system_prompt"),
            seed=int(seed_value) if seed_value is not None else None,
        )

    raise ValueError(
        f"unsupported caption backend {backend!r}; "
        "expected 'deterministic' or 'llama_cpp'"
    )


def caption_manifest(
    manifest: "ProjectManifest",
    *,
    root: Path | str,
    output_dir: Path | str,
    config: CaptionConfig | None = None,
    captioner: "Captioner | None" = None,
) -> CaptionSummary:
    """Write deterministic captions for tagged/captioned records."""
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

    for image in candidates:
        if captioner is None:
            caption = build_caption(
                image.tags,
                subject=manifest.subject,
                trigger=manifest.trigger,
                config=settings,
            )
        else:
            source_path = root_path / (
                image.processed_path or image.path
            )
            caption = captioner.caption_image(
                source_path,
                tags=list(image.tags),
                subject=manifest.subject,
                trigger=manifest.trigger,
                separator=settings.separator,
                trigger_from_subject=settings.trigger_from_subject,
            )

        target = caption_root / f"{image.image_id}.txt"
        write_caption(target, caption)

        image.caption_path = manifest_path(root_path, target)
        image.metadata["caption_stage"] = "captioned"
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
        image.set_status("captioned")

    skipped = sum(
        1
        for image in manifest.images.values()
        if image.status not in {"tagged", "captioned"}
    )

    return CaptionSummary(
        scanned=len(candidates),
        captioned=len(candidates),
        failed=0,
        skipped=skipped,
    )
