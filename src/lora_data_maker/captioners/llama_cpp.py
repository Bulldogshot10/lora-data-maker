from __future__ import annotations

import re
from pathlib import Path

from ..captions import resolve_trigger
from ..llama_cpp import LlamaCppError, LlamaCppRuntime
from .base import Captioner


DEFAULT_SYSTEM_PROMPT = (
    "You write concise image captions for LoRA training. "
    "Describe only what is visibly supported by the image. "
    "Use natural comma-separated visual phrases and no explanation."
)


class LlamaCppCaptioner(Captioner):
    """Vision captioner backed by an embedded llama.cpp runtime."""

    def __init__(
        self,
        *,
        runtime: LlamaCppRuntime,
        temperature: float = 0.2,
        max_tokens: int = 128,
        system_prompt: str | None = None,
        seed: int | None = None,
    ) -> None:
        self.runtime = runtime
        self.temperature = float(temperature)
        self.max_tokens = int(max_tokens)
        self.system_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT
        self.seed = seed

    @property
    def model(self) -> str:
        return str(self.runtime.model_path)

    def caption_image(
        self,
        path: Path | str,
        *,
        tags: list[str],
        subject: str,
        trigger: str,
        separator: str,
        trigger_from_subject: bool,
    ) -> str:
        resolved_trigger = resolve_trigger(
            subject=subject,
            trigger=trigger,
            trigger_from_subject=trigger_from_subject,
        )
        tag_hint = separator.join(tags) if tags else "(none)"

        prompt = (
            "Caption this image for a LoRA training dataset.\n"
            f"Dataset subject: {subject.strip() or '(none)'}\n"
            f"Required trigger: {resolved_trigger or '(none)'}\n"
            f"Existing tag hints: {tag_hint}\n\n"
            "Return one concise comma-separated caption only. "
            "Use the image as the source of truth. "
            "Keep useful appearance, pose, clothing, setting, and composition details. "
            "Do not add explanations, markdown, or a trailing period."
        )

        try:
            generated = self.runtime.chat_with_image(
                path,
                prompt,
                system_prompt=self.system_prompt,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                seed=self.seed,
            )
        except LlamaCppError:
            raise

        caption = _sanitize_caption(generated)
        if not caption:
            raise LlamaCppError("llama.cpp captioner returned an empty caption")

        return _ensure_trigger_prefix(
            caption,
            trigger=resolved_trigger,
            separator=separator,
        )


def _sanitize_caption(value: str) -> str:
    text = value.replace("\r", " ").replace("\n", " ")
    text = re.sub(r"\s+", " ", text).strip()
    text = text.strip("\"' ")
    return text.rstrip(". ").strip()


def _ensure_trigger_prefix(
    caption: str,
    *,
    trigger: str,
    separator: str,
) -> str:
    if not trigger:
        return caption

    normalized_trigger = trigger.strip()
    first_fragment = caption.split(",", 1)[0].strip()
    if first_fragment.lower() == normalized_trigger.lower():
        return caption

    return f"{normalized_trigger}{separator}{caption}"
