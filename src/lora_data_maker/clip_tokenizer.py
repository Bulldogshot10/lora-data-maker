from __future__ import annotations

from functools import lru_cache
from typing import Any

CLIP_TOKENIZER_ID = "openai/clip-vit-large-patch14"
CLIP_CONTEXT_LENGTH = 77


class ClipTokenizerError(RuntimeError):
    """Raised when the exact SD1.x CLIP tokenizer is unavailable."""


@lru_cache(maxsize=1)
def load_clip_tokenizer() -> Any:
    """Load the exact Hugging Face CLIP tokenizer used by the SD1.x text encoder."""
    try:
        from transformers import CLIPTokenizer
    except ImportError as exc:
        raise ClipTokenizerError(
            "exact CLIP caption budgeting requires Transformers; install "
            "lora-data-maker[caption-vlm]"
        ) from exc

    try:
        tokenizer = CLIPTokenizer.from_pretrained(CLIP_TOKENIZER_ID)
    except Exception as exc:
        raise ClipTokenizerError(
            f"could not load CLIP tokenizer {CLIP_TOKENIZER_ID}: {exc}"
        ) from exc

    model_max_length = int(getattr(tokenizer, "model_max_length", CLIP_CONTEXT_LENGTH))
    if model_max_length != CLIP_CONTEXT_LENGTH:
        raise ClipTokenizerError(
            f"{CLIP_TOKENIZER_ID} reported an unexpected context length "
            f"{model_max_length}; expected {CLIP_CONTEXT_LENGTH}"
        )

    return tokenizer


def count_clip_tokens(text: str, *, tokenizer: Any | None = None) -> int:
    """Count BOS/EOS-inclusive CLIP tokens without tokenizer-side truncation."""
    value = str(text).strip()
    if not value:
        return 0

    active = tokenizer if tokenizer is not None else load_clip_tokenizer()
    try:
        encoded = active(
            value,
            add_special_tokens=True,
            truncation=False,
            padding=False,
        )
        input_ids = encoded["input_ids"]
    except Exception as exc:
        raise ClipTokenizerError(
            f"could not tokenize caption text: {exc}"
        ) from exc

    if input_ids and isinstance(input_ids[0], list):
        input_ids = input_ids[0]

    return len(input_ids)


def pack_clip_caption(
    parts: list[str],
    max_tokens: int,
    *,
    separator: str = ", ",
    tokenizer: Any | None = None,
) -> str:
    """Pack ordered caption fields while keeping the exact CLIP budget."""
    if not 1 <= max_tokens <= CLIP_CONTEXT_LENGTH:
        raise ValueError(
            f"caption max tokens must be between 1 and {CLIP_CONTEXT_LENGTH}"
        )
    if not isinstance(separator, str):
        raise ValueError("caption separator must be a string")

    normalized = [str(part).strip(" ,.;:") for part in parts if str(part).strip()]
    if not normalized:
        return ""

    selected: list[str] = []
    for part in normalized:
        candidate = separator.join((*selected, part))
        if count_clip_tokens(candidate, tokenizer=tokenizer) <= max_tokens:
            selected.append(part)
            continue

        # Keep the ordering intact and use any remaining room in the current field.
        words = part.split()
        partial: list[str] = []
        for word in words:
            candidate_part = " ".join((*partial, word))
            candidate = separator.join((*selected, candidate_part))
            if count_clip_tokens(candidate, tokenizer=tokenizer) <= max_tokens:
                partial.append(word)
            else:
                break

        if partial:
            selected.append(" ".join(partial))
        break

    return separator.join(selected).strip(" ,.;:")
