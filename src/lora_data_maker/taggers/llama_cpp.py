from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Mapping

from ..llama_cpp import LlamaCppError, LlamaCppRuntime
from .base import TagResult, Tagger, TaggerError
from .filters import normalize_tag


DEFAULT_SYSTEM_PROMPT = (
    "You are a precise image tagger for LoRA dataset preparation. "
    "Describe visible, useful training concepts only. "
    "Do not write prose, explanations, or a caption."
)

DEFAULT_PROMPT = """Inspect this image and return JSON only.

Use this exact shape:
{"tags":["tag one","tag two"],"scores":{},"ratings":{},"characters":{}}

Rules:
- tags must be short, lowercase visual concepts.
- prefer concrete appearance, pose, clothing, setting, composition, and camera concepts.
- do not add the dataset subject/trigger; that is handled by a later pipeline stage.
- do not invent unseen details.
- scores, ratings, and characters may remain empty objects.
"""


class LlamaCppVLMTagger(Tagger):
    """Image tagger backed by an embedded llama.cpp multimodal runtime."""

    def __init__(
        self,
        *,
        runtime: LlamaCppRuntime,
        temperature: float = 0.0,
        max_tokens: int = 256,
        system_prompt: str | None = None,
        prompt: str | None = None,
        seed: int | None = None,
        max_tags: int = 64,
    ) -> None:
        if max_tags <= 0:
            raise ValueError("max_tags must be greater than zero")

        self.runtime = runtime
        self.temperature = float(temperature)
        self.max_tokens = int(max_tokens)
        self.system_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT
        self.prompt = prompt or DEFAULT_PROMPT
        self.seed = seed
        self.max_tags = int(max_tags)

    @property
    def model(self) -> str:
        return str(self.runtime.model_path)

    def tag_image(self, path: Path | str) -> TagResult:
        try:
            content = self.runtime.chat_with_image(
                path,
                self.prompt,
                system_prompt=self.system_prompt,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                response_format={"type": "json_object"},
                seed=self.seed,
            )
            return _parse_tag_result(content, max_tags=self.max_tags)
        except TaggerError:
            raise
        except LlamaCppError as exc:
            raise TaggerError(str(exc)) from exc
        except Exception as exc:
            raise TaggerError(
                f"llama.cpp tagger failed for {Path(path)}: {exc}"
            ) from exc

    def tag_images(self, paths: list[Path | str]) -> list[TagResult]:
        return [self.tag_image(path) for path in paths]


def _parse_tag_result(content: str, *, max_tags: int) -> TagResult:
    data = _load_json_object(content)

    raw_tags = data.get("tags", [])
    if isinstance(raw_tags, str):
        raw_tags = re.split(r"[,;\n]+", raw_tags)
    if not isinstance(raw_tags, list):
        raise TaggerError("llama.cpp tagger response field 'tags' must be a list")

    tags: list[str] = []
    seen: set[str] = set()
    for raw_tag in raw_tags:
        if not isinstance(raw_tag, str):
            continue
        tag = normalize_tag(raw_tag)
        if not tag or tag in seen:
            continue
        seen.add(tag)
        tags.append(tag)
        if len(tags) >= max_tags:
            break

    return TagResult(
        tags=tags,
        scores=_score_map(data.get("scores")),
        ratings=_score_map(data.get("ratings")),
        characters=_score_map(data.get("characters")),
    )


def _load_json_object(content: str) -> dict[str, Any]:
    candidate = content.strip()
    try:
        value = json.loads(candidate)
    except json.JSONDecodeError:
        start = candidate.find("{")
        end = candidate.rfind("}")
        if start < 0 or end <= start:
            raise TaggerError("llama.cpp tagger returned invalid JSON")
        try:
            value = json.loads(candidate[start : end + 1])
        except json.JSONDecodeError as exc:
            raise TaggerError(
                f"llama.cpp tagger returned invalid JSON: {exc}"
            ) from exc

    if not isinstance(value, dict):
        raise TaggerError("llama.cpp tagger response must be a JSON object")
    return value


def _score_map(value: object) -> dict[str, float]:
    if not isinstance(value, Mapping):
        return {}
    result: dict[str, float] = {}
    for key, score in value.items():
        if not isinstance(key, str):
            continue
        try:
            result[key] = float(score)
        except (TypeError, ValueError):
            continue
    return result
