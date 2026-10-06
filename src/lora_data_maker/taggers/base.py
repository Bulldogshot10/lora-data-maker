from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol


class TaggerError(RuntimeError):
    """Raised when a tagger cannot load or infer."""


@dataclass(frozen=True)
class TagResult:
    tags: list[str]
    scores: dict[str, float] = field(default_factory=dict)
    ratings: dict[str, float] = field(default_factory=dict)
    characters: dict[str, float] = field(default_factory=dict)


class Tagger(Protocol):
    def tag_image(self, path: Path | str) -> TagResult:
        ...

    def tag_images(self, paths: list[Path | str]) -> list[TagResult]:
        ...
