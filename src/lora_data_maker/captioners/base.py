from __future__ import annotations

from pathlib import Path
from typing import Protocol


class Captioner(Protocol):
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
        ...
