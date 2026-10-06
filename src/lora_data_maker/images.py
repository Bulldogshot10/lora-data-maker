from __future__ import annotations

from pathlib import Path
from typing import Iterator

from .models import ImageRecord


IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".gif",
    ".bmp",
    ".tif",
    ".tiff",
    ".avif",
}


def iter_image_files(root: Path | str) -> Iterator[Path]:
    root_path = Path(root).expanduser().resolve()
    if not root_path.exists():
        return

    for path in sorted(root_path.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            yield path


def manifest_paths(records: dict[str, ImageRecord]) -> set[str]:
    return {
        Path(record.path).expanduser().as_posix()
        for record in records.values()
        if record.path
    }
