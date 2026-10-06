from __future__ import annotations

from pathlib import Path
from typing import Iterator

from .base import ImageResult, ScraperError


class DirectURLScraper:
    """Yield image URLs from a plain text file."""

    def search(self, query: str, *, limit: int) -> Iterator[ImageResult]:
        path = Path(query).expanduser().resolve()
        if not path.is_file():
            raise ScraperError(f"URL list not found: {path}")

        yielded = 0
        seen: set[str] = set()
        for raw in path.read_text(encoding="utf-8").splitlines():
            url = raw.strip()
            if not url or url.startswith("#") or url in seen:
                continue
            if not url.startswith(("http://", "https://")):
                raise ScraperError(f"invalid URL in {path}: {url}")
            seen.add(url)
            yielded += 1
            yield ImageResult(image_url=url, rank=yielded)
            if yielded >= limit:
                break
