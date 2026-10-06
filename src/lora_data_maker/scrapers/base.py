from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Protocol
from urllib.parse import urldefrag, urlsplit, urlunsplit


class ScraperError(Exception):
    """Raised for expected scraper failures."""


@dataclass(frozen=True)
class ImageResult:
    image_url: str
    query: str | None = None
    page_url: str | None = None
    rank: int | None = None
    metadata: dict[str, object] | None = None


class Scraper(Protocol):
    def search(self, query: str, *, limit: int) -> Iterator[ImageResult]:
        """Yield image results without downloading them."""


def canonicalize_url(value: str) -> str | None:
    """Normalize a candidate URL for cross-engine deduplication."""
    if not isinstance(value, str):
        return None

    stripped, _fragment = urldefrag(value.strip())
    if not stripped:
        return None

    try:
        parsed = urlsplit(stripped)
    except ValueError:
        return None

    if parsed.scheme.lower() not in {"http", "https"} or not parsed.netloc:
        return None

    return urlunsplit(
        (
            parsed.scheme.lower(),
            parsed.netloc.lower(),
            parsed.path,
            parsed.query,
            "",
        )
    )
