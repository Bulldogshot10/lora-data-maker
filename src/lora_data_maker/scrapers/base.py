from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Protocol


class ScraperError(Exception):
    """Raised for expected scraper failures."""


@dataclass(frozen=True)
class ImageResult:
    image_url: str
    query: str | None = None
    page_url: str | None = None
    rank: int | None = None
    metadata: dict[str, object] | None = None


SCRAPER_BACKENDS = (
    "google",
    "duckduckgo",
    "chromium",
    "playwright-bing",
    "playwright-google",
    "direct",
)


class Scraper(Protocol):
    def search(self, query: str, *, limit: int) -> Iterator[ImageResult]:
        """Yield image results without downloading them."""
