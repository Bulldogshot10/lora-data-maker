from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from .base import Scraper, ScraperError
from .chromium import ChromiumImagesScraper
from .direct import DirectURLScraper
from .duckduckgo import DuckDuckGoImagesScraper
from .google import DEFAULT_USER_AGENT, GoogleImagesScraper


IMAGE_SEARCH_BACKENDS: tuple[str, ...] = (
    "google",
    "duckduckgo",
    "chromium",
)
SCRAPER_BACKENDS: tuple[str, ...] = IMAGE_SEARCH_BACKENDS + ("direct",)


def validate_backend_chain(
    backend: str,
    fallbacks: Iterable[str] = (),
) -> tuple[str, ...]:
    primary = str(backend).strip().lower()
    if primary not in SCRAPER_BACKENDS:
        raise ScraperError(
            f"unknown scraper backend {primary!r}; "
            f"expected {', '.join(SCRAPER_BACKENDS)}"
        )

    if primary == "direct" and fallbacks:
        raise ScraperError(
            "direct cannot have fallback search engines because its query "
            "is a URL-list file"
        )

    chain = [primary]
    seen = {primary}

    for raw in fallbacks:
        name = str(raw).strip().lower()
        if not name:
            continue
        if name not in IMAGE_SEARCH_BACKENDS:
            raise ScraperError(
                f"invalid fallback backend {name!r}; "
                f"fallbacks must be image-search backends: "
                f"{', '.join(IMAGE_SEARCH_BACKENDS)}"
            )
        if name in seen:
            raise ScraperError(
                f"duplicate scraper backend in fallback chain: {name!r}"
            )
        if name == "google" and "chromium" in seen:
            raise ScraperError(
                "google cannot be used with chromium in the same fallback chain "
                "because chromium already uses Google Images"
            )
        if name == "chromium" and "google" in seen:
            raise ScraperError(
                "chromium cannot be used with google in the same fallback chain "
                "because chromium already uses Google Images"
            )
        seen.add(name)
        chain.append(name)

    return tuple(chain)


def create_scraper(
    backend: str,
    *,
    timeout: float,
    retries: int,
    delay: float,
    user_agent: str,
    profile_root: Path,
) -> Scraper:
    if backend == "google":
        return GoogleImagesScraper(
            timeout=timeout,
            retries=retries,
            delay=delay,
            user_agent=user_agent,
        )
    if backend == "duckduckgo":
        return DuckDuckGoImagesScraper(
            timeout=timeout,
            retries=retries,
            delay=delay,
            user_agent=user_agent,
        )
    if backend == "chromium":
        return ChromiumImagesScraper(
            timeout=timeout,
            retries=retries,
            delay=delay,
            user_agent=user_agent,
            profile_root=profile_root,
        )
    if backend == "direct":
        return DirectURLScraper()
    raise ScraperError(f"unknown scraper backend {backend!r}")
