from __future__ import annotations

import html
import re
from html.parser import HTMLParser
from typing import Iterator
from urllib.parse import urlparse

import requests

from .base import ImageResult, Scraper, ScraperError


DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Linux; Android 14; Mobile) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/154.0.0.0 Mobile Safari/537.36"
)

_URL_RE = re.compile(r"https?://[^\s\"'<>\\]+", re.IGNORECASE)


class _ImageAttributeParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.urls: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() != "img":
            return
        for key, value in attrs:
            if key.lower() in {"src", "data-src", "data-iurl", "data-img-url", "data-image-url"}:
                if value:
                    self.urls.append(value)


class GoogleImagesScraper(Scraper):
    """Credential-free Google Images result scraper.

    It uses Google's image-search request endpoint and layered URL extraction.
    It does not attempt CAPTCHA, login, or verification bypasses.
    """

    SEARCH_URL = "https://www.google.com/search"

    def __init__(
        self,
        *,
        timeout: float = 20.0,
        retries: int = 3,
        delay: float = 1.0,
        user_agent: str = DEFAULT_USER_AGENT,
        session: requests.Session | None = None,
        language: str = "en",
        country: str | None = None,
        safe: str | None = None,
    ) -> None:
        self.timeout = timeout
        self.retries = retries
        self.delay = delay
        self.session = session or requests.Session()
        self.session.headers.update({
            "User-Agent": user_agent,
            "Accept": "text/html,application/xhtml+xml,application/json;q=0.9,*/*;q=0.8",
        })
        self.language = language
        self.country = country
        self.safe = safe

    def search(self, query: str, *, limit: int) -> Iterator[ImageResult]:
        if not query.strip():
            raise ScraperError("Google Images query cannot be empty")
        if limit <= 0:
            return

        seen: set[str] = set()
        yielded = 0
        page = 0
        max_pages = max(4, (limit + 49) // 50 * 2)

        while yielded < limit and page < max_pages:
            body = self._fetch_page(query, page)
            self._check_block_page(body)
            candidates = self._extract_urls(body)

            if not candidates:
                if page == 0:
                    raise ScraperError(
                        "Google Images returned no parseable image URLs"
                    )
                break

            for rank, image_url in enumerate(candidates, start=1):
                normalized = self._normalize_url(image_url)
                if not normalized or normalized in seen:
                    continue
                seen.add(normalized)
                yielded += 1
                yield ImageResult(
                    image_url=normalized,
                    query=query,
                    rank=yielded,
                    metadata={"page": page, "page_rank": rank},
                )
                if yielded >= limit:
                    break

            page += 1

    def _fetch_page(self, query: str, page: int) -> str:
        params = {
            "q": query,
            "tbm": "isch",
            "ijn": str(page),
            "hl": self.language,
            "async": "_fmt:json,p:1",
        }
        if self.country:
            params["gl"] = self.country
        if self.safe:
            params["safe"] = self.safe

        last_error: Exception | None = None
        for attempt in range(self.retries + 1):
            try:
                response = self.session.get(
                    self.SEARCH_URL,
                    params=params,
                    timeout=self.timeout,
                )
                response.raise_for_status()
                if self.delay:
                    import time
                    time.sleep(self.delay)
                return response.text
            except requests.RequestException as exc:
                last_error = exc
                if attempt >= self.retries:
                    break
                import time
                time.sleep(min(max(self.delay, 0.25) * (2**attempt), 30.0))

        raise ScraperError(
            f"Google Images request failed after {self.retries + 1} attempts: {last_error}"
        ) from last_error

    @classmethod
    def _extract_urls(cls, body: str) -> list[str]:
        urls: list[str] = []

        try:
            parser = _ImageAttributeParser()
            parser.feed(body)
            urls.extend(parser.urls)
        except Exception:
            pass

        urls.extend(_URL_RE.findall(body))

        decoded: list[str] = []
        for raw in urls:
            value = cls._decode_url(raw)
            if cls._looks_like_image_url(value):
                decoded.append(value)

        # Preserve first occurrence while removing duplicates.
        return list(dict.fromkeys(decoded))

    @staticmethod
    def _decode_url(value: str) -> str:
        text = html.unescape(value)
        text = text.replace("\/", "/")
        text = text.replace("\u003d", "=").replace("\u0026", "&")
        text = text.replace("\u002F", "/").replace("\u002f", "/")
        text = text.rstrip(".,);]}\"'")
        return text

    @staticmethod
    def _normalize_url(value: str) -> str | None:
        if not value.startswith(("http://", "https://")):
            return None
        parsed = urlparse(value)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            return None
        return value

    @staticmethod
    def _looks_like_image_url(value: str) -> bool:
        parsed = urlparse(value)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            return False

        host = parsed.netloc.lower()
        path = parsed.path.lower()

        blocked_hosts = {
            "www.google.com",
            "google.com",
            "www.google.co.uk",
            "consent.google.com",
            "accounts.google.com",
        }
        if host in blocked_hosts:
            return False

        if "/search" in path or "google.com/url" in value.lower():
            return False

        image_exts = (
            ".jpg", ".jpeg", ".png", ".webp", ".gif",
            ".bmp", ".tif", ".tiff", ".avif",
        )
        if path.endswith(image_exts):
            return True

        # Google result payloads frequently contain extensionless CDN/original
        # image URLs. Reject obvious navigation/document URLs and keep the rest.
        if any(token in host for token in ("gstatic.com", "googleusercontent.com")):
            return True

        return bool(path and "." in host)

    @staticmethod
    def _check_block_page(body: str) -> None:
        lowered = body.lower()
        markers = (
            "our systems have detected unusual traffic",
            "detected unusual traffic from your computer network",
            "before you continue to google",
            "consent.google.com",
            "/sorry/",
        )
        if any(marker in lowered for marker in markers):
            raise ScraperError(
                "Google returned a consent/verification page; "
                "the scraper does not bypass verification challenges"
            )
