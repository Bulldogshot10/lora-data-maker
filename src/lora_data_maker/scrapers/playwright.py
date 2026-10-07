from __future__ import annotations

import html
import re
import time
from typing import Iterator
from urllib.parse import urlencode, urlparse

from .base import ImageResult, Scraper, ScraperError


class PlaywrightImagesScraper(Scraper):
    """Image-search scraper driven by Termux-native Chromium through Playwright.

    The Termux-specific Playwright implementation controls the native Chromium
    package through CDP, so no glibc Playwright browser binary is downloaded.
    """

    PROVIDERS = ("bing", "google")
    BING_URL = "https://www.bing.com/images/search"
    GOOGLE_URL = "https://www.google.com/search"

    def __init__(
        self,
        *,
        provider: str = "bing",
        timeout: float = 20.0,
        retries: int = 2,
        delay: float = 1.0,
        max_scrolls: int = 30,
    ) -> None:
        if provider not in self.PROVIDERS:
            raise ValueError(
                f"unknown Playwright provider {provider!r}; "
                f"expected one of {', '.join(self.PROVIDERS)}"
            )
        if timeout <= 0:
            raise ValueError("timeout must be greater than zero")
        if retries < 0:
            raise ValueError("retries cannot be negative")
        if delay < 0:
            raise ValueError("delay cannot be negative")
        if max_scrolls <= 0:
            raise ValueError("max_scrolls must be greater than zero")

        self.provider = provider
        self.timeout = timeout
        self.retries = retries
        self.delay = delay
        self.max_scrolls = max_scrolls

    def search(self, query: str, *, limit: int) -> Iterator[ImageResult]:
        if not query.strip():
            raise ScraperError("Playwright image-search query cannot be empty")
        if limit <= 0:
            return

        last_error: Exception | None = None
        for attempt in range(self.retries + 1):
            try:
                results = self._search_once(query, limit)
                if results:
                    for result in results:
                        yield result
                    return
                raise ScraperError(
                    f"Playwright {self.provider} image search returned no "
                    "parseable original image URLs"
                )
            except ScraperError as exc:
                last_error = exc
                if attempt >= self.retries:
                    break
                print(
                    f"scrape: playwright provider={self.provider} "
                    f"retry={attempt + 1}/{self.retries}: {exc}",
                    flush=True,
                )
                time.sleep(min(max(self.delay, 0.25) * (2**attempt), 15.0))

        raise ScraperError(
            f"Playwright {self.provider} search failed after "
            f"{self.retries + 1} attempts: {last_error}"
        ) from last_error

    def _search_once(self, query: str, limit: int) -> list[ImageResult]:
        try:
            from playwright_termux import Chromium
        except ImportError as exc:
            raise ScraperError(
                "Playwright backend requires playwright-termux; "
                'install with python -m pip install ".[playwright]"'
            ) from exc

        search_url = self._build_url(query)
        print(
            f"scrape: playwright provider={self.provider} query={query}",
            flush=True,
        )

        try:
            with Chromium(headless=True) as browser:
                page = browser.new_page()
                try:
                    self._goto(page, search_url)

                    seen: set[str] = set()
                    results: list[ImageResult] = []
                    stagnant_rounds = 0
                    previous_count = 0

                    for scroll_no in range(1, self.max_scrolls + 1):
                        page_html = page.content()
                        if self.provider == "bing":
                            candidates = self._extract_bing_urls(page_html)
                        else:
                            candidates = self._extract_google_urls(page_html)

                        for image_url in candidates:
                            if image_url in seen:
                                continue
                            seen.add(image_url)
                            results.append(
                                ImageResult(
                                    image_url=image_url,
                                    query=query,
                                    page_url=search_url,
                                    rank=len(results) + 1,
                                    metadata={
                                        "provider": self.provider,
                                        "playwright": True,
                                        "scroll": scroll_no,
                                    },
                                )
                            )
                            if len(results) >= limit:
                                return results

                        print(
                            f"scrape: playwright provider={self.provider} "
                            f"scroll={scroll_no}/{self.max_scrolls} "
                            f"candidates={len(results)}",
                            flush=True,
                        )

                        if len(results) == previous_count:
                            stagnant_rounds += 1
                        else:
                            stagnant_rounds = 0
                        previous_count = len(results)

                        if stagnant_rounds >= 4:
                            break

                        page.evaluate(
                            "window.scrollTo(0, document.body.scrollHeight)"
                        )
                        if self.delay:
                            page.wait_for_timeout(
                                int(max(self.delay, 0.5) * 1000)
                            )

                    return results
                finally:
                    try:
                        page.close()
                    except Exception:
                        pass
        except ScraperError:
            raise
        except Exception as exc:
            raise ScraperError(
                f"Playwright {self.provider} browser error: {exc}"
            ) from exc

    def _goto(self, page, url: str) -> None:
        """Navigate without relying on playwright-termux Page.goto()."""
        timeout = max(float(self.timeout), 1.0)
        navigation_timeout = min(timeout, 5.0)
        initial_url = getattr(page, "url", "about:blank") or "about:blank"
        try:
            navigation = page.session.send(
                "Page.navigate",
                {"url": url},
                timeout=navigation_timeout,
            )
        except Exception as exc:
            raise ScraperError(
                f"Playwright {self.provider} navigation failed: {exc}"
            ) from exc

        if navigation.get("errorText"):
            raise ScraperError(
                f"Playwright {self.provider} navigation failed: "
                f"{navigation['errorText']}"
            )

        deadline = time.monotonic() + timeout
        last_error: Exception | None = None

        while time.monotonic() < deadline:
            try:
                result = page.session.send(
                    "Runtime.evaluate",
                    {
                        "expression": (
                            "[document.readyState, location.href]"
                        ),
                        "returnByValue": True,
                    },
                    timeout=min(1.0, max(deadline - time.monotonic(), 0.1)),
                )
                value = result.get("result", {}).get("value")
                if (
                    isinstance(value, list)
                    and len(value) == 2
                    and value[0] in {"interactive", "complete"}
                    and str(value[1]) not in {"", initial_url, "about:blank"}
                ):
                    page._url = str(value[1])
                    return
            except Exception as exc:
                last_error = exc

            time.sleep(0.05)

        raise ScraperError(
            f"Playwright {self.provider} navigation timed out after "
            f"{timeout:g}s"
            + (f": {last_error}" if last_error else "")
        )

    def _build_url(self, query: str) -> str:
        if self.provider == "bing":
            return f"{self.BING_URL}?{urlencode({'q': query, 'form': 'HDRSC2'})}"
        return (
            f"{self.GOOGLE_URL}?"
            f"{urlencode({'q': query, 'tbm': 'isch', 'hl': 'en'})}"
        )

    @staticmethod
    def _extract_bing_urls(page_html: str) -> list[str]:
        # Bing image results expose original URLs in the JSON stored in murl.
        tag_pattern = re.compile(
            r"""<a\b[^>]*\bclass=["'][^"']*\biusc\b[^"']*["'][^>]*>""",
            re.IGNORECASE | re.DOTALL,
        )
        murl_pattern = re.compile(
            r"""\bm=(["'])(.*?)\1""",
            re.IGNORECASE | re.DOTALL,
        )
        urls: list[str] = []
        seen: set[str] = set()

        for tag in tag_pattern.findall(page_html):
            match = murl_pattern.search(tag)
            if match is None:
                continue
            payload = html.unescape(match.group(2))
            try:
                import json

                data = json.loads(payload)
            except (TypeError, ValueError):
                continue

            image_url = data.get("murl") if isinstance(data, dict) else None
            if not isinstance(image_url, str):
                continue
            if not image_url.startswith(("http://", "https://")):
                continue
            host = (urlparse(image_url).hostname or "").lower().rstrip(".")
            if (
                host == "bing.com"
                or host.endswith(".bing.com")
                or host == "bing.net"
                or host.endswith(".bing.net")
            ):
                continue
            if image_url in seen:
                continue
            seen.add(image_url)
            urls.append(image_url)

        return urls

    @staticmethod
    def _extract_google_urls(page_html: str) -> list[str]:
        lowered = page_html.lower()
        if any(
            marker in lowered
            for marker in (
                "our systems have detected unusual traffic",
                "detected unusual traffic from your computer network",
                "before you continue to google",
                "/sorry/",
            )
        ):
            raise ScraperError(
                "Google returned a consent/verification page; "
                "the scraper does not bypass verification challenges"
            )

        urls: list[str] = []
        seen: set[str] = set()

        # Google image results commonly embed the original URL in an "ou"
        # field. Prefer it so thumbnails are never sent to the downloader.
        pattern = re.compile(
            r"""["']ou["']\s*:\s*["']((?:\\.|[^"'])+)["']"""
        )
        for raw in pattern.findall(page_html):
            value = PlaywrightImagesScraper._decode_json_string(raw)
            if PlaywrightImagesScraper._valid_image_candidate(value, seen):
                seen.add(value)
                urls.append(value)

        if urls:
            return urls

        # Fallback for markup variants. Reuse the legacy parser's layered
        # extraction rather than duplicating its URL heuristics.
        try:
            from .google import GoogleImagesScraper

            for value in GoogleImagesScraper._extract_urls(page_html):
                if value not in seen:
                    seen.add(value)
                    urls.append(value)
        except Exception:
            pass

        return urls

    @staticmethod
    def _decode_json_string(value: str) -> str:
        import json

        try:
            return json.loads(f'"{value}"')
        except (TypeError, ValueError):
            return html.unescape(value).replace(r"\/", "/")

    @staticmethod
    def _valid_image_candidate(value: str, seen: set[str]) -> bool:
        if not value.startswith(("http://", "https://")):
            return False
        host = (urlparse(value).hostname or "").lower().rstrip(".")
        if (
            host == "google.com"
            or host.endswith(".google.com")
            or host == "gstatic.com"
            or host.endswith(".gstatic.com")
        ):
            return False
        if value in seen:
            return False
        return True
