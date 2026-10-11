from __future__ import annotations

import html
import os
import re
import time
from datetime import date
from html.parser import HTMLParser
from typing import Iterator
from urllib.parse import parse_qs, urlencode, urlparse

from .base import ImageResult, Scraper, ScraperError
from .existing_chromium import ExistingChromium



class _BingResultParser(HTMLParser):
    """Extract Bing image metadata only from the main image-results region."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.payloads: list[str] = []
        self._fallback_payloads: list[str] = []
        self._element_stack: list[tuple[str, bool]] = []
        self._result_container_depth = 0
        self._has_result_container = False

    def handle_starttag(
        self,
        tag: str,
        attrs: list[tuple[str, str | None]],
    ) -> None:
        tag = tag.lower()
        attr_map = {
            key.lower(): value
            for key, value in attrs
            if value is not None
        }
        classes = attr_map.get("class", "").split()
        is_result_container = (
            "dgControl_list" in classes
            or attr_map.get("id") == "mmComponent_images_2"
        )
        if is_result_container:
            self._has_result_container = True
            self._result_container_depth += 1
        self._element_stack.append((tag, is_result_container))

        if tag != "a" or "iusc" not in classes:
            return

        payload = attr_map.get("m")
        if not payload:
            return
        if self._result_container_depth:
            self.payloads.append(payload)
        else:
            # Retain support for small/legacy markup fixtures that contain
            # only a result anchor. Real pages with a known results container
            # never use these out-of-grid anchors as search candidates.
            self._fallback_payloads.append(payload)

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        for index in range(len(self._element_stack) - 1, -1, -1):
            if self._element_stack[index][0] != tag:
                continue
            removed = self._element_stack[index:]
            self._result_container_depth -= sum(
                1 for _, is_container in removed if is_container
            )
            del self._element_stack[index:]
            break

    @property
    def result_payloads(self) -> list[str]:
        if self._has_result_container:
            return self.payloads
        return self._fallback_payloads


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
        date_from: str | None = None,
        date_to: str | None = None,
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
        self.date_from, self.date_to = self._normalize_date_range(date_from, date_to)
        if self.provider != "bing" and self.date_from is not None:
            raise ValueError("date filtering is only supported by Playwright Bing")

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

    def _create_browser(self):
        if self.provider == "google" and os.environ.get("CHROME_USER_DATA_DIR", "").strip():
            try:
                return ExistingChromium.from_environment(timeout=self.timeout)
            except ValueError as exc:
                raise ScraperError(str(exc)) from exc

        try:
            from playwright_termux import Chromium
        except ImportError as exc:
            raise ScraperError(
                "Playwright backend requires playwright-termux; "
                'install with python -m pip install ".[playwright]"'
            ) from exc

        return Chromium(headless=True)

    def _search_once(self, query: str, limit: int) -> list[ImageResult]:
        search_url = self._build_url(query)
        print(
            f"scrape: playwright provider={self.provider} query={query}",
            flush=True,
        )

        stage = "creating browser"
        try:
            browser = self._create_browser()
            with browser as browser:
                stage = "opening page"
                page = browser.new_page()
                try:
                    stage = "waiting for matching search results"
                    self._goto(page, search_url)

                    seen: set[str] = set()
                    results: list[ImageResult] = []
                    stagnant_rounds = 0
                    previous_count = 0

                    for scroll_no in range(1, self.max_scrolls + 1):
                        stage = f"reading result page HTML (scroll {scroll_no})"
                        try:
                            page_html = page.content()
                        except Exception as exc:
                            if results:
                                print(
                                    f"scrape: playwright provider={self.provider} "
                                    f"could not read next page state; keeping "
                                    f"{len(results)} candidates: {exc}",
                                    flush=True,
                                )
                                break
                            raise ScraperError(
                                f"could not read rendered result HTML: {exc}"
                            ) from exc

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

                        stage = f"scrolling result page (scroll {scroll_no})"
                        try:
                            page.evaluate(
                                "window.scrollTo(0, "
                                "Math.max(document.documentElement?.scrollHeight || 0, "
                                "document.body?.scrollHeight || 0))"
                            )
                            if self.delay:
                                stage = f"waiting after scroll {scroll_no}"
                                page.wait_for_timeout(
                                    int(max(self.delay, 0.5) * 1000)
                                )
                        except Exception as exc:
                            # A scroll helper failure must not discard valid image
                            # URLs already extracted from the current result grid.
                            print(
                                f"scrape: playwright provider={self.provider} "
                                f"scroll failed; keeping {len(results)} candidates: "
                                f"{type(exc).__name__}: {exc}",
                                flush=True,
                            )
                            break

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
                f"Playwright {self.provider} browser error during {stage}: "
                f"{type(exc).__name__}: {exc}"
            ) from exc

    def _goto(self, page, url: str) -> None:
        """Navigate without relying on playwright-termux Page.goto()."""
        timeout = max(float(self.timeout), 0.1)
        navigation_timeout = min(timeout, 5.0)
        requested_url = urlparse(url)
        requested_query = parse_qs(requested_url.query).get("q", [""])[0]
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
                expression = (
                    "[document.readyState, location.href, "
                    "document.querySelectorAll("
                    "'ul.dgControl_list a.iusc, #mmComponent_images_2 a.iusc'"
                    ").length]"
                    if self.provider == "bing"
                    else "[document.readyState, location.href]"
                )
                result = page.session.send(
                    "Runtime.evaluate",
                    {
                        "expression": expression,
                        "returnByValue": True,
                    },
                    timeout=min(1.0, max(deadline - time.monotonic(), 0.1)),
                )
                value = result.get("result", {}).get("value")
                if isinstance(value, list) and len(value) >= 2:
                    current_url = str(value[1])
                    current = urlparse(current_url)
                    ready_state = str(value[0]).lower()

                    if self.provider == "bing":
                        current_query = parse_qs(current.query).get("q", [""])[0]
                        same_query = (
                            " ".join(current_query.split()).casefold()
                            == " ".join(requested_query.split()).casefold()
                        )
                        on_expected_page = (
                            current.scheme in {"http", "https"}
                            and current.hostname == requested_url.hostname
                            and current.path == requested_url.path
                            and bool(requested_query)
                            and same_query
                        )
                        result_grid_ready = (
                            len(value) >= 3
                            and isinstance(value[2], (int, float))
                            and value[2] > 0
                        )
                        if (
                            on_expected_page
                            and ready_state == "complete"
                            and result_grid_ready
                        ):
                            page._url = current_url
                            return
                    elif (
                        ready_state in {"interactive", "complete"}
                        and current_url not in {"", "about:blank"}
                    ):
                        # Let the Google extractor report consent/verification
                        # pages directly instead of masking them as timeouts.
                        page._url = current_url
                        return
            except Exception as exc:
                last_error = exc

            time.sleep(0.05)

        detail = (
            " (Bing did not render image-result cards for the requested query)"
            if self.provider == "bing"
            else ""
        )
        raise ScraperError(
            f"Playwright {self.provider} navigation timed out after "
            f"{timeout:g}s{detail}"
            + (f": {last_error}" if last_error else "")
        )

    @staticmethod
    def _normalize_date_range(
        date_from: str | None,
        date_to: str | None,
    ) -> tuple[str | None, str | None]:
        date_from = date_from or None
        date_to = date_to or None
        if date_from is None and date_to is None:
            return None, None
        if not date_from or not date_to:
            raise ValueError("date_from and date_to must be provided together")
        try:
            start = date.fromisoformat(date_from)
            end = date.fromisoformat(date_to)
        except (TypeError, ValueError) as exc:
            raise ValueError("dates must use YYYY-MM-DD format") from exc
        if start < date(1970, 1, 1) or end < date(1970, 1, 1):
            raise ValueError("Bing date filtering supports dates from 1970 onward")
        if start > end:
            raise ValueError("date_from must be on or before date_to")
        return start.isoformat(), end.isoformat()

    def _build_url(self, query: str) -> str:
        if self.provider == "bing":
            params = {"q": query, "form": "HDRSC2", "adlt": "off"}
            if self.date_from is not None and self.date_to is not None:
                epoch = date(1970, 1, 1)
                start_day = (date.fromisoformat(self.date_from) - epoch).days
                # Bing's end day is exclusive; add one day to include date_to.
                end_day = (date.fromisoformat(self.date_to) - epoch).days + 1
                params["filters"] = f'ex1:"ez5_{start_day}_{end_day}"'
            return f"{self.BING_URL}?{urlencode(params)}"
        return (
            f"{self.GOOGLE_URL}?"
            f"{urlencode({'q': query, 'tbm': 'isch', 'hl': 'en'})}"
        )

    @staticmethod
    def _extract_bing_urls(page_html: str) -> list[str]:
        parser = _BingResultParser()
        try:
            parser.feed(page_html)
        except Exception:
            return []

        urls: list[str] = []
        seen: set[str] = set()

        import json

        for raw_payload in parser.result_payloads:
            try:
                data = json.loads(html.unescape(raw_payload))
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
