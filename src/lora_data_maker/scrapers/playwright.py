from __future__ import annotations

import html
import os
import re
import socket
import threading
import time
from datetime import date
from pathlib import Path
from html.parser import HTMLParser
from typing import Iterator
from urllib.parse import parse_qs, urlencode, urlparse

from .base import ImageResult, Scraper, ScraperCancelledError, ScraperError
from .existing_chromium import ExistingChromium



class _BingResultParser(HTMLParser):
    """Extract Bing image metadata from individual cards in the main result grid."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.payloads: list[str] = []
        self._fallback_payloads: list[str] = []
        self._element_stack: list[tuple[str, bool, bool]] = []
        self._result_container_depth = 0
        self._has_result_container = False
        self._card_stack: list[dict[str, object]] = []
        self.result_cards: list[dict[str, object]] = []
        self._card_order = 0

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

        is_result_card = tag == "li" and self._result_container_depth > 0
        self._element_stack.append((tag, is_result_container, is_result_card))

        if is_result_card:
            self._card_order += 1
            self._card_stack.append(
                {"payload": None, "text": [], "order": self._card_order}
            )

        if self._card_stack and self._result_container_depth:
            parts = self._card_stack[-1]["text"]
            if isinstance(parts, list):
                for key in ("title", "alt", "aria-label"):
                    value = attr_map.get(key)
                    if value:
                        parts.append(value)

        if tag != "a" or "iusc" not in classes:
            return

        payload = attr_map.get("m")
        if not payload:
            return

        if self._result_container_depth:
            self.payloads.append(payload)
            if self._card_stack:
                self._card_stack[-1]["payload"] = payload
        else:
            # Legacy fallback is kept for the URL-only helper's lightweight
            # fixtures. The actual scraper consumes result_cards only.
            self._fallback_payloads.append(payload)

    def handle_data(self, data: str) -> None:
        if self._card_stack and self._result_container_depth and data.strip():
            parts = self._card_stack[-1]["text"]
            if isinstance(parts, list):
                parts.append(data.strip())

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        for index in range(len(self._element_stack) - 1, -1, -1):
            if self._element_stack[index][0] != tag:
                continue
            removed = self._element_stack[index:]
            self._result_container_depth -= sum(
                1 for _, is_container, _ in removed if is_container
            )
            card_count = sum(1 for _, _, is_card in removed if is_card)
            del self._element_stack[index:]

            for _ in range(card_count):
                if not self._card_stack:
                    break
                card = self._card_stack.pop()
                payload = card.get("payload")
                if isinstance(payload, str) and payload:
                    raw_text = card.get("text")
                    text = " ".join(raw_text) if isinstance(raw_text, list) else ""
                    self.result_cards.append(
                        {
                            "payload": payload,
                            "text": text,
                            "order": card.get("order", 0),
                        }
                    )
            break

    @property
    def result_payloads(self) -> list[str]:
        if self._has_result_container:
            return self.payloads
        return self._fallback_payloads

    def ordered_result_cards(self) -> list[dict[str, object]]:
        if self.result_cards:
            return sorted(
                self.result_cards,
                key=lambda item: int(item.get("order", 0)),
            )

        # Bing sometimes changes the surrounding grid/card markup while keeping
        # the stable a.iusc metadata attribute. Fall back to those URLs, but
        # let the title/description relevance filter decide which ones survive.
        raw_payloads = (
            self.payloads
            if self._has_result_container
            else self._fallback_payloads
        )
        return [
            {"payload": payload, "text": "", "order": index}
            for index, payload in enumerate(raw_payloads, start=1)
        ]

class PlaywrightImagesScraper(Scraper):
    """Image-search scraper driven by Termux-native Chromium through Playwright.

    The Termux-specific Playwright implementation controls the native Chromium
    package through CDP, so no glibc Playwright browser binary is downloaded.
    """

    PROVIDERS = ("bing", "google")
    BING_URL = "https://www.bing.com/images/search"
    GOOGLE_URL = "https://www.google.com/search"
    GOOGLE_IMAGES_URL = "https://www.google.com/search?tbm=isch"

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
        proxy_url: str | None = None,
        safe_search: bool = False,
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

        if not isinstance(safe_search, bool):
            raise ValueError("safe_search must be a boolean")

        self.provider = provider
        self.proxy_url = proxy_url
        self.cancel_event: threading.Event | None = None
        self.safe_search = safe_search
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
                url_kind = "original"
                raise ScraperError(
                    f"Playwright {self.provider} image search returned no "
                    f"parseable {url_kind} image URLs"
                )
            except ScraperCancelledError:
                raise
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

    @staticmethod
    def _headless_requested() -> bool:
        """Headless defaults to true on ordinary Linux without a display, visible on Termux."""
        value = os.environ.get("LORA_DATA_MAKER_HEADLESS", "").strip().lower()
        if value:
            if value in {"1", "true", "yes", "on"}:
                return True
            if value in {"0", "false", "no", "off"}:
                return False
        is_termux = bool(os.environ.get("TERMUX_VERSION")) or (
            "/data/data/com.termux/files/" in os.environ.get("PREFIX", "")
        )
        return not is_termux and not bool(os.environ.get("DISPLAY", "").strip())

    @staticmethod
    def _data_home() -> Path:
        data_home = os.environ.get("XDG_DATA_HOME", "").strip()
        return (
            Path(data_home).expanduser()
            if data_home
            else Path.home() / ".local" / "share"
        )

    def _create_browser(self, initial_url: str | None = None):
        from .existing_chromium import ExistingChromium

        headless = self._headless_requested()
        executable = os.environ.get("CHROME_EXECUTABLE", "").strip() or None

        if self.proxy_url:
            # WireGuard gets a dedicated persistent profile and dynamic CDP port:
            # session cookies survive across runs, while proxy routing stays isolated.
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
                server.bind(("127.0.0.1", 0))
                port = int(server.getsockname()[1])
            profile = (
                self._data_home()
                / "lora-data-maker"
                / f"playwright-{self.provider}-wireguard-profile"
            )
            try:
                return ExistingChromium(
                    user_data_dir=profile,
                    port=port,
                    timeout=min(max(self.timeout, 0.5), 5.0),
                    auto_launch=True,
                    executable=executable,
                    initial_url=initial_url or (
                        self.GOOGLE_IMAGES_URL
                        if self.provider == "google"
                        else self.BING_URL
                    ),
                    proxy_server=self.proxy_url,
                    headless=headless,
                    shutdown_on_close=True,
                )
            except ValueError as exc:
                raise ScraperError(str(exc)) from exc

        if self.provider == "google":
            try:
                return ExistingChromium.from_environment(
                    timeout=self.timeout,
                    initial_url=initial_url or self.GOOGLE_IMAGES_URL,
                    headless=headless,
                )
            except ValueError as exc:
                raise ScraperError(str(exc)) from exc

        # Bing also uses a named profile instead of Chromium's throwaway temp
        # profile, so cookies, language settings, and consent persist between runs.
        profile_value = os.environ.get("CHROME_BING_USER_DATA_DIR", "").strip()
        profile = (
            Path(profile_value).expanduser()
            if profile_value
            else self._data_home() / "lora-data-maker" / "bing-profile"
        )
        raw_port = os.environ.get("CHROME_BING_DEBUG_PORT", "9223").strip()
        try:
            port = int(raw_port)
        except ValueError as exc:
            raise ScraperError(
                f"CHROME_BING_DEBUG_PORT must be an integer, got {raw_port!r}"
            ) from exc
        auto_launch_value = os.environ.get("CHROME_AUTO_LAUNCH", "1").strip().lower()
        auto_launch = auto_launch_value not in {"0", "false", "no", "off"}
        try:
            return ExistingChromium(
                user_data_dir=profile,
                port=port,
                timeout=min(max(self.timeout, 0.5), 5.0),
                auto_launch=auto_launch,
                executable=executable,
                initial_url=initial_url or self.BING_URL,
                headless=headless,
            )
        except ValueError as exc:
            raise ScraperError(str(exc)) from exc

    def _search_once(self, query: str, limit: int) -> list[ImageResult]:
        search_url = self._build_url(query)
        print(
            f"scrape: playwright provider={self.provider} query={query}",
            flush=True,
        )

        stage = "creating browser"
        try:
            browser = self._create_browser(search_url)
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

                        candidate_details: dict[str, dict[str, object]] = {}
                        if self.provider == "bing":
                            extracted = self._extract_bing_results(page_html, query)
                            candidate_details = {
                                str(item["image_url"]): item for item in extracted
                            }
                            candidates = list(candidate_details)
                        else:
                            try:
                                candidates = self._extract_google_original_urls(page_html)
                            except ScraperError as exc:
                                if not self._is_google_verification_error(exc):
                                    raise
                                candidates = self._wait_for_google_verification(
                                    page,
                                    query=query,
                                    initial_html=page_html,
                                )

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
                                        **(
                                            {
                                                "title": candidate_details[image_url].get("title", ""),
                                                "description": candidate_details[image_url].get("description", ""),
                                                "query_terms_matched": candidate_details[image_url].get("matched_terms", 0),
                                                "query_terms_total": candidate_details[image_url].get("query_terms_total", 0),
                                            }
                                            if image_url in candidate_details
                                            else {}
                                        ),
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

    @staticmethod
    def _is_google_verification_error(error: BaseException) -> bool:
        return "Google returned a consent/verification page" in str(error)

    def _wait_for_google_verification(
        self,
        page,
        *,
        query: str,
        initial_html: str,
    ) -> list[str]:
        """Wait on the same visible browser until a person clears Google's challenge."""
        print(
            "scrape: Google consent/verification detected. "
            "Search paused, not retrying. Complete the challenge in the open "
            "Chromium window; scraping will continue automatically.",
            flush=True,
        )
        page_html = initial_html
        next_notice = time.monotonic() + 15.0

        while True:
            if self.cancel_event is not None and self.cancel_event.is_set():
                raise ScraperCancelledError(
                    "scrape cancelled while waiting for Google verification"
                )
            try:
                candidates = self._extract_google_original_urls(page_html)
            except ScraperError as exc:
                if not self._is_google_verification_error(exc):
                    raise
                candidates = []

            if candidates:
                print(
                    "scrape: Google verification cleared; resuming the same search.",
                    flush=True,
                )
                return candidates

            now = time.monotonic()
            if now >= next_notice:
                current_url = str(getattr(page, "_url", "") or "")
                print(
                    "scrape: still waiting for Google verification to finish; "
                    "complete it in Chromium to continue"
                    + (f" (current URL: {current_url})" if current_url else ""),
                    flush=True,
                )
                next_notice = now + 15.0

            time.sleep(1.0)
            try:
                page_html = page.content()
            except Exception:
                # Browser/DOM transitions can briefly make CDP content reads fail.
                # Keep the verification wait alive and try again instead of
                # turning a transient read failure into a search retry.
                continue

    def _goto(self, page, url: str) -> None:
        """Navigate and distinguish slow cards from redirects or blocked pages."""
        timeout = max(float(self.timeout), 0.1)
        navigation_timeout = min(timeout, 5.0)
        requested_url = urlparse(url)
        requested_query = parse_qs(requested_url.query).get("q", [""])[0]
        result_grace = min(2.0, max(0.35, timeout * 0.1))
        redirect_grace = min(3.0, max(0.75, timeout * 0.1))
        ready_without_results_since: float | None = None
        ready_url_without_results: str | None = None
        unexpected_page_since: float | None = None
        last_unexpected_url: str | None = None
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
                    "document.querySelectorAll('a.iusc').length, "
                    "document.title, (document.body?.innerText || '').slice(0, 2000)]"
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
                        current_host = (current.hostname or "").lower()
                        on_bing_host = (
                            current_host == "bing.com"
                            or current_host.endswith(".bing.com")
                        )
                        on_expected_page = (
                            current.scheme in {"http", "https"}
                            and on_bing_host
                            and current.path == requested_url.path
                            and bool(requested_query)
                            and same_query
                        )
                        dom_ready = ready_state in {"interactive", "complete"}
                        result_grid_ready = (
                            len(value) >= 3
                            and isinstance(value[2], (int, float))
                            and value[2] > 0
                        )

                        if on_expected_page and dom_ready:
                            unexpected_page_since = None
                            last_unexpected_url = None
                            page._url = current_url
                            if result_grid_ready:
                                return

                            title = str(value[3]) if len(value) >= 4 else ""
                            body_text = (
                                str(value[4]).casefold() if len(value) >= 5 else ""
                            )
                            block_markers = (
                                "verify you are human",
                                "unusual traffic",
                                "enter the characters you see",
                                "captcha",
                                "automated requests",
                                "access denied",
                                "temporarily blocked",
                            )
                            if any(marker in body_text for marker in block_markers):
                                raise ScraperError(
                                    "Bing returned a verification/block page instead "
                                    "of image results "
                                    f"(title={title!r}, URL={current_url})"
                                )

                            now = time.monotonic()
                            if ready_without_results_since is None:
                                ready_without_results_since = now
                                ready_url_without_results = current_url
                            elif now - ready_without_results_since >= result_grace:
                                print(
                                    "scrape: playwright provider=bing "
                                    "page is ready but no a.iusc result anchors "
                                    f"appeared after {result_grace:g}s; "
                                    "continuing to result parsing",
                                    flush=True,
                                )
                                return
                        else:
                            ready_without_results_since = None
                            ready_url_without_results = None
                            if dom_ready and current_url not in {"", "about:blank"}:
                                now = time.monotonic()
                                if unexpected_page_since is None:
                                    unexpected_page_since = now
                                    last_unexpected_url = current_url
                                else:
                                    last_unexpected_url = current_url
                                    if now - unexpected_page_since >= redirect_grace:
                                        raise ScraperError(
                                            "Bing did not remain on the requested "
                                            "image-search page; possible redirect, "
                                            "consent, or verification page. "
                                            f"Expected query={requested_query!r}, "
                                            f"current URL={current_url}"
                                        )
                            else:
                                unexpected_page_since = None
                                last_unexpected_url = None
                    elif (
                        ready_state in {"interactive", "complete"}
                        and current_url not in {"", "about:blank"}
                    ):
                        # Let the Google extractor report consent/verification
                        # pages directly instead of masking them as timeouts.
                        page._url = current_url
                        return
            except ScraperError:
                raise
            except Exception as exc:
                last_error = exc

            time.sleep(0.05)

        if self.provider == "bing" and ready_url_without_results:
            # The correct search document became ready; missing image anchors
            # are a result-page issue, not a navigation timeout.
            page._url = ready_url_without_results
            print(
                "scrape: playwright provider=bing page is ready but contains "
                "no a.iusc result anchors; continuing to result parsing",
                flush=True,
            )
            return

        if self.provider == "bing" and last_unexpected_url:
            raise ScraperError(
                "Playwright Bing did not reach the requested image-search page; "
                f"expected query={requested_query!r}, "
                f"last URL={last_unexpected_url}"
            ) from last_error

        detail = (
            " (Bing did not reach the requested image-search URL; last URL="
            f"{current_url if 'current_url' in locals() else 'unknown'})"
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
            # Bing's "moderate" filters explicit imagery; "off" allows it.
            params = {
                "q": query,
                "form": "HDRSC2",
                "adlt": "moderate" if self.safe_search else "off",
            }
            if self.date_from is not None and self.date_to is not None:
                epoch = date(1970, 1, 1)
                start_day = (date.fromisoformat(self.date_from) - epoch).days
                # Bing's end day is exclusive; add one day to include date_to.
                end_day = (date.fromisoformat(self.date_to) - epoch).days + 1
                params["filters"] = f'ex1:"ez5_{start_day}_{end_day}"'
            return f"{self.BING_URL}?{urlencode(params)}"
        params = {
            "q": query,
            "tbm": "isch",
            "hl": "en",
            "safe": "active" if self.safe_search else "off",
        }
        return f"{self.GOOGLE_URL}?{urlencode(params)}"

    @staticmethod
    def _query_terms(query: str) -> list[str]:
        words = re.findall(r"[^\W_]+", query.casefold(), flags=re.UNICODE)
        stopwords = {
            "a", "an", "and", "are", "as", "at", "be", "by", "for",
            "from", "in", "into", "is", "of", "on", "or", "the", "to",
            "with", "photo", "photos", "image", "images", "picture", "pictures",
        }
        if len(words) <= 1:
            return list(dict.fromkeys(words))
        return list(dict.fromkeys(word for word in words if word not in stopwords)) or list(dict.fromkeys(words))

    @staticmethod
    def _extract_bing_results(
        page_html: str,
        query: str,
    ) -> list[dict[str, object]]:
        """Extract main-grid Bing cards and reject cards whose text doesn't match the query."""
        parser = _BingResultParser()
        try:
            parser.feed(page_html)
        except Exception:
            return []

        query_terms = PlaywrightImagesScraper._query_terms(query)
        if not query_terms:
            return []

        results: list[dict[str, object]] = []
        seen: set[str] = set()
        rejected = 0
        import json

        for card in parser.ordered_result_cards():
            raw_payload = card.get("payload")
            if not isinstance(raw_payload, str):
                continue
            try:
                data = json.loads(html.unescape(raw_payload))
            except (TypeError, ValueError):
                continue
            if not isinstance(data, dict):
                continue

            image_url = data.get("murl")
            if not isinstance(image_url, str) or not image_url.startswith(("http://", "https://")):
                continue
            host = (urlparse(image_url).hostname or "").lower().rstrip(".")
            if (
                host == "bing.com"
                or host.endswith(".bing.com")
                or host == "bing.net"
                or host.endswith(".bing.net")
                or image_url in seen
            ):
                continue

            title = str(data.get("t") or data.get("title") or "").strip()
            description = str(data.get("desc") or data.get("description") or "").strip()
            card_text = str(card.get("text") or "")
            searchable_text = " ".join((title, description, card_text)).casefold()
            result_words = set(re.findall(r"[^\W_]+", searchable_text, flags=re.UNICODE))
            # Basic plural tolerance for English titles.
            normalized_result_words = set(result_words)
            normalized_result_words.update(
                word[:-1] for word in result_words if len(word) > 4 and word.endswith("s")
            )
            matched_terms = [
                term for term in query_terms
                if term in normalized_result_words
                or (len(term) > 4 and term.endswith("s") and term[:-1] in normalized_result_words)
            ]
            required_matches = (
                1 if len(query_terms) == 1
                else max(2, (3 * len(query_terms) + 4) // 5)
            )

            # Do not silently fall back to arbitrary out-of-grid URLs or images
            # with no textual evidence connecting them to the requested query.
            if len(matched_terms) < required_matches:
                rejected += 1
                continue

            seen.add(image_url)
            results.append(
                {
                    "image_url": image_url,
                    "title": title,
                    "description": description,
                    "matched_terms": len(matched_terms),
                    "query_terms_total": len(query_terms),
                }
            )

        print(
            f"scrape: playwright provider=bing relevance "
            f"accepted={len(results)} rejected={rejected} "
            f"query_terms={len(query_terms)}",
            flush=True,
        )
        return results

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
    def _extract_google_original_urls(page_html: str) -> list[str]:
        """Extract source URLs from Google's ou payload, never cached tu thumbnails."""
        urls: list[str] = []
        seen: set[str] = set()

        # Google Images result payloads pair ou (original/source image URL)
        # with tu (Google's cached thumbnail). Downloading tu can turn a
        # low-resolution preview into a permanent training sample, so only ou
        # is eligible here. The source hosts may reject hotlink downloads; those
        # failures are preferable to silently polluting a dataset with thumbnails.
        original_pattern = re.compile(
            r"""["']ou["']\s*:\s*["']((?:\\.|[^"'])+)["']"""
        )
        for raw in original_pattern.findall(page_html):
            value = html.unescape(
                PlaywrightImagesScraper._decode_json_string(raw)
            ).strip().rstrip(",;")
            if not PlaywrightImagesScraper._valid_image_candidate(value, seen):
                continue
            seen.add(value)
            urls.append(value)

        if urls:
            return urls

        # Detect a blocked Google page only after checking actual source URLs,
        # so challenge text in scripts/templates does not reject valid results.
        challenge_markup = re.sub(
            r"(?is)<(script|style|noscript)\b[^>]*>.*?</\1\s*>",
            " ",
            page_html,
        )
        visible_text = html.unescape(
            re.sub(r"(?s)<[^>]+>", " ", challenge_markup)
        ).casefold()
        if any(
            marker in visible_text
            for marker in (
                "our systems have detected unusual traffic",
                "detected unusual traffic from your computer network",
                "before you continue to google",
                "automated queries",
                "unusual traffic",
                "consent.google.com",
            )
        ):
            raise ScraperError(
                "Google returned a consent/verification page; "
                "the scraper does not bypass verification challenges"
            )

        return []

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
