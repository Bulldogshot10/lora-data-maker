from __future__ import annotations

import html
import json
import shutil
import time
from pathlib import Path
from typing import Iterator
from urllib.parse import urlencode, urlparse

from .base import ImageResult, ScraperError
from .duckduckgo import DuckDuckGoImagesScraper


DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Linux; Android 14; Mobile) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/154.0.0.0 Mobile Safari/537.36"
)


class ChromiumImagesScraper:
    """Hybrid image-search scraper using Bing via Chromium, then DuckDuckGo.

    Bing is the primary source for this backend. DuckDuckGo is used to fill
    any remaining candidate budget when Bing returns too few results or its
    browser backend is unavailable.
    """

    BING_URL = "https://www.bing.com/images/search"

    def __init__(
        self,
        *,
        timeout: float = 20.0,
        retries: int = 3,
        delay: float = 1.0,
        user_agent: str = DEFAULT_USER_AGENT,
        chromium_binary: str = "chromium-browser",
        chromedriver_binary: str = "chromedriver",
        max_scrolls: int = 30,
    ) -> None:
        if timeout <= 0:
            raise ValueError("timeout must be greater than zero")
        if retries < 0:
            raise ValueError("retries cannot be negative")
        if delay < 0:
            raise ValueError("delay cannot be negative")
        if max_scrolls <= 0:
            raise ValueError("max_scrolls must be greater than zero")

        self.timeout = timeout
        self.retries = retries
        self.delay = delay
        self.user_agent = user_agent
        self.chromium_binary = chromium_binary
        self.chromedriver_binary = chromedriver_binary
        self.max_scrolls = max_scrolls

    def search(self, query: str, *, limit: int) -> Iterator[ImageResult]:
        if not query.strip():
            raise ScraperError("Chromium image-search query cannot be empty")
        if limit <= 0:
            return

        seen: set[str] = set()
        ordered_candidates: list[ImageResult] = []
        bing_error: Exception | None = None
        duckduckgo_error: Exception | None = None

        # Chromium is the Bing-backed option: prefer its results, then use
        # DuckDuckGo only to fill any remaining candidate budget.
        print("scrape: chromium starting browser...", flush=True)
        driver = None
        try:
            driver = self._start_driver()
            target_url = self._build_bing_search_url(query)
            print(
                f"scrape: chromium provider=bing query={query}",
                flush=True,
            )
            self._load_page(driver, target_url)

            last_candidate_count = 0
            stagnant_rounds = 0

            for scroll_no in range(1, self.max_scrolls + 1):
                raw_items = driver.execute_script(
                    """
                    return Array.from(document.querySelectorAll('a.iusc'))
                        .map(a => a.getAttribute('m'))
                        .filter(Boolean);
                    """
                )
                candidates = self._extract_bing_image_urls(raw_items)

                for candidate in candidates:
                    if candidate in seen:
                        continue
                    seen.add(candidate)
                    ordered_candidates.append(
                        ImageResult(
                            image_url=candidate,
                            query=query,
                            page_url=self.BING_URL,
                            metadata={"provider": "bing"},
                        )
                    )

                print(
                    f"scrape: chromium provider=bing "
                    f"scroll={scroll_no}/{self.max_scrolls} "
                    f"candidates={len(ordered_candidates)}",
                    flush=True,
                )

                if len(ordered_candidates) >= limit:
                    break

                if len(ordered_candidates) == last_candidate_count:
                    stagnant_rounds += 1
                else:
                    stagnant_rounds = 0
                last_candidate_count = len(ordered_candidates)

                if stagnant_rounds >= 4:
                    break

                driver.execute_script(
                    "window.scrollTo(0, document.body.scrollHeight);"
                )
                time.sleep(max(self.delay, 0.5))
        except ScraperError as exc:
            bing_error = exc
            print(
                f"scrape: Bing unavailable; trying DuckDuckGo fallback: {exc}",
                flush=True,
            )
        except Exception as exc:
            bing_error = exc
            print(
                "scrape: Bing failed; trying DuckDuckGo fallback: "
                f"{type(exc).__name__}: {exc}",
                flush=True,
            )
        finally:
            if driver is not None:
                try:
                    driver.quit()
                except Exception:
                    pass

        remaining = limit - len(ordered_candidates)
        if remaining > 0:
            print(
                f"scrape: chromium provider=duckduckgo "
                f"query={query} limit={remaining}",
                flush=True,
            )
            duckduckgo = DuckDuckGoImagesScraper(
                timeout=self.timeout,
                retries=self.retries,
                delay=self.delay,
                user_agent=self.user_agent,
            )
            try:
                for result in duckduckgo.search(query, limit=remaining):
                    if result.image_url in seen:
                        continue
                    seen.add(result.image_url)
                    original_metadata = dict(result.metadata or {})
                    original_metadata["provider"] = "duckduckgo"
                    ordered_candidates.append(
                        ImageResult(
                            image_url=result.image_url,
                            query=query,
                            page_url=(
                                result.page_url or DuckDuckGoImagesScraper.SITE_URL
                            ),
                            metadata=original_metadata,
                        )
                    )
                    if len(ordered_candidates) >= limit:
                        break
            except ScraperError as exc:
                duckduckgo_error = exc
                print(
                    f"scrape: DuckDuckGo fallback unavailable: {exc}",
                    flush=True,
                )

        if ordered_candidates:
            for rank, result in enumerate(ordered_candidates[:limit], start=1):
                metadata = dict(result.metadata or {})
                metadata.setdefault("provider", "chromium")
                yield ImageResult(
                    image_url=result.image_url,
                    query=query,
                    page_url=result.page_url,
                    rank=rank,
                    metadata=metadata,
                )
            return

        details = []
        if bing_error is not None:
            details.append(f"Bing: {bing_error}")
        if duckduckgo_error is not None:
            details.append(f"DuckDuckGo: {duckduckgo_error}")
        suffix = f" ({'; '.join(details)})" if details else ""
        raise ScraperError(
            "Chromium image search returned no parseable image URLs "
            f"from Bing or DuckDuckGo{suffix}"
        )

    @staticmethod
    def _build_bing_search_url(query: str) -> str:
        params = {"q": query, "form": "HDRSC2"}
        return f"{ChromiumImagesScraper.BING_URL}?{urlencode(params)}"

    @staticmethod
    def _extract_bing_image_urls(payloads: object) -> list[str]:
        if not isinstance(payloads, list):
            return []

        urls: list[str] = []
        seen: set[str] = set()

        for raw_payload in payloads:
            if not isinstance(raw_payload, str):
                continue

            try:
                data = json.loads(html.unescape(raw_payload))
            except (TypeError, ValueError):
                continue

            if not isinstance(data, dict):
                continue

            image_url = data.get("murl")
            if (
                not isinstance(image_url, str)
                or not image_url.startswith(("http://", "https://"))
            ):
                continue

            host = urlparse(image_url).netloc.lower()
            if (
                host == "bing.com"
                or host.endswith(".bing.com")
                or host == "mm.bing.net"
                or host.endswith(".mm.bing.net")
            ):
                continue
            if image_url in seen:
                continue

            seen.add(image_url)
            urls.append(image_url)

        return urls

    def _load_page(self, driver, target_url: str) -> bool:
        try:
            from selenium.common.exceptions import TimeoutException, WebDriverException
        except ImportError as exc:
            raise ScraperError(
                "Chromium backend requires Selenium; install the "
                "chromium optional dependency"
            ) from exc

        last_timeout: Exception | None = None
        for attempt in range(self.retries + 1):
            try:
                driver.get(target_url)
                deadline = time.monotonic() + min(max(self.timeout, 1.0), 10.0)
                while time.monotonic() < deadline:
                    ready_state = str(
                        driver.execute_script(
                            "return document.readyState || '';"
                        )
                    ).lower()
                    if ready_state in {"interactive", "complete"}:
                        break
                    time.sleep(0.1)

                time.sleep(max(self.delay, 0.25))
                return True
            except TimeoutException as exc:
                last_timeout = exc
                try:
                    driver.execute_script("window.stop();")
                except Exception:
                    pass

                if attempt < self.retries:
                    print(
                        "scrape: chromium provider=bing "
                        f"page load timeout; retry={attempt + 1}/{self.retries}",
                        flush=True,
                    )
                    time.sleep(min(max(self.delay, 0.25) * (2**attempt), 30.0))
                    continue

                raise ScraperError(
                    "Chromium Bing page load timed out after "
                    f"{self.retries + 1} attempts"
                ) from exc
            except WebDriverException as exc:
                if attempt < self.retries:
                    print(
                        "scrape: chromium provider=bing "
                        f"webdriver error; retry={attempt + 1}/{self.retries}",
                        flush=True,
                    )
                    time.sleep(min(max(self.delay, 0.25) * (2**attempt), 30.0))
                    continue

                raise ScraperError(
                    f"Chromium Bing webdriver error: {exc}"
                ) from exc

        raise ScraperError(
            f"Chromium Bing page load failed: {last_timeout}"
        )

    def _start_driver(self):
        try:
            from selenium import webdriver
            from selenium.common.exceptions import WebDriverException
            from selenium.webdriver.chrome.options import Options
            from selenium.webdriver.chrome.service import Service
        except ImportError as exc:
            raise ScraperError(
                "Chromium backend requires Selenium; install the "
                "chromium optional dependency"
            ) from exc

        chromium_path = self._resolve_executable(self.chromium_binary)
        driver_path = self._resolve_executable(self.chromedriver_binary)

        options = Options()
        options.binary_location = chromium_path
        options.page_load_strategy = "none"
        options.add_argument("--headless=new")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--disable-gpu")
        options.add_argument("--disable-extensions")
        options.add_argument("--no-first-run")
        options.add_argument("--no-default-browser-check")
        options.add_argument(f"--user-agent={self.user_agent}")
        options.add_argument("--window-size=1365,2200")
        service = Service(executable_path=driver_path)

        try:
            driver = webdriver.Chrome(service=service, options=options)
            driver.set_page_load_timeout(self.timeout)
            return driver
        except WebDriverException as exc:
            raise ScraperError(
                "could not start Chromium WebDriver: "
                f"{exc}. Check that chromium-browser and chromedriver "
                "are installed and compatible."
            ) from exc

    @staticmethod
    def _resolve_executable(value: str) -> str:
        path = Path(value).expanduser()
        if path.is_file():
            return str(path)

        resolved = shutil.which(value)
        if resolved:
            return resolved

        raise ScraperError(f"executable not found: {value}")
