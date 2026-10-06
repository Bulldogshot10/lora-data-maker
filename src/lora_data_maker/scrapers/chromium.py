from __future__ import annotations

import html
import json
import shutil
import time
from pathlib import Path
from typing import Iterator
from urllib.parse import parse_qs, urlencode, urljoin, urlparse

from .base import ImageResult, ScraperError
from .google import DEFAULT_USER_AGENT, GoogleImagesScraper


class ChromiumImagesScraper:
    """Image-search scraper driven by a local headless Chromium browser.

    Selenium is imported lazily so the core package remains usable without the
    optional Chromium stack.
    """

    GOOGLE_URL = "https://www.google.com/search"
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
        profile_root: Path | str | None = None,
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
        # Kept for backwards compatibility. Selenium manages its own temporary
        # browser profile; forcing --user-data-dir breaks Chromium/ChromeDriver
        # startup on some Termux/Android installations.
        self.profile_root = (
            Path(profile_root).expanduser().resolve()
            if profile_root is not None
            else None
        )

    def search(self, query: str, *, limit: int) -> Iterator[ImageResult]:
        if not query.strip():
            raise ScraperError("Chromium image-search query cannot be empty")
        if limit <= 0:
            return

        print("scrape: chromium starting browser...", flush=True)
        driver = self._start_driver()

        try:
            providers = (
                ("google", self.GOOGLE_URL),
                ("bing", self.BING_URL),
            )

            for provider, base_url in providers:
                seen: set[str] = set()
                last_candidate_count = 0
                stagnant_rounds = 0

                target_url = self._build_search_url(provider, base_url, query)
                print(
                    f"scrape: chromium provider={provider} query={query}",
                    flush=True,
                )

                try:
                    driver.get(target_url)
                    if provider == "google":
                        self._check_page(driver)
                except ScraperError as exc:
                    if provider == "google" and "verification" in str(exc).lower():
                        print(
                            "scrape: google verification detected; "
                            "falling back to Bing Images",
                            flush=True,
                        )
                        continue
                    raise

                for scroll_no in range(1, self.max_scrolls + 1):
                    if provider == "google":
                        raw_items = driver.execute_script(
                            """
                            return Array.from(document.querySelectorAll('a[href]'))
                                .map(a => a.href)
                                .filter(Boolean);
                            """
                        )
                        candidates = self._extract_image_urls(raw_items)
                    else:
                        raw_items = driver.execute_script(
                            """
                            return Array.from(document.querySelectorAll('a.iusc'))
                                .map(a => a.getAttribute('m'))
                                .filter(Boolean);
                            """
                        )
                        candidates = self._extract_bing_image_urls(raw_items)

                    for candidate in candidates:
                        seen.add(candidate)

                    print(
                        f"scrape: chromium provider={provider} "
                        f"scroll={scroll_no}/{self.max_scrolls} "
                        f"candidates={len(seen)}",
                        flush=True,
                    )

                    if len(seen) >= limit:
                        break

                    if len(seen) == last_candidate_count:
                        stagnant_rounds += 1
                    else:
                        stagnant_rounds = 0
                    last_candidate_count = len(seen)

                    if stagnant_rounds >= 4:
                        break

                    driver.execute_script(
                        "window.scrollTo(0, document.body.scrollHeight);"
                    )
                    time.sleep(max(self.delay, 0.5))

                    if provider == "google":
                        self._check_page(driver)

                if seen:
                    for rank, image_url in enumerate(seen, start=1):
                        yield ImageResult(
                            image_url=image_url,
                            query=query,
                            page_url=base_url,
                            rank=rank,
                            metadata={"provider": provider},
                        )
                        if rank >= limit:
                            break
                    return

                if provider == "google":
                    print(
                        "scrape: google returned no image candidates; "
                        "falling back to Bing Images",
                        flush=True,
                    )

            raise ScraperError(
                "Chromium image search returned no parseable image URLs "
                "from Google or Bing"
            )
        except ScraperError:
            raise
        except Exception as exc:
            raise ScraperError(f"Chromium scraper failed: {exc}") from exc
        finally:
            driver.quit()

    @staticmethod
    def _build_search_url(provider: str, base_url: str, query: str) -> str:
        if provider == "google":
            params = {"tbm": "isch", "q": query, "hl": "en"}
        else:
            params = {"q": query, "form": "HDRSC2"}
        return f"{base_url}?{urlencode(params)}"

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
            if host.endswith("bing.com") or host.endswith("mm.bing.net"):
                continue
            if image_url in seen:
                continue

            seen.add(image_url)
            urls.append(image_url)

        return urls

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
        options.add_argument("--headless=new")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--disable-gpu")
        options.add_argument("--disable-extensions")
        options.add_argument("--disable-background-networking")
        options.add_argument("--disable-popup-blocking")
        options.add_argument("--no-first-run")
        options.add_argument("--no-default-browser-check")
        options.add_argument("--lang=en-US")
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

    @staticmethod
    def _extract_image_urls(hrefs: object) -> list[str]:
        if not isinstance(hrefs, list):
            return []

        urls: list[str] = []
        seen: set[str] = set()

        for raw_href in hrefs:
            if not isinstance(raw_href, str):
                continue

            absolute = urljoin("https://www.google.com", raw_href)
            parsed = urlparse(absolute)

            if parsed.netloc.lower().endswith(".google.com") or (
                parsed.netloc.lower() == "google.com"
            ):
                values = parse_qs(parsed.query)
                image_values = values.get("imgurl", [])
                for image_url in image_values:
                    normalized = image_url.strip()
                    if not normalized.startswith(("http://", "https://")):
                        continue
                    image_host = urlparse(normalized).netloc.lower()
                    if image_host == "encrypted-tbn0.gstatic.com":
                        continue
                    if not GoogleImagesScraper._looks_like_image_url(normalized):
                        continue
                    if normalized in seen:
                        continue
                    seen.add(normalized)
                    urls.append(normalized)
                continue

            if "imgurl=" not in absolute:
                continue

            values = parse_qs(parsed.query)
            for image_url in values.get("imgurl", []):
                normalized = image_url.strip()
                if not normalized.startswith(("http://", "https://")):
                    continue
                if not GoogleImagesScraper._looks_like_image_url(normalized):
                    continue
                if normalized in seen:
                    continue
                seen.add(normalized)
                urls.append(normalized)

        return urls

    @staticmethod
    def _check_page(driver) -> None:
        try:
            text = str(
                driver.execute_script(
                    "return document.body ? document.body.innerText : '';"
                )
            ).lower()
            title = str(driver.title).lower()
        except Exception as exc:
            raise ScraperError(f"could not inspect Chromium page: {exc}") from exc

        markers = (
            "our systems have detected unusual traffic",
            "detected unusual traffic from your computer network",
            "before you continue to google",
            "unusual traffic",
        )
        if any(marker in text or marker in title for marker in markers):
            raise ScraperError(
                "Google returned a consent/verification page; "
                "the Chromium scraper does not bypass verification challenges"
            )
