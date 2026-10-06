from __future__ import annotations

import shutil
import tempfile
import time
from pathlib import Path
from typing import Iterator
from urllib.parse import parse_qs, urljoin, urlparse

from .base import ImageResult, ScraperError
from .google import DEFAULT_USER_AGENT, GoogleImagesScraper


class ChromiumGoogleImagesScraper:
    """Google Images scraper driven by a local headless Chromium browser.

    Selenium is imported lazily so the core package remains usable without the
    optional Chromium stack.
    """

    SEARCH_URL = "https://www.google.com/search"

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
        self.profile_root = (
            Path(profile_root).expanduser().resolve()
            if profile_root is not None
            else Path(tempfile.gettempdir()) / "lora-data-maker-chromium"
        )

    def search(self, query: str, *, limit: int) -> Iterator[ImageResult]:
        if not query.strip():
            raise ScraperError("Chromium Google Images query cannot be empty")
        if limit <= 0:
            return

        driver = self._start_driver()
        profile_dir = Path(driver._lora_profile_dir)

        try:
            url = self.SEARCH_URL
            params = {
                "tbm": "isch",
                "q": query,
                "hl": "en",
            }

            from urllib.parse import urlencode

            target_url = f"{url}?{urlencode(params)}"
            driver.get(target_url)
            self._check_page(driver)

            seen: set[str] = set()
            last_candidate_count = 0
            stagnant_rounds = 0

            for scroll_no in range(1, self.max_scrolls + 1):
                hrefs = driver.execute_script(
                    """
                    return Array.from(document.querySelectorAll('a[href]'))
                        .map(a => a.href)
                        .filter(Boolean);
                    """
                )

                candidates = self._extract_image_urls(hrefs)
                new_candidates = [
                    candidate for candidate in candidates
                    if candidate not in seen
                ]

                for candidate in new_candidates:
                    seen.add(candidate)

                print(
                    f"scrape: chromium scroll {scroll_no}/{self.max_scrolls} "
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
                self._check_page(driver)

            for rank, image_url in enumerate(seen, start=1):
                yield ImageResult(
                    image_url=image_url,
                    query=query,
                    page_url=self.SEARCH_URL,
                    rank=rank,
                )
                if rank >= limit:
                    break

        except ScraperError:
            raise
        except Exception as exc:
            raise ScraperError(f"Chromium scraper failed: {exc}") from exc
        finally:
            try:
                driver.quit()
            finally:
                shutil.rmtree(profile_dir, ignore_errors=True)

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

        self.profile_root.mkdir(parents=True, exist_ok=True)
        profile_dir = Path(
            tempfile.mkdtemp(
                prefix="profile-",
                dir=str(self.profile_root),
            )
        )

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
        options.add_argument(f"--user-data-dir={profile_dir}")

        service = Service(executable_path=driver_path)

        try:
            driver = webdriver.Chrome(service=service, options=options)
            driver._lora_profile_dir = str(profile_dir)
            driver.set_page_load_timeout(self.timeout)
            return driver
        except WebDriverException as exc:
            shutil.rmtree(profile_dir, ignore_errors=True)
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
