from __future__ import annotations

from pathlib import Path
from collections.abc import Callable, Iterable

from .archive import DownloadArchive
from .base import (
    SCRAPER_BACKENDS,
    ImageResult,
    Scraper,
    ScraperCancelledError,
    ScraperError,
)
from .chromium import ChromiumImagesScraper
from .direct import DirectURLScraper
from .duckduckgo import DuckDuckGoImagesScraper
from .downloader import DownloadResult, ImageDownloader
from .google import DEFAULT_USER_AGENT, GoogleImagesScraper
from .playwright import PlaywrightImagesScraper


class ScraperManager:
    def __init__(
        self,
        *,
        backend: str,
        output_dir: Path | str,
        cache_dir: Path | str,
        timeout: float = 20.0,
        retries: int = 3,
        delay: float = 1.0,
        user_agent: str | None = None,
        max_scrolls: int = 30,
    ) -> None:
        self.backend = backend
        self.output_dir = Path(output_dir).expanduser().resolve()
        self.cache_dir = Path(cache_dir).expanduser().resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        effective_user_agent = user_agent or DEFAULT_USER_AGENT

        if backend == "google":
            self.scraper: Scraper = GoogleImagesScraper(
                timeout=timeout,
                retries=retries,
                delay=delay,
                user_agent=effective_user_agent,
            )
        elif backend == "chromium":
            self.scraper = ChromiumImagesScraper(
                timeout=timeout,
                retries=retries,
                delay=delay,
                user_agent=effective_user_agent,
            )
        elif backend == "duckduckgo":
            self.scraper = DuckDuckGoImagesScraper(
                timeout=timeout,
                retries=retries,
                delay=delay,
                user_agent=effective_user_agent,
            )
        elif backend == "playwright-bing":
            self.scraper = PlaywrightImagesScraper(
                provider="bing",
                timeout=timeout,
                retries=retries,
                delay=delay,
                max_scrolls=max_scrolls,
            )
        elif backend == "playwright-google":
            self.scraper = PlaywrightImagesScraper(
                provider="google",
                timeout=timeout,
                retries=retries,
                delay=delay,
                max_scrolls=max_scrolls,
            )
        elif backend == "direct":
            self.scraper = DirectURLScraper()
        else:
            raise ScraperError(
                f"unknown scraper backend {backend!r}; expected "
                + ", ".join(SCRAPER_BACKENDS)
            )

        self.downloader = ImageDownloader(
            timeout=timeout,
            retries=retries,
            delay=delay,
            user_agent=effective_user_agent,
        )
        self.archive = DownloadArchive(self.cache_dir / "downloads.txt")
        self.last_search_error: str | None = None
        self.cancelled = False

    def run(
        self,
        query: str,
        *,
        count: int,
        known_urls: Iterable[str] = (),
        progress: Callable[[int, int, str], None] | None = None,
        cancel_event: threading.Event | None = None,
    ) -> tuple[list[DownloadResult], list[tuple[ImageResult, str]]]:
        if count <= 0:
            raise ScraperError("count must be greater than zero")

        downloads: list[DownloadResult] = []
        failures: list[tuple[ImageResult, str]] = []
        known_url_set = set(known_urls)
        self.last_search_error = None
        self.cancelled = False
        next_index = self._next_index()

        if cancel_event is not None and cancel_event.is_set():
            self.cancelled = True
            if progress is not None:
                progress(0, count, f"scrape cancelled: downloaded 0/{count}")
            return downloads, failures

        # Ask the source for more candidates than the target because some
        # candidates can already be archived or fail to download.
        search_limit = max(count, count * 3)

        if progress is not None:
            progress(0, count, f"searching for up to {search_limit} candidates...")
        print(f"scrape: searching for up to {search_limit} candidates...", flush=True)

        try:
            candidates = self.scraper.search(query, limit=search_limit)
            for result in candidates:
                if cancel_event is not None and cancel_event.is_set():
                    self.cancelled = True
                    break
                if result.image_url in self.archive or result.image_url in known_url_set:
                    continue

                try:
                    downloaded = self.downloader.download(
                        result.image_url,
                        self.output_dir,
                        index=next_index,
                        archive=self.archive,
                        cancel_event=cancel_event,
                    )
                    if downloaded is None:
                        continue
                    downloads.append(downloaded)
                    next_index += 1
                    message = f"downloaded {len(downloads)}/{count}"
                    if progress is not None:
                        progress(len(downloads), count, message)
                    print(
                        f"scrape: {message}",
                        flush=True,
                    )
                except ScraperCancelledError:
                    self.cancelled = True
                    break
                except ScraperError as exc:
                    failures.append((result, str(exc)))
                    print(
                        f"scrape: skipped failed image: {result.image_url} :: {exc}",
                        flush=True,
                    )
                    continue

                if len(downloads) >= count:
                    break
        except ScraperError as exc:
            self.last_search_error = str(exc)
            if downloads:
                print(
                    f"scrape: search source stopped; keeping "
                    f"{len(downloads)} downloaded images: {exc}",
                    flush=True,
                )
            else:
                print(
                    f"scrape: search source failed: {exc}",
                    flush=True,
                )

        if progress is not None:
            if self.cancelled:
                progress(
                    len(downloads),
                    count,
                    f"scrape cancelled: downloaded {len(downloads)}/{count}",
                )
            elif downloads:
                progress(
                    len(downloads),
                    count,
                    f"scrape finished: downloaded {len(downloads)}/{count}",
                )
        return downloads, failures

    def _next_index(self) -> int:
        highest = 0
        for path in self.output_dir.iterdir():
            if not path.is_file():
                continue
            stem = path.stem
            if stem.isdigit():
                highest = max(highest, int(stem))
        return highest + 1
