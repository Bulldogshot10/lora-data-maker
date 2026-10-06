from __future__ import annotations

from pathlib import Path
from collections.abc import Iterable

from .archive import DownloadArchive
from .base import ImageResult, Scraper, ScraperError
from .direct import DirectURLScraper
from .downloader import DownloadResult, ImageDownloader
from .google import GoogleImagesScraper


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
    ) -> None:
        self.backend = backend
        self.output_dir = Path(output_dir).expanduser().resolve()
        self.cache_dir = Path(cache_dir).expanduser().resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        effective_user_agent = user_agent or GoogleImagesScraper.DEFAULT_USER_AGENT

        if backend == "google":
            self.scraper: Scraper = GoogleImagesScraper(
                timeout=timeout,
                retries=retries,
                delay=delay,
                user_agent=effective_user_agent,
            )
        elif backend == "direct":
            self.scraper = DirectURLScraper()
        else:
            raise ScraperError(
                f"unknown scraper backend {backend!r}; expected google or direct"
            )

        self.downloader = ImageDownloader(
            timeout=timeout,
            retries=retries,
            delay=delay,
            user_agent=effective_user_agent,
        )
        self.archive = DownloadArchive(self.cache_dir / "downloads.txt")

    def run(
        self,
        query: str,
        *,
        count: int,
        known_urls: Iterable[str] = (),
    ) -> tuple[list[DownloadResult], list[tuple[ImageResult, str]]]:
        if count <= 0:
            raise ScraperError("count must be greater than zero")

        downloads: list[DownloadResult] = []
        failures: list[tuple[ImageResult, str]] = []
        known_url_set = set(known_urls)
        next_index = self._next_index()

        # Ask the source for more candidates than the target because some
        # candidates can already be archived or fail to download.
        search_limit = max(count, count * 3)

        for result in self.scraper.search(query, limit=search_limit):
            if result.image_url in self.archive or result.image_url in known_url_set:
                continue

            try:
                downloaded = self.downloader.download(
                    result.image_url,
                    self.output_dir,
                    index=next_index,
                    archive=self.archive,
                )
                if downloaded is None:
                    continue
                downloads.append(downloaded)
                next_index += 1
            except ScraperError as exc:
                failures.append((result, str(exc)))
                continue

            if len(downloads) >= count:
                break

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
