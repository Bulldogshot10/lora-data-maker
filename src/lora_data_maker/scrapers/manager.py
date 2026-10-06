from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from .archive import DownloadArchive
from .base import ImageResult, Scraper, ScraperError, canonicalize_url
from .downloader import DownloadResult, ImageDownloader
from .google import DEFAULT_USER_AGENT
from .registry import create_scraper, validate_backend_chain


class ScraperManager:
    def __init__(
        self,
        *,
        backend: str,
        fallbacks: Iterable[str] = (),
        output_dir: Path | str,
        cache_dir: Path | str,
        timeout: float = 20.0,
        retries: int = 3,
        delay: float = 1.0,
        user_agent: str | None = None,
    ) -> None:
        self.backends = validate_backend_chain(backend, fallbacks)
        self.backend = self.backends[0]
        self.fallbacks = self.backends[1:]

        self.output_dir = Path(output_dir).expanduser().resolve()
        self.cache_dir = Path(cache_dir).expanduser().resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        effective_user_agent = user_agent or DEFAULT_USER_AGENT
        profile_root = self.cache_dir / "chromium-profiles"

        self.scrapers: dict[str, Scraper] = {
            name: create_scraper(
                name,
                timeout=timeout,
                retries=retries,
                delay=delay,
                user_agent=effective_user_agent,
                profile_root=profile_root,
            )
            for name in self.backends
        }
        # Backward-compatible primary scraper attribute.
        self.scraper = self.scrapers[self.backend]

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
        next_index = self._next_index()

        # One URL set is shared by every engine in this run.
        seen_urls: set[str] = set()
        for raw_url in (*known_urls, *self.archive.urls):
            normalized = canonicalize_url(raw_url)
            if normalized:
                seen_urls.add(normalized)

        # Also prevent exact-content overlap when different engines expose
        # different URLs for the same image.
        seen_hashes: set[str] = set()

        for engine_index, backend in enumerate(self.backends):
            remaining = count - len(downloads)
            if remaining <= 0:
                break

            search_limit = max(remaining, remaining * 3)
            print(
                f"scrape: engine={backend} remaining={remaining} "
                f"search_limit={search_limit}",
                flush=True,
            )

            scraper = (
                self.scraper
                if backend == self.backend
                else self.scrapers[backend]
            )

            try:
                for result in scraper.search(query, limit=search_limit):
                    candidate_url = canonicalize_url(result.image_url)
                    if not candidate_url or candidate_url in seen_urls:
                        continue

                    # Mark before download so a failed URL is not retried
                    # through another fallback engine in the same run.
                    seen_urls.add(candidate_url)

                    try:
                        downloaded = self.downloader.download(
                            candidate_url,
                            self.output_dir,
                            index=next_index,
                            archive=self.archive,
                        )
                        if downloaded is None:
                            continue

                        if downloaded.sha256 in seen_hashes:
                            try:
                                downloaded.path.unlink(missing_ok=True)
                            except OSError:
                                pass
                            print(
                                f"scrape: skip duplicate image content from {backend}",
                                flush=True,
                            )
                            continue

                        seen_hashes.add(downloaded.sha256)
                        downloads.append(downloaded)
                        next_index += 1

                        print(
                            f"scrape: downloaded {len(downloads)}/{count} "
                            f"via {backend}",
                            flush=True,
                        )
                    except ScraperError as exc:
                        failures.append((result, str(exc)))

                    if len(downloads) >= count:
                        break

            except ScraperError as exc:
                failures.append(
                    (
                        ImageResult(
                            image_url=f"[{backend}:search]",
                            query=query,
                            metadata={"backend": backend, "stage": "search"},
                        ),
                        str(exc),
                    )
                )

            if len(downloads) >= count:
                break

            if engine_index + 1 < len(self.backends):
                next_backend = self.backends[engine_index + 1]
                print(
                    f"scrape: fallback {backend} -> {next_backend} "
                    f"(target not yet reached)",
                    flush=True,
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
