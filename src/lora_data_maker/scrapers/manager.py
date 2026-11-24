from __future__ import annotations

import threading
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
from .site import PlaywrightSiteScraper
from ..processing.quality import evaluate_quality


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
        date_from: str | None = None,
        date_to: str | None = None,
        proxy_url: str | None = None,
        safe_search: bool = False,
        min_image_width: int = 256,
        min_image_height: int = 256,
        min_blur_score: float = 0.0,
        selector: str | None = None,
    ) -> None:
        if not isinstance(safe_search, bool):
            raise ScraperError("safe_search must be a boolean")
        if min_image_width < 1 or min_image_height < 1:
            raise ScraperError("minimum scraped image dimensions must be positive")
        if min_blur_score < 0:
            raise ScraperError("min_blur_score cannot be negative")
        self.backend = backend
        self.proxy_url = proxy_url
        self.safe_search = safe_search
        self.min_image_width = int(min_image_width)
        self.min_image_height = int(min_image_height)
        self.min_blur_score = float(min_blur_score)
        if selector is not None and not selector.strip():
            raise ScraperError("selector must not be empty")
        if selector and backend != "playwright-site":
            raise ScraperError("--selector is supported only by the playwright-site backend")
        if proxy_url and backend not in {"playwright-bing", "playwright-google"}:
            raise ScraperError("WireGuard proxy is supported only by Playwright backends")
        self.output_dir = Path(output_dir).expanduser().resolve()
        self.cache_dir = Path(cache_dir).expanduser().resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        effective_user_agent = user_agent or DEFAULT_USER_AGENT

        if (date_from or date_to) and backend not in {"playwright-bing", "playwright-google"}:
            raise ScraperError(
                "date filters are only supported by the Playwright Bing backend "
                "or Playwright Google backend"
            )

        if backend == "google":
            self.scraper: Scraper = GoogleImagesScraper(
                timeout=timeout,
                retries=retries,
                delay=delay,
                user_agent=effective_user_agent,
                safe="active" if safe_search else "off",
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
                date_from=date_from,
                date_to=date_to,
                proxy_url=proxy_url,
                safe_search=safe_search,
            )
        elif backend == "playwright-google":
            self.scraper = PlaywrightImagesScraper(
                provider="google",
                timeout=timeout,
                retries=retries,
                delay=delay,
                max_scrolls=max_scrolls,
                date_from=date_from,
                date_to=date_to,
                proxy_url=proxy_url,
                safe_search=safe_search,
            )
        elif backend == "playwright-site":
            self.scraper = PlaywrightSiteScraper(
                timeout=timeout,
                retries=retries,
                delay=delay,
                max_scrolls=max_scrolls,
                selector=selector,
            )
        elif backend == "direct":
            self.scraper = DirectURLScraper()
        else:
            raise ScraperError(
                f"unknown scraper backend {backend!r}; expected "
                + ", ".join(SCRAPER_BACKENDS)
            )

        # Google source hosts may reject direct downloads. Cap retries only for this
        # backend; other backends keep the configured retry count.
        download_retries = 2 if backend == "playwright-google" else retries
        self.downloader = ImageDownloader(
            timeout=timeout,
            retries=download_retries,
            delay=delay,
            user_agent=effective_user_agent,
            proxy_url=proxy_url,
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
        search_limit = max(count, count * (10 if self.backend == "playwright-google" else 3))

        if progress is not None:
            progress(0, count, f"searching for up to {search_limit} candidates...")
        print(f"scrape: searching for up to {search_limit} candidates...", flush=True)

        if isinstance(self.scraper, PlaywrightImagesScraper):
            self.scraper.cancel_event = cancel_event

        try:
            candidates = self.scraper.search(query, limit=search_limit)
            for result in candidates:
                if isinstance(self.scraper, PlaywrightImagesScraper):
                    self._sync_playwright_session()
                elif self.backend == "playwright-site":
                    self._sync_site_cookies()
                if cancel_event is not None and cancel_event.is_set():
                    self.cancelled = True
                    break
                if result.image_url in self.archive or result.image_url in known_url_set:
                    continue

                metadata = result.metadata if isinstance(result.metadata, dict) else {}
                candidate_urls = [result.image_url]
                if self.backend in {"playwright-google", "playwright-bing"}:
                    fallback_url = metadata.get("thumbnail_url")
                    if (
                        isinstance(fallback_url, str)
                        and fallback_url.startswith(("http://", "https://"))
                        and fallback_url != result.image_url
                    ):
                        candidate_urls.append(fallback_url)

                referer = None
                if self.backend in {"playwright-google", "playwright-bing"}:
                    referer = result.page_url or (
                        "https://www.google.com/"
                        if self.backend == "playwright-google"
                        else "https://www.bing.com/"
                    )
                elif self.backend == "playwright-site":
                    referer = result.page_url or query
                download_options = {
                    "archive": self.archive,
                    "cancel_event": cancel_event,
                }
                if referer:
                    download_options["referer"] = referer

                downloaded = None
                candidate_errors: list[str] = []
                for candidate_index, candidate_url in enumerate(candidate_urls):
                    if candidate_url in self.archive or candidate_url in known_url_set:
                        continue
                    try:
                        downloaded = self.downloader.download(
                            candidate_url,
                            self.output_dir,
                            index=next_index,
                            **download_options,
                        )
                    except ScraperCancelledError:
                        self.cancelled = True
                        break
                    except ScraperError as exc:
                        candidate_errors.append(f"{candidate_url}: {exc}")
                        print(
                            f"scrape: skipped failed image: {candidate_url} :: {exc}",
                            flush=True,
                        )
                        continue
                    if downloaded is None:
                        continue

                    # Only original Google source downloads use this scrape-time
                    # quality gate. A cached Google/Bing thumbnail is already a
                    # provider-rendered fallback and must remain available.
                    is_google_thumbnail = bool(
                        metadata.get("google_thumbnail_fallback")
                    ) or candidate_index > 0
                    if (
                        self.backend == "playwright-google"
                        and candidate_index == 0
                        and not is_google_thumbnail
                    ):
                        try:
                            quality = evaluate_quality(
                                downloaded.path,
                                min_width=self.min_image_width,
                                min_height=self.min_image_height,
                                blur_threshold=self.min_blur_score,
                            )
                            rejection_reason = quality.reason if not quality.valid else None
                        except Exception as exc:
                            rejection_reason = f"image quality check failed: {exc}"

                        if rejection_reason:
                            try:
                                downloaded.path.unlink(missing_ok=True)
                            except OSError:
                                pass
                            try:
                                self.archive.discard(candidate_url)
                            except ScraperError as exc:
                                print(
                                    "scrape: warning: could not unarchive "
                                    f"rejected image URL: {exc}",
                                    flush=True,
                                )
                            candidate_errors.append(
                                f"{candidate_url}: quality filter rejected image: "
                                f"{rejection_reason}"
                            )
                            print(
                                "scrape: original Google image rejected by quality "
                                f"gate ({rejection_reason}); trying thumbnail fallback",
                                flush=True,
                            )
                            downloaded = None
                            continue

                    if candidate_index > 0:
                        print(
                            "scrape: original image was unavailable; accepted "
                            f"provider thumbnail fallback: {candidate_url}",
                            flush=True,
                        )
                    break

                if self.cancelled:
                    break
                if downloaded is None:
                    error_message = "; ".join(candidate_errors) or (
                        f"no downloadable candidate remained for {result.image_url}"
                    )
                    failures.append((result, error_message))
                    continue

                downloads.append(downloaded)
                next_index += 1
                message = f"downloaded {len(downloads)}/{count}"
                if progress is not None:
                    progress(len(downloads), count, message)
                print(f"scrape: {message}", flush=True)

                if len(downloads) >= count:
                    break
        except ScraperCancelledError:
            self.cancelled = True
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

    def _sync_playwright_session(self) -> None:
        """Reuse Playwright browser cookies and user-agent for image HTTP requests."""
        cookies = getattr(self.scraper, "cookies", ())
        for cookie in cookies:
            if not isinstance(cookie, dict):
                continue
            name, value = cookie.get("name"), cookie.get("value")
            domain = cookie.get("domain")
            if (
                not isinstance(name, str) or not name
                or not isinstance(value, str)
                or not isinstance(domain, str) or not domain
            ):
                continue
            options: dict[str, object] = {
                "domain": domain,
                "path": cookie.get("path") if isinstance(cookie.get("path"), str) else "/",
                "secure": bool(cookie.get("secure", False)),
            }
            expires = cookie.get("expires")
            if isinstance(expires, (int, float)) and expires > 0:
                options["expires"] = int(expires)
            try:
                self.downloader.session.cookies.set(name, value, **options)
            except (TypeError, ValueError):
                continue

        user_agent = getattr(self.scraper, "browser_user_agent", None)
        if isinstance(user_agent, str) and user_agent.strip():
            self.downloader.session.headers["User-Agent"] = user_agent.strip()

    def _sync_site_cookies(self) -> None:
        """Copy browser cookies into the per-run HTTP jar without persisting them."""
        for cookie in getattr(self.scraper, "cookies", ()):
            if not isinstance(cookie, dict):
                continue
            name, value = cookie.get("name"), cookie.get("value")
            if not isinstance(name, str) or not name or not isinstance(value, str):
                continue
            options: dict[str, object] = {
                "path": cookie.get("path") if isinstance(cookie.get("path"), str) else "/",
                "secure": bool(cookie.get("secure", False)),
            }
            domain = cookie.get("domain")
            if isinstance(domain, str) and domain:
                options["domain"] = domain
            expires = cookie.get("expires")
            if isinstance(expires, (int, float)) and expires > 0:
                options["expires"] = int(expires)
            self.downloader.session.cookies.set(name, value, **options)

    def _next_index(self) -> int:
        highest = 0
        for path in self.output_dir.iterdir():
            if not path.is_file():
                continue
            stem = path.stem
            if stem.isdigit():
                highest = max(highest, int(stem))
        return highest + 1
