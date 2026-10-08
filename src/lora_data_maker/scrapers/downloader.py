from __future__ import annotations

import hashlib
import mimetypes
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

import requests
import urllib3.util.connection as urllib3_connection

from .archive import DownloadArchive
from .base import ScraperCancelledError, ScraperError


def _is_termux_environment() -> bool:
    """Return True when running inside a native Termux environment."""
    if os.environ.get("TERMUX__ROOTFS") or os.environ.get("TERMUX_VERSION"):
        return True
    return "/data/data/com.termux/files/" in os.environ.get("PREFIX", "")


def _configure_termux_network() -> None:
    """Prefer IPv4 in Termux where IPv6 paths can stall HTTP connects."""
    if _is_termux_environment():
        urllib3_connection.HAS_IPV6 = False




def _retryable_request_error(exc: requests.RequestException) -> bool:
    response = getattr(exc, "response", None)
    status = getattr(response, "status_code", None)
    if status is None:
        return True
    return status == 408 or status == 429 or status >= 500

IMAGE_MIME_TYPES = {
    "image/jpeg": ".jpg",
    "image/png": ".png",
    "image/webp": ".webp",
    "image/gif": ".gif",
    "image/bmp": ".bmp",
    "image/tiff": ".tiff",
    "image/avif": ".avif",
}


@dataclass(frozen=True)
class DownloadResult:
    url: str
    path: Path
    sha256: str
    file_size: int
    content_type: str | None


class ImageDownloader:
    def __init__(
        self,
        *,
        timeout: float = 20.0,
        retries: int = 3,
        delay: float = 1.0,
        user_agent: str,
        session: requests.Session | None = None,
    ) -> None:
        if timeout <= 0:
            raise ValueError("timeout must be greater than zero")
        if retries < 0:
            raise ValueError("retries cannot be negative")
        if delay < 0:
            raise ValueError("delay cannot be negative")

        self.timeout = timeout
        self.retries = retries
        self.delay = delay
        self.session = session or requests.Session()
        self.session.headers.update({"User-Agent": user_agent})
        _configure_termux_network()

    def download(
        self,
        url: str,
        output_dir: Path | str,
        *,
        index: int,
        archive: DownloadArchive | None = None,
        cancel_event: threading.Event | None = None,
    ) -> DownloadResult | None:
        if cancel_event is not None and cancel_event.is_set():
            raise ScraperCancelledError("scrape cancelled")
        if archive is not None and url in archive:
            return None

        output_path: Path | None = None
        response: requests.Response | None = None
        last_error: Exception | None = None

        for attempt in range(self.retries + 1):
            try:
                if cancel_event is not None and cancel_event.is_set():
                    raise ScraperCancelledError("scrape cancelled")
                request_timeout: float | tuple[float, float]
                if _is_termux_environment():
                    request_timeout = (min(self.timeout, 8.0), self.timeout)
                else:
                    request_timeout = self.timeout

                response = self.session.get(
                    url,
                    timeout=request_timeout,
                    allow_redirects=True,
                    stream=True,
                )
                response.raise_for_status()

                content_type = self._normalize_content_type(
                    response.headers.get("Content-Type")
                )
                if content_type and not content_type.startswith("image/"):
                    raise ScraperError(
                        f"URL did not return an image: {url} "
                        f"(content-type: {content_type})"
                    )

                output_dir_path = Path(output_dir).expanduser().resolve()
                output_dir_path.mkdir(parents=True, exist_ok=True)

                suffix = self._choose_suffix(url, content_type)
                output_path = output_dir_path / f"{index:06d}{suffix}"
                temp_path = output_path.with_name(f".{output_path.name}.part")

                digest = hashlib.sha256()
                total = 0
                with temp_path.open("wb") as handle:
                    for chunk in response.iter_content(chunk_size=1024 * 128):
                        if cancel_event is not None and cancel_event.is_set():
                            raise ScraperCancelledError("scrape cancelled")
                        if not chunk:
                            continue
                        handle.write(chunk)
                        digest.update(chunk)
                        total += len(chunk)

                temp_path.replace(output_path)

                if archive is not None:
                    archive.add(url)

                if self.delay:
                    if cancel_event is not None:
                        cancel_event.wait(self.delay)
                    else:
                        time.sleep(self.delay)

                return DownloadResult(
                    url=url,
                    path=output_path,
                    sha256=digest.hexdigest(),
                    file_size=total,
                    content_type=content_type,
                )
            except ScraperCancelledError:
                if output_path is not None:
                    part_path = output_path.with_name(
                        f".{output_path.name}.part"
                    )
                    try:
                        part_path.unlink(missing_ok=True)
                        output_path.unlink(missing_ok=True)
                    except OSError:
                        pass
                raise
            except (requests.RequestException, OSError, ScraperError) as exc:
                last_error = exc
                if output_path is not None:
                    part_path = output_path.with_name(
                        f".{output_path.name}.part"
                    )
                    try:
                        part_path.unlink(missing_ok=True)
                        output_path.unlink(missing_ok=True)
                    except OSError:
                        pass
                if attempt >= self.retries:
                    break
                if isinstance(exc, ScraperError):
                    break
                if isinstance(exc, requests.RequestException) and not _retryable_request_error(exc):
                    break
                backoff = min(max(self.delay, 0.25) * (2**attempt), 30.0)
                if cancel_event is not None:
                    if cancel_event.wait(backoff):
                        raise ScraperCancelledError("scrape cancelled")
                else:
                    time.sleep(backoff)
            finally:
                if response is not None:
                    response.close()
                    response = None

        raise ScraperError(
            f"download failed after {self.retries + 1} attempts: {url}: {last_error}"
        ) from last_error

    @staticmethod
    def _normalize_content_type(value: str | None) -> str | None:
        if not value:
            return None
        return value.split(";", 1)[0].strip().lower()

    @staticmethod
    def _choose_suffix(url: str, content_type: str | None) -> str:
        if content_type in IMAGE_MIME_TYPES:
            return IMAGE_MIME_TYPES[content_type]

        suffix = Path(urlparse(url).path).suffix.lower()
        if suffix in {
            ".jpg",
            ".jpeg",
            ".png",
            ".webp",
            ".gif",
            ".bmp",
            ".tif",
            ".tiff",
            ".avif",
        }:
            return ".jpg" if suffix == ".jpeg" else suffix

        guessed = mimetypes.guess_extension(content_type or "")
        allowed = {
            ".jpg",
            ".png",
            ".webp",
            ".gif",
            ".bmp",
            ".tif",
            ".tiff",
            ".avif",
        }
        return guessed if guessed in allowed else ".img"
