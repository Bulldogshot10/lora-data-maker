from __future__ import annotations

import re
import time
from typing import Iterator

import requests

from .base import ImageResult, ScraperError


DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Linux; Android 14; Mobile) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/154.0.0.0 Mobile Safari/537.36"
)


class DuckDuckGoImagesScraper:
    """Credential-free DuckDuckGo Images scraper using its JSON image endpoint."""

    SITE_URL = "https://duckduckgo.com/"
    IMAGES_URL = "https://duckduckgo.com/i.js"

    def __init__(
        self,
        *,
        timeout: float = 20.0,
        retries: int = 3,
        delay: float = 1.0,
        user_agent: str = DEFAULT_USER_AGENT,
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
        self.session.headers.update(
            {
                "User-Agent": user_agent,
                "Accept": "application/json,text/javascript,*/*;q=0.8",
            }
        )

    def search(self, query: str, *, limit: int) -> Iterator[ImageResult]:
        if not query.strip():
            raise ScraperError("DuckDuckGo Images query cannot be empty")
        if limit <= 0:
            return

        vqd = self._get_vqd(query)
        seen: set[str] = set()
        yielded = 0
        offset = 0

        while yielded < limit:
            payload = self._fetch_results(query, vqd, offset)
            results = payload.get("results", [])
            if not isinstance(results, list) or not results:
                break

            added = 0
            for item in results:
                if not isinstance(item, dict):
                    continue
                image_url = item.get("image")
                if not isinstance(image_url, str) or not image_url.startswith(("http://", "https://")):
                    continue
                if image_url in seen:
                    continue

                seen.add(image_url)
                yielded += 1
                added += 1
                yield ImageResult(
                    image_url=image_url,
                    query=query,
                    page_url=item.get("url") if isinstance(item.get("url"), str) else None,
                    rank=yielded,
                    metadata={
                        "thumbnail": item.get("thumbnail"),
                        "title": item.get("title"),
                        "source": item.get("source"),
                        "width": item.get("width"),
                        "height": item.get("height"),
                        "offset": offset,
                    },
                )
                if yielded >= limit:
                    break

            batch_size = len(results)
            if added == 0 or batch_size <= 0:
                break
            offset += batch_size

            if self.delay:
                time.sleep(self.delay)

    def _get_vqd(self, query: str) -> str:
        params = {
            "q": query,
            "iax": "images",
            "ia": "images",
        }
        body = self._request(self.SITE_URL, params=params, referer=self.SITE_URL)
        match = re.search(r"""vqd[='"]+\s*[:=]\s*['"]([^'"]+)['"]""", body)
        if not match:
            match = re.search(r"""vqd=['"]([^'"]+)['"]""", body)
        if not match:
            raise ScraperError("DuckDuckGo Images did not return a usable vqd token")
        return match.group(1)

    def _fetch_results(
        self,
        query: str,
        vqd: str,
        offset: int,
    ) -> dict[str, object]:
        params = {
            "q": query,
            "o": "json",
            "l": "us-en",
            "vqd": vqd,
            "f": ",,,",
            "p": "1",
            "s": str(offset),
        }
        body = self._request(
            self.IMAGES_URL,
            params=params,
            referer=self.SITE_URL,
            accept_json=True,
        )
        try:
            payload = requests.models.complexjson.loads(body)
        except ValueError as exc:
            raise ScraperError("DuckDuckGo Images returned invalid JSON") from exc
        if not isinstance(payload, dict):
            raise ScraperError("DuckDuckGo Images returned an unexpected response")
        return payload

    def _request(
        self,
        url: str,
        *,
        params: dict[str, str],
        referer: str,
        accept_json: bool = False,
    ) -> str:
        last_error: Exception | None = None
        headers = {
            "Referer": referer,
            "X-Requested-With": "XMLHttpRequest",
        }
        if accept_json:
            headers["Accept"] = "application/json,text/javascript,*/*;q=0.8"

        for attempt in range(self.retries + 1):
            try:
                response = self.session.get(
                    url,
                    params=params,
                    headers=headers,
                    timeout=(min(self.timeout, 8.0), self.timeout),
                )
                response.raise_for_status()
                return response.text
            except requests.RequestException as exc:
                last_error = exc
                if attempt >= self.retries:
                    break
                time.sleep(min(max(self.delay, 0.25) * (2**attempt), 30.0))

        raise ScraperError(
            f"DuckDuckGo request failed after {self.retries + 1} attempts: {last_error}"
        ) from last_error
