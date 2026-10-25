from __future__ import annotations

import json
import os
import time
from pathlib import Path
from urllib.parse import urljoin, urlparse

from .base import ImageResult, ScraperError


DEFAULT_SITE_URL = "https://www.google.com/"


def site_profile_dir() -> Path:
    """Return the dedicated persistent profile used for authenticated websites."""
    configured = os.environ.get("CHROME_SITE_USER_DATA_DIR", "").strip()
    if configured:
        return Path(configured).expanduser().resolve()
    data_home = os.environ.get("XDG_DATA_HOME", "").strip()
    root = Path(data_home).expanduser() if data_home else Path.home() / ".local" / "share"
    return (root / "lora-data-maker" / "playwright-site-profile").resolve()


def site_debug_port() -> int:
    raw = os.environ.get("CHROME_SITE_DEBUG_PORT", "9224").strip()
    try:
        port = int(raw)
    except ValueError as exc:
        raise ScraperError(
            f"CHROME_SITE_DEBUG_PORT must be an integer, got {raw!r}"
        ) from exc
    if not 1 <= port <= 65535:
        raise ScraperError("CHROME_SITE_DEBUG_PORT must be between 1 and 65535")
    return port


def create_site_browser(
    *,
    timeout: float = 5.0,
    auto_launch: bool = False,
    headless: bool = False,
):
    """Create the CDP adapter for the isolated, persistent website profile."""
    from .existing_chromium import ExistingChromium

    return ExistingChromium(
        user_data_dir=site_profile_dir(),
        port=site_debug_port(),
        timeout=min(max(float(timeout), 0.5), 5.0),
        auto_launch=auto_launch,
        executable=os.environ.get("CHROME_EXECUTABLE", "").strip() or None,
        initial_url="about:blank",
        headless=headless,
        shutdown_on_close=False,
    )


def validate_page_url(value: str) -> str:
    """Validate a user-supplied page URL before passing it to Chromium."""
    url = (value or "").strip()
    parsed = urlparse(url)
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
    ):
        raise ScraperError(
            "playwright-site requires a full http:// or https:// page URL "
            "without embedded credentials"
        )
    return url


def open_login_browser(url: str = DEFAULT_SITE_URL, *, timeout: float = 5.0) -> Path:
    """Start/attach to the visible site browser and open a tab without scraping."""
    target_url = validate_page_url(url)
    browser = create_site_browser(timeout=timeout, auto_launch=True, headless=False)
    try:
        with browser:
            if browser.cdp is None:
                raise ScraperError("could not connect to the website browser session")
            browser.cdp.send("Target.createTarget", {"url": target_url})
    except ScraperError:
        raise
    except Exception as exc:
        raise ScraperError(f"could not open the website login browser: {exc}") from exc
    return site_profile_dir()


class PlaywrightSiteScraper:
    """Capture image elements from an authenticated, user-controlled Chromium page."""

    def __init__(
        self,
        *,
        timeout: float = 20.0,
        retries: int = 2,
        delay: float = 1.0,
        max_scrolls: int = 30,
        selector: str | None = None,
    ) -> None:
        if timeout <= 0:
            raise ValueError("timeout must be greater than zero")
        if retries < 0:
            raise ValueError("retries cannot be negative")
        if delay < 0:
            raise ValueError("delay cannot be negative")
        if max_scrolls <= 0:
            raise ValueError("max_scrolls must be greater than zero")
        if selector is not None and not selector.strip():
            raise ValueError("selector must not be empty")
        self.timeout = float(timeout)
        self.retries = int(retries)
        self.delay = float(delay)
        self.max_scrolls = int(max_scrolls)
        self.selector = selector.strip() if selector else "img"
        self.cookies: list[dict[str, object]] = []

    @staticmethod
    def _image_expression(selector: str) -> str:
        """Build a page-side extractor. Uses an invoked IIFE for playwright-termux."""
        return """(() => {
          const selector = """ + json.dumps(selector) + """;
          let nodes;
          try {
            nodes = Array.from(document.querySelectorAll(selector));
          } catch (error) {
            return {error: String(error)};
          }
          const imageExtensions = /\\.(?:avif|bmp|gif|jpe?g|png|tiff?|webp)(?:$|[?#])/i;
          const isHttpImage = (value) => {
            if (!value || typeof value !== "string") return false;
            try {
              const parsed = new URL(value, location.href);
              return (parsed.protocol === "http:" || parsed.protocol === "https:")
                && !parsed.username && !parsed.password;
            } catch (_) { return false; }
          };
          const largestSrcset = (value) => {
            if (!value) return "";
            const choices = value.split(",").map((part) => {
              const match = part.trim().match(/^(\\S+)\\s+(\\d+(?:\\.\\d+)?)(w|x)?$/);
              return match ? {url: match[1], size: Number(match[2])} : null;
            }).filter(Boolean).filter((item) => isHttpImage(item.url));
            choices.sort((a, b) => b.size - a.size);
            return choices.length ? choices[0].url : "";
          };
          return nodes.map((node, index) => {
            const tag = (node.tagName || "").toLowerCase();
            const image = tag === "img" ? node : (node.querySelector ? node.querySelector("img") : null);
            if (!image) return null;
            const anchor = (node.closest && node.closest("a[href]"))
              || (image.closest && image.closest("a[href]"));
            const attr = (name) => image.getAttribute(name) || "";
            const preferred = [
              attr("data-original-src"), attr("data-full-src"), attr("data-original"),
              attr("data-full"), attr("data-image"), attr("data-url"), attr("data-src"),
              largestSrcset(attr("data-srcset")), largestSrcset(attr("srcset"))
            ];
            const href = anchor ? anchor.href : "";
            let url = preferred.find(isHttpImage) || "";
            if (!url && href && imageExtensions.test(href) && isHttpImage(href)) url = href;
            if (!url) url = [image.currentSrc, image.src, attr("src")].find(isHttpImage) || "";
            if (!url) url = largestSrcset(attr("srcset"));
            if (!url && href && isHttpImage(href) && imageExtensions.test(href)) url = href;
            if (!isHttpImage(url)) return null;
            return {
              url: new URL(url, location.href).href,
              alt: image.alt || image.getAttribute("aria-label") || "",
              width: Number(image.naturalWidth || image.width || 0),
              height: Number(image.naturalHeight || image.height || 0),
              index
            };
          }).filter((item) => item && item.url && item.width >= 64 && item.height >= 64);
        })()"""

    @staticmethod
    def _normalize_candidate(raw: object, base_url: str) -> tuple[str, dict[str, object]] | None:
        if not isinstance(raw, dict):
            return None
        candidate = raw.get("url")
        if not isinstance(candidate, str) or not candidate.strip():
            return None
        url = urljoin(base_url, candidate.strip())
        parsed = urlparse(url)
        if (
            parsed.scheme not in {"http", "https"}
            or not parsed.hostname
            or parsed.username is not None
            or parsed.password is not None
        ):
            return None
        width = raw.get("width", 0)
        height = raw.get("height", 0)
        alt = raw.get("alt", "")
        metadata: dict[str, object] = {
            "image_source": "playwright-site-dom",
            "image_width": int(width) if isinstance(width, (int, float)) else 0,
            "image_height": int(height) if isinstance(height, (int, float)) else 0,
            "alt": alt[:500] if isinstance(alt, str) else "",
        }
        return url, metadata

    @staticmethod
    def _read_cookies(page) -> list[dict[str, object]]:
        """Read live cookies from CDP into memory only; never write them to disk."""
        try:
            response = page.session.send("Network.getAllCookies", {})
        except Exception:
            return []
        values = response.get("cookies", []) if isinstance(response, dict) else []
        result: list[dict[str, object]] = []
        for item in values:
            if not isinstance(item, dict):
                continue
            name, value = item.get("name"), item.get("value")
            if not isinstance(name, str) or not name or not isinstance(value, str):
                continue
            result.append({
                "name": name,
                "value": value,
                "domain": item.get("domain", ""),
                "path": item.get("path", "/"),
                "secure": bool(item.get("secure", False)),
                "expires": item.get("expires"),
            })
        return result

    def _navigate(self, page, url: str) -> None:
        try:
            response = page.session.send(
                "Page.navigate",
                {"url": url},
                timeout=min(self.timeout, 5.0),
            )
            if isinstance(response, dict) and response.get("errorText"):
                raise ScraperError(
                    f"Chromium navigation failed: {response['errorText']}"
                )
        except ScraperError:
            raise
        except Exception as exc:
            if "Timeout waiting for CDP command: Page.navigate" not in str(exc):
                raise ScraperError(f"could not navigate to the requested page: {exc}") from exc
            print(
                "scrape: Playwright site navigation acknowledgement timed out; "
                "waiting for the live page state",
                flush=True,
            )

        deadline = time.monotonic() + min(self.timeout, 12.0)
        while time.monotonic() < deadline:
            try:
                ready = page.evaluate("document.readyState")
                if ready in {"interactive", "complete"}:
                    return
            except Exception:
                pass
            page.wait_for_timeout(200)
        # Some SPA pages keep transitioning and never settle at "complete".
        # The extractor below can still read their current DOM.

    def _capture(self, url: str, limit: int) -> list[ImageResult]:
        browser = create_site_browser(timeout=self.timeout, auto_launch=False, headless=False)
        results: list[ImageResult] = []
        seen: set[str] = set()
        with browser:
            page = browser.new_page()
            try:
                try:
                    page.bring_to_front()
                except Exception:
                    pass
                self._navigate(page, url)
                current_url = url
                try:
                    actual = page.evaluate("location.href")
                    if isinstance(actual, str) and actual:
                        current_url = actual
                except Exception:
                    pass
                self.cookies = self._read_cookies(page)
                print(
                    f"scrape: Playwright site session ready; "
                    f"{len(self.cookies)} browser cookie(s) available in memory",
                    flush=True,
                )

                stagnant_rounds = 0
                previous_count = 0
                for scroll_no in range(1, self.max_scrolls + 1):
                    raw_candidates = page.evaluate(self._image_expression(self.selector))
                    if isinstance(raw_candidates, dict) and raw_candidates.get("error"):
                        raise ScraperError(
                            f"invalid image selector {self.selector!r}: "
                            f"{raw_candidates['error']}"
                        )
                    if not isinstance(raw_candidates, list):
                        raw_candidates = []

                    for raw in raw_candidates:
                        normalized = self._normalize_candidate(raw, current_url)
                        if normalized is None:
                            continue
                        image_url, metadata = normalized
                        if image_url in seen:
                            continue
                        seen.add(image_url)
                        results.append(
                            ImageResult(
                                image_url=image_url,
                                query=url,
                                page_url=current_url,
                                rank=len(results) + 1,
                                metadata={
                                    **metadata,
                                    "scroll": scroll_no,
                                    "selector": self.selector,
                                },
                            )
                        )
                        if len(results) >= limit:
                            return results

                    print(
                        f"scrape: playwright-site scroll={scroll_no}/{self.max_scrolls} "
                        f"images={len(results)}",
                        flush=True,
                    )
                    if len(results) == previous_count:
                        stagnant_rounds += 1
                    else:
                        stagnant_rounds = 0
                    previous_count = len(results)
                    if stagnant_rounds >= 3:
                        break

                    page.evaluate(
                        "window.scrollTo(0, Math.max(window.scrollY + "
                        "Math.max(window.innerHeight || 0, 600), "
                        "document.documentElement?.scrollHeight || 0))"
                    )
                    if self.delay:
                        page.wait_for_timeout(int(min(self.delay, 5.0) * 1000))
            finally:
                try:
                    page.close()
                except Exception:
                    pass
        return results

    def search(self, query: str, *, limit: int):
        if limit <= 0:
            return
        url = validate_page_url(query)
        last_error: Exception | None = None
        for attempt in range(self.retries + 1):
            try:
                results = self._capture(url, limit)
                if results:
                    yield from results
                    return
                last_error = ScraperError(
                    f"no image elements matched selector {self.selector!r} "
                    f"on {url}; verify login or pass --selector"
                )
            except ScraperError as exc:
                last_error = exc
            if attempt < self.retries:
                print(
                    f"scrape: playwright-site retry={attempt + 1}/{self.retries}: "
                    f"{last_error}",
                    flush=True,
                )
                if self.delay:
                    time.sleep(min(self.delay * (2 ** attempt), 10.0))
        raise ScraperError(
            f"Playwright site capture failed after {self.retries + 1} attempt(s): "
            f"{last_error}"
        ) from last_error
