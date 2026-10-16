from __future__ import annotations

import json
import os
from http.client import HTTPConnection, HTTPException
from pathlib import Path
from urllib.parse import urlsplit

from .base import ScraperError


DEFAULT_CDP_HOST = "127.0.0.1"
DEFAULT_CDP_PORT = 9222
DEFAULT_CDP_TIMEOUT = 5.0


class ExistingChromium:
    """Attach Playwright-Termux to an already-running Chromium profile.

    The pinned playwright-termux 2.0.0 Chromium wrapper always launches a
    temporary profile. Authenticated Google mode therefore uses its CDP
    primitives directly so the existing user profile is never copied,
    replaced, or deleted.
    """

    def __init__(
        self,
        *,
        user_data_dir: Path | str,
        host: str = DEFAULT_CDP_HOST,
        port: int = DEFAULT_CDP_PORT,
        timeout: float = DEFAULT_CDP_TIMEOUT,
    ) -> None:
        profile = Path(user_data_dir).expanduser().resolve()
        if not str(user_data_dir).strip():
            raise ValueError("user_data_dir must not be empty")
        if host not in {"127.0.0.1", "localhost", "::1"}:
            raise ValueError(
                "CDP host must be local; expected 127.0.0.1, localhost, or ::1"
            )
        if port < 1 or port > 65535:
            raise ValueError("CDP port must be between 1 and 65535")
        if timeout <= 0:
            raise ValueError("CDP timeout must be greater than zero")

        self.user_data_dir = profile
        self.host = host
        self.port = port
        self.timeout = timeout
        self.cdp = None
        self._pages = []

    @classmethod
    def from_environment(cls, *, timeout: float) -> "ExistingChromium":
        profile = os.environ.get("CHROME_USER_DATA_DIR", "").strip()
        if not profile:
            raise ValueError(
                "CHROME_USER_DATA_DIR must be set for authenticated Google mode"
            )

        raw_port = os.environ.get(
            "CHROME_DEBUG_PORT",
            str(DEFAULT_CDP_PORT),
        ).strip()
        try:
            port = int(raw_port)
        except ValueError as exc:
            raise ValueError(
                f"CHROME_DEBUG_PORT must be an integer, got {raw_port!r}"
            ) from exc

        return cls(
            user_data_dir=profile,
            port=port,
            timeout=min(max(timeout, 0.5), 5.0),
        )

    def connect(self) -> "ExistingChromium":
        if not self.user_data_dir.is_dir():
            raise ScraperError(
                "CHROME_USER_DATA_DIR does not exist or is not a directory"
            )

        websocket_url = self._discover_websocket_url()

        try:
            from playwright_termux.cdp import CDPConnection
        except ImportError as exc:
            raise ScraperError(
                "Playwright backend requires playwright-termux; "
                'install with python -m pip install ".[playwright]"'
            ) from exc

        try:
            self.cdp = CDPConnection(
                websocket_url,
                timeout=min(max(self.timeout, 0.5), 10.0),
            ).connect()
        except Exception as exc:
            raise ScraperError(
                "could not connect to the existing Chromium CDP session: "
                f"{exc}"
            ) from exc

        return self

    def new_page(self):
        if self.cdp is None:
            raise ScraperError("existing Chromium is not connected")

        try:
            from playwright_termux.page import Page

            target = self.cdp.send(
                "Target.createTarget",
                {"url": "about:blank"},
            )
            target_id = target.get("targetId")
            if not target_id:
                raise RuntimeError("Chromium did not return a targetId")

            session = self.cdp.create_session(target_id)
            page = Page(self, session)
        except ScraperError:
            raise
        except Exception as exc:
            raise ScraperError(
                "could not create a page in the existing Chromium session: "
                f"{exc}"
            ) from exc

        self._pages.append(page)
        return page

    def _discover_websocket_url(self) -> str:
        display_host = f"[{self.host}]" if ":" in self.host else self.host
        endpoint = f"http://{display_host}:{self.port}/json/version"
        connection = HTTPConnection(
            self.host,
            self.port,
            timeout=min(max(self.timeout, 0.5), 5.0),
        )
        try:
            # Use a direct loopback HTTP connection instead of urllib, which
            # follows redirects by default. CDP discovery must never redirect
            # this local request to a different destination.
            connection.request(
                "GET",
                "/json/version",
                headers={"Connection": "close"},
            )
            response = connection.getresponse()
            if response.status != 200:
                raise ScraperError(
                    f"Chromium CDP endpoint {endpoint} returned HTTP {response.status}; "
                    "redirects and non-success responses are not accepted"
                )

            # The metadata document is tiny. Bound reads in case another local
            # service is listening on the configured debug port.
            raw_payload = response.read(1024 * 1024 + 1)
            if len(raw_payload) > 1024 * 1024:
                raise ScraperError(
                    f"Chromium CDP endpoint {endpoint} returned metadata larger than 1 MiB"
                )
            payload = json.loads(raw_payload.decode("utf-8"))
        except ScraperError:
            raise
        except (OSError, HTTPException, TypeError, ValueError, UnicodeError) as exc:
            raise ScraperError(
                f"Chromium CDP is unavailable at {endpoint}. "
                "Start the logged-in Chromium profile with "
                f"--remote-debugging-port={self.port}: {exc}"
            ) from exc
        finally:
            connection.close()

        websocket_url = (
            payload.get("webSocketDebuggerUrl")
            if isinstance(payload, dict)
            else None
        )
        if not isinstance(websocket_url, str):
            raise ScraperError(
                f"Chromium CDP endpoint {endpoint} did not provide "
                "a valid webSocketDebuggerUrl"
            )

        try:
            parsed = urlsplit(websocket_url)
            websocket_host = (parsed.hostname or "").lower().rstrip(".")
            websocket_port = parsed.port
        except ValueError as exc:
            raise ScraperError(
                f"Chromium CDP endpoint {endpoint} returned an invalid "
                "webSocketDebuggerUrl"
            ) from exc

        if (
            parsed.scheme not in {"ws", "wss"}
            or websocket_host not in {"127.0.0.1", "localhost", "::1"}
            or websocket_port != self.port
            or parsed.username is not None
            or parsed.password is not None
        ):
            raise ScraperError(
                f"Chromium CDP endpoint {endpoint} returned a non-local "
                "webSocketDebuggerUrl"
            )

        return websocket_url

    def close(self) -> None:
        for page in reversed(self._pages):
            try:
                page.close()
            except Exception:
                pass
        self._pages.clear()

        if self.cdp is not None:
            try:
                self.cdp.close()
            except Exception:
                pass
            self.cdp = None

    def __enter__(self) -> "ExistingChromium":
        return self.connect()

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
