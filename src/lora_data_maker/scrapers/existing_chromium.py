from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
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
        auto_launch: bool = False,
        executable: str | None = None,
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
        self.auto_launch = auto_launch
        self.executable = executable
        self.process = None
        self.cdp = None
        self._pages = []

    @classmethod
    def from_environment(cls, *, timeout: float) -> "ExistingChromium":
        profile = os.environ.get("CHROME_USER_DATA_DIR", "").strip()
        if not profile:
            data_home = os.environ.get("XDG_DATA_HOME", "").strip()
            base_dir = (
                Path(data_home).expanduser()
                if data_home
                else Path.home() / ".local" / "share"
            )
            profile = str(base_dir / "lora-data-maker" / "google-profile")

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

        auto_launch_value = os.environ.get("CHROME_AUTO_LAUNCH", "1").strip().lower()
        auto_launch = auto_launch_value not in {"0", "false", "no", "off"}
        executable = os.environ.get("CHROME_EXECUTABLE", "").strip() or None

        return cls(
            user_data_dir=profile,
            port=port,
            timeout=min(max(timeout, 0.5), 5.0),
            auto_launch=auto_launch,
            executable=executable,
        )

    def connect(self) -> "ExistingChromium":
        profile_existed = self.user_data_dir.is_dir()
        if not profile_existed and not self.auto_launch:
            raise ScraperError(
                "CHROME_USER_DATA_DIR does not exist or is not a directory"
            )

        if self.auto_launch:
            try:
                self.user_data_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
                if os.name == "posix":
                    self.user_data_dir.chmod(0o700)
            except OSError as exc:
                raise ScraperError(
                    f"could not prepare the private Google browser profile: {exc}"
                ) from exc

        try:
            websocket_url = self._discover_websocket_url()
        except ScraperError as exc:
            if not self.auto_launch or "CDP is unavailable at" not in str(exc):
                raise
            self._launch_browser()
            websocket_url = self._wait_for_websocket_url()
        else:
            if self.auto_launch and not profile_existed:
                raise ScraperError(
                    f"a CDP server is already listening on port {self.port}, "
                    "but the configured Google profile did not exist yet. "
                    "Refusing to attach to an unrelated browser. Stop that "
                    "debugging session or set CHROME_DEBUG_PORT to a free port."
                )

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
            if not (self.auto_launch and self._is_cdp_origin_rejection(exc)):
                raise ScraperError(
                    "could not connect to the existing Chromium CDP session: "
                    f"{exc}"
                ) from exc

            # A Chromium instance launched before the origin allowlist was
            # added rejects the CDP WebSocket handshake. Restart only the
            # browser using this dedicated profile and local debugging port.
            try:
                self._restart_browser_for_cdp_origin()
                websocket_url = self._wait_for_websocket_url()
                self.cdp = CDPConnection(
                    websocket_url,
                    timeout=min(max(self.timeout, 0.5), 10.0),
                ).connect()
            except Exception as retry_exc:
                raise ScraperError(
                    "Chromium rejected the CDP WebSocket origin. The scraper "
                    "tried to restart its dedicated browser with the required "
                    "origin flag, but reconnecting failed: "
                    f"{retry_exc}"
                ) from retry_exc

        return self

    @staticmethod
    def _is_cdp_origin_rejection(error: Exception) -> bool:
        message = str(error).lower()
        return "403 forbidden" in message and (
            "remote-allow-origins" in message
            or "rejected an incoming websocket connection" in message
        )

    def _matching_browser_pids(self, proc_root: Path = Path("/proc")) -> list[int]:
        """Find the Chromium parent launched for this exact profile and CDP port."""
        profile_arg = f"--user-data-dir={self.user_data_dir}"
        port_arg = f"--remote-debugging-port={self.port}"
        matches: list[int] = []
        try:
            process_entries = list(proc_root.iterdir())
        except OSError:
            return matches

        for entry in process_entries:
            if not entry.name.isdigit():
                continue
            try:
                raw = (entry / "cmdline").read_bytes()
            except OSError:
                continue
            argv = [part.decode("utf-8", errors="replace") for part in raw.split(b"\0") if part]
            if not argv or "chrom" not in Path(argv[0]).name.lower():
                continue
            # Only the parent browser owns the debugging listener. Never
            # signal renderer/zygote processes or a different browser profile.
            if any(arg.startswith("--type=") for arg in argv):
                continue
            if profile_arg in argv and port_arg in argv:
                matches.append(int(entry.name))
        return matches

    def _restart_browser_for_cdp_origin(self) -> None:
        """Restart only this profile when an old Chromium rejects the CDP origin."""
        import signal

        pids = self._matching_browser_pids()
        if not pids:
            raise ScraperError(
                "the current Chromium rejects the CDP origin, but the scraper "
                "could not identify its parent process for the configured profile. "
                "Close the Google scraping Chromium window once, then tap scrape again."
            )

        for pid in pids:
            try:
                os.kill(pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            except OSError as exc:
                raise ScraperError(
                    f"could not stop the old Google Chromium process {pid}: {exc}"
                ) from exc

        deadline = time.monotonic() + 8.0
        while time.monotonic() < deadline:
            if not self._matching_browser_pids():
                break
            time.sleep(0.2)
        else:
            raise ScraperError(
                "the old Google Chromium process did not close cleanly. "
                "Close the scraping browser once, then tap scrape again."
            )

        self.process = None
        self._launch_browser()
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

    @staticmethod
    def _x11_socket_path(display: str) -> Path:
        """Return the Termux X11 Unix socket path for a DISPLAY value."""
        display_number = display.rsplit(":", 1)[-1].split(".", 1)[0]
        if not display_number.isdigit():
            raise ScraperError(
                f"unsupported Termux:X11 display {display!r}; use a value such as ':1'"
            )
        tmpdir = os.environ.get("TMPDIR", "").strip() or "/tmp"
        return Path(tmpdir) / ".X11-unix" / f"X{display_number}"

    def _ensure_display(self) -> str:
        """Start Termux:X11 and open its Android activity on first Google scrape."""
        display = os.environ.get("DISPLAY", "").strip()
        if display:
            return display

        display = os.environ.get("LORA_DATA_MAKER_X11_DISPLAY", ":1").strip() or ":1"
        socket_path = self._x11_socket_path(display)
        termux_x11 = shutil.which("termux-x11")
        if not termux_x11:
            raise ScraperError(
                "Google scraping needs Termux:X11, but termux-x11 was not found. "
                "Install the Termux:X11 Android app and its Termux package "
                "(pkg install x11-repo termux-x11-nightly), then tap scrape again."
            )

        am = shutil.which("am")
        if not am and Path("/system/bin/am").is_file():
            am = "/system/bin/am"
        if not am:
            raise ScraperError(
                "could not find Android am to open Termux:X11. "
                "Run the WebUI from the Termux app, not from a Linux container."
            )

        try:
            activity = subprocess.run(
                [
                    am,
                    "start",
                    "--user",
                    "0",
                    "-n",
                    "com.termux.x11/com.termux.x11.MainActivity",
                ],
                check=False,
                capture_output=True,
                text=True,
                timeout=10,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            raise ScraperError(
                f"could not open the Termux:X11 Android app automatically: {exc}"
            ) from exc
        if activity.returncode != 0:
            details = (activity.stderr or activity.stdout or "").strip()
            hint = f" Details: {details}" if details else ""
            raise ScraperError(
                "Android could not open Termux:X11. Make sure the app is installed "
                "and launchable, then retry scrape." + hint
            )

        process = None
        if not socket_path.exists():
            env = os.environ.copy()
            env["DISPLAY"] = display
            try:
                process = subprocess.Popen(
                    [termux_x11, display],
                    env=env,
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    start_new_session=True,
                )
            except OSError as exc:
                raise ScraperError(
                    f"could not start the Termux:X11 display server: {exc}"
                ) from exc

            deadline = time.monotonic() + 10.0
            while time.monotonic() < deadline and not socket_path.exists():
                if process.poll() is not None:
                    raise ScraperError(
                        "Termux:X11 exited before its display socket became ready. "
                        "Check that the Termux:X11 Android app and "
                        "termux-x11-nightly package are both installed."
                    )
                time.sleep(0.2)

        if not socket_path.exists():
            raise ScraperError(
                f"Termux:X11 did not become ready on {display} within 10 seconds. "
                "Keep the Termux:X11 app installed and retry scrape."
            )

        # The WebUI server may have started before X11; update its process
        # environment before Chromium is launched in this scrape worker.
        os.environ["DISPLAY"] = display
        return display

    def _launch_browser(self) -> None:
        """Start visible Termux Chromium with a persistent, local-only CDP profile."""
        self._ensure_display()

        executable = (
            self.executable
            or shutil.which("chromium-browser")
            or shutil.which("chromium")
        )
        if not executable:
            raise ScraperError(
                "could not find Termux Chromium. Install it with "
                "`pkg install x11-repo chromium`."
            )

        try:
            self.user_data_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
            if os.name == "posix":
                self.user_data_dir.chmod(0o700)
        except OSError as exc:
            raise ScraperError(
                f"could not prepare the private Google browser profile: {exc}"
            ) from exc

        args = [
            executable,
            f"--user-data-dir={self.user_data_dir}",
            f"--remote-debugging-port={self.port}",
            "--remote-debugging-address=127.0.0.1",
            f"--remote-allow-origins=http://127.0.0.1:{self.port}",
            "--no-first-run",
            "--no-default-browser-check",
            "--new-window",
            "https://www.google.com/",
        ]
        try:
            self.process = subprocess.Popen(
                args,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
        except OSError as exc:
            raise ScraperError(
                f"could not launch Termux Chromium for Google Images: {exc}"
            ) from exc

    def _wait_for_websocket_url(self) -> str:
        deadline = time.monotonic() + 20.0
        last_error: ScraperError | None = None
        while time.monotonic() < deadline:
            try:
                return self._discover_websocket_url()
            except ScraperError as exc:
                if "CDP is unavailable at" not in str(exc):
                    raise
                last_error = exc
                if self.process is not None and self.process.poll() is not None:
                    raise ScraperError(
                        "Chromium exited before its local CDP endpoint became "
                        "ready. The profile may already be open in a Chromium "
                        "process that was not started with remote debugging; "
                        "close that browser and retry."
                    ) from exc
                time.sleep(0.25)

        raise ScraperError(
            f"Chromium did not expose CDP on 127.0.0.1:{self.port} within "
            "20 seconds. Check Termux:X11, DISPLAY, and whether another "
            "process is using CHROME_DEBUG_PORT."
        ) from last_error

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
