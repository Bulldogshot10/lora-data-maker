from __future__ import annotations

import base64
import binascii
import ipaddress
import json
import os
import re
import shutil
import socket
import subprocess
import tempfile
import time
from pathlib import Path
from urllib.parse import urlsplit

from .scrapers.base import ScraperCancelledError


MAX_WIREGUARD_CONFIG_BYTES = 64 * 1024
WIREPROXY_START_TIMEOUT = 12.0


def _decode_key(value: str, field: str) -> None:
    try:
        decoded = base64.b64decode(value.strip(), validate=True)
    except (ValueError, binascii.Error) as exc:
        raise ValueError(f"WireGuard {field} must be a valid base64 key") from exc
    if len(decoded) != 32:
        raise ValueError(f"WireGuard {field} must decode to exactly 32 bytes")


def validate_wireguard_config(content: bytes | str) -> str:
    """Validate a client WireGuard config without ever echoing private key values."""
    if isinstance(content, bytes):
        if len(content) > MAX_WIREGUARD_CONFIG_BYTES:
            raise ValueError("WireGuard config exceeds the 64 KiB limit")
        try:
            text = content.decode("utf-8-sig")
        except UnicodeDecodeError as exc:
            raise ValueError("WireGuard config must be UTF-8 text") from exc
    elif isinstance(content, str):
        text = content
        if len(text.encode("utf-8")) > MAX_WIREGUARD_CONFIG_BYTES:
            raise ValueError("WireGuard config exceeds the 64 KiB limit")
    else:
        raise ValueError("WireGuard config must be text")

    if not text.strip():
        raise ValueError("WireGuard config is empty")
    if "\x00" in text or any(ord(char) < 32 and char not in "\r\n\t" for char in text):
        raise ValueError("WireGuard config contains unsupported control characters")

    sections: list[tuple[str, dict[str, str]]] = []
    current: dict[str, str] | None = None
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith(("#", ";")):
            continue
        line = re.split(r"\s+[;#]", line, maxsplit=1)[0].strip()
        if not line:
            continue
        if line.startswith("[") and line.endswith("]"):
            name = line[1:-1].strip().casefold()
            if name not in {"interface", "peer"}:
                raise ValueError(f"unsupported WireGuard section on line {line_number}")
            current = {}
            sections.append((name, current))
            continue
        if current is None or "=" not in line:
            raise ValueError(f"invalid WireGuard setting on line {line_number}")
        key, value = line.split("=", 1)
        key = key.strip().casefold()
        value = value.strip()
        if not key or not value:
            raise ValueError(f"empty WireGuard setting on line {line_number}")
        if key in current:
            raise ValueError(f"duplicate WireGuard setting on line {line_number}")
        current[key] = value

    interfaces = [settings for name, settings in sections if name == "interface"]
    peers = [settings for name, settings in sections if name == "peer"]
    if len(interfaces) != 1:
        raise ValueError("WireGuard config must contain exactly one [Interface] section")
    if not peers:
        raise ValueError("WireGuard config must contain at least one [Peer] section")

    interface = interfaces[0]
    private_key = interface.get("privatekey")
    if not private_key:
        raise ValueError("WireGuard [Interface] is missing PrivateKey")
    _decode_key(private_key, "PrivateKey")
    addresses = interface.get("address", "")
    if not addresses:
        raise ValueError("WireGuard [Interface] is missing Address")
    try:
        for address in addresses.split(","):
            ipaddress.ip_interface(address.strip())
    except ValueError as exc:
        raise ValueError("WireGuard Address must contain valid IP interface addresses") from exc

    dns_servers = interface.get("dns", "")
    if not dns_servers:
        raise ValueError(
            "WireGuard [Interface] is missing DNS; add your VPN provider's DNS IP "
            "to avoid using the device's system resolver"
        )
    try:
        for dns_server in dns_servers.split(","):
            ipaddress.ip_address(dns_server.strip())
    except ValueError as exc:
        raise ValueError("WireGuard DNS must contain IP addresses") from exc

    has_endpoint = False
    for peer in peers:
        public_key = peer.get("publickey")
        allowed_ips = peer.get("allowedips")
        if not public_key or not allowed_ips:
            raise ValueError("each WireGuard [Peer] requires PublicKey and AllowedIPs")
        _decode_key(public_key, "PublicKey")
        if peer.get("presharedkey"):
            _decode_key(peer["presharedkey"], "PresharedKey")
        try:
            for network in allowed_ips.split(","):
                ipaddress.ip_network(network.strip(), strict=False)
        except ValueError as exc:
            raise ValueError("WireGuard AllowedIPs must contain valid IP networks") from exc

        endpoint = peer.get("endpoint", "")
        if endpoint:
            try:
                parsed = urlsplit("//" + endpoint)
                port = parsed.port
            except ValueError as exc:
                raise ValueError("WireGuard Endpoint must be host:port") from exc
            if (
                not parsed.hostname
                or port is None
                or not 1 <= port <= 65535
                or parsed.username is not None
                or parsed.password is not None
                or parsed.path
                or parsed.query
                or parsed.fragment
            ):
                raise ValueError("WireGuard Endpoint must be host:port")
            has_endpoint = True
    if not has_endpoint:
        raise ValueError("WireGuard config needs a peer Endpoint for client VPN use")

    return text.rstrip() + "\n"


class WireGuardConfigStore:
    """Keep the uploaded WireGuard credentials outside dataset folders."""

    def __init__(self, directory: Path | str) -> None:
        self.directory = Path(directory).expanduser().absolute()
        self.config_path = self.directory / "wireguard.conf"
        self.settings_path = self.directory / "settings.json"

    def _ensure_private_directory(self) -> None:
        if self.directory.is_symlink():
            raise ValueError("WireGuard storage directory cannot be a symlink")
        self.directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        if os.name == "posix":
            self.directory.chmod(0o700)

    def _read_settings(self) -> dict[str, object]:
        if self.settings_path.is_symlink():
            raise ValueError("WireGuard settings file cannot be a symlink")
        try:
            value = json.loads(self.settings_path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return {}
        except (OSError, UnicodeError, json.JSONDecodeError):
            return {}
        return value if isinstance(value, dict) else {}

    @staticmethod
    def _write_private_file(path: Path, data: bytes) -> None:
        if path.is_symlink():
            raise ValueError("refusing to overwrite a symlinked WireGuard file")
        fd, temporary_name = tempfile.mkstemp(
            prefix=".wireguard-",
            suffix=".tmp",
            dir=path.parent,
        )
        try:
            if os.name == "posix":
                os.fchmod(fd, 0o600)
            with os.fdopen(fd, "wb") as handle:
                fd = -1
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_name, path)
            if os.name == "posix":
                path.chmod(0o600)
        except Exception:
            if fd >= 0:
                os.close(fd)
            try:
                os.unlink(temporary_name)
            except OSError:
                pass
            raise

    def save_config(self, content: bytes, filename: str) -> dict[str, object]:
        validate_wireguard_config(content)
        self._ensure_private_directory()
        if self.config_path.is_symlink():
            raise ValueError("refusing to overwrite a symlinked WireGuard config")
        clean_name = Path(str(filename or "wireguard.conf").replace("\\", "/")).name
        clean_name = "".join(ch for ch in clean_name if ch.isprintable())[:96]
        if not clean_name.lower().endswith(".conf"):
            raise ValueError("upload a WireGuard .conf file")
        self._write_private_file(self.config_path, content)
        settings = self._read_settings()
        settings["filename"] = clean_name
        settings.setdefault("enabled", False)
        self._write_private_file(
            self.settings_path,
            (json.dumps(settings, sort_keys=True) + "\n").encode("utf-8"),
        )
        return self.public_state()

    def set_enabled(self, enabled: bool) -> dict[str, object]:
        if not isinstance(enabled, bool):
            raise ValueError("enabled must be a boolean")
        self._ensure_private_directory()
        if enabled:
            if self.config_path.is_symlink() or not self.config_path.is_file():
                raise ValueError("upload a WireGuard config before enabling the VPN")
            validate_wireguard_config(self.config_path.read_bytes())
        settings = self._read_settings()
        settings["enabled"] = enabled
        self._write_private_file(
            self.settings_path,
            (json.dumps(settings, sort_keys=True) + "\n").encode("utf-8"),
        )
        return self.public_state()

    def clear_config(self) -> dict[str, object]:
        self._ensure_private_directory()
        if self.config_path.is_symlink():
            raise ValueError("refusing to delete a symlinked WireGuard config")
        self.config_path.unlink(missing_ok=True)
        settings = self._read_settings()
        settings["enabled"] = False
        settings["filename"] = ""
        self._write_private_file(
            self.settings_path,
            (json.dumps(settings, sort_keys=True) + "\n").encode("utf-8"),
        )
        return self.public_state()

    def is_enabled(self) -> bool:
        # Preserve the user's opt-in even if the file disappears or becomes
        # invalid: the job then fails closed instead of silently scraping direct.
        try:
            return bool(self._read_settings().get("enabled"))
        except (OSError, ValueError):
            return False

    def public_state(self) -> dict[str, object]:
        try:
            settings = self._read_settings()
            configured = self.config_path.is_file() and not self.config_path.is_symlink()
            enabled = bool(settings.get("enabled")) and configured
            filename = str(settings.get("filename", "")) if configured else ""
        except (OSError, ValueError):
            configured, enabled, filename = False, False, ""
        return {
            "configured": configured,
            "enabled": enabled,
            "filename": filename,
            "wireproxy_available": bool(WireGuardProxy.find_binary()),
        }


class WireGuardProxy:
    """Run wireproxy as a local HTTP CONNECT proxy for one scrape only."""

    def __init__(
        self,
        config_path: Path | str,
        *,
        startup_timeout: float = WIREPROXY_START_TIMEOUT,
    ) -> None:
        self.config_path = Path(config_path).expanduser().absolute()
        self.startup_timeout = startup_timeout
        self.process: subprocess.Popen | None = None
        self.temp_dir: tempfile.TemporaryDirectory | None = None
        self.proxy_url: str | None = None

    @staticmethod
    def find_binary() -> str | None:
        for name in ("LORA_DATA_MAKER_WIREPROXY_BIN", "WIREPROXY_BIN"):
            configured = os.environ.get(name, "").strip()
            if configured:
                path = Path(configured).expanduser()
                if path.is_file() and os.access(path, os.X_OK):
                    return str(path)
                return None
        return shutil.which("wireproxy")

    def start(self, *, cancel_event=None) -> "WireGuardProxy":
        if self.process is not None:
            return self
        binary = self.find_binary()
        if not binary:
            raise RuntimeError(
                "wireproxy was not found. Install the wireproxy executable in PATH "
                "or set LORA_DATA_MAKER_WIREPROXY_BIN."
            )
        if self.config_path.is_symlink() or not self.config_path.is_file():
            raise RuntimeError("WireGuard config is missing or is not a regular file")
        try:
            validate_wireguard_config(self.config_path.read_bytes())
        except (OSError, ValueError) as exc:
            raise RuntimeError(f"WireGuard config validation failed: {exc}") from exc

        try:
            self.temp_dir = tempfile.TemporaryDirectory(prefix="ldm-wireproxy-")
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
                server.bind(("127.0.0.1", 0))
                port = int(server.getsockname()[1])
            self.proxy_url = f"http://127.0.0.1:{port}"
            proxy_config = Path(self.temp_dir.name) / "wireproxy.conf"
            config_text = (
                f"WGConfig = {self.config_path}\n\n"
                "[http]\n"
                f"BindAddress = 127.0.0.1:{port}\n"
            )
            self._write_proxy_config(proxy_config, config_text)
            self.process = subprocess.Popen(
                [binary, "-c", str(proxy_config)],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                close_fds=True,
            )
        except OSError as exc:
            self.close()
            raise RuntimeError(f"could not start wireproxy: {exc}") from exc

        deadline = time.monotonic() + self.startup_timeout
        while time.monotonic() < deadline:
            if cancel_event is not None and cancel_event.is_set():
                self.close()
                raise ScraperCancelledError("scrape cancelled")
            if self.process is None or self.process.poll() is not None:
                self.close()
                raise RuntimeError(
                    "wireproxy exited before its local HTTP proxy became ready. "
                    "Check the WireGuard config and wireproxy build."
                )
            try:
                with socket.create_connection(("127.0.0.1", port), timeout=0.25):
                    return self
            except OSError:
                time.sleep(0.1)

        self.close()
        raise RuntimeError(
            f"wireproxy did not open its local HTTP proxy within {self.startup_timeout:g} seconds"
        )

    @staticmethod
    def _write_proxy_config(path: Path, content: str) -> None:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())

    def close(self) -> None:
        process, self.process = self.process, None
        if process is not None:
            try:
                process.terminate()
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                try:
                    process.kill()
                    process.wait(timeout=2)
                except Exception:
                    pass
            except OSError:
                pass
        temp_dir, self.temp_dir = self.temp_dir, None
        if temp_dir is not None:
            temp_dir.cleanup()
        self.proxy_url = None

    def __enter__(self) -> "WireGuardProxy":
        return self.start()

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
