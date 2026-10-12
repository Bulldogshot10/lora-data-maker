from __future__ import annotations

import ipaddress
import socket
from pathlib import PurePosixPath
from urllib.parse import urlsplit


MAX_JSON_REQUEST_BYTES = 2 * 1024 * 1024
MAX_IMAGE_UPLOAD_BYTES = 25 * 1024 * 1024
MAX_IMAGE_UPLOAD_REQUEST_BYTES = 128 * 1024 * 1024
MAX_VLM_UPLOAD_BYTES = 2 * 1024 * 1024 * 1024
MAX_IMAGE_PIXELS = 40_000_000

IMAGE_SUFFIXES = frozenset(
    {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tif", ".tiff", ".avif"}
)
_LOCAL_HOSTS = frozenset({"localhost", "ip6-localhost"})


def _host_is_loopback_name(host: str) -> bool:
    normalized = host.rstrip(".").lower()
    return normalized in _LOCAL_HOSTS


def validate_api_base_url(value: str) -> str:
    """Validate the syntax and safe defaults for an API base URL.

    Public endpoints must use HTTPS. Plain HTTP is allowed only for loopback
    services such as a local Ollama instance. Credentials, query strings, and
    fragments are never accepted in the URL.
    """
    raw = str(value or "").strip()
    if not raw:
        return ""

    if len(raw) > 2048 or any(ord(char) < 32 or ord(char) == 127 for char in raw):
        raise ValueError("API base URL is too long or contains control characters")

    try:
        parts = urlsplit(raw)
        host = parts.hostname
        port = parts.port
    except ValueError as exc:
        raise ValueError("invalid API base URL") from exc

    if (
        parts.scheme.lower() not in {"http", "https"}
        or not host
        or parts.username is not None
        or parts.password is not None
        or parts.query
        or parts.fragment
        or "\\" in parts.netloc
    ):
        raise ValueError(
            "API base URL must be an http(s) URL without credentials, query, or fragment"
        )

    if port is not None and not 1 <= port <= 65535:
        raise ValueError("API port must be between 1 and 65535")

    normalized_host = host.rstrip(".").lower()
    local_name = _host_is_loopback_name(normalized_host)
    try:
        address = ipaddress.ip_address(normalized_host.split("%", 1)[0])
    except ValueError:
        address = None

    if address is not None:
        if not address.is_global and not address.is_loopback:
            raise ValueError("API base URL cannot target a private or reserved IP address")
        local_name = address.is_loopback

    if parts.scheme.lower() == "http" and not local_name:
        raise ValueError("public API base URLs must use HTTPS")
    if normalized_host.endswith((".local", ".internal", ".lan", ".home.arpa")):
        raise ValueError("local network hostnames are not allowed as API endpoints")

    # Accessing parts.port above validates that any explicit port is numeric
    # and in the TCP port range. Recompose without userinfo/query/fragment.
    del port
    return raw.rstrip("/")


def assert_safe_outbound_url(
    value: str,
    *,
    allow_loopback: bool = False,
    api_base: bool = False,
) -> str:
    """Fail closed on unsafe outbound destinations and DNS results.

    Every DNS answer must be globally routable, except for an explicitly
    allowed loopback endpoint. DNS is checked immediately before each request,
    including each redirect hop.
    """
    raw = str(value or "").strip()
    if api_base:
        raw = validate_api_base_url(raw)
        if not raw:
            raise ValueError("API base URL is required")

    if len(raw) > 4096 or any(ord(char) < 32 or ord(char) == 127 for char in raw):
        raise ValueError("outbound URL is too long or contains control characters")

    try:
        parts = urlsplit(raw)
        host = parts.hostname
        explicit_port = parts.port
        port = explicit_port if explicit_port is not None else (443 if parts.scheme.lower() == "https" else 80)
    except ValueError as exc:
        raise ValueError("invalid outbound URL") from exc

    if (
        parts.scheme.lower() not in {"http", "https"}
        or not host
        or parts.username is not None
        or parts.password is not None
    ):
        raise ValueError("outbound URL must be an http(s) URL without embedded credentials")

    if not 1 <= port <= 65535:
        raise ValueError("outbound port must be between 1 and 65535")

    normalized_host = host.rstrip(".").lower()
    loopback_name = _host_is_loopback_name(normalized_host)
    try:
        literal_ip = ipaddress.ip_address(normalized_host.split("%", 1)[0])
    except ValueError:
        literal_ip = None

    if parts.scheme.lower() == "http" and not (
        allow_loopback and (loopback_name or (literal_ip is not None and literal_ip.is_loopback))
    ):
        raise ValueError("outbound HTTP is allowed only for explicitly permitted loopback services")

    if literal_ip is not None:
        if literal_ip.is_loopback:
            if allow_loopback:
                return raw
            raise ValueError("loopback outbound destinations are blocked")
        if not literal_ip.is_global:
            raise ValueError("private, link-local, and reserved outbound IP addresses are blocked")
        return raw

    if normalized_host.endswith((".local", ".internal", ".lan", ".home.arpa")):
        raise ValueError("local network hostnames are blocked")

    try:
        answers = socket.getaddrinfo(
            normalized_host,
            port,
            type=socket.SOCK_STREAM,
        )
    except socket.gaierror as exc:
        raise ValueError(f"could not resolve outbound host {normalized_host!r}") from exc

    addresses: set[ipaddress.IPv4Address | ipaddress.IPv6Address] = set()
    for answer in answers:
        try:
            addresses.add(ipaddress.ip_address(answer[4][0].split("%", 1)[0]))
        except (ValueError, IndexError, TypeError):
            continue

    if not addresses:
        raise ValueError(f"outbound host {normalized_host!r} has no valid IP addresses")

    for address in addresses:
        if loopback_name and not address.is_loopback:
            raise ValueError("localhost hostname resolved to a non-loopback address")
        if address.is_loopback:
            if allow_loopback and loopback_name:
                continue
            raise ValueError("loopback outbound destinations are blocked")
        if not address.is_global:
            raise ValueError("DNS resolved to a private, link-local, or reserved IP address")

    return raw


def is_safe_media_path(relative: str) -> bool:
    """Allow only image files in public-facing dataset image directories."""
    path = PurePosixPath(str(relative).replace("\\", "/"))
    if path.is_absolute() or not path.parts or any(part in {"", ".", ".."} for part in path.parts):
        return False
    if path.parts[0] not in {"raw", "work", "output"}:
        return False
    return path.suffix.lower() in IMAGE_SUFFIXES
