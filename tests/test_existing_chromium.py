import json

import pytest

from lora_data_maker.scrapers import existing_chromium as existing_chromium_module
from lora_data_maker.scrapers.base import ScraperError


class FakeResponse:
    def __init__(self, status: int, payload: bytes = b"") -> None:
        self.status = status
        self.payload = payload
        self.read_sizes: list[int] = []

    def read(self, size: int = -1) -> bytes:
        self.read_sizes.append(size)
        return self.payload[:size] if size >= 0 else self.payload


class FakeConnection:
    def __init__(self, response: FakeResponse) -> None:
        self.response = response
        self.requests: list[tuple[str, str, dict[str, str]]] = []
        self.closed = False

    def request(self, method: str, path: str, *, headers: dict[str, str]) -> None:
        self.requests.append((method, path, headers))

    def getresponse(self) -> FakeResponse:
        return self.response

    def close(self) -> None:
        self.closed = True


def _chromium(tmp_path):
    return existing_chromium_module.ExistingChromium(
        user_data_dir=tmp_path,
        host="127.0.0.1",
        port=9222,
        timeout=1.0,
    )


def test_cdp_discovery_uses_direct_request_and_rejects_redirect(monkeypatch, tmp_path):
    response = FakeResponse(302, b'{"webSocketDebuggerUrl":"ws://attacker.example/devtools"}')
    connection = FakeConnection(response)
    monkeypatch.setattr(
        existing_chromium_module,
        "HTTPConnection",
        lambda host, port, timeout: connection,
    )

    with pytest.raises(ScraperError, match="HTTP 302"):
        _chromium(tmp_path)._discover_websocket_url()

    assert connection.requests == [
        ("GET", "/json/version", {"Connection": "close"})
    ]
    assert connection.closed
    assert response.read_sizes == []


def test_cdp_discovery_bounds_metadata_response(monkeypatch, tmp_path):
    response = FakeResponse(200, b"x" * (1024 * 1024 + 1))
    connection = FakeConnection(response)
    monkeypatch.setattr(
        existing_chromium_module,
        "HTTPConnection",
        lambda host, port, timeout: connection,
    )

    with pytest.raises(ScraperError, match="larger than 1 MiB"):
        _chromium(tmp_path)._discover_websocket_url()

    assert response.read_sizes == [1024 * 1024 + 1]
    assert connection.closed


def test_cdp_discovery_accepts_local_websocket_url(monkeypatch, tmp_path):
    payload = json.dumps(
        {"webSocketDebuggerUrl": "ws://127.0.0.1:9222/devtools/browser/test"}
    ).encode("utf-8")
    response = FakeResponse(200, payload)
    connection = FakeConnection(response)
    monkeypatch.setattr(
        existing_chromium_module,
        "HTTPConnection",
        lambda host, port, timeout: connection,
    )

    assert (
        _chromium(tmp_path)._discover_websocket_url()
        == "ws://127.0.0.1:9222/devtools/browser/test"
    )
    assert connection.closed


def test_cdp_discovery_rejects_external_websocket_url(monkeypatch, tmp_path):
    payload = json.dumps(
        {"webSocketDebuggerUrl": "ws://attacker.example:9222/devtools/browser/test"}
    ).encode("utf-8")
    connection = FakeConnection(FakeResponse(200, payload))
    monkeypatch.setattr(
        existing_chromium_module,
        "HTTPConnection",
        lambda host, port, timeout: connection,
    )

    with pytest.raises(ScraperError, match="non-local"):
        _chromium(tmp_path)._discover_websocket_url()

    assert connection.closed
