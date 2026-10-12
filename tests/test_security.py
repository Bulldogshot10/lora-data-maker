from __future__ import annotations

import socket

import pytest

from lora_data_maker.security import (
    assert_safe_outbound_url,
    is_safe_media_path,
    validate_api_base_url,
)


def _dns_answer(ip: str):
    family = socket.AF_INET6 if ":" in ip else socket.AF_INET
    sockaddr = (ip, 443, 0, 0) if family == socket.AF_INET6 else (ip, 443)
    return [(family, socket.SOCK_STREAM, 6, "", sockaddr)]


def test_public_https_api_endpoint_is_accepted(monkeypatch):
    monkeypatch.setattr(
        "lora_data_maker.security.socket.getaddrinfo",
        lambda host, port, **kwargs: _dns_answer("93.184.216.34"),
    )
    assert assert_safe_outbound_url(
        "https://provider.example.org/v1",
        allow_loopback=True,
        api_base=True,
    ) == "https://provider.example.org/v1"


@pytest.mark.parametrize(
    "url",
    [
        "http://provider.example.org/v1",
        "https://user:secret@provider.example.org/v1",
        "https://provider.example.org/v1?token=secret",
        "https://127.0.0.1/v1",
        "https://10.0.0.4/v1",
        "https://169.254.169.254/latest/meta-data",
        "https://192.168.1.10/v1",
    ],
)
def test_api_base_rejects_unsafe_url_shapes_and_private_ip_addresses(url):
    with pytest.raises(ValueError):
        validate_api_base_url(url)


def test_plain_http_is_allowed_for_loopback_ollama():
    assert validate_api_base_url("http://127.0.0.1:11434/v1") == (
        "http://127.0.0.1:11434/v1"
    )
    assert validate_api_base_url("http://localhost:11434/v1") == (
        "http://localhost:11434/v1"
    )


@pytest.mark.parametrize("address", ["127.0.0.1", "10.2.3.4", "169.254.169.254", "::1"])
def test_scraper_cannot_fetch_loopback_or_private_literal_addresses(address):
    with pytest.raises(ValueError):
        assert_safe_outbound_url(f"https://{f'[{address}]' if ':' in address else address}/image.jpg")


def test_dns_resolving_to_private_address_is_rejected(monkeypatch):
    monkeypatch.setattr(
        "lora_data_maker.security.socket.getaddrinfo",
        lambda host, port, **kwargs: _dns_answer("192.168.1.99"),
    )
    with pytest.raises(ValueError, match="private"):
        assert_safe_outbound_url("https://attacker-controlled.example.org/image.jpg")


def test_dns_resolving_to_loopback_is_rejected_even_for_a_hostname(monkeypatch):
    monkeypatch.setattr(
        "lora_data_maker.security.socket.getaddrinfo",
        lambda host, port, **kwargs: _dns_answer("127.0.0.1"),
    )
    with pytest.raises(ValueError, match="loopback"):
        assert_safe_outbound_url("https://rebind.example.org/image.jpg")


def test_media_route_policy_only_allows_image_artifacts():
    assert is_safe_media_path("raw/001.jpg")
    assert is_safe_media_path("work/processed/001.png")
    assert is_safe_media_path("output/1_julia/001.webp")
    assert not is_safe_media_path("config.toml")
    assert not is_safe_media_path("manifest.json")
    assert not is_safe_media_path("models/model.gguf")
    assert not is_safe_media_path("work/captions/001.txt")
    assert not is_safe_media_path("raw/../config.toml")
    assert not is_safe_media_path("/etc/passwd")
