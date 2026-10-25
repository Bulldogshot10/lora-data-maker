from __future__ import annotations

import pytest

from lora_data_maker.scrapers.base import ScraperError
from lora_data_maker.scrapers.site import (
    PlaywrightSiteScraper,
    site_debug_port,
    site_profile_dir,
    validate_page_url,
)


def test_site_page_url_requires_http_or_https():
    assert validate_page_url("https://example.com/gallery") == "https://example.com/gallery"
    assert validate_page_url("http://localhost:8000/images") == "http://localhost:8000/images"
    for value in ("", "example.com", "file:///tmp/image.jpg", "ftp://example.com/a.jpg",
                  "https://user:password@example.com/gallery"):
        with pytest.raises(ScraperError):
            validate_page_url(value)


def test_site_profile_and_cdp_port_are_dedicated_and_configurable(monkeypatch, tmp_path):
    monkeypatch.delenv("CHROME_SITE_USER_DATA_DIR", raising=False)
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    monkeypatch.setenv("CHROME_SITE_DEBUG_PORT", "9450")
    assert site_profile_dir() == (tmp_path / "data" / "lora-data-maker" / "playwright-site-profile").resolve()
    assert site_debug_port() == 9450

    monkeypatch.setenv("CHROME_SITE_DEBUG_PORT", "not-a-port")
    with pytest.raises(ScraperError):
        site_debug_port()


def test_image_expression_uses_selected_css_selector_and_invoked_iife():
    expression = PlaywrightSiteScraper._image_expression("article.gallery img")
    assert expression.lstrip().startswith("(() =>")
    assert expression.rstrip().endswith("()")
    assert "document.querySelectorAll(selector)" in expression
    assert "article.gallery img" in expression
    assert "data-original-src" in expression
    assert "naturalWidth" in expression


def test_normalize_candidate_resolves_relative_url_and_retains_image_metadata():
    result = PlaywrightSiteScraper._normalize_candidate(
        {"url": "/media/large.jpg", "width": 1024, "height": 768, "alt": "portrait"},
        "https://example.com/gallery/page",
    )
    assert result is not None
    url, metadata = result
    assert url == "https://example.com/media/large.jpg"
    assert metadata["image_width"] == 1024
    assert metadata["image_height"] == 768
    assert metadata["alt"] == "portrait"
    assert PlaywrightSiteScraper._normalize_candidate(
        {"url": "data:image/png;base64,abc", "width": 512, "height": 512},
        "https://example.com/gallery",
    ) is None


def test_site_scraper_reuses_browser_session_and_captures_cookies(monkeypatch):
    from lora_data_maker.scrapers import site as site_module

    class FakeSession:
        def __init__(self):
            self.sent = []

        def send(self, method, params, **kwargs):
            self.sent.append((method, params))
            if method == "Network.getAllCookies":
                return {
                    "cookies": [{
                        "name": "sessionid",
                        "value": "secret-cookie-value",
                        "domain": ".example.com",
                        "path": "/",
                        "secure": True,
                        "expires": -1,
                    }]
                }
            return {}

    class FakePage:
        def __init__(self):
            self.session = FakeSession()
            self.closed = False

        def bring_to_front(self):
            pass

        def evaluate(self, expression):
            if expression == "document.readyState":
                return "complete"
            if expression == "location.href":
                return "https://example.com/gallery"
            if "document.querySelectorAll(selector)" in expression:
                return [
                    {"url": "https://cdn.example.com/a.jpg", "width": 900, "height": 1200, "alt": "a"},
                    {"url": "https://cdn.example.com/b.webp", "width": 800, "height": 1000, "alt": "b"},
                ]
            return None

        def wait_for_timeout(self, _timeout):
            pass

        def close(self):
            self.closed = True

    class FakeBrowser:
        def __init__(self):
            self.page = FakePage()

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            pass

        def new_page(self):
            return self.page

    browser = FakeBrowser()
    calls = []

    def fake_create_site_browser(**kwargs):
        calls.append(kwargs)
        return browser

    monkeypatch.setattr(site_module, "create_site_browser", fake_create_site_browser)
    scraper = PlaywrightSiteScraper(timeout=3, retries=0, delay=0, max_scrolls=5)
    results = list(scraper.search("https://example.com/gallery", limit=2))

    assert [result.image_url for result in results] == [
        "https://cdn.example.com/a.jpg",
        "https://cdn.example.com/b.webp",
    ]
    assert calls == [{"timeout": 3.0, "auto_launch": False, "headless": False}]
    assert scraper.cookies[0]["name"] == "sessionid"
    assert scraper.cookies[0]["value"] == "secret-cookie-value"
    assert browser.page.closed
