from lora_data_maker.scrapers.archive import DownloadArchive
from lora_data_maker.scrapers.google import GoogleImagesScraper

import pytest


@pytest.fixture(autouse=True)
def mock_downloader_url_validation(monkeypatch):
    # Downloader tests use fake HTTP sessions and reserved example hosts.
    monkeypatch.setattr("lora_data_maker.scrapers.downloader.assert_safe_outbound_url", lambda value, **kwargs: value)


def test_download_archive_deduplicates(tmp_path):
    archive = DownloadArchive(tmp_path / "downloads.txt")
    assert "https://example.com/a.jpg" not in archive
    assert archive.add("https://example.com/a.jpg") is True
    assert archive.add("https://example.com/a.jpg") is False

    reloaded = DownloadArchive(tmp_path / "downloads.txt")
    assert "https://example.com/a.jpg" in reloaded


def test_google_extracts_image_urls_from_markup():
    body = '''
    <img src="https://cdn.example.com/one.jpg">
    <img data-iurl="https://cdn.example.com/two.png">
    <script>
      var data = "https://cdn.example.com/three.webp";
      var unrelated = "https://www.google.com/search?q=abc";
    </script>
    '''
    urls = GoogleImagesScraper._extract_urls(body)
    assert urls == [
        "https://cdn.example.com/one.jpg",
        "https://cdn.example.com/two.png",
        "https://cdn.example.com/three.webp",
    ]


def test_google_rejects_obvious_navigation_urls():
    assert GoogleImagesScraper._looks_like_image_url(
        "https://www.google.com/search?q=julia"
    ) is False
    assert GoogleImagesScraper._looks_like_image_url(
        "https://example.com/profile/julia"
    ) is True
    assert GoogleImagesScraper._looks_like_image_url(
        "https://support.google.com/websearch"
    ) is False
    assert GoogleImagesScraper._looks_like_image_url(
        "https://support.google.com/websearch/answer/134479"
    ) is False


class _FakeResponse:
    def __init__(
        self,
        text: str,
        *,
        content: bytes = b"",
        content_type: str = "image/jpeg",
    ) -> None:
        self.text = text
        self.content = content
        self.headers = {"Content-Type": content_type}
        self.status_code = 200
        self.closed = False

    def raise_for_status(self):
        return None

    def iter_content(self, chunk_size=0):
        yield self.content

    def close(self):
        self.closed = True


class _FakeSession:
    def __init__(self):
        self.headers = {}
        self.calls = []
        self.responses = [_FakeResponse("", content=b"fake-image")]

    def get(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return self.responses.pop(0)


def test_downloader_sends_referer_for_protected_image_hosts(tmp_path):
    from lora_data_maker.scrapers.downloader import ImageDownloader

    session = _FakeSession()
    downloader = ImageDownloader(
        timeout=5,
        retries=0,
        delay=0,
        user_agent="test",
        session=session,
    )

    downloader.download(
        "https://cdn.example.com/original.jpg",
        tmp_path,
        index=1,
        referer="https://www.google.com/",
    )

    options = session.calls[0][1]
    assert options["headers"]["Referer"] == "https://www.google.com/"
    assert "image/*" in options["headers"]["Accept"]
    assert options["headers"]["Accept-Language"] == "en-US,en;q=0.9"


def test_direct_downloader_writes_stable_filename(tmp_path):
    from lora_data_maker.scrapers.downloader import ImageDownloader

    session = _FakeSession()
    downloader = ImageDownloader(
        timeout=5,
        retries=1,
        delay=0,
        user_agent="test",
        session=session,
    )
    result = downloader.download(
        "https://cdn.example.com/original.jpg",
        tmp_path,
        index=1,
        archive=DownloadArchive(tmp_path / "archive.txt"),
    )

    assert result is not None
    assert result.path.name == "000001.jpg"
    assert result.file_size == len(b"fake-image")
    assert result.sha256
    assert result.path.read_bytes() == b"fake-image"
    assert session.calls


def test_downloader_rejects_non_image_content(tmp_path):
    from lora_data_maker.scrapers.downloader import ImageDownloader

    session = _FakeSession()
    session.responses = [
        _FakeResponse("", content=b"html", content_type="text/html"),
        _FakeResponse("", content=b"html", content_type="text/html"),
    ]
    downloader = ImageDownloader(
        timeout=5,
        retries=1,
        delay=0,
        user_agent="test",
        session=session,
    )

    try:
        downloader.download(
            "https://example.com/page",
            tmp_path,
            index=1,
            archive=DownloadArchive(tmp_path / "archive.txt"),
        )
    except Exception as exc:
        assert "did not return an image" in str(exc)
    else:
        raise AssertionError("expected non-image content to fail")


def test_google_decodes_escaped_json_urls():
    body = r'''
    <script>
      var payload = "https:\/\/cdn.example.com\/photo.jpg?x\u003d1\u0026y\u003d2";
    </script>
    '''
    urls = GoogleImagesScraper._extract_urls(body)
    assert urls == [
        "https://cdn.example.com/photo.jpg?x=1&y=2"
    ]


def test_google_detects_verification_page():
    try:
        GoogleImagesScraper._check_block_page(
            "Our systems have detected unusual traffic from your computer network."
        )
    except Exception as exc:
        assert "verification" in str(exc).lower()
    else:
        raise AssertionError("expected verification page to fail")


def test_scraper_manager_skips_urls_already_in_manifest(tmp_path):
    from lora_data_maker.scrapers.base import ImageResult
    from lora_data_maker.scrapers.manager import ScraperManager

    manager = ScraperManager(
        backend="direct",
        output_dir=tmp_path / "raw",
        cache_dir=tmp_path / "cache",
        delay=0,
    )

    class FakeScraper:
        def search(self, query, *, limit):
            yield ImageResult(
                image_url="https://example.com/already.jpg",
                query=query,
            )

    calls = []

    class FakeDownloader:
        def download(self, *args, **kwargs):
            calls.append(args[0])
            raise AssertionError("known URL should not reach downloader")

    manager.scraper = FakeScraper()
    manager.downloader = FakeDownloader()

    downloads, failures = manager.run(
        "julia",
        count=1,
        known_urls={"https://example.com/already.jpg"},
    )

    assert downloads == []
    assert failures == []
    assert calls == []


def test_scraper_manager_skips_failed_downloads_and_continues(tmp_path):
    from pathlib import Path

    from lora_data_maker.scrapers.base import ImageResult, ScraperError
    from lora_data_maker.scrapers.downloader import DownloadResult
    from lora_data_maker.scrapers.manager import ScraperManager

    manager = ScraperManager(
        backend="direct",
        output_dir=tmp_path / "raw",
        cache_dir=tmp_path / "cache",
        delay=0,
        retries=5,
    )
    assert manager.downloader.retries == 5

    results = [
        ImageResult("https://example.com/a.jpg", query="julia"),
        ImageResult("https://example.com/b.jpg", query="julia"),
        ImageResult("https://example.com/c.jpg", query="julia"),
    ]

    class FakeScraper:
        def search(self, query, *, limit):
            return iter(results)

    manager.scraper = FakeScraper()

    calls = []

    class FakeDownloader:
        def download(self, url, output_dir, *, index, archive, cancel_event=None):
            calls.append(url)
            if url.endswith("/b.jpg"):
                raise ScraperError("source refused image")
            return DownloadResult(
                url=url,
                path=Path(output_dir) / f"{index:06d}.jpg",
                sha256="0" * 64,
                file_size=123,
                content_type="image/jpeg",
            )

    manager.downloader = FakeDownloader()

    downloads, failures = manager.run(
        "julia",
        count=2,
        known_urls=(),
    )

    assert calls == [
        "https://example.com/a.jpg",
        "https://example.com/b.jpg",
        "https://example.com/c.jpg",
    ]
    assert [item.url for item in downloads] == [
        "https://example.com/a.jpg",
        "https://example.com/c.jpg",
    ]
    assert len(failures) == 1
    assert failures[0][0].image_url.endswith("/b.jpg")



def test_playwright_google_manager_only_downloads_thumbnail_url(tmp_path):
    from lora_data_maker.scrapers.base import ImageResult, ScraperError
    from lora_data_maker.scrapers.manager import ScraperManager

    original = "https://cdn.example.com/original.jpg"
    thumbnail = "https://encrypted-tbn0.gstatic.com/images?q=tbn%3Aexample"
    manager = ScraperManager(
        backend="playwright-google",
        output_dir=tmp_path / "raw",
        cache_dir=tmp_path / "cache",
        delay=0,
        retries=5,
    )
    assert manager.downloader.retries == 2

    class FakeScraper:
        def search(self, query, *, limit):
            # Google extraction supplies the thumbnail as image_url. Any legacy
            # original URL in metadata is deliberately ignored by the manager.
            yield ImageResult(
                thumbnail,
                query=query,
                metadata={"original_url": original},
            )

    manager.scraper = FakeScraper()
    calls = []

    class FakeDownloader:
        def download(
            self, url, output_dir, *, index, archive=None,
            cancel_event=None, referer=None,
        ):
            calls.append((url, referer))
            raise ScraperError("403 Client Error: Forbidden")

    manager.downloader = FakeDownloader()
    downloads, failures = manager.run("portrait", count=1)

    assert downloads == []
    assert len(failures) == 1
    assert failures[0][0].image_url == thumbnail
    assert calls == [(thumbnail, "https://www.google.com/")]



def test_scraper_manager_preserves_downloads_when_search_fails(tmp_path):
    from pathlib import Path

    from lora_data_maker.scrapers.base import ImageResult, ScraperError
    from lora_data_maker.scrapers.downloader import DownloadResult
    from lora_data_maker.scrapers.manager import ScraperManager

    manager = ScraperManager(
        backend="direct",
        output_dir=tmp_path / "raw",
        cache_dir=tmp_path / "cache",
        delay=0,
    )

    class FakeScraper:
        def search(self, query, *, limit):
            yield ImageResult("https://example.com/good.jpg", query=query)
            raise ScraperError("search source interrupted")

    class FakeDownloader:
        def download(self, url, output_dir, *, index, archive, cancel_event=None):
            return DownloadResult(
                url=url,
                path=Path(output_dir) / f"{index:06d}.jpg",
                sha256="0" * 64,
                file_size=123,
                content_type="image/jpeg",
            )

    manager.scraper = FakeScraper()
    manager.downloader = FakeDownloader()

    downloads, failures = manager.run(
        "julia",
        count=2,
        known_urls=(),
    )

    assert [item.url for item in downloads] == [
        "https://example.com/good.jpg"
    ]
    assert failures == []
def test_downloader_prefers_ipv4_in_termux(monkeypatch):
    import urllib3.util.connection as urllib3_connection
    from lora_data_maker.scrapers.downloader import ImageDownloader

    for key in ("TERMUX__ROOTFS", "TERMUX_VERSION", "PREFIX"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("TERMUX_VERSION", "test")

    original = urllib3_connection.HAS_IPV6
    try:
        urllib3_connection.HAS_IPV6 = True
        ImageDownloader(timeout=5, retries=0, delay=0, user_agent="test")
        assert urllib3_connection.HAS_IPV6 is False
    finally:
        urllib3_connection.HAS_IPV6 = original


def test_duckduckgo_extracts_image_results():
    from lora_data_maker.scrapers.duckduckgo import DuckDuckGoImagesScraper

    class FakeSession:
        def __init__(self):
            self.headers = {}

        def get(self, url, **kwargs):
            if url == DuckDuckGoImagesScraper.SITE_URL:
                return _FakeTextResponse("<html>vqd='12345'</html>")
            return _FakeTextResponse(
                '{"results": ['
                '{"image":"https://cdn.example.com/a.jpg","title":"A"},'
                '{"image":"https://cdn.example.com/b.webp","title":"B"}'
                ']}'
            )

    class _FakeTextResponse:
        def __init__(self, text):
            self.text = text

        def raise_for_status(self):
            return None

    scraper = DuckDuckGoImagesScraper(
        timeout=5,
        retries=0,
        delay=0,
        session=FakeSession(),
    )
    results = list(scraper.search("julia", limit=2))
    assert [item.image_url for item in results] == [
        "https://cdn.example.com/a.jpg",
        "https://cdn.example.com/b.webp",
    ]


def test_chromium_uses_duckduckgo_without_launching_bing(monkeypatch):
    from lora_data_maker.scrapers import chromium as chromium_module

    class FakeDuckDuckGo:
        SITE_URL = "https://duckduckgo.com/"

        def __init__(self, **kwargs):
            pass

        def search(self, query, *, limit):
            from lora_data_maker.scrapers.base import ImageResult

            assert query == "julia"
            assert limit == 1
            yield ImageResult("https://cdn.example.com/ddg.jpg", query=query)

    monkeypatch.setattr(
        chromium_module,
        "DuckDuckGoImagesScraper",
        FakeDuckDuckGo,
    )

    scraper = chromium_module.ChromiumImagesScraper(delay=0)
    monkeypatch.setattr(
        scraper,
        "_start_driver",
        lambda: (_ for _ in ()).throw(AssertionError("Bing should not run")),
    )

    results = list(scraper.search("julia", limit=1))

    assert [item.image_url for item in results] == [
        "https://cdn.example.com/ddg.jpg"
    ]
    assert results[0].metadata["provider"] == "duckduckgo"

def test_bing_rejects_bing_hosted_thumbnails():
    from lora_data_maker.scrapers.chromium import ChromiumImagesScraper

    payloads = [
        '{"murl":"https://tse1.mm.bing.net/thumb.jpg"}',
        '{"murl":"https://cdn.example.com/real.jpg"}',
        '{"murl":"https://evilbing.com/not-image.jpg"}',
    ]

    assert ChromiumImagesScraper._extract_bing_image_urls(payloads) == [
        "https://cdn.example.com/real.jpg",
        "https://evilbing.com/not-image.jpg",
    ]

def test_bing_extracts_original_image_urls():
    from lora_data_maker.scrapers.chromium import ChromiumImagesScraper

    payloads = [
        '{"murl":"https://cdn.example.com/a.jpg","turl":"https://tse1.mm.bing.net/thumb.jpg"}',
        '{"murl":"https://cdn.example.com/b.png"}',
        '{"murl":"https://tse1.mm.bing.net/thumb.jpg"}',
        '{"murl":"https://cdn.example.com/a.jpg"}',
    ]

    assert ChromiumImagesScraper._extract_bing_image_urls(payloads) == [
        "https://cdn.example.com/a.jpg",
        "https://cdn.example.com/b.png",
    ]


def test_chromium_does_not_force_custom_user_data_dir(monkeypatch, tmp_path):
    import sys
    import types

    from lora_data_maker.scrapers.chromium import ChromiumImagesScraper

    captured = {}

    class FakeOptions:
        def __init__(self):
            self.arguments = []
            self.binary_location = None
            self.page_load_strategy = None

        def add_argument(self, value):
            self.arguments.append(value)

    class FakeService:
        def __init__(self, executable_path):
            self.executable_path = executable_path

    class FakeDriver:
        def __init__(self):
            self.timeout = None

        def set_page_load_timeout(self, timeout):
            self.timeout = timeout

        def quit(self):
            pass

    def fake_chrome(*, service, options):
        captured["service"] = service
        captured["options"] = options
        return FakeDriver()

    selenium = types.ModuleType("selenium")
    webdriver = types.ModuleType("selenium.webdriver")
    webdriver.Chrome = fake_chrome
    selenium.webdriver = webdriver

    common = types.ModuleType("selenium.common")
    exceptions = types.ModuleType("selenium.common.exceptions")
    exceptions.WebDriverException = type("WebDriverException", (Exception,), {})
    common.exceptions = exceptions

    chrome = types.ModuleType("selenium.webdriver.chrome")
    options_module = types.ModuleType("selenium.webdriver.chrome.options")
    options_module.Options = FakeOptions
    service_module = types.ModuleType("selenium.webdriver.chrome.service")
    service_module.Service = FakeService

    monkeypatch.setitem(sys.modules, "selenium", selenium)
    monkeypatch.setitem(sys.modules, "selenium.webdriver", webdriver)
    monkeypatch.setitem(sys.modules, "selenium.common", common)
    monkeypatch.setitem(sys.modules, "selenium.common.exceptions", exceptions)
    monkeypatch.setitem(sys.modules, "selenium.webdriver.chrome", chrome)
    monkeypatch.setitem(
        sys.modules,
        "selenium.webdriver.chrome.options",
        options_module,
    )
    monkeypatch.setitem(
        sys.modules,
        "selenium.webdriver.chrome.service",
        service_module,
    )

    scraper = ChromiumImagesScraper(timeout=7)
    monkeypatch.setattr(
        scraper,
        "_resolve_executable",
        lambda value: f"/fake/{value}",
    )

    driver = scraper._start_driver()

    assert driver.timeout == 7
    assert captured["service"].executable_path == "/fake/chromedriver"
    assert captured["options"].binary_location == "/fake/chromium-browser"
    assert captured["options"].page_load_strategy == "none"
    assert not any(
        argument.startswith("--user-data-dir=")
        for argument in captured["options"].arguments
    )


def test_chromium_preserves_duckduckgo_candidate_order(monkeypatch):
    from lora_data_maker.scrapers import chromium as chromium_module
    from lora_data_maker.scrapers.base import ImageResult

    class FakeDuckDuckGo:
        SITE_URL = "https://duckduckgo.com/"

        def __init__(self, **kwargs):
            pass

        def search(self, query, *, limit):
            yield ImageResult("https://cdn.example.com/first.jpg", query=query)
            yield ImageResult("https://cdn.example.com/second.jpg", query=query)

    monkeypatch.setattr(
        chromium_module,
        "DuckDuckGoImagesScraper",
        FakeDuckDuckGo,
    )

    scraper = chromium_module.ChromiumImagesScraper(delay=0)
    monkeypatch.setattr(
        scraper,
        "_start_driver",
        lambda: (_ for _ in ()).throw(AssertionError("Bing unavailable in this test")),
    )
    results = list(scraper.search("julia", limit=2))

    assert [item.image_url for item in results] == [
        "https://cdn.example.com/first.jpg",
        "https://cdn.example.com/second.jpg",
    ]
    assert [item.metadata["provider"] for item in results] == [
        "duckduckgo",
        "duckduckgo",
    ]

def test_chromium_bing_page_timeout_retries(monkeypatch):
    import sys
    import types

    from lora_data_maker.scrapers.base import ScraperError
    from lora_data_maker.scrapers.chromium import ChromiumImagesScraper

    class TimeoutException(Exception):
        pass

    WebDriverException = type("WebDriverException", (Exception,), {})

    common = types.ModuleType("selenium.common")
    exceptions = types.ModuleType("selenium.common.exceptions")
    exceptions.TimeoutException = TimeoutException
    exceptions.WebDriverException = WebDriverException
    common.exceptions = exceptions

    monkeypatch.setitem(sys.modules, "selenium.common", common)
    monkeypatch.setitem(sys.modules, "selenium.common.exceptions", exceptions)

    class FakeDriver:
        def __init__(self):
            self.get_calls = 0
            self.stop_calls = 0

        def get(self, _url):
            self.get_calls += 1
            raise TimeoutException("timed out")

        def execute_script(self, script):
            if "window.stop" in script:
                self.stop_calls += 1
            return None

    driver = FakeDriver()
    scraper = ChromiumImagesScraper(retries=1, delay=0)

    try:
        scraper._load_page(
            driver,
            "https://www.bing.com/images/search?q=julia",
        )
    except ScraperError as exc:
        assert "timed out" in str(exc).lower()
    else:
        raise AssertionError("expected Bing page timeout")

    assert driver.get_calls == 2
    assert driver.stop_calls == 2

def test_chromium_prefers_bing_then_fills_from_duckduckgo(monkeypatch):
    import sys
    import types

    from lora_data_maker.scrapers import chromium as chromium_module
    from lora_data_maker.scrapers.base import ImageResult

    TimeoutException = type("TimeoutException", (Exception,), {})
    WebDriverException = type("WebDriverException", (Exception,), {})
    common = types.ModuleType("selenium.common")
    exceptions = types.ModuleType("selenium.common.exceptions")
    exceptions.TimeoutException = TimeoutException
    exceptions.WebDriverException = WebDriverException
    common.exceptions = exceptions
    monkeypatch.setitem(sys.modules, "selenium.common", common)
    monkeypatch.setitem(sys.modules, "selenium.common.exceptions", exceptions)

    class FakeDuckDuckGo:
        SITE_URL = "https://duckduckgo.com/"

        def __init__(self, **kwargs):
            pass

        def search(self, query, *, limit):
            yield ImageResult("https://cdn.example.com/ddg.jpg", query=query)

    class FakeDriver:
        quit_called = False

        def get(self, _url):
            pass

        def execute_script(self, script):
            if "document.readyState" in script:
                return "complete"
            if "querySelectorAll" in script:
                return ['{"murl":"https://cdn.example.com/bing.jpg"}']
            return None

        def quit(self):
            self.quit_called = True

    driver = FakeDriver()
    scraper = chromium_module.ChromiumImagesScraper(
        delay=0,
        retries=0,
        max_scrolls=1,
    )
    monkeypatch.setattr(
        chromium_module,
        "DuckDuckGoImagesScraper",
        FakeDuckDuckGo,
    )
    monkeypatch.setattr(scraper, "_start_driver", lambda: driver)

    results = list(scraper.search("julia", limit=2))

    assert [item.image_url for item in results] == [
        "https://cdn.example.com/bing.jpg",
        "https://cdn.example.com/ddg.jpg",
    ]
    assert [item.metadata["provider"] for item in results] == [
        "bing",
        "duckduckgo",
    ]
    assert driver.quit_called is True

def test_downloader_extensionless_image_is_visible_to_validator(tmp_path):
    from PIL import Image
    from lora_data_maker.processing.validator import validate_image
    from lora_data_maker.scrapers.archive import DownloadArchive
    from lora_data_maker.scrapers.downloader import ImageDownloader

    fixture = tmp_path / "fixture.png"
    Image.new("RGB", (32, 32), (20, 30, 40)).save(fixture, format="PNG")

    session = _FakeSession()
    session.responses = [
        _FakeResponse(
            "",
            content=fixture.read_bytes(),
            content_type="",
        )
    ]
    downloader = ImageDownloader(
        timeout=5,
        retries=0,
        delay=0,
        user_agent="test",
        session=session,
    )
    result = downloader.download(
        "https://cdn.example.com/image-without-extension",
        tmp_path,
        index=1,
        archive=DownloadArchive(tmp_path / "archive.txt"),
    )

    assert result is not None
    assert result.path.suffix == ".img"

    validation = validate_image(result.path, min_width=16, min_height=16)
    assert validation.valid is True
    assert validation.format == "PNG"



def test_chromium_keeps_duckduckgo_when_bing_fails(monkeypatch):
    import sys
    import types

    from lora_data_maker.scrapers import chromium as chromium_module
    from lora_data_maker.scrapers.base import ImageResult

    TimeoutException = type("TimeoutException", (Exception,), {})
    WebDriverException = type("WebDriverException", (Exception,), {})
    common = types.ModuleType("selenium.common")
    exceptions = types.ModuleType("selenium.common.exceptions")
    exceptions.TimeoutException = TimeoutException
    exceptions.WebDriverException = WebDriverException
    common.exceptions = exceptions
    monkeypatch.setitem(sys.modules, "selenium.common", common)
    monkeypatch.setitem(sys.modules, "selenium.common.exceptions", exceptions)

    class FakeDuckDuckGo:
        SITE_URL = "https://duckduckgo.com/"

        def __init__(self, **kwargs):
            pass

        def search(self, query, *, limit):
            yield ImageResult("https://cdn.example.com/ddg.jpg", query=query)

    class FakeDriver:
        def get(self, _url):
            pass

        def execute_script(self, script):
            if "document.readyState" in script:
                return "complete"
            if "querySelectorAll" in script:
                raise RuntimeError("Bing DOM changed")
            return None

        def quit(self):
            self.quit_called = True

    driver = FakeDriver()
    scraper = chromium_module.ChromiumImagesScraper(
        delay=0,
        retries=0,
        max_scrolls=1,
    )
    monkeypatch.setattr(
        chromium_module,
        "DuckDuckGoImagesScraper",
        FakeDuckDuckGo,
    )
    monkeypatch.setattr(scraper, "_start_driver", lambda: driver)

    results = list(scraper.search("julia", limit=2))

    assert [item.image_url for item in results] == [
        "https://cdn.example.com/ddg.jpg"
    ]
    assert results[0].metadata["provider"] == "duckduckgo"
    assert driver.quit_called is True

def test_download_archive_rejects_invalid_utf8(tmp_path):
    from lora_data_maker.scrapers.archive import DownloadArchive
    from lora_data_maker.scrapers.base import ScraperError

    path = tmp_path / "downloads.txt"
    path.write_bytes(b"valid-url\n\xff\xfe")

    import pytest
    with pytest.raises(ScraperError, match="could not read download archive"):
        DownloadArchive(path)


def test_google_stops_when_page_has_only_seen_urls(monkeypatch):
    from lora_data_maker.scrapers.google import GoogleImagesScraper

    scraper = GoogleImagesScraper(retries=0, delay=0)

    bodies = [
        "https://cdn.example.com/a.jpg",
        "https://cdn.example.com/a.jpg",
    ]

    calls = {"count": 0}

    def fetch_page(_query, _page):
        calls["count"] += 1
        return bodies[min(calls["count"] - 1, len(bodies) - 1)]

    monkeypatch.setattr(scraper, "_fetch_page", fetch_page)

    results = list(scraper.search("julia", limit=5))

    assert [item.image_url for item in results] == [
        "https://cdn.example.com/a.jpg"
    ]
    assert calls["count"] == 2


def test_google_scraper_validates_parameters():
    from lora_data_maker.scrapers.google import GoogleImagesScraper
    import pytest

    with pytest.raises(ValueError, match="timeout"):
        GoogleImagesScraper(timeout=0)
    with pytest.raises(ValueError, match="retries"):
        GoogleImagesScraper(retries=-1)
    with pytest.raises(ValueError, match="delay"):
        GoogleImagesScraper(delay=-1)


def test_downloader_rolls_back_file_when_archive_update_fails(tmp_path):
    from PIL import Image
    from lora_data_maker.scrapers.archive import DownloadArchive
    from lora_data_maker.scrapers.downloader import ImageDownloader

    fixture = tmp_path / "fixture.jpg"
    Image.new("RGB", (32, 32), (20, 30, 40)).save(fixture, format="JPEG")

    session = _FakeSession()
    session.responses = [
        _FakeResponse("", content=fixture.read_bytes(), content_type="image/jpeg")
    ]

    archive = DownloadArchive(tmp_path / "downloads.txt")

    def fail_add(_url):
        raise OSError("archive is read-only")

    archive.add = fail_add

    downloader = ImageDownloader(
        timeout=5,
        retries=0,
        delay=0,
        user_agent="test",
        session=session,
    )

    import pytest
    with pytest.raises(Exception, match="download failed"):
        downloader.download(
            "https://cdn.example.com/a.jpg",
            tmp_path / "raw",
            index=1,
            archive=archive,
        )

    assert not (tmp_path / "raw" / "000001.jpg").exists()


def test_scrape_command_accepts_partial_success_after_skipping_failures(tmp_path, monkeypatch):
    from argparse import Namespace

    from lora_data_maker.cli import command_init, command_scrape
    from lora_data_maker.scrapers.downloader import DownloadResult

    root = tmp_path / "project"
    command_init(
        Namespace(
            path=str(root),
            name="julia",
            subject="julia",
            trigger=None,
            force=False,
        )
    )

    result = DownloadResult(
        url="https://cdn.example.com/001.jpg",
        path=root / "raw" / "000001.jpg",
        sha256="0" * 64,
        file_size=123,
        content_type="image/jpeg",
    )

    class FakeManager:
        last_search_error = None

        def __init__(self, **kwargs):
            pass

        def run(self, query, *, count, known_urls):
            return [result], [
                (
                    type(
                        "Result",
                        (),
                        {"image_url": "https://bad.example/2.jpg"},
                    )(),
                    "source refused image",
                )
            ]

    monkeypatch.setattr("lora_data_maker.cli.ScraperManager", FakeManager)

    assert command_scrape(
        Namespace(
            backend="direct",
            query="unused.txt",
            count=3,
            path=str(root),
        )
    ) == 0




def test_scrape_command_fails_when_search_source_fails_without_downloads(tmp_path, monkeypatch):
    from argparse import Namespace

    from lora_data_maker.cli import command_init, command_scrape

    root = tmp_path / "project"
    command_init(
        Namespace(
            path=str(root),
            name="julia",
            subject="julia",
            trigger=None,
            force=False,
        )
    )

    class FakeManager:
        last_search_error = "provider unavailable"

        def __init__(self, **kwargs):
            pass

        def run(self, query, *, count, known_urls):
            return [], []

    monkeypatch.setattr("lora_data_maker.cli.ScraperManager", FakeManager)

    assert command_scrape(
        Namespace(
            backend="direct",
            query="unused.txt",
            count=3,
            path=str(root),
        )
    ) == 1


def test_playwright_bing_extracts_original_urls():
    from lora_data_maker.scrapers.playwright import PlaywrightImagesScraper

    body = """
    <a class="iusc x" m='{"murl":"https://cdn.example.com/a.jpg"}'></a>
    <a m='{"murl":"https://cdn.example.com/b.jpg"}' class="iusc"></a>
    <a class="iusc" m='{"murl":"https://tse1.mm.bing.net/thumb.jpg"}'></a>
    <a class="iusc" m='{"murl":"https://cdn.example.com/a.jpg"}'></a>
    """

    assert PlaywrightImagesScraper._extract_bing_urls(body) == [
        "https://cdn.example.com/a.jpg",
        "https://cdn.example.com/b.jpg",
    ]



def test_playwright_bing_ignores_iusc_anchors_outside_results_grid():
    from lora_data_maker.scrapers.playwright import PlaywrightImagesScraper

    body = """
    <a class="iusc" m='{"murl":"https://cdn.example.com/unrelated.jpg"}'></a>
    <ul class="dgControl_list">
      <li><a class="iusc" m='{"murl":"https://cdn.example.com/query-result.jpg"}'></a></li>
    </ul>
    """

    assert PlaywrightImagesScraper._extract_bing_urls(body) == [
        "https://cdn.example.com/query-result.jpg"
    ]


def test_playwright_bing_relevance_fallback_when_grid_class_changes():
    from lora_data_maker.scrapers.playwright import PlaywrightImagesScraper

    # Some Bing layouts omit the known dgControl_list wrapper. The stable
    # iusc metadata still contains title/description and can be relevance-checked.
    body = """
    <div class="results-experiment">
      <a class="iusc" m='{"murl":"https://cdn.example.com/julia.jpg","t":"Julia Patty portrait"}'></a>
      <a class="iusc" m='{"murl":"https://cdn.example.com/zelda.jpg","t":"Zelda gameplay screenshot"}'></a>
    </div>
    """

    results = PlaywrightImagesScraper._extract_bing_results(body, "Julia Patty")

    assert [item["image_url"] for item in results] == [
        "https://cdn.example.com/julia.jpg"
    ]


def test_playwright_bing_relevance_filter_rejects_unrelated_cards():
    from lora_data_maker.scrapers.playwright import PlaywrightImagesScraper

    body = """
    <a class="iusc" m='{"murl":"https://cdn.example.com/outside.jpg"}'></a>
    <ul class="dgControl_list">
      <li>
        <a class="iusc" m='{"murl":"https://cdn.example.com/zelda.jpg","desc":"game"}'></a>
        <div class="infnmpt"><a>Zelda character gameplay screenshot</a></div>
      </li>
      <li>
        <a class="iusc" m='{"murl":"https://cdn.example.com/julia.jpg","desc":"portrait photo"}'></a>
        <div class="infnmpt"><a>Julia Patty portrait photo</a></div>
      </li>
    </ul>
    """

    results = PlaywrightImagesScraper._extract_bing_results(
        body,
        "Julia Patty portrait",
    )

    assert [item["image_url"] for item in results] == [
        "https://cdn.example.com/julia.jpg"
    ]
    assert results[0]["matched_terms"] == 3
    assert results[0]["query_terms_total"] == 3


def test_playwright_bing_relevance_filter_uses_title_or_description():
    from lora_data_maker.scrapers.playwright import PlaywrightImagesScraper

    body = """
    <ul class="dgControl_list">
      <li>
        <a class="iusc" m='{"murl":"https://cdn.example.com/one.jpg","desc":"julia patty portrait photo"}'></a>
      </li>
      <li>
        <a class="iusc" m='{"murl":"https://cdn.example.com/two.jpg","desc":"zelda video game"}'></a>
      </li>
    </ul>
    """

    results = PlaywrightImagesScraper._extract_bing_results(body, "Julia Patty")

    assert [item["image_url"] for item in results] == [
        "https://cdn.example.com/one.jpg"
    ]


def test_playwright_bing_parser_handles_html_entities_and_quote_styles():
    from lora_data_maker.scrapers.playwright import PlaywrightImagesScraper

    body = r'''
    <a class="iusc" m='{"murl":"https://cdn.example.com/single.jpg"}'></a>
    <a class="x iusc y" m="{&quot;murl&quot;:&quot;https://cdn.example.com/double.png&quot;}"></a>
    '''

    assert PlaywrightImagesScraper._extract_bing_urls(body) == [
        "https://cdn.example.com/single.jpg",
        "https://cdn.example.com/double.png",
    ]



def test_playwright_url_filters_use_hostname_boundaries():
    from lora_data_maker.scrapers.playwright import PlaywrightImagesScraper

    assert PlaywrightImagesScraper._valid_image_candidate(
        "https://evilgoogle.com/image.jpg", set()
    )
    assert not PlaywrightImagesScraper._valid_image_candidate(
        "https://gstatic.com/image.jpg", set()
    )
    assert PlaywrightImagesScraper._extract_bing_urls(
        '<a class="iusc" m=\'{"murl":"https://evilbing.com/image.jpg"}\'></a>'
    ) == ["https://evilbing.com/image.jpg"]


def test_playwright_manager_passes_max_scrolls(monkeypatch, tmp_path):
    import lora_data_maker.scrapers.manager as manager_module

    captured = {}

    class FakePlaywrightScraper:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(
        manager_module,
        "PlaywrightImagesScraper",
        FakePlaywrightScraper,
    )

    manager_module.ScraperManager(
        backend="playwright-bing",
        output_dir=tmp_path / "raw",
        cache_dir=tmp_path / "cache",
        timeout=17,
        retries=2,
        delay=0.75,
        max_scrolls=11,
    )

    assert captured == {
        "provider": "bing",
        "timeout": 17,
        "retries": 2,
        "delay": 0.75,
        "max_scrolls": 11,
        "date_from": None,
        "date_to": None,
    }


def test_playwright_google_builds_direct_google_images_url():
    from urllib.parse import parse_qs, urlparse

    from lora_data_maker.scrapers.playwright import PlaywrightImagesScraper

    url = PlaywrightImagesScraper(provider="google")._build_url(
        "julia russian woman"
    )
    parsed = urlparse(url)

    assert parsed.scheme == "https"
    assert parsed.netloc == "www.google.com"
    assert parsed.path == "/search"
    params = parse_qs(parsed.query)
    assert params["q"] == ["julia russian woman"]
    assert params["tbm"] == ["isch"]
    assert params["hl"] == ["en"]


def test_playwright_google_extracts_only_google_thumbnails():
    from lora_data_maker.scrapers.playwright import PlaywrightImagesScraper

    body = r'''
    <script>
      var first = {
        "ou":"https://cdn.example.com/original.jpg",
        "tu":"https://encrypted-tbn0.gstatic.com/images?q=tbn%3Aexample"
      };
      var second = {
        "ou":"https://images.example.net/portrait.png",
        "tu":"https://lh3.googleusercontent.com/example-thumb"
      };
      var third = {
        "ou":"https://cdn.example.com/not-used.webp",
        "tu":"https://unrelated.example.com/not-a-google-thumb.jpg"
      };
      var duplicate = {
        "tu":"https://encrypted-tbn0.gstatic.com/images?q=tbn%3Aexample"
      };
    </script>
    '''

    assert PlaywrightImagesScraper._extract_google_thumbnail_urls(body) == [
        "https://encrypted-tbn0.gstatic.com/images?q=tbn%3Aexample",
        "https://lh3.googleusercontent.com/example-thumb",
    ]


def test_playwright_google_extracts_thumbnails_from_rendered_img_markup():
    from lora_data_maker.scrapers.playwright import PlaywrightImagesScraper

    # Rendered Google Images pages may expose thumbnails in img attributes
    # without any serialized "tu" fields.
    body = """
    <html><body>
      <img src="https://encrypted-tbn0.gstatic.com/images?q=tbn%3Afirst">
      <img data-src="https://lh3.googleusercontent.com/second-thumb">
      <picture>
        <source srcset="
          https://encrypted-tbn1.gstatic.com/images?q=tbn%3Athird 1x,
          https://encrypted-tbn2.gstatic.com/images?q=tbn%3Afourth 2x
        ">
      </picture>
      <img src="https://cdn.example.com/original.jpg">
    </body></html>
    """

    assert PlaywrightImagesScraper._extract_google_thumbnail_urls(body) == [
        "https://encrypted-tbn0.gstatic.com/images?q=tbn%3Afirst",
        "https://lh3.googleusercontent.com/second-thumb",
        "https://encrypted-tbn1.gstatic.com/images?q=tbn%3Athird",
        "https://encrypted-tbn2.gstatic.com/images?q=tbn%3Afourth",
    ]


def test_playwright_google_decodes_escaped_urls_and_rejects_non_google_hosts():
    from lora_data_maker.scrapers.playwright import PlaywrightImagesScraper

    body = r'''
    <script>
      {"tu":"https:\/\/encrypted-tbn0.gstatic.com\/images?q=tbn%3Aone\u0026x\u003d2"}
      {"tu":"http:\/\/lh3.googleusercontent.com\/not-https"}
      {"tu":"https:\/\/encrypted-tbn0.gstatic.com.evil.example\/spoof"}
    </script>
    '''

    assert PlaywrightImagesScraper._extract_google_thumbnail_urls(body) == [
        "https://encrypted-tbn0.gstatic.com/images?q=tbn%3Aone&x=2"
    ]


def test_playwright_google_ignores_challenge_markers_inside_valid_results():
    from lora_data_maker.scrapers.playwright import PlaywrightImagesScraper

    body = r'''
    <html>
      <head>
        <script>
          var hiddenTemplate = "Before you continue to Google";
          var staleRedirectPath = "/sorry/";
          var imageData = {
            "ou":"https://cdn.example.com/original.jpg",
            "tu":"https://encrypted-tbn0.gstatic.com/images?q=tbn%3Aexample"
          };
        </script>
      </head>
      <body>
        <div class="image-results">Google Images results</div>
      </body>
    </html>
    '''

    assert PlaywrightImagesScraper._extract_google_thumbnail_urls(body) == [
        "https://encrypted-tbn0.gstatic.com/images?q=tbn%3Aexample"
    ]


def test_playwright_google_rejects_verification_page():
    import pytest

    from lora_data_maker.scrapers.base import ScraperError
    from lora_data_maker.scrapers.playwright import PlaywrightImagesScraper

    with pytest.raises(ScraperError, match="verification"):
        PlaywrightImagesScraper._extract_google_thumbnail_urls(
            "Our systems have detected unusual traffic from your computer network"
        )


def test_playwright_scraper_uses_termux_chromium(monkeypatch):
    import sys
    import types

    from lora_data_maker.scrapers.playwright import PlaywrightImagesScraper

    class FakeSession:
        def __init__(self):
            self.evaluate_calls = 0

        def send(self, method, params=None, timeout=None):
            if method == "Page.navigate":
                assert params["url"].startswith(
                    "https://www.bing.com/images/search?"
                )
                return {"loaderId": "loader-1"}
            if method == "Runtime.evaluate":
                assert "document.querySelectorAll('a.iusc').length" in params["expression"]
                self.evaluate_calls += 1
                if self.evaluate_calls == 1:
                    href = "about:blank"
                    result_count = 0
                elif self.evaluate_calls == 2:
                    href = "https://www.bing.com/images/search?q=wrong-query"
                    result_count = 1
                else:
                    href = "https://www.bing.com/images/search?q=julia"
                    result_count = 1
                return {"result": {"value": ["complete", href, result_count]}}
            raise AssertionError(f"unexpected CDP method: {method}")

    class FakePage:
        def __init__(self):
            self.closed = False
            self.session = FakeSession()
            self.urls = []

        def content(self):
            return (
                '<ul class="dgControl_list"><li><a class="iusc" '
                'm=\'{"murl":"https://cdn.example.com/a.jpg","t":"Julia portrait"}\'></a>'
                '<div class="infnmpt"><a>Julia portrait</a></div>'
                '</li></ul>'
            )

        def evaluate(self, _script):
            return None

        def wait_for_timeout(self, _ms):
            return None

        def close(self):
            self.closed = True

    class FakeBrowser:
        def __init__(self):
            self.page = FakePage()
            self.closed = False

        def new_page(self):
            return self.page

        def close(self):
            self.closed = True

    browser = FakeBrowser()

    class FakeChromium:
        def __init__(self, *, headless):
            assert headless is True

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            browser.close()

        def new_page(self):
            return browser.new_page()

    monkeypatch.setitem(
        sys.modules,
        "playwright_termux",
        types.SimpleNamespace(Chromium=FakeChromium),
    )

    scraper = PlaywrightImagesScraper(
        provider="bing",
        retries=0,
        delay=0,
        max_scrolls=1,
    )
    results = list(scraper.search("julia", limit=1))

    assert [item.image_url for item in results] == [
        "https://cdn.example.com/a.jpg"
    ]
    assert browser.page.closed is True
    assert browser.closed is True


def test_playwright_keeps_collected_results_if_scroll_throws(monkeypatch):
    import sys
    import types

    from lora_data_maker.scrapers.playwright import PlaywrightImagesScraper

    class FakeSession:
        def send(self, method, params=None, timeout=None):
            if method == "Page.navigate":
                return {"loaderId": "loader-1"}
            if method == "Runtime.evaluate":
                return {
                    "result": {
                        "value": [
                            "complete",
                            "https://www.bing.com/images/search?q=julia",
                            1,
                        ]
                    }
                }
            raise AssertionError(f"unexpected CDP method: {method}")

    class FakePage:
        def __init__(self):
            self.session = FakeSession()
            self.closed = False

        def content(self):
            return (
                '<ul class="dgControl_list"><li><a class="iusc" '
                'm=\'{"murl":"https://cdn.example.com/first.jpg","t":"Julia portrait"}\'></a>'
                '<div class="infnmpt"><a>Julia portrait</a></div>'
                '</li></ul>'
            )

        def evaluate(self, _script):
            raise RuntimeError("Uncaught")

        def wait_for_timeout(self, _ms):
            pass

        def close(self):
            self.closed = True

    page = FakePage()

    class FakeBrowser:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            pass

        def new_page(self):
            return page

    class FakeChromium:
        def __init__(self, *, headless):
            assert headless is True

        def __enter__(self):
            return FakeBrowser()

        def __exit__(self, *_args):
            pass

    monkeypatch.setitem(
        sys.modules,
        "playwright_termux",
        types.SimpleNamespace(Chromium=FakeChromium),
    )

    scraper = PlaywrightImagesScraper(
        provider="bing",
        retries=0,
        delay=0,
        max_scrolls=3,
    )
    results = list(scraper.search("julia", limit=2))

    assert [result.image_url for result in results] == [
        "https://cdn.example.com/first.jpg"
    ]
    assert page.closed is True


def test_playwright_scraper_validates_provider():
    import pytest

    from lora_data_maker.scrapers.playwright import PlaywrightImagesScraper

    with pytest.raises(ValueError, match="provider"):
        PlaywrightImagesScraper(provider="duckduckgo")


def test_downloader_honors_pre_cancel(tmp_path):
    import pytest

    from lora_data_maker.scrapers.base import ScraperCancelledError
    from lora_data_maker.scrapers.downloader import ImageDownloader

    downloader = ImageDownloader(
        timeout=5,
        retries=0,
        delay=0,
        user_agent="test",
        session=_FakeSession(),
    )
    cancel_event = __import__("threading").Event()
    cancel_event.set()

    with pytest.raises(ScraperCancelledError, match="cancelled"):
        downloader.download(
            "https://cdn.example.com/a.jpg",
            tmp_path,
            index=1,
            cancel_event=cancel_event,
        )

    assert not (tmp_path / "000001.jpg").exists()


def test_playwright_bing_builds_custom_date_range_url():
    from urllib.parse import parse_qs, urlparse

    from lora_data_maker.scrapers.playwright import PlaywrightImagesScraper

    scraper = PlaywrightImagesScraper(
        provider="bing",
        date_from="2019-01-01",
        date_to="2019-12-31",
    )
    parsed = urlparse(scraper._build_url("julia russian woman"))

    assert parsed.path == "/images/search"
    params = parse_qs(parsed.query)
    assert params["q"] == ["julia russian woman"]
    assert params["filters"] == ['ex1:"ez5_17897_18262"']
    assert params["adlt"] == ["off"]


def test_playwright_bing_leaves_date_filter_off_by_default():
    from urllib.parse import parse_qs, urlparse

    from lora_data_maker.scrapers.playwright import PlaywrightImagesScraper

    scraper = PlaywrightImagesScraper(provider="bing")
    params = parse_qs(urlparse(scraper._build_url("julia")).query)

    assert "filters" not in params


def test_playwright_bing_validates_custom_date_range():
    import pytest

    from lora_data_maker.scrapers.playwright import PlaywrightImagesScraper

    with pytest.raises(ValueError, match="provided together"):
        PlaywrightImagesScraper(provider="bing", date_from="2019-01-01")
    with pytest.raises(ValueError, match="on or before"):
        PlaywrightImagesScraper(
            provider="bing",
            date_from="2020-01-01",
            date_to="2019-12-31",
        )
    with pytest.raises(ValueError, match="only supported"):
        PlaywrightImagesScraper(
            provider="google",
            date_from="2019-01-01",
            date_to="2019-12-31",
        )


def test_existing_chromium_reads_google_profile_environment(monkeypatch, tmp_path):
    from lora_data_maker.scrapers.existing_chromium import ExistingChromium

    profile = tmp_path / "google-profile"
    profile.mkdir()
    monkeypatch.setenv("CHROME_USER_DATA_DIR", str(profile))
    monkeypatch.delenv("CHROME_DEBUG_PORT", raising=False)

    browser = ExistingChromium.from_environment(timeout=20)

    assert browser.user_data_dir == profile.resolve()
    assert browser.port == 9222
    assert browser.timeout == 5.0


def test_existing_chromium_discovers_local_cdp_endpoint(monkeypatch, tmp_path):
    from lora_data_maker.scrapers import existing_chromium as module
    from lora_data_maker.scrapers.existing_chromium import ExistingChromium

    profile = tmp_path / "google-profile"
    profile.mkdir()

    class FakeResponse:
        status = 200

        def read(self, limit):
            assert limit == 1024 * 1024 + 1
            return b'{"webSocketDebuggerUrl":"ws://127.0.0.1:9222/devtools/browser/test"}'

    class FakeConnection:
        def __init__(self, host, port, timeout):
            assert host == "127.0.0.1"
            assert port == 9222
            assert 0 < timeout <= 5.0

        def request(self, method, path, headers=None):
            assert method == "GET"
            assert path == "/json/version"
            assert headers == {"Connection": "close"}

        def getresponse(self):
            return FakeResponse()

        def close(self):
            pass

    monkeypatch.setattr(module, "HTTPConnection", FakeConnection)

    browser = ExistingChromium(
        user_data_dir=profile,
        port=9222,
        timeout=20,
    )
    assert browser._discover_websocket_url() == (
        "ws://127.0.0.1:9222/devtools/browser/test"
    )

def test_existing_chromium_rejects_remote_websocket_url(monkeypatch, tmp_path):
    import pytest

    from lora_data_maker.scrapers import existing_chromium as module
    from lora_data_maker.scrapers.existing_chromium import ExistingChromium

    profile = tmp_path / "google-profile"
    profile.mkdir()

    class FakeResponse:
        status = 200

        def read(self, limit):
            assert limit == 1024 * 1024 + 1
            return b'{"webSocketDebuggerUrl":"wss://attacker.example:9222/devtools/browser/test"}'

    class FakeConnection:
        def __init__(self, host, port, timeout):
            assert host == "127.0.0.1"
            assert port == 9222

        def request(self, method, path, headers=None):
            assert method == "GET"
            assert path == "/json/version"

        def getresponse(self):
            return FakeResponse()

        def close(self):
            pass

    monkeypatch.setattr(module, "HTTPConnection", FakeConnection)
    browser = ExistingChromium(user_data_dir=profile, port=9222, timeout=3)

    with pytest.raises(module.ScraperError, match="non-local"):
        browser._discover_websocket_url()

def test_existing_chromium_rejects_non_local_cdp_host(tmp_path):
    import pytest

    from lora_data_maker.scrapers.existing_chromium import ExistingChromium

    with pytest.raises(ValueError, match="CDP host must be local"):
        ExistingChromium(
            user_data_dir=tmp_path,
            host="192.168.1.10",
        )


def test_existing_chromium_rejects_invalid_debug_port(monkeypatch, tmp_path):
    import pytest

    from lora_data_maker.scrapers.existing_chromium import ExistingChromium

    profile = tmp_path / "google-profile"
    profile.mkdir()
    monkeypatch.setenv("CHROME_USER_DATA_DIR", str(profile))
    monkeypatch.setenv("CHROME_DEBUG_PORT", "not-a-port")

    with pytest.raises(ValueError, match="CHROME_DEBUG_PORT"):
        ExistingChromium.from_environment(timeout=20)


def test_playwright_google_uses_existing_profile_when_configured(monkeypatch, tmp_path):
    from lora_data_maker.scrapers import playwright as playwright_module
    from lora_data_maker.scrapers.existing_chromium import ExistingChromium

    profile = tmp_path / "google-profile"
    profile.mkdir()
    monkeypatch.setenv("CHROME_USER_DATA_DIR", str(profile))

    scraper = playwright_module.PlaywrightImagesScraper(
        provider="google",
        delay=0,
        retries=0,
        max_scrolls=1,
    )

    search_url = scraper._build_url("julia russian woman")
    browser = scraper._create_browser(search_url)

    assert isinstance(browser, ExistingChromium)
    assert browser.user_data_dir == profile.resolve()
    assert browser.initial_url == search_url
    browser.close()


def test_scraper_manager_forwards_bing_date_range(tmp_path):
    from lora_data_maker.scrapers.manager import ScraperManager

    manager = ScraperManager(
        backend="playwright-bing",
        output_dir=tmp_path / "raw",
        cache_dir=tmp_path / "cache",
        date_from="2019-01-01",
        date_to="2019-12-31",
    )

    assert manager.scraper.date_from == "2019-01-01"
    assert manager.scraper.date_to == "2019-12-31"


def test_scraper_manager_rejects_date_filter_for_other_backends(tmp_path):
    import pytest

    from lora_data_maker.scrapers.base import ScraperError
    from lora_data_maker.scrapers.manager import ScraperManager

    with pytest.raises(ScraperError, match="playwright-bing"):
        ScraperManager(
            backend="duckduckgo",
            output_dir=tmp_path / "raw",
            cache_dir=tmp_path / "cache",
            date_from="2019-01-01",
            date_to="2019-12-31",
        )


def test_chromium_prefers_bing_and_preserves_duckduckgo_metadata(monkeypatch):
    import json

    from lora_data_maker.scrapers import chromium as chromium_module
    from lora_data_maker.scrapers.base import ImageResult
    from lora_data_maker.scrapers.chromium import ChromiumImagesScraper

    class FakeDriver:
        closed = False

        def execute_script(self, script):
            if "querySelectorAll('a.iusc')" in script:
                return [
                    json.dumps(
                        {"murl": "https://cdn.example.com/bing.jpg"}
                    )
                ]
            return None

        def quit(self):
            self.closed = True

    calls = {}
    driver = FakeDriver()

    class FakeDuckDuckGo:
        def __init__(self, **kwargs):
            pass

        def search(self, query, *, limit):
            calls["query"] = query
            calls["limit"] = limit
            yield ImageResult(
                image_url="https://cdn.example.com/ddg.jpg",
                query=query,
                page_url="https://example.com/ddg-result",
                metadata={
                    "title": "Julia Patty portrait",
                    "source": "example.com",
                    "width": 640,
                },
            )

    scraper = ChromiumImagesScraper(delay=0, max_scrolls=1)
    monkeypatch.setattr(scraper, "_start_driver", lambda: driver)
    monkeypatch.setattr(scraper, "_load_page", lambda _driver, _url: True)
    monkeypatch.setattr(
        chromium_module,
        "DuckDuckGoImagesScraper",
        FakeDuckDuckGo,
    )
    monkeypatch.setattr(chromium_module.time, "sleep", lambda _seconds: None)

    results = list(scraper.search("Julia Patty", limit=2))

    assert [result.image_url for result in results] == [
        "https://cdn.example.com/bing.jpg",
        "https://cdn.example.com/ddg.jpg",
    ]
    assert results[0].metadata == {"provider": "bing"}
    assert results[1].metadata == {
        "title": "Julia Patty portrait",
        "source": "example.com",
        "width": 640,
        "provider": "duckduckgo",
    }
    assert results[1].page_url == "https://example.com/ddg-result"
    assert calls == {"query": "Julia Patty", "limit": 1}
    assert driver.closed is True


def test_chromium_uses_duckduckgo_when_bing_browser_is_unavailable(monkeypatch):
    from lora_data_maker.scrapers import chromium as chromium_module
    from lora_data_maker.scrapers.base import ImageResult, ScraperError
    from lora_data_maker.scrapers.chromium import ChromiumImagesScraper

    class FakeDuckDuckGo:
        SITE_URL = "https://duckduckgo.com/"

        def __init__(self, **kwargs):
            pass

        def search(self, query, *, limit):
            yield ImageResult(
                image_url="https://cdn.example.com/ddg.jpg",
                query=query,
                metadata={"title": "Julia Patty portrait"},
            )

    scraper = ChromiumImagesScraper(delay=0)
    monkeypatch.setattr(
        scraper,
        "_start_driver",
        lambda: (_ for _ in ()).throw(ScraperError("chromium unavailable")),
    )
    monkeypatch.setattr(
        chromium_module,
        "DuckDuckGoImagesScraper",
        FakeDuckDuckGo,
    )

    results = list(scraper.search("Julia Patty", limit=1))

    assert len(results) == 1
    assert results[0].image_url == "https://cdn.example.com/ddg.jpg"
    assert results[0].metadata == {
        "title": "Julia Patty portrait",
        "provider": "duckduckgo",
    }


def test_playwright_bing_ready_page_without_result_anchors_is_not_navigation_timeout():
    from lora_data_maker.scrapers.playwright import PlaywrightImagesScraper

    class FakeSession:
        def send(self, method, params=None, timeout=None):
            if method == "Page.navigate":
                return {"loaderId": "loader-1"}
            if method == "Runtime.evaluate":
                assert "document.querySelectorAll('a.iusc').length" in params["expression"]
                return {
                    "result": {
                        "value": [
                            "complete",
                            "https://www.bing.com/images/search?q=julia",
                            0,
                        ]
                    }
                }
            raise AssertionError(f"unexpected CDP method: {method}")

    class FakePage:
        def __init__(self):
            self.session = FakeSession()
            self._url = "about:blank"

    scraper = PlaywrightImagesScraper(
        provider="bing",
        timeout=1,
        retries=0,
        delay=0,
    )
    page = FakePage()

    scraper._goto(page, scraper._build_url("julia"))

    assert page._url == "https://www.bing.com/images/search?q=julia"


def test_playwright_bing_readiness_reports_verification_page():
    import pytest

    from lora_data_maker.scrapers.base import ScraperError
    from lora_data_maker.scrapers.playwright import PlaywrightImagesScraper

    class FakeSession:
        def send(self, method, params=None, timeout=None):
            if method == "Page.navigate":
                return {"loaderId": "loader-1"}
            if method == "Runtime.evaluate":
                return {
                    "result": {
                        "value": [
                            "complete",
                            "https://www.bing.com/images/search?q=julia",
                            0,
                            "Verify you are human",
                            "Please verify you are human before continuing.",
                        ]
                    }
                }
            raise AssertionError(f"unexpected CDP method: {method}")

    class FakePage:
        def __init__(self):
            self.session = FakeSession()
            self._url = "about:blank"

    scraper = PlaywrightImagesScraper(
        provider="bing",
        timeout=1,
        retries=0,
        delay=0,
    )

    with pytest.raises(ScraperError, match="verification/block page"):
        scraper._goto(FakePage(), scraper._build_url("julia"))



def test_existing_chromium_defaults_to_private_persistent_google_profile(
    monkeypatch, tmp_path
):
    from lora_data_maker.scrapers.existing_chromium import ExistingChromium

    monkeypatch.delenv("CHROME_USER_DATA_DIR", raising=False)
    monkeypatch.delenv("CHROME_AUTO_LAUNCH", raising=False)
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))

    browser = ExistingChromium.from_environment(timeout=20)

    assert browser.user_data_dir == (
        tmp_path / "data" / "lora-data-maker" / "google-profile"
    ).resolve()
    assert browser.auto_launch is True


def test_existing_chromium_launches_visible_browser_with_local_cdp(
    monkeypatch, tmp_path
):
    from lora_data_maker.scrapers import existing_chromium as module
    from lora_data_maker.scrapers.existing_chromium import ExistingChromium

    monkeypatch.setenv("DISPLAY", ":1")
    profile = tmp_path / "private-google-profile"
    calls = {}

    class FakeProcess:
        def poll(self):
            return None

    def fake_popen(args, **kwargs):
        calls["args"] = args
        calls["kwargs"] = kwargs
        return FakeProcess()

    monkeypatch.setattr(module.subprocess, "Popen", fake_popen)
    search_url = "https://www.google.com/search?tbm=isch&q=julia+russian+woman"
    browser = ExistingChromium(
        user_data_dir=profile,
        port=9222,
        auto_launch=True,
        executable="/mock/chromium-browser",
        initial_url=search_url,
    )

    browser._launch_browser()

    args = calls["args"]
    assert args[0] == "/mock/chromium-browser"
    assert f"--user-data-dir={profile.resolve()}" in args
    assert "--remote-debugging-port=9222" in args
    assert "--remote-debugging-address=127.0.0.1" in args
    assert "--remote-allow-origins=http://127.0.0.1:9222" in args
    assert search_url in args
    assert calls["kwargs"]["stdout"] is module.subprocess.DEVNULL
    assert calls["kwargs"]["stderr"] is module.subprocess.DEVNULL
    assert profile.is_dir()


def test_existing_chromium_only_matches_its_own_browser_process(tmp_path):
    from lora_data_maker.scrapers.existing_chromium import ExistingChromium

    profile = (tmp_path / "google-profile").resolve()
    proc_root = tmp_path / "proc"
    proc_root.mkdir()

    def write_process(pid, argv):
        process = proc_root / str(pid)
        process.mkdir()
        (process / "cmdline").write_bytes(b"\0".join(arg.encode() for arg in argv) + b"\0")

    write_process(101, [
        "/data/data/com.termux/files/usr/bin/chromium-browser",
        f"--user-data-dir={profile}",
        "--remote-debugging-port=9222",
    ])
    write_process(102, [
        "/data/data/com.termux/files/usr/bin/chromium-browser",
        f"--user-data-dir={profile}",
        "--remote-debugging-port=9222",
        "--type=renderer",
    ])
    write_process(103, [
        "/data/data/com.termux/files/usr/bin/chromium-browser",
        f"--user-data-dir={(tmp_path / 'other-profile').resolve()}",
        "--remote-debugging-port=9222",
    ])

    browser = ExistingChromium(user_data_dir=profile, port=9222, auto_launch=True)
    assert browser._matching_browser_pids(proc_root) == [101]


def test_existing_chromium_retries_after_cdp_origin_rejection(monkeypatch, tmp_path):
    import sys
    import types

    from lora_data_maker.scrapers.existing_chromium import ExistingChromium

    profile = tmp_path / "google-profile"
    profile.mkdir()
    calls = {"connect": 0, "restart": 0}
    websocket_url = "ws://127.0.0.1:9222/devtools/browser/test"

    class FakeCDPConnection:
        def __init__(self, url, *, timeout):
            assert url == websocket_url
            assert timeout > 0

        def connect(self):
            calls["connect"] += 1
            if calls["connect"] == 1:
                raise RuntimeError(
                    "Handshake status 403 Forbidden: Rejected an incoming WebSocket "
                    "connection. Use --remote-allow-origins=* to allow connections."
                )
            return self

    package = types.ModuleType("playwright_termux")
    package.__path__ = []
    cdp_module = types.ModuleType("playwright_termux.cdp")
    cdp_module.CDPConnection = FakeCDPConnection
    monkeypatch.setitem(sys.modules, "playwright_termux", package)
    monkeypatch.setitem(sys.modules, "playwright_termux.cdp", cdp_module)

    browser = ExistingChromium(user_data_dir=profile, port=9222, auto_launch=True)
    monkeypatch.setattr(browser, "_discover_websocket_url", lambda: websocket_url)
    monkeypatch.setattr(browser, "_wait_for_websocket_url", lambda: websocket_url)

    def restart():
        calls["restart"] += 1

    monkeypatch.setattr(browser, "_restart_browser_for_cdp_origin", restart)
    assert browser.connect() is browser
    assert calls == {"connect": 2, "restart": 1}
    browser.close()

def test_existing_chromium_starts_termux_x11_when_display_is_unset(
    monkeypatch, tmp_path
):
    from types import SimpleNamespace

    from lora_data_maker.scrapers import existing_chromium as module
    from lora_data_maker.scrapers.existing_chromium import ExistingChromium

    monkeypatch.delenv("DISPLAY", raising=False)
    monkeypatch.setenv("TMPDIR", str(tmp_path))
    monkeypatch.setenv("LORA_DATA_MAKER_X11_DISPLAY", ":1")
    socket_path = tmp_path / ".X11-unix" / "X1"
    calls = {}

    def fake_which(name):
        return {"termux-x11": "/mock/termux-x11", "am": "/mock/am"}.get(name)

    def fake_run(args, **kwargs):
        calls["activity"] = args
        return SimpleNamespace(returncode=0, stdout="Starting", stderr="")

    class FakeProcess:
        def poll(self):
            return None

    def fake_popen(args, **kwargs):
        calls["server"] = args
        calls["server_kwargs"] = kwargs
        socket_path.parent.mkdir(parents=True, exist_ok=True)
        socket_path.touch()
        return FakeProcess()

    monkeypatch.setattr(module.shutil, "which", fake_which)
    monkeypatch.setattr(module.subprocess, "run", fake_run)
    monkeypatch.setattr(module.subprocess, "Popen", fake_popen)

    browser = ExistingChromium(
        user_data_dir=tmp_path / "google-profile",
        auto_launch=True,
        executable="/mock/chromium-browser",
    )
    assert browser._ensure_display() == ":1"
    assert module.os.environ["DISPLAY"] == ":1"
    assert calls["activity"] == [
        "/mock/am", "start", "--user", "0", "-n",
        "com.termux.x11/com.termux.x11.MainActivity",
    ]
    assert calls["server"] == ["/mock/termux-x11", ":1"]
    assert calls["server_kwargs"]["env"]["DISPLAY"] == ":1"


def test_existing_chromium_reports_missing_termux_x11_package(
    monkeypatch, tmp_path
):
    import pytest

    from lora_data_maker.scrapers.base import ScraperError
    from lora_data_maker.scrapers.existing_chromium import ExistingChromium

    monkeypatch.delenv("DISPLAY", raising=False)
    from lora_data_maker.scrapers import existing_chromium as module
    monkeypatch.setattr(module.shutil, "which", lambda name: None)
    browser = ExistingChromium(
        user_data_dir=tmp_path / "google-profile",
        auto_launch=True,
        executable="/mock/chromium-browser",
    )

    with pytest.raises(ScraperError, match="termux-x11"):
        browser._ensure_display()


def test_playwright_google_defaults_to_persistent_chromium_profile(
    monkeypatch, tmp_path
):
    from lora_data_maker.scrapers.existing_chromium import ExistingChromium
    from lora_data_maker.scrapers.playwright import PlaywrightImagesScraper

    monkeypatch.delenv("CHROME_USER_DATA_DIR", raising=False)
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    browser = PlaywrightImagesScraper(provider="google")._create_browser()

    assert isinstance(browser, ExistingChromium)
    assert browser.auto_launch is True
    assert browser.user_data_dir == (
        tmp_path / "data" / "lora-data-maker" / "google-profile"
    ).resolve()
