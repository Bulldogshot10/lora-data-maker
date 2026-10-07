from lora_data_maker.scrapers.archive import DownloadArchive
from lora_data_maker.scrapers.google import GoogleImagesScraper


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
    )

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
        def download(self, url, output_dir, *, index, archive):
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
        def download(self, url, output_dir, *, index, archive):
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

def test_chromium_continues_to_bing_when_duckduckgo_has_too_few_candidates(monkeypatch):
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
        "https://cdn.example.com/ddg.jpg",
        "https://cdn.example.com/bing.jpg",
    ]
    assert [item.metadata["provider"] for item in results] == [
        "duckduckgo",
        "bing",
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
    }


def test_playwright_google_prefers_original_ou_url():
    from lora_data_maker.scrapers.playwright import PlaywrightImagesScraper

    body = r'''
    <script>
    {"ou":"https://cdn.example.com/original.jpg","tu":"https://encrypted-tbn.example/thumb.jpg"}
    </script>
    '''

    assert PlaywrightImagesScraper._extract_google_urls(body) == [
        "https://cdn.example.com/original.jpg"
    ]


def test_playwright_google_rejects_verification_page():
    import pytest

    from lora_data_maker.scrapers.base import ScraperError
    from lora_data_maker.scrapers.playwright import PlaywrightImagesScraper

    with pytest.raises(ScraperError, match="verification"):
        PlaywrightImagesScraper._extract_google_urls(
            "Our systems have detected unusual traffic from your computer network"
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
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return None

        def read(self):
            return b'{"webSocketDebuggerUrl":"ws://127.0.0.1:9222/devtools/browser/test"}'

    def fake_urlopen(url, timeout):
        assert url == "http://127.0.0.1:9222/json/version"
        assert 0 < timeout <= 5.0
        return FakeResponse()

    monkeypatch.setattr(module, "urlopen", fake_urlopen)

    browser = ExistingChromium(
        user_data_dir=profile,
        port=9222,
        timeout=20,
    )
    assert (
        browser._discover_websocket_url()
        == "ws://127.0.0.1:9222/devtools/browser/test"
    )


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

    browser = scraper._create_browser()

    assert isinstance(browser, ExistingChromium)
    assert browser.user_data_dir == profile.resolve()
    browser.close()


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
                self.evaluate_calls += 1
                href = (
                    "about:blank"
                    if self.evaluate_calls == 1
                    else "https://www.bing.com/images/search?q=julia"
                )
                return {"result": {"value": ["complete", href]}}
            raise AssertionError(f"unexpected CDP method: {method}")

    class FakePage:
        def __init__(self):
            self.closed = False
            self.session = FakeSession()
            self.urls = []

        def content(self):
            return (
                '<a class="iusc" m=\'{"murl":"https://cdn.example.com/a.jpg"}\'></a>'
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


def test_playwright_scraper_validates_provider():
    import pytest

    from lora_data_maker.scrapers.playwright import PlaywrightImagesScraper

    with pytest.raises(ValueError, match="provider"):
        PlaywrightImagesScraper(provider="duckduckgo")
