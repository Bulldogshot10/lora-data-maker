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


def test_chromium_extracts_google_original_urls():
    from lora_data_maker.scrapers.chromium import ChromiumImagesScraper

    hrefs = [
        "https://www.google.com/imgres?imgurl=https%3A%2F%2Fcdn.example.com%2Fa.jpg&imgrefurl=https%3A%2F%2Fexample.com%2Fa",
        "/imgres?imgurl=https%3A%2F%2Fcdn.example.com%2Fb.png",
        "https://www.google.com/search?q=not-an-image",
    ]

    assert ChromiumImagesScraper._extract_image_urls(hrefs) == [
        "https://cdn.example.com/a.jpg",
        "https://cdn.example.com/b.png",
    ]


def test_chromium_rejects_google_thumbnail_urls():
    from lora_data_maker.scrapers.chromium import ChromiumImagesScraper

    hrefs = [
        "https://www.google.com/imgres?imgurl=https%3A%2F%2Fencrypted-tbn0.gstatic.com%2Fimages%2Fthumb.jpg"
    ]

    assert ChromiumImagesScraper._extract_image_urls(hrefs) == []


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

    scraper = ChromiumImagesScraper(
        timeout=7,
        profile_root=tmp_path / "chromium-profiles",
    )
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
    assert not (tmp_path / "chromium-profiles").exists()


def test_chromium_preserves_candidate_order(monkeypatch):
    import sys
    import types

    from lora_data_maker.scrapers.chromium import ChromiumImagesScraper

    class TimeoutException(Exception):
        pass

    selenium = types.ModuleType("selenium")
    common = types.ModuleType("selenium.common")
    exceptions = types.ModuleType("selenium.common.exceptions")
    exceptions.TimeoutException = TimeoutException
    exceptions.WebDriverException = WebDriverException
    common.exceptions = exceptions

    monkeypatch.setitem(sys.modules, "selenium", selenium)
    monkeypatch.setitem(sys.modules, "selenium.common", common)
    monkeypatch.setitem(sys.modules, "selenium.common.exceptions", exceptions)

    class FakeDriver:
        title = "Google Images"

        def __init__(self):
            self.quit_called = False

        def get(self, _url):
            return None

        def execute_script(self, script):
            if "document.readyState" in script:
                return "complete"
            if "document.body" in script:
                return "normal results"
            if "querySelectorAll" in script:
                return [
                    "https://www.google.com/imgres?imgurl=https%3A%2F%2Fcdn.example.com%2Ffirst.jpg",
                    "https://www.google.com/imgres?imgurl=https%3A%2F%2Fcdn.example.com%2Fsecond.jpg",
                ]
            return None

        def quit(self):
            self.quit_called = True

    driver = FakeDriver()
    scraper = ChromiumImagesScraper(delay=0)
    monkeypatch.setattr(scraper, "_start_driver", lambda: driver)

    results = list(scraper.search("julia", limit=2))

    assert [item.image_url for item in results] == [
        "https://cdn.example.com/first.jpg",
        "https://cdn.example.com/second.jpg",
    ]
    assert driver.quit_called is True


def test_chromium_page_timeout_retries_and_falls_back_to_bing(monkeypatch):
    import sys
    import types

    from lora_data_maker.scrapers.chromium import ChromiumImagesScraper

    class TimeoutException(Exception):
        pass

    WebDriverException = type("WebDriverException", (Exception,), {})

    common = types.ModuleType("selenium.common")
    exceptions = types.ModuleType("selenium.common.exceptions")
    exceptions.TimeoutException = TimeoutException
    common.exceptions = exceptions

    monkeypatch.setitem(
        sys.modules,
        "selenium.common",
        common,
    )
    monkeypatch.setitem(
        sys.modules,
        "selenium.common.exceptions",
        exceptions,
    )

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

    assert scraper._load_page(
        driver,
        "https://www.google.com/search?tbm=isch&q=julia",
        provider="google",
    ) is False
    assert driver.get_calls == 2
    assert driver.stop_calls == 2


def test_chromium_continues_to_bing_when_google_has_too_few_candidates(monkeypatch):
    from lora_data_maker.scrapers.chromium import ChromiumImagesScraper

    class FakeDriver:
        title = "Images"

        def __init__(self):
            self.provider_calls = 0
            self.quit_called = False

        def get(self, url):
            self.provider_calls += 1
            self.last_url = url

        def execute_script(self, script):
            if "document.body" in script:
                return "normal results"
            if "querySelectorAll" in script:
                if "a[href]" in script and "i.iusc" not in script:
                    return [
                        "https://www.google.com/imgres?imgurl=https%3A%2F%2Fcdn.example.com%2Fgoogle.jpg"
                    ]
                return [
                    '{"murl":"https://cdn.example.com/bing.jpg"}'
                ]
            return "complete"

        def quit(self):
            self.quit_called = True

    driver = FakeDriver()
    scraper = ChromiumImagesScraper(delay=0, retries=0, max_scrolls=1)
    monkeypatch.setattr(scraper, "_start_driver", lambda: driver)

    results = list(scraper.search("julia", limit=2))

    assert [item.image_url for item in results] == [
        "https://cdn.example.com/google.jpg",
        "https://cdn.example.com/bing.jpg",
    ]
    assert [item.metadata["provider"] for item in results] == [
        "google",
        "bing",
    ]
    assert driver.quit_called is True
