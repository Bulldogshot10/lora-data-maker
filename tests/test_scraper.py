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

def test_manager_fallbacks_are_sequential_and_non_overlapping(tmp_path):
    from lora_data_maker.scrapers.base import ImageResult
    from lora_data_maker.scrapers.manager import ScraperManager

    manager = ScraperManager(
        backend="google",
        fallbacks=["duckduckgo", "chromium"],
        output_dir=tmp_path / "raw",
        cache_dir=tmp_path / "cache",
        delay=0,
    )

    calls = []

    class FakeScraper:
        def __init__(self, name, results):
            self.name = name
            self.results = list(results)

        def search(self, query, *, limit):
            calls.append(self.name)
            yield from self.results

    manager.scrapers["google"] = FakeScraper("google", [
        ImageResult("https://CDN.example.com/a.jpg#fragment"),
    ])
    manager.scrapers["duckduckgo"] = FakeScraper("duckduckgo", [
        ImageResult("https://cdn.example.com/a.jpg"),
        ImageResult("https://cdn.example.com/b.jpg"),
    ])
    manager.scrapers["chromium"] = FakeScraper("chromium", [
        ImageResult("https://cdn.example.com/c.jpg"),
    ])
    manager.scraper = manager.scrapers["google"]

    class FakeDownload:
        def __init__(self, path, url):
            self.path = path
            self.url = url
            self.sha256 = url
            self.file_size = 1
            self.content_type = "image/jpeg"

    downloaded_urls = []

    def fake_download(url, output_dir, *, index, archive):
        downloaded_urls.append(url)
        path = tmp_path / "raw" / f"{index:06d}.jpg"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"x")
        return FakeDownload(path, url)

    manager.downloader.download = fake_download
    downloads, failures = manager.run("julia", count=3)

    assert failures == []
    assert len(downloads) == 3
    assert calls == ["google", "duckduckgo", "chromium"]
    assert downloaded_urls == [
        "https://cdn.example.com/a.jpg",
        "https://cdn.example.com/b.jpg",
        "https://cdn.example.com/c.jpg",
    ]


def test_manager_stops_before_fallback_when_target_is_reached(tmp_path):
    from lora_data_maker.scrapers.base import ImageResult
    from lora_data_maker.scrapers.manager import ScraperManager

    manager = ScraperManager(
        backend="google",
        fallbacks=["duckduckgo"],
        output_dir=tmp_path / "raw",
        cache_dir=tmp_path / "cache",
        delay=0,
    )
    calls = []

    class FakeScraper:
        def __init__(self, name, results):
            self.name = name
            self.results = list(results)

        def search(self, query, *, limit):
            calls.append(self.name)
            yield from self.results

    manager.scrapers["google"] = FakeScraper("google", [
        ImageResult("https://cdn.example.com/a.jpg"),
        ImageResult("https://cdn.example.com/b.jpg"),
    ])
    manager.scrapers["duckduckgo"] = FakeScraper("duckduckgo", [])
    manager.scraper = manager.scrapers["google"]

    class FakeDownload:
        def __init__(self, path, url):
            self.path = path
            self.url = url
            self.sha256 = url
            self.file_size = 1
            self.content_type = "image/jpeg"

    def fake_download(url, output_dir, *, index, archive):
        path = tmp_path / "raw" / f"{index:06d}.jpg"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"x")
        return FakeDownload(path, url)

    manager.downloader.download = fake_download
    downloads, failures = manager.run("julia", count=2)

    assert len(downloads) == 2
    assert failures == []
    assert calls == ["google"]


def test_manager_discards_exact_content_overlap_between_engines(tmp_path):
    from lora_data_maker.scrapers.base import ImageResult
    from lora_data_maker.scrapers.manager import ScraperManager

    manager = ScraperManager(
        backend="google",
        fallbacks=["duckduckgo"],
        output_dir=tmp_path / "raw",
        cache_dir=tmp_path / "cache",
        delay=0,
    )

    class FakeScraper:
        def __init__(self, results):
            self.results = list(results)

        def search(self, query, *, limit):
            yield from self.results

    manager.scrapers["google"] = FakeScraper([
        ImageResult("https://cdn.example.com/a.jpg"),
    ])
    manager.scrapers["duckduckgo"] = FakeScraper([
        ImageResult("https://other.example.com/a.jpg"),
        ImageResult("https://other.example.com/b.jpg"),
    ])
    manager.scraper = manager.scrapers["google"]

    class FakeDownload:
        def __init__(self, path, url, digest):
            self.path = path
            self.url = url
            self.sha256 = digest
            self.file_size = 1
            self.content_type = "image/jpeg"

    def fake_download(url, output_dir, *, index, archive):
        path = tmp_path / "raw" / f"{index:06d}.jpg"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"x")
        digest = "same" if url.endswith("/a.jpg") else "unique"
        return FakeDownload(path, url, digest)

    manager.downloader.download = fake_download
    downloads, failures = manager.run("julia", count=2)

    assert failures == []
    assert len(downloads) == 1
    assert downloads[0].url == "https://cdn.example.com/a.jpg"
    assert not (tmp_path / "raw" / "000002.jpg").exists()


def test_manager_falls_back_after_search_error(tmp_path):
    from lora_data_maker.scrapers.base import ImageResult, ScraperError
    from lora_data_maker.scrapers.manager import ScraperManager

    manager = ScraperManager(
        backend="google",
        fallbacks=["duckduckgo"],
        output_dir=tmp_path / "raw",
        cache_dir=tmp_path / "cache",
        delay=0,
    )
    calls = []

    class FailingScraper:
        def search(self, query, *, limit):
            calls.append("google")
            raise ScraperError("google search failed")
            yield  # pragma: no cover

    class WorkingScraper:
        def search(self, query, *, limit):
            calls.append("duckduckgo")
            yield ImageResult("https://cdn.example.com/a.jpg")

    manager.scrapers["google"] = FailingScraper()
    manager.scrapers["duckduckgo"] = WorkingScraper()
    manager.scraper = manager.scrapers["google"]

    class FakeDownload:
        def __init__(self, path, url):
            self.path = path
            self.url = url
            self.sha256 = url
            self.file_size = 1
            self.content_type = "image/jpeg"

    def fake_download(url, output_dir, *, index, archive):
        path = tmp_path / "raw" / f"{index:06d}.jpg"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"x")
        return FakeDownload(path, url)

    manager.downloader.download = fake_download
    downloads, failures = manager.run("julia", count=1)

    assert len(downloads) == 1
    assert calls == ["google", "duckduckgo"]
    assert len(failures) == 1
    assert failures[0][1] == "google search failed"


def test_manager_rejects_invalid_fallback_chain(tmp_path):
    from lora_data_maker.scrapers.manager import ScraperManager
    import pytest

    with pytest.raises(Exception, match="duplicate scraper backend"):
        ScraperManager(
            backend="google",
            fallbacks=["duckduckgo", "duckduckgo"],
            output_dir=tmp_path / "raw",
            cache_dir=tmp_path / "cache",
        )

    with pytest.raises(Exception, match="direct cannot have fallback"):
        ScraperManager(
            backend="direct",
            fallbacks=["duckduckgo"],
            output_dir=tmp_path / "raw",
            cache_dir=tmp_path / "cache",
        )

def test_manager_rejects_provider_overlap_in_fallback_chain(tmp_path):
    from lora_data_maker.scrapers.manager import ScraperManager
    import pytest

    with pytest.raises(Exception, match="because chromium already uses Google"):
        ScraperManager(
            backend="google",
            fallbacks=["chromium"],
            output_dir=tmp_path / "raw",
            cache_dir=tmp_path / "cache",
        )

    with pytest.raises(Exception, match="because chromium already uses Google"):
        ScraperManager(
            backend="chromium",
            fallbacks=["google"],
            output_dir=tmp_path / "raw",
            cache_dir=tmp_path / "cache",
        )
