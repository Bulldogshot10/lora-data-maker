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
