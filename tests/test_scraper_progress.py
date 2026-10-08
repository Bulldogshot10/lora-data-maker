from pathlib import Path
import threading

from lora_data_maker.scrapers.base import ImageResult, ScraperCancelledError
from lora_data_maker.scrapers.downloader import DownloadResult
from lora_data_maker.scrapers.manager import ScraperManager


def test_scraper_manager_reports_real_download_progress(tmp_path: Path) -> None:
    class FakeScraper:
        def search(self, query: str, *, limit: int):
            assert query == "julia"
            assert limit == 6
            yield ImageResult("https://example.com/1.jpg", query=query, rank=1)
            yield ImageResult("https://example.com/2.jpg", query=query, rank=2)

    class FakeDownloader:
        def download(self, url, output_dir, *, index, archive=None, cancel_event=None):
            path = Path(output_dir) / f"{index:06d}.jpg"
            path.write_bytes(b"fake")
            return DownloadResult(
                url=url,
                path=path,
                sha256="a" * 64,
                file_size=4,
                content_type="image/jpeg",
            )

    class FakeArchive:
        def __contains__(self, value):
            return False

        def add(self, value):
            return None

    manager = ScraperManager.__new__(ScraperManager)
    manager.backend = "fake"
    manager.output_dir = tmp_path / "raw"
    manager.output_dir.mkdir()
    manager.cache_dir = tmp_path / "cache"
    manager.cache_dir.mkdir()
    manager.scraper = FakeScraper()
    manager.downloader = FakeDownloader()
    manager.archive = FakeArchive()
    manager.last_search_error = None

    progress = []
    downloads, failures = manager.run(
        "julia",
        count=2,
        known_urls=[],
        progress=lambda current, total, message: progress.append(
            (current, total, message)
        ),
    )

    assert failures == []
    assert len(downloads) == 2
    assert progress[0] == (0, 2, "searching for up to 6 candidates...")
    assert progress[-1] == (2, 2, "scrape finished: downloaded 2/2")
    assert [item[0] for item in progress] == [0, 1, 2, 2]


def test_scraper_manager_cancels_without_discarding_completed_downloads(
    tmp_path: Path,
) -> None:
    manager = ScraperManager.__new__(ScraperManager)
    manager.backend = "fake"
    manager.output_dir = tmp_path / "raw"
    manager.output_dir.mkdir()
    manager.cache_dir = tmp_path / "cache"
    manager.cache_dir.mkdir()
    manager.archive = type(
        "Archive",
        (),
        {"__contains__": lambda self, value: False, "add": lambda self, value: None},
    )()
    manager.last_search_error = None

    results = [
        ImageResult("https://example.com/1.jpg", query="julia", rank=1),
        ImageResult("https://example.com/2.jpg", query="julia", rank=2),
    ]

    class FakeScraper:
        def search(self, query, *, limit):
            return iter(results)

    class FakeDownloader:
        def download(
            self,
            url,
            output_dir,
            *,
            index,
            archive=None,
            cancel_event=None,
        ):
            if url.endswith("/2.jpg"):
                raise ScraperCancelledError("scrape cancelled")
            path = Path(output_dir) / f"{index:06d}.jpg"
            path.write_bytes(b"fake")
            return DownloadResult(
                url=url,
                path=path,
                sha256="a" * 64,
                file_size=4,
                content_type="image/jpeg",
            )

    manager.scraper = FakeScraper()
    manager.downloader = FakeDownloader()

    downloads, failures = manager.run(
        "julia",
        count=2,
        known_urls=(),
        cancel_event=threading.Event(),
    )

    assert [item.url for item in downloads] == ["https://example.com/1.jpg"]
    assert failures == []
    assert manager.cancelled is True
