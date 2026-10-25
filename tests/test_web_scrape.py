from pathlib import Path
import threading
import time

from fastapi.testclient import TestClient

from lora_data_maker.config import write_default_config
from lora_data_maker.dataset import DatasetProject
from lora_data_maker.paths import ProjectPaths
from lora_data_maker.scrapers.downloader import DownloadResult
from lora_data_maker.web import create_app


def make_project(tmp_path: Path) -> Path:
    root = tmp_path / "dataset"
    paths = ProjectPaths.from_root(root)
    paths.create_directories()
    write_default_config(
        paths.config,
        name="test",
        subject="person",
        trigger="person",
    )
    DatasetProject.create(paths)
    return root


def test_web_exposes_scraper_controls_and_progress(tmp_path: Path, monkeypatch) -> None:
    root = make_project(tmp_path)

    class FakeScraperManager:
        last_search_error = None

        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def run(self, query, *, count, known_urls=(), progress=None, cancel_event=None):
            assert query == "julia russian woman portrait"
            assert count == 3
            assert list(known_urls) == []
            if progress is not None:
                progress(0, count, "searching...")
            output = root / "raw" / "000001.jpg"
            output.write_bytes(b"fake image")
            if progress is not None:
                progress(1, count, "downloaded 1/3")
            return [
                DownloadResult(
                    url="https://example.com/image.jpg",
                    path=output,
                    sha256="a" * 64,
                    file_size=10,
                    content_type="image/jpeg",
                )
            ], []

    monkeypatch.setattr("lora_data_maker.web.ScraperManager", FakeScraperManager)

    client = TestClient(create_app(root))

    html = client.get("/")
    assert html.status_code == 200
    assert 'id="scraper_backend"' in html.text
    assert 'id="scrape_query"' in html.text
    assert 'id="scrape_count"' in html.text
    assert 'id="progress-fill"' in html.text

    settings = client.post(
        "/api/settings",
        json={
            "name": "test",
            "subject": "person",
            "trigger": "person",
            "preset": "sweetspot",
            "scraper_backend": "duckduckgo",
            "scrape_query": "julia russian woman portrait",
            "scrape_count": 3,
            "mode": "cover",
            "width": 512,
            "height": 512,
            "quality": 95,
            "face_crop": False,
            "face_padding": 1.75,
            "person_check": False,
            "max_persons": 1,
            "person_threshold": 0.35,
            "min_subject_area": 0.12,
            "min_subject_visibility": 0.35,
        },
    )
    assert settings.status_code == 200

    state = client.get("/api/state").json()
    assert state["scraper"]["backend"] == "duckduckgo"
    assert state["scraper"]["query"] == "julia russian woman portrait"
    assert state["scraper"]["count"] == 3

    started = client.post("/api/run/scrape")
    assert started.status_code == 200
    job_id = started.json()["job_id"]

    deadline = time.monotonic() + 3
    job = None
    while time.monotonic() < deadline:
        job = client.get(f"/api/jobs/{job_id}").json()
        if job["status"] != "running":
            break
        time.sleep(0.01)

    assert job is not None
    assert job["status"] == "done"
    assert job["progress"]["current"] == 1
    assert job["progress"]["total"] == 3
    assert job["progress"]["percent"] == 33.3
    assert "downloaded 1/3" in job["progress"]["label"]

    final_state = client.get("/api/state").json()
    assert final_state["counts"]["downloaded"] == 1


def test_web_safe_search_setting_persists_and_can_be_disabled(tmp_path: Path) -> None:
    root = make_project(tmp_path)
    client = TestClient(create_app(root))

    initial_state = client.get("/api/state").json()
    assert initial_state["scraper"]["safe_search"] is False

    payload = {
        "name": "test",
        "subject": "person",
        "trigger": "person",
        "preset": "sweetspot",
        "scraper_backend": "playwright-google",
        "scrape_query": "portrait",
        "scrape_count": 1,
        "scrape_safe_search": True,
        "mode": "cover",
        "width": 512,
        "height": 512,
        "quality": 95,
        "face_crop": False,
        "face_padding": 1.75,
        "person_check": False,
        "max_persons": 1,
        "person_threshold": 0.35,
        "min_subject_area": 0.12,
        "min_subject_visibility": 0.35,
        "text_check": True,
        "text_confidence": 0.5,
        "text_min_area_ratio": 0.0004,
    }
    saved = client.post("/api/settings", json=payload)
    assert saved.status_code == 200
    state = client.get("/api/state").json()
    assert state["scraper"]["safe_search"] is True

    config = (root / "config.toml").read_text(encoding="utf-8")
    assert "safe_search = true" in config

    payload["scrape_safe_search"] = False
    saved_off = client.post("/api/settings", json=payload)
    assert saved_off.status_code == 200
    state_off = client.get("/api/state").json()
    assert state_off["scraper"]["safe_search"] is False
    config_off = (root / "config.toml").read_text(encoding="utf-8")
    assert "safe_search = false" in config_off


def test_web_can_cancel_running_scrape(tmp_path: Path, monkeypatch) -> None:
    root = make_project(tmp_path)
    started = threading.Event()

    class FakeScraperManager:
        last_search_error = None
        cancelled = False

        def __init__(self, **kwargs):
            pass

        def run(
            self,
            query,
            *,
            count,
            known_urls=(),
            progress=None,
            cancel_event=None,
        ):
            started.set()
            while cancel_event is not None and not cancel_event.is_set():
                time.sleep(0.01)
            self.cancelled = True
            if progress is not None:
                progress(0, count, "scrape cancelled: downloaded 0/3")
            return [], []

    monkeypatch.setattr("lora_data_maker.web.ScraperManager", FakeScraperManager)

    client = TestClient(create_app(root))
    settings = client.post(
        "/api/settings",
        json={
            "name": "test",
            "subject": "person",
            "trigger": "person",
            "preset": "sweetspot",
            "scraper_backend": "duckduckgo",
            "scrape_query": "julia",
            "scrape_count": 3,
            "mode": "cover",
            "width": 512,
            "height": 512,
            "quality": 95,
            "face_crop": False,
            "face_padding": 1.75,
            "person_check": False,
            "max_persons": 1,
            "person_threshold": 0.35,
            "min_subject_area": 0.12,
            "min_subject_visibility": 0.35,
        },
    )
    assert settings.status_code == 200

    started_job = client.post("/api/run/scrape")
    assert started_job.status_code == 200
    job_id = started_job.json()["job_id"]
    assert started.wait(3)

    cancelled = client.post(f"/api/jobs/{job_id}/cancel")
    assert cancelled.status_code == 200
    assert cancelled.json()["cancel_requested"] is True

    deadline = time.monotonic() + 3
    job = None
    while time.monotonic() < deadline:
        job = client.get(f"/api/jobs/{job_id}").json()
        if job["status"] != "running":
            break
        time.sleep(0.01)

    assert job is not None
    assert job["status"] == "cancelled"
    assert "downloaded 0/3" in job["message"]



def test_web_settings_allow_google_custom_date_range(tmp_path: Path) -> None:
    import tomllib

    root = make_project(tmp_path)
    client = TestClient(create_app(root))
    payload = {
        "name": "test",
        "subject": "person",
        "trigger": "person",
        "preset": "sweetspot",
        "scraper_backend": "playwright-google",
        "scrape_query": "portrait",
        "scrape_count": 2,
        "scrape_date_mode": "range",
        "scrape_year": 0,
        "scrape_date_from": "2019-04-01",
        "scrape_date_to": "2019-09-30",
        "scrape_safe_search": False,
        "mode": "cover",
        "width": 512,
        "height": 512,
        "quality": 95,
        "face_crop": False,
        "face_padding": 1.75,
        "person_check": False,
        "max_persons": 1,
        "person_threshold": 0.35,
        "min_subject_area": 0.12,
        "min_subject_visibility": 0.35,
        "text_check": True,
        "text_confidence": 0.5,
        "text_min_area_ratio": 0.0004,
    }

    response = client.post("/api/settings", json=payload)
    assert response.status_code == 200, response.text

    with (root / "config.toml").open("rb") as stream:
        saved = tomllib.load(stream)
    assert saved["scraper"]["backend"] == "playwright-google"
    assert saved["scraper"]["date_from"] == "2019-04-01"
    assert saved["scraper"]["date_to"] == "2019-09-30"
