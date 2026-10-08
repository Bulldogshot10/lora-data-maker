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
    write_default_config(paths.config, name="test", subject="person", trigger="person")
    DatasetProject.create(paths)
    return root


def test_web_routes_and_state(tmp_path: Path) -> None:
    root = make_project(tmp_path)
    client = TestClient(create_app(root))

    response = client.get("/")
    assert response.status_code == 200
    assert "LoRA Data Maker" in response.text

    state = client.get("/api/state")
    assert state.status_code == 200
    assert state.json()["project"]["subject"] == "person"

    settings = client.post(
        "/api/settings",
        json={
            "name": "test",
            "subject": "new_person",
            "trigger": "new_person",
            "preset": "sweetspot",
            "mode": "cover",
            "width": 768,
            "height": 512,
            "quality": 92,
            "face_crop": False,
            "face_padding": 1.75,
        },
    )
    assert settings.status_code == 200

    updated = client.get("/api/state").json()
    assert updated["project"]["subject"] == "new_person"
    assert updated["processing"]["width"] == 768
    assert updated["processing"]["height"] == 512
    assert updated["processing"]["quality"] == 92

    joytag = client.post(
        "/api/settings",
        json={
            "name": "test",
            "subject": "new_person",
            "trigger": "new_person",
            "preset": "joytag",
            "mode": "cover",
            "width": 768,
            "height": 512,
            "quality": 92,
            "face_crop": False,
            "face_padding": 1.75,
        },
    )
    assert joytag.status_code == 200

    joytag_state = client.get("/api/state").json()
    assert joytag_state["tagger"]["backend"] == "joytag"
    assert joytag_state["tagger"]["preset"] == "joytag"


def test_new_project_switches_active_project(tmp_path: Path) -> None:
    root = make_project(tmp_path)
    client = TestClient(create_app(root))

    response = client.post(
        "/api/project/new",
        json={"name": "second", "subject": "cat", "trigger": "catx"},
    )
    assert response.status_code == 200
    assert response.json()["name"] == "second"

    state = client.get("/api/state").json()
    assert state["project"]["name"] == "second"
    assert state["project"]["subject"] == "cat"
    assert state["project"]["trigger"] == "catx"



def test_web_quality_settings_round_trip(tmp_path: Path) -> None:
    root = make_project(tmp_path)
    client = TestClient(create_app(root))

    response = client.post(
        "/api/settings",
        json={
            "name": "test",
            "subject": "person",
            "trigger": "person",
            "preset": "sweetspot",
            "mode": "cover",
            "width": 512,
            "height": 512,
            "quality": 95,
            "face_crop": False,
            "face_padding": 1.75,
            "person_check": True,
            "max_persons": 1,
            "person_threshold": 0.4,
            "min_subject_area": 0.15,
            "min_subject_visibility": 0.4,
        },
    )
    assert response.status_code == 200

    state = client.get("/api/state").json()
    assert state["quality"]["person_check"] is True
    assert state["quality"]["max_persons"] == 1
    assert state["quality"]["person_threshold"] == 0.4
    assert state["quality"]["min_subject_area"] == 0.15
    assert state["quality"]["min_subject_visibility"] == 0.4


def test_dataset_explorer_lists_and_switches_projects(tmp_path: Path) -> None:
    root = make_project(tmp_path)
    second = tmp_path / "second"
    second_paths = ProjectPaths.from_root(second)
    second_paths.create_directories()
    write_default_config(
        second_paths.config,
        name="second",
        subject="cat",
        trigger="catx",
    )
    DatasetProject.create(second_paths)

    client = TestClient(create_app(root))

    response = client.get("/api/projects")
    assert response.status_code == 200
    data = response.json()
    assert data["workspace"] == tmp_path.as_posix()
    assert {item["name"] for item in data["projects"]} == {"dataset", "second"}

    page = client.get("/")
    assert 'id="project_explorer"' in page.text
    assert "dataset explorer" in page.text

    selected = client.post(
        "/api/projects/select",
        json={"name": "second"},
    )
    assert selected.status_code == 200
    assert selected.json()["name"] == "second"

    state = client.get("/api/state").json()
    assert state["project"]["name"] == "second"
    assert state["project"]["subject"] == "cat"


def test_running_job_stays_bound_to_original_project_after_switch(
    tmp_path: Path,
    monkeypatch,
) -> None:
    root = make_project(tmp_path)
    second = tmp_path / "second"
    second_paths = ProjectPaths.from_root(second)
    second_paths.create_directories()
    write_default_config(
        second_paths.config,
        name="second",
        subject="cat",
        trigger="catx",
    )
    DatasetProject.create(second_paths)

    client = TestClient(create_app(root))

    save = client.post(
        "/api/settings",
        json={
            "name": "test",
            "subject": "person",
            "trigger": "person",
            "preset": "sweetspot",
            "scraper_backend": "duckduckgo",
            "scrape_query": "julia",
            "scrape_count": 1,
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
    assert save.status_code == 200

    started = threading.Event()
    release = threading.Event()

    class FakeScraperManager:
        last_search_error = None

        def __init__(self, **kwargs):
            assert Path(kwargs["output_dir"]).resolve() == (root / "raw").resolve()

        def run(self, query, *, count, known_urls=(), progress=None):
            assert query == "julia"
            assert count == 1
            started.set()
            assert release.wait(3)
            output = root / "raw" / "000001.jpg"
            output.write_bytes(b"fake image")
            if progress is not None:
                progress(1, 1, "downloaded 1/1")
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

    started_job = client.post("/api/run/scrape")
    assert started_job.status_code == 200
    job_id = started_job.json()["job_id"]
    assert started.wait(3)

    selected = client.post("/api/projects/select", json={"name": "second"})
    assert selected.status_code == 200
    release.set()

    deadline = time.monotonic() + 3
    job = None
    while time.monotonic() < deadline:
        job = client.get(f"/api/jobs/{job_id}").json()
        if job["status"] != "running":
            break
        time.sleep(0.01)

    assert job is not None
    assert job["status"] == "done"
    assert job["project_path"] == root.as_posix()
    assert (root / "raw" / "000001.jpg").is_file()

    state = client.get("/api/state").json()
    assert state["project"]["name"] == "second"
