from pathlib import Path
import re
import shutil
import subprocess
import tempfile
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

    assert updated["caption"]["max_tokens"] == 77

    caption_budget = client.post(
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
            "caption_max_tokens": 64,
            "face_crop": False,
            "face_padding": 1.75,
        },
    )
    assert caption_budget.status_code == 200
    budget_state = client.get("/api/state").json()
    assert budget_state["caption"]["max_tokens"] == 64

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


def test_webui_shell_has_production_navigation_and_safe_interactions(tmp_path: Path) -> None:
    root = make_project(tmp_path)
    client = TestClient(create_app(root))

    html = client.get("/").text
    assert 'data-view="overview"' in html
    assert 'data-view="collect"' in html
    assert 'data-view="prepare"' in html
    assert 'data-view="settings"' in html
    assert 'id="new-modal"' in html
    assert 'id="confirm-modal"' in html
    assert 'aria-live="polite"' in html
    assert 'id="project_explorer"' in html
    assert 'id="progress-fill"' in html
    assert 'id="caption_max_tokens"' in html
    assert 'max="77"' in html
    assert 'id="preview-modal"' in html
    assert 'rel="icon" href="/favicon.ico"' in html
    assert 'sizes="32x32"' in html
    assert "onclick=" not in html

    for path in (
        "/favicon.ico",
        "/favicon-16x16.png",
        "/favicon-32x32.png",
    ):
        response = TestClient(create_app(make_project(tmp_path))).get(path)
        assert response.status_code == 200
        assert response.content


def test_webui_javascript_syntax_when_node_is_available(tmp_path: Path) -> None:
    node = shutil.which("node")
    if not node:
        return

    html = TestClient(create_app(make_project(tmp_path))).get("/").text
    scripts = re.findall(r"<script>(.*?)</script>", html, flags=re.DOTALL)
    assert scripts

    for index, script in enumerate(scripts):
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            suffix=".js",
            prefix=f"webui_{index}_",
        ) as handle:
            handle.write(script)
            handle.flush()
            checked = subprocess.run(
                [node, "--check", handle.name],
                capture_output=True,
                text=True,
                timeout=10,
            )
        assert checked.returncode == 0, checked.stderr



def test_web_can_start_from_empty_workspace_and_create_dataset(tmp_path: Path) -> None:
    workspace = tmp_path / "datasets"
    workspace.mkdir()
    client = TestClient(create_app(workspace))

    state = client.get("/api/state")
    assert state.status_code == 200
    assert state.json()["project"] is None
    assert state.json()["workspace"] == workspace.as_posix()
    assert state.json()["project_count"] == 0
    assert client.post("/api/run/validate").status_code == 409

    page = client.get("/")
    assert page.status_code == 200
    assert "create your first dataset" in page.text
    assert 'id="new-dataset"' in page.text

    created = client.post(
        "/api/project/new",
        json={
            "name": "julia",
            "subject": "julia",
            "trigger": "juliax",
        },
    )
    assert created.status_code == 200
    assert created.json()["name"] == "julia"
    assert (workspace / "julia" / "config.toml").is_file()
    assert (workspace / "julia" / "manifest.json").is_file()

    state = client.get("/api/state")
    assert state.status_code == 200
    data = state.json()
    assert data["project"]["name"] == "julia"
    assert data["project"]["subject"] == "julia"
    assert data["project"]["trigger"] == "juliax"
    assert data["project_count"] == 1
    assert data["projects"][0]["active"] is True


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
            "text_check": True,
            "text_confidence": 0.6,
            "text_min_area_ratio": 0.0008,
        },
    )
    assert response.status_code == 200

    state = client.get("/api/state").json()
    assert state["quality"]["person_check"] is True
    assert state["quality"]["max_persons"] == 1
    assert state["quality"]["person_threshold"] == 0.4
    assert state["quality"]["min_subject_area"] == 0.15
    assert state["quality"]["min_subject_visibility"] == 0.4
    assert state["quality"]["text_check"] is True
    assert state["quality"]["text_confidence"] == 0.6
    assert state["quality"]["text_min_area_ratio"] == 0.0008


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

        def run(self, query, *, count, known_urls=(), progress=None, cancel_event=None):
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


def test_dataset_explorer_deletes_inactive_project(tmp_path: Path) -> None:
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

    deleted = client.delete("/api/projects/second")
    assert deleted.status_code == 200
    assert deleted.json() == {"ok": True, "name": "second"}
    assert not second.exists()

    projects = client.get("/api/projects").json()["projects"]
    assert {item["name"] for item in projects} == {"dataset"}


def test_dataset_explorer_refuses_active_project_delete(tmp_path: Path) -> None:
    root = make_project(tmp_path)
    client = TestClient(create_app(root))

    response = client.delete("/api/projects/dataset")
    assert response.status_code == 409
    assert "active project" in response.json()["detail"]
    assert root.exists()


def test_web_delete_image_removes_record_and_artifacts(tmp_path: Path) -> None:
    from lora_data_maker.manifest import register_image

    root = make_project(tmp_path)
    paths = ProjectPaths.from_root(root)

    raw = paths.raw / "000001.jpg"
    processed = paths.work / "processed" / "000001.jpg"
    caption = paths.work / "captions" / "000001.txt"
    exported = paths.output / "1_person" / "image.jpg"
    exported_caption = paths.output / "1_person" / "image.txt"

    for target, content in (
        (raw, b"raw"),
        (processed, b"processed"),
        (caption, b"caption"),
        (exported, b"exported"),
        (exported_caption, b"exported caption"),
    ):
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)

    project = DatasetProject.open(paths)
    record = register_image(
        project.manifest,
        path="raw/000001.jpg",
        status="exported",
    )
    record.processed_path = "work/processed/000001.jpg"
    record.caption_path = "work/captions/000001.txt"
    record.metadata["export_image_path"] = "output/1_person/image.jpg"
    record.metadata["export_caption_path"] = "output/1_person/image.txt"
    project.save()

    client = TestClient(create_app(root))
    response = client.delete(f"/api/images/{record.image_id}")

    assert response.status_code == 200
    assert response.json()["ok"] is True
    assert record.image_id not in DatasetProject.open(paths).manifest.images
    assert not raw.exists()
    assert not processed.exists()
    assert not caption.exists()
    assert not exported.exists()
    assert not exported_caption.exists()


def test_webui_shell_exposes_curation_controls(tmp_path: Path) -> None:
    html = TestClient(create_app(make_project(tmp_path))).get("/").text
    assert 'data-job-cancel=' in html
    assert 'data-image-delete=' in html
    assert '/api/images/' in html
