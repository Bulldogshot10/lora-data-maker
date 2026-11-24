from pathlib import Path

from fastapi.testclient import TestClient

from lora_data_maker.config import write_default_config
from lora_data_maker.dataset import DatasetProject
from lora_data_maker.paths import ProjectPaths
from lora_data_maker.web import create_app


def make_project(tmp_path: Path) -> Path:
    root = tmp_path / "dataset"
    paths = ProjectPaths.from_root(root)
    paths.create_directories()
    write_default_config(paths.config, name="test", subject="person", trigger="person")
    DatasetProject.create(paths)
    return root


def test_upload_skips_exact_duplicate_content_in_batch_and_later_requests(
    tmp_path: Path,
) -> None:
    root = make_project(tmp_path)
    client = TestClient(create_app(root))
    image_bytes = b"same image bytes"

    first = client.post(
        "/api/upload",
        files=[
            ("files", ("portrait.jpg", image_bytes, "image/jpeg")),
            ("files", ("renamed-copy.png", image_bytes, "image/png")),
        ],
    )

    assert first.status_code == 200
    assert first.json() == {"uploaded": 1, "skipped": 1}
    raw_images = sorted((root / "raw").glob("*"))
    assert [path.name for path in raw_images] == ["portrait.jpg"]
    assert len(DatasetProject.open(ProjectPaths.from_root(root)).manifest.images) == 1

    second = client.post(
        "/api/upload",
        files={"files": ("another-name.jpg", image_bytes, "image/jpeg")},
    )

    assert second.status_code == 200
    assert second.json() == {"uploaded": 0, "skipped": 1}
    assert [path.name for path in sorted((root / "raw").glob("*"))] == ["portrait.jpg"]
    assert len(DatasetProject.open(ProjectPaths.from_root(root)).manifest.images) == 1


def test_upload_deduplicates_against_raw_file_missing_from_manifest(
    tmp_path: Path,
) -> None:
    root = make_project(tmp_path)
    orphan = root / "raw" / "already-there.jpg"
    image_bytes = b"orphaned image bytes"
    orphan.write_bytes(image_bytes)
    client = TestClient(create_app(root))

    response = client.post(
        "/api/upload",
        files={"files": ("new-name.jpg", image_bytes, "image/jpeg")},
    )

    assert response.status_code == 200
    assert response.json() == {"uploaded": 0, "skipped": 1}
    assert orphan.read_bytes() == image_bytes
    assert len(DatasetProject.open(ProjectPaths.from_root(root)).manifest.images) == 0


def test_upload_ui_prevents_concurrent_duplicate_submission(tmp_path: Path) -> None:
    client = TestClient(create_app(make_project(tmp_path)))
    html = client.get("/").text

    assert "if(appState.uploadInProgress)return;" in html
    assert "appState.uploadInProgress=true;" in html
    assert 'appState.uploadInProgress=false;' in html
