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
