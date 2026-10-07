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
        },
    )
    assert settings.status_code == 200

    updated = client.get("/api/state").json()
    assert updated["project"]["subject"] == "new_person"
    assert updated["processing"]["width"] == 768
    assert updated["processing"]["height"] == 512
    assert updated["processing"]["quality"] == 92
