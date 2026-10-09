from __future__ import annotations

import json
import os
import stat
from pathlib import Path

from fastapi.testclient import TestClient

from lora_data_maker.config import write_default_config
from lora_data_maker.credentials import load_api_keys, save_api_keys
from lora_data_maker.dataset import DatasetProject
from lora_data_maker.paths import ProjectPaths
from lora_data_maker.web import create_app


def make_project(root: Path) -> Path:
    paths = ProjectPaths.from_root(root)
    paths.create_directories()
    write_default_config(paths.config, name="test", subject="person", trigger="person")
    DatasetProject.create(paths)
    return root


def test_api_key_store_is_private_and_keeps_provider_keys(tmp_path: Path) -> None:
    path = tmp_path / "private" / "api-keys.json"
    save_api_keys(
        path,
        {
            "GEMINI_API_KEY": "gemini-secret",
            "OPENAI_API_KEY": "openai-secret",
            "IGNORED_KEY": "must-not-be-written",
        },
    )

    assert load_api_keys(path) == {
        "GEMINI_API_KEY": "gemini-secret",
        "OPENAI_API_KEY": "openai-secret",
    }
    if os.name == "posix":
        assert stat.S_IMODE(path.stat().st_mode) == 0o600
        assert stat.S_IMODE(path.parent.stat().st_mode) == 0o700

    saved = json.loads(path.read_text(encoding="utf-8"))
    assert "IGNORED_KEY" not in saved


def test_webui_api_keys_survive_project_switch_and_restart(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    workspace = tmp_path / "workspace"
    first_root = make_project(workspace / "first")
    make_project(workspace / "second")
    key_file = tmp_path / "private" / "api-keys.json"

    first_client = TestClient(create_app(first_root, api_key_store_path=key_file))
    gemini = first_client.post(
        "/api/vlm/api-key",
        json={"runtime": "gemini", "api_key": "gemini-secret"},
    )
    assert gemini.status_code == 200
    assert gemini.json()["saved"] is True

    openai = first_client.post(
        "/api/vlm/api-key",
        json={"runtime": "openai_compatible", "api_key": "openai-secret"},
    )
    assert openai.status_code == 200
    assert openai.json()["saved"] is True

    project_config = (first_root / "config.toml").read_text(encoding="utf-8")
    assert "gemini-secret" not in project_config
    assert "openai-secret" not in project_config

    # Simulate a server process restart: keys are restored from the local store.
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    restarted_client = TestClient(create_app(first_root, api_key_store_path=key_file))
    state = restarted_client.get("/api/state").json()
    assert state["caption"]["api_keys_set"] == {
        "gemini": True,
        "openai_compatible": True,
    }

    reused = restarted_client.post(
        "/api/vlm/api-key",
        json={"runtime": "gemini", "api_key": ""},
    )
    assert reused.status_code == 200
    assert reused.json()["key_set"] is True

    selected = restarted_client.post("/api/projects/select", json={"name": "second"})
    assert selected.status_code == 200
    switched_state = restarted_client.get("/api/state").json()
    assert switched_state["caption"]["api_keys_set"] == {
        "gemini": True,
        "openai_compatible": True,
    }

    cleared = restarted_client.post(
        "/api/vlm/api-key",
        json={"runtime": "gemini", "clear": True},
    )
    assert cleared.status_code == 200
    assert cleared.json()["key_set"] is False
    after_clear = restarted_client.get("/api/state").json()
    assert after_clear["caption"]["api_keys_set"] == {
        "gemini": False,
        "openai_compatible": True,
    }


def test_gemini_model_discovery_is_cached(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "test-gemini-key")
    root = make_project(tmp_path / "workspace" / "dataset")
    calls: list[tuple[str, dict]] = []

    class Response:
        ok = True
        status_code = 200
        text = ""

        def json(self):
            return {
                "models": [
                    {
                        "name": "models/gemini-fast",
                        "displayName": "Gemini Fast",
                        "supportedGenerationMethods": ["generateContent"],
                    }
                ],
                "nextPageToken": "",
            }

    def fake_get(url: str, **kwargs):
        calls.append((url, kwargs))
        return Response()

    monkeypatch.setattr("requests.get", fake_get)
    client = TestClient(
        create_app(root, api_key_store_path=tmp_path / "private" / "api-keys.json")
    )

    first = client.get("/api/vlm/models?runtime=gemini")
    second = client.get("/api/vlm/models?runtime=gemini")
    assert first.status_code == 200
    assert second.status_code == 200
    assert first.json()["models"][0]["id"] == "gemini-fast"
    assert second.json()["cached"] is True
    assert len(calls) == 1
