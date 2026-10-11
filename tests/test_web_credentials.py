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

def test_openai_model_discovery_uses_saved_base_url_not_query_parameter(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "trusted-openai-key")
    root = make_project(tmp_path / "workspace" / "dataset")
    config_path = root / "config.toml"
    config_text = config_path.read_text(encoding="utf-8")
    assert 'api_base_url = ""' in config_text
    config_path.write_text(
        config_text.replace(
            'api_base_url = ""',
            'api_base_url = "https://trusted-provider.example/v1"',
            1,
        ),
        encoding="utf-8",
    )

    calls: list[tuple[str, dict]] = []

    class Response:
        ok = True
        status_code = 200
        text = ""

        def json(self):
            return {"data": [{"id": "vision-model"}]}

    def fake_get(url: str, **kwargs):
        calls.append((url, kwargs))
        return Response()

    monkeypatch.setattr("requests.get", fake_get)
    client = TestClient(
        create_app(root, api_key_store_path=tmp_path / "private" / "api-keys.json")
    )

    unapproved = client.get(
        "/api/vlm/models"
        "?runtime=openai_compatible"
        "&base_url=https%3A%2F%2Fattacker.example%2Fv1"
    )
    assert unapproved.status_code == 409
    assert calls == []

    response = client.get(
        "/api/vlm/models"
        "?runtime=openai_compatible"
        "&base_url=https%3A%2F%2Fattacker.example%2Fv1",
        headers={"x-ldm-explicit-model-discovery": "1"},
    )

    assert response.status_code == 200
    assert response.json()["models"] == [
        {"id": "vision-model", "label": "vision-model"}
    ]
    assert len(calls) == 1
    url, kwargs = calls[0]
    assert url == "https://trusted-provider.example/v1/models"
    assert kwargs["headers"]["Authorization"] == "Bearer trusted-openai-key"
    assert "attacker.example" not in url

def test_vlm_credential_routes_reject_cross_origin_browser_requests(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "test-gemini-key")
    root = make_project(tmp_path / "workspace" / "dataset")
    calls: list[str] = []

    def fake_get(url: str, **kwargs):
        calls.append(url)
        raise AssertionError("cross-origin request must not contact provider")

    monkeypatch.setattr("requests.get", fake_get)
    client = TestClient(
        create_app(root, api_key_store_path=tmp_path / "private" / "api-keys.json")
    )
    cross_origin_headers = {
        "origin": "https://attacker.example",
        "sec-fetch-site": "cross-site",
    }

    discovery = client.get(
        "/api/vlm/models?runtime=gemini",
        headers=cross_origin_headers,
    )
    mutate_key = client.post(
        "/api/vlm/api-key",
        json={"runtime": "gemini", "clear": True},
        headers=cross_origin_headers,
    )
    start_job = client.post(
        "/api/run/scrape",
        headers=cross_origin_headers,
    )

    assert discovery.status_code == 403
    assert mutate_key.status_code == 403
    assert start_job.status_code == 403
    assert calls == []
    assert load_api_keys(tmp_path / "private" / "api-keys.json") == {}

def test_saving_api_keys_does_not_change_existing_parent_permissions(tmp_path: Path) -> None:
    shared_dir = tmp_path / "shared"
    shared_dir.mkdir()
    if os.name == "posix":
        os.chmod(shared_dir, 0o755)

    path = shared_dir / "api-keys.json"
    save_api_keys(path, {"GEMINI_API_KEY": "test-key"})

    if os.name == "posix":
        assert stat.S_IMODE(shared_dir.stat().st_mode) == 0o755
        assert stat.S_IMODE(path.stat().st_mode) == 0o600



def test_explicit_openai_model_discovery_accepts_unsaved_provider_url(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "ollama-cloud-key")
    root = make_project(tmp_path / "workspace" / "dataset")
    calls: list[tuple[str, dict]] = []

    class Response:
        ok = True
        status_code = 200
        text = ""

        def json(self):
            return {"data": [{"id": "qwen3.5:cloud"}]}

    def fake_get(url: str, **kwargs):
        calls.append((url, kwargs))
        return Response()

    monkeypatch.setattr("requests.get", fake_get)
    client = TestClient(
        create_app(root, api_key_store_path=tmp_path / "private" / "api-keys.json")
    )
    payload = {
        "runtime": "openai_compatible",
        "base_url": "https://ollama.com/v1",
    }

    unapproved = client.post("/api/vlm/models", json=payload)
    assert unapproved.status_code == 403
    assert calls == []

    invalid = client.post(
        "/api/vlm/models",
        json={**payload, "base_url": "file:///etc/passwd"},
        headers={"x-ldm-explicit-model-discovery": "1"},
    )
    assert invalid.status_code == 400
    assert calls == []

    response = client.post(
        "/api/vlm/models",
        json=payload,
        headers={"x-ldm-explicit-model-discovery": "1"},
    )

    assert response.status_code == 200
    assert response.json()["models"] == [
        {"id": "qwen3.5:cloud", "label": "qwen3.5:cloud"}
    ]
    assert len(calls) == 1
    url, kwargs = calls[0]
    assert url == "https://ollama.com/v1/models"
    assert kwargs["headers"]["Authorization"] == "Bearer ollama-cloud-key"


def test_explicit_model_discovery_post_rejects_cross_origin_requests(
    tmp_path: Path,
) -> None:
    root = make_project(tmp_path / "workspace" / "dataset")
    client = TestClient(create_app(root))
    response = client.post(
        "/api/vlm/models",
        json={
            "runtime": "openai_compatible",
            "base_url": "https://attacker.example/v1",
        },
        headers={
            "origin": "https://attacker.example",
            "sec-fetch-site": "cross-site",
            "x-ldm-explicit-model-discovery": "1",
        },
    )
    assert response.status_code == 403



def test_gemini_model_discovery_follows_all_pages(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "test-gemini-key")
    root = make_project(tmp_path / "workspace" / "dataset")
    calls: list[tuple[str, dict]] = []
    pages = {
        "": {
            "models": [
                {
                    "name": "models/gemini-page-1",
                    "displayName": "Gemini Page 1",
                    "supportedGenerationMethods": ["generateContent"],
                }
            ],
            "nextPageToken": "page-2",
        },
        "page-2": {
            "models": [
                {
                    "name": "models/gemini-page-2",
                    "displayName": "Gemini Page 2",
                    "supportedGenerationMethods": ["generateContent"],
                }
            ],
            "nextPageToken": "page-3",
        },
        "page-3": {
            "models": [
                {
                    "name": "models/gemini-page-3",
                    "displayName": "Gemini Page 3",
                    "supportedGenerationMethods": ["generateContent"],
                }
            ],
            "nextPageToken": "",
        },
    }

    class Response:
        ok = True
        status_code = 200
        text = ""

        def __init__(self, payload):
            self.payload = payload

        def json(self):
            return self.payload

    def fake_get(url: str, **kwargs):
        calls.append((url, kwargs))
        params = kwargs.get("params", {})
        return Response(pages[params.get("pageToken", "")])

    monkeypatch.setattr("requests.get", fake_get)
    client = TestClient(
        create_app(root, api_key_store_path=tmp_path / "private" / "api-keys.json")
    )

    response = client.get("/api/vlm/models?runtime=gemini")

    assert response.status_code == 200
    assert [model["id"] for model in response.json()["models"]] == [
        "gemini-page-1",
        "gemini-page-2",
        "gemini-page-3",
    ]
    assert len(calls) == 3
    assert all(call[1]["params"]["pageSize"] == "1000" for call in calls)
