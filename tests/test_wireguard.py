from __future__ import annotations

import base64
import os
import stat
from pathlib import Path

import pytest

from lora_data_maker.wireguard import (
    WireGuardConfigStore,
    WireGuardProxy,
    validate_wireguard_config,
)


def wireguard_config() -> str:
    key = base64.b64encode(bytes(32)).decode("ascii")
    return f"""[Interface]
PrivateKey = {key}
Address = 10.64.0.2/32
DNS = 10.64.0.1

[Peer]
PublicKey = {key}
AllowedIPs = 0.0.0.0/0, ::/0
Endpoint = 198.51.100.10:51820
PersistentKeepalive = 25
"""


def test_wireguard_config_validates_without_echoing_secret():
    config = wireguard_config()
    assert validate_wireguard_config(config) == config
    with pytest.raises(ValueError, match="PrivateKey"):
        validate_wireguard_config(config.replace(
            "PrivateKey = " + base64.b64encode(bytes(32)).decode("ascii"),
            "PrivateKey = not-a-key",
        ))


@pytest.mark.parametrize(
    ("replacement", "error"),
    [
        ("DNS = 10.64.0.1\n", "DNS"),
        ("Address = 10.64.0.2/32\n", "Address"),
        ("AllowedIPs = 0.0.0.0/0, ::/0\n", "Endpoint"),
    ],
)
def test_wireguard_config_requires_dns_address_and_endpoint(replacement, error):
    config = wireguard_config()
    if error == "DNS":
        config = config.replace(replacement, "")
    elif error == "Address":
        config = config.replace(replacement, "")
    else:
        config = config.replace("Endpoint = 198.51.100.10:51820\n", "")
    with pytest.raises(ValueError, match=error):
        validate_wireguard_config(config)


def test_wireguard_store_is_private_and_public_state_never_exposes_keys(
    tmp_path: Path, monkeypatch
):
    monkeypatch.setattr(
        WireGuardProxy,
        "find_binary",
        staticmethod(lambda: "/mock/wireproxy"),
    )
    store = WireGuardConfigStore(tmp_path / "private" / "wireguard")
    config = wireguard_config().encode()
    state = store.save_config(config, "private-vpn.conf")

    assert state["configured"] is True
    assert state["enabled"] is False
    assert state["filename"] == "private-vpn.conf"
    assert base64.b64encode(bytes(32)).decode("ascii") not in repr(state)
    assert store.config_path.read_bytes() == config
    if os.name == "posix":
        assert stat.S_IMODE(store.directory.stat().st_mode) == 0o700
        assert stat.S_IMODE(store.config_path.stat().st_mode) == 0o600
        assert stat.S_IMODE(store.settings_path.stat().st_mode) == 0o600

    enabled = store.set_enabled(True)
    assert enabled["enabled"] is True
    assert store.is_enabled() is True

    removed = store.clear_config()
    assert removed["configured"] is False
    assert removed["enabled"] is False
    assert not store.config_path.exists()


def test_wireguard_store_cannot_enable_without_config(tmp_path: Path):
    store = WireGuardConfigStore(tmp_path / "wireguard")
    with pytest.raises(ValueError, match="upload"):
        store.set_enabled(True)


def test_webui_wireguard_config_upload_toggle_and_remove(tmp_path: Path, monkeypatch):
    from fastapi.testclient import TestClient

    from lora_data_maker.config import write_default_config
    from lora_data_maker.dataset import DatasetProject
    from lora_data_maker.paths import ProjectPaths
    from lora_data_maker.web import create_app

    project_root = tmp_path / "dataset"
    project_paths = ProjectPaths.from_root(project_root)
    project_paths.create_directories()
    write_default_config(project_paths.config, name="test", subject="person", trigger="person")
    DatasetProject.create(project_paths)
    monkeypatch.setattr(
        WireGuardProxy,
        "find_binary",
        staticmethod(lambda: "/mock/wireproxy"),
    )
    client = TestClient(
        create_app(project_root, wireguard_store_dir=tmp_path / "private" / "wireguard")
    )

    before = client.get("/api/wireguard/state").json()
    assert before["configured"] is False
    uploaded = client.post(
        "/api/wireguard/config",
        files={"file": ("test.conf", wireguard_config().encode(), "text/plain")},
    )
    assert uploaded.status_code == 200
    assert uploaded.json()["configured"] is True
    assert uploaded.json()["enabled"] is False
    assert "PrivateKey" not in str(uploaded.json())

    enabled = client.post("/api/wireguard/settings", json={"enabled": True})
    assert enabled.status_code == 200
    assert enabled.json()["enabled"] is True

    removed = client.delete("/api/wireguard/config")
    assert removed.status_code == 200
    assert removed.json()["configured"] is False
    assert removed.json()["enabled"] is False


def test_wireproxy_requires_an_executable(tmp_path: Path, monkeypatch):
    config_path = tmp_path / "wireguard.conf"
    config_path.write_text(wireguard_config(), encoding="utf-8")
    monkeypatch.setattr(WireGuardProxy, "find_binary", staticmethod(lambda: None))
    with pytest.raises(RuntimeError, match="wireproxy was not found"):
        WireGuardProxy(config_path).start()


def test_wireproxy_uses_private_config_and_cleans_up(
    tmp_path: Path, monkeypatch
):
    import lora_data_maker.wireguard as module

    config_path = tmp_path / "wireguard.conf"
    config_path.write_text(wireguard_config(), encoding="utf-8")
    monkeypatch.setattr(
        WireGuardProxy,
        "find_binary",
        staticmethod(lambda: "/mock/wireproxy"),
    )
    captured = {}

    class FakeProcess:
        returncode = None

        def poll(self):
            return self.returncode

        def terminate(self):
            self.returncode = 0

        def wait(self, timeout=None):
            return self.returncode

        def kill(self):
            self.returncode = -9

    def fake_popen(args, **kwargs):
        captured["args"] = args
        captured["config_path"] = Path(args[-1])
        captured["config_text"] = Path(args[-1]).read_text(encoding="utf-8")
        captured["kwargs"] = kwargs
        return FakeProcess()

    class FakeSocket:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

    monkeypatch.setattr(module.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(module.socket, "create_connection", lambda *args, **kwargs: FakeSocket())

    proxy = WireGuardProxy(config_path)
    proxy.start()
    generated_path = captured["config_path"]

    assert captured["args"][0] == "/mock/wireproxy"
    assert captured["args"][1] == "-c"
    assert f"WGConfig = {config_path}" in captured["config_text"]
    assert "[http]" in captured["config_text"]
    assert f"BindAddress = {proxy.proxy_url.removeprefix('http://')}" in captured["config_text"]
    assert proxy.proxy_url.startswith("http://127.0.0.1:")
    if os.name == "posix":
        assert stat.S_IMODE(generated_path.stat().st_mode) == 0o600

    proxy.close()
    assert proxy.proxy_url is None
    assert not generated_path.exists()
