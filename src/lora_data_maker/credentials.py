from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any


API_KEY_ENV_NAMES = ("GEMINI_API_KEY", "OPENAI_API_KEY")


def default_api_key_store_path() -> Path:
    """Return a per-user credential path outside every dataset project."""
    override = os.environ.get("LORA_DATA_MAKER_API_KEY_FILE", "").strip()
    if override:
        return Path(override).expanduser().resolve()

    config_home = os.environ.get("XDG_CONFIG_HOME", "").strip()
    base = Path(config_home).expanduser() if config_home else Path.home() / ".config"
    return (base / "lora-data-maker" / "api-keys.json").resolve()


def load_api_keys(path: Path | str) -> dict[str, str]:
    """Load only recognized provider keys. Never expose this mapping to the UI."""
    target = Path(path).expanduser()
    try:
        raw: Any = json.loads(target.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    except (OSError, UnicodeError, json.JSONDecodeError):
        return {}

    if not isinstance(raw, dict):
        return {}
    return {
        name: value.strip()
        for name in API_KEY_ENV_NAMES
        if isinstance((value := raw.get(name)), str) and value.strip()
    }


def save_api_keys(path: Path | str, keys: dict[str, str]) -> None:
    """Atomically save API keys with owner-only file and directory permissions."""
    target = Path(path).expanduser().resolve()
    safe_keys = {
        name: value.strip()
        for name in API_KEY_ENV_NAMES
        if isinstance((value := keys.get(name)), str) and value.strip()
    }
    target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    try:
        os.chmod(target.parent, 0o700)
    except OSError:
        pass

    fd, temporary_name = tempfile.mkstemp(
        prefix=".api-keys-",
        suffix=".tmp",
        dir=target.parent,
    )
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(safe_keys, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, target)
        try:
            os.chmod(target, 0o600)
        except OSError:
            pass
    except Exception:
        try:
            os.close(fd)
        except OSError:
            pass
        try:
            os.unlink(temporary_name)
        except OSError:
            pass
        raise
