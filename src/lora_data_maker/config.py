from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any

from .exceptions import ConfigError


DEFAULT_CONFIG_TOML = """[project]
name = "my-dataset"
subject = "my-dataset"
trigger = "my-dataset"

[paths]
raw = "raw"
work = "work"
output = "output"
cache = "cache"

[scraper]
backend = "google"
query = "my-dataset"
count = 300
timeout = 20
retries = 3
delay = 1.0
min_width = 512
min_height = 512

[processing]
target_size = 512
face_crop = false
blur_threshold = 0.0

[dedup]
exact = true
perceptual = true
perceptual_distance = 6

[tagger]
backend = "wd14"
threshold = 0.35
model = "SmilingWolf/wd-eva02-large-tagger-v3"

[tag_rules]
remove = [
  "watermark",
  "signature",
  "logo",
  "text",
]

[caption]
separator = ", "
subject_position = "first"
trigger_from_subject = true
"""


def load_config(path: Path | str) -> dict[str, Any]:
    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise ConfigError(f"config file not found: {config_path}")

    try:
        with config_path.open("rb") as handle:
            data = tomllib.load(handle)
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"invalid TOML in {config_path}: {exc}") from exc
    except OSError as exc:
        raise ConfigError(f"could not read {config_path}: {exc}") from exc

    if not isinstance(data, dict):
        raise ConfigError(f"invalid config root in {config_path}")

    return data


def write_default_config(
    path: Path | str,
    *,
    name: str | None = None,
    subject: str | None = None,
    trigger: str | None = None,
) -> Path:
    config_path = Path(path).expanduser().resolve()
    dataset_name = name or "my-dataset"
    dataset_subject = subject or dataset_name
    dataset_trigger = trigger or dataset_subject

    text = DEFAULT_CONFIG_TOML
    text = text.replace('name = "my-dataset"', f'name = "{_toml_string(dataset_name)}"', 1)
    text = text.replace(
        'subject = "my-dataset"',
        f'subject = "{_toml_string(dataset_subject)}"',
        1,
    )
    text = text.replace(
        'trigger = "my-dataset"',
        f'trigger = "{_toml_string(dataset_trigger)}"',
        1,
    )
    text = text.replace(
        'query = "my-dataset"',
        f'query = "{_toml_string(dataset_subject)}"',
        1,
    )

    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(text, encoding="utf-8")
    return config_path


def _toml_string(value: str) -> str:
    return (
        str(value)
        .replace("\\", "\\\\")
        .replace('"', '\\\"')
        .replace("\n", "\\n")
        .replace("\r", "\\r")
    )
