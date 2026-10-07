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
backend = "duckduckgo"
query = "my-dataset"
count = 300
timeout = 20
retries = 3
delay = 1.0
max_scrolls = 30

[validation]
min_width = 1
min_height = 1
max_width = 0
max_height = 0
min_file_size = 1
max_file_size = 0

[quality]
min_width = 512
min_height = 512
blur_threshold = 0.0
max_aspect_ratio = 0.0
person_check = false
person_threshold = 0.35
max_persons = 1
min_subject_area = 0.12
min_subject_visibility = 0.35

[processing]
target_width = 512
target_height = 512
resize_mode = "cover"
jpeg_quality = 95
face_crop = false
face_padding = 1.75

[dedup]
exact = true
perceptual = true
perceptual_distance = 6

[tagger]
backend = "wd14"
preset = "sweetspot"
threshold = 0.35
character_threshold = 0.85
use_mcut = false
batch_size = 1
download_timeout = 60

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

[export]
repeats = 1
folder_name = ""
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
