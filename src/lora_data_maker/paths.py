from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping


@dataclass(frozen=True)
class ProjectPaths:
    root: Path
    config: Path
    raw: Path
    work: Path
    output: Path
    cache: Path

    @classmethod
    def from_root(
        cls,
        root: Path | str,
        configured: Mapping[str, str] | None = None,
    ) -> "ProjectPaths":
        root_path = Path(root).expanduser().resolve()
        configured = configured or {}

        def resolve_dir(key: str, default: str) -> Path:
            value = configured.get(key, default)
            candidate = Path(str(value)).expanduser()
            return candidate if candidate.is_absolute() else root_path / candidate

        return cls(
            root=root_path,
            config=root_path / "config.toml",
            raw=resolve_dir("raw", "raw"),
            work=resolve_dir("work", "work"),
            output=resolve_dir("output", "output"),
            cache=resolve_dir("cache", "cache"),
        )

    def create_directories(self) -> None:
        for directory in (self.raw, self.work, self.output, self.cache):
            directory.mkdir(parents=True, exist_ok=True)


def find_project_root(start: Path | str = ".") -> Path:
    current = Path(start).expanduser().resolve()
    if current.is_file():
        current = current.parent

    for candidate in (current, *current.parents):
        if (candidate / "config.toml").is_file():
            return candidate

    raise FileNotFoundError(
        f"could not find config.toml from {current} or any parent directory"
    )
