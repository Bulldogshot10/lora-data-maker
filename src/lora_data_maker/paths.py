from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from collections.abc import Mapping


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

        paths = {
            "raw": resolve_dir("raw", "raw"),
            "work": resolve_dir("work", "work"),
            "output": resolve_dir("output", "output"),
            "cache": resolve_dir("cache", "cache"),
        }

        if not isinstance(configured, Mapping):
            raise ValueError("config section [paths] must be a table")

        labels = list(paths.items())
        for index, (left_name, left_path) in enumerate(labels):
            for right_name, right_path in labels[index + 1:]:
                if _paths_overlap(left_path, right_path):
                    raise ValueError(
                        f"configured paths overlap: {left_name}={left_path} "
                        f"and {right_name}={right_path}"
                    )

        return cls(
            root=root_path,
            config=root_path / "config.toml",
            raw=paths["raw"],
            work=paths["work"],
            output=paths["output"],
            cache=paths["cache"],
        )

    def create_directories(self) -> None:
        for directory in (self.raw, self.work, self.output, self.cache):
            directory.mkdir(parents=True, exist_ok=True)


def manifest_path(root: Path | str, path: Path | str) -> str:
    """Represent a path relative to root when possible, otherwise absolute."""
    root_path = Path(root).expanduser().resolve()
    target = Path(path).expanduser().resolve()
    try:
        return target.relative_to(root_path).as_posix()
    except ValueError:
        return target.as_posix()


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


def _paths_overlap(first: Path, second: Path) -> bool:
    return first == second or first in second.parents or second in first.parents
