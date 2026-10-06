from __future__ import annotations

from pathlib import Path
from typing import Any

from .config import load_config
from .exceptions import ProjectError
from .manifest import MANIFEST_FILENAME, create_manifest, load_manifest, save_manifest
from .models import ProjectManifest
from .paths import ProjectPaths


class DatasetProject:
    """Persistent project facade used by later pipeline stages."""

    def __init__(
        self,
        paths: ProjectPaths,
        manifest: ProjectManifest,
    ) -> None:
        self.paths = paths
        self.manifest = manifest

    @property
    def manifest_path(self) -> Path:
        return self.paths.root / MANIFEST_FILENAME

    @classmethod
    def create(cls, paths: ProjectPaths) -> "DatasetProject":
        if not paths.config.is_file():
            raise ProjectError(f"project config not found: {paths.config}")

        config = load_config(paths.config)
        project = config.get("project", {})
        if not isinstance(project, dict):
            raise ProjectError("config section [project] must be a table")

        scraper = config.get("scraper", {})
        if not isinstance(scraper, dict):
            raise ProjectError("config section [scraper] must be a table")

        name = _required_text(project, "name")
        subject = _required_text(project, "subject")
        trigger = str(project.get("trigger") or subject)

        manifest = create_manifest(
            project_name=name,
            subject=subject,
            trigger=trigger,
            source={
                "backend": scraper.get("backend"),
                "query": scraper.get("query"),
            },
            config_snapshot=config,
        )

        save_manifest(manifest, paths.root / MANIFEST_FILENAME)
        return cls(paths, manifest)

    @classmethod
    def open(cls, paths: ProjectPaths) -> "DatasetProject":
        return cls(paths, load_manifest(paths.root / MANIFEST_FILENAME))

    def save(self) -> Path:
        self.manifest.touch()
        return save_manifest(self.manifest, self.manifest_path)

    def counts(self) -> dict[str, int]:
        return self.manifest.counts()


def _required_text(section: dict[str, Any], key: str) -> str:
    value = section.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ProjectError(f"config field project.{key} must be a non-empty string")
    return value.strip()
