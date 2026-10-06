from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping

from .exceptions import LoraDataMakerError
from .models import ImageRecord, ProjectManifest


MANIFEST_FILENAME = "manifest.json"


class ManifestError(LoraDataMakerError):
    """Raised when a dataset manifest cannot be loaded or saved."""


def load_manifest(path: Path | str) -> ProjectManifest:
    manifest_path = Path(path).expanduser().resolve()
    if not manifest_path.is_file():
        raise ManifestError(f"manifest file not found: {manifest_path}")

    try:
        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(raw, Mapping):
            raise ValueError("manifest root must be an object")
        return ProjectManifest.from_dict(raw)
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
        raise ManifestError(f"could not load manifest {manifest_path}: {exc}") from exc


def save_manifest(
    manifest: ProjectManifest,
    path: Path | str,
) -> Path:
    manifest_path = Path(path).expanduser().resolve()
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest.touch()

    payload = json.dumps(
        manifest.to_dict(),
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    )

    fd = -1
    temp_name: str | None = None
    try:
        fd, temp_name = tempfile.mkstemp(
            prefix=f".{manifest_path.name}.",
            suffix=".tmp",
            dir=manifest_path.parent,
            text=True,
        )
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            fd = -1
            handle.write(payload)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())

        os.replace(temp_name, manifest_path)
        temp_name = None
    except OSError as exc:
        raise ManifestError(f"could not save manifest {manifest_path}: {exc}") from exc
    finally:
        if fd >= 0:
            os.close(fd)
        if temp_name is not None:
            try:
                os.unlink(temp_name)
            except OSError:
                pass

    return manifest_path


def create_manifest(
    *,
    project_name: str,
    subject: str,
    trigger: str,
    source: Mapping[str, Any] | None = None,
    config_snapshot: Mapping[str, Any] | None = None,
) -> ProjectManifest:
    return ProjectManifest(
        project_name=project_name,
        subject=subject,
        trigger=trigger,
        source=dict(source or {}),
        config_snapshot=dict(config_snapshot or {}),
    )


def register_image(
    manifest: ProjectManifest,
    *,
    path: str,
    source_url: str | None = None,
    source_query: str | None = None,
    status: str = "discovered",
) -> ImageRecord:
    image = ImageRecord(
        path=path,
        source_url=source_url,
        source_query=source_query,
        status=status,
    )
    image.set_status(status)
    return manifest.add_image(image)


def find_image_by_path(
    manifest: ProjectManifest,
    path: str,
) -> ImageRecord | None:
    normalized = Path(path).as_posix()
    for image in manifest.images.values():
        if Path(image.path).as_posix() == normalized:
            return image
    return None


def find_image_by_source_url(
    manifest: ProjectManifest,
    source_url: str,
) -> ImageRecord | None:
    for image in manifest.images.values():
        if image.source_url == source_url:
            return image
    return None
