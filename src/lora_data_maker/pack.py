from __future__ import annotations

import hashlib
import os
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .models import ProjectManifest


@dataclass(frozen=True)
class PackSummary:
    archive_path: str
    file_count: int
    size_bytes: int
    sha256: str


def _sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def pack_dataset(
    *,
    root: Path | str,
    dataset_dir: Path | str,
    archive_path: Path | str,
    manifest_path: Path | str | None = None,
) -> PackSummary:
    """Create a ZIP containing the exported dataset and manifest.json."""
    root_path = Path(root).expanduser().resolve()
    dataset_root = Path(dataset_dir).expanduser().resolve()
    archive = Path(archive_path).expanduser().resolve()
    manifest = (
        Path(manifest_path).expanduser().resolve()
        if manifest_path is not None
        else root_path / "manifest.json"
    )

    if not dataset_root.is_dir():
        raise FileNotFoundError(f"exported dataset directory not found: {dataset_root}")
    if not manifest.is_file():
        raise FileNotFoundError(f"manifest file not found: {manifest}")
    if archive == manifest:
        raise ValueError("archive path must differ from manifest path")

    try:
        archive.relative_to(dataset_root)
    except ValueError:
        pass
    else:
        raise ValueError(
            "archive path must be outside the exported dataset directory"
        )

    files = sorted(
        path
        for path in dataset_root.rglob("*")
        if path.is_file()
    )
    if not files:
        raise ValueError(f"exported dataset directory is empty: {dataset_root}")

    archive.parent.mkdir(parents=True, exist_ok=True)
    try:
        archive_prefix = dataset_root.relative_to(root_path)
    except ValueError:
        archive_prefix = Path(dataset_root.name)

    archive_names = [
        (archive_prefix / path.relative_to(dataset_root)).as_posix()
        for path in files
    ]

    manifest_name = manifest.relative_to(root_path).as_posix()
    fd, temp_name = tempfile.mkstemp(
        prefix=f".{archive.name}.",
        suffix=".tmp",
        dir=archive.parent,
    )
    os.close(fd)
    temp_archive = Path(temp_name)

    try:
        with zipfile.ZipFile(
            temp_archive,
            mode="w",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=6,
        ) as bundle:
            for path, arcname in zip(files, archive_names):
                bundle.write(path, arcname=arcname)
            bundle.write(manifest, arcname=manifest_name)

        temp_archive.replace(archive)
    except Exception:
        temp_archive.unlink(missing_ok=True)
        raise

    return PackSummary(
        archive_path=archive.as_posix(),
        file_count=len(files) + 1,
        size_bytes=archive.stat().st_size,
        sha256=_sha256_file(archive),
    )
