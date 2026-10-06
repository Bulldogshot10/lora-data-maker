from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from .images import iter_image_files
from .manifest import find_image_by_path, register_image
from .paths import manifest_path

if TYPE_CHECKING:
    from .models import ProjectManifest


@dataclass(frozen=True)
class ReconcileSummary:
    scanned: int = 0
    added: int = 0
    existing: int = 0
    missing: int = 0


def reconcile_raw(
    manifest: "ProjectManifest",
    *,
    root: Path | str,
    raw_dir: Path | str,
) -> ReconcileSummary:
    """Reconcile raw files into the manifest without deleting or regressing state."""
    root_path = Path(root).expanduser().resolve()
    raw_root = Path(raw_dir).expanduser().resolve()

    scanned = added = existing = 0
    present_paths: set[str] = set()

    for image_path in iter_image_files(raw_root):
        relative_path = manifest_path(root_path, image_path)
        present_paths.add(relative_path)
        scanned += 1

        record = find_image_by_path(manifest, relative_path)
        if record is None:
            record = register_image(
                manifest,
                path=relative_path,
                status="discovered",
            )
            record.metadata["reconcile_added"] = True
            added += 1
        else:
            existing += 1
            if record.metadata.pop("reconcile_raw_missing", None):
                record.metadata["reconcile_raw_restored"] = True

        record.metadata["reconcile_last_seen_raw"] = True

    missing = 0
    for record in manifest.images.values():
        record_path = Path(record.path).as_posix()
        if not record_path:
            continue

        try:
            absolute = (root_path / record_path).resolve()
            absolute.relative_to(raw_root)
        except ValueError:
            continue

        if record_path not in present_paths:
            missing += 1
            record.metadata["reconcile_raw_missing"] = True

    return ReconcileSummary(
        scanned=scanned,
        added=added,
        existing=existing,
        missing=missing,
    )
