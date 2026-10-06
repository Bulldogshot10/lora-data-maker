from __future__ import annotations

import os
import re
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from .paths import manifest_path

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .models import ProjectManifest


@dataclass(frozen=True)
class ExportSummary:
    scanned: int = 0
    exported: int = 0
    failed: int = 0
    skipped: int = 0
    output_dir: str = ""


def safe_component(value: str) -> str:
    """Convert a human label into one safe deterministic directory component."""
    text = str(value).strip()
    text = re.sub(r"\s+", "_", text)
    text = re.sub(r"[^\w.-]+", "_", text, flags=re.UNICODE)
    text = text.strip("._")
    return text or "dataset"


def kohya_folder_name(
    *,
    subject: str,
    repeats: int = 1,
    folder_name: str | None = None,
) -> str:
    if repeats < 1:
        raise ValueError("repeats must be at least 1")

    if folder_name is not None and folder_name.strip():
        candidate = folder_name.strip()
        if Path(candidate).is_absolute() or Path(candidate).name != candidate:
            raise ValueError("folder_name must be a single directory name")
        if candidate in {".", ".."} or ".." in Path(candidate).parts:
            raise ValueError("folder_name cannot contain '..'")
        return candidate

    return f"{repeats}_{safe_component(subject)}"


def _copy_atomic(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)

    fd, temp_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    os.close(fd)
    temp_path = Path(temp_name)

    try:
        shutil.copy2(source, temp_path)
        os.replace(temp_path, destination)
    except Exception:
        try:
            temp_path.unlink()
        except FileNotFoundError:
            pass
        raise


def export_manifest(
    manifest: "ProjectManifest",
    *,
    root: Path | str,
    output_dir: Path | str,
    repeats: int = 1,
    folder_name: str | None = None,
) -> ExportSummary:
    """Export captioned records as Kohya-style image/.txt pairs."""
    root_path = Path(root).expanduser().resolve()
    output_root = Path(output_dir).expanduser().resolve()
    folder = kohya_folder_name(
        subject=manifest.subject,
        repeats=repeats,
        folder_name=folder_name,
    )
    dataset_root = output_root / folder

    candidates = sorted(
        (
            image
            for image in manifest.images.values()
            if image.status in {"captioned", "exported"}
        ),
        key=lambda image: image.path,
    )

    exported = failed = 0

    for image in candidates:
        if not image.processed_path or not image.caption_path:
            failed += 1
            image.metadata["export_stage"] = "failed"
            image.metadata["export_error"] = (
                "processed_path and caption_path are required"
            )
            continue

        source_image = (root_path / image.processed_path).resolve()
        source_caption = (root_path / image.caption_path).resolve()

        if not source_image.is_file():
            failed += 1
            image.metadata["export_stage"] = "failed"
            image.metadata["export_error"] = (
                f"processed image not found: {source_image}"
            )
            continue

        if not source_caption.is_file():
            failed += 1
            image.metadata["export_stage"] = "failed"
            image.metadata["export_error"] = (
                f"caption file not found: {source_caption}"
            )
            continue

        destination_image = dataset_root / f"{image.image_id}{source_image.suffix.lower()}"
        destination_caption = dataset_root / f"{image.image_id}.txt"

        try:
            _copy_atomic(source_image, destination_image)
            _copy_atomic(source_caption, destination_caption)
        except (OSError, ValueError, shutil.Error) as exc:
            failed += 1
            image.metadata["export_stage"] = "failed"
            image.metadata["export_error"] = str(exc)
            continue

        image.metadata.update(
            {
                "export_stage": "exported",
                "export_layout": "kohya",
                "export_repeats": repeats,
                "export_folder": folder,
                "export_image_path": manifest_path(root_path, destination_image),
                "export_caption_path": manifest_path(root_path, destination_caption),
            }
        )
        image.set_status("exported")
        exported += 1

    skipped = sum(
        1
        for image in manifest.images.values()
        if image.status not in {"captioned", "exported"}
    )

    return ExportSummary(
        scanned=len(candidates),
        exported=exported,
        failed=failed,
        skipped=skipped,
        output_dir=dataset_root.as_posix(),
    )
