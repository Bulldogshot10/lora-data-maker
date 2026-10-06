from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from .filters import clean_tags

if TYPE_CHECKING:
    from ..models import ProjectManifest


@dataclass(frozen=True)
class TagCleanSummary:
    scanned: int = 0
    cleaned: int = 0
    skipped: int = 0


def clean_tag_manifest(
    manifest: "ProjectManifest",
    *,
    remove_tags: list[str] | tuple[str, ...] = (),
) -> TagCleanSummary:
    """Clean tagged image records and inject manifest.subject exactly once."""
    candidates = sorted(
        (
            image
            for image in manifest.images.values()
            if image.status == "tagged"
        ),
        key=lambda image: image.path,
    )

    cleaned = 0
    for image in candidates:
        before = list(image.tags)
        image.tags = clean_tags(
            before,
            remove=remove_tags,
            subject=manifest.subject,
        )

        image.metadata["tag_clean_stage"] = "cleaned"
        image.metadata["tag_clean_removed"] = [
            tag for tag in before
            if tag not in image.tags
        ]
        image.metadata["tag_clean_subject"] = manifest.subject
        image.metadata["tag_clean_trigger"] = manifest.trigger

        # Phase 8 mutates only tags/metadata. The image remains tagged;
        # phase 9 owns caption-file generation and trigger injection.
        image.set_status("tagged")
        cleaned += 1

    return TagCleanSummary(
        scanned=len(candidates),
        cleaned=cleaned,
        skipped=sum(
            1
            for image in manifest.images.values()
            if image.status != "tagged"
        ),
    )
