from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from .filters import clean_tags, normalize_tag

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

    blacklist = {
        normalize_tag(tag)
        for tag in remove_tags
        if normalize_tag(tag)
    }

    for image in candidates:
        before = list(image.tags)
        normalized_before = [
            normalize_tag(tag)
            for tag in before
            if normalize_tag(tag)
        ]
        image.tags = clean_tags(
            before,
            remove=remove_tags,
            subject=manifest.subject,
        )

        image.metadata["tag_clean_stage"] = "cleaned"
        image.metadata["tag_clean_input_count"] = len(before)
        image.metadata["tag_clean_output_count"] = len(image.tags)
        image.metadata["tag_clean_blacklist"] = sorted(blacklist)
        image.metadata["tag_clean_subject"] = manifest.subject
        image.metadata["tag_clean_trigger"] = manifest.trigger
        image.metadata["tag_clean_duplicates_removed"] = max(
            0,
            len(set(normalized_before)) - len(
                [tag for tag in normalized_before if tag not in blacklist]
            ),
        )

        # Phase 8 mutates only tags/metadata. The image remains tagged;
        # phase 9 owns caption-file generation and trigger injection.
        image.set_status("tagged")

    return TagCleanSummary(
        scanned=len(candidates),
        cleaned=len(candidates),
        skipped=sum(
            1
            for image in manifest.images.values()
            if image.status != "tagged"
        ),
    )
