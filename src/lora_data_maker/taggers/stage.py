from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from .base import Tagger, TaggerError

if TYPE_CHECKING:
    from ..models import ProjectManifest


@dataclass(frozen=True)
class TagSummary:
    scanned: int = 0
    tagged: int = 0
    failed: int = 0
    skipped: int = 0


def tag_manifest(
    manifest: "ProjectManifest",
    *,
    root: Path | str,
    tagger: Tagger,
    use_processed_images: bool = True,
) -> TagSummary:
    root_path = Path(root).expanduser().resolve()
    candidates = sorted(
        (
            image
            for image in manifest.images.values()
            if image.status in {"validated", "processed"} and image.path
        ),
        key=lambda image: image.path,
    )

    if not candidates:
        return TagSummary()

    processed_paths: list[Path] = []
    for image in candidates:
        source_path: Path | None = None
        if use_processed_images and image.processed_path:
            candidate = (root_path / str(image.processed_path)).resolve()
            try:
                candidate.relative_to(root_path)
            except ValueError:
                candidate = None
            if candidate is not None and candidate.is_file():
                source_path = candidate

        if source_path is None:
            candidate = (root_path / image.path).resolve()
            try:
                candidate.relative_to(root_path)
            except ValueError:
                candidate = None
            if candidate is not None and candidate.is_file():
                source_path = candidate

        if source_path is None:
            raise TaggerError(
                f"could not find raw or processed image for tagging: {image.path}"
            )
        processed_paths.append(source_path)

    try:
        results = tagger.tag_images(processed_paths)
    except TaggerError:
        raise
    except Exception as exc:
        raise TaggerError(f"tagger inference failed: {exc}") from exc

    if len(results) != len(candidates):
        raise TaggerError(
            f"tagger returned {len(results)} results for "
            f"{len(candidates)} images"
        )

    for image, result in zip(candidates, results):
        image.tags = list(result.tags)
        image.metadata["tagger_scores"] = dict(result.scores)
        image.metadata["tagger_ratings"] = dict(result.ratings)
        image.metadata["tagger_characters"] = dict(result.characters)
        image.metadata["tagger_stage"] = "tagged"
        image.set_status("tagged")

    return TagSummary(
        scanned=len(candidates),
        tagged=len(candidates),
        failed=0,
        skipped=sum(
            1
            for image in manifest.images.values()
            if image.status != "processed"
        ),
    )
