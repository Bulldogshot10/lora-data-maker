from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PIL import UnidentifiedImageError
from PIL.Image import DecompressionBombError

from .dedup import (
    find_perceptual_duplicates,
    perceptual_hash,
    sha256_file,
)
from .quality import evaluate_quality
from .person import PersonDetector


@dataclass(frozen=True)
class CleanSummary:
    scanned: int = 0
    exact_duplicates: int = 0
    perceptual_duplicates: int = 0
    quality_rejected: int = 0
    kept: int = 0


def clean_manifest(
    manifest,
    *,
    root: Path | str,
    exact: bool = True,
    perceptual: bool = True,
    perceptual_distance: int = 6,
    min_width: int = 512,
    min_height: int = 512,
    blur_threshold: float = 0.0,
    max_aspect_ratio: float = 0.0,
    person_detector: PersonDetector | None = None,
    max_persons: int = 1,
    min_subject_area: float = 0.12,
    min_subject_visibility: float = 0.35,
) -> CleanSummary:
    if perceptual_distance < 0:
        raise ValueError("perceptual_distance cannot be negative")

    root_path = Path(root).expanduser().resolve()
    candidates = sorted(
        (
            image
            for image in manifest.images.values()
            if image.status == "validated"
        ),
        key=lambda image: image.path,
    )

    quality_rejected = 0
    quality_candidates = list(candidates)

    for image in quality_candidates:
        try:
            result = evaluate_quality(
                root_path / image.path,
                min_width=min_width,
                min_height=min_height,
                blur_threshold=blur_threshold,
                max_aspect_ratio=max_aspect_ratio,
                person_detector=person_detector,
                max_persons=max_persons,
                min_subject_area=min_subject_area,
                min_subject_visibility=min_subject_visibility,
            )
        except (OSError, UnidentifiedImageError, DecompressionBombError):
            image.set_status(
                "rejected",
                reason="quality check could not read image",
            )
            image.metadata["quality_stage"] = "rejected"
            quality_rejected += 1
            continue

        image.width = result.width
        image.height = result.height
        image.metadata["quality_blur_score"] = result.blur_score
        image.metadata["quality_aspect_ratio"] = result.aspect_ratio
        image.metadata["quality_score"] = result.quality_score
        image.metadata["quality_contrast_score"] = result.contrast_score
        image.metadata["quality_exposure_score"] = result.exposure_score
        image.metadata["quality_reasons"] = list(result.reasons)
        if result.person_count is not None:
            image.metadata["quality_person_count"] = result.person_count
            image.metadata["quality_person_confidence"] = result.person_confidence
            image.metadata["quality_subject_area_ratio"] = result.subject_area_ratio
            image.metadata["quality_subject_visibility"] = result.subject_visibility

        if result.valid:
            continue

        image.set_status(
            "rejected",
            reason=result.reason or "quality filter rejected image",
        )
        image.metadata["quality_stage"] = "rejected"
        quality_rejected += 1

    dedup_candidates = [
        image
        for image in candidates
        if image.status == "validated"
    ]

    exact_duplicates = _exact_deduplicate(
        dedup_candidates,
        root_path,
        enabled=exact,
    )

    perceptual_duplicates = 0
    if perceptual:
        remaining = [
            image
            for image in dedup_candidates
            if image.status == "validated"
        ]
        hashes: list[tuple[str, str]] = []

        for image in remaining:
            path = root_path / image.path
            try:
                image.perceptual_hash = perceptual_hash(path)
            except (OSError, ValueError):
                image.set_status(
                    "rejected",
                    reason="could not compute perceptual hash",
                )
                continue
            hashes.append((image.image_id, image.perceptual_hash))

        for match in find_perceptual_duplicates(
            hashes,
            max_distance=perceptual_distance,
        ):
            image = manifest.images[match.image_id]
            winner = manifest.images[match.duplicate_of]
            image.set_status(
                "duplicate",
                reason=(
                    f"near-duplicate of {winner.path}; "
                    f"pHash distance={match.distance}"
                ),
            )
            image.metadata["duplicate_of"] = winner.image_id
            image.metadata["duplicate_kind"] = "perceptual"
            image.metadata["perceptual_distance"] = match.distance
            perceptual_duplicates += 1


    kept = sum(image.status == "validated" for image in candidates)

    return CleanSummary(
        scanned=len(candidates),
        exact_duplicates=exact_duplicates,
        perceptual_duplicates=perceptual_duplicates,
        quality_rejected=quality_rejected,
        kept=kept,
    )


def _exact_deduplicate(candidates, root: Path, *, enabled: bool) -> int:
    if not enabled:
        for image in candidates:
            try:
                image.sha256 = sha256_file(root / image.path)
            except (OSError, ValueError):
                image.set_status(
                    "rejected",
                    reason="could not compute sha256",
                )
                image.metadata["dedup_stage"] = "rejected"
        return 0

    seen: dict[str, object] = {}
    duplicate_count = 0

    for image in candidates:
        path = root / image.path
        try:
            image.sha256 = sha256_file(path)
        except (OSError, ValueError):
            image.set_status(
                "rejected",
                reason="could not compute sha256",
            )
            image.metadata["dedup_stage"] = "rejected"
            continue

        previous = seen.get(image.sha256)
        if previous is None:
            seen[image.sha256] = image
            continue

        image.set_status(
            "duplicate",
            reason=f"exact duplicate of {previous.path}; sha256={image.sha256}",
        )
        image.metadata["duplicate_of"] = previous.image_id
        image.metadata["duplicate_kind"] = "exact"
        duplicate_count += 1

    return duplicate_count
