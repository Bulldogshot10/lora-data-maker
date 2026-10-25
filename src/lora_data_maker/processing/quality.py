from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Sequence

from PIL import Image, ImageFilter, ImageStat


class _PersonBoxLike(Protocol):
    left: int
    top: int
    right: int
    bottom: int
    confidence: float

    @property
    def area(self) -> int: ...


class _PersonDetectorLike(Protocol):
    def detect(self, image: Image.Image) -> Sequence[_PersonBoxLike]:
        ...


@dataclass(frozen=True)
class QualityResult:
    valid: bool
    width: int
    height: int
    aspect_ratio: float
    blur_score: float
    reason: str | None = None
    quality_score: float = 0.0
    person_count: int | None = None
    person_confidence: float | None = None
    subject_area_ratio: float | None = None
    subject_visibility: float | None = None
    text_count: int | None = None
    text_confidence: float | None = None
    contrast_score: float = 0.0
    exposure_score: float = 0.0
    reasons: tuple[str, ...] = ()


def laplacian_variance(
    image: Image.Image,
    *,
    sample_size: int = 256,
) -> float:
    if sample_size < 16:
        raise ValueError("sample_size must be at least 16")

    working = image.convert("L")
    working.thumbnail((sample_size, sample_size), Image.Resampling.BILINEAR)
    filtered = working.filter(
        ImageFilter.Kernel(
            (3, 3),
            [0, -1, 0, -1, 4, -1, 0, -1, 0],
            scale=1,
            offset=0,
        )
    )
    # Pillow's edge handling creates artificial gradients around the image
    # boundary for a zero-sum Laplacian kernel. Exclude that one-pixel rim so
    # a flat/blurred image is not mistakenly rated sharp from border artifacts.
    if filtered.width > 2 and filtered.height > 2:
        filtered = filtered.crop(
            (1, 1, filtered.width - 1, filtered.height - 1)
        )
    return float(ImageStat.Stat(filtered).var[0])


def image_contrast_score(image: Image.Image) -> float:
    gray = image.convert("L")
    rms = float(ImageStat.Stat(gray).rms[0])
    mean = float(ImageStat.Stat(gray).mean[0])
    spread = max(0.0, rms * rms - mean * mean) ** 0.5
    return _clamp(spread / 96.0)


def image_exposure_score(image: Image.Image) -> float:
    gray = image.convert("L")
    histogram = gray.histogram()
    total = max(1, sum(histogram))
    clipped = sum(histogram[:8]) + sum(histogram[-8:])
    clipped_ratio = clipped / total
    return _clamp(1.0 - clipped_ratio / 0.25)


def evaluate_quality(
    path: Path | str,
    *,
    min_width: int = 512,
    min_height: int = 512,
    blur_threshold: float = 0.0,
    max_aspect_ratio: float = 0.0,
    person_detector: _PersonDetectorLike | None = None,
    max_persons: int = 1,
    min_subject_area: float = 0.12,
    min_subject_visibility: float = 0.35,
    text_detector = None,
    text_confidence_threshold: float = 0.5,
    text_min_area_ratio: float = 0.0004,
) -> QualityResult:
    if min_width < 1 or min_height < 1:
        raise ValueError("minimum dimensions must be greater than zero")
    if blur_threshold < 0:
        raise ValueError("blur_threshold cannot be negative")
    if max_aspect_ratio < 0:
        raise ValueError("max_aspect_ratio cannot be negative")
    if max_persons < 1:
        raise ValueError("max_persons must be at least one")
    if not 0 <= min_subject_area <= 1:
        raise ValueError("min_subject_area must be between 0 and 1")
    if not 0 <= min_subject_visibility <= 1:
        raise ValueError("min_subject_visibility must be between 0 and 1")
    if not 0 < text_confidence_threshold <= 1:
        raise ValueError("text_confidence_threshold must be between 0 and 1")
    if not 0 <= text_min_area_ratio <= 1:
        raise ValueError("text_min_area_ratio must be between 0 and 1")

    with Image.open(Path(path).expanduser().resolve()) as image:
        image.load()
        width, height = image.size
        blur_score = laplacian_variance(image)
        contrast_score = image_contrast_score(image)
        exposure_score = image_exposure_score(image)

        detected = None
        text_detected = None
        if person_detector is not None:
            detected = tuple(person_detector.detect(image))
        if text_detector is not None:
            text_detected = tuple(text_detector.detect(image))

    aspect_ratio = max(width / height, height / width)
    reasons: list[str] = []

    if width < min_width or height < min_height:
        reasons.append(
            f"resolution too small: {width}x{height} < {min_width}x{min_height}"
        )

    if max_aspect_ratio > 0 and aspect_ratio > max_aspect_ratio:
        reasons.append(
            f"aspect ratio too extreme: {aspect_ratio:.3f} > {max_aspect_ratio:.3f}"
        )

    if blur_threshold > 0 and blur_score < blur_threshold:
        reasons.append(f"too blurry: score {blur_score:.3f} < {blur_threshold:.3f}")

    person_count: int | None = None
    person_confidence: float | None = None
    subject_area_ratio: float | None = None
    subject_visibility: float | None = None
    text_count: int | None = None
    text_confidence: float | None = None

    if text_detected is not None:
        text_count = len(text_detected)
        if text_count > 0:
            text_confidence = max(float(item.confidence) for item in text_detected)
            reasons.append(
                "text detected: "
                f"{text_count} region(s), max confidence {text_confidence:.3f}"
            )

    if detected is not None:
        person_count = len(detected)
        if person_count == 0:
            reasons.append("subject not visible: no person detected")
        elif person_count > max_persons:
            reasons.append(
                f"multiple people: detected {person_count}, maximum {max_persons}"
            )

        if person_count > 0:
            primary = max(
                detected,
                key=lambda item: (item.area, item.confidence),
            )
            person_confidence = float(primary.confidence)
        if person_count > 0:
            subject_area_ratio = _clamp(
                primary.area / max(1, width * height)
            )
            subject_visibility = _subject_visibility(
                primary,
                width=width,
                height=height,
                area_ratio=subject_area_ratio,
            )
            if subject_area_ratio < min_subject_area:
                reasons.append(
                    f"subject too small: area {subject_area_ratio:.3f} < {min_subject_area:.3f}"
                )
            elif subject_visibility < min_subject_visibility:
                reasons.append(
                    "subject not visible: bounding box visibility "
                    f"{subject_visibility:.3f} < {min_subject_visibility:.3f}"
                )

    sharpness_score = _sigmoid_score(blur_score, scale=120.0)
    subject_score = (
        1.0 if subject_visibility is None else subject_visibility
    )
    quality_score = (
        0.45 * sharpness_score
        + 0.20 * contrast_score
        + 0.10 * exposure_score
        + 0.25 * subject_score
    )
    if detected is not None and person_count == 0:
        quality_score *= 0.25
    if detected is not None and person_count is not None and person_count > max_persons:
        quality_score *= 0.5

    valid = not reasons
    return QualityResult(
        valid=valid,
        width=width,
        height=height,
        aspect_ratio=aspect_ratio,
        blur_score=blur_score,
        reason="; ".join(reasons) if reasons else None,
        quality_score=_clamp(quality_score),
        person_count=person_count,
        person_confidence=person_confidence,
        subject_area_ratio=subject_area_ratio,
        subject_visibility=subject_visibility,
        text_count=text_count,
        text_confidence=text_confidence,
        contrast_score=contrast_score,
        exposure_score=exposure_score,
        reasons=tuple(reasons),
    )


def _subject_visibility(
    person: _PersonBoxLike,
    *,
    width: int,
    height: int,
    area_ratio: float,
) -> float:
    if person.right <= person.left or person.bottom <= person.top:
        return 0.0

    edge_touch = sum(
        (
            person.left <= 1,
            person.top <= 1,
            person.right >= width - 1,
            person.bottom >= height - 1,
        )
    )
    edge_score = max(0.4, 1.0 - edge_touch * 0.15)
    area_score = _clamp(area_ratio / 0.30)
    confidence_score = _clamp(float(person.confidence) / 0.75)
    return _clamp(0.55 * area_score + 0.30 * confidence_score + 0.15 * edge_score)


def _sigmoid_score(value: float, *, scale: float) -> float:
    if value <= 0:
        return 0.0
    return _clamp(value / (value + scale))


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, float(value)))
