from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageFilter, ImageStat


@dataclass(frozen=True)
class QualityResult:
    valid: bool
    width: int
    height: int
    aspect_ratio: float
    blur_score: float
    reason: str | None = None


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
    return float(ImageStat.Stat(filtered).var[0])


def evaluate_quality(
    path: Path | str,
    *,
    min_width: int = 512,
    min_height: int = 512,
    blur_threshold: float = 0.0,
    max_aspect_ratio: float = 0.0,
) -> QualityResult:
    if min_width < 1 or min_height < 1:
        raise ValueError("minimum dimensions must be greater than zero")
    if blur_threshold < 0:
        raise ValueError("blur_threshold cannot be negative")
    if max_aspect_ratio < 0:
        raise ValueError("max_aspect_ratio cannot be negative")

    with Image.open(Path(path).expanduser().resolve()) as image:
        image.load()
        width, height = image.size
        blur_score = laplacian_variance(image)

    aspect_ratio = max(width / height, height / width)

    if width < min_width or height < min_height:
        return QualityResult(
            valid=False,
            width=width,
            height=height,
            aspect_ratio=aspect_ratio,
            blur_score=blur_score,
            reason=(
                f"resolution too small: {width}x{height} "
                f"< {min_width}x{min_height}"
            ),
        )

    if max_aspect_ratio > 0 and aspect_ratio > max_aspect_ratio:
        return QualityResult(
            valid=False,
            width=width,
            height=height,
            aspect_ratio=aspect_ratio,
            blur_score=blur_score,
            reason=(
                f"aspect ratio too extreme: {aspect_ratio:.3f} "
                f"> {max_aspect_ratio:.3f}"
            ),
        )

    if blur_threshold > 0 and blur_score < blur_threshold:
        return QualityResult(
            valid=False,
            width=width,
            height=height,
            aspect_ratio=aspect_ratio,
            blur_score=blur_score,
            reason=(
                f"too blurry: score {blur_score:.3f} "
                f"< {blur_threshold:.3f}"
            ),
        )

    return QualityResult(
        valid=True,
        width=width,
        height=height,
        aspect_ratio=aspect_ratio,
        blur_score=blur_score,
    )
