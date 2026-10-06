from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from PIL import Image, ImageOps

from .faces import FaceDetector, largest_face, square_crop_around_face

if TYPE_CHECKING:
    from ..models import ProjectManifest


ResizeMode = Literal["cover", "fit", "stretch"]


@dataclass(frozen=True)
class ProcessConfig:
    target_width: int = 512
    target_height: int = 512
    resize_mode: ResizeMode = "cover"
    jpeg_quality: int = 95
    face_crop: bool = False
    face_padding: float = 1.75
    background: tuple[int, int, int] = (255, 255, 255)

    def __post_init__(self) -> None:
        if self.target_width < 1 or self.target_height < 1:
            raise ValueError("target dimensions must be greater than zero")
        if self.resize_mode not in {"cover", "fit", "stretch"}:
            raise ValueError(
                f"invalid resize mode {self.resize_mode!r}; "
                "expected 'cover', 'fit', or 'stretch'"
            )
        if not 1 <= self.jpeg_quality <= 100:
            raise ValueError("jpeg_quality must be between 1 and 100")
        if self.face_padding <= 0:
            raise ValueError("face_padding must be greater than zero")
        if len(self.background) != 3 or any(
            not 0 <= value <= 255 for value in self.background
        ):
            raise ValueError("background must contain three RGB bytes")


@dataclass(frozen=True)
class ProcessResult:
    output_path: Path
    source_size: tuple[int, int]
    output_size: tuple[int, int]
    crop_box: tuple[int, int, int, int] | None
    face_detected: bool
    source_orientation: int | None


def process_image(
    source_path: Path | str,
    output_path: Path | str,
    *,
    config: ProcessConfig | None = None,
    face_detector: FaceDetector | None = None,
) -> ProcessResult:
    settings = config or ProcessConfig()
    source = Path(source_path).expanduser().resolve()
    destination = Path(output_path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)

    if settings.face_crop and face_detector is None:
        raise ValueError("face crop is enabled but no face detector is configured")

    with Image.open(source) as opened:
        source_size = opened.size
        exif = opened.getexif()
        orientation = exif.get(274)
        image = ImageOps.exif_transpose(opened)
        image.load()
        image = image.convert("RGB")

    face_detected = False
    if settings.face_crop:
        faces = list(face_detector.detect(image))
        face = largest_face(faces)
        if face is not None:
            image = square_crop_around_face(
                image,
                face,
                padding=settings.face_padding,
            )
            face_detected = True

    processed, crop_box = resize_and_crop(
        image,
        settings.target_width,
        settings.target_height,
        mode=settings.resize_mode,
        background=settings.background,
    )

    processed.save(
        destination,
        format="JPEG",
        quality=settings.jpeg_quality,
        optimize=True,
        progressive=True,
    )

    return ProcessResult(
        output_path=destination,
        source_size=source_size,
        output_size=processed.size,
        crop_box=crop_box,
        face_detected=face_detected,
        source_orientation=orientation,
    )


def resize_and_crop(
    image: Image.Image,
    target_width: int,
    target_height: int,
    *,
    mode: ResizeMode = "cover",
    background: tuple[int, int, int] = (255, 255, 255),
) -> tuple[Image.Image, tuple[int, int, int, int] | None]:
    if target_width < 1 or target_height < 1:
        raise ValueError("target dimensions must be greater than zero")

    source_width, source_height = image.size
    if source_width < 1 or source_height < 1:
        raise ValueError("source image has invalid dimensions")

    if mode == "stretch":
        return (
            image.resize(
                (target_width, target_height),
                Image.Resampling.LANCZOS,
            ),
            None,
        )

    if mode == "fit":
        fitted = ImageOps.contain(
            image,
            (target_width, target_height),
            method=Image.Resampling.LANCZOS,
        )
        canvas = Image.new("RGB", (target_width, target_height), background)
        left = (target_width - fitted.width) // 2
        top = (target_height - fitted.height) // 2
        canvas.paste(fitted, (left, top))
        return canvas, None

    if mode != "cover":
        raise ValueError(f"unsupported resize mode: {mode}")

    scale = max(
        target_width / source_width,
        target_height / source_height,
    )
    resized_width = max(1, round(source_width * scale))
    resized_height = max(1, round(source_height * scale))

    resized = image.resize(
        (resized_width, resized_height),
        Image.Resampling.LANCZOS,
    )

    left = max(0, (resized_width - target_width) // 2)
    top = max(0, (resized_height - target_height) // 2)
    right = left + target_width
    bottom = top + target_height

    cropped = resized.crop((left, top, right, bottom))
    return cropped, (left, top, right, bottom)


def process_manifest(
    manifest: "ProjectManifest",
    *,
    root: Path | str,
    output_dir: Path | str,
    config: ProcessConfig | None = None,
    face_detector: FaceDetector | None = None,
) -> tuple[int, int, int]:
    settings = config or ProcessConfig()
    root_path = Path(root).expanduser().resolve()
    output_root = Path(output_dir).expanduser().resolve()
    processed_root = output_root / "processed"

    processed_count = failed_count = skipped_count = 0

    for image in sorted(
        manifest.images.values(),
        key=lambda item: item.path,
    ):
        if image.status != "validated":
            skipped_count += 1
            continue

        source = root_path / image.path
        destination = processed_root / f"{image.image_id}.jpg"

        try:
            result = process_image(
                source,
                destination,
                config=settings,
                face_detector=face_detector,
            )
        except (OSError, ValueError, RuntimeError, SyntaxError) as exc:
            image.set_status(
                "rejected",
                reason=f"processing failed: {exc}",
            )
            image.metadata["processing_stage"] = "rejected"
            failed_count += 1
            continue

        image.processed_path = result.output_path.relative_to(root_path).as_posix()
        image.metadata.update(
            {
                "processing_stage": "processed",
                "processing_mode": settings.resize_mode,
                "processing_output_format": "JPEG",
                "processing_output_size": list(result.output_size),
                "processing_source_size": list(result.source_size),
                "processing_face_detected": result.face_detected,
            }
        )
        if result.crop_box is not None:
            image.metadata["processing_crop_box"] = list(result.crop_box)
        if result.source_orientation is not None:
            image.metadata["processing_source_orientation"] = result.source_orientation

        image.set_status("processed")
        processed_count += 1

    return processed_count, failed_count, skipped_count
