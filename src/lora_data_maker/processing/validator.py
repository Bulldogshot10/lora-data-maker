from __future__ import annotations

import mimetypes
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from PIL import Image, UnidentifiedImageError
from PIL.Image import DecompressionBombError


SUPPORTED_MIME_BY_FORMAT = {
    "JPEG": "image/jpeg",
    "PNG": "image/png",
    "WEBP": "image/webp",
    "GIF": "image/gif",
    "BMP": "image/bmp",
    "TIFF": "image/tiff",
    "AVIF": "image/avif",
}


@dataclass(frozen=True)
class ImageValidationResult:
    path: Path
    valid: bool
    width: int | None = None
    height: int | None = None
    file_size: int | None = None
    format: str | None = None
    mime_type: str | None = None
    exif_orientation: int | None = None
    has_exif: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)
    reason: str | None = None


def validate_image(
    path: Path | str,
    *,
    min_width: int = 1,
    min_height: int = 1,
    max_width: int | None = None,
    max_height: int | None = None,
    min_file_size: int = 1,
    max_file_size: int | None = None,
) -> ImageValidationResult:
    image_path = Path(path).expanduser().resolve()

    if min_width < 1 or min_height < 1:
        raise ValueError("minimum dimensions must be greater than zero")
    if min_file_size < 0:
        raise ValueError("minimum file size cannot be negative")
    if max_width is not None and max_width < min_width:
        raise ValueError("max_width cannot be smaller than min_width")
    if max_height is not None and max_height < min_height:
        raise ValueError("max_height cannot be smaller than min_height")
    if max_file_size is not None and max_file_size < min_file_size:
        raise ValueError("max_file_size cannot be smaller than min_file_size")

    try:
        file_size = image_path.stat().st_size
    except FileNotFoundError:
        return ImageValidationResult(
            path=image_path,
            valid=False,
            reason="file not found",
        )
    except OSError as exc:
        return ImageValidationResult(
            path=image_path,
            valid=False,
            reason=f"cannot stat file: {exc}",
        )

    if file_size < min_file_size:
        return ImageValidationResult(
            path=image_path,
            valid=False,
            file_size=file_size,
            reason=f"file too small: {file_size} bytes < {min_file_size}",
        )
    if max_file_size is not None and file_size > max_file_size:
        return ImageValidationResult(
            path=image_path,
            valid=False,
            file_size=file_size,
            reason=f"file too large: {file_size} bytes > {max_file_size}",
        )

    try:
        # verify() must be called immediately after open(). Some Pillow
        # plugins invalidate their internal file pointer after metadata access.
        with Image.open(image_path) as image:
            image.verify()

        with Image.open(image_path) as image:
            image_format = (image.format or "").upper() or None
            width, height = image.size
            info = dict(image.info)

            exif = image.getexif()
            orientation = exif.get(274)
            has_exif = bool(exif)

            mime_type = (
                SUPPORTED_MIME_BY_FORMAT.get(image_format or "")
                or mimetypes.guess_type(image_path.name)[0]
            )

            # Fully decode the pixel data. This catches corruption that a
            # container/header verification may not expose.
            image.load()

            metadata = _metadata(image_path, image_format, info)
            metadata["mode"] = image.mode
            metadata["has_exif"] = has_exif
            if orientation is not None:
                metadata["exif_orientation"] = orientation

        if width < min_width or height < min_height:
            return ImageValidationResult(
                path=image_path,
                valid=False,
                width=width,
                height=height,
                file_size=file_size,
                format=image_format,
                mime_type=mime_type,
                exif_orientation=orientation,
                has_exif=has_exif,
                metadata=metadata,
                reason=(
                    f"dimensions too small: {width}x{height} "
                    f"< {min_width}x{min_height}"
                ),
            )

        if max_width is not None and width > max_width:
            return ImageValidationResult(
                path=image_path,
                valid=False,
                width=width,
                height=height,
                file_size=file_size,
                format=image_format,
                mime_type=mime_type,
                exif_orientation=orientation,
                has_exif=has_exif,
                metadata=metadata,
                reason=f"width too large: {width} > {max_width}",
            )

        if max_height is not None and height > max_height:
            return ImageValidationResult(
                path=image_path,
                valid=False,
                width=width,
                height=height,
                file_size=file_size,
                format=image_format,
                mime_type=mime_type,
                exif_orientation=orientation,
                has_exif=has_exif,
                metadata=metadata,
                reason=f"height too large: {height} > {max_height}",
            )

        return ImageValidationResult(
            path=image_path,
            valid=True,
            width=width,
            height=height,
            file_size=file_size,
            format=image_format,
            mime_type=mime_type,
            exif_orientation=orientation,
            has_exif=has_exif,
            metadata=metadata,
        )

    except (UnidentifiedImageError, DecompressionBombError):
        return ImageValidationResult(
            path=image_path,
            valid=False,
            file_size=file_size,
            reason="unidentified image, corrupt image, or unsafe image dimensions",
        )
    except (OSError, ValueError, SyntaxError, RuntimeError) as exc:
        return ImageValidationResult(
            path=image_path,
            valid=False,
            file_size=file_size,
            reason=f"image decode failed: {exc}",
        )


def _metadata(
    path: Path,
    image_format: str | None,
    info: dict[str, Any],
) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "format": image_format,
        "suffix": path.suffix.lower(),
    }

    for key in ("dpi", "duration", "loop", "background", "transparency"):
        if key not in info:
            continue
        value = _json_safe(value=info[key])
        if value is not None:
            metadata[key] = value

    return metadata


def _json_safe(value: Any) -> Any:
    """Convert Pillow metadata values into JSON-native types."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value

    if isinstance(value, dict):
        return {
            str(key): safe_value
            for key, item in value.items()
            if (safe_value := _json_safe(item)) is not None
        }

    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]

    # Pillow's IFDRational and similar numeric metadata types implement
    # float conversion but are not JSON serializable.
    try:
        return float(value)
    except (TypeError, ValueError, OverflowError):
        return str(value)
