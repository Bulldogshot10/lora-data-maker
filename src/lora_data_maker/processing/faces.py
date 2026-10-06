from __future__ import annotations

import hashlib
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Protocol, Sequence

from PIL import Image


@dataclass(frozen=True)
class FaceBox:
    left: int
    top: int
    right: int
    bottom: int

    @property
    def width(self) -> int:
        return max(0, self.right - self.left)

    @property
    def height(self) -> int:
        return max(0, self.bottom - self.top)

    @property
    def area(self) -> int:
        return self.width * self.height

    @property
    def center(self) -> tuple[float, float]:
        return (
            (self.left + self.right) / 2,
            (self.top + self.bottom) / 2,
        )


class FaceDetector(Protocol):
    def detect(self, image: Image.Image) -> Sequence[FaceBox]:
        """Return face bounding boxes in source-image coordinates."""


class YuNetFaceDetector:
    """OpenCV YuNet face detector backed by the pinned 2023mar ONNX model."""

    MODEL_FILENAME = "face_detection_yunet_2023mar.onnx"
    MODEL_URL = (
        "https://github.com/opencv/opencv_zoo/raw/"
        "47534e27c9851bb1128ccc0102f1145e27f23f98/"
        "models/face_detection_yunet/"
        "face_detection_yunet_2023mar.onnx"
    )
    MODEL_SHA256 = (
        "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4"
    )
    MODEL_REVISION = "47534e27c9851bb1128ccc0102f1145e27f23f98"

    def __init__(
        self,
        *,
        cache_dir: Path | str | None = None,
        model_path: Path | str | None = None,
        confidence_threshold: float = 0.6,
        nms_threshold: float = 0.3,
        top_k: int = 5000,
        detector_factory: Callable[..., object] | None = None,
    ) -> None:
        if not 0 < confidence_threshold <= 1:
            raise ValueError("confidence_threshold must be between 0 and 1")
        if not 0 < nms_threshold <= 1:
            raise ValueError("nms_threshold must be between 0 and 1")
        if top_k < 1:
            raise ValueError("top_k must be greater than zero")

        try:
            import cv2 as cv
            import numpy as np
        except ImportError as exc:
            raise RuntimeError(
                "face crop requires OpenCV (cv2) and NumPy; "
                "on Termux install: pkg install x11-repo opencv-python; "
                "on Linux install: python -m pip install '.[face]'"
            ) from exc

        if not hasattr(cv, "FaceDetectorYN"):
            raise RuntimeError(
                "installed OpenCV does not provide FaceDetectorYN; "
                "install a current OpenCV build"
            )

        self._cv = cv
        self._np = np
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        self.top_k = top_k
        self.model_path = self._resolve_model_path(
            cache_dir=cache_dir,
            model_path=model_path,
        )
        factory = detector_factory or cv.FaceDetectorYN.create
        self._detector = factory(
            model=str(self.model_path),
            config="",
            input_size=(320, 320),
            score_threshold=confidence_threshold,
            nms_threshold=nms_threshold,
            top_k=top_k,
            backend_id=0,
            target_id=0,
        )

    def detect(self, image: Image.Image) -> Sequence[FaceBox]:
        rgb = image.convert("RGB")
        array = self._np.asarray(rgb, dtype=self._np.uint8)
        frame = self._cv.cvtColor(array, self._cv.COLOR_RGB2BGR)

        self._detector.setInputSize((frame.shape[1], frame.shape[0]))
        _, faces = self._detector.detect(frame)
        if faces is None:
            return ()

        height, width = frame.shape[:2]
        boxes: list[FaceBox] = []
        for row in faces:
            x, y, box_width, box_height = (float(value) for value in row[:4])

            left = max(0, int(round(x)))
            top = max(0, int(round(y)))
            right = min(width, int(round(x + box_width)))
            bottom = min(height, int(round(y + box_height)))

            if right <= left or bottom <= top:
                continue

            boxes.append(
                FaceBox(
                    left=left,
                    top=top,
                    right=right,
                    bottom=bottom,
                )
            )

        return boxes

    def _resolve_model_path(
        self,
        *,
        cache_dir: Path | str | None,
        model_path: Path | str | None,
    ) -> Path:
        if model_path is not None:
            resolved = Path(model_path).expanduser().resolve()
            if not resolved.is_file():
                raise RuntimeError(f"face detector model not found: {resolved}")
            self._verify_model(resolved)
            return resolved

        root = (
            Path(cache_dir).expanduser().resolve()
            if cache_dir is not None
            else Path.home() / ".cache" / "lora-data-maker" / "face"
        )
        root.mkdir(parents=True, exist_ok=True)
        resolved = root / self.MODEL_FILENAME

        if resolved.is_file() and self._matches_model(resolved):
            return resolved

        self._download_model(resolved)
        return resolved

    def _matches_model(self, path: Path) -> bool:
        try:
            return _sha256_file(path) == self.MODEL_SHA256
        except OSError:
            return False

    def _verify_model(self, path: Path) -> None:
        try:
            digest = _sha256_file(path)
        except OSError as exc:
            raise RuntimeError(
                f"could not read face detector model: {path}"
            ) from exc

        if digest != self.MODEL_SHA256:
            raise RuntimeError(
                "face detector model checksum mismatch: "
                f"{path}"
            )

    def _download_model(self, destination: Path) -> None:
        import requests

        fd, temp_name = tempfile.mkstemp(
            prefix=f".{destination.name}.",
            suffix=".part",
            dir=str(destination.parent),
        )
        temp_path = Path(temp_name)

        try:
            with os.fdopen(fd, "wb") as handle:
                try:
                    with requests.get(
                        self.MODEL_URL,
                        stream=True,
                        timeout=(5, 60),
                    ) as response:
                        response.raise_for_status()
                        for chunk in response.iter_content(chunk_size=64 * 1024):
                            if chunk:
                                handle.write(chunk)
                except requests.RequestException as exc:
                    raise RuntimeError(
                        f"could not download face detector model: {exc}"
                    ) from exc

                handle.flush()
                os.fsync(handle.fileno())

            self._verify_model(temp_path)
            os.replace(temp_path, destination)
        except Exception:
            try:
                temp_path.unlink()
            except FileNotFoundError:
                pass
            raise


def largest_face(faces: Sequence[FaceBox]) -> FaceBox | None:
    if not faces:
        return None
    return max(faces, key=lambda face: face.area)


def square_crop_around_face(
    image: Image.Image,
    face: FaceBox,
    *,
    padding: float = 1.75,
) -> Image.Image:
    if padding <= 0:
        raise ValueError("face crop padding must be greater than zero")

    width, height = image.size
    side = max(face.width, face.height) * padding
    if side <= 0:
        raise ValueError("face bounding box has zero area")
    side = min(side, float(width), float(height))

    center_x, center_y = face.center
    half = side / 2
    left = int(round(center_x - half))
    top = int(round(center_y - half))
    right = int(round(center_x + half))
    bottom = int(round(center_y + half))

    if left < 0:
        right -= left
        left = 0
    if top < 0:
        bottom -= top
        top = 0
    if right > width:
        left -= right - width
        right = width
    if bottom > height:
        top -= bottom - height
        bottom = height

    left = max(0, left)
    top = max(0, top)
    right = min(width, right)
    bottom = min(height, bottom)

    if right <= left or bottom <= top:
        raise ValueError("face crop produced an empty region")

    return image.crop((left, top, right, bottom))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
