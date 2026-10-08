from __future__ import annotations

import hashlib
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

from PIL import Image


@dataclass(frozen=True)
class TextRegion:
    left: int
    top: int
    right: int
    bottom: int
    confidence: float

    @property
    def width(self) -> int:
        return max(0, self.right - self.left)

    @property
    def height(self) -> int:
        return max(0, self.bottom - self.top)

    @property
    def area(self) -> int:
        return self.width * self.height


class PPOCRTextDetector:
    """PP-OCRv3 English scene-text detector via OpenCV DNN."""

    MODEL_FILENAME = "text_detection_en_ppocrv3_2023may.onnx"
    MODEL_REVISION = "eb8f1ab7550dc8d895a1c950f6a7e9ba52df0d27"
    MODEL_URL = (
        "https://huggingface.co/opencv/text_detection_ppocr/resolve/"
        f"{MODEL_REVISION}/text_detection_en_ppocrv3_2023may.onnx"
    )
    MODEL_SHA256 = (
        "03f550c6b406fda8bf54bd8327815f6c7e2edd98cea02348c93d879254366587"
    )
    INPUT_SIZE = (736, 736)

    def __init__(
        self,
        *,
        cache_dir: Path | str | None = None,
        model_path: Path | str | None = None,
        confidence_threshold: float = 0.5,
        min_area_ratio: float = 0.0004,
        providers: Sequence[str] | None = None,
        detector_factory: Callable[..., object] | None = None,
    ) -> None:
        if not 0 < confidence_threshold <= 1:
            raise ValueError("text confidence_threshold must be between 0 and 1")
        if not 0 <= min_area_ratio <= 1:
            raise ValueError("text min_area_ratio must be between 0 and 1")

        self.confidence_threshold = float(confidence_threshold)
        self.min_area_ratio = float(min_area_ratio)
        self.model_path = self._resolve_model_path(
            cache_dir=cache_dir,
            model_path=model_path,
        )

        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError(
                "text quality checks require OpenCV; "
                "install: python -m pip install '.[quality]'"
            ) from exc

        self._cv2 = cv2
        net = cv2.dnn.readNet(str(self.model_path))
        if detector_factory is None:
            detector = cv2.dnn_TextDetectionModel_DB(net)
        else:
            detector = detector_factory(net)

        detector.setBinaryThreshold(0.3)
        detector.setPolygonThreshold(0.5)
        detector.setUnclipRatio(2.0)
        detector.setMaxCandidates(200)
        detector.setInputSize(self.INPUT_SIZE)
        detector.setInputMean((123.675, 116.28, 103.53))
        import numpy as np
        detector.setInputScale(1.0 / 255.0 / np.array([0.229, 0.224, 0.225]))
        self._detector = detector

    def _resolve_model_path(
        self,
        *,
        cache_dir: Path | str | None,
        model_path: Path | str | None,
    ) -> Path:
        if model_path is not None:
            resolved = Path(model_path).expanduser().resolve()
            if not resolved.is_file():
                raise RuntimeError(f"text detector model not found: {resolved}")
            self._verify_model(resolved)
            return resolved

        root = (
            Path(cache_dir).expanduser().resolve()
            if cache_dir is not None
            else Path.home() / ".cache" / "lora-data-maker" / "text"
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
                f"could not read text detector model: {path}"
            ) from exc
        if digest != self.MODEL_SHA256:
            raise RuntimeError(f"text detector model checksum mismatch: {path}")

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
                        f"could not download text detector model: {exc}"
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

    def detect(self, image: Image.Image) -> Sequence[TextRegion]:
        cv2 = self._cv2
        import numpy as np

        rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        height, width = bgr.shape[:2]
        if width < 1 or height < 1:
            return ()

        working = cv2.resize(bgr, self.INPUT_SIZE, interpolation=cv2.INTER_LINEAR)
        boxes, scores = self._detector.detect(working)
        if boxes is None:
            return ()

        result: list[TextRegion] = []
        scale_x = width / self.INPUT_SIZE[0]
        scale_y = height / self.INPUT_SIZE[1]
        image_area = max(1, width * height)

        for box, score in zip(boxes, scores):
            confidence = float(score)
            if confidence < self.confidence_threshold:
                continue
            points = np.asarray(box, dtype=np.float32).reshape(-1, 2)
            left = max(0, min(width, int(round(float(points[:, 0].min()) * scale_x))))
            top = max(0, min(height, int(round(float(points[:, 1].min()) * scale_y))))
            right = max(left, min(width, int(round(float(points[:, 0].max()) * scale_x))))
            bottom = max(top, min(height, int(round(float(points[:, 1].max()) * scale_y))))
            region = TextRegion(left, top, right, bottom, confidence)
            if region.area / image_area < self.min_area_ratio:
                continue
            if region.width < 8 or region.height < 4:
                continue
            result.append(region)

        return tuple(result)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
