from __future__ import annotations

import hashlib
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Protocol, Sequence

from PIL import Image
import numpy as np

try:
    import numpy as _np
except ImportError:
    _np = None


@dataclass(frozen=True)
class PersonBox:
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


class PersonDetector(Protocol):
    def detect(self, image: Image.Image) -> Sequence[PersonBox]:
        """Return person bounding boxes in source-image coordinates."""


class YOLOXNanoPersonDetector:
    """YOLOX-Nano COCO person detector using ONNX Runtime only."""

    MODEL_FILENAME = "yolox_nano.onnx"
    MODEL_URL = (
        "https://huggingface.co/hr16/yolox-onnx/resolve/"
        "761141cbf90ddd9438943b3cd1b1c46ade29c714/yolox_nano.onnx"
    )
    MODEL_SHA256 = (
        "c789161ed43c8269fcd4e67c67eeeb4e80c622da2eb296a20bc6007bd18a0b7d"
    )
    MODEL_REVISION = "761141cbf90ddd9438943b3cd1b1c46ade29c714"
    INPUT_SIZE = (416, 416)
    PERSON_CLASS_ID = 0

    def __init__(
        self,
        *,
        cache_dir: Path | str | None = None,
        model_path: Path | str | None = None,
        confidence_threshold: float = 0.35,
        nms_threshold: float = 0.45,
        providers: Sequence[str] | None = None,
        session_factory: Callable[..., object] | None = None,
    ) -> None:
        if not 0 < confidence_threshold <= 1:
            raise ValueError("confidence_threshold must be between 0 and 1")
        if not 0 < nms_threshold <= 1:
            raise ValueError("nms_threshold must be between 0 and 1")

        self.confidence_threshold = float(confidence_threshold)
        self.nms_threshold = float(nms_threshold)
        self.model_path = self._resolve_model_path(
            cache_dir=cache_dir,
            model_path=model_path,
        )

        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise RuntimeError(
                "person quality checks require NumPy and ONNX Runtime; "
                "install: python -m pip install '.[quality]'"
            ) from exc

        factory = session_factory or ort.InferenceSession
        session_options = None
        if providers is None:
            providers = ("CPUExecutionProvider",)
        self._session = factory(
            str(self.model_path),
            sess_options=session_options,
            providers=list(providers),
        )

        self._input_name = self._get_input_name()
        self._np = _np if _np is not None else np

    def detect(self, image: Image.Image) -> Sequence[PersonBox]:
        array = self._preprocess(image)
        outputs = self._session.run(None, {self._input_name: array})
        if not outputs:
            return ()
        decoded = self._decode_outputs(outputs[0], image.size)
        if decoded.size == 0:
            return ()

        boxes = decoded[:, :4]
        scores = decoded[:, 4]
        keep = _nms(boxes, scores, self.nms_threshold)

        result: list[PersonBox] = []
        width, height = image.size
        for index in keep:
            x1, y1, x2, y2, score = decoded[index].tolist()
            left = max(0, min(width, int(round(x1))))
            top = max(0, min(height, int(round(y1))))
            right = max(left, min(width, int(round(x2))))
            bottom = max(top, min(height, int(round(y2))))
            if right <= left or bottom <= top:
                continue
            result.append(
                PersonBox(
                    left=left,
                    top=top,
                    right=right,
                    bottom=bottom,
                    confidence=float(score),
                )
            )
        return tuple(result)

    def _get_input_name(self) -> str:
        inputs = self._session.get_inputs()
        if not inputs:
            raise RuntimeError("YOLOX-Nano model has no inputs")
        return str(inputs[0].name)

    def _preprocess(self, image: Image.Image) -> np.ndarray:
        working = np.asarray(image.convert("RGB"), dtype=np.uint8)[..., ::-1]
        original_height, original_width = working.shape[:2]
        target_width, target_height = self.INPUT_SIZE

        ratio = min(
            target_width / max(1, original_width),
            target_height / max(1, original_height),
        )
        resized_width = max(1, int(round(original_width * ratio)))
        resized_height = max(1, int(round(original_height * ratio)))

        resized = Image.fromarray(working).resize(
            (resized_width, resized_height),
            Image.Resampling.BILINEAR,
        )
        canvas = np.full(
            (target_height, target_width, 3),
            114,
            dtype=np.uint8,
        )
        resized_array = np.asarray(resized, dtype=np.uint8)
        pad_y = (target_height - resized_height) // 2
        pad_x = (target_width - resized_width) // 2
        canvas[
            pad_y : pad_y + resized_height,
            pad_x : pad_x + resized_width,
        ] = resized_array

        tensor = canvas.astype(np.float32) / 255.0
        return tensor.transpose(2, 0, 1)[None, ...]

    def _decode_outputs(
        self,
        output: np.ndarray,
        image_size: tuple[int, int],
    ) -> np.ndarray:
        output_array = np.asarray(output)
        if output_array.ndim == 3:
            output_array = output_array[0]
        if output_array.ndim != 2:
            raise RuntimeError(
                f"unexpected YOLOX output shape: {tuple(np.asarray(output).shape)}"
            )
        if output_array.shape[1] <= 5 and output_array.shape[0] > 5:
            output_array = output_array.T
        if output_array.shape[1] < 6:
            raise RuntimeError(
                f"unexpected YOLOX output shape: {tuple(output_array.shape)}"
            )

        count = output_array.shape[0]
        grids: list[np.ndarray] = []
        expanded_strides: list[np.ndarray] = []
        for stride in (8, 16, 32):
            height = self.INPUT_SIZE[1] // stride
            width = self.INPUT_SIZE[0] // stride
            yy, xx = np.meshgrid(
                np.arange(height),
                np.arange(width),
                indexing="ij",
            )
            grid = np.stack((xx, yy), axis=2).reshape(1, -1, 2)
            grids.append(grid)
            expanded_strides.append(
                np.full((1, height * width, 1), stride, dtype=np.float32)
            )
        grid_array = np.concatenate(grids, axis=1)[0]
        stride_array = np.concatenate(expanded_strides, axis=1)[0]

        if len(grid_array) != count:
            raise RuntimeError(
                f"YOLOX output candidate count {count} does not match "
                f"416x416 decode grid {len(grid_array)}"
            )

        decoded = output_array.astype(np.float32, copy=True)
        decoded[:, :2] = (decoded[:, :2] + grid_array) * stride_array
        decoded[:, 2:4] = np.exp(np.clip(decoded[:, 2:4], -10, 10)) * stride_array

        objectness = 1.0 / (1.0 + np.exp(-np.clip(decoded[:, 4], -60, 60)))
        class_logits = decoded[:, 5:]
        person_logits = class_logits[:, self.PERSON_CLASS_ID]
        person_scores = 1.0 / (
            1.0 + np.exp(-np.clip(person_logits, -60, 60))
        )
        scores = objectness * person_scores

        centers = decoded[:, :2]
        half = decoded[:, 2:4] * 0.5
        x1y1 = centers - half
        x2y2 = centers + half

        width, height = image_size
        input_width, input_height = self.INPUT_SIZE
        ratio = min(
            input_width / max(1, width),
            input_height / max(1, height),
        )
        pad_x = (input_width - width * ratio) / 2
        pad_y = (input_height - height * ratio) / 2

        boxes = np.empty((count, 5), dtype=np.float32)
        boxes[:, :2] = (x1y1 - (pad_x, pad_y)) / ratio
        boxes[:, 2:4] = (x2y2 - (pad_x, pad_y)) / ratio
        boxes[:, :4] = boxes[:, :4].clip(
            (0, 0, 0, 0),
            (width, height, width, height),
        )
        boxes[:, 4] = scores
        return boxes[scores >= self.confidence_threshold]


def _nms(
    boxes: np.ndarray,
    scores: np.ndarray,
    threshold: float,
) -> list[int]:
    if len(boxes) == 0:
        return []

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    order = scores.argsort()[::-1]
    keep: list[int] = []

    while order.size:
        index = int(order[0])
        keep.append(index)
        if order.size == 1:
            break

        rest = order[1:]
        intersection_left = np.maximum(x1[index], x1[rest])
        intersection_top = np.maximum(y1[index], y1[rest])
        intersection_right = np.minimum(x2[index], x2[rest])
        intersection_bottom = np.minimum(y2[index], y2[rest])
        intersection = (
            np.maximum(0, intersection_right - intersection_left)
            * np.maximum(0, intersection_bottom - intersection_top)
        )
        union = areas[index] + areas[rest] - intersection
        iou = np.divide(
            intersection,
            union,
            out=np.zeros_like(intersection, dtype=np.float32),
            where=union > 0,
        )
        order = rest[iou <= threshold]

    return keep


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
