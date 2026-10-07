from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from PIL import Image

from .base import TagResult, TaggerError
from .model import HuggingFaceModelCache


JOYTAG_MEAN = (0.48145466, 0.4578275, 0.40821073)
JOYTAG_STD = (0.26862954, 0.26130258, 0.27577711)
JOYTAG_DEFAULT_THRESHOLD = 0.4


class JoyTagLabels:
    def __init__(self, names: list[str]) -> None:
        self.names = names

    @classmethod
    def load(cls, path: Path | str) -> "JoyTagLabels":
        try:
            names = [
                line.strip()
                for line in Path(path).read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        except OSError as exc:
            raise TaggerError(f"could not read JoyTag labels: {path}") from exc

        if not names:
            raise TaggerError("JoyTag labels are empty")

        return cls(names)


class JoyTagTagger:
    def __init__(
        self,
        *,
        model_cache: HuggingFaceModelCache,
        model_sha256: str | None = None,
        threshold: float = JOYTAG_DEFAULT_THRESHOLD,
        batch_size: int = 1,
        session_factory: Callable[[str], Any] | None = None,
    ) -> None:
        if not 0 <= threshold <= 1:
            raise ValueError("JoyTag threshold must be between 0 and 1")
        if batch_size < 1:
            raise ValueError("batch_size must be greater than zero")

        self.model_cache = model_cache
        self.model_sha256 = model_sha256
        self.threshold = threshold
        self.batch_size = batch_size
        self.session_factory = session_factory

        self._session: Any | None = None
        self._labels: JoyTagLabels | None = None
        self._target_size: int | None = None
        self._input_layout: str | None = None

    def _load(self) -> None:
        if (
            self._session is not None
            and self._labels is not None
            and self._target_size is not None
            and self._input_layout is not None
        ):
            return

        try:
            import numpy as np
        except ImportError as exc:
            raise TaggerError(
                "JoyTag tagger requires numpy; install the 'tagger' extra"
            ) from exc

        try:
            if self.session_factory is None:
                import onnxruntime as ort

                session_factory = ort.InferenceSession
            else:
                session_factory = self.session_factory
        except ImportError as exc:
            raise TaggerError(
                "JoyTag tagger requires optional onnxruntime; "
                "install the 'tagger' extra"
            ) from exc

        try:
            files = self.model_cache.ensure(
                model_sha256=self.model_sha256,
            )
            session = session_factory(str(files.model_path))
            inputs = session.get_inputs()
            if not inputs:
                raise TaggerError("JoyTag ONNX model has no inputs")

            input_info = inputs[0]
            shape = input_info.shape
            if len(shape) != 4:
                raise TaggerError(
                    f"unexpected JoyTag ONNX input shape: {shape}"
                )

            target_size = None
            layout = None

            if _is_fixed_dim(shape[1], 3) or shape[1] in (None, "None"):
                if _is_fixed_dim(shape[2], shape[3]) and _is_fixed_dim(
                    shape[2], 448
                ):
                    target_size = 448
                    layout = "nchw"
            if (
                layout is None
                and _is_fixed_dim(shape[3], 3)
                and _is_fixed_dim(shape[1], shape[2])
            ):
                target_size = int(shape[1])
                layout = "nhwc"

            if target_size is None or layout is None:
                raise TaggerError(
                    "unexpected JoyTag ONNX input shape; expected "
                    "[batch, 3, 448, 448] or [batch, 448, 448, 3]: "
                    f"{shape}"
                )

            input_type = str(getattr(input_info, "type", "")).lower()
            if input_type and "float" not in input_type:
                raise TaggerError(
                    f"unexpected JoyTag ONNX input type: {input_type}"
                )

            labels = JoyTagLabels.load(files.labels_path)
        except TaggerError:
            raise
        except Exception as exc:
            raise TaggerError(
                f"could not load JoyTag tagger model: {exc}"
            ) from exc

        self._session = session
        self._labels = labels
        self._target_size = target_size
        self._input_layout = layout

    def tag_image(self, path: Path | str) -> TagResult:
        return self.tag_images([path])[0]

    def tag_images(self, paths: list[Path | str]) -> list[TagResult]:
        if not paths:
            return []

        self._load()

        results: list[TagResult] = []
        for start in range(0, len(paths), self.batch_size):
            batch_paths = paths[start : start + self.batch_size]
            batch = self._prepare_batch(batch_paths)

            input_name = self._session.get_inputs()[0].name
            output_name = self._session.get_outputs()[0].name
            predictions = self._session.run(
                [output_name],
                {input_name: batch},
            )[0]

            probabilities = self._sigmoid_batch(predictions)
            if len(probabilities) != len(batch_paths):
                raise TaggerError(
                    f"JoyTag returned {len(probabilities)} predictions for "
                    f"{len(batch_paths)} images"
                )

            results.extend(
                self._decode_prediction(prediction)
                for prediction in probabilities
            )

        return results

    def _prepare_batch(self, paths: list[Path | str]):
        import numpy as np

        if self._target_size is None or self._input_layout is None:
            raise TaggerError("JoyTag model input is not loaded")

        arrays = []
        for path in paths:
            try:
                with Image.open(
                    Path(path).expanduser().resolve()
                ) as image:
                    rgba = image.convert("RGBA")
                    canvas = Image.new(
                        "RGBA",
                        rgba.size,
                        (255, 255, 255, 255),
                    )
                    canvas.alpha_composite(rgba)
                    rgb = canvas.convert("RGB")

                    max_dim = max(rgb.size)
                    ratio = float(self._target_size) / float(max_dim)
                    new_size = (
                        max(1, int(rgb.width * ratio)),
                        max(1, int(rgb.height * ratio)),
                    )
                    resized = rgb.resize(
                        new_size,
                        Image.Resampling.BICUBIC,
                    )
                    square = Image.new(
                        "RGB",
                        (self._target_size, self._target_size),
                        (255, 255, 255),
                    )
                    square.paste(
                        resized,
                        (
                            (self._target_size - new_size[0]) // 2,
                            (self._target_size - new_size[1]) // 2,
                        ),
                    )

                    array = np.asarray(square, dtype=np.float32) / 255.0
                    mean = np.asarray(JOYTAG_MEAN, dtype=np.float32)
                    std = np.asarray(JOYTAG_STD, dtype=np.float32)
                    array = (array - mean) / std

                    if self._input_layout == "nchw":
                        array = np.transpose(array, (2, 0, 1))
                    arrays.append(array)
            except (OSError, ValueError) as exc:
                raise TaggerError(
                    f"could not prepare image for JoyTag: {path}"
                ) from exc

        return np.stack(arrays, axis=0)

    @staticmethod
    def _sigmoid_batch(predictions):
        import numpy as np

        values = np.asarray(predictions, dtype=np.float32)
        if values.ndim == 1:
            values = values.reshape(1, -1)
        if values.ndim != 2:
            raise TaggerError(
                f"JoyTag returned unexpected output shape: {values.shape}"
            )

        values = np.clip(values, -60.0, 60.0)
        return 1.0 / (1.0 + np.exp(-values))

    def _decode_prediction(self, probabilities) -> TagResult:
        if self._labels is None:
            raise TaggerError("JoyTag labels are not loaded")

        if len(probabilities) != len(self._labels.names):
            raise TaggerError(
                f"JoyTag output has {len(probabilities)} labels; "
                f"expected {len(self._labels.names)}"
            )

        scores = {
            name: float(score)
            for name, score in zip(self._labels.names, probabilities)
            if float(score) >= self.threshold
        }
        ordered_tags = [
            name
            for name, _score in sorted(
                scores.items(),
                key=lambda item: item[1],
                reverse=True,
            )
        ]

        return TagResult(
            tags=ordered_tags,
            scores=scores,
        )


def _is_fixed_dim(value: Any, expected: int) -> bool:
    try:
        return int(value) == expected
    except (TypeError, ValueError):
        return False
