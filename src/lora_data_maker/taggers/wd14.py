from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

from PIL import Image

from .base import TagResult, TaggerError
from .model import HuggingFaceModelCache


KAOMOJIS = {
    "0_0",
    "(o)_(o)",
    "+_+",
    "+_-",
    "._.",
    "<o>_<o>",
    "<|>_<|>",
    "=_=",
    ">_<",
    "3_3",
    "6_9",
    ">_o",
    "@_@",
    "^_^",
    "o_o",
    "u_u",
    "x_x",
    "|_|",
    "||_||",
}


@dataclass(frozen=True)
class WD14Labels:
    names: list[str]
    rating_indexes: list[int]
    general_indexes: list[int]
    character_indexes: list[int]

    @classmethod
    def load(cls, path: Path | str) -> "WD14Labels":
        names: list[str] = []
        rating_indexes: list[int] = []
        general_indexes: list[int] = []
        character_indexes: list[int] = []

        try:
            with Path(path).open(
                "r",
                encoding="utf-8",
                newline="",
            ) as handle:
                reader = csv.DictReader(handle)
                if reader.fieldnames is None or not {
                    "name",
                    "category",
                }.issubset(reader.fieldnames):
                    raise TaggerError(
                        "invalid WD14 selected_tags.csv: "
                        "expected name/category columns"
                    )

                for index, row in enumerate(reader):
                    raw_name = (row.get("name") or "").strip()
                    if not raw_name:
                        raise TaggerError(
                            "invalid WD14 selected_tags.csv row "
                            f"{index + 2}: empty tag name"
                        )

                    try:
                        category = int(row.get("category") or -1)
                    except ValueError as exc:
                        raise TaggerError(
                            f"invalid WD14 category at row {index + 2}"
                        ) from exc

                    name = (
                        raw_name
                        if raw_name in KAOMOJIS
                        else raw_name.replace("_", " ")
                    )
                    names.append(name)

                    if category == 9:
                        rating_indexes.append(index)
                    elif category == 0:
                        general_indexes.append(index)
                    elif category == 4:
                        character_indexes.append(index)
        except OSError as exc:
            raise TaggerError(
                f"could not read WD14 labels: {path}"
            ) from exc

        if not names or not rating_indexes or not general_indexes:
            raise TaggerError(
                "WD14 labels are missing required rating/general categories"
            )

        return cls(
            names=names,
            rating_indexes=rating_indexes,
            general_indexes=general_indexes,
            character_indexes=character_indexes,
        )


def mcut_threshold(probabilities: Sequence[float]) -> float:
    if len(probabilities) < 2:
        return 0.0

    ordered = sorted(
        (float(value) for value in probabilities),
        reverse=True,
    )
    differences = [
        ordered[index] - ordered[index + 1]
        for index in range(len(ordered) - 1)
    ]
    index = max(range(len(differences)), key=differences.__getitem__)
    return (ordered[index] + ordered[index + 1]) / 2


class WD14Tagger:
    def __init__(
        self,
        *,
        model_cache: HuggingFaceModelCache,
        threshold: float = 0.35,
        character_threshold: float = 0.85,
        use_mcut: bool = False,
        character_mcut_floor: float = 0.15,
        batch_size: int = 1,
        session_factory: Callable[[str], Any] | None = None,
    ) -> None:
        if not 0 <= threshold <= 1:
            raise ValueError("general threshold must be between 0 and 1")
        if not 0 <= character_threshold <= 1:
            raise ValueError("character threshold must be between 0 and 1")
        if character_mcut_floor < 0:
            raise ValueError("character_mcut_floor cannot be negative")
        if batch_size < 1:
            raise ValueError("batch_size must be greater than zero")

        self.model_cache = model_cache
        self.threshold = threshold
        self.character_threshold = character_threshold
        self.use_mcut = use_mcut
        self.character_mcut_floor = character_mcut_floor
        self.batch_size = batch_size
        self.session_factory = session_factory

        self._session: Any | None = None
        self._labels: WD14Labels | None = None
        self._target_size: int | None = None

    def _load(self) -> None:
        if self._session is not None and self._labels is not None:
            return

        try:
            import numpy as np
        except ImportError as exc:
            raise TaggerError(
                "WD14 tagger requires the optional numpy dependency; "
                "install the 'tagger' extra"
            ) from exc

        try:
            if self.session_factory is None:
                import onnxruntime as ort
                session_factory = ort.InferenceSession
            else:
                session_factory = self.session_factory
        except ImportError as exc:
            raise TaggerError(
                "WD14 tagger requires optional onnxruntime; "
                "install the 'tagger' extra"
            ) from exc

        files = self.model_cache.ensure()
        session = session_factory(str(files.model_path))
        inputs = session.get_inputs()
        if not inputs:
            raise TaggerError("WD14 ONNX model has no inputs")

        shape = inputs[0].shape
        if len(shape) != 4:
            raise TaggerError(
                f"unexpected WD14 ONNX input shape: {shape}"
            )

        target_size = shape[1]
        channels = shape[3]
        if not isinstance(target_size, int) or target_size < 1:
            raise TaggerError(
                f"dynamic/invalid WD14 target height: {target_size}"
            )
        if channels != 3:
            raise TaggerError(
                f"unexpected WD14 channel count: {channels}"
            )

        self._session = session
        self._labels = WD14Labels.load(files.labels_path)
        self._target_size = target_size

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
            output_name = self._session.get_outputs()[0].name
            input_name = self._session.get_inputs()[0].name
            predictions = self._session.run(
                [output_name],
                {input_name: batch},
            )[0]

            if len(predictions) != len(batch_paths):
                raise TaggerError(
                    f"WD14 returned {len(predictions)} predictions for "
                    f"{len(batch_paths)} images"
                )

            for prediction in predictions:
                results.append(self._decode_prediction(prediction))

        return results

    def _prepare_batch(self, paths: list[Path | str]):
        import numpy as np

        if self._target_size is None:
            raise TaggerError("WD14 model target size is not loaded")

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
                    pad_left = (max_dim - rgb.width) // 2
                    pad_top = (max_dim - rgb.height) // 2
                    padded = Image.new(
                        "RGB",
                        (max_dim, max_dim),
                        (255, 255, 255),
                    )
                    padded.paste(rgb, (pad_left, pad_top))

                    if max_dim != self._target_size:
                        padded = padded.resize(
                            (self._target_size, self._target_size),
                            Image.Resampling.BICUBIC,
                        )

                    array = np.asarray(
                        padded,
                        dtype=np.float32,
                    )[:, :, ::-1]
                    arrays.append(array)
            except (OSError, ValueError) as exc:
                raise TaggerError(
                    f"could not prepare image for WD14: {path}"
                ) from exc

        return np.stack(arrays, axis=0)

    def _decode_prediction(self, prediction) -> TagResult:
        if self._labels is None:
            raise TaggerError("WD14 labels are not loaded")

        if len(prediction) != len(self._labels.names):
            raise TaggerError(
                f"WD14 output has {len(prediction)} labels; "
                f"expected {len(self._labels.names)}"
            )

        def selected(
            indexes: list[int],
            threshold: float,
        ) -> dict[str, float]:
            return {
                self._labels.names[index]: float(prediction[index])
                for index in indexes
                if float(prediction[index]) >= threshold
            }

        ratings = {
            self._labels.names[index]: float(prediction[index])
            for index in self._labels.rating_indexes
        }

        general_threshold = self.threshold
        general_scores = [
            float(prediction[index])
            for index in self._labels.general_indexes
        ]
        if self.use_mcut:
            general_threshold = max(
                self.threshold,
                mcut_threshold(general_scores),
            )

        characters = selected(
            self._labels.character_indexes,
            self.character_threshold,
        )
        tags = selected(
            self._labels.general_indexes,
            general_threshold,
        )
        tags.update(characters)

        ordered_tags = [
            name
            for name, _score in sorted(
                tags.items(),
                key=lambda item: item[1],
                reverse=True,
            )
        ]

        return TagResult(
            tags=ordered_tags,
            scores=dict(tags),
            ratings=ratings,
            characters=characters,
        )
