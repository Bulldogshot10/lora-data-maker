from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from PIL import Image

from .base import TagResult, TaggerError
from .model import HuggingFaceModelCache
from .wd14 import KAOMOJIS


@dataclass(frozen=True)
class MobileWD14Labels:
    names: list[str]
    rating_indexes: list[int]
    general_indexes: list[int]
    character_indexes: list[int]

    @classmethod
    def load(cls, path: Path | str) -> "MobileWD14Labels":
        try:
            with Path(path).open("r", encoding="utf-8") as handle:
                raw = json.load(handle)
        except (OSError, json.JSONDecodeError) as exc:
            raise TaggerError(f"could not read mobile WD14 labels: {path}") from exc

        if not isinstance(raw, dict) or not raw:
            raise TaggerError(
                "invalid mobile WD14 tags.json: expected a non-empty object"
            )

        entries: dict[int, tuple[str, int]] = {}
        for key, value in raw.items():
            try:
                index = int(key)
            except (TypeError, ValueError) as exc:
                raise TaggerError(
                    f"invalid mobile WD14 tag index: {key!r}"
                ) from exc

            if (
                not isinstance(value, list)
                or len(value) < 2
                or not isinstance(value[0], str)
            ):
                raise TaggerError(
                    f"invalid mobile WD14 tag entry at index {key!r}"
                )

            try:
                category = int(value[1])
            except (TypeError, ValueError) as exc:
                raise TaggerError(
                    f"invalid mobile WD14 category at index {key!r}"
                ) from exc

            if index < 0:
                raise TaggerError(f"invalid mobile WD14 tag index: {index}")

            entries[index] = (value[0], category)

        max_index = max(entries)
        if len(entries) != max_index + 1:
            raise TaggerError("mobile WD14 tags.json has missing tag indexes")

        names: list[str] = []
        rating_indexes: list[int] = []
        general_indexes: list[int] = []
        character_indexes: list[int] = []

        for index in range(max_index + 1):
            raw_name, category = entries[index]
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

        if not rating_indexes or not general_indexes:
            raise TaggerError(
                "mobile WD14 tags.json is missing required rating/general categories"
            )

        return cls(
            names=names,
            rating_indexes=rating_indexes,
            general_indexes=general_indexes,
            character_indexes=character_indexes,
        )


class MobileWD14Tagger:
    def __init__(
        self,
        *,
        model_cache: HuggingFaceModelCache,
        model_sha256: str | None = None,
        threshold: float = 0.35,
        character_threshold: float = 0.85,
        session_factory: Callable[[str], Any] | None = None,
    ) -> None:
        if not 0 <= threshold <= 1:
            raise ValueError("general threshold must be between 0 and 1")
        if not 0 <= character_threshold <= 1:
            raise ValueError("character threshold must be between 0 and 1")

        self.model_cache = model_cache
        self.model_sha256 = model_sha256
        self.threshold = threshold
        self.character_threshold = character_threshold
        self.session_factory = session_factory

        self._session: Any | None = None
        self._labels: MobileWD14Labels | None = None

    def _load(self) -> None:
        if self._session is not None and self._labels is not None:
            return

        try:
            import numpy as np
        except ImportError as exc:
            raise TaggerError(
                "mobile WD14 tagger requires numpy; install the 'tagger' extra"
            ) from exc

        try:
            if self.session_factory is None:
                import onnxruntime as ort
                session_factory = ort.InferenceSession
            else:
                session_factory = self.session_factory
        except ImportError as exc:
            raise TaggerError(
                "mobile WD14 tagger requires optional onnxruntime; "
                "install the 'tagger' extra"
            ) from exc

        files = self.model_cache.ensure(model_sha256=self.model_sha256)
        session = session_factory(str(files.model_path))
        inputs = session.get_inputs()
        if not inputs:
            raise TaggerError("mobile WD14 ONNX model has no inputs")

        shape = inputs[0].shape
        if len(shape) != 4:
            raise TaggerError(
                f"unexpected mobile WD14 input shape: {shape}"
            )

        if shape[3] not in (3, "3", None):
            raise TaggerError(
                f"unexpected mobile WD14 channel count: {shape[3]}"
            )

        input_type = str(getattr(inputs[0], "type", "")).lower()
        if input_type and "uint8" not in input_type:
            raise TaggerError(
                f"unexpected mobile WD14 input type: {input_type}"
            )

        self._session = session
        self._labels = MobileWD14Labels.load(files.labels_path)

    def tag_image(self, path: Path | str) -> TagResult:
        return self.tag_images([path])[0]

    def tag_images(self, paths: list[Path | str]) -> list[TagResult]:
        if not paths:
            return []

        self._load()
        results: list[TagResult] = []

        for path in paths:
            array = self._prepare_image(path)
            input_name = self._session.get_inputs()[0].name
            output_name = self._session.get_outputs()[0].name
            predictions = self._session.run(
                [output_name],
                {input_name: array},
            )[0]
            results.append(self._decode_prediction(predictions))

        return results

    def _prepare_image(self, path: Path | str):
        import numpy as np

        try:
            with Image.open(Path(path).expanduser().resolve()) as image:
                rgba = image.convert("RGBA")
                canvas = Image.new(
                    "RGBA",
                    rgba.size,
                    (255, 255, 255, 255),
                )
                canvas.alpha_composite(rgba)
                rgb = canvas.convert("RGB")
                array = np.asarray(rgb, dtype=np.uint8)
                return np.expand_dims(array, axis=0)
        except (OSError, ValueError) as exc:
            raise TaggerError(
                f"could not prepare image for mobile WD14: {path}"
            ) from exc

    def _decode_prediction(self, prediction) -> TagResult:
        if self._labels is None:
            raise TaggerError("mobile WD14 labels are not loaded")

        import numpy as np

        probabilities = np.asarray(prediction, dtype=np.float32)
        if probabilities.ndim == 2:
            if probabilities.shape[0] != 1:
                raise TaggerError(
                    "mobile WD14 returned unexpected batch shape: "
                    f"{probabilities.shape}"
                )
            probabilities = probabilities[0]
        else:
            probabilities = probabilities.reshape(-1)

        if len(probabilities) != len(self._labels.names):
            raise TaggerError(
                f"mobile WD14 output has {len(probabilities)} labels; "
                f"expected {len(self._labels.names)}"
            )

        ratings = {
            self._labels.names[index]: float(probabilities[index])
            for index in self._labels.rating_indexes
        }
        characters = {
            self._labels.names[index]: float(probabilities[index])
            for index in self._labels.character_indexes
            if float(probabilities[index]) >= self.character_threshold
        }
        tags = {
            self._labels.names[index]: float(probabilities[index])
            for index in self._labels.general_indexes
            if float(probabilities[index]) >= self.threshold
        }
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
