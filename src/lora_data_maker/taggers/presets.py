from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TaggerPreset:
    name: str
    kind: str
    repo_id: str
    revision: str
    model_filename: str
    labels_filename: str
    model_sha256: str
    default_threshold: float = 0.35
    default_character_threshold: float = 0.85
    default_threshold: float = 0.35
    default_character_threshold: float = 0.85


SWEETSPOT = TaggerPreset(
    name="sweetspot",
    kind="wd14",
    repo_id="SmilingWolf/wd-vit-tagger-v3",
    revision="790b0e92cefd2a0221451604e7831fe643ab7c4f",
    model_filename="model.onnx",
    labels_filename="selected_tags.csv",
    model_sha256=(
        "35f23693620b668f4d53fd3c62bf65e40af739bc52c7eb0fbc49258b58d065b6"
    ),
)

POTATO = TaggerPreset(
    name="potato",
    kind="mobile",
    repo_id="Smashinfries/wd-swinv2-tagger-v3-onnx-mobile",
    revision="7e2509c9dcc5e9c39970f8701dbea602a2b6e095",
    model_filename="wd-swinv2-tagger-v3-quint8-mobile.onnx",
    labels_filename="tags.json",
    model_sha256=(
        "dddaebcc030a614d16b6783e2d386a5686461ce809a0c73ce143d6ba231b7f67"
    ),
)

JOYTAG = TaggerPreset(
    name="joytag",
    kind="joytag",
    repo_id="fancyfeast/joytag",
    revision="6b7f16331a6ccf0fdce37d5a9564715f6e772b22",
    model_filename="model.onnx",
    labels_filename="top_tags.txt",
    model_sha256=(
        "f85b7130e6e549b5b0822537007b7482e8c4c8e754c8d9a5bee08e27050e1097"
    ),
    default_threshold=0.4,
)

TAGGER_PRESETS = {
    SWEETSPOT.name: SWEETSPOT,
    POTATO.name: POTATO,
    JOYTAG.name: JOYTAG,
    JOYTAG.name: JOYTAG,
}


def resolve_tagger_preset(name: str | None = None) -> TaggerPreset:
    key = str(name or SWEETSPOT.name).strip().lower()
    try:
        return TAGGER_PRESETS[key]
    except KeyError as exc:
        expected = ", ".join(TAGGER_PRESETS)
        raise ValueError(
            f"unknown tagger preset {name!r}; expected one of {expected}"
        ) from exc
