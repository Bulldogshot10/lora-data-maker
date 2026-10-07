from __future__ import annotations

from pathlib import Path
from typing import Any

from .base import Tagger, TaggerError
from .joytag import JoyTagTagger
from .mobile import MobileWD14Tagger
from .model import HuggingFaceModelCache
from .presets import resolve_tagger_preset
from .wd14 import WD14Tagger


def build_tagger(
    config: dict[str, Any],
    *,
    cache_root: Path | str | None = None,
    preset: str | None = None,
) -> Tagger:
    backend = str(config.get("backend", "wd14")).strip().lower()
    if backend not in {"wd14", "joytag"}:
        raise TaggerError(
            f"unsupported tagger backend {backend!r}; expected 'wd14' or 'joytag'"
        )

    profile = resolve_tagger_preset(
        preset if preset is not None else config.get("preset")
    )

    configured_cache = config.get("cache_dir")
    if configured_cache:
        cache_dir = Path(str(configured_cache)).expanduser()
    elif cache_root is not None:
        cache_dir = Path(cache_root).expanduser()
    else:
        cache_dir = (
            Path.home()
            / ".cache"
            / "lora-data-maker"
            / "taggers"
            / "wd14"
        )

    model_cache = HuggingFaceModelCache(
        cache_dir,
        repo_id=profile.repo_id,
        revision=profile.revision,
        timeout=float(config.get("download_timeout", 60.0)),
        model_filename=profile.model_filename,
        labels_filename=profile.labels_filename,
    )

    common = {
        "model_cache": model_cache,
        "model_sha256": profile.model_sha256,
        "threshold": float(
            config.get("threshold", profile.default_threshold)
        ),
        "character_threshold": float(
            config.get(
                "character_threshold",
                profile.default_character_threshold,
            )
        ),
    }

    if profile.kind == "joytag":
        return JoyTagTagger(
            **common,
            batch_size=int(config.get("batch_size", 1)),
        )

    if profile.kind == "mobile":
        return MobileWD14Tagger(**common)

    if profile.kind == "wd14":
        return WD14Tagger(
            **common,
            use_mcut=bool(config.get("use_mcut", False)),
            batch_size=int(config.get("batch_size", 1)),
        )

    raise TaggerError(f"unsupported tagger preset kind {profile.kind!r}")
