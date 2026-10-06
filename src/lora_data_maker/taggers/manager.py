from __future__ import annotations

from pathlib import Path
from typing import Any

from .base import Tagger, TaggerError
from .model import HuggingFaceModelCache
from .wd14 import WD14Tagger


def build_tagger(
    config: dict[str, Any],
    *,
    cache_root: Path | str | None = None,
) -> Tagger:
    backend = str(config.get("backend", "wd14")).strip().lower()
    if backend != "wd14":
        raise TaggerError(
            f"unsupported tagger backend {backend!r}; expected 'wd14'"
        )

    repo_id = str(
        config.get(
            "model",
            "SmilingWolf/wd-eva02-large-tagger-v3",
        )
    ).strip()
    revision = str(
        config.get(
            "revision",
            "b25b82a03f7282e41aa2f257a52c7583b710bd1c",
        )
    ).strip()

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
        repo_id=repo_id,
        revision=revision,
        timeout=float(config.get("download_timeout", 60.0)),
    )

    return WD14Tagger(
        model_cache=model_cache,
        model_sha256=str(config.get("model_sha256", "")).strip() or None,
        threshold=float(config.get("threshold", 0.35)),
        character_threshold=float(
            config.get("character_threshold", 0.85)
        ),
        use_mcut=bool(config.get("use_mcut", False)),
        batch_size=int(config.get("batch_size", 1)),
    )
