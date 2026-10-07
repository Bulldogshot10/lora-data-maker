from __future__ import annotations

from pathlib import Path
from typing import Any

from .base import Tagger, TaggerError
from .llama_cpp import LlamaCppVLMTagger
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
    if backend == "llama_cpp":
        llama_config = config.get("llama_cpp", {})
        if not isinstance(llama_config, dict):
            raise TaggerError("config section [tagger.llama_cpp] must be a table")

        seed_value = llama_config.get("seed")
        return LlamaCppVLMTagger(
            base_url=str(llama_config.get("base_url", "http://127.0.0.1:8080/v1")),
            model=str(llama_config.get("model", "local-model")),
            timeout=float(llama_config.get("timeout", 120.0)),
            temperature=float(llama_config.get("temperature", 0.0)),
            max_tokens=int(llama_config.get("max_tokens", 256)),
            system_prompt=llama_config.get("system_prompt"),
            prompt=llama_config.get("prompt"),
            seed=int(seed_value) if seed_value is not None else None,
            max_tags=int(llama_config.get("max_tags", 64)),
        )

    if backend != "wd14":
        raise TaggerError(
            f"unsupported tagger backend {backend!r}; expected 'wd14' or 'llama_cpp'"
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
        "threshold": float(config.get("threshold", 0.35)),
        "character_threshold": float(
            config.get("character_threshold", 0.85)
        ),
    }

    if profile.kind == "mobile":
        return MobileWD14Tagger(**common)

    if profile.kind == "wd14":
        return WD14Tagger(
            **common,
            use_mcut=bool(config.get("use_mcut", False)),
            batch_size=int(config.get("batch_size", 1)),
        )

    raise TaggerError(f"unsupported tagger preset kind {profile.kind!r}")
