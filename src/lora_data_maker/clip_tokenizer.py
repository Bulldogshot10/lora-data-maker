from __future__ import annotations

import json
import os
import re
import tempfile
import unicodedata
from functools import lru_cache
from pathlib import Path
from typing import Any

import requests

from .exceptions import LoraDataMakerError

CLIP_TOKENIZER_ID = "openai/clip-vit-large-patch14"
CLIP_CONTEXT_LENGTH = 77
CLIP_TOKENIZER_PATH_ENV = "LORA_DATA_MAKER_CLIP_TOKENIZER_PATH"
CLIP_TOKENIZER_OFFLINE_ENV = "LORA_DATA_MAKER_CLIP_TOKENIZER_OFFLINE"
_CLIP_CACHE_DIR = "lora-data-maker/clip-vit-large-patch14"
_CLIP_VOCAB_URL = (
    "https://huggingface.co/openai/clip-vit-large-patch14/resolve/main/vocab.json"
)
_CLIP_MERGES_URL = (
    "https://huggingface.co/openai/clip-vit-large-patch14/resolve/main/merges.txt"
)


class ClipTokenizerError(LoraDataMakerError):
    """Raised when the exact SD1.x CLIP tokenizer is unavailable."""


@lru_cache(maxsize=1)
def _bytes_to_unicode() -> dict[int, str]:
    """Return the byte-to-unicode mapping used by CLIP's byte-level BPE."""
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    extra = 0
    for value in range(256):
        if value not in bs:
            bs.append(value)
            cs.append(256 + extra)
            extra += 1
    return {byte: chr(codepoint) for byte, codepoint in zip(bs, cs, strict=True)}


def _unicode_letter(value: str) -> bool:
    return unicodedata.category(value).startswith("L")


def _unicode_number(value: str) -> bool:
    return unicodedata.category(value).startswith("N")


def _split_clip_text(text: str) -> list[str]:
    """Match the CLIP regex using only the Python standard library."""
    value = unicodedata.normalize("NFC", str(text))
    value = re.sub(r"\s+", " ", value).strip().lower()
    if not value:
        return []

    contractions = ("'s", "'t", "'re", "'ve", "'m", "'ll", "'d")
    specials = ("<|startoftext|>", "<|endoftext|>")
    pieces: list[str] = []
    index = 0

    while index < len(value):
        if value[index].isspace():
            index += 1
            continue

        matched_special = None
        for special in specials:
            if value.startswith(special, index):
                matched_special = special
                break
        if matched_special is not None:
            pieces.append(matched_special)
            index += len(matched_special)
            continue

        matched_contraction = None
        for contraction in contractions:
            if value.startswith(contraction, index):
                matched_contraction = contraction
                break
        if matched_contraction is not None:
            pieces.append(matched_contraction)
            index += len(matched_contraction)
            continue

        char = value[index]
        if _unicode_letter(char):
            end = index + 1
            while end < len(value) and _unicode_letter(value[end]):
                end += 1
        elif _unicode_number(char):
            end = index + 1
            while end < len(value) and _unicode_number(value[end]):
                end += 1
        else:
            end = index + 1
            while (
                end < len(value)
                and not value[end].isspace()
                and not _unicode_letter(value[end])
                and not _unicode_number(value[end])
            ):
                end += 1

        pieces.append(value[index:end])
        index = end

    return pieces


def _get_pairs(word: tuple[str, ...]) -> set[tuple[str, str]]:
    if len(word) < 2:
        return set()
    return set(zip(word, word[1:], strict=False))


class PurePythonCLIPTokenizer:
    """Small dependency-free CLIP BPE tokenizer for Termux/ABI-constrained builds."""

    model_max_length = CLIP_CONTEXT_LENGTH
    name_or_path = CLIP_TOKENIZER_ID

    def __init__(self, vocab: dict[str, int], merges: list[tuple[str, str]]) -> None:
        self.encoder = {str(token): int(token_id) for token, token_id in vocab.items()}
        self.bpe_ranks = {merge: rank for rank, merge in enumerate(merges)}
        self.byte_encoder = _bytes_to_unicode()
        self.bos_token_id = self.encoder.get("<|startoftext|>", 49406)
        self.eos_token_id = self.encoder.get("<|endoftext|>", 49407)
        self.pad_token_id = self.eos_token_id
        self._cache: dict[str, str] = {}

        if "<|startoftext|>" not in self.encoder or "<|endoftext|>" not in self.encoder:
            raise ClipTokenizerError(
                f"{CLIP_TOKENIZER_ID} tokenizer assets are missing CLIP special tokens"
            )

    def _bpe(self, token: str) -> str:
        cached = self._cache.get(token)
        if cached is not None:
            return cached

        word = tuple(token[:-1]) + (token[-1] + "</w>",)
        pairs = _get_pairs(word)
        if not pairs:
            result = token + "</w>"
            self._cache[token] = result
            return result

        while True:
            bigram = min(
                pairs,
                key=lambda pair: self.bpe_ranks.get(pair, float("inf")),
            )
            if bigram not in self.bpe_ranks:
                break

            first, second = bigram
            new_word: list[str] = []
            index = 0
            while index < len(word):
                try:
                    next_index = word.index(first, index)
                    new_word.extend(word[index:next_index])
                    index = next_index
                except ValueError:
                    new_word.extend(word[index:])
                    break

                if (
                    index < len(word) - 1
                    and word[index] == first
                    and word[index + 1] == second
                ):
                    new_word.append(first + second)
                    index += 2
                else:
                    new_word.append(word[index])
                    index += 1

            word = tuple(new_word)
            if len(word) == 1:
                break
            pairs = _get_pairs(word)

        result = " ".join(word)
        self._cache[token] = result
        return result

    def encode(self, text: str) -> list[int]:
        byte_encoder = self.byte_encoder
        tokens: list[int] = []

        for piece in _split_clip_text(text):
            if piece in {"<|startoftext|>", "<|endoftext|>"}:
                tokens.append(self.encoder[piece])
                continue

            encoded = "".join(byte_encoder[value] for value in piece.encode("utf-8"))
            for token in self._bpe(encoded).split(" "):
                try:
                    tokens.append(self.encoder[token])
                except KeyError as exc:
                    raise ClipTokenizerError(
                        f"CLIP vocabulary is missing BPE token {token!r}"
                    ) from exc

        return tokens

    def __call__(self, text: str, **_: Any) -> dict[str, list[int]]:
        value = str(text)
        input_ids = [self.bos_token_id]
        input_ids.extend(self.encode(value))
        input_ids.append(self.eos_token_id)
        return {"input_ids": input_ids}


def _tokenizer_cache_dir() -> Path:
    hf_home = os.environ.get("HF_HOME", "").strip()
    if hf_home:
        return Path(hf_home).expanduser() / "lora-data-maker" / "clip-vit-large-patch14"

    xdg_cache = os.environ.get("XDG_CACHE_HOME", "").strip()
    base = Path(xdg_cache).expanduser() if xdg_cache else Path.home() / ".cache"
    return base / _CLIP_CACHE_DIR


def _find_existing_hf_snapshot() -> Path | None:
    hf_home = os.environ.get("HF_HOME", "").strip()
    if hf_home:
        hub_root = Path(hf_home).expanduser() / "hub"
    else:
        xdg_cache = os.environ.get("XDG_CACHE_HOME", "").strip()
        base = Path(xdg_cache).expanduser() if xdg_cache else Path.home() / ".cache"
        hub_root = base / "huggingface" / "hub"

    snapshots = hub_root / "models--openai--clip-vit-large-patch14" / "snapshots"
    if not snapshots.is_dir():
        return None

    candidates = [
        path
        for path in snapshots.iterdir()
        if path.is_dir()
        and (path / "vocab.json").is_file()
        and (path / "merges.txt").is_file()
    ]
    return max(candidates, key=lambda path: path.stat().st_mtime) if candidates else None


def _read_pure_assets(root: Path) -> PurePythonCLIPTokenizer:
    vocab_path = root / "vocab.json"
    merges_path = root / "merges.txt"
    if not vocab_path.is_file() or not merges_path.is_file():
        raise ClipTokenizerError(
            f"CLIP tokenizer assets require vocab.json and merges.txt under {root}"
        )

    try:
        vocab = json.loads(vocab_path.read_text(encoding="utf-8"))
        merge_lines = merges_path.read_text(encoding="utf-8").splitlines()
    except (OSError, json.JSONDecodeError) as exc:
        raise ClipTokenizerError(
            f"could not read CLIP tokenizer assets from {root}: {exc}"
        ) from exc

    if not isinstance(vocab, dict):
        raise ClipTokenizerError(f"invalid CLIP vocabulary in {vocab_path}")

    merges = [
        tuple(line.split())
        for line in merge_lines
        if line.strip() and not line.startswith("#")
    ]
    if any(len(merge) != 2 for merge in merges):
        raise ClipTokenizerError(f"invalid CLIP merges in {merges_path}")

    return PurePythonCLIPTokenizer(vocab, merges)


def _download_text_asset(url: str, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(
        prefix=f".{target.name}.",
        suffix=".part",
        dir=target.parent,
    )
    try:
        with os.fdopen(fd, "wb") as handle:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    handle.write(chunk)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, target)
    except Exception:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass
        raise


def _load_pure_python_tokenizer() -> PurePythonCLIPTokenizer:
    local_path = os.environ.get(CLIP_TOKENIZER_PATH_ENV, "").strip()
    offline = os.environ.get(CLIP_TOKENIZER_OFFLINE_ENV, "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }

    if local_path:
        root = Path(local_path).expanduser().resolve()
    else:
        root = _find_existing_hf_snapshot() or _tokenizer_cache_dir()

    try:
        tokenizer = _read_pure_assets(root)
    except ClipTokenizerError as local_exc:
        if offline:
            raise ClipTokenizerError(
                f"CLIP tokenizer {CLIP_TOKENIZER_ID} is not available locally and "
                "offline mode is enabled. Provide LORA_DATA_MAKER_CLIP_TOKENIZER_PATH "
                "with vocab.json and merges.txt, or pre-cache the tokenizer."
            ) from local_exc

        if local_path or _find_existing_hf_snapshot() is not None:
            raise

        try:
            _download_text_asset(_CLIP_VOCAB_URL, root / "vocab.json")
            _download_text_asset(_CLIP_MERGES_URL, root / "merges.txt")
            tokenizer = _read_pure_assets(root)
        except Exception as exc:
            raise ClipTokenizerError(
                f"could not download CLIP tokenizer assets for {CLIP_TOKENIZER_ID}: {exc}"
            ) from exc

    return tokenizer


@lru_cache(maxsize=1)
def load_clip_tokenizer() -> Any:
    """Load the exact SD1.x CLIP tokenizer, with a pure-Python Termux fallback."""
    try:
        from transformers import CLIPTokenizer
    except Exception:
        # Transformers v5 imports the Rust-backed tokenizers package for CLIP.
        # Termux/Python builds can have an ABI-incompatible tokenizers .so. The
        # CLIP BPE itself does not require Transformers or PyTorch, so fall back
        # to the same vocab/merges assets with a dependency-free implementation.
        return _load_pure_python_tokenizer()

    local_path = os.environ.get(CLIP_TOKENIZER_PATH_ENV, "").strip()
    offline = os.environ.get(CLIP_TOKENIZER_OFFLINE_ENV, "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }

    if local_path:
        source: str | Path = Path(local_path).expanduser().resolve()
        try:
            tokenizer = CLIPTokenizer.from_pretrained(
                source,
                local_files_only=True,
            )
        except Exception as exc:
            raise ClipTokenizerError(
                f"could not load local CLIP tokenizer from {source}: {exc}"
            ) from exc
    else:
        try:
            tokenizer = CLIPTokenizer.from_pretrained(
                CLIP_TOKENIZER_ID,
                local_files_only=True,
            )
        except Exception as local_exc:
            if offline:
                raise ClipTokenizerError(
                    f"CLIP tokenizer {CLIP_TOKENIZER_ID} is not available in the "
                    "local Hugging Face cache and offline mode is enabled. "
                    f"Unset {CLIP_TOKENIZER_OFFLINE_ENV} or pre-cache the tokenizer."
                ) from local_exc
            try:
                tokenizer = CLIPTokenizer.from_pretrained(CLIP_TOKENIZER_ID)
            except Exception as remote_exc:
                raise ClipTokenizerError(
                    f"could not load CLIP tokenizer {CLIP_TOKENIZER_ID} from local "
                    f"cache or Hugging Face: {remote_exc}"
                ) from remote_exc

    model_max_length = int(
        getattr(tokenizer, "model_max_length", CLIP_CONTEXT_LENGTH)
    )
    if model_max_length != CLIP_CONTEXT_LENGTH:
        if model_max_length >= 1_000_000:
            tokenizer.model_max_length = CLIP_CONTEXT_LENGTH
        else:
            raise ClipTokenizerError(
                f"{CLIP_TOKENIZER_ID} reported an unexpected context length "
                f"{model_max_length}; expected {CLIP_CONTEXT_LENGTH}"
            )

    return tokenizer


def count_clip_tokens(text: str, *, tokenizer: Any | None = None) -> int:
    """Count BOS/EOS-inclusive CLIP tokens without tokenizer-side truncation."""
    value = str(text).strip()
    if not value:
        return 0

    active = tokenizer if tokenizer is not None else load_clip_tokenizer()
    try:
        encoded = active(
            value,
            add_special_tokens=True,
            truncation=False,
            padding=False,
        )
        input_ids = encoded["input_ids"]
    except Exception as exc:
        raise ClipTokenizerError(
            f"could not tokenize caption text: {exc}"
        ) from exc

    if input_ids and isinstance(input_ids[0], list):
        input_ids = input_ids[0]

    return len(input_ids)


def pack_clip_caption(
    parts: list[str],
    max_tokens: int,
    *,
    separator: str = ", ",
    tokenizer: Any | None = None,
) -> str:
    """Pack ordered caption fields while keeping the exact CLIP budget."""
    if not 1 <= max_tokens <= CLIP_CONTEXT_LENGTH:
        raise ValueError(
            f"caption max tokens must be between 1 and {CLIP_CONTEXT_LENGTH}"
        )
    if not isinstance(separator, str):
        raise ValueError("caption separator must be a string")

    normalized = [str(part).strip(" ,.;:") for part in parts if str(part).strip()]
    if not normalized:
        return ""

    selected: list[str] = []
    for part in normalized:
        candidate = separator.join((*selected, part))
        if count_clip_tokens(candidate, tokenizer=tokenizer) <= max_tokens:
            selected.append(part)
            continue

        words = part.split()
        partial: list[str] = []
        for word in words:
            candidate_part = " ".join((*partial, word))
            candidate = separator.join((*selected, candidate_part))
            if count_clip_tokens(candidate, tokenizer=tokenizer) <= max_tokens:
                partial.append(word)
            else:
                break

        if partial:
            selected.append(" ".join(partial))
        break

    return separator.join(selected).strip(" ,.;:")
