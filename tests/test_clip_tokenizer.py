import pytest

from lora_data_maker.clip_tokenizer import (
    CLIP_CONTEXT_LENGTH,
    CLIP_TOKENIZER_ID,
    ClipTokenizerError,
    count_clip_tokens,
    pack_clip_caption,
)
from lora_data_maker.exceptions import LoraDataMakerError


class FakeClipTokenizer:
    def __call__(self, text, **kwargs):
        value = str(text)
        return {"input_ids": list(range(len(value.split()) + 2))}


def test_count_clip_tokens_preserves_bos_and_eos():
    tokenizer = FakeClipTokenizer()
    assert count_clip_tokens("julia portrait", tokenizer=tokenizer) == 4


def test_pack_clip_caption_preserves_field_order_and_budget():
    tokenizer = FakeClipTokenizer()
    caption = pack_clip_caption(
        ["julia", "standing", "black dress", "smiling", "indoors soft lighting"],
        8,
        tokenizer=tokenizer,
    )
    assert caption == "julia, standing, black dress, smiling, indoors"
    assert count_clip_tokens(caption, tokenizer=tokenizer) == 8


def test_pack_clip_caption_uses_custom_separator():
    tokenizer = FakeClipTokenizer()
    caption = pack_clip_caption(
        ["julia", "standing", "black dress"],
        8,
        separator="; ",
        tokenizer=tokenizer,
    )
    assert caption == "julia; standing; black dress"


def test_pure_python_clip_tokenizer_fallback_works_without_tokenizers(
    tmp_path, monkeypatch
):
    import sys
    import types

    from lora_data_maker import clip_tokenizer

    byte_encoder = clip_tokenizer._bytes_to_unicode()
    vocab = {str(value): index for index, value in enumerate(byte_encoder.values())}
    vocab["a</w>"] = 300
    vocab["b</w>"] = 301
    vocab["c</w>"] = 302
    vocab["<|startoftext|>"] = 49406
    vocab["<|endoftext|>"] = 49407
    (tmp_path / "vocab.json").write_text(
        __import__("json").dumps(vocab),
        encoding="utf-8",
    )
    (tmp_path / "merges.txt").write_text("#version: 0.2\n", encoding="utf-8")

    monkeypatch.setitem(
        sys.modules,
        "transformers",
        types.SimpleNamespace(),
    )
    monkeypatch.setenv(
        clip_tokenizer.CLIP_TOKENIZER_PATH_ENV,
        str(tmp_path),
    )

    clip_tokenizer.load_clip_tokenizer.cache_clear()
    try:
        tokenizer = clip_tokenizer.load_clip_tokenizer()
        assert isinstance(
            tokenizer,
            clip_tokenizer.PurePythonCLIPTokenizer,
        )
        assert tokenizer.model_max_length == 77
        assert tokenizer.name_or_path == clip_tokenizer.CLIP_TOKENIZER_ID
        assert clip_tokenizer.count_clip_tokens("abc", tokenizer=tokenizer) == 5
    finally:
        clip_tokenizer.load_clip_tokenizer.cache_clear()



def test_clip_tokenizer_error_is_application_error():
    assert issubclass(ClipTokenizerError, LoraDataMakerError)


class FakeRealTokenizer:
    model_max_length = 77
    name_or_path = "openai/clip-vit-large-patch14"
    bos_token_id = 49406
    eos_token_id = 49407

    def __call__(self, text, **kwargs):
        return {"input_ids": [self.bos_token_id, 1, self.eos_token_id]}


def test_load_clip_tokenizer_prefers_local_hf_cache(monkeypatch):
    import sys
    import types

    calls = []

    class FakeCLIPTokenizer:
        @classmethod
        def from_pretrained(cls, source, **kwargs):
            calls.append((source, kwargs))
            if kwargs.get("local_files_only"):
                return FakeRealTokenizer()
            raise AssertionError("remote tokenizer load should not be attempted")

    monkeypatch.setitem(
        sys.modules,
        "transformers",
        types.SimpleNamespace(CLIPTokenizer=FakeCLIPTokenizer),
    )

    from lora_data_maker import clip_tokenizer

    clip_tokenizer.load_clip_tokenizer.cache_clear()
    try:
        tokenizer = clip_tokenizer.load_clip_tokenizer()
    finally:
        clip_tokenizer.load_clip_tokenizer.cache_clear()

    assert tokenizer.name_or_path == clip_tokenizer.CLIP_TOKENIZER_ID
    assert calls == [(clip_tokenizer.CLIP_TOKENIZER_ID, {"local_files_only": True})]


def test_load_clip_tokenizer_offline_fails_cleanly_when_cache_is_missing(
    monkeypatch,
):
    import sys
    import types

    class FakeCLIPTokenizer:
        @classmethod
        def from_pretrained(cls, source, **kwargs):
            raise OSError("not cached")

    monkeypatch.setitem(
        sys.modules,
        "transformers",
        types.SimpleNamespace(CLIPTokenizer=FakeCLIPTokenizer),
    )
    monkeypatch.setenv("LORA_DATA_MAKER_CLIP_TOKENIZER_OFFLINE", "1")

    from lora_data_maker import clip_tokenizer

    clip_tokenizer.load_clip_tokenizer.cache_clear()
    try:
        with pytest.raises(
            clip_tokenizer.ClipTokenizerError,
            match="local Hugging Face cache.*offline mode is enabled",
        ):
            clip_tokenizer.load_clip_tokenizer()
    finally:
        clip_tokenizer.load_clip_tokenizer.cache_clear()


def test_load_clip_tokenizer_normalizes_huge_length_sentinel(monkeypatch):
    import sys
    import types

    class FakeCLIPTokenizer:
        @classmethod
        def from_pretrained(cls, source, **kwargs):
            tokenizer = FakeRealTokenizer()
            tokenizer.model_max_length = 10**30
            return tokenizer

    monkeypatch.setitem(
        sys.modules,
        "transformers",
        types.SimpleNamespace(CLIPTokenizer=FakeCLIPTokenizer),
    )

    from lora_data_maker import clip_tokenizer

    clip_tokenizer.load_clip_tokenizer.cache_clear()
    try:
        tokenizer = clip_tokenizer.load_clip_tokenizer()
    finally:
        clip_tokenizer.load_clip_tokenizer.cache_clear()

    assert tokenizer.model_max_length == clip_tokenizer.CLIP_CONTEXT_LENGTH
