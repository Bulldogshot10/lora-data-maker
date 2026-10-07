from pathlib import Path

import pytest

from lora_data_maker.captioners import LlamaCppCaptioner
from lora_data_maker.captions import CaptionConfig, build_captioner, caption_manifest
from lora_data_maker.llama_cpp import LlamaCppError, LlamaCppRuntime
from lora_data_maker.models import ImageRecord, ProjectManifest
from lora_data_maker.taggers import LlamaCppVLMTagger, TaggerError


class FakeRuntime:
    def __init__(self, response):
        self.response = response
        self.calls = []

        self.model_path = Path("/models/vision.gguf")
        self.mmproj_path = Path("/models/mmproj.gguf")

    def chat_with_image(self, path, prompt, **kwargs):
        self.calls.append((Path(path), prompt, kwargs))
        return self.response


def test_runtime_rejects_missing_model(tmp_path):
    with pytest.raises(LlamaCppError, match="model not found"):
        LlamaCppRuntime(model_path=tmp_path / "missing.gguf")


def test_runtime_rejects_missing_mmproj(tmp_path):
    model = tmp_path / "model.gguf"
    model.write_bytes(b"model")

    with pytest.raises(LlamaCppError, match="mmproj not found"):
        LlamaCppRuntime(
            model_path=model,
            mmproj_path=tmp_path / "missing-mmproj.gguf",
        )


def test_llama_cpp_tagger_parses_json_and_normalizes_tags(tmp_path):
    image = tmp_path / "image.jpg"
    image.write_bytes(b"fake")

    runtime = FakeRuntime(
        '{"tags":["blue_eyes","blue eyes","portrait"],'
        '"ratings":{"safe":0.99},"characters":{"someone":0.91}}'
    )
    tagger = LlamaCppVLMTagger(runtime=runtime, seed=1)

    result = tagger.tag_image(image)

    assert result.tags == ["blue eyes", "portrait"]
    assert result.ratings == {"safe": 0.99}
    assert result.characters == {"someone": 0.91}
    assert runtime.calls[0][2]["response_format"] == {"type": "json_object"}
    assert runtime.calls[0][0] == image


def test_llama_cpp_tagger_rejects_invalid_json(tmp_path):
    image = tmp_path / "image.jpg"
    image.write_bytes(b"fake")

    tagger = LlamaCppVLMTagger(
        runtime=FakeRuntime("not-json"),
    )

    with pytest.raises(TaggerError, match="invalid JSON"):
        tagger.tag_image(image)


def test_llama_cpp_tagger_can_share_runtime(tmp_path):
    image = tmp_path / "image.jpg"
    image.write_bytes(b"fake")

    runtime = FakeRuntime('{"tags":["portrait"]}')
    tagger = LlamaCppVLMTagger(runtime=runtime)

    assert tagger.runtime is runtime


def test_llama_cpp_captioner_adds_trigger_prefix(tmp_path):
    image = tmp_path / "image.jpg"
    image.write_bytes(b"fake")

    runtime = FakeRuntime("blonde hair, white top, portrait.")

    captioner = LlamaCppCaptioner(
        runtime=runtime,
    )
    result = captioner.caption_image(
        image,
        tags=["blonde hair", "portrait"],
        subject="julia",
        trigger="julia",
        separator=", ",
        trigger_from_subject=True,
    )

    assert result == "julia, blonde hair, white top, portrait"
    assert runtime.calls[0][0] == image
    assert "Existing tag hints: blonde hair, portrait" in runtime.calls[0][1]


def test_build_tagger_supports_llama_cpp_backend(tmp_path):
    from lora_data_maker.taggers import build_tagger

    model = tmp_path / "vision.gguf"
    mmproj = tmp_path / "mmproj.gguf"
    model.write_bytes(b"model")
    mmproj.write_bytes(b"mmproj")

    tagger = build_tagger(
        {
            "backend": "llama_cpp",
            "llama_cpp": {
                "model_path": str(model),
                "mmproj_path": str(mmproj),
            },
        }
    )

    assert isinstance(tagger, LlamaCppVLMTagger)
    assert tagger.runtime.model_path == model.resolve()
    assert tagger.runtime.mmproj_path == mmproj.resolve()


def test_build_captioner_supports_llama_cpp_backend(tmp_path):
    model = tmp_path / "caption.gguf"
    mmproj = tmp_path / "mmproj.gguf"
    model.write_bytes(b"model")
    mmproj.write_bytes(b"mmproj")

    captioner = build_captioner(
        {
            "backend": "llama_cpp",
            "llama_cpp": {
                "model_path": str(model),
                "mmproj_path": str(mmproj),
            },
        }
    )

    assert isinstance(captioner, LlamaCppCaptioner)
    assert captioner.runtime.model_path == model.resolve()


def test_caption_manifest_uses_llama_cpp_captioner(tmp_path):
    processed = tmp_path / "work" / "processed"
    processed.mkdir(parents=True)
    image_path = processed / "image.jpg"
    image_path.write_bytes(b"fake")

    manifest = ProjectManifest(
        project_name="demo",
        subject="julia",
        trigger="julia",
    )
    image = ImageRecord(
        path="raw/image.jpg",
        processed_path="work/processed/image.jpg",
        status="tagged",
        tags=["portrait"],
    )
    manifest.add_image(image)

    runtime = FakeRuntime("portrait, blonde hair.")
    captioner = LlamaCppCaptioner(runtime=runtime)
    summary = caption_manifest(
        manifest,
        root=tmp_path,
        output_dir=tmp_path / "work" / "captions",
        config=CaptionConfig(),
        captioner=captioner,
    )

    assert summary.captioned == 1
    assert image.status == "captioned"
    assert (
        (tmp_path / image.caption_path).read_text(encoding="utf-8")
        == "julia, portrait, blonde hair\n"
    )
    assert runtime.calls[0][0] == image_path


def test_runtime_import_is_lazy(tmp_path, monkeypatch):
    model = tmp_path / "model.gguf"
    model.write_bytes(b"model")
    runtime = LlamaCppRuntime(model_path=model)

    monkeypatch.setitem(__import__("sys").modules, "llama_cpp", None)

    with pytest.raises(LlamaCppError, match="requires llama-cpp-python"):
        runtime.load()
