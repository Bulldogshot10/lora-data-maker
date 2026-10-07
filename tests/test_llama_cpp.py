from pathlib import Path

import pytest

from lora_data_maker.captioners import LlamaCppCaptioner
from lora_data_maker.captions import CaptionConfig, build_captioner, caption_manifest
from lora_data_maker.llama_cpp import LlamaCppClient, LlamaCppError
from lora_data_maker.models import ImageRecord, ProjectManifest
from lora_data_maker.taggers import LlamaCppVLMTagger, TaggerError


class FakeResponse:
    def __init__(self, payload, *, status=200):
        self.payload = payload
        self.status = status

    @property
    def ok(self):
        return self.status < 400

    def raise_for_status(self):
        if self.status >= 400:
            import requests
            raise requests.HTTPError(f"HTTP {self.status}")

    def json(self):
        return self.payload


class FakeSession:
    def __init__(self, content):
        self.content = content
        self.calls = []

    def post(self, url, *, json, timeout):
        self.calls.append((url, json, timeout))
        return FakeResponse(
            {
                "choices": [
                    {
                        "message": {
                            "content": self.content,
                        }
                    }
                ]
            }
        )

    def get(self, url, *, timeout):
        self.calls.append((url, timeout))
        return FakeResponse({"data": []})


def test_llama_cpp_client_sends_openai_compatible_image_payload(tmp_path):
    image = tmp_path / "image.jpg"
    image.write_bytes(b"fake-jpeg")
    session = FakeSession("ok")

    client = LlamaCppClient(
        base_url="http://127.0.0.1:8080/v1/",
        model="qwen-vl",
        timeout=9,
        session=session,
    )

    result = client.chat_with_image(
        image,
        "Describe this image",
        system_prompt="System",
        temperature=0.1,
        max_tokens=32,
        seed=7,
    )

    assert result == "ok"
    url, payload, timeout = session.calls[0]
    assert url == "http://127.0.0.1:8080/v1/chat/completions"
    assert timeout == 9
    assert payload["model"] == "qwen-vl"
    assert payload["temperature"] == 0.1
    assert payload["max_tokens"] == 32
    assert payload["seed"] == 7
    assert payload["messages"][0]["role"] == "system"
    content = payload["messages"][1]["content"]
    assert content[0]["type"] == "image_url"
    assert content[0]["image_url"]["url"].startswith("data:image/jpeg;base64,")
    assert content[1]["text"] == "Describe this image"


def test_llama_cpp_client_health_uses_models_endpoint():
    session = FakeSession("ok")
    client = LlamaCppClient(session=session)

    assert client.health() is True
    assert session.calls[0][0] == "http://127.0.0.1:8080/v1/models"


def test_llama_cpp_tagger_parses_json_and_normalizes_tags(tmp_path):
    image = tmp_path / "image.jpg"
    image.write_bytes(b"fake")
    session = FakeSession(
        '{"tags":["blue_eyes","blue eyes","portrait"],'
        '"ratings":{"safe":0.99},"characters":{"someone":0.91}}'
    )

    tagger = LlamaCppVLMTagger(
        model="vision-model",
        session=session,
        seed=1,
    )
    result = tagger.tag_image(image)

    assert result.tags == ["blue eyes", "portrait"]
    assert result.ratings == {"safe": 0.99}
    assert result.characters == {"someone": 0.91}
    assert session.calls[0][1]["response_format"] == {"type": "json_object"}


def test_llama_cpp_tagger_rejects_invalid_json(tmp_path):
    image = tmp_path / "image.jpg"
    image.write_bytes(b"fake")
    tagger = LlamaCppVLMTagger(session=FakeSession("not-json"))

    with pytest.raises(TaggerError, match="invalid JSON"):
        tagger.tag_image(image)


def test_build_tagger_supports_llama_cpp_backend():
    from lora_data_maker.taggers import build_tagger

    tagger = build_tagger(
        {
            "backend": "llama_cpp",
            "llama_cpp": {
                "base_url": "http://127.0.0.1:8080/v1",
                "model": "vision-model",
            },
        }
    )

    assert isinstance(tagger, LlamaCppVLMTagger)
    assert tagger.model == "vision-model"


def test_llama_cpp_captioner_adds_trigger_prefix(tmp_path):
    image = tmp_path / "image.jpg"
    image.write_bytes(b"fake")
    session = FakeSession("blonde hair, white top, portrait.")

    captioner = LlamaCppCaptioner(
        model="vision-model",
        session=session,
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


def test_build_captioner_supports_llama_cpp_backend():
    captioner = build_captioner(
        {
            "backend": "llama_cpp",
            "llama_cpp": {
                "model": "caption-model",
            },
        }
    )

    assert isinstance(captioner, LlamaCppCaptioner)
    assert captioner.model == "caption-model"


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

    captioner = LlamaCppCaptioner(
        session=FakeSession("portrait, blonde hair."),
    )
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


def test_llama_cpp_rejects_http_errors():
    class ErrorSession:
        def post(self, url, *, json, timeout):
            return FakeResponse({}, status=503)

    client = LlamaCppClient(session=ErrorSession())

    with pytest.raises(LlamaCppError, match="request failed"):
        client.chat([{"role": "user", "content": "hi"}])
