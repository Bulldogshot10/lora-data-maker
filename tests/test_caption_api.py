from pathlib import Path

import requests
from PIL import Image

from lora_data_maker.caption_api import API_VLM_PROMPT, GeminiVlmCaptioner, OpenAICompatibleVlmCaptioner
from lora_data_maker.caption_vlm import VlmConfig
from lora_data_maker.captions import CaptionConfig, caption_manifest
from lora_data_maker.models import ImageRecord, ProjectManifest


class FakeResponse:
    def __init__(self, payload, *, status_code=200, ok=True):
        self._payload = payload
        self.status_code = status_code
        self.ok = ok
        self.text = str(payload)

    def json(self):
        return self._payload


class FakeSession:
    payloads = []
    responses = []

    def post(self, url, **kwargs):
        type(self).payloads.append((url, kwargs))
        return type(self).responses.pop(0)

    def close(self):
        pass


def test_gemini_caption_uses_fixed_prompt_and_inline_image(monkeypatch, tmp_path):
    monkeypatch.setattr(requests, "Session", FakeSession)
    FakeSession.payloads.clear()
    FakeSession.responses[:] = [
        FakeResponse(
            {
                "candidates": [
                    {
                        "content": {
                            "parts": [
                                {
                                    "text": (
                                        "pose: standing straight-on\nclothing: white top\nexpression: neutral\nbackground: indoors with soft lighting"
                                    )
                                }
                            ]
                        }
                    }
                ]
            }
        )
    ]

    image = tmp_path / "image.jpg"
    Image.new("RGB", (16, 16), "white").save(image)

    monkeypatch.setenv("TEST_GEMINI_KEY", "secret")
    runtime = GeminiVlmCaptioner(
        VlmConfig(
            runtime="gemini",
            api_key_env="TEST_GEMINI_KEY",
            api_model="gemini-test",
            api_base_url="https://example.test/v1beta",
        )
    )
    try:
        raw = runtime.caption(image)
    finally:
        runtime.close()

    assert raw.startswith("pose: standing")
    url, kwargs = FakeSession.payloads[-1]
    assert url == "https://example.test/v1beta/models/gemini-test:generateContent"
    assert kwargs["headers"]["x-goog-api-key"] == "secret"
    payload = kwargs["json"]
    assert payload["contents"][0]["parts"][1]["text"] == API_VLM_PROMPT
    assert payload["contents"][0]["parts"][0]["inline_data"]["mime_type"] == "image/jpeg"


def test_openai_compatible_caption_uses_chat_completions(monkeypatch, tmp_path):
    monkeypatch.setattr(requests, "Session", FakeSession)
    FakeSession.payloads.clear()
    FakeSession.responses[:] = [
        FakeResponse(
            {
                "choices": [
                    {
                        "message": {
                            "content": (
                                "pose: sitting straight-on\nclothing: black dress\nexpression: smiling\nbackground: indoors with soft lighting"
                            )
                        }
                    }
                ]
            }
        )
    ]
    monkeypatch.setenv("TEST_OPENAI_KEY", "secret")

    image = tmp_path / "image.jpg"
    Image.new("RGB", (16, 16), "white").save(image)

    runtime = OpenAICompatibleVlmCaptioner(
        VlmConfig(
            runtime="openai_compatible",
            api_key_env="TEST_OPENAI_KEY",
            api_model="vision-model",
            api_base_url="https://example.test/v1",
        )
    )
    try:
        raw = runtime.caption(image)
    finally:
        runtime.close()

    assert raw.startswith("pose: sitting")
    url, kwargs = FakeSession.payloads[-1]
    assert url == "https://example.test/v1/chat/completions"
    assert kwargs["headers"]["Authorization"] == "Bearer secret"
    payload = kwargs["json"]
    content = payload["messages"][0]["content"]
    assert content[0]["type"] == "image_url"
    assert content[1]["type"] == "text"
    assert content[1]["text"] == API_VLM_PROMPT


def test_api_vlm_caption_injects_trigger_from_four_fields(monkeypatch, tmp_path):
    monkeypatch.setattr(requests, "Session", FakeSession)
    FakeSession.payloads.clear()
    FakeSession.responses[:] = [
        FakeResponse(
            {
                "candidates": [
                    {
                        "content": {
                            "parts": [
                                {
                                    "text": (
                                        "pose: standing straight-on\nclothing: white spandex top\nexpression: neutral\nbackground: indoor soft lighting"
                                    )
                                }
                            ]
                        }
                    }
                ]
            }
        )
    ]
    monkeypatch.setenv("TEST_GEMINI_KEY", "secret")

    root = tmp_path / "project"
    processed = root / "work" / "processed" / "0.jpg"
    processed.parent.mkdir(parents=True)
    Image.new("RGB", (32, 32), "white").save(processed)

    manifest = ProjectManifest(
        project_name="julia",
        subject="julia",
        trigger="julia",
    )
    manifest.add_image(
        ImageRecord(
            path="raw/0.jpg",
            status="processed",
            processed_path="work/processed/0.jpg",
        )
    )

    runtime = GeminiVlmCaptioner(
        VlmConfig(
            runtime="gemini",
            api_key_env="TEST_GEMINI_KEY",
            api_model="gemini-test",
            api_base_url="https://example.test/v1beta",
        )
    )
    try:
        summary = caption_manifest(
            manifest,
            root=root,
            output_dir=root / "work" / "captions",
            config=CaptionConfig(
                backend="vlm",
                vlm=VlmConfig(
                    runtime="gemini",
                    api_key_env="TEST_GEMINI_KEY",
                    api_model="gemini-test",
                    api_base_url="https://example.test/v1beta",
                ),
            ),
            captioner=runtime,
        )
    finally:
        runtime.close()

    assert summary.captioned == 1
    record = next(iter(manifest.images.values()))
    caption = (root / record.caption_path).read_text(encoding="utf-8")
    assert caption == "julia, standing straight-on, wearing a white spandex top, neutral expression, indoors with soft lighting\n"
