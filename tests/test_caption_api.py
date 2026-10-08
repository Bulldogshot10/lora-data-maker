from pathlib import Path

import requests
from PIL import Image

from lora_data_maker.caption_api import GeminiVlmCaptioner, OpenAICompatibleVlmCaptioner
from lora_data_maker.caption_vlm import DEFAULT_VLM_PROMPT, VlmConfig
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
                                        "pose: standing straight-on\\n"
                                        "clothing: white top\\n"
                                        "expression: neutral\\n"
                                        "background: indoor soft lighting"
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

    runtime = GeminiVlmCaptioner(
        VlmConfig(
            runtime="gemini",
            api_key_env="TEST_GEMINI_KEY",
            api_model="gemini-test",
            api_base_url="https://example.test/v1beta",
        )
    )
    monkeypatch.setenv("TEST_GEMINI_KEY", "secret")
    runtime.close()

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
    assert payload["contents"][0]["parts"][0]["text"] == DEFAULT_VLM_PROMPT
    assert payload["generationConfig"]["maxOutputTokens"] == 64
    assert payload["contents"][0]["parts"][1]["inline_data"]["mime_type"] == "image/jpeg"


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
                                "pose: sitting straight-on\\n"
                                "clothing: black dress\\n"
                                "expression: smiling\\n"
                                "background: indoors soft lighting"
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
    content = kwargs["json"]["messages"][0]["content"]
    assert content[0]["type"] == "image_url"
    assert content[1]["type"] == "text"
    assert content[1]["text"] == DEFAULT_VLM_PROMPT


def test_api_vlm_caption_injects_trigger_and_keeps_constraints(monkeypatch, tmp_path):
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
                                        "pose: standing straight-on\\n"
                                        "clothing: white spandex top\\n"
                                        "expression: neutral\\n"
                                        "background: indoor soft lighting"
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
    assert caption == "julia, standing straight-on, white spandex top, neutral, indoor soft lighting\\n"
