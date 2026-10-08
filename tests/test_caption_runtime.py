from pathlib import Path

from lora_data_maker.caption_vlm import (
    LlamaCppVlmCaptioner,
    VlmConfig,
    _Qwen35NoThinkingChatHandler,
    _qwen35_retry_prompt,
    _strip_qwen35_thinking,
)
from lora_data_maker.captions import caption_config_from_dict, caption_manifest
from lora_data_maker.models import ImageRecord, ProjectManifest


def test_caption_config_resolves_relative_vlm_paths(tmp_path):
    settings = caption_config_from_dict(
        {
            "backend": "vlm",
            "vlm": {
                "model_path": "models/model.gguf",
                "mmproj_path": "models/mmproj.gguf",
            },
        },
        root=tmp_path,
    )
    assert settings.vlm.model_path == str(
        (tmp_path / "models" / "model.gguf").resolve()
    )
    assert settings.vlm.mmproj_path == str(
        (tmp_path / "models" / "mmproj.gguf").resolve()
    )


def test_caption_config_accepts_qwen35_chat_format(tmp_path):
    settings = caption_config_from_dict(
        {
            "backend": "vlm",
            "vlm": {
                "model_path": "models/model.gguf",
                "mmproj_path": "models/mmproj.gguf",
                "chat_format": "qwen3.5",
            },
        },
        root=tmp_path,
    )
    assert settings.vlm.chat_format == "qwen3.5"


def test_caption_config_accepts_smolvlm_chat_format(tmp_path):
    settings = caption_config_from_dict(
        {
            "backend": "vlm",
            "vlm": {
                "model_path": "models/model.gguf",
                "mmproj_path": "models/mmproj.gguf",
                "chat_format": "smolvlm",
            },
        },
        root=tmp_path,
    )
    assert settings.vlm.chat_format == "smolvlm"


def test_qwen35_handler_disables_thinking():
    seen = {}

    class FakeHandler:
        def __call__(self, **kwargs):
            seen.update(kwargs)
            return "ok"

    handler = _Qwen35NoThinkingChatHandler(FakeHandler())

    assert handler(messages=[], temperature=0.1) == "ok"
    assert seen["enable_thinking"] is False


def test_qwen35_prompt_helpers():
    assert _qwen35_retry_prompt("describe the image").endswith("\n/no_think")
    assert _qwen35_retry_prompt("describe the image\n/no_think") == (
        "describe the image\n/no_think"
    )
    assert _strip_qwen35_thinking(
        "<think>internal</think>\npose: standing"
    ) == "pose: standing"


def test_qwen35_blank_caption_retries_without_thinking(tmp_path):
    image = tmp_path / "image.jpg"
    image.write_bytes(b"image-data")

    class FakeLlm:
        def __init__(self):
            self.calls = []

        def create_chat_completion(self, **kwargs):
            self.calls.append(kwargs)
            if len(self.calls) == 1:
                return {
                    "choices": [
                        {
                            "message": {
                                "content": "<think>\n\n</think>\n\n",
                            }
                        }
                    ]
                }
            return {
                "choices": [
                    {
                        "message": {
                            "content": (
                                "pose: standing\n"
                                "clothing: white top\n"
                                "expression: neutral\n"
                                "background: indoors"
                            ),
                        }
                    }
                ]
            }

    runtime = LlamaCppVlmCaptioner.__new__(LlamaCppVlmCaptioner)
    runtime.config = VlmConfig(
        chat_format="qwen3.5",
        model_path="model.gguf",
        mmproj_path="mmproj.gguf",
    )
    runtime._llm = FakeLlm()

    content = runtime.caption(image)

    assert content.startswith("pose: standing")
    assert len(runtime._llm.calls) == 2
    retry = runtime._llm.calls[1]
    retry_text = retry["messages"][0]["content"][1]["text"]
    assert retry_text.endswith("/no_think")
    assert retry["max_tokens"] == 256
    assert retry["temperature"] == 0.7
    assert retry["top_p"] == 0.8
    assert retry["top_k"] == 20


def test_caption_manifest_loads_owned_vlm_runtime_once(tmp_path, monkeypatch):
    root = tmp_path / "project"
    manifest = ProjectManifest(
        project_name="julia-set",
        subject="julia",
        trigger="julia",
    )
    for index in range(2):
        image = ImageRecord(
            path=f"raw/{index}.jpg",
            status="processed",
            tags=["1girl", "blue_eyes"],
            processed_path=f"work/processed/{index}.jpg",
        )
        manifest.add_image(image)
        processed = root / image.processed_path
        processed.parent.mkdir(parents=True, exist_ok=True)
        processed.write_bytes(b"processed-image")

    state = {"init": 0, "caption": 0, "close": 0}

    class FakeRuntime:
        def __init__(self, config):
            state["init"] += 1
            assert config.model_path == str((root / "model.gguf").resolve())

        def caption(self, image_path):
            state["caption"] += 1
            assert Path(image_path).is_file()
            return (
                "pose: standing with arms raised\n"
                "clothing: white top\n"
                "expression: neutral\n"
                "background: indoors"
            )

        def close(self):
            state["close"] += 1

    monkeypatch.setattr(
        "lora_data_maker.captions.LlamaCppVlmCaptioner",
        FakeRuntime,
    )
    config = caption_config_from_dict(
        {
            "backend": "vlm",
            "vlm": {
                "model_path": "model.gguf",
                "mmproj_path": "mmproj.gguf",
            },
        },
        root=root,
    )
    summary = caption_manifest(
        manifest,
        root=root,
        output_dir=root / "work" / "captions",
        config=config,
    )

    assert summary.captioned == 2
    assert summary.fallback == 0
    assert state == {"init": 1, "caption": 2, "close": 1}
