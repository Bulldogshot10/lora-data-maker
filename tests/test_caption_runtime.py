from pathlib import Path

from lora_data_maker.caption_vlm import LlamaCppVlmCaptioner, VlmConfig
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




def test_caption_config_accepts_onnx_runtime_without_mmproj(tmp_path):
    model_dir = tmp_path / "models" / "SmolVLM2-500M-Video-Instruct"
    model_dir.mkdir(parents=True)
    settings = caption_config_from_dict(
        {
            "backend": "vlm",
            "vlm": {
                "runtime": "onnx",
                "model_path": "models/SmolVLM2-500M-Video-Instruct",
            },
        },
        root=tmp_path,
    )
    assert settings.vlm.runtime == "onnx"
    assert settings.vlm.model_path == str(
        (tmp_path / "models" / "SmolVLM2-500M-Video-Instruct").resolve()
    )
    assert settings.vlm.mmproj_path == ""


def test_caption_manifest_routes_onnx_vlm_runtime(tmp_path, monkeypatch):
    root = tmp_path / "project"
    manifest = ProjectManifest(
        project_name="julia-set",
        subject="julia",
        trigger="julia",
    )
    image = ImageRecord(
        path="raw/0.jpg",
        status="processed",
        tags=[],
        processed_path="work/processed/0.jpg",
    )
    manifest.add_image(image)
    processed = root / image.processed_path
    processed.parent.mkdir(parents=True, exist_ok=True)
    processed.write_bytes(b"processed-image")

    state = {"init": 0, "caption": 0, "close": 0}

    class FakeRuntime:
        def __init__(self, config):
            state["init"] += 1
            assert config.runtime == "onnx"
            assert config.mmproj_path == ""
            assert config.model_path == str((root / "model").resolve())

        def caption(self, image_path):
            state["caption"] += 1
            assert Path(image_path).is_file()
            return (
                "pose: standing\n"
                "clothing: white top\n"
                "expression: neutral\n"
                "background: indoors"
            )

        def close(self):
            state["close"] += 1

    import lora_data_maker.caption_onnx as caption_onnx

    monkeypatch.setattr(caption_onnx, "SmolVlmOnnxCaptioner", FakeRuntime)
    config = caption_config_from_dict(
        {
            "backend": "vlm",
            "vlm": {
                "runtime": "onnx",
                "model_path": "model",
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

    assert summary.captioned == 1
    assert summary.failed == 0
    assert state == {"init": 1, "caption": 1, "close": 1}


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


def test_blank_vlm_completion_includes_generation_diagnostics(tmp_path):
    image = tmp_path / "image.jpg"
    image.write_bytes(b"image-data")

    class FakeLlm:
        def create_chat_completion(self, **kwargs):
            return {
                "choices": [
                    {
                        "message": {
                            "content": "<think>\n\n</think>\n\n",
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"completion_tokens": 0},
            }

    runtime = LlamaCppVlmCaptioner.__new__(LlamaCppVlmCaptioner)
    runtime.config = VlmConfig(
        chat_format="qwen3.5",
        model_path="model.gguf",
        mmproj_path="mmproj.gguf",
    )
    runtime._llm = FakeLlm()

    try:
        runtime.caption(image)
    except Exception as exc:
        message = str(exc)
    else:
        raise AssertionError("expected blank completion to fail")

    assert "finish_reason='stop'" in message
    assert "completion_tokens=0" in message
    assert "raw_content='<think>\\n\\n</think>\\n\\n'" in message


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


def test_onnx_model_resolver_auto_downloads_default_repo(tmp_path, monkeypatch):
    import sys
    import types

    from lora_data_maker.caption_onnx import SmolVlmOnnxCaptioner

    snapshot_root = tmp_path / "hf-snapshot"
    calls = {}

    def fake_snapshot_download(*, repo_id, allow_patterns):
        calls["repo_id"] = repo_id
        calls["allow_patterns"] = list(allow_patterns)
        return str(snapshot_root)

    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        types.SimpleNamespace(snapshot_download=fake_snapshot_download),
    )

    resolved = SmolVlmOnnxCaptioner._resolve_model_root("")

    assert resolved == snapshot_root.resolve()
    assert calls["repo_id"] == SmolVlmOnnxCaptioner._DEFAULT_REPO_ID
    assert "onnx/decoder_model_merged_q4f16.onnx" in calls["allow_patterns"]
    assert "onnx/embed_tokens_q4f16.onnx" in calls["allow_patterns"]
    assert "onnx/vision_encoder_q4f16.onnx" in calls["allow_patterns"]


def test_onnx_cache_shape_resolver_uses_empty_sequence():
    from lora_data_maker.caption_onnx import _resolve_empty_cache_shape

    shape = _resolve_empty_cache_shape(
        ("batch_size", 5, "past_sequence_length", 64),
        batch_size=1,
        num_kv_heads=5,
    )

    assert shape == (1, 5, 0, 64)


def test_onnx_position_ids_match_transformers_cache_semantics():
    import numpy as np

    from lora_data_maker.caption_onnx import SmolVlmOnnxCaptioner

    runtime = SmolVlmOnnxCaptioner.__new__(SmolVlmOnnxCaptioner)
    mask = np.ones((1, 5), dtype=np.int64)

    np.testing.assert_array_equal(
        runtime._position_ids(mask),
        np.array([[0, 1, 2, 3, 4]], dtype=np.int64),
    )
    np.testing.assert_array_equal(
        runtime._position_ids(mask, last_only=True),
        np.array([[4]], dtype=np.int64),
    )
