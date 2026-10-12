from argparse import Namespace
from pathlib import Path

import pytest

from lora_data_maker.captions import (
    CaptionConfig,
    build_caption,
    build_api_vlm_caption,
    build_vlm_caption,
    caption_config_from_dict,
    caption_manifest,
    clean_vlm_caption,
    estimate_caption_tokens,
    parse_vlm_caption,
    truncate_caption,
    resolve_trigger,
    reset_vlm_captions_for_deterministic,
    write_caption,
)
from lora_data_maker.caption_vlm import DEFAULT_VLM_PROMPT, VlmCaptionerError, VlmConfig
from lora_data_maker.cli import command_caption, command_init
from lora_data_maker.dataset import DatasetProject
from lora_data_maker.models import ImageRecord, ProjectManifest
from lora_data_maker.paths import ProjectPaths


def test_resolve_trigger_defaults_to_subject():
    assert resolve_trigger(
        subject="Julia",
        trigger="old-trigger",
        trigger_from_subject=True,
    ) == "julia"


def test_resolve_trigger_can_use_explicit_trigger():
    assert resolve_trigger(
        subject="Julia",
        trigger="JULIA_V1",
        trigger_from_subject=False,
    ) == "julia v1"


def test_build_caption_injects_trigger_once():
    assert build_caption(
        ["blue_eyes", "julia", "blue eyes", "portrait"],
        subject="Julia",
        trigger="ignored",
    ) == "julia, blue eyes, portrait"


def test_build_caption_keeps_subject_when_trigger_differs():
    assert build_caption(
        ["julia", "blue eyes", "portrait"],
        subject="Julia",
        trigger="julia v1",
        config=CaptionConfig(trigger_from_subject=False),
    ) == "julia v1, julia, blue eyes, portrait"


def test_write_caption_is_deterministic(tmp_path):
    path = tmp_path / "work" / "captions" / "a.txt"
    write_caption(path, "julia, portrait")
    first = path.read_text(encoding="utf-8")
    write_caption(path, "julia, portrait\n\n")
    second = path.read_text(encoding="utf-8")
    assert first == second == "julia, portrait\n"


def test_caption_manifest_writes_and_transitions(tmp_path):
    root = tmp_path / "project"
    manifest = ProjectManifest(
        project_name="julia-set",
        subject="julia",
        trigger="julia-trigger",
    )
    image = ImageRecord(
        path="raw/001.jpg",
        status="tagged",
        tags=["1girl", "julia", "blue_eyes"],
    )
    manifest.add_image(image)

    summary = caption_manifest(
        manifest,
        root=root,
        output_dir=root / "work" / "captions",
    )

    assert summary.scanned == 1
    assert summary.captioned == 1
    assert image.status == "captioned"
    assert image.caption_path == f"work/captions/{image.image_id}.txt"
    assert (
        (root / image.caption_path).read_text(encoding="utf-8")
        == "julia, 1girl, blue eyes\n"
    )

    second = caption_manifest(
        manifest,
        root=root,
        output_dir=root / "work" / "captions",
    )
    assert second.scanned == 1
    assert (
        (root / image.caption_path).read_text(encoding="utf-8")
        == "julia, 1girl, blue eyes\n"
    )


def test_caption_command_persists_manifest(tmp_path):
    root = tmp_path / "project"
    command_init(
        Namespace(
            path=str(root),
            name="julia-set",
            subject="julia",
            trigger=None,
            force=False,
        )
    )

    project = DatasetProject.open(ProjectPaths.from_root(root))
    project.manifest.add_image(
        ImageRecord(
            path="raw/001.jpg",
            status="tagged",
            tags=["1girl", "julia", "blue_eyes"],
        )
    )
    project.save()

    assert command_caption(Namespace(path=str(root))) == 0

    refreshed = DatasetProject.open(ProjectPaths.from_root(root))
    record = next(iter(refreshed.manifest.images.values()))
    assert record.status == "captioned"
    assert record.caption_path is not None
    assert (root / record.caption_path).read_text(encoding="utf-8") == (
        "julia, 1girl, blue eyes\n"
    )
    assert refreshed.manifest.source["last_caption_trigger"] == "julia"
    assert refreshed.manifest.source["last_caption_max_tokens"] == 77


def test_clean_vlm_caption_removes_common_prose_and_markdown():
    raw = """Caption: The image shows a woman in a black dress.
She is sitting outdoors."""
    assert clean_vlm_caption(raw) == "a woman in a black dress. She is sitting outdoors."


def test_default_vlm_prompt_uses_five_structured_fields():
    assert DEFAULT_VLM_PROMPT.count("\n") >= 4
    assert "exactly these 5 fields, one per line" in DEFAULT_VLM_PROMPT
    assert "maximum of 5 words per field" in DEFAULT_VLM_PROMPT
    assert "angle camera: describe the visible camera angle or view" in DEFAULT_VLM_PROMPT
    assert "pose: describe body pose" in DEFAULT_VLM_PROMPT
    assert "prefer camera terms: straight-on, from above, from below, high-angle shot, low-angle shot, overhead shot." in DEFAULT_VLM_PROMPT
    assert "do not mention hair color, eye color, face, age, or gender" in DEFAULT_VLM_PROMPT


def test_parse_vlm_caption_normalizes_five_fields():
    assert parse_vlm_caption(
        "pose: standing with arms raised\n"
        "angle camera: straight-on\n"
        "clothing: white halter top, black skirt\n"
        "expression: slight smile\n"
        "background: indoors, soft window lighting"
    ) == (
        "standing with arms raised, "
        "straight-on, "
        "white halter top, black skirt, "
        "slight smile, "
        "indoors, soft window lighting"
    )


def test_parse_vlm_caption_rejects_malformed_output():
    with pytest.raises(VlmCaptionerError, match="missing required fields"):
        parse_vlm_caption("pose: standing\nclothing: black shirt")


def test_parse_vlm_caption_rejects_unstructured_prose():
    with pytest.raises(VlmCaptionerError, match="does not contain the expected"):
        parse_vlm_caption("a woman standing in a black dress outdoors")


def test_parse_vlm_caption_accepts_camera_angle_alias():
    assert parse_vlm_caption(
        "pose: standing\n"
        "camera angle: straight-on\n"
        "clothing: black shirt\n"
        "expression: neutral\n"
        "background: indoors"
    ) == "standing, straight-on, black shirt, neutral, indoors"


def test_parse_vlm_caption_tolerates_reordered_fields():
    assert parse_vlm_caption(
        "background: indoors, soft lighting\n"
        "expression: slight smile\n"
        "angle camera: straight-on\n"
        "clothing: black shirt\n"
        "pose: standing"
    ) == "standing, straight-on, black shirt, slight smile, indoors, soft lighting"


def test_parse_vlm_caption_ignores_unknown_extra_fields():
    assert parse_vlm_caption(
        "pose: standing\n"
        "angle camera: straight-on\n"
        "clothing: black shirt\n"
        "expression: slight smile\n"
        "hair: blonde\n"
        "background: indoors"
    ) == "standing, straight-on, black shirt, slight smile, indoors"


def test_parse_vlm_caption_tolerates_markdown_formatting():
    fence = "\x60\x60\x60"
    raw = (
        fence + "text\n"
        "- pose: standing\n"
        "- angle camera: straight-on\n"
        "- clothing: white top, black skirt, casual jacket\n"
        "- expression: slight smile\n"
        "- background: indoors, soft window lighting\n"
        + fence
    )
    assert parse_vlm_caption(raw) == (
        "standing, straight-on, white top, black skirt, casual jacket, "
        "slight smile, indoors, soft window lighting"
    )


def test_parse_vlm_caption_accepts_json_structured_output():
    assert parse_vlm_caption(
        '{"pose":"standing", "angle camera":"straight-on", '
        '"clothing":"white top", "expression":"neutral", '
        '"background":"indoors, soft lighting"}'
    ) == "standing, straight-on, white top, neutral, indoors, soft lighting"


def test_parse_vlm_caption_accepts_json_camera_angle_alias():
    assert parse_vlm_caption(
        '{"pose":"standing", "camera angle":"straight-on", '
        '"clothing":"black shirt", "expression":"neutral", '
        '"background":"indoors"}'
    ) == "standing, straight-on, black shirt, neutral, indoors"


def test_parse_vlm_caption_accepts_longer_fields_without_failing():
    assert parse_vlm_caption(
        "pose: standing with both arms raised\n"
        "angle camera: low-angle shot\n"
        "clothing: black shirt\n"
        "expression: slight smile\n"
        "background: indoors"
    ) == (
        "standing with both arms raised, low-angle shot, black shirt, "
        "slight smile, indoors"
    )


def test_build_api_vlm_caption_injects_trigger_from_five_fields():
    config = CaptionConfig(
        backend="vlm",
        vlm=VlmConfig(
            runtime="openai_compatible",
            api_key_env="OPENAI_API_KEY",
            api_model="vision-model",
            api_base_url="https://api.openai.com/v1",
        ),
    )
    assert build_api_vlm_caption(
        "pose: standing in place\n"
        "angle camera: straight-on\n"
        "clothing: white top\n"
        "expression: smiling\n"
        "background: indoors with soft lighting",
        subject="Julia",
        trigger="ignored",
        config=config,
    ) == "julia, standing in place, white top, smiling, indoors with soft lighting"



def test_build_api_vlm_caption_keeps_unstructured_output_for_audit():
    config = CaptionConfig(
        backend="vlm",
        vlm=VlmConfig(
            runtime="openai_compatible",
            api_key_env="OPENAI_API_KEY",
            api_model="vision-model",
            api_base_url="https://api.openai.com/v1",
        ),
    )
    assert build_api_vlm_caption(
        "A woman standing in a black dress indoors.",
        subject="Julia",
        trigger="ignored",
        config=config,
    ) == "julia, A woman standing in a black dress indoors."


def test_caption_manifest_keeps_unstructured_api_caption_and_marks_warning(tmp_path):
    root = tmp_path / "project"
    manifest = ProjectManifest(
        project_name="julia-set",
        subject="julia",
        trigger="julia-trigger",
    )
    image = ImageRecord(
        path="raw/001.jpg",
        status="processed",
        tags=[],
        processed_path="work/processed/001.jpg",
    )
    manifest.add_image(image)
    processed = root / image.processed_path
    processed.parent.mkdir(parents=True, exist_ok=True)
    processed.write_bytes(b"processed-image")

    class UnstructuredCaptioner:
        def caption(self, image_path):
            return "A woman standing in a black dress indoors."

        def close(self):
            raise AssertionError("caller must not close injected captioner")

    config = CaptionConfig(
        backend="vlm",
        vlm=VlmConfig(
            runtime="openai_compatible",
            api_key_env="OPENAI_API_KEY",
            api_model="vision-model",
            api_base_url="https://ollama.com/v1",
        ),
    )
    summary = caption_manifest(
        manifest,
        root=root,
        output_dir=root / "work" / "captions",
        config=config,
        captioner=UnstructuredCaptioner(),
    )

    assert summary.captioned == 1
    assert summary.failed == 0
    assert image.status == "captioned"
    assert image.metadata["caption_vlm_format_warning"].startswith(
        "VLM caption does not contain the expected"
    )
    assert image.metadata["caption_vlm_raw_output"] == (
        "A woman standing in a black dress indoors."
    )
    assert (
        (root / image.caption_path).read_text(encoding="utf-8")
        == "julia, A woman standing in a black dress indoors.\n"
    )


def test_build_vlm_caption_removes_repeated_trigger_mentions():
    config = CaptionConfig(
        backend="vlm",
        vlm=VlmConfig(
            model_path="model.gguf",
            mmproj_path="mmproj.gguf",
        ),
    )
    assert build_vlm_caption(
        "pose: portrait pose\n"
        "angle camera: straight-on\n"
        "clothing: black shirt\n"
        "expression: smiling\n"
        "background: indoor room",
        subject="Julia",
        trigger="ignored",
        config=config,
    ) == "julia, portrait pose, black shirt, smiling, indoor room"


def test_caption_config_rejects_llama_vlm_without_paths():
    with pytest.raises(
        ValueError,
        match="caption.vlm.model_path is required for the llama_cpp runtime",
    ):
        CaptionConfig(backend="vlm")


def test_caption_config_allows_inactive_incomplete_api_runtime():
    settings = caption_config_from_dict(
        {
            "backend": "deterministic",
            "vlm": {
                "runtime": "openai_compatible",
                "api_key_env": "",
                "api_model": "",
                "api_base_url": "",
            },
        }
    )
    assert settings.backend == "deterministic"
    assert settings.vlm.runtime == "openai_compatible"
    assert settings.vlm.api_model == ""


def test_caption_config_from_dict_keeps_deterministic_default():
    settings = caption_config_from_dict(
        {
            "separator": ", ",
            "subject_position": "first",
            "trigger_from_subject": True,
        }
    )
    assert settings.backend == "deterministic"
    assert settings.vlm.model_path == ""


def test_caption_manifest_uses_injected_vlm_captioner(tmp_path):
    root = tmp_path / "project"
    manifest = ProjectManifest(
        project_name="julia-set",
        subject="julia",
        trigger="julia-trigger",
    )
    image = ImageRecord(
        path="raw/001.jpg",
        status="tagged",
        tags=["1girl", "julia", "blue_eyes"],
        processed_path="work/processed/001.jpg",
    )
    manifest.add_image(image)
    processed = root / image.processed_path
    processed.parent.mkdir(parents=True, exist_ok=True)
    processed.write_bytes(b"processed-image")

    class FakeCaptioner:
        def caption(self, image_path):
            assert Path(image_path) == processed.resolve()
            return (
                "pose: standing with arms raised\n"
                "angle camera: straight-on\n"
                "clothing: white halter top, black skirt\n"
                "expression: slight smile\n"
                "background: indoors, soft window lighting"
            )

        def close(self):
            raise AssertionError("caller must not close injected captioner")

    config = CaptionConfig(
        backend="vlm",
        vlm=VlmConfig(
            model_path="model.gguf",
            mmproj_path="mmproj.gguf",
        ),
    )

    summary = caption_manifest(
        manifest,
        root=root,
        output_dir=root / "work" / "captions",
        config=config,
        captioner=FakeCaptioner(),
    )

    assert summary.scanned == 1
    assert summary.captioned == 1
    assert summary.fallback == 0
    assert image.metadata["caption_backend"] == "vlm"
    assert image.metadata["caption_fallback"] is False
    assert (
        (root / image.caption_path).read_text(encoding="utf-8")
        == (
            "julia, standing with arms raised, white halter top, black skirt, "
            "slight smile, indoors, soft window lighting\n"
        )
    )


def test_reset_vlm_captions_for_deterministic_requeues_records(tmp_path):
    root = tmp_path / "project"
    manifest = ProjectManifest(
        project_name="julia-set",
        subject="julia",
        trigger="julia",
    )
    image = ImageRecord(
        path="raw/001.jpg",
        status="captioned",
        processed_path="work/processed/001.jpg",
        caption_path="work/captions/001.txt",
        metadata={
            "caption_backend": "vlm",
            "caption_stage": "captioned",
        },
    )
    manifest.add_image(image)

    stale = root / image.caption_path
    stale.parent.mkdir(parents=True, exist_ok=True)
    stale.write_text("julia, stale vlm caption\n", encoding="utf-8")

    reset = reset_vlm_captions_for_deterministic(
        manifest,
        root=root,
    )

    assert reset == 1
    assert image.status == "processed"
    assert image.caption_path is None
    assert image.metadata["caption_previous_backend"] == "vlm"
    assert not stale.exists()


def test_caption_manifest_fails_without_deterministic_fallback(tmp_path):
    root = tmp_path / "project"
    manifest = ProjectManifest(
        project_name="julia-set",
        subject="julia",
        trigger="julia-trigger",
    )
    image = ImageRecord(
        path="raw/001.jpg",
        status="processed",
        tags=["1girl", "julia", "blue_eyes"],
        processed_path="work/processed/001.jpg",
    )
    manifest.add_image(image)
    processed = root / image.processed_path
    processed.parent.mkdir(parents=True, exist_ok=True)
    processed.write_bytes(b"processed-image")

    class FailingCaptioner:
        def caption(self, image_path):
            raise VlmCaptionerError("boom")

        def close(self):
            pass

    config = CaptionConfig(
        backend="vlm",
        vlm=VlmConfig(
            model_path="model.gguf",
            mmproj_path="mmproj.gguf",
        ),
    )

    summary = caption_manifest(
        manifest,
        root=root,
        output_dir=root / "work" / "captions",
        config=config,
        captioner=FailingCaptioner(),
    )

    assert summary.scanned == 1
    assert summary.captioned == 0
    assert summary.failed == 1
    assert summary.fallback == 0
    assert image.status == "processed"
    assert image.caption_path is None
    assert image.metadata["caption_stage"] == "failed"
    assert "boom" in image.metadata["caption_vlm_error"]
    assert not (root / "work" / "captions" / f"{image.image_id}.txt").exists()


def test_caption_manifest_clears_stale_vlm_caption_after_failure(tmp_path):
    root = tmp_path / "project"
    manifest = ProjectManifest(
        project_name="julia-set",
        subject="julia",
        trigger="julia",
    )
    image = ImageRecord(
        path="raw/001.jpg",
        status="captioned",
        tags=["booru", "blue_eyes"],
        processed_path="work/processed/001.jpg",
        caption_path="work/captions/001.txt",
    )
    manifest.add_image(image)

    processed = root / image.processed_path
    processed.parent.mkdir(parents=True, exist_ok=True)
    processed.write_bytes(b"processed-image")
    stale = root / image.caption_path
    stale.parent.mkdir(parents=True, exist_ok=True)
    stale.write_text("julia, stale caption\n", encoding="utf-8")

    class FailingCaptioner:
        def caption(self, image_path):
            raise VlmCaptionerError("boom")

        def close(self):
            pass

    config = CaptionConfig(
        backend="vlm",
        vlm=VlmConfig(
            model_path="model.gguf",
            mmproj_path="mmproj.gguf",
        ),
    )

    summary = caption_manifest(
        manifest,
        root=root,
        output_dir=root / "work" / "captions",
        config=config,
        captioner=FailingCaptioner(),
    )

    assert summary.failed == 1
    assert image.status == "processed"
    assert image.caption_path is None
    assert not stale.exists()


def test_truncate_caption_preserves_custom_separator():
    caption = truncate_caption(
        "julia; portrait; black dress; indoors",
        8,
        separator="; ",
    )
    assert "; " in caption
    assert estimate_caption_tokens(caption) <= 8


def test_caption_token_budget_is_enforced():
    config = CaptionConfig(max_tokens=12)
    caption = build_caption(
        ["julia", "long blonde hair", "blue eyes", "black dress", "portrait"],
        subject="julia",
        trigger="julia",
        config=config,
    )
    assert estimate_caption_tokens(caption) <= 12


def test_truncate_caption_drops_later_content_when_budget_is_full():
    caption = truncate_caption(
        "julia, a woman with long blonde hair and blue eyes, black dress, indoors",
        12,
    )
    assert caption.startswith("julia")
    assert estimate_caption_tokens(caption) <= 12


def test_vlm_caption_ignores_tagger_tags_and_max_token_setting():
    config = CaptionConfig(
        backend="vlm",
        max_tokens=1,
        vlm=VlmConfig(
            model_path="model.gguf",
            mmproj_path="mmproj.gguf",
        ),
    )
    caption = build_vlm_caption(
        "pose: standing\n"
        "angle camera: straight-on\n"
        "clothing: black dress\n"
        "expression: smiling\n"
        "background: indoors soft lighting",
        subject="Julia",
        trigger="ignored",
        config=config,
    )
    assert caption == "julia, standing, black dress, smiling, indoors soft lighting"


def test_vlm_caption_can_use_raw_image_when_processing_is_bypassed(tmp_path):
    from PIL import Image

    root = tmp_path / "project"
    raw_path = root / "raw" / "001.jpg"
    raw_path.parent.mkdir(parents=True)
    Image.new("RGB", (48, 32), (80, 90, 100)).save(raw_path)

    manifest = ProjectManifest(
        project_name="julia-set",
        subject="julia",
        trigger="julia",
    )
    image = ImageRecord(path="raw/001.jpg", status="validated")
    manifest.add_image(image)

    class RawImageCaptioner:
        def __init__(self):
            self.image_path = None

        def caption(self, image_path):
            self.image_path = Path(image_path)
            return (
                "pose: standing\n"
                "angle camera: straight-on\n"
                "clothing: white top\n"
                "expression: neutral\n"
                "background: indoors"
            )

    captioner = RawImageCaptioner()
    settings = CaptionConfig(
        backend="vlm",
        vlm=VlmConfig(
            runtime="llama_cpp",
            model_path="model.gguf",
            mmproj_path="mmproj.gguf",
        ),
    )

    summary = caption_manifest(
        manifest,
        root=root,
        output_dir=root / "work" / "captions",
        config=settings,
        captioner=captioner,
        use_processed_images=False,
    )

    assert summary.captioned == 1
    assert summary.failed == 0
    assert captioner.image_path == raw_path.resolve()
    assert image.status == "captioned"
    assert image.metadata["caption_vlm_image_source"] == "raw"
    assert (root / image.caption_path).is_file()


def test_vlm_caption_uses_raw_image_when_processing_is_disabled(tmp_path):
    from PIL import Image

    root = tmp_path / "project"
    raw = root / "raw" / "001.png"
    raw.parent.mkdir(parents=True)
    Image.new("RGB", (80, 60), (20, 30, 40)).save(raw)

    manifest = ProjectManifest(
        project_name="raw-caption",
        subject="julia",
        trigger="julia",
    )
    record = ImageRecord(path="raw/001.png", status="validated")
    manifest.add_image(record)

    class RecordingCaptioner:
        def __init__(self):
            self.paths = []

        def caption(self, image_path):
            self.paths.append(Path(image_path).resolve())
            return (
                "pose: standing\n"
                "angle camera: straight-on\n"
                "clothing: black shirt\n"
                "expression: neutral\n"
                "background: indoors"
            )

    captioner = RecordingCaptioner()
    config = CaptionConfig(
        backend="vlm",
        vlm=VlmConfig(
            runtime="openai_compatible",
            api_key_env="OPENAI_API_KEY",
            api_model="vision-model",
            api_base_url="https://example.com/v1",
        ),
    )

    summary = caption_manifest(
        manifest,
        root=root,
        output_dir=root / "work" / "captions",
        config=config,
        captioner=captioner,
        use_processed_images=False,
    )

    assert summary.captioned == 1
    assert summary.failed == 0
    assert captioner.paths == [raw.resolve()]
    assert record.status == "captioned"
    assert record.metadata["caption_vlm_image_source"] == "raw"
    assert (root / record.caption_path).is_file()


def test_vlm_caption_failure_keeps_raw_tagged_record_retryable(tmp_path):
    from PIL import Image

    root = tmp_path / "project"
    raw = root / "raw" / "001.png"
    raw.parent.mkdir(parents=True)
    Image.new("RGB", (80, 60), (20, 30, 40)).save(raw)

    manifest = ProjectManifest(
        project_name="raw-caption",
        subject="julia",
        trigger="julia",
    )
    record = ImageRecord(
        path="raw/001.png",
        status="tagged",
        tags=["portrait"],
    )
    manifest.add_image(record)

    class FailingCaptioner:
        def caption(self, image_path):
            assert Path(image_path).resolve() == raw.resolve()
            raise RuntimeError("temporary API failure")

    config = CaptionConfig(
        backend="vlm",
        vlm=VlmConfig(
            runtime="openai_compatible",
            api_key_env="OPENAI_API_KEY",
            api_model="vision-model",
            api_base_url="https://example.com/v1",
        ),
    )
    summary = caption_manifest(
        manifest,
        root=root,
        output_dir=root / "work" / "captions",
        config=config,
        captioner=FailingCaptioner(),
        use_processed_images=False,
    )

    assert summary.scanned == 1
    assert summary.captioned == 0
    assert summary.failed == 1
    assert record.status == "tagged"
    assert record.caption_path is None
    assert record.metadata["caption_vlm_error"] == "temporary API failure"
