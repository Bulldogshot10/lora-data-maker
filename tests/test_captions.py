from argparse import Namespace
from pathlib import Path

import pytest

from lora_data_maker.captions import (
    CaptionConfig,
    build_caption,
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


def test_default_vlm_prompt_uses_four_structured_fields():
    assert DEFAULT_VLM_PROMPT.count("\n") >= 4
    assert "exactly these 4 fields, one per line" in DEFAULT_VLM_PROMPT
    assert "maximum of 5 words per field" in DEFAULT_VLM_PROMPT
    assert "camera view term" in DEFAULT_VLM_PROMPT
    assert "pose: include body pose plus one camera view term when visible" in DEFAULT_VLM_PROMPT
    assert "prefer camera terms: straight-on, from above, from below, high-angle shot, low-angle shot, overhead shot." in DEFAULT_VLM_PROMPT
    assert "do not mention hair color, eye color, face, age, or gender" in DEFAULT_VLM_PROMPT


def test_parse_vlm_caption_normalizes_four_fields():
    assert parse_vlm_caption(
        "pose: standing with arms raised\n"
        "clothing: white halter top, black skirt\n"
        "expression: slight smile\n"
        "background: indoors, soft window lighting"
    ) == (
        "standing with arms raised, "
        "white halter top, black skirt, "
        "slight smile, "
        "indoors, soft window lighting"
    )


def test_parse_vlm_caption_rejects_malformed_output():
    with pytest.raises(VlmCaptionerError, match="exactly four"):
        parse_vlm_caption("pose: standing\nclothing: black shirt")


def test_parse_vlm_caption_rejects_unstructured_prose():
    with pytest.raises(VlmCaptionerError, match="exactly four"):
        parse_vlm_caption("a woman standing in a black dress outdoors")


def test_parse_vlm_caption_rejects_wrong_field_order():
    with pytest.raises(VlmCaptionerError, match="pose, clothing, expression, background"):
        parse_vlm_caption(
            "clothing: black shirt\n"
            "pose: standing\n"
            "expression: slight smile\n"
            "background: indoors"
        )


def test_parse_vlm_caption_rejects_unknown_extra_field():
    with pytest.raises(VlmCaptionerError, match="exactly four"):
        parse_vlm_caption(
            "pose: standing\n"
            "clothing: black shirt\n"
            "expression: slight smile\n"
            "background: indoors\n"
            "hair: blonde"
        )


def test_parse_vlm_caption_rejects_more_than_five_words_per_field():
    with pytest.raises(VlmCaptionerError, match="5-word limit"):
        parse_vlm_caption(
            "pose: standing with both arms raised toward camera\n"
            "clothing: black shirt\n"
            "expression: slight smile\n"
            "background: indoors"
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
        "clothing: black dress\n"
        "expression: smiling\n"
        "background: indoors soft lighting",
        subject="Julia",
        trigger="ignored",
        config=config,
    )
    assert caption == "julia, standing, black dress, smiling, indoors soft lighting"
