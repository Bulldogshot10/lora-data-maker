from argparse import Namespace
from pathlib import Path

from lora_data_maker.captions import (
    CaptionConfig,
    build_caption,
    caption_manifest,
    resolve_trigger,
    write_caption,
)
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
