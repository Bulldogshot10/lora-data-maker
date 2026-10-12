from argparse import Namespace

import pytest
from PIL import Image

from lora_data_maker.cli import command_export, command_init
from lora_data_maker.dataset import DatasetProject
from lora_data_maker.export import (
    export_manifest,
    kohya_folder_name,
    safe_component,
)
from lora_data_maker.models import ImageRecord, ProjectManifest
from lora_data_maker.paths import ProjectPaths


def test_safe_component_and_kohya_folder_name():
    assert safe_component("Julia Girl!") == "Julia_Girl"
    assert kohya_folder_name(subject="julia", repeats=10) == "10_julia"


def test_kohya_folder_name_rejects_nested_paths():
    with pytest.raises(ValueError):
        kohya_folder_name(subject="julia", repeats=1, folder_name="../bad")


def test_export_manifest_writes_matching_image_and_caption(tmp_path):
    root = tmp_path / "project"
    processed = root / "work" / "processed"
    captions = root / "work" / "captions"
    processed.mkdir(parents=True)
    captions.mkdir(parents=True)

    image_id = "abc123"
    source_image = processed / f"{image_id}.jpg"
    source_caption = captions / f"{image_id}.txt"
    Image.new("RGB", (32, 32), (20, 30, 40)).save(source_image)
    source_caption.write_text("julia, portrait\n", encoding="utf-8")

    manifest = ProjectManifest(
        project_name="julia-set",
        subject="julia",
        trigger="julia",
    )
    record = ImageRecord(
        image_id=image_id,
        path="raw/001.jpg",
        status="captioned",
        processed_path="work/processed/abc123.jpg",
        caption_path="work/captions/abc123.txt",
        tags=["julia", "portrait"],
    )
    manifest.add_image(record)

    summary = export_manifest(
        manifest,
        root=root,
        output_dir=root / "output",
        repeats=10,
    )

    destination = root / "output" / "10_julia"
    assert summary.scanned == 1
    assert summary.exported == 1
    assert record.status == "exported"
    assert (destination / "abc123.jpg").is_file()
    assert (destination / "abc123.txt").read_text(encoding="utf-8") == (
        "julia, portrait\n"
    )
    assert record.metadata["export_layout"] == "kohya"


def test_export_command_persists_manifest(tmp_path):
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
    processed = root / "work" / "processed"
    captions = root / "work" / "captions"
    processed.mkdir(parents=True, exist_ok=True)
    captions.mkdir(parents=True, exist_ok=True)

    image_id = "abc123"
    Image.new("RGB", (32, 32), (20, 30, 40)).save(processed / f"{image_id}.jpg")
    (captions / f"{image_id}.txt").write_text(
        "julia, portrait\n",
        encoding="utf-8",
    )
    project.manifest.add_image(
        ImageRecord(
            image_id=image_id,
            path="raw/001.jpg",
            status="captioned",
            processed_path=f"work/processed/{image_id}.jpg",
            caption_path=f"work/captions/{image_id}.txt",
            tags=["julia", "portrait"],
        )
    )
    project.save()

    assert command_export(
        Namespace(path=str(root), repeats=5, folder_name=None)
    ) == 0

    refreshed = DatasetProject.open(ProjectPaths.from_root(root))
    record = next(iter(refreshed.manifest.images.values()))
    assert record.status == "exported"
    assert refreshed.manifest.source["last_export_repeats"] == 5
    assert (root / "output" / "5_julia" / f"{image_id}.txt").is_file()


def test_export_command_returns_failure_when_candidate_fails(tmp_path):
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
    processed = root / "work" / "processed"
    captions = root / "work" / "captions"
    processed.mkdir(parents=True, exist_ok=True)
    captions.mkdir(parents=True, exist_ok=True)

    image_id = "good"
    Image.new("RGB", (32, 32), (20, 30, 40)).save(processed / f"{image_id}.jpg")
    (captions / f"{image_id}.txt").write_text("julia\\n", encoding="utf-8")
    project.manifest.add_image(
        ImageRecord(
            image_id=image_id,
            path="raw/good.jpg",
            status="captioned",
            processed_path=f"work/processed/{image_id}.jpg",
            caption_path=f"work/captions/{image_id}.txt",
        )
    )
    project.manifest.add_image(
        ImageRecord(
            image_id="bad",
            path="raw/bad.jpg",
            status="captioned",
            processed_path="work/processed/missing.jpg",
            caption_path="work/captions/missing.txt",
        )
    )
    project.save()

    assert command_export(
        Namespace(path=str(root), repeats=1, folder_name=None)
    ) == 1


def test_export_rolls_back_partial_pair_on_caption_failure(tmp_path, monkeypatch):
    root = tmp_path / "project"
    processed = root / "work" / "processed"
    captions = root / "work" / "captions"
    processed.mkdir(parents=True)
    captions.mkdir(parents=True)

    image_id = "abc123"
    source_image = processed / f"{image_id}.jpg"
    source_caption = captions / f"{image_id}.txt"
    Image.new("RGB", (32, 32), (20, 30, 40)).save(source_image)
    source_caption.write_text("julia\n", encoding="utf-8")

    manifest = ProjectManifest(
        project_name="julia-set",
        subject="julia",
        trigger="julia",
    )
    manifest.add_image(
        ImageRecord(
            image_id=image_id,
            path="raw/001.jpg",
            status="captioned",
            processed_path=source_image.relative_to(root).as_posix(),
            caption_path=source_caption.relative_to(root).as_posix(),
        )
    )

    from lora_data_maker import export as export_module
    original = export_module._copy_atomic

    def fail_caption(source, destination):
        if destination.suffix == ".txt":
            raise OSError("simulated caption copy failure")
        original(source, destination)

    monkeypatch.setattr(export_module, "_copy_atomic", fail_caption)

    summary = export_manifest(
        manifest,
        root=root,
        output_dir=root / "output",
        repeats=1,
    )

    destination = root / "output" / "1_julia"
    assert summary.failed == 1
    assert not (destination / f"{image_id}.jpg").exists()
    assert not (destination / f"{image_id}.txt").exists()
    assert manifest.images[image_id].status == "captioned"


def test_export_manifest_uses_raw_image_when_processing_is_bypassed(tmp_path):
    root = tmp_path / "project"
    raw_dir = root / "raw"
    captions = root / "work" / "captions"
    raw_dir.mkdir(parents=True)
    captions.mkdir(parents=True)

    image_id = "raw-image"
    source_image = raw_dir / "source.png"
    source_caption = captions / f"{image_id}.txt"
    Image.new("RGB", (48, 32), (30, 40, 50)).save(source_image)
    source_caption.write_text("julia, portrait\n", encoding="utf-8")

    manifest = ProjectManifest(
        project_name="julia-set",
        subject="julia",
        trigger="julia",
    )
    record = ImageRecord(
        image_id=image_id,
        path="raw/source.png",
        status="captioned",
        caption_path=source_caption.relative_to(root).as_posix(),
    )
    manifest.add_image(record)

    summary = export_manifest(
        manifest,
        root=root,
        output_dir=root / "output",
        repeats=1,
        use_processed_images=False,
    )

    destination = root / "output" / "1_julia"
    assert summary.exported == 1
    assert (destination / f"{image_id}.png").is_file()
    assert (destination / f"{image_id}.txt").read_text(encoding="utf-8") == "julia, portrait\n"


def test_export_manifest_copies_raw_image_when_processing_is_disabled(tmp_path):
    root = tmp_path / "project"
    raw_dir = root / "raw"
    captions = root / "work" / "captions"
    raw_dir.mkdir(parents=True)
    captions.mkdir(parents=True)

    raw_image = raw_dir / "001.png"
    Image.new("RGB", (80, 60), (20, 30, 40)).save(raw_image)
    caption_path = captions / "abc123.txt"
    caption_path.write_text("julia, portrait\n", encoding="utf-8")

    manifest = ProjectManifest(
        project_name="raw-export",
        subject="julia",
        trigger="julia",
    )
    record = ImageRecord(
        image_id="abc123",
        path="raw/001.png",
        status="captioned",
        caption_path="work/captions/abc123.txt",
    )
    manifest.add_image(record)

    summary = export_manifest(
        manifest,
        root=root,
        output_dir=root / "output",
        repeats=1,
        use_processed_images=False,
    )

    exported_image = root / "output" / "1_julia" / "abc123.png"
    exported_caption = root / "output" / "1_julia" / "abc123.txt"
    assert summary.exported == 1
    assert summary.failed == 0
    assert record.status == "exported"
    assert exported_image.read_bytes() == raw_image.read_bytes()
    assert exported_caption.read_text(encoding="utf-8") == "julia, portrait\n"
