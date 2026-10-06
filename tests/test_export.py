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
