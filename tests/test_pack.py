from argparse import Namespace
from pathlib import Path
from zipfile import ZipFile

from PIL import Image

from lora_data_maker.cli import command_export, command_init, command_pack
from lora_data_maker.dataset import DatasetProject
from lora_data_maker.export import export_manifest
from lora_data_maker.models import ImageRecord
from lora_data_maker.pack import pack_dataset
from lora_data_maker.paths import ProjectPaths


def test_pack_dataset_contains_only_export_and_manifest(tmp_path):
    root = tmp_path / "project"
    dataset = root / "output" / "10_julia"
    dataset.mkdir(parents=True)
    (dataset / "abc123.jpg").write_bytes(b"image")
    (dataset / "abc123.txt").write_text("julia, portrait\n", encoding="utf-8")
    (root / "manifest.json").write_text("{\"project_name\": \"julia\"}\n", encoding="utf-8")

    archive = root / "output" / "julia.zip"
    summary = pack_dataset(
        root=root,
        dataset_dir=dataset,
        archive_path=archive,
    )

    assert summary.file_count == 3
    assert summary.size_bytes > 0
    assert len(summary.sha256) == 64

    with ZipFile(archive) as bundle:
        assert bundle.namelist() == [
            "manifest.json",
            "output/10_julia/abc123.jpg",
            "output/10_julia/abc123.txt",
        ] or bundle.namelist() == [
            "output/10_julia/abc123.jpg",
            "output/10_julia/abc123.txt",
            "manifest.json",
        ]


def test_pack_command_uses_last_export_and_records_archive(tmp_path):
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
    Image.new("RGB", (16, 16)).save(processed / f"{image_id}.jpg")
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
        )
    )
    project.save()

    command_export(Namespace(path=str(root), repeats=10, folder_name=None))
    assert command_pack(
        Namespace(path=str(root), archive=None, dataset_dir=None)
    ) == 0

    refreshed = DatasetProject.open(ProjectPaths.from_root(root))
    archive = Path(refreshed.manifest.source["last_pack_archive"])
    assert archive.is_file()
    assert refreshed.manifest.source["last_pack_file_count"] == 3

    with ZipFile(archive) as bundle:
        names = set(bundle.namelist())
    assert "manifest.json" in names
    assert f"output/10_julia/{image_id}.jpg" in names
    assert f"output/10_julia/{image_id}.txt" in names


def test_pack_rejects_archive_inside_dataset(tmp_path):
    import pytest

    root = tmp_path / "project"
    dataset = root / "output" / "10_julia"
    dataset.mkdir(parents=True)
    (dataset / "abc.jpg").write_bytes(b"image")
    (root / "manifest.json").write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="outside the exported dataset"):
        pack_dataset(
            root=root,
            dataset_dir=dataset,
            archive_path=dataset / "bundle.zip",
        )
