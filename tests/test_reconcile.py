from argparse import Namespace

from PIL import Image

from lora_data_maker.cli import command_init, command_reconcile, command_validate
from lora_data_maker.dataset import DatasetProject
from lora_data_maker.models import ImageRecord
from lora_data_maker.paths import ProjectPaths
from lora_data_maker.reconcile import reconcile_raw


def test_reconcile_adds_raw_files_without_touching_advanced_state(tmp_path):
    root = tmp_path / "project"
    raw = root / "raw"
    raw.mkdir(parents=True)

    Image.new("RGB", (16, 16)).save(raw / "001.jpg")
    Image.new("RGB", (16, 16)).save(raw / "002.jpg")

    manifest = DatasetProject(
        ProjectPaths.from_root(root),
        __import__("lora_data_maker.manifest", fromlist=["create_manifest"]).create_manifest(
            project_name="julia",
            subject="julia",
            trigger="julia",
        ),
    ).manifest
    advanced = ImageRecord(
        path="raw/001.jpg",
        status="tagged",
        tags=["julia", "portrait"],
    )
    manifest.add_image(advanced)

    summary = reconcile_raw(
        manifest,
        root=root,
        raw_dir=raw,
    )

    assert summary.scanned == 2
    assert summary.added == 1
    assert summary.existing == 1
    assert advanced.status == "tagged"
    assert any(
        record.path == "raw/002.jpg" and record.status == "discovered"
        for record in manifest.images.values()
    )


def test_reconcile_marks_missing_raw_without_deleting_record(tmp_path):
    root = tmp_path / "project"
    manifest = __import__("lora_data_maker.manifest", fromlist=["create_manifest"]).create_manifest(
        project_name="julia",
        subject="julia",
        trigger="julia",
    )
    record = ImageRecord(path="raw/missing.jpg", status="captioned")
    manifest.add_image(record)

    summary = reconcile_raw(
        manifest,
        root=root,
        raw_dir=root / "raw",
    )

    assert summary.missing == 1
    assert record.status == "captioned"
    assert record.metadata["reconcile_raw_missing"] is True


def test_validate_does_not_regress_completed_records(tmp_path):
    root = tmp_path / "project"
    command_init(
        Namespace(
            path=str(root),
            name="julia",
            subject="julia",
            trigger=None,
            force=False,
        )
    )

    raw = root / "raw"
    Image.new("RGB", (32, 32)).save(raw / "001.jpg")

    project = DatasetProject.open(ProjectPaths.from_root(root))
    record = ImageRecord(
        path="raw/001.jpg",
        status="captioned",
        tags=["julia"],
    )
    project.manifest.add_image(record)
    project.save()

    assert command_validate(Namespace(path=str(root))) == 0

    refreshed = DatasetProject.open(ProjectPaths.from_root(root))
    refreshed_record = next(iter(refreshed.manifest.images.values()))
    assert refreshed_record.status == "captioned"


def test_reconcile_command_persists_summary(tmp_path):
    root = tmp_path / "project"
    command_init(
        Namespace(
            path=str(root),
            name="julia",
            subject="julia",
            trigger=None,
            force=False,
        )
    )

    Image.new("RGB", (32, 32)).save(root / "raw" / "001.jpg")
    assert command_reconcile(Namespace(path=str(root))) == 0

    refreshed = DatasetProject.open(ProjectPaths.from_root(root))
    assert refreshed.manifest.source["last_reconcile_added"] == 1
    assert refreshed.manifest.source["last_reconcile_scanned"] == 1
