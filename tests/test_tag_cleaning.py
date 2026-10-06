from argparse import Namespace

from lora_data_maker.cli import command_clean_tags, command_init
from lora_data_maker.dataset import DatasetProject
from lora_data_maker.models import ImageRecord
from lora_data_maker.paths import ProjectPaths
from lora_data_maker.taggers.filters import clean_tags, normalize_tag, normalize_tags
from lora_data_maker.taggers.clean_stage import clean_tag_manifest


def test_normalize_tag():
    assert normalize_tag("  Blue_Eyes  ") == "blue eyes"
    assert normalize_tag("1girl") == "1girl"


def test_normalize_tags_deduplicates_in_input_order():
    assert normalize_tags(
        ["Blue_Eyes", "blue eyes", "SMILE", "smile", "", "  "]
    ) == ["blue eyes", "smile"]


def test_clean_tags_filters_and_injects_subject_idempotently():
    tags = [
        "Blue_Eyes",
        "watermark",
        "1girl",
        "Julia",
        "julia",
        "  smile ",
        "logo",
    ]

    cleaned = clean_tags(
        tags,
        remove=["Watermark", "LOGO"],
        subject="Julia",
    )

    assert cleaned == ["julia", "blue eyes", "1girl", "smile"]

    assert clean_tags(
        cleaned,
        remove=["watermark", "logo"],
        subject="Julia",
    ) == cleaned


def test_clean_tag_manifest_keeps_tagged_status_and_records_identity():
    manifest = ProjectManifest(
        project_name="julia-set",
        subject="julia",
        trigger="julia trigger",
    )
    image = ImageRecord(
        path="raw/001.jpg",
        status="tagged",
        tags=["1girl", "watermark", "julia", "julia", "blue_eyes"],
    )
    manifest.add_image(image)

    summary = clean_tag_manifest(
        manifest,
        remove_tags=["watermark"],
    )

    record = manifest.images[image.image_id]
    assert summary.scanned == 1
    assert summary.cleaned == 1
    assert record.status == "tagged"
    assert record.tags == ["julia", "1girl", "blue eyes"]
    assert record.metadata["tag_clean_subject"] == "julia"
    assert record.metadata["tag_clean_trigger"] == "julia trigger"
    assert record.metadata["tag_clean_blacklist_removed"] == 1
    assert record.metadata["tag_clean_duplicates_removed"] == 1


def test_clean_tags_command_updates_manifest(tmp_path):
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
    image = ImageRecord(
        path="raw/001.jpg",
        status="tagged",
        tags=["1girl", "watermark", "julia", "julia"],
    )
    project.manifest.add_image(image)
    project.save()

    assert command_clean_tags(Namespace(path=str(root))) == 0

    refreshed = DatasetProject.open(ProjectPaths.from_root(root))
    record = next(iter(refreshed.manifest.images.values()))
    assert record.tags == ["julia", "1girl"]
    assert record.status == "tagged"
    assert refreshed.manifest.source["last_tag_clean_subject"] == "julia"
