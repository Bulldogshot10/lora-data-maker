from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter

from lora_data_maker.models import ImageRecord, ProjectManifest
from lora_data_maker.processing.clean import clean_manifest


def _make_subject_image(path: Path, *, blur: float = 0.0, variant: int = 0):
    image = Image.new("RGB", (640, 640), "white")
    draw = ImageDraw.Draw(image)
    draw.ellipse((100, 100, 540, 540), fill=(80, 120, 180))
    if variant == 0:
        draw.rectangle((240, 180, 400, 460), fill=(220, 80, 80))
    else:
        draw.polygon(
            ((180, 420), (320, 140), (460, 420)),
            fill=(70, 180, 100),
        )
    if blur:
        image = image.filter(ImageFilter.GaussianBlur(blur))
    image.save(path, format="PNG")


def _manifest_for(*paths: str) -> ProjectManifest:
    manifest = ProjectManifest(project_name="example", subject="example", trigger="example")
    for path in paths:
        manifest.add_image(ImageRecord(path=path, status="validated"))
    return manifest


def test_clean_exact_duplicate_and_quality_filter(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()

    _make_subject_image(raw / "001.png")
    (raw / "002.png").write_bytes((raw / "001.png").read_bytes())
    _make_subject_image(raw / "003.png", blur=8)
    _make_subject_image(raw / "004.png", variant=1)

    manifest = _manifest_for(
        "raw/001.png",
        "raw/002.png",
        "raw/003.png",
        "raw/004.png",
    )

    summary = clean_manifest(
        manifest,
        root=tmp_path,
        exact=True,
        perceptual=True,
        perceptual_distance=6,
        min_width=512,
        min_height=512,
        blur_threshold=10,
    )

    records = {image.path: image for image in manifest.images.values()}

    assert summary.scanned == 4
    assert summary.exact_duplicates == 1
    assert summary.quality_rejected == 1
    assert records["raw/001.png"].status == "validated"
    assert records["raw/002.png"].status == "duplicate"
    assert records["raw/003.png"].status == "rejected"
    assert records["raw/004.png"].status == "validated"
    assert records["raw/002.png"].metadata["duplicate_kind"] == "exact"
    assert records["raw/003.png"].metadata["quality_stage"] == "rejected"


def test_clean_command_is_successful_when_all_records_are_already_advanced(tmp_path):
    from argparse import Namespace
    from lora_data_maker.cli import command_clean, command_init
    from lora_data_maker.dataset import DatasetProject
    from lora_data_maker.models import ImageRecord
    from lora_data_maker.paths import ProjectPaths

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

    project = DatasetProject.open(ProjectPaths.from_root(root))
    project.manifest.add_image(
        ImageRecord(
            path="raw/a.jpg",
            status="exported",
        )
    )
    project.save()

    assert command_clean(Namespace(path=str(root))) == 0


def test_clean_quality_rejection_happens_before_perceptual_dedup(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()

    sharp = raw / "001.png"
    blurry = raw / "002.png"
    _make_subject_image(sharp)
    _make_subject_image(blurry, blur=8)

    manifest = _manifest_for("raw/001.png", "raw/002.png")

    summary = clean_manifest(
        manifest,
        root=tmp_path,
        exact=False,
        perceptual=True,
        perceptual_distance=6,
        min_width=512,
        min_height=512,
        blur_threshold=10,
    )

    records = {image.path: image for image in manifest.images.values()}
    assert records["raw/001.png"].status == "validated"
    assert records["raw/002.png"].status == "rejected"
    assert summary.perceptual_duplicates == 0
    assert summary.quality_rejected == 1


def test_clean_command_rejects_string_boolean_config(tmp_path):
    import pytest
    from argparse import Namespace
    from lora_data_maker.cli import command_init, command_clean

    root = tmp_path / "project"
    command_init(
        Namespace(
            path=str(root),
            name="example",
            subject="example",
            trigger=None,
            force=False,
        )
    )
    config = (root / "config.toml").read_text(encoding="utf-8")
    config = config.replace("exact = true", 'exact = "false"', 1)
    (root / "config.toml").write_text(config, encoding="utf-8")

    with pytest.raises(Exception, match="dedup.exact"):
        command_clean(Namespace(path=str(root)))


def test_clean_command_keeps_legacy_person_gate_opt_in(tmp_path, monkeypatch):
    from argparse import Namespace
    from lora_data_maker.cli import command_clean, command_init
    from lora_data_maker.paths import ProjectPaths

    root = tmp_path / "project"
    command_init(
        Namespace(
            path=str(root),
            name="legacy",
            subject="legacy",
            trigger=None,
            force=False,
        )
    )

    image = root / "raw" / "001.png"
    _make_subject_image(image)

    project = __import__("lora_data_maker.dataset", fromlist=["DatasetProject"]).DatasetProject.open(
        ProjectPaths.from_root(root)
    )
    project.manifest.add_image(
        ImageRecord(path="raw/001.png", status="validated")
    )
    project.save()

    config_path = root / "config.toml"
    config = config_path.read_text(encoding="utf-8")
    config = config.replace("person_check = false\n", "", 1)
    config = config.replace("text_check = true\n", "text_check = false\n", 1)
    config_path.write_text(config, encoding="utf-8")

    class UnexpectedDetector:
        def __init__(self, *args, **kwargs):
            raise AssertionError("legacy config unexpectedly enabled person gate")

    monkeypatch.setattr(
        "lora_data_maker.cli.YOLOXNanoPersonDetector",
        UnexpectedDetector,
    )

    assert command_clean(Namespace(path=str(root))) == 0


def test_clean_manifest_rejects_pillow_image_decode_failure(tmp_path, monkeypatch):
    from lora_data_maker.processing import clean as clean_module
    from PIL.Image import DecompressionBombError

    manifest = _manifest_for("raw/bad.png")

    def broken_quality(*args, **kwargs):
        raise DecompressionBombError("image too large")

    monkeypatch.setattr(clean_module, "evaluate_quality", broken_quality)

    summary = clean_module.clean_manifest(
        manifest,
        root=tmp_path,
        exact=False,
        perceptual=False,
    )

    record = manifest.images[next(iter(manifest.images))]
    assert summary.quality_rejected == 1
    assert record.status == "rejected"
    assert record.rejection_reason == "quality check could not read image"
