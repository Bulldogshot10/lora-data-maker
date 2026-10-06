from argparse import Namespace

from PIL import Image, ImageDraw

from lora_data_maker.cli import command_clean, command_init, command_process, command_validate
from lora_data_maker.dataset import DatasetProject
from lora_data_maker.exceptions import LoraDataMakerError
from lora_data_maker.paths import ProjectPaths


def _make_image(path):
    image = Image.new("RGB", (800, 640), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((200, 50, 600, 590), fill=(30, 120, 200))
    image.save(path, format="PNG")


def test_process_command_writes_work_and_preserves_raw(tmp_path):
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

    source = root / "raw" / "input.png"
    _make_image(source)
    raw_before = source.read_bytes()

    assert command_validate(Namespace(path=str(root))) == 0
    assert command_clean(Namespace(path=str(root))) == 0

    assert command_process(Namespace(path=str(root))) == 0

    project = DatasetProject.open(ProjectPaths.from_root(root))
    record = next(iter(project.manifest.images.values()))

    assert record.status == "processed"
    assert record.processed_path == f"work/processed/{record.image_id}.jpg"
    assert source.read_bytes() == raw_before
    assert (root / record.processed_path).is_file()

    with Image.open(root / record.processed_path) as output:
        assert output.size == (512, 512)
        assert output.mode == "RGB"
        assert output.format == "JPEG"

    assert record.metadata["processing_stage"] == "processed"
    assert record.metadata["processing_mode"] == "cover"


def test_process_command_rejects_unavailable_cli_face_crop(tmp_path):
    import pytest

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
    config = config.replace("face_crop = false", "face_crop = true", 1)
    (root / "config.toml").write_text(config, encoding="utf-8")

    with pytest.raises(LoraDataMakerError, match="face_crop"):
        command_process(Namespace(path=str(root)))


def test_process_command_returns_failure_when_any_image_fails(tmp_path):
    from lora_data_maker.models import ImageRecord

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
    Image.new("RGB", (640, 640)).save(root / "raw" / "valid.jpg")
    assert command_validate(Namespace(path=str(root))) == 0
    assert command_clean(Namespace(path=str(root))) == 0

    project = DatasetProject.open(ProjectPaths.from_root(root))
    project.manifest.add_image(
        ImageRecord(
            path="raw/missing.jpg",
            status="validated",
        )
    )
    project.save()

    assert command_process(Namespace(path=str(root))) == 1


def test_process_command_rejects_string_face_crop_boolean(tmp_path):
    import pytest

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
    config = config.replace("face_crop = false", 'face_crop = "false"', 1)
    (root / "config.toml").write_text(config, encoding="utf-8")

    with pytest.raises(Exception, match="processing.face_crop"):
        command_process(Namespace(path=str(root)))
