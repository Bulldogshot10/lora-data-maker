from argparse import Namespace

from PIL import Image

from lora_data_maker.cli import command_init, command_validate
from lora_data_maker.dataset import DatasetProject
from lora_data_maker.paths import ProjectPaths


def test_validate_command_registers_and_updates_images(tmp_path):
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

    Image.new("RGB", (640, 480), (10, 20, 30)).save(
        root / "raw" / "valid.jpg",
        format="JPEG",
    )
    (root / "raw" / "broken.jpg").write_text(
        "not really an image",
        encoding="utf-8",
    )

    result = command_validate(Namespace(path=str(root)))
    assert result == 0

    project = DatasetProject.open(ProjectPaths.from_root(root))
    records = {
        record.path: record
        for record in project.manifest.images.values()
    }

    assert records["raw/valid.jpg"].status == "validated"
    assert records["raw/valid.jpg"].width == 640
    assert records["raw/broken.jpg"].status == "rejected"
    assert "unidentified image" in (records["raw/broken.jpg"].rejection_reason or "")
