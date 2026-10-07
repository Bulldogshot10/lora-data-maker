from argparse import Namespace
from pathlib import Path
from zipfile import ZipFile

from PIL import Image

from lora_data_maker.cli import (
    command_caption,
    command_clean,
    command_clean_tags,
    command_export,
    command_init,
    command_pack,
    command_process,
    command_reconcile,
    command_validate,
)
from lora_data_maker.dataset import DatasetProject
from lora_data_maker.models import ImageRecord
from lora_data_maker.paths import ProjectPaths


def test_v1_end_to_end_offline(tmp_path):
    root = tmp_path / "acceptance"

    assert command_init(
        Namespace(
            path=str(root),
            name="julia",
            subject="julia",
            trigger=None,
            force=False,
        )
    ) == 0

    Image.new("RGB", (640, 640)).save(root / "raw" / "001.jpg")
    assert command_reconcile(Namespace(path=str(root))) == 0
    assert command_validate(Namespace(path=str(root))) == 0
    config = (root / "config.toml").read_text(encoding="utf-8")
    config = config.replace("person_check = true", "person_check = false", 1)
    (root / "config.toml").write_text(config, encoding="utf-8")
    config = (root / "config.toml").read_text(encoding="utf-8")
    config = config.replace("person_check = true", "person_check = false", 1)
    (root / "config.toml").write_text(config, encoding="utf-8")
    assert command_clean(Namespace(path=str(root))) == 0
    assert command_process(Namespace(path=str(root))) == 0

    project = DatasetProject.open(ProjectPaths.from_root(root))
    record = next(iter(project.manifest.images.values()))
    record.tags = ["julia", "1girl", "blue_eyes"]
    record.set_status("tagged")
    project.save()

    assert command_clean_tags(Namespace(path=str(root))) == 0
    assert command_caption(Namespace(path=str(root))) == 0
    assert command_export(
        Namespace(path=str(root), repeats=10, folder_name=None)
    ) == 0
    assert command_pack(
        Namespace(path=str(root), archive=None, dataset_dir=None)
    ) == 0

    refreshed = DatasetProject.open(ProjectPaths.from_root(root))
    exported = next(iter(refreshed.manifest.images.values()))
    assert exported.status == "exported"
    assert exported.caption_path
    assert exported.metadata["export_layout"] == "kohya"

    dataset = root / "output" / "10_julia"
    assert (dataset / f"{exported.image_id}.jpg").is_file()
    assert (dataset / f"{exported.image_id}.txt").read_text(
        encoding="utf-8"
    ) == "julia, 1girl, blue eyes\n"

    archive = Path(refreshed.manifest.source["last_pack_archive"])
    assert archive.is_file()
    with ZipFile(archive) as bundle:
        names = set(bundle.namelist())
    assert f"output/10_julia/{exported.image_id}.jpg" in names
    assert f"output/10_julia/{exported.image_id}.txt" in names
    assert "manifest.json" in names
