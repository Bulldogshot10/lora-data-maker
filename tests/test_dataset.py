from argparse import Namespace

from lora_data_maker.cli import command_init
from lora_data_maker.dataset import DatasetProject
from lora_data_maker.paths import ProjectPaths


def test_project_create_and_open(tmp_path):
    args = Namespace(
        path=str(tmp_path / "julia"),
        name="julia",
        subject="julia",
        trigger=None,
        force=False,
    )

    assert command_init(args) == 0

    root = tmp_path / "julia"
    assert (root / "config.toml").is_file()
    assert (root / "manifest.json").is_file()

    project = DatasetProject.open(ProjectPaths.from_root(root))
    assert project.manifest.project_name == "julia"
    assert project.manifest.subject == "julia"
    assert project.manifest.trigger == "julia"
    assert len(project.manifest.images) == 0


def test_project_paths_reject_overlapping_directories(tmp_path):
    import pytest
    from lora_data_maker.paths import ProjectPaths

    with pytest.raises(ValueError, match="paths overlap"):
        ProjectPaths.from_root(
            tmp_path / "project",
            {
                "raw": "data",
                "work": "data/work",
            },
        )


def test_project_paths_reject_non_table_paths_config(tmp_path):
    import pytest
    from lora_data_maker.paths import ProjectPaths

    with pytest.raises(ValueError, match="paths.*table"):
        ProjectPaths.from_root(
            tmp_path / "project",
            "raw",
        )
