from argparse import Namespace

import pytest

from lora_data_maker.build import BUILD_STAGES, run_build, select_stages
from lora_data_maker.cli import command_init
from lora_data_maker.dataset import DatasetProject
from lora_data_maker.paths import ProjectPaths


def test_select_stages_and_skip():
    assert select_stages(
        from_stage="validate",
        to_stage="caption",
        skip=["tag"],
    ) == (
        "validate",
        "clean",
        "process",
        "clean-tags",
        "caption",
    )


def test_select_stages_rejects_reverse_range():
    with pytest.raises(ValueError, match="occurs after"):
        select_stages(from_stage="caption", to_stage="tag")


def test_run_build_runs_stages_in_order_and_records_summary(tmp_path, monkeypatch):
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

    import lora_data_maker.cli as cli

    calls = []

    def fake_command(name):
        def runner(args):
            calls.append((name, args.path))
            return 0

        return runner

    for stage in BUILD_STAGES:
        monkeypatch.setattr(cli, f"command_{stage.replace('-', '_')}", fake_command(stage))

    summary = run_build(
        path=root,
        skip=["scrape", "tag"],
    )

    assert summary.success is True
    assert [item.name for item in summary.stages] == [
        "validate",
        "clean",
        "process",
        "clean-tags",
        "caption",
        "export",
        "pack",
    ]
    assert [name for name, _ in calls] == [
        "validate",
        "clean",
        "process",
        "clean-tags",
        "caption",
        "export",
        "pack",
    ]

    project = DatasetProject.open(ProjectPaths.from_root(root))
    assert project.manifest.source["last_build_success"] is True
    assert project.manifest.source["last_build_selected_stages"] == [
        "validate",
        "clean",
        "process",
        "clean-tags",
        "caption",
    ]
    assert project.manifest.source["last_build_skipped_stages"] == ["scrape", "tag"]
    assert len(project.manifest.source["last_build_stages"]) == 7
    assert all(
        stage["status"] == "completed"
        for stage in project.manifest.source["last_build_stages"]
    )


def test_run_build_stops_on_failure_and_records_failed_stage(tmp_path, monkeypatch):
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

    import lora_data_maker.cli as cli

    calls = []

    def validate(args):
        calls.append("validate")
        return 1

    def clean(args):
        calls.append("clean")
        return 0

    monkeypatch.setattr(cli, "command_validate", validate)
    monkeypatch.setattr(cli, "command_clean", clean)

    summary = run_build(
        path=root,
        from_stage="validate",
        to_stage="clean",
    )

    assert summary.success is False
    assert summary.failed_stage == "validate"
    assert calls == ["validate"]

    project = DatasetProject.open(ProjectPaths.from_root(root))
    assert project.manifest.source["last_build_success"] is False
    assert project.manifest.source["last_build_failed_stage"] == "validate"
    assert project.manifest.source["last_build_stages"][0]["exit_code"] == 1
