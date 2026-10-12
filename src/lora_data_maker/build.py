from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


BUILD_STAGES: tuple[str, ...] = (
    "scrape",
    "validate",
    "clean",
    "process",
    "tag",
    "clean-tags",
    "caption",
    "export",
    "pack",
)


@dataclass(frozen=True)
class BuildStageResult:
    name: str
    status: str
    exit_code: int
    before_counts: dict[str, int]
    after_counts: dict[str, int]


@dataclass(frozen=True)
class BuildSummary:
    success: bool
    stages: tuple[BuildStageResult, ...]
    started_at: str
    finished_at: str

    @property
    def failed_stage(self) -> str | None:
        for stage in self.stages:
            if stage.status == "failed":
                return stage.name
        return None


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def normalize_stage_names(stages: Iterable[str]) -> tuple[str, ...]:
    normalized: list[str] = []
    seen: set[str] = set()

    for stage in stages:
        name = str(stage).strip().lower()
        if not name:
            continue
        if name not in BUILD_STAGES:
            raise ValueError(
                f"unknown build stage {name!r}; "
                f"expected one of {', '.join(BUILD_STAGES)}"
            )
        if name not in seen:
            seen.add(name)
            normalized.append(name)

    return tuple(normalized)


def select_stages(
    *,
    from_stage: str = BUILD_STAGES[0],
    to_stage: str = BUILD_STAGES[-1],
    skip: Iterable[str] = (),
) -> tuple[str, ...]:
    start = str(from_stage).strip().lower()
    end = str(to_stage).strip().lower()

    if start not in BUILD_STAGES:
        raise ValueError(f"unknown build stage {from_stage!r}")
    if end not in BUILD_STAGES:
        raise ValueError(f"unknown build stage {to_stage!r}")

    start_index = BUILD_STAGES.index(start)
    end_index = BUILD_STAGES.index(end)
    if start_index > end_index:
        raise ValueError(
            f"build from-stage {start!r} occurs after to-stage {end!r}"
        )

    skipped = set(normalize_stage_names(skip))
    return tuple(
        stage
        for stage in BUILD_STAGES[start_index : end_index + 1]
        if stage not in skipped
    )


def run_build(
    *,
    path: Path | str = ".",
    from_stage: str = BUILD_STAGES[0],
    to_stage: str = BUILD_STAGES[-1],
    skip: Iterable[str] = (),
) -> BuildSummary:
    """Run selected pipeline stages in order.

    Existing stage implementations provide the resume behavior by operating
    only on records eligible for their current state.
    """
    from .scrapers import SCRAPER_BACKENDS
    from .cli import (
        command_caption,
        command_clean,
        command_clean_tags,
        command_process,
        command_scrape,
        command_tag,
        command_validate,
        command_export,
        command_pack,
    )
    from .config import load_config
    from .dataset import DatasetProject
    from .paths import ProjectPaths, find_project_root

    started_at = utc_now()
    root = find_project_root(path)
    config = load_config(root / "config.toml")
    paths = ProjectPaths.from_root(root, config.get("paths", {}))
    selected = select_stages(
        from_stage=from_stage,
        to_stage=to_stage,
        skip=skip,
    )

    processing_config = config.get("processing", {})
    if not isinstance(processing_config, dict):
        raise ValueError("config section [processing] must be a table")
    processing_enabled = processing_config.get("enabled", True)
    if not isinstance(processing_enabled, bool):
        raise ValueError("config field processing.enabled must be a boolean")
    if not processing_enabled:
        selected = tuple(stage for stage in selected if stage != "process")

    caption_config = config.get("caption", {})
    if not isinstance(caption_config, dict):
        raise ValueError("config section [caption] must be a table")
    caption_backend = str(
        caption_config.get("backend", "deterministic")
    ).strip().lower()
    if caption_backend not in {"deterministic", "vlm"}:
        raise ValueError(
            "caption.backend must be 'deterministic' or 'vlm'"
        )

    tagger_config = config.get("tagger", {})
    if not isinstance(tagger_config, dict):
        raise ValueError("config section [tagger] must be a table")
    tagger_disabled = (
        str(tagger_config.get("preset", "sweetspot")).strip().lower() == "off"
        or str(tagger_config.get("backend", "wd14")).strip().lower() == "off"
    )

    if caption_backend == "vlm" or tagger_disabled:
        selected = tuple(
            stage for stage in selected
            if stage not in {"tag", "clean-tags"}
        )

    commands = {
        "scrape": command_scrape,
        "validate": command_validate,
        "clean": command_clean,
        "process": command_process,
        "tag": command_tag,
        "clean-tags": command_clean_tags,
        "caption": command_caption,
        "export": command_export,
        "pack": command_pack,
    }

    backend = "duckduckgo"
    if "scrape" in selected:
        scraper_config = config.get("scraper", {})
        if not isinstance(scraper_config, dict):
            raise ValueError("config section [scraper] must be a table")

        backend = str(scraper_config.get("backend", "duckduckgo"))
        if backend not in SCRAPER_BACKENDS:
            raise ValueError(
                f"unsupported scraper backend {backend!r}; "
                "expected one of: " + ", ".join(SCRAPER_BACKENDS)
            )

    project = DatasetProject.open(paths)
    results: list[BuildStageResult] = []

    project.manifest.source.update(
        {
            "last_build_started_at": started_at,
            "last_build_selected_stages": list(selected),
            "last_build_skipped_stages": [
                stage for stage in BUILD_STAGES
                if stage not in selected
                and BUILD_STAGES.index(stage)
                >= BUILD_STAGES.index(str(from_stage).strip().lower())
                and BUILD_STAGES.index(stage)
                <= BUILD_STAGES.index(str(to_stage).strip().lower())
            ],
        }
    )
    project.save()

    for index, stage in enumerate(selected, start=1):
        project = DatasetProject.open(paths)
        before_counts = project.manifest.counts()

        print(f"[{index}/{len(selected)}] {stage}")

        if stage == "scrape":
            args = argparse.Namespace(
                backend=backend,
                query=None,
                count=None,
                path=str(root),
            )
        elif stage == "export":
            args = argparse.Namespace(
                path=str(root),
                repeats=None,
                folder_name=None,
            )
        elif stage == "pack":
            args = argparse.Namespace(
                path=str(root),
                archive=None,
                dataset_dir=None,
            )
        else:
            args = argparse.Namespace(path=str(root))

        stage_error = None
        try:
            exit_code = int(commands[stage](args))
        except Exception as exc:
            stage_error = str(exc)
            print(f"build stage error: {stage}: {exc}")
            exit_code = 1

        try:
            project = DatasetProject.open(paths)
            after_counts = project.manifest.counts()
        except Exception:
            after_counts = {}

        status = "completed" if exit_code == 0 else "failed"
        results.append(
            BuildStageResult(
                name=stage,
                status=status,
                exit_code=exit_code,
                before_counts=before_counts,
                after_counts=after_counts,
            )
        )

        project = DatasetProject.open(paths)
        project.manifest.source["last_build_stages"] = [
            {
                "name": item.name,
                "status": item.status,
                "exit_code": item.exit_code,
                "before_counts": item.before_counts,
                "after_counts": item.after_counts,
            }
            for item in results
        ]
        project.manifest.source["last_build_finished_at"] = utc_now()
        project.manifest.source["last_build_success"] = exit_code == 0
        if stage_error is not None:
            project.manifest.source["last_build_stage_error"] = stage_error
        elif exit_code == 0:
            project.manifest.source.pop("last_build_stage_error", None)
        if exit_code != 0:
            project.manifest.source["last_build_failed_stage"] = stage
        elif "last_build_failed_stage" in project.manifest.source:
            project.manifest.source.pop("last_build_failed_stage", None)
        project.save()

        if exit_code != 0:
            print(f"build stopped: {stage} exited with code {exit_code}")
            return BuildSummary(
                success=False,
                stages=tuple(results),
                started_at=started_at,
                finished_at=utc_now(),
            )

    project = DatasetProject.open(paths)
    project.manifest.source["last_build_stages"] = [
        {
            "name": item.name,
            "status": item.status,
            "exit_code": item.exit_code,
            "before_counts": item.before_counts,
            "after_counts": item.after_counts,
        }
        for item in results
    ]
    project.manifest.source["last_build_finished_at"] = utc_now()
    project.manifest.source["last_build_success"] = True
    project.manifest.source.pop("last_build_failed_stage", None)
    project.save()

    print("build complete")
    return BuildSummary(
        success=True,
        stages=tuple(results),
        started_at=started_at,
        finished_at=utc_now(),
    )
