from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterable
from pathlib import Path


BUILD_STAGES: tuple[str, ...] = (
    "scrape",
    "validate",
    "clean",
    "process",
    "tag",
    "clean-tags",
    "caption",
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
    from .cli import (
        command_caption,
        command_clean,
        command_clean_tags,
        command_process,
        command_scrape,
        command_tag,
        command_validate,
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

    commands = {
        "scrape": command_scrape,
        "validate": command_validate,
        "clean": command_clean,
        "process": command_process,
        "tag": command_tag,
        "clean-tags": command_clean_tags,
        "caption": command_caption,
    }

    scraper_config = config.get("scraper", {})
    if not isinstance(scraper_config, dict):
        raise ValueError("config section [scraper] must be a table")

    backend = str(scraper_config.get("backend", "google"))
    if backend not in {"google", "direct"}:
        raise ValueError(
            f"unsupported scraper backend {backend!r}; "
            "expected 'google' or 'direct'"
        )

    try:
        project = DatasetProject.open(paths)
    except Exception:
        project = None

    results: list[BuildStageResult] = []

    for index, stage in enumerate(selected, start=1):
        if project is None:
            before_counts = {}
        else:
            before_counts = project.manifest.counts()

        print(f"[{index}/{len(selected)}] {stage}")

        if stage == "scrape":
            args = __import__("argparse").Namespace(
                backend=backend,
                query=None,
                count=None,
                path=str(root),
            )
        else:
            args = __import__("argparse").Namespace(path=str(root))

        exit_code = int(commands[stage](args))

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

        if exit_code != 0:
            print(f"build stopped: {stage} exited with code {exit_code}")
            return BuildSummary(
                success=False,
                stages=tuple(results),
                started_at=started_at,
                finished_at=utc_now(),
            )

    print("build complete")
    return BuildSummary(
        success=True,
        stages=tuple(results),
        started_at=started_at,
        finished_at=utc_now(),
    )
