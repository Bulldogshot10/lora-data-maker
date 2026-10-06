from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

from . import __version__
from .config import load_config, write_default_config
from .dataset import DatasetProject
from .exceptions import LoraDataMakerError
from .logging import configure_logging
from .paths import ProjectPaths, find_project_root


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="lora-data-maker",
        description="Termux-first toolkit for LoRA image dataset preparation.",
    )
    parser.add_argument("--version", action="version", version=__version__)
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="enable debug logging"
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    init_parser = subparsers.add_parser(
        "init", help="initialize a dataset project"
    )
    init_parser.add_argument(
        "path",
        nargs="?",
        default=".",
        help="project directory (default: current directory)",
    )
    init_parser.add_argument("--name", help="dataset/project name")
    init_parser.add_argument("--subject", help="dataset subject/identity")
    init_parser.add_argument(
        "--trigger", help="trigger token; defaults to subject"
    )
    init_parser.add_argument(
        "--force",
        action="store_true",
        help="replace an existing config.toml",
    )

    info_parser = subparsers.add_parser(
        "info", help="show project information"
    )
    info_parser.add_argument(
        "path",
        nargs="?",
        default=".",
        help="project directory or child path (default: current directory)",
    )

    return parser


def command_init(args: argparse.Namespace) -> int:
    root = Path(args.path).expanduser().resolve()
    paths = ProjectPaths.from_root(root)

    if paths.config.exists() and not args.force:
        raise LoraDataMakerError(
            f"project already initialized at {root}; "
            "use --force to replace config.toml"
        )

    paths.create_directories()
    write_default_config(
        paths.config,
        name=args.name,
        subject=args.subject,
        trigger=args.trigger,
    )

    project = DatasetProject.create(paths)

    print(f"initialized project: {root}")
    print(f"config: {paths.config}")
    print(f"manifest: {project.manifest_path}")
    print("directories: raw/ work/ output/ cache/")
    return 0


def command_info(args: argparse.Namespace) -> int:
    root = find_project_root(args.path)
    config_path = root / "config.toml"
    config = load_config(config_path)

    project_config = config.get("project", {})
    paths_config = config.get("paths", {})
    paths = ProjectPaths.from_root(root, paths_config)
    project = DatasetProject.open(paths)
    counts = project.counts()

    print(f"name: {project.manifest.project_name}")
    print(f"subject: {project.manifest.subject}")
    print(f"trigger: {project.manifest.trigger}")
    print(f"root: {root}")
    print(f"manifest: {project.manifest_path}")
    print(f"images: {len(project.manifest.images)}")
    print(f"status: {counts}")
    print(f"raw: {paths.raw}")
    print(f"work: {paths.work}")
    print(f"output: {paths.output}")
    print(f"cache: {paths.cache}")

    if project_config and not isinstance(project_config, dict):
        raise LoraDataMakerError("config section [project] must be a table")

    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logger = configure_logging(args.verbose)

    try:
        if args.command == "init":
            return command_init(args)
        if args.command == "info":
            return command_info(args)
    except (LoraDataMakerError, FileNotFoundError, OSError, ValueError, KeyError) as exc:
        logger.error(str(exc))
        return 1

    parser.error(f"unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
