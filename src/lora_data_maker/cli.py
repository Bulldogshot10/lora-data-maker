from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

from . import __version__
from .build import BUILD_STAGES, run_build
from .captions import CaptionConfig, caption_manifest
from .export import export_manifest, kohya_folder_name, safe_component
from .config import load_config, write_default_config
from .dataset import DatasetProject
from .exceptions import LoraDataMakerError
from .logging import configure_logging
from .images import iter_image_files
from .manifest import find_image_by_source_url, register_image
from .pack import pack_dataset
from .paths import ProjectPaths, find_project_root, manifest_path
from .reconcile import reconcile_raw
from .processing import (
    ProcessConfig,
    YuNetFaceDetector,
    clean_manifest,
    process_manifest,
    validate_image,
)
from .scrapers import SCRAPER_BACKENDS, ScraperError, ScraperManager
from .taggers import (
    TAGGER_PRESETS,
    TaggerError,
    build_tagger,
    clean_tag_manifest,
    resolve_tagger_preset,
    tag_manifest,
)


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
        help="replace config.toml and recreate manifest.json",
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

    scrape_parser = subparsers.add_parser(
        "scrape", help="search for and download images into the project"
    )
    scrape_parser.add_argument(
        "backend",
        choices=SCRAPER_BACKENDS,
        help="scraper backend",
    )
    scrape_parser.add_argument(
        "query",
        nargs="?",
        help="search query for google/duckduckgo/chromium, or a text file of URLs for direct",
    )
    scrape_parser.add_argument(
        "--count",
        type=int,
        default=None,
        help="number of successfully downloaded images (default: config)",
    )
    scrape_parser.add_argument(
        "--path",
        default=".",
        help="project directory (default: current directory)",
    )

    validate_parser = subparsers.add_parser(
        "validate", help="validate image files in the project's raw directory"
    )
    validate_parser.add_argument(
        "--path",
        default=".",
        help="project directory (default: current directory)",
    )

    reconcile_parser = subparsers.add_parser(
        "reconcile", help="reconcile raw image files into the manifest"
    )
    reconcile_parser.add_argument(
        "--path",
        default=".",
        help="project directory (default: current directory)",
    )

    clean_parser = subparsers.add_parser(
        "clean", help="deduplicate and quality-filter validated images"
    )
    clean_parser.add_argument(
        "--path",
        default=".",
        help="project directory (default: current directory)",
    )

    process_parser = subparsers.add_parser(
        "process", help="resize, crop, orient, and export validated images to work/"
    )
    process_parser.add_argument(
        "--path",
        default=".",
        help="project directory (default: current directory)",
    )

    tag_parser = subparsers.add_parser(
        "tag", help="run the configured image tagger on processed images"
    )
    tag_parser.add_argument(
        "--path",
        default=".",
        help="project directory (default: current directory)",
    )
    tag_parser.add_argument(
        "--preset",
        choices=tuple(TAGGER_PRESETS),
        default=None,
        help="tagger preset (default: config, otherwise sweetspot)",
    )

    clean_tags_parser = subparsers.add_parser(
        "clean-tags", help="normalize, filter, and inject the dataset subject into tags"
    )
    clean_tags_parser.add_argument(
        "--path",
        default=".",
        help="project directory (default: current directory)",
    )

    caption_parser = subparsers.add_parser(
        "caption", help="write deterministic caption files from cleaned tags"
    )
    caption_parser.add_argument(
        "--path",
        default=".",
        help="project directory (default: current directory)",
    )

    build_parser = subparsers.add_parser(
        "build", help="run the dataset preparation pipeline"
    )
    build_parser.add_argument(
        "--path",
        default=".",
        help="project directory (default: current directory)",
    )
    build_parser.add_argument(
        "--from-stage",
        choices=BUILD_STAGES,
        default=BUILD_STAGES[0],
        help="first stage to run (default: scrape)",
    )
    build_parser.add_argument(
        "--to-stage",
        choices=BUILD_STAGES,
        default=BUILD_STAGES[-1],
        help="last stage to run (default: pack)",
    )
    build_parser.add_argument(
        "--skip",
        action="append",
        choices=BUILD_STAGES,
        default=[],
        help="skip a stage; may be repeated",
    )

    export_parser = subparsers.add_parser(
        "export", help="export captioned images as a Kohya-style dataset"
    )
    export_parser.add_argument(
        "--path",
        default=".",
        help="project directory (default: current directory)",
    )
    export_parser.add_argument(
        "--repeats",
        type=int,
        default=None,
        help="Kohya folder repeat count (default: config)",
    )
    export_parser.add_argument(
        "--folder-name",
        default=None,
        help="explicit output dataset folder name",
    )

    pack_parser = subparsers.add_parser(
        "pack", help="package the exported dataset and manifest into a ZIP"
    )
    pack_parser.add_argument(
        "--path",
        default=".",
        help="project directory (default: current directory)",
    )
    pack_parser.add_argument(
        "--archive",
        default=None,
        help="archive path (default: output/<project-name>.zip)",
    )
    pack_parser.add_argument(
        "--dataset-dir",
        default=None,
        help="exported dataset directory; defaults to the last export",
    )

    web_parser = subparsers.add_parser(
        "web", help="start the localhost WebUI"
    )
    web_parser.add_argument(
        "--path",
        default=".",
        help="dataset project directory (default: current directory)",
    )
    web_parser.add_argument("--host", default="127.0.0.1")
    web_parser.add_argument("--port", type=int, default=8787)

    return parser


def command_init(args: argparse.Namespace) -> int:
    root = Path(args.path).expanduser().resolve()
    paths = ProjectPaths.from_root(root)

    if paths.config.exists() and not args.force:
        raise LoraDataMakerError(
            f"project already initialized at {root}; "
            "use --force to replace config.toml and recreate manifest.json"
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


def command_validate(args: argparse.Namespace) -> int:
    root = find_project_root(args.path)
    config = load_config(root / "config.toml")
    paths = ProjectPaths.from_root(root, config.get("paths", {}))
    project = DatasetProject.open(paths)

    validation_config = config.get("validation", {})
    if not isinstance(validation_config, dict):
        raise LoraDataMakerError("config section [validation] must be a table")

    min_width = int(validation_config.get("min_width", 1))
    min_height = int(validation_config.get("min_height", 1))
    max_width = int(validation_config.get("max_width", 0)) or None
    max_height = int(validation_config.get("max_height", 0)) or None
    min_file_size = int(validation_config.get("min_file_size", 1))
    max_file_size = int(validation_config.get("max_file_size", 0)) or None

    reconcile = reconcile_raw(
        project.manifest,
        root=root,
        raw_dir=paths.raw,
    )

    existing = {
        path: record
        for path, record in (
            (record.path.replace("\\", "/"), record)
            for record in project.manifest.images.values()
        )
        if path
    }

    scanned = valid_count = rejected_count = skipped_count = 0
    for image_path in iter_image_files(paths.raw):
        relative_path = manifest_path(root, image_path)
        record = existing.get(relative_path)
        if record is None:
            record = register_image(
                project.manifest,
                path=relative_path,
                status="discovered",
            )
            existing[relative_path] = record

        if record.status not in {"discovered", "downloaded"}:
            skipped_count += 1
            continue

        scanned += 1

        result = validate_image(
            image_path,
            min_width=min_width,
            min_height=min_height,
            max_width=max_width,
            max_height=max_height,
            min_file_size=min_file_size,
            max_file_size=max_file_size,
        )

        record.width = result.width
        record.height = result.height
        record.file_size = result.file_size
        record.mime_type = result.mime_type
        if result.metadata:
            record.metadata.update(result.metadata)
        if result.format:
            record.metadata["format"] = result.format
        if result.exif_orientation is not None:
            record.metadata["exif_orientation"] = result.exif_orientation
        record.metadata["has_exif"] = result.has_exif

        if result.valid:
            record.set_status("validated")
            valid_count += 1
        else:
            record.set_status("rejected", reason=result.reason or "validation failed")
            rejected_count += 1

    project.manifest.source.update(
        {
            "last_validation_scanned": scanned,
            "last_validation_valid": valid_count,
            "last_validation_rejected": rejected_count,
            "last_validation_skipped": skipped_count,
            "last_validation_reconcile_added": reconcile.added,
            "last_validation_reconcile_missing": reconcile.missing,
        }
    )
    project.save()

    print(f"scanned: {scanned}")
    print(f"valid: {valid_count}")
    print(f"rejected: {rejected_count}")
    print(f"skipped: {skipped_count}")
    print(f"raw: {paths.raw}")

    return 0 if scanned or skipped_count else 1


def command_reconcile(args: argparse.Namespace) -> int:
    root = find_project_root(args.path)
    config = load_config(root / "config.toml")
    paths = ProjectPaths.from_root(root, config.get("paths", {}))
    project = DatasetProject.open(paths)

    summary = reconcile_raw(
        project.manifest,
        root=root,
        raw_dir=paths.raw,
    )

    project.manifest.source.update(
        {
            "last_reconcile_scanned": summary.scanned,
            "last_reconcile_added": summary.added,
            "last_reconcile_existing": summary.existing,
            "last_reconcile_missing": summary.missing,
        }
    )
    project.save()

    print(f"scanned: {summary.scanned}")
    print(f"added: {summary.added}")
    print(f"existing: {summary.existing}")
    print(f"missing: {summary.missing}")
    print(f"raw: {paths.raw}")

    return 0 if summary.scanned or summary.missing else 1


def command_clean(args: argparse.Namespace) -> int:
    root = find_project_root(args.path)
    config = load_config(root / "config.toml")
    paths = ProjectPaths.from_root(root, config.get("paths", {}))
    project = DatasetProject.open(paths)

    dedup_config = config.get("dedup", {})
    quality_config = config.get("quality", {})
    processing_config = config.get("processing", {})

    if not isinstance(dedup_config, dict):
        raise LoraDataMakerError("config section [dedup] must be a table")
    if not isinstance(quality_config, dict):
        raise LoraDataMakerError("config section [quality] must be a table")
    if not isinstance(processing_config, dict):
        raise LoraDataMakerError("config section [processing] must be a table")
    for key in ("exact", "perceptual"):
        if key in dedup_config and not isinstance(dedup_config[key], bool):
            raise LoraDataMakerError(
                f"config field dedup.{key} must be a boolean"
            )

    summary = clean_manifest(
        project.manifest,
        root=root,
        exact=dedup_config.get("exact", True),
        perceptual=dedup_config.get("perceptual", True),
        perceptual_distance=int(dedup_config.get("perceptual_distance", 6)),
        min_width=int(quality_config.get("min_width", 512)),
        min_height=int(quality_config.get("min_height", 512)),
        blur_threshold=float(
            quality_config.get(
                "blur_threshold",
                processing_config.get("blur_threshold", 0.0),
            )
        ),
        max_aspect_ratio=float(quality_config.get("max_aspect_ratio", 0.0)),
    )

    project.manifest.source.update(
        {
            "last_clean_scanned": summary.scanned,
            "last_clean_exact_duplicates": summary.exact_duplicates,
            "last_clean_perceptual_duplicates": summary.perceptual_duplicates,
            "last_clean_quality_rejected": summary.quality_rejected,
            "last_clean_kept": summary.kept,
        }
    )
    project.save()

    print(f"scanned: {summary.scanned}")
    print(f"exact duplicates: {summary.exact_duplicates}")
    print(f"perceptual duplicates: {summary.perceptual_duplicates}")
    print(f"quality rejected: {summary.quality_rejected}")
    print(f"kept: {summary.kept}")

    return 0 if summary.scanned or bool(project.manifest.images) else 1


def command_process(args: argparse.Namespace) -> int:
    root = find_project_root(args.path)
    config = load_config(root / "config.toml")
    paths = ProjectPaths.from_root(root, config.get("paths", {}))
    project = DatasetProject.open(paths)

    processing_config = config.get("processing", {})
    if not isinstance(processing_config, dict):
        raise LoraDataMakerError("config section [processing] must be a table")

    target_size = int(processing_config.get("target_size", 512))
    target_width = int(processing_config.get("target_width", target_size))
    target_height = int(processing_config.get("target_height", target_size))

    resize_mode = str(processing_config.get("resize_mode", "cover")).lower()
    jpeg_quality = int(processing_config.get("jpeg_quality", 95))
    face_crop = processing_config.get("face_crop", False)
    if not isinstance(face_crop, bool):
        raise LoraDataMakerError("config field processing.face_crop must be a boolean")
    face_padding = float(processing_config.get("face_padding", 1.75))

    face_detector = None
    if face_crop:
        try:
            face_detector = YuNetFaceDetector(
                cache_dir=paths.cache / "face",
            )
        except RuntimeError as exc:
            raise LoraDataMakerError(str(exc)) from exc
        print("face detector: yunet_2023mar")

    settings = ProcessConfig(
        target_width=target_width,
        target_height=target_height,
        resize_mode=resize_mode,
        jpeg_quality=jpeg_quality,
        face_crop=face_crop,
        face_padding=face_padding,
    )

    processed, failed, skipped = process_manifest(
        project.manifest,
        root=root,
        output_dir=paths.work,
        config=settings,
        face_detector=face_detector,
    )

    project.manifest.source.update(
        {
            "last_process_processed": processed,
            "last_process_failed": failed,
            "last_process_skipped": skipped,
            "last_process_target": [target_width, target_height],
            "last_process_resize_mode": resize_mode,
        }
    )
    project.save()

    print(f"processed: {processed}")
    print(f"failed: {failed}")
    print(f"skipped: {skipped}")
    print(f"work: {paths.work / 'processed'}")

    return 0 if failed == 0 and (processed or skipped) else 1


def command_tag(args: argparse.Namespace) -> int:
    root = find_project_root(args.path)
    config = load_config(root / "config.toml")
    paths = ProjectPaths.from_root(root, config.get("paths", {}))
    project = DatasetProject.open(paths)

    tagger_config = config.get("tagger", {})
    if not isinstance(tagger_config, dict):
        raise LoraDataMakerError("config section [tagger] must be a table")

    selected_preset = getattr(args, "preset", None) or str(
        tagger_config.get("preset", "sweetspot")
    )
    resolved_preset = resolve_tagger_preset(selected_preset)

    cache_root = (
        Path.home()
        / ".cache"
        / "lora-data-maker"
        / "taggers"
        / ("joytag" if resolved_preset.kind == "joytag" else "wd14")
    )
    tagger = build_tagger(
        tagger_config,
        cache_root=cache_root,
        preset=resolved_preset.name,
    )

    summary = tag_manifest(
        project.manifest,
        root=root,
        tagger=tagger,
    )

    project.manifest.source.update(
        {
            "last_tag_scanned": summary.scanned,
            "last_tag_tagged": summary.tagged,
            "last_tag_failed": summary.failed,
            "last_tag_skipped": summary.skipped,
            "last_tag_backend": str(tagger_config.get("backend", "wd14")),
            "last_tag_preset": resolved_preset.name,
            "last_tag_model": resolved_preset.repo_id,
            "last_tag_revision": resolved_preset.revision,
        }
    )
    project.save()

    print(f"scanned: {summary.scanned}")
    print(f"tagged: {summary.tagged}")
    print(f"failed: {summary.failed}")
    print(f"skipped: {summary.skipped}")

    return 0 if summary.tagged or summary.skipped else 1


def command_clean_tags(args: argparse.Namespace) -> int:
    root = find_project_root(args.path)
    config = load_config(root / "config.toml")
    paths = ProjectPaths.from_root(root, config.get("paths", {}))
    project = DatasetProject.open(paths)

    tag_rules = config.get("tag_rules", {})
    if not isinstance(tag_rules, dict):
        raise LoraDataMakerError("config section [tag_rules] must be a table")

    remove_tags = tag_rules.get("remove", [])
    if not isinstance(remove_tags, list) or not all(
        isinstance(tag, str) for tag in remove_tags
    ):
        raise LoraDataMakerError(
            "config field tag_rules.remove must be an array of strings"
        )

    summary = clean_tag_manifest(
        project.manifest,
        remove_tags=remove_tags,
    )

    project.manifest.source.update(
        {
            "last_tag_clean_scanned": summary.scanned,
            "last_tag_clean_cleaned": summary.cleaned,
            "last_tag_clean_skipped": summary.skipped,
            "last_tag_clean_subject": project.manifest.subject,
            "last_tag_clean_trigger": project.manifest.trigger,
            "last_tag_clean_remove_rules": list(remove_tags),
        }
    )
    project.save()

    print(f"scanned: {summary.scanned}")
    print(f"cleaned: {summary.cleaned}")
    print(f"skipped: {summary.skipped}")
    print(f"subject: {project.manifest.subject}")

    return 0 if summary.cleaned or summary.skipped else 1


def command_caption(args: argparse.Namespace) -> int:
    root = find_project_root(args.path)
    config = load_config(root / "config.toml")
    paths = ProjectPaths.from_root(root, config.get("paths", {}))
    project = DatasetProject.open(paths)

    caption_config = config.get("caption", {})
    if not isinstance(caption_config, dict):
        raise LoraDataMakerError("config section [caption] must be a table")

    separator = caption_config.get("separator", ", ")
    subject_position = caption_config.get("subject_position", "first")
    trigger_from_subject = caption_config.get("trigger_from_subject", True)

    if not isinstance(separator, str):
        raise LoraDataMakerError("config field caption.separator must be a string")
    if not isinstance(subject_position, str):
        raise LoraDataMakerError(
            "config field caption.subject_position must be a string"
        )
    if not isinstance(trigger_from_subject, bool):
        raise LoraDataMakerError(
            "config field caption.trigger_from_subject must be a boolean"
        )

    settings = CaptionConfig(
        separator=separator,
        subject_position=subject_position,
        trigger_from_subject=trigger_from_subject,
    )

    summary = caption_manifest(
        project.manifest,
        root=root,
        output_dir=paths.work / "captions",
        config=settings,
    )

    project.manifest.source.update(
        {
            "last_caption_scanned": summary.scanned,
            "last_caption_captioned": summary.captioned,
            "last_caption_failed": summary.failed,
            "last_caption_skipped": summary.skipped,
            "last_caption_separator": separator,
            "last_caption_subject": project.manifest.subject,
            "last_caption_trigger": (
                project.manifest.subject
                if trigger_from_subject
                else project.manifest.trigger
            ),
            "last_caption_trigger_from_subject": trigger_from_subject,
        }
    )
    project.save()

    print(f"scanned: {summary.scanned}")
    print(f"captioned: {summary.captioned}")
    print(f"failed: {summary.failed}")
    print(f"skipped: {summary.skipped}")
    print(f"captions: {paths.work / 'captions'}")

    return 0 if summary.captioned or summary.skipped else 1



def command_export(args: argparse.Namespace) -> int:
    root = find_project_root(args.path)
    config = load_config(root / "config.toml")
    paths = ProjectPaths.from_root(root, config.get("paths", {}))
    project = DatasetProject.open(paths)

    export_config = config.get("export", {})
    if not isinstance(export_config, dict):
        raise LoraDataMakerError("config section [export] must be a table")

    repeats = (
        args.repeats
        if args.repeats is not None
        else int(export_config.get("repeats", 1))
    )
    folder_name = (
        args.folder_name
        if args.folder_name is not None
        else str(export_config.get("folder_name", "") or "")
    )

    dataset_folder = kohya_folder_name(
        subject=project.manifest.subject,
        repeats=repeats,
        folder_name=folder_name,
    )
    summary = export_manifest(
        project.manifest,
        root=root,
        output_dir=paths.output,
        repeats=repeats,
        folder_name=folder_name,
    )

    project.manifest.source.update(
        {
            "last_export_scanned": summary.scanned,
            "last_export_exported": summary.exported,
            "last_export_failed": summary.failed,
            "last_export_skipped": summary.skipped,
            "last_export_repeats": repeats,
            "last_export_folder": dataset_folder,
            "last_export_dir": (
                manifest_path(root, summary.output_dir)
            ),
        }
    )
    project.save()

    print(f"scanned: {summary.scanned}")
    print(f"exported: {summary.exported}")
    print(f"failed: {summary.failed}")
    print(f"skipped: {summary.skipped}")
    print(f"dataset: {summary.output_dir}")

    return 0 if summary.failed == 0 and (summary.exported or summary.skipped) else 1


def command_pack(args: argparse.Namespace) -> int:
    root = find_project_root(args.path)
    config = load_config(root / "config.toml")
    paths = ProjectPaths.from_root(root, config.get("paths", {}))
    project = DatasetProject.open(paths)

    dataset_dir = args.dataset_dir
    if dataset_dir:
        candidate = Path(dataset_dir).expanduser()
        dataset_root = (
            candidate if candidate.is_absolute() else root / candidate
        ).resolve()
    else:
        last_export_dir = project.manifest.source.get("last_export_dir")
        if isinstance(last_export_dir, str) and last_export_dir.strip():
            dataset_root = (root / last_export_dir).resolve()
        else:
            export_config = config.get("export", {})
            if not isinstance(export_config, dict):
                raise LoraDataMakerError(
                    "config section [export] must be a table"
                )
            repeats = int(export_config.get("repeats", 1))
            folder_name = str(export_config.get("folder_name", "") or "")
            dataset_root = (
                paths.output
                / kohya_folder_name(
                    subject=project.manifest.subject,
                    repeats=repeats,
                    folder_name=folder_name,
                )
            ).resolve()

    archive = (
        Path(args.archive).expanduser()
        if args.archive
        else paths.output / f"{safe_component(project.manifest.project_name)}.zip"
    )
    archive = archive if archive.is_absolute() else root / archive
    archive = archive.resolve()

    summary = pack_dataset(
        root=root,
        dataset_dir=dataset_root,
        archive_path=archive,
        manifest_path=project.manifest_path,
    )

    project.manifest.source.update(
        {
            "last_pack_archive": summary.archive_path,
            "last_pack_file_count": summary.file_count,
            "last_pack_size_bytes": summary.size_bytes,
            "last_pack_sha256": summary.sha256,
        }
    )
    project.save()

    print(f"archive: {summary.archive_path}")
    print(f"files: {summary.file_count}")
    print(f"size: {summary.size_bytes}")
    print(f"sha256: {summary.sha256}")

    return 0



def command_scrape(args: argparse.Namespace) -> int:
    root = find_project_root(args.path)
    config = load_config(root / "config.toml")
    paths = ProjectPaths.from_root(root, config.get("paths", {}))
    project = DatasetProject.open(paths)

    scraper_config = config.get("scraper", {})
    if not isinstance(scraper_config, dict):
        raise LoraDataMakerError("config section [scraper] must be a table")

    query = args.query or scraper_config.get("query")
    if not isinstance(query, str) or not query.strip():
        raise LoraDataMakerError(
            "scrape query is required; provide it on the CLI or in [scraper].query"
        )

    count = args.count if args.count is not None else int(
        scraper_config.get("count", 300)
    )
    timeout = float(scraper_config.get("timeout", 20))
    retries = int(scraper_config.get("retries", 3))
    delay = float(scraper_config.get("delay", 1.0))
    max_scrolls = int(scraper_config.get("max_scrolls", 30))

    manager = ScraperManager(
        backend=args.backend,
        output_dir=paths.raw,
        cache_dir=paths.cache,
        timeout=timeout,
        retries=retries,
        delay=delay,
        max_scrolls=max_scrolls,
    )
    known_urls = {
        record.source_url
        for record in project.manifest.images.values()
        if record.source_url
    }
    downloads, failures = manager.run(
        query,
        count=count,
        known_urls=known_urls,
    )
    search_error = getattr(manager, "last_search_error", None)

    added_records = 0
    for result in downloads:
        relative_path = manifest_path(root, result.path)
        record = find_image_by_source_url(project.manifest, result.url)
        if record is None:
            record = register_image(
                project.manifest,
                path=relative_path,
                source_url=result.url,
                source_query=query,
                status="downloaded",
            )
            added_records += 1
        record.sha256 = result.sha256
        record.file_size = result.file_size
        record.mime_type = result.content_type

    project.manifest.source.update(
        {
            "backend": args.backend,
            "query": query,
            "last_scrape_requested": count,
            "last_scrape_downloaded": len(downloads),
            "last_scrape_added_records": added_records,
            "last_scrape_failed": len(failures),
        }
    )
    project.save()

    print(f"scrape backend: {args.backend}")
    print(f"query: {query}")
    print(f"requested: {count}")
    print(f"downloaded: {len(downloads)}")
    print(f"failed: {len(failures)}")
    print(f"raw: {paths.raw}")

    for result, error in failures[:10]:
        print(f"failed: {result.image_url} :: {error}")
    if len(failures) > 10:
        print(f"... and {len(failures) - 10} more failures")

    if len(downloads) >= count:
        return 0

    if search_error and not downloads:
        print(
            f"scrape failed: downloaded 0/{count} requested; "
            f"search source error: {search_error}",
        )
        return 1

    if not downloads and known_urls and not failures:
        print("no new images downloaded")
        return 0

    if downloads:
        if search_error:
            print(
                f"scrape complete with source error: downloaded "
                f"{len(downloads)}/{count} requested; "
                f"skipped {len(failures)} failed candidates; "
                f"source: {search_error}",
            )
        else:
            print(
                f"scrape complete with skips: downloaded {len(downloads)}/{count} requested; "
                f"skipped {len(failures)} failed candidates",
            )
        return 0

    print(
        f"scrape failed: downloaded 0/{count} requested; "
        f"failed candidates: {len(failures)}",
    )
    return 1


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logger = configure_logging(args.verbose)

    try:
        if args.command == "init":
            return command_init(args)
        if args.command == "info":
            return command_info(args)
        if args.command == "scrape":
            return command_scrape(args)
        if args.command == "validate":
            return command_validate(args)
        if args.command == "reconcile":
            return command_reconcile(args)
        if args.command == "clean":
            return command_clean(args)
        if args.command == "process":
            return command_process(args)
        if args.command == "tag":
            return command_tag(args)
        if args.command == "clean-tags":
            return command_clean_tags(args)
        if args.command == "caption":
            return command_caption(args)
        if args.command == "build":
            summary = run_build(
                path=args.path,
                from_stage=args.from_stage,
                to_stage=args.to_stage,
                skip=args.skip,
            )
            return 0 if summary.success else 1
        if args.command == "export":
            return command_export(args)
        if args.command == "pack":
            return command_pack(args)
        if args.command == "web":
            from .web import serve
            serve(args.path, host=args.host, port=args.port)
            return 0
    except KeyboardInterrupt:
        logger.error("operation cancelled by user")
        return 130
    except (
        LoraDataMakerError,
        FileNotFoundError,
        OSError,
        ValueError,
        KeyError,
        ScraperError,
        TaggerError,
    ) as exc:
        logger.error(str(exc))
        return 1

    parser.error(f"unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
