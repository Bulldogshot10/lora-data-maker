from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, ClassVar, Mapping
from uuid import uuid4


IMAGE_STATUSES: tuple[str, ...] = (
    "discovered",
    "downloaded",
    "validated",
    "duplicate",
    "rejected",
    "processed",
    "tagged",
    "captioned",
    "exported",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


@dataclass
class ImageRecord:
    image_id: str = field(default_factory=lambda: uuid4().hex)
    path: str = ""
    status: str = "discovered"
    source_url: str | None = None
    source_query: str | None = None
    sha256: str | None = None
    perceptual_hash: str | None = None
    width: int | None = None
    height: int | None = None
    file_size: int | None = None
    mime_type: str | None = None
    processed_path: str | None = None
    caption_path: str | None = None
    tags: list[str] = field(default_factory=list)
    rejection_reason: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=utc_now)
    updated_at: str = field(default_factory=utc_now)

    VALID_STATUSES: ClassVar[set[str]] = set(IMAGE_STATUSES)

    def set_status(self, status: str, *, reason: str | None = None) -> None:
        if status not in self.VALID_STATUSES:
            raise ValueError(
                f"invalid image status {status!r}; "
                f"expected one of {sorted(self.VALID_STATUSES)}"
            )
        self.status = status
        if reason is not None:
            self.rejection_reason = reason
        elif status not in {"duplicate", "rejected"}:
            self.rejection_reason = None
        self.updated_at = utc_now()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ImageRecord":
        values = dict(data)
        values.setdefault("image_id", uuid4().hex)
        values.setdefault("path", "")
        values.setdefault("status", "discovered")
        values.setdefault("tags", [])
        values.setdefault("metadata", {})
        values.setdefault("created_at", utc_now())
        values.setdefault("updated_at", values["created_at"])
        record = cls(**values)
        if record.status not in cls.VALID_STATUSES:
            raise ValueError(
                f"invalid image status {record.status!r}; "
                f"expected one of {sorted(cls.VALID_STATUSES)}"
            )
        if not isinstance(record.tags, list) or not all(
            isinstance(tag, str) for tag in record.tags
        ):
            raise ValueError("image record tags must be a list of strings")
        if not isinstance(record.metadata, dict):
            raise ValueError("image record metadata must be an object")
        return record


@dataclass
class ProjectManifest:
    project_name: str
    subject: str
    trigger: str
    schema_version: int = 1
    created_at: str = field(default_factory=utc_now)
    updated_at: str = field(default_factory=utc_now)
    source: dict[str, Any] = field(default_factory=dict)
    config_snapshot: dict[str, Any] = field(default_factory=dict)
    images: dict[str, ImageRecord] = field(default_factory=dict)

    def add_image(self, image: ImageRecord) -> ImageRecord:
        if image.image_id in self.images:
            raise ValueError(f"image id already exists: {image.image_id}")
        self.images[image.image_id] = image
        self.touch()
        return image

    def upsert_image(self, image: ImageRecord) -> ImageRecord:
        self.images[image.image_id] = image
        self.touch()
        return image

    def get_image(self, image_id: str) -> ImageRecord:
        try:
            return self.images[image_id]
        except KeyError as exc:
            raise KeyError(f"image not found in manifest: {image_id}") from exc

    def touch(self) -> None:
        self.updated_at = utc_now()

    def counts(self) -> dict[str, int]:
        counts = {status: 0 for status in IMAGE_STATUSES}
        for image in self.images.values():
            counts[image.status] = counts.get(image.status, 0) + 1
        return counts

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "project_name": self.project_name,
            "subject": self.subject,
            "trigger": self.trigger,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "source": self.source,
            "config_snapshot": self.config_snapshot,
            "images": {
                image_id: image.to_dict()
                for image_id, image in self.images.items()
            },
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ProjectManifest":
        images_data = data.get("images", {})
        if not isinstance(images_data, Mapping):
            raise ValueError("manifest field 'images' must be an object")

        images: dict[str, ImageRecord] = {}
        for image_id, image_data in images_data.items():
            if not isinstance(image_data, Mapping):
                raise ValueError(
                    f"manifest image {image_id!r} must be an object"
                )
            record = ImageRecord.from_dict(image_data)
            key = str(image_id)
            if record.image_id != key:
                raise ValueError(
                    f"manifest image key {key!r} does not match "
                    f"record image_id {record.image_id!r}"
                )
            if key in images:
                raise ValueError(
                    f"duplicate manifest image id: {key!r}"
                )
            images[key] = record

        return cls(
            schema_version=int(data.get("schema_version", 1)),
            project_name=str(data.get("project_name", "")),
            subject=str(data.get("subject", "")),
            trigger=str(data.get("trigger", "")),
            created_at=str(data.get("created_at", utc_now())),
            updated_at=str(data.get("updated_at", data.get("created_at", utc_now()))),
            source=dict(data.get("source", {})),
            config_snapshot=dict(data.get("config_snapshot", {})),
            images=images,
        )
