from __future__ import annotations

import hashlib
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

import requests


DEFAULT_HF_BASE_URL = "https://huggingface.co"
_SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9._-]+")


@dataclass(frozen=True)
class ModelFiles:
    model_path: Path
    labels_path: Path


class ModelDownloadError(RuntimeError):
    """Raised when a tagger model cannot be downloaded or verified."""


def cache_key(repo_id: str, revision: str) -> str:
    raw = f"{repo_id}@{revision}".encode("utf-8")
    digest = hashlib.sha256(raw).hexdigest()[:16]
    repo_name = _SAFE_NAME_RE.sub("-", repo_id).strip("-").lower()
    revision_name = _SAFE_NAME_RE.sub("-", revision).strip("-").lower()
    return f"{repo_name}-{revision_name}-{digest}"


class HuggingFaceModelCache:
    def __init__(
        self,
        cache_dir: Path | str,
        *,
        repo_id: str,
        revision: str = "main",
        timeout: float = 60.0,
        model_filename: str = "model.onnx",
        labels_filename: str = "selected_tags.csv",
        session: requests.Session | None = None,
    ) -> None:
        if not repo_id.strip():
            raise ValueError("repo_id must not be empty")
        if not revision.strip():
            raise ValueError("revision must not be empty")
        if timeout <= 0:
            raise ValueError("timeout must be greater than zero")
        if not _is_safe_filename(model_filename):
            raise ValueError("model_filename must be a simple filename")
        if not _is_safe_filename(labels_filename):
            raise ValueError("labels_filename must be a simple filename")

        self.root = Path(cache_dir).expanduser().resolve() / cache_key(
            repo_id,
            revision,
        )
        self.repo_id = repo_id
        self.revision = revision
        self.timeout = timeout
        self.model_filename = model_filename
        self.labels_filename = labels_filename
        self.session = session or requests.Session()

    @property
    def model_path(self) -> Path:
        return self.root / self.model_filename

    @property
    def labels_path(self) -> Path:
        return self.root / self.labels_filename

    def ensure(
        self,
        *,
        model_sha256: str | None = None,
    ) -> ModelFiles:
        self.root.mkdir(parents=True, exist_ok=True)

        self._ensure_file(
            self.model_filename,
            self.model_path,
            expected_sha256=model_sha256,
        )
        self._ensure_file(
            self.labels_filename,
            self.labels_path,
        )

        return ModelFiles(
            model_path=self.model_path,
            labels_path=self.labels_path,
        )

    def _ensure_file(
        self,
        filename: str,
        destination: Path,
        *,
        expected_sha256: str | None = None,
    ) -> None:
        if destination.is_file() and (
            expected_sha256 is None
            or _sha256_path(destination) == expected_sha256.lower()
        ):
            return

        if destination.exists():
            destination.unlink()

        url = (
            f"{DEFAULT_HF_BASE_URL}/{self.repo_id}"
            f"/resolve/{self.revision}/{filename}"
        )
        fd, temp_name = tempfile.mkstemp(
            prefix=f".{destination.name}.",
            suffix=".part",
            dir=destination.parent,
        )
        temp = Path(temp_name)
        open_fd = fd
        digest = hashlib.sha256()

        try:
            with self.session.get(
                url,
                stream=True,
                timeout=self.timeout,
                allow_redirects=True,
            ) as response:
                response.raise_for_status()
                with os.fdopen(open_fd, "wb") as handle:
                    open_fd = -1
                    _stream_response(response, handle, digest)
                    handle.flush()
                    os.fsync(handle.fileno())

            if expected_sha256 and digest.hexdigest().lower() != expected_sha256.lower():
                raise ModelDownloadError(
                    f"sha256 mismatch for {filename}: "
                    f"expected {expected_sha256}, got {digest.hexdigest()}"
                )

            temp.replace(destination)
        except ModelDownloadError:
            temp.unlink(missing_ok=True)
            raise
        except (OSError, requests.RequestException) as exc:
            temp.unlink(missing_ok=True)
            raise ModelDownloadError(
                f"failed downloading {self.repo_id}/{filename}: {exc}"
            ) from exc
        finally:
            if open_fd >= 0:
                os.close(open_fd)


def _is_safe_filename(filename: str) -> bool:
    value = str(filename).strip()
    return bool(value) and Path(value).name == value and value not in {".", ".."}


def _stream_response(
    response,
    handle: BinaryIO,
    digest: "hashlib._Hash",
    *,
    chunk_size: int = 1024 * 1024,
) -> None:
    for chunk in response.iter_content(chunk_size=chunk_size):
        if not chunk:
            continue
        handle.write(chunk)
        digest.update(chunk)


def _sha256_path(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()
