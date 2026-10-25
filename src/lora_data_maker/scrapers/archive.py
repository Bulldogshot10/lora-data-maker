from __future__ import annotations

from pathlib import Path

from .base import ScraperError


class DownloadArchive:
    """Small persistent URL archive used to make downloads resumable."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path).expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._urls: set[str] = set()
        if self.path.is_file():
            try:
                content = self.path.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError) as exc:
                raise ScraperError(
                    f"could not read download archive: {self.path}"
                ) from exc
            self._urls = {
                line.strip()
                for line in content.splitlines()
                if line.strip()
            }

    def __contains__(self, url: str) -> bool:
        return url in self._urls

    def add(self, url: str) -> bool:
        if url in self._urls:
            return False
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(url)
            handle.write("\n")
            handle.flush()
        self._urls.add(url)
        return True

    def discard(self, url: str) -> bool:
        """Remove a rejected URL from the persistent archive so it can be retried."""
        if url not in self._urls:
            return False

        remaining = self._urls - {url}
        temporary = self.path.with_name(self.path.name + ".tmp")
        try:
            with temporary.open("w", encoding="utf-8") as handle:
                for item in sorted(remaining):
                    handle.write(item)
                    handle.write("\n")
                handle.flush()
            temporary.replace(self.path)
        except OSError as exc:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass
            raise ScraperError(
                f"could not update download archive: {self.path}"
            ) from exc

        self._urls = remaining
        return True

    @property
    def urls(self) -> frozenset[str]:
        return frozenset(self._urls)
