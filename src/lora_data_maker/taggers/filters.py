from __future__ import annotations

import re
from collections.abc import Iterable, Sequence


_WHITESPACE_RE = re.compile(r"\s+")


def normalize_tag(tag: str) -> str:
    """Normalize one WD14-style tag into stable caption/tag text."""
    value = str(tag).strip().replace("_", " ")
    value = _WHITESPACE_RE.sub(" ", value).strip().lower()
    return value


def normalize_tags(tags: Iterable[str]) -> list[str]:
    """Normalize tags and remove duplicates while preserving first-seen order."""
    result: list[str] = []
    seen: set[str] = set()

    for raw_tag in tags:
        tag = normalize_tag(raw_tag)
        if not tag or tag in seen:
            continue
        seen.add(tag)
        result.append(tag)

    return result


def clean_tags(
    tags: Sequence[str],
    *,
    remove: Iterable[str] = (),
    subject: str | None = None,
) -> list[str]:
    """Normalize, filter, deduplicate, then inject the canonical subject."""
    normalized = normalize_tags(tags)
    blacklist = {
        normalize_tag(tag)
        for tag in remove
        if normalize_tag(tag)
    }

    filtered = [
        tag
        for tag in normalized
        if tag not in blacklist
    ]

    if subject is not None:
        canonical_subject = normalize_tag(subject)
        if canonical_subject:
            filtered = [
                tag
                for tag in filtered
                if tag != canonical_subject
            ]
            filtered.insert(0, canonical_subject)

    return filtered


def tag_set(tags: Iterable[str]) -> set[str]:
    return set(normalize_tags(tags))
