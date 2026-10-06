from __future__ import annotations

import hashlib
from dataclasses import dataclass
from math import cos, pi, sqrt
from pathlib import Path
from typing import Iterable

from PIL import Image


@dataclass(frozen=True)
class DuplicateMatch:
    image_id: str
    duplicate_of: str
    kind: str
    distance: int | None = None


def sha256_file(path: Path | str, *, chunk_size: int = 1024 * 1024) -> str:
    if chunk_size < 1:
        raise ValueError("chunk_size must be greater than zero")

    digest = hashlib.sha256()
    with Path(path).expanduser().resolve().open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def perceptual_hash(
    path: Path | str,
    *,
    size: int = 32,
    low_freq_size: int = 8,
) -> str:
    if size < 8:
        raise ValueError("perceptual hash size must be at least 8")
    if low_freq_size < 2 or low_freq_size > size:
        raise ValueError("low_freq_size must be between 2 and size")

    with Image.open(Path(path).expanduser().resolve()) as image:
        gray = image.convert("L").resize(
            (size, size),
            Image.Resampling.LANCZOS,
        )
        pixels = list(gray.get_flattened_data())

    alpha = [sqrt(1.0 / size)] + [
        sqrt(2.0 / size) for _ in range(low_freq_size - 1)
    ]
    cos_table = [
        [
            cos(pi * (2 * x + 1) * u / (2 * size))
            for x in range(size)
        ]
        for u in range(low_freq_size)
    ]

    row_coefficients: list[list[float]] = []
    for y in range(size):
        row = pixels[y * size : (y + 1) * size]
        row_coefficients.append(
            [
                alpha[u] * sum(
                    row[x] * cos_table[u][x]
                    for x in range(size)
                )
                for u in range(low_freq_size)
            ]
        )

    coefficients: list[float] = []
    for v in range(low_freq_size):
        for u in range(low_freq_size):
            coefficients.append(
                alpha[v] * sum(
                    row_coefficients[y][u] * cos_table[v][y]
                    for y in range(size)
                )
            )

    dc = coefficients[0]
    del dc
    low_frequency = coefficients[1:]
    median = sorted(low_frequency)[len(low_frequency) // 2]

    bits = 0
    for coefficient in coefficients:
        bits = (bits << 1) | int(coefficient > median)

    width = (low_freq_size * low_freq_size + 3) // 4
    return f"{bits:0{width}x}"


def hamming_distance(first: str, second: str) -> int:
    try:
        left = int(first, 16)
        right = int(second, 16)
    except ValueError as exc:
        raise ValueError("perceptual hashes must be hexadecimal strings") from exc

    return (left ^ right).bit_count()


def find_perceptual_duplicates(
    items: Iterable[tuple[str, str]],
    *,
    max_distance: int,
) -> list[DuplicateMatch]:
    if max_distance < 0:
        raise ValueError("max_distance cannot be negative")

    accepted: list[tuple[str, str]] = []
    duplicates: list[DuplicateMatch] = []

    for image_id, image_hash in sorted(items, key=lambda item: item[0]):
        match: tuple[str, int] | None = None
        for accepted_id, accepted_hash in accepted:
            distance = hamming_distance(image_hash, accepted_hash)
            if distance <= max_distance:
                match = (accepted_id, distance)
                break

        if match is None:
            accepted.append((image_id, image_hash))
            continue

        duplicate_of, distance = match
        duplicates.append(
            DuplicateMatch(
                image_id=image_id,
                duplicate_of=duplicate_of,
                kind="perceptual",
                distance=distance,
            )
        )

    return duplicates
