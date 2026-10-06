from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

from PIL import Image


@dataclass(frozen=True)
class FaceBox:
    left: int
    top: int
    right: int
    bottom: int

    @property
    def width(self) -> int:
        return max(0, self.right - self.left)

    @property
    def height(self) -> int:
        return max(0, self.bottom - self.top)

    @property
    def area(self) -> int:
        return self.width * self.height

    @property
    def center(self) -> tuple[float, float]:
        return (
            (self.left + self.right) / 2,
            (self.top + self.bottom) / 2,
        )


class FaceDetector(Protocol):
    def detect(self, image: Image.Image) -> Sequence[FaceBox]:
        """Return face bounding boxes in source-image coordinates."""


def largest_face(faces: Sequence[FaceBox]) -> FaceBox | None:
    if not faces:
        return None
    return max(faces, key=lambda face: face.area)


def square_crop_around_face(
    image: Image.Image,
    face: FaceBox,
    *,
    padding: float = 1.75,
) -> Image.Image:
    if padding <= 0:
        raise ValueError("face crop padding must be greater than zero")

    width, height = image.size
    side = max(face.width, face.height) * padding
    if side <= 0:
        raise ValueError("face bounding box has zero area")
    side = min(side, float(width), float(height))

    center_x, center_y = face.center
    half = side / 2
    left = int(round(center_x - half))
    top = int(round(center_y - half))
    right = int(round(center_x + half))
    bottom = int(round(center_y + half))

    if left < 0:
        right -= left
        left = 0
    if top < 0:
        bottom -= top
        top = 0
    if right > width:
        left -= right - width
        right = width
    if bottom > height:
        top -= bottom - height
        bottom = height

    left = max(0, left)
    top = max(0, top)
    right = min(width, right)
    bottom = min(height, bottom)

    if right <= left or bottom <= top:
        raise ValueError("face crop produced an empty region")

    return image.crop((left, top, right, bottom))
