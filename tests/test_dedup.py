import shutil

from PIL import Image, ImageDraw, ImageFilter

from lora_data_maker.processing.dedup import (
    find_perceptual_duplicates,
    hamming_distance,
    perceptual_hash,
    sha256_file,
)


def _make_subject_image(path):
    image = Image.new("RGB", (640, 640), "white")
    draw = ImageDraw.Draw(image)
    draw.ellipse((100, 100, 540, 540), fill=(80, 120, 180))
    draw.rectangle((240, 180, 400, 460), fill=(220, 80, 80))
    image.save(path, format="PNG")


def test_sha256_is_stable(tmp_path):
    first = tmp_path / "first.png"
    second = tmp_path / "second.png"
    _make_subject_image(first)
    shutil.copyfile(first, second)

    assert sha256_file(first) == sha256_file(second)


def test_perceptual_hash_identical_and_near_duplicate(tmp_path):
    first = tmp_path / "first.png"
    second = tmp_path / "second.png"
    blurred = tmp_path / "blurred.png"
    _make_subject_image(first)
    shutil.copyfile(first, second)

    with Image.open(first) as image:
        image.filter(ImageFilter.GaussianBlur(1)).save(blurred, format="PNG")

    first_hash = perceptual_hash(first)
    second_hash = perceptual_hash(second)
    blurred_hash = perceptual_hash(blurred)

    assert hamming_distance(first_hash, second_hash) == 0
    assert hamming_distance(first_hash, blurred_hash) <= 6


def test_perceptual_duplicate_matching_preserves_input_order():
    matches = find_perceptual_duplicates(
        [("first", "0000000000000000"), ("second", "0000000000000001")],
        max_distance=1,
    )

    assert len(matches) == 1
    assert matches[0].image_id == "second"
    assert matches[0].duplicate_of == "first"
    assert matches[0].distance == 1
