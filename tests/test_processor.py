from pathlib import Path

from PIL import Image

from lora_data_maker.processing.faces import FaceBox, largest_face, square_crop_around_face
from lora_data_maker.processing.processor import ProcessConfig, process_image, resize_and_crop


def test_cover_crop_hits_target_size():
    image = Image.new("RGB", (800, 400), (10, 20, 30))

    processed, crop_box = resize_and_crop(
        image,
        512,
        512,
        mode="cover",
    )

    assert processed.size == (512, 512)
    assert crop_box is not None


def test_fit_preserves_full_image_without_padding_borders():
    image = Image.new("RGB", (800, 400), (10, 20, 30))

    processed, crop_box = resize_and_crop(
        image,
        512,
        512,
        mode="fit",
    )

    # Fit stays proportional and keeps the entire source image. With a
    # 2:1 input and square target box, output is 512x256, not padded 512x512.
    assert processed.size == (512, 256)
    assert crop_box is None
    assert processed.getpixel((0, 0)) == (10, 20, 30)
    assert processed.getpixel((511, 255)) == (10, 20, 30)


def test_stretch_has_no_crop():
    image = Image.new("RGB", (800, 400), (10, 20, 30))

    processed, crop_box = resize_and_crop(
        image,
        512,
        512,
        mode="stretch",
    )

    assert processed.size == (512, 512)
    assert crop_box is None


def test_face_helpers_choose_largest_and_crop():
    image = Image.new("RGB", (800, 600), "white")
    small = FaceBox(50, 50, 150, 150)
    large = FaceBox(250, 100, 550, 400)

    assert largest_face([small, large]) == large

    cropped = square_crop_around_face(image, large, padding=1.5)
    assert cropped.width == cropped.height
    assert cropped.size[0] > large.width


class FakeFaceDetector:
    def detect(self, image):
        return [FaceBox(250, 100, 550, 400)]


def test_process_image_writes_rgb_jpeg(tmp_path):
    source = tmp_path / "input.png"
    output = tmp_path / "work" / "processed.jpg"
    image = Image.new("RGBA", (800, 600), (100, 80, 60, 255))
    image.save(source, format="PNG")

    result = process_image(
        source,
        output,
        config=ProcessConfig(
            target_width=512,
            target_height=512,
            resize_mode="cover",
        ),
    )

    assert result.output_path == output.resolve()
    assert result.output_size == (512, 512)

    with Image.open(output) as saved:
        assert saved.size == (512, 512)
        assert saved.mode == "RGB"
        assert saved.format == "JPEG"


def test_process_image_uses_face_detector(tmp_path):
    source = tmp_path / "input.jpg"
    output = tmp_path / "face.jpg"
    Image.new("RGB", (800, 600), (20, 30, 40)).save(source, format="JPEG")

    result = process_image(
        source,
        output,
        config=ProcessConfig(face_crop=True),
        face_detector=FakeFaceDetector(),
    )

    assert result.face_detected is True
    assert result.output_size == (512, 512)


def test_face_crop_requires_detector(tmp_path):
    source = tmp_path / "input.jpg"
    output = tmp_path / "face.jpg"
    Image.new("RGB", (800, 600)).save(source, format="JPEG")

    try:
        process_image(
            source,
            output,
            config=ProcessConfig(face_crop=True),
        )
    except ValueError as exc:
        assert "no face detector" in str(exc)
    else:
        raise AssertionError("expected face detector configuration error")


def test_exif_orientation_is_transposed(tmp_path):
    source = tmp_path / "oriented.jpg"
    output = tmp_path / "oriented-output.jpg"

    image = Image.new("RGB", (80, 40), "red")
    exif = image.getexif()
    exif[274] = 6
    image.save(source, format="JPEG", exif=exif.tobytes())

    result = process_image(source, output)

    assert result.source_size == (80, 40)
    assert result.output_size == (512, 512)
    assert result.source_orientation == 6


def test_face_crop_stays_square_when_requested_crop_exceeds_image(tmp_path):
    from lora_data_maker.processing.faces import FaceBox, square_crop_around_face
    image = Image.new("RGB", (400, 800), "white")
    face = FaceBox(100, 300, 300, 500)

    cropped = square_crop_around_face(image, face, padding=4.0)

    assert cropped.width == cropped.height
    assert cropped.size == (400, 400)
