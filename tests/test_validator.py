from PIL import Image

from lora_data_maker.processing.validator import validate_image


def test_valid_jpeg(tmp_path):
    path = tmp_path / "ok.jpg"
    Image.new("RGB", (640, 480), (10, 20, 30)).save(path, format="JPEG")

    result = validate_image(
        path,
        min_width=512,
        min_height=256,
    )

    assert result.valid is True
    assert result.width == 640
    assert result.height == 480
    assert result.format == "JPEG"
    assert result.mime_type == "image/jpeg"
    assert result.file_size > 0


def test_rejects_non_image(tmp_path):
    path = tmp_path / "bad.jpg"
    path.write_text("not an image", encoding="utf-8")

    result = validate_image(path)

    assert result.valid is False
    assert "unidentified image" in (result.reason or "")


def test_rejects_small_image(tmp_path):
    path = tmp_path / "small.png"
    Image.new("RGB", (128, 128)).save(path, format="PNG")

    result = validate_image(path, min_width=512, min_height=512)

    assert result.valid is False
    assert result.width == 128
    assert "dimensions too small" in (result.reason or "")


def test_reads_webp_metadata(tmp_path):
    path = tmp_path / "animated.webp"
    frames = [
        Image.new("RGB", (32, 32), (255, 0, 0)),
        Image.new("RGB", (32, 32), (0, 255, 0)),
    ]
    frames[0].save(
        path,
        format="WEBP",
        save_all=True,
        append_images=frames[1:],
        duration=100,
        loop=0,
    )

    result = validate_image(path)

    assert result.valid is True
    assert result.format == "WEBP"
    assert result.mime_type == "image/webp"
    assert result.metadata.get("loop") == 0


def test_missing_file(tmp_path):
    result = validate_image(tmp_path / "missing.jpg")
    assert result.valid is False
    assert result.reason == "file not found"
def test_rejects_truncated_jpeg(tmp_path):
    path = tmp_path / "truncated.jpg"
    Image.new("RGB", (640, 480), (10, 20, 30)).save(path, format="JPEG")
    path.write_bytes(path.read_bytes()[:20])

    result = validate_image(path)

    assert result.valid is False
    assert any(token in (result.reason or "").lower() for token in ("corrupt", "decode failed"))


def test_reads_exif_orientation(tmp_path):
    path = tmp_path / "oriented.jpg"
    image = Image.new("RGB", (640, 480), (30, 40, 50))
    exif = image.getexif()
    exif[274] = 6
    image.save(path, format="JPEG", exif=exif.tobytes())

    result = validate_image(path)

    assert result.valid is True
    assert result.exif_orientation == 6
    assert result.has_exif is True
    assert result.metadata["exif_orientation"] == 6


def test_reads_jpeg_dpi_as_json_safe_value(tmp_path):
    import json

    path = tmp_path / "dpi.jpg"
    Image.new("RGB", (640, 480), (10, 20, 30)).save(
        path,
        format="JPEG",
        dpi=(300, 300),
    )

    result = validate_image(path)

    assert result.valid is True
    assert result.metadata["dpi"]
    json.dumps(result.metadata)
