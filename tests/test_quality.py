from PIL import Image, ImageDraw, ImageFilter

from lora_data_maker.processing.quality import evaluate_quality, laplacian_variance


def _make_checkerboard(size=640):
    image = Image.new("L", (size, size), 255)
    draw = ImageDraw.Draw(image)
    for y in range(0, size, 16):
        for x in range(0, size, 16):
            if (x // 16 + y // 16) % 2 == 0:
                draw.rectangle((x, y, x + 15, y + 15), fill=0)
    return image


def test_sharp_image_has_higher_laplacian_variance():
    sharp = _make_checkerboard()
    blurry = sharp.filter(ImageFilter.GaussianBlur(8))

    assert laplacian_variance(sharp) > laplacian_variance(blurry)


def test_quality_rejects_low_resolution(tmp_path):
    path = tmp_path / "small.png"
    _make_checkerboard(256).save(path)

    result = evaluate_quality(path, min_width=512, min_height=512)

    assert result.valid is False
    assert result.width == 256
    assert result.height == 256
    assert "resolution too small" in (result.reason or "")


def test_quality_rejects_blur(tmp_path):
    path = tmp_path / "blurry.png"
    image = _make_checkerboard(640).filter(ImageFilter.GaussianBlur(8))
    image.save(path)

    result = evaluate_quality(path, min_width=512, min_height=512, blur_threshold=1000)

    assert result.valid is False
    assert "too blurry" in (result.reason or "")


def test_quality_accepts_good_image(tmp_path):
    path = tmp_path / "good.png"
    _make_checkerboard(640).save(path)

    result = evaluate_quality(path, min_width=512, min_height=512)

    assert result.valid is True
    assert result.width == 640
    assert result.height == 640
