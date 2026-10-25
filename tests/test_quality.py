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



def test_laplacian_variance_ignores_flat_image_border_artifacts():
    image = Image.new("L", (64, 64), 128)

    assert laplacian_variance(image) == 0.0


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



def test_quality_rejects_multiple_people(tmp_path):
    from lora_data_maker.processing.person import PersonBox

    class Detector:
        def detect(self, image):
            return (
                PersonBox(20, 20, 500, 600, 0.95),
                PersonBox(100, 60, 450, 520, 0.90),
            )

    path = tmp_path / "multiple.png"
    _make_checkerboard(640).save(path)

    result = evaluate_quality(
        path,
        person_detector=Detector(),
        max_persons=1,
    )

    assert result.valid is False
    assert result.person_count == 2
    assert "multiple people" in (result.reason or "")
    assert any("multiple people" in reason for reason in result.reasons)


def test_quality_rejects_missing_subject(tmp_path):
    class Detector:
        def detect(self, image):
            return ()

    path = tmp_path / "empty.png"
    _make_checkerboard(640).save(path)

    result = evaluate_quality(path, person_detector=Detector())

    assert result.valid is False
    assert result.person_count == 0
    assert result.subject_visibility is None
    assert "subject not visible" in (result.reason or "")


def test_quality_rejects_tiny_subject(tmp_path):
    from lora_data_maker.processing.person import PersonBox

    class Detector:
        def detect(self, image):
            return (PersonBox(300, 300, 350, 350, 0.95),)

    path = tmp_path / "tiny.png"
    _make_checkerboard(640).save(path)

    result = evaluate_quality(
        path,
        person_detector=Detector(),
        min_subject_area=0.12,
    )

    assert result.valid is False
    assert result.person_count == 1
    assert result.subject_area_ratio < 0.12
    assert "subject too small" in (result.reason or "")


def test_quality_accepts_visible_single_subject(tmp_path):
    from lora_data_maker.processing.person import PersonBox

    class Detector:
        def detect(self, image):
            return (PersonBox(120, 40, 520, 610, 0.95),)

    path = tmp_path / "person.png"
    _make_checkerboard(640).save(path)

    result = evaluate_quality(path, person_detector=Detector())

    assert result.valid is True
    assert result.person_count == 1
    assert result.subject_visibility is not None
    assert 0.0 <= result.quality_score <= 1.0


def test_clean_manifest_propagates_person_detector_runtime_error(tmp_path):
    from lora_data_maker.models import ImageRecord, ProjectManifest
    from lora_data_maker.processing.clean import clean_manifest
    import pytest

    path = tmp_path / "person.png"
    _make_checkerboard(640).save(path)

    manifest = ProjectManifest(
        project_name="example",
        subject="example",
        trigger="example",
    )
    record = ImageRecord(path="person.png", status="validated")
    manifest.add_image(record)

    class BrokenDetector:
        def detect(self, image):
            raise RuntimeError("onnx inference failed")

    with pytest.raises(RuntimeError, match="onnx inference failed"):
        clean_manifest(
            manifest,
            root=tmp_path,
            person_detector=BrokenDetector(),
        )

    assert record.status == "validated"


def test_quality_rejects_detected_text(tmp_path):
    from lora_data_maker.processing.text import TextRegion

    class Detector:
        def detect(self, image):
            return (TextRegion(80, 100, 560, 150, 0.91),)

    path = tmp_path / "text.png"
    _make_checkerboard(640).save(path)

    result = evaluate_quality(path, text_detector=Detector())

    assert result.valid is False
    assert result.text_count == 1
    assert result.text_confidence == 0.91
    assert "text detected" in (result.reason or "")
