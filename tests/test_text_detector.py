from PIL import Image
import pytest

from lora_data_maker.processing.text import PPOCRTextDetector, TextRegion


def test_text_region_properties():
    region = TextRegion(10, 20, 110, 70, 0.9)
    assert region.width == 100
    assert region.height == 50
    assert region.area == 5000


def test_text_detector_validates_parameters_before_runtime(tmp_path):
    model = tmp_path / "model.onnx"
    model.write_bytes(b"model")

    with pytest.raises(ValueError, match="text confidence_threshold"):
        PPOCRTextDetector(
            model_path=model,
            confidence_threshold=0,
        )

    with pytest.raises(ValueError, match="text min_area_ratio"):
        PPOCRTextDetector(
            model_path=model,
            min_area_ratio=-0.1,
        )


def test_text_detector_scales_boxes_and_filters_small_regions():
    import numpy as np

    class CV2:
        COLOR_RGB2BGR = 1
        @staticmethod
        def cvtColor(array, code):
            return array[:, :, ::-1]
        @staticmethod
        def resize(array, size, interpolation=None):
            return array
        INTER_LINEAR = 1

    class FakeDetector:
        def detect(self, image):
            return (
                [np.array([[10, 20], [110, 20], [110, 30], [10, 30]], dtype=np.float32)],
                [0.9],
            )

    detector = PPOCRTextDetector.__new__(PPOCRTextDetector)
    detector._cv2 = CV2
    detector._detector = FakeDetector()
    detector.INPUT_SIZE = (736, 736)
    detector.confidence_threshold = 0.5
    detector.min_area_ratio = 0.0001

    regions = detector.detect(Image.new("RGB", (1472, 736), "white"))

    assert len(regions) == 1
    assert regions[0].left == 20
    assert regions[0].top == 20
    assert regions[0].right == 220
    assert regions[0].bottom == 30
    assert regions[0].confidence == pytest.approx(0.9)


def test_ppocr_real_model_smoke():
    import os

    model_path = os.environ.get("LORA_DATA_MAKER_TEXTOCR_MODEL")
    image_path = os.environ.get("LORA_DATA_MAKER_TEXTOCR_IMAGE")
    if not model_path or not image_path:
        pytest.skip("real PP-OCR model/image not provided")

    detector = PPOCRTextDetector(model_path=model_path)
    with Image.open(image_path) as image:
        regions = detector.detect(image.convert("RGB"))

    assert isinstance(regions, tuple)
    assert len(regions) > 0
