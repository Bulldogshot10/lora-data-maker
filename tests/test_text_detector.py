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
