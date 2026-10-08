import os

import numpy as np
import pytest
from PIL import Image

from lora_data_maker.processing.person import (
    PersonBox,
    YOLOXNanoPersonDetector,
    _nms,
    _sha256_file,
)


def test_person_detector_validates_parameters_before_runtime_import(tmp_path):
    model = tmp_path / "model.onnx"
    model.write_bytes(b"model")

    with pytest.raises(ValueError, match="confidence_threshold"):
        YOLOXNanoPersonDetector(model_path=model, confidence_threshold=0)

    with pytest.raises(ValueError, match="nms_threshold"):
        YOLOXNanoPersonDetector(model_path=model, nms_threshold=0)


def test_person_box_properties():
    box = PersonBox(10, 20, 110, 220, 0.9)
    assert box.width == 100
    assert box.height == 200
    assert box.area == 20_000


def test_nms_keeps_highest_overlapping_box():
    boxes = np.array(
        [[0, 0, 100, 100], [10, 10, 110, 110], [200, 200, 300, 300]],
        dtype=np.float32,
    )
    scores = np.array([0.95, 0.80, 0.70], dtype=np.float32)

    assert _nms(boxes, scores, 0.5) == [0, 2]


def test_yolox_output_decode_416():
    detector = YOLOXNanoPersonDetector.__new__(YOLOXNanoPersonDetector)
    detector._np = np
    detector.INPUT_SIZE = (416, 416)
    detector.PERSON_CLASS_ID = 0
    detector.confidence_threshold = 0.5

    raw = np.zeros((1, 3549, 85), dtype=np.float32)
    raw[0, 0, :4] = [10.0, 10.0, 1.0, 1.0]
    raw[0, 0, 4] = 0.9
    raw[0, 0, 5] = 0.9

    decoded = detector._decode_outputs(raw, (416, 416))
    assert decoded.shape == (1, 5)
    assert decoded[0, 4] == pytest.approx(0.81, rel=1e-5)
    assert decoded[0, 0] < 90
    assert decoded[0, 1] < 90


def test_person_detector_fake_session():
    class FakeInput:
        name = "images"

    class FakeSession:
        def get_inputs(self):
            return [FakeInput()]

        def run(self, output_names, inputs):
            raw = np.zeros((1, 3549, 85), dtype=np.float32)
            raw[0, 0, :4] = [10, 10, 3, 3]
            raw[0, 0, 4] = 0.9
            raw[0, 0, 5] = 0.9
            return [raw]

    detector = YOLOXNanoPersonDetector.__new__(YOLOXNanoPersonDetector)
    detector._session = FakeSession()
    detector._input_name = "images"
    detector._np = np
    detector.confidence_threshold = 0.5
    detector.nms_threshold = 0.45

    boxes = detector.detect(Image.new("RGB", (416, 416), "white"))
    assert len(boxes) == 1
    assert boxes[0].confidence == pytest.approx(0.81, rel=1e-5)


def test_sha256_helper(tmp_path):
    path = tmp_path / "sample.bin"
    path.write_bytes(b"abc")
    assert _sha256_file(path) == (
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    )


def test_yolox_preprocess_uses_top_left_float32_pixel_contract():
    detector = YOLOXNanoPersonDetector.__new__(YOLOXNanoPersonDetector)
    detector._np = np
    detector.INPUT_SIZE = (416, 416)

    image = Image.new("RGB", (208, 416), (255, 0, 0))
    tensor = detector._preprocess(image)

    assert tensor.shape == (1, 3, 416, 416)
    assert tensor.dtype == np.float32
    assert tensor.flags["C_CONTIGUOUS"]
    assert tensor[0, 0, 0, 0] == pytest.approx(0.0)
    assert tensor[0, 1, 0, 0] == pytest.approx(0.0)
    assert tensor[0, 2, 0, 0] == pytest.approx(255.0)
    assert tensor[0, :, 0, 208].tolist() == pytest.approx([114.0, 114.0, 114.0])


def test_yolox_decode_uses_top_left_padding_geometry():
    detector = YOLOXNanoPersonDetector.__new__(YOLOXNanoPersonDetector)
    detector._np = np
    detector.INPUT_SIZE = (416, 416)
    detector.PERSON_CLASS_ID = 0
    detector.confidence_threshold = 0.5

    raw = np.zeros((1, 3549, 85), dtype=np.float32)
    raw[0, 0, :4] = [10.0, 10.0, 1.0, 1.0]
    raw[0, 0, 4] = 0.9
    raw[0, 0, 5] = 0.9

    decoded = detector._decode_outputs(raw, (208, 416))

    assert decoded.shape == (1, 5)
    assert decoded[0, 0] == pytest.approx(69.1274, abs=1e-4)
    assert decoded[0, 1] == pytest.approx(69.1274, abs=1e-4)


def test_yolox_real_model_smoke():
    model_path = os.environ.get("LORA_DATA_MAKER_YOLOX_MODEL")
    if not model_path:
        pytest.skip("real YOLOX model not provided")

    detector = YOLOXNanoPersonDetector(model_path=model_path)
    boxes = detector.detect(Image.new("RGB", (416, 416), "black"))

    assert isinstance(boxes, tuple)
