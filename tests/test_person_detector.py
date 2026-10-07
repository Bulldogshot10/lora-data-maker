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
    detector.INPUT_SIZE = (416, 416)
    detector.PERSON_CLASS_ID = 0
    detector.confidence_threshold = 0.5

    raw = np.zeros((1, 3549, 85), dtype=np.float32)
    raw[0, 0, :4] = [10.0, 10.0, 1.0, 1.0]
    raw[0, 0, 4] = 8.0
    raw[0, 0, 5] = 8.0

    decoded = detector._decode_outputs(raw, (416, 416))
    assert decoded.shape == (1, 5)
    assert decoded[0, 4] > 0.95
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
            raw[0, 0, 4] = 8
            raw[0, 0, 5] = 8
            return [raw]

    model = pytest.importorskip("tempfile")
    del model
    detector = YOLOXNanoPersonDetector.__new__(YOLOXNanoPersonDetector)
    detector._session = FakeSession()
    detector._input_name = "images"
    detector.confidence_threshold = 0.5
    detector.nms_threshold = 0.45

    boxes = detector.detect(Image.new("RGB", (416, 416), "white"))
    assert len(boxes) == 1
    assert boxes[0].confidence > 0.95


def test_sha256_helper(tmp_path):
    path = tmp_path / "sample.bin"
    path.write_bytes(b"abc")
    assert _sha256_file(path) == (
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    )
