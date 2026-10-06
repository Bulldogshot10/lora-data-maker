import pytest
from PIL import Image

from lora_data_maker.processing.faces import FaceBox, YuNetFaceDetector, _sha256_file


def test_yunet_validates_parameters_before_optional_import():
    with pytest.raises(ValueError, match="confidence_threshold"):
        YuNetFaceDetector(confidence_threshold=0)

    with pytest.raises(ValueError, match="nms_threshold"):
        YuNetFaceDetector(nms_threshold=0)

    with pytest.raises(ValueError, match="top_k"):
        YuNetFaceDetector(top_k=0)


def test_yunet_coordinate_conversion_with_fake_runtime():
    class FakeArray:
        shape = (150, 200, 3)

    class FakeNP:
        uint8 = "uint8"

        @staticmethod
        def asarray(image, dtype=None):
            return FakeArray()

    class FakeCV:
        COLOR_RGB2BGR = 1

        @staticmethod
        def cvtColor(array, code):
            return array

    class FakeDetector:
        def setInputSize(self, size):
            self.input_size = size

        def detect(self, frame):
            return None, [
                [10.2, -5.4, 100.6, 120.8, 0.95],
                [-20, 30, 5, 5, 0.90],
            ]

    detector = YuNetFaceDetector.__new__(YuNetFaceDetector)
    detector._np = FakeNP()
    detector._cv = FakeCV()
    detector._detector = FakeDetector()

    faces = detector.detect(Image.new("RGB", (200, 150), "white"))

    assert faces == [FaceBox(left=10, top=0, right=111, bottom=115)]
    assert detector._detector.input_size == (200, 150)


def test_yunet_returns_empty_sequence_when_runtime_has_no_faces():
    class FakeArray:
        shape = (240, 320, 3)

    class FakeNP:
        uint8 = "uint8"

        @staticmethod
        def asarray(image, dtype=None):
            return FakeArray()

    class FakeCV:
        COLOR_RGB2BGR = 1

        @staticmethod
        def cvtColor(array, code):
            return array

    class FakeDetector:
        def setInputSize(self, size):
            self.input_size = size

        def detect(self, frame):
            return None, None

    detector = YuNetFaceDetector.__new__(YuNetFaceDetector)
    detector._np = FakeNP()
    detector._cv = FakeCV()
    detector._detector = FakeDetector()

    assert detector.detect(Image.new("RGB", (320, 240), "white")) == ()


@pytest.mark.face_smoke
def test_yunet_real_model_smoke(tmp_path):
    detector = YuNetFaceDetector(cache_dir=tmp_path / "face-cache")
    assert detector.detect(Image.new("RGB", (320, 240), "white")) == ()


def test_sha256_helper(tmp_path):
    path = tmp_path / "sample.bin"
    path.write_bytes(b"abc")
    assert _sha256_file(path) == (
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    )
