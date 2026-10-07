from pathlib import Path

from lora_data_maker.models import ImageRecord, ProjectManifest
from lora_data_maker.taggers import TagResult, TaggerError, WD14Labels, mcut_threshold, tag_manifest
from lora_data_maker.taggers.model import cache_key


def test_wd14_labels_loads_categories_and_normalizes_names(tmp_path):
    labels = tmp_path / "selected_tags.csv"
    labels.write_text(
        "tag_id,name,category,count\n"
        "0,safe,9,1\n"
        "1,blue_eyes,0,10\n"
        "2,julia,4,10\n"
        "3,1girl,0,10\n"
        "4,smile,0,10\n"
        "5,^_^,0,10\n",
        encoding="utf-8",
    )

    result = WD14Labels.load(labels)

    assert result.names == ["safe", "blue eyes", "julia", "1girl", "smile", "^_^"]
    assert result.rating_indexes == [0]
    assert result.general_indexes == [1, 3, 4, 5]
    assert result.character_indexes == [2]


def test_mcut_threshold_finds_largest_probability_gap():
    assert mcut_threshold([0.95, 0.91, 0.90, 0.20, 0.10]) == 0.55
    assert mcut_threshold([0.8]) == 0.0


def test_cache_key_is_stable_and_safe():
    first = cache_key("SmilingWolf/wd-eva02-large-tagger-v3", "main")
    second = cache_key("SmilingWolf/wd-eva02-large-tagger-v3", "main")

    assert first == second
    assert "/" not in first
    assert "@" not in first


class FakeTagger:
    def tag_images(self, paths):
        return [
            TagResult(
                tags=["1girl", "smile"],
                scores={"1girl": 0.99, "smile": 0.88},
                ratings={"safe": 0.97},
                characters={"julia": 0.91},
            )
            for _ in paths
        ]


def test_tag_manifest_updates_processed_records(tmp_path):
    processed = tmp_path / "work" / "processed"
    processed.mkdir(parents=True)

    source = processed / "image.jpg"
    source.write_bytes(b"placeholder")

    manifest = ProjectManifest(
        project_name="example",
        subject="example",
        trigger="example",
    )
    record = ImageRecord(
        path="raw/source.jpg",
        processed_path="work/processed/image.jpg",
        status="processed",
    )
    manifest.add_image(record)

    summary = tag_manifest(
        manifest,
        root=tmp_path,
        tagger=FakeTagger(),
    )

    assert summary.scanned == 1
    assert summary.tagged == 1
    assert manifest.images[record.image_id].status == "tagged"
    assert manifest.images[record.image_id].tags == ["1girl", "smile"]
    assert manifest.images[record.image_id].metadata["tagger_ratings"]["safe"] == 0.97


def test_tag_manifest_no_candidates():
    manifest = ProjectManifest(
        project_name="example",
        subject="example",
        trigger="example",
    )

    summary = tag_manifest(
        manifest,
        root=Path("."),
        tagger=FakeTagger(),
    )

    assert summary.scanned == 0
    assert summary.tagged == 0


class FakeInput:
    shape = [1, 448, 448, 3]
    name = "input"


class FakeOutput:
    name = "output"


class FakeSession:
    def get_inputs(self):
        return [FakeInput()]

    def get_outputs(self):
        return [FakeOutput()]

    def run(self, output_names, inputs):
        batch = next(iter(inputs.values()))
        import numpy as np

        predictions = np.zeros((len(batch), 5), dtype=np.float32)
        predictions[:, 0] = 0.99
        predictions[:, 1] = 0.92
        predictions[:, 2] = 0.87
        predictions[:, 3] = 0.80
        predictions[:, 4] = 0.70
        return [predictions]


class FakeCache:
    def __init__(self, model_path, labels_path):
        from lora_data_maker.taggers.model import ModelFiles

        self.files = ModelFiles(
            model_path=model_path,
            labels_path=labels_path,
        )

    def ensure(self, *, model_sha256=None):
        return self.files


def test_wd14_fake_inference_runs_preprocess_and_decode(tmp_path):
    import numpy as np

    model_path = tmp_path / "model.onnx"
    labels_path = tmp_path / "selected_tags.csv"
    model_path.write_bytes(b"fake")

    labels_path.write_text(
        "tag_id,name,category,count\n"
        "0,safe,9,1\n"
        "1,blue_eyes,0,10\n"
        "2,julia,4,10\n"
        "3,smile,0,10\n"
        "4,watermark,0,10\n",
        encoding="utf-8",
    )

    first = tmp_path / "first.jpg"
    second = tmp_path / "second.jpg"
    from PIL import Image

    Image.new("RGB", (800, 600), (20, 30, 40)).save(first, format="JPEG")
    Image.new("RGB", (400, 800), (40, 30, 20)).save(second, format="JPEG")

    from lora_data_maker.taggers.wd14 import WD14Tagger

    tagger = WD14Tagger(
        model_cache=FakeCache(model_path, labels_path),
        threshold=0.75,
        character_threshold=0.85,
        batch_size=2,
        session_factory=lambda _path: FakeSession(),
    )

    results = tagger.tag_images([first, second])

    assert len(results) == 2
    assert results[0].tags == ["blue eyes", "julia", "smile"]
    assert abs(results[0].ratings["safe"] - 0.99) < 1e-6



def test_mobile_wd14_labels_loads_tags_json(tmp_path):
    import json

    labels = tmp_path / "tags.json"
    labels.write_text(
        json.dumps(
            {
                "0": ["safe", 9, 1],
                "1": ["blue_eyes", 0, 10],
                "2": ["julia", 4, 10],
                "3": ["smile", 0, 10],
            }
        ),
        encoding="utf-8",
    )

    from lora_data_maker.taggers.mobile import MobileWD14Labels

    result = MobileWD14Labels.load(labels)

    assert result.names == ["safe", "blue eyes", "julia", "smile"]
    assert result.rating_indexes == [0]
    assert result.general_indexes == [1, 3]
    assert result.character_indexes == [2]


class MobileFakeInput:
    shape = [1, "height", "width", 3]
    name = "pixel_values"
    type = "tensor(uint8)"


class MobileFakeOutput:
    name = "probabilities"


class MobileFakeSession:
    def get_inputs(self):
        return [MobileFakeInput()]

    def get_outputs(self):
        return [MobileFakeOutput()]

    def run(self, output_names, inputs):
        batch = next(iter(inputs.values()))
        import numpy as np

        assert batch.dtype == np.uint8
        assert batch.ndim == 4
        assert batch.shape[0] == 1
        return [
            np.array(
                [[0.99, 0.92, 0.87, 0.80]],
                dtype=np.float32,
            )
        ]


def test_mobile_wd14_fake_inference_uses_uint8_input(tmp_path):
    import json
    from PIL import Image

    model_path = tmp_path / "model.onnx"
    labels_path = tmp_path / "tags.json"
    model_path.write_bytes(b"fake")
    labels_path.write_text(
        json.dumps(
            {
                "0": ["safe", 9, 1],
                "1": ["blue_eyes", 0, 10],
                "2": ["julia", 4, 10],
                "3": ["smile", 0, 10],
            }
        ),
        encoding="utf-8",
    )

    image = tmp_path / "image.png"
    Image.new("RGB", (320, 240), (20, 30, 40)).save(image)

    from lora_data_maker.taggers.mobile import MobileWD14Tagger

    tagger = MobileWD14Tagger(
        model_cache=FakeCache(model_path, labels_path),
        threshold=0.75,
        character_threshold=0.85,
        session_factory=lambda _path: MobileFakeSession(),
    )

    result = tagger.tag_image(image)

    assert result.tags == ["blue eyes", "julia", "smile"]
    assert abs(result.characters["julia"] - 0.87) < 1e-6
    assert abs(result.ratings["safe"] - 0.99) < 1e-6


def test_build_tagger_presets():
    from lora_data_maker.taggers import POTATO, SWEETSPOT
    from lora_data_maker.taggers.manager import build_tagger
    from lora_data_maker.taggers.mobile import MobileWD14Tagger
    from lora_data_maker.taggers.wd14 import WD14Tagger

    sweetspot = build_tagger({}, preset="sweetspot")
    potato = build_tagger({}, preset="potato")

    assert sweetspot.model_cache.repo_id == SWEETSPOT.repo_id
    assert isinstance(sweetspot, WD14Tagger)
    assert potato.model_cache.repo_id == POTATO.repo_id
    assert potato.model_cache.model_filename == POTATO.model_filename
    assert isinstance(potato, MobileWD14Tagger)


def test_wd14_model_load_failure_becomes_tagger_error(tmp_path):
    from lora_data_maker.taggers.wd14 import WD14Tagger

    class BrokenCache:
        def ensure(self, *, model_sha256=None):
            raise RuntimeError("download exploded")

    tagger = WD14Tagger(
        model_cache=BrokenCache(),
        session_factory=lambda _path: None,
    )

    import pytest
    with pytest.raises(TaggerError, match="could not load WD14 tagger model"):
        tagger.tag_image(tmp_path / "missing.jpg")


def test_mobile_model_load_failure_becomes_tagger_error(tmp_path):
    from lora_data_maker.taggers.mobile import MobileWD14Tagger

    class BrokenCache:
        def ensure(self, *, model_sha256=None):
            raise RuntimeError("download exploded")

    tagger = MobileWD14Tagger(
        model_cache=BrokenCache(),
        session_factory=lambda _path: None,
    )

    import pytest
    with pytest.raises(TaggerError, match="could not load mobile WD14 tagger model"):
        tagger.tag_image(tmp_path / "missing.jpg")



class JoyTagFakeInput:
    shape = [1, 3, 448, 448]
    name = "input"
    type = "tensor(float)"


class JoyTagFakeOutput:
    name = "output"


class JoyTagFakeSession:
    def get_inputs(self):
        return [JoyTagFakeInput()]

    def get_outputs(self):
        return [JoyTagFakeOutput()]

    def run(self, output_names, inputs):
        batch = next(iter(inputs.values()))
        import numpy as np

        assert batch.dtype == np.float32
        assert batch.ndim == 4
        assert batch.shape[0] == 2
        assert batch.shape[1:] == (3, 448, 448)
        return [
            np.array(
                [
                    [2.0, 0.0, -2.0, 1.0],
                    [-2.0, 2.0, 0.0, -1.0],
                ],
                dtype=np.float32,
            )
        ]


def test_joytag_labels_loads_top_tags(tmp_path):
    labels = tmp_path / "top_tags.txt"
    labels.write_text(
        "1girl\n"
        "brown_hair\n"
        "photo_(medium)\n",
        encoding="utf-8",
    )

    from lora_data_maker.taggers.joytag import JoyTagLabels

    result = JoyTagLabels.load(labels)

    assert result.names == ["1girl", "brown_hair", "photo_(medium)"]


def test_joytag_fake_inference_uses_nchw_and_sigmoid(tmp_path):
    import numpy as np
    from PIL import Image

    model_path = tmp_path / "model.onnx"
    labels_path = tmp_path / "top_tags.txt"
    model_path.write_bytes(b"fake")
    labels_path.write_text(
        "tag_a\n"
        "tag_b\n"
        "tag_c\n"
        "tag_d\n",
        encoding="utf-8",
    )

    first = tmp_path / "first.jpg"
    second = tmp_path / "second.jpg"
    Image.new("RGB", (800, 600), (255, 0, 0)).save(first, format="JPEG")
    Image.new("RGB", (400, 800), (0, 255, 0)).save(second, format="JPEG")

    from lora_data_maker.taggers.joytag import JoyTagTagger

    tagger = JoyTagTagger(
        model_cache=FakeCache(model_path, labels_path),
        threshold=0.6,
        batch_size=2,
        session_factory=lambda _path: JoyTagFakeSession(),
    )

    results = tagger.tag_images([first, second])

    assert len(results) == 2
    assert results[0].tags == ["tag_a", "tag_d"]
    assert results[1].tags == ["tag_b"]
    assert abs(results[0].scores["tag_a"] - 0.880797) < 1e-5
    assert abs(results[0].scores["tag_d"] - 0.731059) < 1e-5


def test_joytag_uses_profile_defaults_and_preset():
    from lora_data_maker.taggers import JOYTAG
    from lora_data_maker.taggers.joytag import JoyTagTagger
    from lora_data_maker.taggers.manager import build_tagger

    tagger = build_tagger({}, preset="joytag")

    assert JOYTAG.repo_id == "fancyfeast/joytag"
    assert JOYTAG.model_filename == "model.onnx"
    assert JOYTAG.labels_filename == "top_tags.txt"
    assert JOYTAG.default_threshold == 0.4
    assert isinstance(tagger, JoyTagTagger)
    assert abs(tagger.threshold - 0.4) < 1e-9
    assert tagger.model_cache.revision == JOYTAG.revision


def test_joytag_invalid_input_shape_becomes_tagger_error(tmp_path):
    from lora_data_maker.taggers.joytag import JoyTagTagger

    class BrokenInput:
        shape = [1, 4, 448, 448]
        name = "input"
        type = "tensor(float)"

    class BrokenSession:
        def get_inputs(self):
            return [BrokenInput()]

        def get_outputs(self):
            return [JoyTagFakeOutput()]

    model_path = tmp_path / "model.onnx"
    labels_path = tmp_path / "top_tags.txt"
    model_path.write_bytes(b"fake")
    labels_path.write_text("tag\n", encoding="utf-8")

    tagger = JoyTagTagger(
        model_cache=FakeCache(model_path, labels_path),
        session_factory=lambda _path: BrokenSession(),
    )

    import pytest

    with pytest.raises(TaggerError, match="unexpected JoyTag ONNX input shape"):
        tagger.tag_image(tmp_path / "missing.jpg")
