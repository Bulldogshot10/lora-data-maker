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
