import json

from lora_data_maker.manifest import create_manifest, load_manifest, register_image, save_manifest


def test_manifest_atomic_round_trip(tmp_path):
    manifest = create_manifest(
        project_name="julia",
        subject="julia",
        trigger="julia",
        source={"backend": "google", "query": "julia"},
    )
    image = register_image(
        manifest,
        path="raw/001.jpg",
        source_url="https://example.com/001.jpg",
        source_query="julia",
        status="downloaded",
    )
    assert image.status == "downloaded"

    path = tmp_path / "manifest.json"
    save_manifest(manifest, path)

    raw = json.loads(path.read_text(encoding="utf-8"))
    assert raw["subject"] == "julia"
    assert raw["trigger"] == "julia"
    assert raw["images"][image.image_id]["source_url"] == "https://example.com/001.jpg"

    restored = load_manifest(path)
    assert restored.images[image.image_id].status == "downloaded"


def test_missing_manifest_fails(tmp_path):
    try:
        load_manifest(tmp_path / "manifest.json")
    except Exception as exc:
        assert "manifest file not found" in str(exc)
    else:
        raise AssertionError("expected manifest load to fail")
