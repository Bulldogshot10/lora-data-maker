from lora_data_maker.models import ImageRecord, ProjectManifest


def test_image_status_update():
    image = ImageRecord(path="raw/001.jpg")
    image.set_status("validated")
    assert image.status == "validated"
    assert image.rejection_reason is None


def test_rejected_image_keeps_reason():
    image = ImageRecord(path="raw/001.jpg")
    image.set_status("rejected", reason="too small")
    assert image.status == "rejected"
    assert image.rejection_reason == "too small"


def test_manifest_round_trip():
    manifest = ProjectManifest(project_name="julia", subject="julia", trigger="julia")
    image = ImageRecord(path="raw/001.jpg", sha256="abc")
    manifest.add_image(image)

    restored = ProjectManifest.from_dict(manifest.to_dict())
    assert restored.project_name == "julia"
    assert restored.images[image.image_id].sha256 == "abc"
    assert restored.counts()["discovered"] == 1


def test_manifest_rejects_invalid_image_status():
    import pytest

    manifest = {
        "schema_version": 1,
        "project_name": "julia",
        "subject": "julia",
        "trigger": "julia",
        "images": {
            "abc": {
                "image_id": "abc",
                "path": "raw/a.jpg",
                "status": "bogus",
                "tags": [],
                "metadata": {},
            }
        },
    }

    with pytest.raises(ValueError, match="invalid image status"):
        ProjectManifest.from_dict(manifest)


def test_manifest_rejects_non_string_tags():
    import pytest

    manifest = {
        "schema_version": 1,
        "project_name": "julia",
        "subject": "julia",
        "trigger": "julia",
        "images": {
            "abc": {
                "image_id": "abc",
                "path": "raw/a.jpg",
                "status": "discovered",
                "tags": ["ok", 123],
                "metadata": {},
            }
        },
    }

    with pytest.raises(ValueError, match="tags must be a list of strings"):
        ProjectManifest.from_dict(manifest)


def test_manifest_rejects_mismatched_image_key():
    import pytest

    manifest = {
        "schema_version": 1,
        "project_name": "julia",
        "subject": "julia",
        "trigger": "julia",
        "images": {
            "dictionary-key": {
                "image_id": "record-id",
                "path": "raw/a.jpg",
                "status": "discovered",
                "tags": [],
                "metadata": {},
            }
        },
    }

    with pytest.raises(ValueError, match="does not match"):
        ProjectManifest.from_dict(manifest)


def test_image_record_rejects_unsafe_image_id():
    import pytest

    for value in ("../evil", "sub/evil", r"sub\evil", "C:\\evil"):
        with pytest.raises(ValueError, match="invalid image id"):
            ImageRecord(image_id=value, path="raw/a.jpg")


def test_image_record_rejects_non_string_id():
    import pytest

    with pytest.raises(ValueError, match="image id must be a string"):
        ImageRecord(image_id=123, path="raw/a.jpg")
