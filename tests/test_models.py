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
