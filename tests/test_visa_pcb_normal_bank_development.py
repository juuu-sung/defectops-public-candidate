"""Post-test VisA bank fit separates official normal fit/calibration and excludes test."""

from __future__ import annotations

import csv
import hashlib
import io
import tarfile
import tempfile
from pathlib import Path

from PIL import Image

from defectops.visa_pcb_external_evaluation import _pixel_sha
from defectops.visa_pcb_normal_bank_development import (
    audit_training_normals, load_execution, training_members,
)


def _csv_for_pcb3() -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.writer(stream)
    writer.writerow(["object", "split", "label", "image", "mask"])
    for index in range(905):
        writer.writerow(["pcb3", "train", "normal", f"pcb3/Data/Images/Normal/{index:04d}.JPG", ""])
    writer.writerow(["pcb3", "test", "anomaly", "pcb3/Data/Images/Anomaly/000.JPG",
                     "pcb3/Data/Masks/Anomaly/000.png"])
    return stream.getvalue().encode("utf-8")


def test_training_uses_train_normals_minus_fixed_calibration_only() -> None:
    selected = [f"pcb3/Data/Images/Normal/{index:04d}.JPG" for index in range(56)]
    plan = {"records": [
        {"category": "pcb3", "protocol_split": "calibration", "source_image_member": name}
        for name in selected
    ]}
    fitted = training_members(_csv_for_pcb3(), plan, "pcb3")
    assert len(fitted) == 849
    assert fitted == sorted(fitted)
    assert not set(fitted).intersection(selected)
    assert not any("/Anomaly/" in name for name in fitted)
    assert fitted[0] == "pcb3/Data/Images/Normal/0056.JPG"


def test_training_rejects_calibration_not_present_in_official_train() -> None:
    plan = {"records": [
        {"category": "pcb3", "protocol_split": "calibration",
         "source_image_member": f"pcb3/Data/Images/Normal/{index:04d}.JPG"}
        for index in range(55)
    ] + [{"category": "pcb3", "protocol_split": "calibration",
           "source_image_member": "pcb3/Data/Images/Normal/9999.JPG"}]}
    try:
        training_members(_csv_for_pcb3(), plan, "pcb3")
    except ValueError as error:
        assert "split differs" in str(error)
    else:
        raise AssertionError("Foreign calibration image should fail")


def test_config_is_explicit_post_test_and_keeps_prior_architecture() -> None:
    path = Path(__file__).resolve().parents[1] / "configs" / "visa-pcb-normal-bank-development-v1.toml"
    config = load_execution(path)
    assert config["status"] == "post_test_exploratory_normal_bank_development"
    assert config["first_pilot_category"] == "pcb3"
    assert config["layers"] == ["layer1", "layer2"]
    assert config["top_fraction"] == 0.01
    assert config["official_test_payloads_allowed"] is False
    assert config["confirmatory_claim_allowed"] is False


def test_normal_audit_rejects_exact_calibration_duplicate() -> None:
    image = Image.new("RGB", (16, 16), "#123456")
    stream = io.BytesIO()
    image.save(stream, format="JPEG")
    data = stream.getvalue()
    name = "pcb3/Data/Images/Normal/0001.JPG"
    with tempfile.TemporaryDirectory() as temporary:
        path = Path(temporary) / "source.tar"
        with tarfile.open(path, "w") as archive:
            member = tarfile.TarInfo(name)
            member.size = len(data)
            archive.addfile(member, io.BytesIO(data))
        rows = audit_training_normals(path, [name], [], {}, "pcb3")
        assert len(rows) == 1
        assert rows[0]["image_sha256"] == hashlib.sha256(data).hexdigest()
        try:
            audit_training_normals(
                path, [name], [{"source_image_member": "calibration"}],
                {"calibration": {"decoded_pixel_sha256": _pixel_sha(Image.open(io.BytesIO(data))) }},
                "pcb3",
            )
        except ValueError as error:
            assert "duplicate" in str(error)
        else:
            raise AssertionError("Exact calibration duplicate should fail")
