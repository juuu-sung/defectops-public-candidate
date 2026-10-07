"""Export selected ORIGINAL normal PNGs and saved maps; no GPU or new inference."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import zipfile

import numpy as np
from PIL import Image

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parents[1]/"src")]
from scripts import analyze_mpi_patchcore_scaling_sensitivity as a
from scripts import run_mpi_patchcore_scaling as engine

NAME = "mpi-patchcore-scaling-normal-review-v1"


def selection(report):
    selected = {}
    def add(key, region, reason):
        selected.setdefault((key, region), set()).add(reason)
    for c in report["normal_cases"]:
        add(c["key"], c["candidate_winning_region"], "new_fp_or_frozen_high_normal")
    for scope, summary in report["summaries"].items():
        for x in summary["bad_draw_boundary_owners"][:3]:
            add(x["key"], x["region"], scope+"_frequent_bad_boundary")
        pivot = summary["worst_draw_development_pivots"]["q90_higher"]
        add(pivot["key"], report["dominant_defect_region"], scope+"_q90_pivot")
    pivot = report["fixed_development_pivots"]["q90_higher"]
    add(pivot["key"], report["dominant_defect_region"], "fixed_q90_pivot")
    return [{"key": k, "region": r, "reasons": sorted(why)} for (k, r), why in sorted(selected.items())]


def checked_png(source, row):
    raw = source.read(row["member"])
    if hashlib.sha256(raw).hexdigest() != row["member_sha256"]:
        raise ValueError("Source image bytes differ")
    with Image.open(io.BytesIO(raw)) as image:
        if image.format != "PNG" or image.mode != "RGB" or image.size != (256, 256):
            raise ValueError("Expected 256x256 RGB PNG")
        image.verify()
    with Image.open(io.BytesIO(raw)) as image:
        image.load()
        if hashlib.sha256(image.tobytes()).hexdigest() != row["decoded_rgb_sha256"]:
            raise ValueError("Source pixels differ")
    return raw


def write_bundle(path, payload):
    manifest = {n: hashlib.sha256(raw).hexdigest() for n, raw in payload.items()}
    if path.exists():
        with zipfile.ZipFile(path) as z:
            if (z.testzip() is not None or len(z.namelist()) != len(set(z.namelist())) or
                    set(z.namelist()) != set(payload) | {"handoff-manifest.json"} or
                    json.loads(z.read("handoff-manifest.json")) != manifest or
                    any(hashlib.sha256(z.read(n)).hexdigest() != sha for n, sha in manifest.items())):
                raise ValueError("Existing review ZIP differs; preserve it")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(prefix=".scaling-review-", suffix=".zip", dir=path.parent, delete=False) as f:
        temp = Path(f.name)
    with zipfile.ZipFile(temp, "w", zipfile.ZIP_DEFLATED) as z:
        for name, raw in payload.items():
            z.writestr(name, raw)
        z.writestr("handoff-manifest.json", json.dumps(manifest, sort_keys=True))
    if path.exists():
        raise ValueError("Output appeared during export; preserve temporary ZIP")
    temp.rename(path)


def run(project, archive):
    handoff = project/"artifacts/handoff"
    data = a.checked_bundle(handoff/"mpi-patchcore-scaling-v1.zip", a.INPUT_SHA)
    base = a.checked_bundle(handoff/"mpi-score-validation-v1.zip", a.BASE_SHA)
    report = a.analyze(data, base)
    plan = data["plan.json"]
    if a.p.fit.sha256_file(archive) != plan["archive_sha256"]:
        raise ValueError("Original archive differs")
    sig = a.p.fit.digest(plan)
    selected = selection(report)
    by_key = {p["key"]: (g, i, p) for g, ps in plan["cohorts"].items() for i, p in enumerate(ps)}
    payload = {}
    for_source = {}
    source_cache = {}
    with zipfile.ZipFile(archive) as z:
        for n, entry in enumerate(selected):
            group, i, photo = by_key[entry["key"]]
            if photo["label"] != 0:
                raise ValueError("Normal-only review; refusing anomaly")
            stage = "first-"+group if group in ("development", "calibration") else group
            root = project/"artifacts"/(a.p.NAME if stage.startswith("first-") else "analyses/"+a.p.NAME)
            name = f"photo-{i:03d}"
            receipts = data["prediction-receipts.json" if stage.startswith("first-") else "evaluation/prediction-receipts.json"]
            expected = receipts[f"{stage}\\{name}\\receipt.json"]
            if entry["key"] not in source_cache:
                actual = a.p.fit.read_phase(root/stage, name, sig, ["sample.json", "maps.npy"])
                if actual != expected:
                    raise ValueError("Saved map receipt differs from reviewed handoff")
                sample = engine.read_sample(root/stage, name, sig, photo, data["fit-report.json"]["checkpoint_sha256"])
                source_cache[entry["key"]] = sample
            else:
                sample = source_cache[entry["key"]]
            j = a.p.fit.REGIONS.index(entry["region"])
            values = a.p.array(root/stage/name/"maps.npy", (12, 256, 256))[j]
            buffer = io.BytesIO(); np.save(buffer, values, allow_pickle=False)
            prefix = f"tiles/{n:03d}"
            payload[prefix+"/source.png"] = checked_png(z, photo["regions"][j])
            payload[prefix+"/map.npy"] = buffer.getvalue()
            for_source[prefix] = {**entry, "source_row": photo["regions"][j],
                "sample_region": sample["regions"][j], "photo_sha256": a.p.fit.digest(photo),
                "saved_receipt": expected, "contract": data["contract.json"]}
            print(f"Read-only normal map export {n+1}/{len(selected)}", flush=True)
    metadata = {"status": "normal_map_review_exported_no_inference", "selection": for_source,
                "input_sha256": a.INPUT_SHA, "archive_sha256": plan["archive_sha256"],
                "checkpoint_sha256": data["fit-report.json"]["checkpoint_sha256"],
                "automatic_pass_allowed": False, "service_model_changed": False,
                "rights": "Private project review only; do not publish source images or maps",
                "scope": "Candidate saved maps only; not nearest-training-patch attribution or visual causal proof"}
    payload["review.json"] = json.dumps(metadata, ensure_ascii=False, sort_keys=True).encode()
    payload["sensitivity.json"] = json.dumps(report, ensure_ascii=False, sort_keys=True).encode()
    output = handoff/(NAME+".zip")
    with a.p.fit.locked_run(project/"artifacts/analyses"/(NAME+"-export-lock")):
        write_bundle(output, payload)
    return {"status": metadata["status"], "tiles": len(selected), "output": str(output)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--archive", type=Path, required=True)
    args = parser.parse_args(argv)
    print(json.dumps(run(args.project, args.archive), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
