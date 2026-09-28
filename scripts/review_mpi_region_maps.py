"""Review saved shared/same-region MPI maps on CPU; never invoke a model."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import shutil
import sys
import zipfile
from pathlib import Path

import numpy as np

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parents[1]/"src")]
from scripts import compare_mpi_regions as region
from scripts import mpi_region_map_review_protocol as q
from scripts import mpi_region_map_review_cards as cards

fit, ev, p = q.fit, q.ev, q.p
CASE_FILES = ["case.json", "card-1.jpg", "card-2.jpg"]
REFERENCE_NAMES = [f"cards/calibration-0411-{suffix}.jpg" for suffix in ("1", "2", "neighbors")]


def close_numbers(a, b):
    """Only permit cross-platform floating point round-off in metric replay."""
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(close_numbers(a[k], b[k]) for k in a)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(close_numbers(x, y) for x, y in zip(a, b, strict=True))
    if isinstance(a, float) and isinstance(b, float):
        return np.isfinite(a) and np.isfinite(b) and abs(a-b) <= 1e-12
    return type(a) is type(b) and a == b


def load_inputs(project, handoffs):
    data = ev.checked_bundle(handoffs/f"{p.NAME}.zip", q.REGION_ZIP_SHA256)
    baseline = ev.checked_bundle(handoffs/"mpi-pcb-normal-bank-evaluation-v1.zip", p.scoring.EVALUATION_SHA256)
    reviewed = ev.checked_bundle(handoffs/"mpi-pcb-score-probe-v1.zip", p.trace.SCORE_ZIP_SHA256)
    previous, _ = ev.load_prior(project, handoffs)
    identity = data["plan.json"]["identity"]
    if (fit.digest(identity) != data["comparison.json"]["execution_sha256"]
            or identity["policy"] != p.POLICY
            or data["shared-A-calibration.json"] != reviewed["calibration.json"]):
        raise ValueError("Frozen comparison identity or shared contract changed")
    hashes = identity["implementation"]["source_files_lf_sha256"]
    for name, value in hashes.items():
        if hashlib.sha256((project/name).read_bytes().replace(b"\r\n", b"\n")).hexdigest() != value:
            raise ValueError(f"Frozen implementation changed: {name}")
    raw = data["raw-scores.json"]
    contract = p.calibrate(*[[s["arms"][p.ARMS[1]] for s in raw[g]] for g in ("development", "calibration")])
    if contract != data["candidate-calibration.json"]:
        raise ValueError("Candidate normal-only contract does not reproduce")
    cs = q.contracts(data)
    for group, samples in raw.items():
        if len(samples) != len(reviewed["raw-scores.json"][group]):
            raise ValueError("Frozen sample cohort differs")
        for sample, expected in zip(samples, reviewed["raw-scores.json"][group], strict=True):
            p.check_shared(sample["arms"][p.ARMS[0]], expected, data["shared-A-calibration.json"])
    shared = p.classify([s["arms"][p.ARMS[0]] for s in raw["test"]], **cs[p.ARMS[0]])
    report, scores = p.compare([s["arms"][p.ARMS[1]] for s in raw["test"]], contract, shared)
    if (scores != data["scores.json"] or shared != reviewed["scores.json"][p.trace.A]
            or any(not close_numbers(v, data["comparison.json"][k]) for k, v in report.items())):
        raise ValueError("Frozen scores or classification do not reproduce")
    keys = previous["cohort-amendment.json"]["photo_keys"]
    if ([s["key"] for s in shared] != keys or [s["key"] for s in baseline["scores.json"]] != keys
            or len(keys) != 109 or sum(s["label"] for s in shared) != 55):
        raise ValueError("Fixed 109-photo cohort differs")
    if fit.digest(baseline["input-lock.json"]["identity"]) != baseline["evaluation.json"]["execution_sha256"]:
        raise ValueError("Baseline map identity differs")
    selection = q.select(scores)
    if len(selection) != 30 or sum(v["group"] == "new_fn" for v in selection) != 15:
        raise ValueError("Reviewed 23 changes plus seven controls differ")
    # Check the historical ZIP before copying any reference images.
    trace = ev.checked_bundle(handoffs/"mpi-pcb-patch-trace-v1.zip", p.TRACE_SHA256)
    if trace["A-calibration.json"] != data["shared-A-calibration.json"]:
        raise ValueError("Historical reference belongs to another A contract")
    return data, baseline, previous, selection


def read_maps(path, receipt):
    raw = (path/"maps.npz").read_bytes()
    if hashlib.sha256(raw).hexdigest() != receipt["files"]["maps.npz"]:
        raise ValueError("Map changed after receipt verification")
    with np.load(io.BytesIO(raw), allow_pickle=False) as z:
        return z["anomaly_maps"].copy()


def read_pair(baseline_root, candidate_root, ordinal, photo, data, baseline):
    name = f"photo-{ordinal:03d}"
    before, br = ev.read_sample(baseline_root/"samples", name,
                              fit.digest(baseline["input-lock.json"]["identity"]), photo)
    if before["photo"] != baseline["scores.json"][ordinal-1]:
        raise ValueError("Baseline cached photo differs from pinned evaluation")
    signature = fit.digest(data["plan.json"]["identity"])
    after, cr = region.read_phase(candidate_root/"test", name, signature, region.sample_files(True))
    after = after["sample.json"]
    expected = data["raw-scores.json"]["test"][ordinal-1]
    if after != expected:
        raise ValueError("Candidate cached sample differs from pinned comparison")
    if (cr["photo_sha256"] != fit.digest(photo) or cr["routing_sha256"] != fit.digest(data["routing.json"])
            or cr["contract_sha256"] != fit.digest(data["candidate-calibration.json"])):
        raise ValueError("Candidate receipt belongs to another photo/routing/contract")
    region.check_sample(after, photo, expected["arms"][p.ARMS[0]],
                        data["shared-A-calibration.json"], True, candidate_root/"test"/name)
    old_top = {k: before["photo"][k] for k in ("key", "photo_id", "label")} | {"regions": [
        {"region": r["region"], "member": r["member"], "top1": r["score"]} for r in before["photo"]["regions"]]}
    p.check_shared(old_top, expected["arms"][p.ARMS[0]], data["shared-A-calibration.json"])
    arrays = {p.ARMS[0]: read_maps(baseline_root/"samples"/name, br),
              p.ARMS[1]: read_maps(candidate_root/"test"/name, cr)}
    return after["arms"], arrays, {p.ARMS[0]: br, p.ARMS[1]: cr}


def execute(root, source, baseline_root, candidate_root, recovered, data, baseline, previous,
            selection, identity, handoff, reference_images):
    signature = fit.digest(identity)
    audit = {s["key"]: s for s in previous["test-audit.json"]["photos"]}
    cases, receipts = [], []
    with fit.locked_run(root):
        if not (root/"identity").exists():
            if any((root/n).exists() for n in ("cases", "result")):
                raise ValueError("Review outputs without identity; preserved")
            region.publish(root, "identity", signature, {"plan.json": {"identity": identity, "selection": selection}})
        region.read_phase(root, "identity", signature, ["plan.json"])
        (root/"cases").mkdir(exist_ok=True)
        for entry in selection:
            photo = audit[entry["key"]]
            ordinal = entry["cohort_ordinal"]
            if baseline["scores.json"][ordinal-1]["key"] != entry["key"]:
                raise ValueError("Selected ordinal differs from fixed cohort")
            samples, arrays, inputs = read_pair(baseline_root, candidate_root, ordinal, photo, data, baseline)
            masks = ev.read_masks(source, photo)
            case = q.describe(photo, samples, arrays, masks, q.contracts(data)) | {
                "group": entry["group"], "cohort_ordinal": ordinal,
                "input_cache_receipts": inputs, "audit_photo_sha256": fit.digest(photo)}
            for arm in p.ARMS:
                expected = data["scores.json"][arm][ordinal-1]
                if (abs(case["arms"][arm]["score"]-expected["score"]) > p.POLICY["replay_tolerance"]
                        or case["arms"][arm]["decision"] != expected["decision"]):
                    raise ValueError("Saved map review changed the frozen decision")
            name = f"photo-{ordinal:03d}"
            images = []
            try:
                # Even on resume, verify every original image and mask against audited hashes.
                for row in photo["regions"]:
                    images.append(ev.read_tile(source, row, recovered))
                if (root/"cases"/name).exists():
                    cached, _ = region.read_phase(root/"cases", name, signature, CASE_FILES)
                    if cached["case.json"] != case:
                        raise ValueError("Saved review differs from current verified inputs")
                else:
                    with fit.phase_stage(root/"cases", name) as stage:
                        fit.write_json(stage/"case.json", case)
                        for page in range(2):
                            cards.render(case, images, arrays, masks, page).save(stage/f"card-{page+1}.jpg", quality=90)
                        fit.seal_phase(stage, name, signature, {"audit_photo_sha256": fit.digest(photo)}, CASE_FILES)
            finally:
                for image in images:
                    image.close()
            cases.append(case | {"cards": [f"cards/{name}-{page}.jpg" for page in (1, 2)]})
            receipts.append(inputs[p.ARMS[1]])
            print(f"saved-map review {len(cases)}/{len(selection)}: {entry['key']} ({entry['group']})", flush=True)
        gates = region.verify_gates(candidate_root, receipts, data["comparison.json"]["bank_fingerprint"])
        candidate_gates = region.verify_candidate_gates(candidate_root, receipts, data["routing.json"])
        if (any(g not in data["comparison.json"]["normal_gates"] for g in gates)
                or any(g not in data["comparison.json"]["candidate_gates"] for g in candidate_gates)):
            raise ValueError("Contributing gates differ from pinned comparison")
        report = {"status": "saved_region_maps_reviewed_exploratory", "policy": q.POLICY,
                  "execution_sha256": signature, "summary": q.summarize(cases),
                  "frozen_classification": {k: data["comparison.json"][k] for k in
                      ("models", "paired_changes_from_shared_A", "sensitivity_without_reconstructed_photo")},
                  "calibration_reference": q.calibration_reference(data),
                  "normal_candidate_maps_created": 0, "new_inference_calls": 0,
                  "source_attribution": ev.maps.SOURCE_ATTRIBUTION,
                  "automatic_pass_allowed": False, "automatic_cause_assigned": False}
        reference_paths = [f"references/{Path(n).name}" for n in reference_images]
        jsons = {"analysis.json": report, "cases.json": cases, "plan.json": {"identity": identity, "selection": selection}}
        names = list(jsons) + ["index.html", "handoff-manifest.json"] + reference_paths + [c for v in cases for c in v["cards"]]
        if (root/"result").exists():
            old, _ = region.read_phase(root, "result", signature, names)
            if old != (jsons | {"handoff-manifest.json": old["handoff-manifest.json"]}):
                raise ValueError("Published read-only review no longer reproduces")
        else:
            with fit.phase_stage(root, "result") as stage:
                (stage/"cards").mkdir(); (stage/"references").mkdir()
                for v in cases:
                    for page, target in enumerate(v["cards"], 1):
                        shutil.copyfile(root/"cases"/f"photo-{v['cohort_ordinal']:03d}"/f"card-{page}.jpg", stage/target)
                for name, raw in reference_images.items():
                    (stage/"references"/Path(name).name).write_bytes(raw)
                for name, value in jsons.items():
                    fit.write_json(stage/name, value)
                (stage/"index.html").write_text(cards.index(cases, report, reference_paths), encoding="utf-8")
                fit.write_json(stage/"handoff-manifest.json", {n: fit.sha256_file(stage/n) for n in names if n != "handoff-manifest.json"})
                fit.seal_phase(stage, "result", signature, {}, names)
        region.old.export(root/"result", handoff)
    return report


def run(project, archive):
    handoffs = project/"artifacts/handoff"
    print("Checking pinned results, original data and saved-map receipts; no model load or inference...", flush=True)
    data, baseline, previous, selection = load_inputs(project, handoffs)
    if fit.sha256_file(archive) != fit.EXPECTED_ARCHIVE_SHA256:
        raise ValueError("Original MPI archive differs; no repair or replacement")
    recovered = ev.recovered_bytes(project/"artifacts/evaluations/mpi-pcb-external-evaluation-v2/recovery/grid1-3-defect-0054.png", previous)
    with zipfile.ZipFile(handoffs/"mpi-pcb-patch-trace-v1.zip") as z:
        references = {n: z.read(n) for n in REFERENCE_NAMES}
    files = ["review_mpi_region_maps.py", "mpi_region_map_review_protocol.py", "mpi_region_map_review_cards.py"]
    identity = {"policy": q.POLICY, "candidate_execution_sha256": data["comparison.json"]["execution_sha256"],
                "baseline_execution_sha256": baseline["evaluation.json"]["execution_sha256"],
                "source_files_lf_sha256": {f"scripts/{n}": hashlib.sha256((project/"scripts"/n).read_bytes().replace(b"\r\n", b"\n")).hexdigest() for n in files},
                "selection_sha256": fit.digest(selection), "reference_image_sha256": {n: hashlib.sha256(v).hexdigest() for n, v in references.items()}}
    with zipfile.ZipFile(archive) as source:
        return execute(project/"artifacts/analyses"/q.NAME, source,
                       project/"artifacts/evaluations/mpi-pcb-normal-bank-evaluation-v1",
                       project/"artifacts/analyses"/p.NAME, recovered, data, baseline, previous,
                       selection, identity, handoffs/f"{q.NAME}.zip", references)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--archive", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        report = run(args.project.resolve(), args.archive.resolve())
        print(json.dumps({k: report[k] for k in ("status", "summary", "new_inference_calls", "automatic_pass_allowed")}, ensure_ascii=False, indent=2))
        return 0
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}. Existing inputs and completed review cards were preserved.", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
