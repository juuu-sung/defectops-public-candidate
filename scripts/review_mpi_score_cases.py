"""Replay five reviewed cases; retain maps, attribution and frozen score boundaries."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tempfile
from pathlib import Path
import zipfile

import numpy as np

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parents[1] / "src")]
from scripts import validate_mpi_scores as validation
from scripts import mpi_score_case_cards as cards

p, fit, ev = validation.p, validation.fit, validation.ev
NAME = "mpi-score-case-review-v1"
VALIDATION_SHA256 = "4f995562c08b3c768ff591025a5f22db01daad5d52020b2d296912f4814bd73a"
SELECTION = [("validation/0970", "added_normal_alert"), ("validation/0998", "added_normal_alert"),
             ("validation/1415", "removed_normal_alert"), ("defect/0019", "remaining_miss"),
             ("defect/0024", "remaining_miss")]
CASE_FILES = ["sample.json", "maps.npz", "case.json", "card-1.jpg", "card-2.jpg", "focus.jpg"]


def load_inputs(project, handoffs):
    data = ev.checked_bundle(handoffs / "mpi-score-validation-v1.zip", VALIDATION_SHA256)
    frozen, cohorts, original, config, preparation, _ = validation.prepare(project, handoffs)
    if (frozen != data["freeze.json"] or frozen != data["plan.json"]["freeze"]
            or frozen != json.loads((project / validation.FREEZE_PATH).read_bytes())):
        raise ValueError("Frozen validation contract differs")
    identity = data["plan.json"]["identity"]
    if identity["freeze_sha256"] != fit.digest(frozen) or identity["cohorts_sha256"] != fit.digest(cohorts):
        raise ValueError("Validation input identity differs")
    for path, digest in identity["implementation"]["source_files_lf_sha256"].items():
        if hashlib.sha256((project / path).read_bytes().replace(b"\r\n", b"\n")).hexdigest() != digest:
            raise ValueError(f"Reviewed inference implementation changed: {path}")
    raw, contract = data["raw-scores.json"], frozen["contract"]
    maximum_delta = 0.
    for group, photos in cohorts.items():
        p.validate_samples(raw[group], group if group != "test" else None)
        for i, (sample, photo) in enumerate(zip(raw[group], photos, strict=True)):
            p.old.validate_sample(sample, photo)
            if group != "validation":
                maximum_delta = max(maximum_delta, p.replay(sample, original[group][i], contract))
    report = data["report.json"]
    if (report["policy"] != p.POLICY or not report["replay"]["passed"]
            or report["replay"]["photos"] != 406
            or report["replay"]["maximum_absolute_raw_difference"] != maximum_delta
            or p.observed_metrics(raw["test"], contract) != report["replay"]["observed_test"]
            or p.normal_summary(raw["validation"], contract) != report["normal_validation"]):
        raise ValueError("Reviewed validation report does not reproduce")
    by_key = {s["key"]: s for samples in raw.values() for s in samples}
    photo_by_key = {s["key"]: s for photos in cohorts.values() for s in photos}
    selected = []
    for key, reason in SELECTION:
        sample, photo = by_key[key], photo_by_key[key]
        a, q = [p.score(sample, contract[k])["alert"] for k in ("control_top1_mad", "candidate_top1_q90")]
        valid = {"added_normal_alert": not photo["label"] and not a and q,
                 "removed_normal_alert": not photo["label"] and a and not q,
                 "remaining_miss": photo["label"] == 1 and not q}
        if not valid[reason]:
            raise ValueError("Frozen review selection no longer describes the recorded decision")
        selected.append({"photo": photo, "expected": sample, "reason": reason})
    records = ev.load_fit(handoffs / "mpi-pcb-normal-bank-fit-v1.zip", project, config, preparation)
    return data, selected, config, preparation, records


def describe(sample, arrays, masks, contract, reason):
    if arrays.shape != (12, 256, 256) or arrays.dtype != np.float32:
        raise ValueError("Expected twelve float32 maps")
    rows = []
    for i, row in enumerate(sample["regions"]):
        stats = ev.maps.region_statistics(arrays[i], masks[i], row["top1"])
        methods = {}
        for name, method in contract.items():
            normal = method["regions"][i]
            if normal["region"] != row["region"]:
                raise ValueError("Normalization region order differs")
            value = (row[method["field"]] - normal["center"]) / normal["scale"]
            methods[name] = {"field": method["field"], "normalized_score": value,
                             "margin": value - method["threshold"], "alert": value >= method["threshold"],
                             "raw_boundary": normal["center"] + method["threshold"] * normal["scale"]}
        rows.append({"region": row["region"], "methods": methods, "map": stats})
    methods = {name: p.score(sample, method) | {"threshold": method["threshold"],
               "margin": p.score(sample, method)["score"] - method["threshold"]}
               for name, method in contract.items()}
    return {"key": sample["key"], "label": sample["label"], "reason": reason,
            "sample": sample, "methods": methods, "regions": rows,
            "automatic_pass_allowed": False, "automatic_cause_assigned": False}


def create_case(source, entry, runtime, contract, stage):
    photo = entry["photo"]
    images, arrays, rows = [], [], []
    try:
        masks = ev.read_masks(source, photo) if photo["label"] else [np.zeros((256, 256), dtype=bool) for _ in fit.REGIONS]
        if any(mask is None for mask in masks):
            raise ValueError("Selected review requires intact official masks")
        for row in photo["regions"]:
            # None is safe only for these fixed, unrepaired cases. Hash checks still apply.
            image = ev.read_tile(source, row, None) if photo["label"] else fit.checked_image(source, row)
            images.append(image)
            raw, values = runtime.predict(image)
            runtime.synchronize()
            values = np.asarray(values, dtype=np.float32)
            if values.shape != (256, 256) or not np.isfinite(values).all() or np.any(values < 0):
                raise ValueError("Invalid replay anomaly map")
            arrays.append(values.copy())
            rows.append({k: row[k] for k in ("region", "member")} | {
                "top1": fit.candidate_image_score(raw, values, fit.CANDIDATE), "native_raw": float(raw)})
        sample = {k: photo[k] for k in ("key", "photo_id", "label")} | {"regions": rows}
        delta = p.replay(sample, entry["expected"], contract)
        case = describe(sample, np.stack(arrays), masks, contract, entry["reason"])
        case["maximum_absolute_replay_difference"] = delta
        fit.write_json(stage / "sample.json", sample)
        fit.write_json(stage / "case.json", case)
        with (stage / "maps.npz").open("xb") as target:
            np.savez_compressed(target, anomaly_maps=np.stack(arrays), regions=np.array(fit.REGIONS))
        for page in range(2):
            with cards.render(case, images, arrays, masks, page) as card:
                card.save(stage / f"card-{page+1}.jpg", quality=92)
        with cards.focus(case, images, arrays, masks) as card:
            card.save(stage / "focus.jpg", quality=92)
        return case
    finally:
        for image in images:
            image.close()


def read_case(root, name, signature, entry, records, contract):
    receipt = fit.read_phase(root / "cases", name, signature, CASE_FILES)
    if receipt["entry_sha256"] != fit.digest(entry):
        raise ValueError("Cached case belongs to another input")
    gate_path = (root / receipt["probe_path"]).resolve()
    gate_path.relative_to(root.resolve())
    if fit.sha256_file(gate_path) != receipt["probe_sha256"]:
        raise ValueError("Cached runtime evidence changed")
    gate = json.loads(gate_path.read_bytes())
    if not gate["passed"] or gate["runtime"] != records["reload.json"]["receipt"]["runtime"]:
        raise ValueError("Cached runtime gate invalid")
    case = json.loads((root / "cases" / name / "case.json").read_bytes())
    sample = json.loads((root / "cases" / name / "sample.json").read_bytes())
    if sample != case["sample"] or case["reason"] != entry["reason"]:
        raise ValueError("Cached case/sample differs")
    if p.replay(sample, entry["expected"], contract) != case["maximum_absolute_replay_difference"]:
        raise ValueError("Cached replay differs")
    return case, receipt


def execute(root, source, checkpoint, config, preparation, records, selected, contract, identity,
            handoff, runtime_factory=fit.TorchRuntime):
    signature = fit.digest(identity)
    with fit.locked_run(root):
        if not (root / "identity").exists() and any((root / n).exists() for n in ("cases", "result")):
            raise ValueError("Review output without input identity; preserved")
        validation.publish(root, "identity", signature, {"plan.json": {"identity": identity}})
        (root / "cases").mkdir(exist_ok=True)
        cases, receipts = {}, {}
        for entry in selected:
            name = entry["photo"]["key"].replace("/", "-")
            if (root / "cases" / name).exists():
                cases[name], receipts[name] = read_case(root, name, signature, entry, records, contract)
        if (root / "result").exists() and len(cases) != len(selected):
            raise ValueError("Published review has missing cases")
        runtime = None
        attempt = Path(tempfile.mkdtemp(prefix="attempt-", dir=root)) if len(cases) < len(selected) else None
        try:
            if attempt is not None:
                runtime = runtime_factory(config)
                runtime.load(checkpoint, records["fit.json"]["checkpoint_context"])
                if runtime.environment() != records["reload.json"]["receipt"]["runtime"]:
                    raise ValueError("Runtime differs from reviewed checkpoint reload")
                gate = ev.normal_probe(source, preparation, records["reload.json"]["photos"], runtime)
                gate["runtime"] = runtime.environment()
                fit.write_json(attempt / "normal-probe.json", gate)
                if not gate["passed"]:
                    raise ValueError(f"Normal probe blocked: {gate['reasons']}")
                probe_path = (attempt / "normal-probe.json").relative_to(root).as_posix()
            for entry in selected:
                name = entry["photo"]["key"].replace("/", "-")
                if name not in cases:
                    with fit.phase_stage(root / "cases", name) as stage:
                        create_case(source, entry, runtime, contract, stage)
                        fit.seal_phase(stage, name, signature, {"entry_sha256": fit.digest(entry),
                            "probe_path": probe_path, "probe_sha256": fit.sha256_file(root / probe_path)}, CASE_FILES)
                    cases[name], receipts[name] = read_case(root, name, signature, entry, records, contract)
                print(f"Case review {len(cases)}/{len(selected)}: {entry['photo']['key']}", flush=True)
            ordered = [cases[e["photo"]["key"].replace("/", "-")] for e in selected]
            report = {"status": "five_case_score_review_complete_exploratory", "case_count": len(ordered),
                      "execution_sha256": signature, "validation_sha256": VALIDATION_SHA256,
                      "maximum_absolute_replay_difference": max(c["maximum_absolute_replay_difference"] for c in ordered),
                      "training_performed": False, "thresholds_changed": False, "new_performance_evaluation": False,
                      "automatic_pass_allowed": False, "automatic_cause_assigned": False,
                      "source_attribution": ev.maps.SOURCE_ATTRIBUTION}
            payloads = {"report.json": report, "cases.json": ordered, "freeze.json": contract,
                        "plan.json": {"identity": identity},
                        "runtime-probes.json": {r["probe_path"]: json.loads((root / r["probe_path"]).read_bytes()) for r in receipts.values()}}
            paths = {f"cards/{name}-{suffix}.jpg": root / "cases" / name / filename
                     for name in cases for suffix, filename in (("1", "card-1.jpg"), ("2", "card-2.jpg"), ("focus", "focus.jpg"))}
            paths.update({f"maps/{name}.npz": root / "cases" / name / "maps.npz" for name in cases})
            names = list(payloads) + list(paths) + ["index.html", "handoff-manifest.json"]
            if not (root / "result").exists():
                with fit.phase_stage(root, "result") as stage:
                    for name, value in payloads.items():
                        fit.write_json(stage / name, value)
                    for name, path in paths.items():
                        (stage / name).parent.mkdir(exist_ok=True)
                        shutil.copyfile(path, stage / name)
                    (stage / "index.html").write_text(cards.index(ordered), encoding="utf-8")
                    fit.write_json(stage / "handoff-manifest.json", {n: fit.sha256_file(stage / n) for n in names if n != "handoff-manifest.json"})
                    fit.seal_phase(stage, "result", signature, {}, names)
            fit.read_phase(root, "result", signature, names)
            for name, value in payloads.items():
                if json.loads((root / "result" / name).read_bytes()) != value:
                    raise ValueError("Published review content differs")
            validation.probe.export(root / "result", handoff)
            return report
        except Exception as exc:
            if attempt is not None:
                fit.write_json(attempt / "failure.json", {"error": f"{type(exc).__name__}: {exc}", "completed_cases": len(cases)})
                print(f"Failure evidence: {attempt / 'failure.json'}", file=sys.stderr)
            raise
        finally:
            if runtime is not None:
                runtime.release()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--archive", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        project = args.project.resolve()
        handoffs = project / "artifacts/handoff"
        print("Checking frozen validation and selecting exactly five reviewed cases...", flush=True)
        data, selected, config, preparation, records = load_inputs(project, handoffs)
        checkpoint = project / "artifacts/mpi-pcb-normal-bank-pilot-v1/fit/model.pt"
        if fit.sha256_file(args.archive) != fit.EXPECTED_ARCHIVE_SHA256 or fit.sha256_file(checkpoint) != ev.CHECKPOINT_SHA256:
            raise ValueError("Original archive or checkpoint changed")
        identity = {"validation_sha256": VALIDATION_SHA256, "contract_sha256": fit.digest(data["freeze.json"]["contract"]),
                    "selection": selected, "source_validation_identity": data["plan.json"]["identity"],
                    "source_files_lf_sha256": {f"scripts/{n}": hashlib.sha256((project / "scripts" / n).read_bytes().replace(b"\r\n", b"\n")).hexdigest()
                                              for n in ("review_mpi_score_cases.py", "mpi_score_case_cards.py", "mpi_score_probe_cards.py")}}
        with zipfile.ZipFile(args.archive) as source:
            report = execute(project / "artifacts/analyses" / NAME, source, checkpoint, config, preparation,
                             records, selected, data["freeze.json"]["contract"], identity, handoffs / f"{NAME}.zip")
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}. Existing data/results preserved.", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
