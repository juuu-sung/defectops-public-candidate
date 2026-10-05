"""Inspect only saved maps for 12 observed misses and 8 deterministic normal controls."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile
import zipfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from scripts import mpi_supersimplenet_map_diagnostic as q
from scripts import mpi_supersimplenet_map_cards as cards
from scripts.run_mpi_supersimplenet_probe import publish

fit, ev = q.fit, q.ev
FILES = ["case.json", "card-1.png", "card-2.png"]


def images_and_masks(archive, entry, recovery, previous):
    photo = entry["photo"]
    recovered = q.evaluation.recovery_for_photo(photo, None, recovery, previous)
    images = []
    try:
        for row in photo["regions"]:
            images.append(ev.read_tile(archive, row, recovered) if entry["group"] == "test"
                          else fit.checked_image(archive, row))
        masks = ev.read_masks(archive, photo) if entry["group"] == "test" else [np.zeros((256, 256), dtype=np.bool_) for _ in images]
        return images, masks
    except BaseException:
        for image in images:
            image.close()
        raise


def export(root, handoff, signature, identity, report, cases):
    paths = ["plan.json", "report.json", "cases.json", "index.html"]
    paths += [f"cases/case-{i:02d}/{n}" for i in range(len(cases)) for n in FILES + ["receipt.json"]]
    if not (root / "bundle").exists():
        with fit.phase_stage(root, "bundle") as stage:
            fit.write_json(stage / "plan.json", identity)
            fit.write_json(stage / "report.json", report)
            fit.write_json(stage / "cases.json", cases)
            (stage / "index.html").write_text(cards.index(cases), encoding="utf-8")
            for rel in paths[4:]:
                (stage / rel).parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(root / rel, stage / rel)
            fit.write_json(stage / "handoff-manifest.json", {n: fit.sha256_file(stage / n) for n in paths})
            fit.seal_phase(stage, "bundle", signature, {}, paths+["handoff-manifest.json"])
    fit.read_phase(root, "bundle", signature, paths+["handoff-manifest.json"])
    bundle = root / "bundle"
    names = paths+["handoff-manifest.json"]
    if handoff.exists():
        with zipfile.ZipFile(handoff) as z:
            if len(z.namelist()) != len(names) or set(z.namelist()) != set(names):
                raise ValueError("Existing diagnostic ZIP differs; preserved")
            for n in names:
                if hashlib.sha256(z.read(n)).hexdigest() != fit.sha256_file(bundle / n):
                    raise ValueError("Existing diagnostic ZIP payload changed; preserved")
        return
    handoff.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation: never overwrite an existing or partial delivery.
    staged = Path(tempfile.mkdtemp(prefix=".supersimplenet-map-export-", dir=handoff.parent)) / handoff.name
    with zipfile.ZipFile(staged, "x", zipfile.ZIP_DEFLATED) as z:
        for n in names:
            z.write(bundle / n, n)
    staged.rename(handoff)


def execute(root, archive_path, fit_root, evaluation_root, recovery, data, records, source, identity, handoff):
    signature = fit.digest(identity)
    with fit.locked_run(root):
        if not (root / "identity").exists() and any(p.name != ".run.lock" for p in root.iterdir()):
            raise ValueError("Diagnostic output without identity; preserved")
        publish(root, "identity", signature, identity)
        (root / "cases").mkdir(exist_ok=True)
        cases = []
        with zipfile.ZipFile(archive_path) as archive:
            for i, entry in enumerate(identity["entries"]):
                raw, sample, receipt = q.load_maps(entry, data, records, fit_root, evaluation_root)
                baseline = q.load_baseline(entry, source, fit_root.parent / "evaluations/mpi-pcb-normal-bank-evaluation-v1")
                name = f"case-{i:02d}"
                metadata = {"entry_sha256": fit.digest(entry), "map_receipt_sha256": receipt["receipt_sha256"]}
                if (root / "cases" / name).exists():
                    cached = fit.read_phase(root / "cases", name, signature, FILES)
                    if any(cached[k] != v for k, v in metadata.items()):
                        raise ValueError("Diagnostic cache source changed")
                    case = json.loads((root / "cases" / name / "case.json").read_bytes())
                    if case["sample"] != sample:
                        raise ValueError("Diagnostic cache score changed")
                else:
                    images, masks = images_and_masks(archive, entry, recovery, source["previous"])
                    try:
                        case, arrays = q.describe(entry, raw, masks, sample, identity["contract"], baseline)
                        with fit.phase_stage(root / "cases", name) as stage:
                            fit.write_json(stage / "case.json", case)
                            for page in range(2):
                                with cards.render(case, images, arrays, masks, page) as card:
                                    card.save(stage / f"card-{page+1}.png")
                            fit.seal_phase(stage, name, signature, metadata, FILES)
                    finally:
                        for image in images:
                            image.close()
                cases.append(case)
                print(f"Saved-map review {i+1}/{len(identity['entries'])}: {entry['photo']['key']}", flush=True)
        report = {"status": "saved_map_diagnostic_completed_exploratory", "cases": len(cases),
                  "defect_photos": sum(c["group"] == "test" for c in cases),
                  "normal_photos": sum(c["group"] != "test" for c in cases),
                  "map_tiles": len(cases)*12, "baseline_compared_photos": sum(c["baseline_available"] for c in cases), "cards": len(cases)*2,
                  "location_excluded_photos": q.POLICY["location_excluded_photos"],
                  "model_inference_calls": 0, "training_performed": False, "threshold_changed": False,
                  "automatic_pass_allowed": False, "service_model_changed": False,
                  "automatic_cause_assigned": False, "independent_performance_claim_allowed": False,
                  "source_attribution": ev.maps.SOURCE_ATTRIBUTION,
                  "next": "review_segmentation_vs_classification_and_normal_diversity_before_new_training"}
        publish(root, "result", signature, report)
        export(root, handoff, signature, identity, report, cases)
        return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=ROOT)
    parser.add_argument("--archive", type=Path, required=True)
    args = parser.parse_args(argv)
    project = args.project.resolve()
    fit_root = project / "artifacts" / q.p.NAME
    evaluation_root = project / "artifacts/analyses" / q.e.NAME
    if not fit_root.is_dir() or not evaluation_root.is_dir():
        raise ValueError("Existing fit/evaluation folders required; do not retrain")
    if fit.sha256_file(args.archive) != fit.EXPECTED_ARCHIVE_SHA256:
        raise ValueError("Original MPI archive changed")
    if shutil.disk_usage(project).free < 1024**3:
        raise ValueError("Need 1 GiB free disk; no automatic deletion")
    with fit.locked_run(fit_root), fit.locked_run(evaluation_root):
        data, records, source, identity = q.prepare(project, project / "artifacts/handoff", fit_root, evaluation_root)
        report = execute(project / "artifacts/analyses" / q.NAME, args.archive, fit_root, evaluation_root,
            project / "artifacts/evaluations/mpi-pcb-external-evaluation-v2/recovery/grid1-3-defect-0054.png",
            data, records, source, identity, project / "artifacts/handoff" / (q.NAME + ".zip"))
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (Exception, KeyboardInterrupt) as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}. Keep all files; no retraining or threshold changes.", file=sys.stderr)
        raise SystemExit(1)
