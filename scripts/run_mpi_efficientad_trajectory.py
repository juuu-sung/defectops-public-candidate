"""Audit inputs, replay the final model, compare two saved checkpoints; no training."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile
import time
import traceback
import zipfile
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from scripts import mpi_efficientad_trajectory as t
from scripts import mpi_efficientad_trajectory_cards as cards
from scripts.mpi_efficientad_trajectory_runtime import Runtime, environment
from scripts.review_mpi_efficientad_maps import images_and_masks
from scripts.run_mpi_efficientad_probe import publish, attempts

fit, q = t.fit, t.q
Monitor = q.evaluation.Monitor


def export(root, handoff, signature, payload, files):
    names = sorted([*payload, *files])
    if len(names) != len(set(names)):
        raise ValueError("Duplicate export inventory")
    if not (root / "bundle").exists():
        with fit.phase_stage(root, "bundle") as stage:
            for name, value in payload.items():
                if name.endswith('.html'):
                    (stage / name).write_text(value, encoding="utf-8")
                else:
                    fit.write_json(stage / name, value)
            for name in files:
                (stage / name).parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(root / name, stage / name)
            fit.write_json(stage / "handoff-manifest.json", {n: fit.sha256_file(stage/n) for n in names})
            fit.seal_phase(stage, "bundle", signature, {}, names+["handoff-manifest.json"])
    fit.read_phase(root, "bundle", signature, names+["handoff-manifest.json"])
    names.append("handoff-manifest.json")
    if handoff.exists():
        with zipfile.ZipFile(handoff) as z:
            if len(z.namelist()) != len(names) or set(z.namelist()) != set(names) or any(
                    hashlib.sha256(z.read(n)).hexdigest() != fit.sha256_file(root/"bundle"/n) for n in names):
                raise ValueError("Existing ZIP differs; preserved")
        return
    handoff.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".trajectory-export-", dir=handoff.parent))/handoff.name
    with zipfile.ZipFile(staging, "x", zipfile.ZIP_DEFLATED) as z:
        for name in names:
            z.write(root/"bundle"/name, name)
    staging.rename(handoff)


def atlas_phase(root, archive, training, signature, check):
    names = [f"region-{i:02d}-page-{j}.png" for i in range(12) for j in range(1, 5)]
    if (root / "training-atlas").exists():
        fit.read_phase(root, "training-atlas", signature, names)
    else:
        with fit.phase_stage(root, "training-atlas") as stage:
            for i, region in enumerate(fit.REGIONS):
                for page in range(4):
                    images = []
                    ids = training["photo_ids"][page*16:(page+1)*16]
                    try:
                        for j in range(page*16, (page+1)*16):
                            check()
                            images.append(fit.checked_image(archive, training["rows"][j*12+i]))
                        with cards.atlas(images, ids, region, page) as image:
                            image.save(stage/f"region-{i:02d}-page-{page+1}.png")
                    finally:
                        for image in images: image.close()
                print(f"Training coverage atlas {i+1}/12 regions", flush=True)
            fit.seal_phase(stage, "training-atlas", signature, {}, names)
    return ["training-atlas/"+n for n in names]+["training-atlas/receipt.json"]


def execute(root, archive_path, fit_root, evaluation_root, recovery, data, records, source,
            identity, handoff, runtime_factory=Runtime, monitor_factory=Monitor):
    signature = fit.digest(identity)
    with fit.locked_run(root):
        if not (root/"identity").exists() and any(p.name != ".run.lock" for p in root.iterdir()):
            raise ValueError("Existing output without identity; preserved")
        publish(root, "identity", signature, identity)
        if (root/"result").exists():
            report = q.e.value(root, "result", signature)
            # Completed export can be recovered without loading a GPU model.
            return finish(root, handoff, signature, identity, report)
        previous = attempts(root)
        prior = sum(v["seconds"] for v in previous)
        attempt = Path(tempfile.mkdtemp(prefix="attempt-", dir=root))
        fit.write_json(attempt/"started.json", {"identity": signature, "unix_seconds": time.time()})
        started = time.perf_counter()
        monitor = monitor_factory(None, prior, started, t.POLICY)
        runtime, error, status = None, None, "failed"
        try:
            with monitor, zipfile.ZipFile(archive_path) as archive:
                monitor.check()
                print("1/4 Training64 x 12-region coverage atlas (CPU, no fitting).", flush=True)
                atlas_phase(root, archive, identity["training"], signature, monitor.check)
                fixed = None
                for step in t.STEPS:
                    print(f"2/4 Saved checkpoint {step}: 20 fixed cases, raw maps only.", flush=True)
                    directory = root/f"step-{step:06d}"
                    directory.mkdir(exist_ok=True)
                    names = [f"case-{i:02d}" for i in range(20)]
                    complete = all((directory/n).exists() for n in names)
                    audits = []
                    if not complete:
                        runtime = runtime_factory(fit_root/f"step-{step:06d}/model.pt", identity["fit_identity"], step)
                        monitor.runtime = runtime
                        monitor.check()
                    for i, entry in enumerate(identity["entries"]):
                        phase = names[i]
                        if (directory/phase).exists():
                            fit.read_phase(directory, phase, signature, ["maps.npy", "summary.json"])
                            summary = json.loads((directory/phase/"summary.json").read_bytes())
                        else:
                            monitor.check()
                            images, masks = images_and_masks(archive, entry, recovery, source["previous"])
                            try:
                                input_audit = runtime.input_audit(images[0])
                                outputs = []
                                for image in images:
                                    monitor.check()
                                    outputs.append(t.p.validate_maps(runtime.predict(image)))
                                raw = np.stack(outputs)
                                replay = None
                                if step == 70000:
                                    expected, _, _ = q.load_maps(entry, data, records, fit_root, evaluation_root)
                                    replay = t.final_replay(raw, expected, entry["photo"], identity["contract"])
                                summary = {"step": step, "key": entry["photo"]["key"], "group": entry["group"],
                                    "entry_sha256": fit.digest(entry), "input_audit": input_audit, "final_replay": replay,
                                    "regions": t.raw_summary(raw, masks, entry["photo"]["key"] not in t.POLICY["location_excluded_photos"])}
                                monitor.check()
                                with fit.phase_stage(directory, phase) as stage:
                                    np.save(stage/"maps.npy", raw, allow_pickle=False)
                                    fit.write_json(stage/"summary.json", summary)
                                    fit.seal_phase(stage, phase, signature, {}, ["maps.npy", "summary.json"])
                            finally:
                                for image in images: image.close()
                        if summary["entry_sha256"] != fit.digest(entry) or summary["step"] != step:
                            raise ValueError("Cached checkpoint sample identity differs")
                        fingerprints = summary["input_audit"]["fixed_teacher_and_statistics_sha256"]
                        if fixed is None: fixed = fingerprints
                        if fingerprints != fixed:
                            raise ValueError("Teacher weights or training statistics differ across checkpoints")
                        audits.append(summary)
                        print(f"Checkpoint {step}: {i+1}/20", flush=True)
                    if runtime is not None:
                        monitor.check()
                        monitor.runtime = None
                        runtime.release(); runtime = None
                    if step == 70000 and not all(a["final_replay"] and a["final_replay"]["passed"] for a in audits):
                        raise ValueError("Final replay must pass before earlier checkpoint comparison")
                print("3/4 Preparing common-scale raw-map comparison cards.", flush=True)
                make_cards(root, archive, recovery, source, identity, signature, monitor.check)
                monitor.check()
                status = "completed"
        except BaseException as exc:
            error = f"{type(exc).__name__}: {exc}"
            if isinstance(exc, KeyboardInterrupt):
                status = "interrupted"
            raise
        finally:
            monitor.runtime = None
            if runtime is not None: runtime.release()
            fit.write_json(attempt/"finished.json", {"status": status, "error": error,
                "seconds": time.perf_counter()-started, "memory": monitor.snapshot()})
        report = {"status": "checkpoint_trajectory_completed_exploratory", "steps": list(t.STEPS),
            "photos_per_step": 20, "tiles_per_step": 240, "training_atlas_photos": 64,
            "training_atlas_tiles": 768, "training_performed": False, "thresholds_fitted": 0,
            "earlier_checkpoint_decisions_computed": 0, "service_model_changed": False,
            "automatic_pass_allowed": False, "automatic_cause_assigned": False,
            "cost_scope": "includes atlas, inference and cards; excludes input hash preflight and ZIP export; not latency benchmark",
            "attempts": attempts(root)}
        publish(root, "result", signature, report)
        return finish(root, handoff, signature, identity, report)


def make_cards(root, archive, recovery, source, identity, signature, check):
    parent = root/"cards"; parent.mkdir(exist_ok=True)
    for i, entry in enumerate(identity["entries"]):
        check()
        name = f"case-{i:02d}"
        if (parent/name).exists():
            fit.read_phase(parent, name, signature, ["page-1.png", "page-2.png"])
            continue
        maps, summaries = {}, {}
        for step in t.STEPS:
            directory = root/f"step-{step:06d}"/name
            fit.read_phase(directory.parent, name, signature, ["maps.npy", "summary.json"])
            maps[str(step)] = t.p.validate_maps(np.load(directory/"maps.npy", allow_pickle=False))
            summaries[str(step)] = json.loads((directory/"summary.json").read_bytes())
        images, masks = images_and_masks(archive, entry, recovery, source["previous"])
        try:
            with fit.phase_stage(parent, name) as stage:
                for page in range(2):
                    with cards.render(entry, images, masks, maps, summaries, identity["contract"]["map_quantiles"], page) as card:
                        card.save(stage/f"page-{page+1}.png")
                fit.seal_phase(stage, name, signature, {}, ["page-1.png", "page-2.png"])
        finally:
            for image in images: image.close()
        print(f"Comparison cards {i+1}/20", flush=True)


def finish(root, handoff, signature, identity, report):
    summaries, receipts, files = {}, {}, []
    for step in t.STEPS:
        summaries[str(step)] = []
        for i in range(20):
            phase = f"case-{i:02d}"
            directory = root/f"step-{step:06d}"
            receipts[f"{directory.name}/{phase}"] = fit.read_phase(directory, phase, signature, ["maps.npy", "summary.json"])
            summaries[str(step)].append(json.loads((directory/phase/"summary.json").read_bytes()))
    page = cards.index()
    for i, entry in enumerate(identity["entries"]):
        relative = [f"cards/case-{i:02d}/page-{j}.png" for j in (1, 2)]
        fit.read_phase(root/"cards", f"case-{i:02d}", signature, ["page-1.png", "page-2.png"])
        files += relative+[f"cards/case-{i:02d}/receipt.json"]
        page += cards.section(entry["group"]+"/"+entry["photo"]["photo_id"], relative)
    page += '<h2>학습 정상64장: 전체12영역</h2><p>모든 학습 사진을 원래 순서로 표시합니다. 특정 외관이 있다는 것과 촬영 조건의 다양성을 충분히 학습했다는 것은 다릅니다.</p>'
    atlas_names = [f"region-{i:02d}-page-{j}.png" for i in range(12) for j in range(1, 5)]
    fit.read_phase(root, "training-atlas", signature, atlas_names)
    files += ["training-atlas/"+n for n in atlas_names]+["training-atlas/receipt.json"]
    for i, region in enumerate(fit.REGIONS):
        page += cards.section(region, [f"training-atlas/region-{i:02d}-page-{j}.png" for j in range(1, 5)])
    page += '<p>'+q.ev.maps.SOURCE_ATTRIBUTION+'</p></html>'
    export(root, handoff, signature, {"plan.json": identity, "report.json": report,
        "summaries.json": summaries, "map-receipts.json": receipts, "index.html": page}, files)
    print(f"4/4 Handoff ready: {handoff}", flush=True)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=ROOT)
    parser.add_argument("--archive", type=Path, required=True)
    args = parser.parse_args(argv)
    project = args.project.resolve()
    fit_root = project/"artifacts"/t.p.NAME
    evaluation_root = project/"artifacts/analyses"/q.e.NAME
    if not fit_root.is_dir() or not evaluation_root.is_dir():
        raise ValueError("Existing fit and evaluation folders required; do not retrain")
    if shutil.disk_usage(project).free < t.POLICY["free_disk_GiB"]*1024**3:
        raise ValueError("Need 5 GiB free disk; no automatic deletion")
    print("Checking original archive, reviewed fit/evaluation and three checkpoint receipts...", flush=True)
    if fit.sha256_file(args.archive) != fit.EXPECTED_ARCHIVE_SHA256:
        raise ValueError("Original archive differs")
    with fit.locked_run(fit_root), fit.locked_run(evaluation_root):
        data, records, source, prior = q.prepare(project, project/"artifacts/handoff", fit_root, evaluation_root)
        current = environment()
        if current != records["plan.json"]["environment"]:
            raise ValueError("GPU software/model source differs from reviewed fit")
        identity = {"name": t.NAME, "policy": t.POLICY, "entries": prior["entries"],
            "contract": prior["contract"], "fit_identity": prior["fit_identity"],
            "fit_zip_sha256": q.e.FIT_SHA, "evaluation_zip_sha256": q.EVALUATION_SHA,
            "archive_sha256": fit.EXPECTED_ARCHIVE_SHA256, "environment": current,
            "checkpoints": t.checkpoints(fit_root, records), "sources_lf_sha256": t.sources(project),
            "training": records["plan.json"]["cohorts"]["training"]}
        result = execute(project/"artifacts/analyses"/t.NAME, args.archive, fit_root, evaluation_root,
            project/"artifacts/evaluations/mpi-pcb-external-evaluation-v2/recovery/grid1-3-defect-0054.png",
            data, records, source, identity, project/"artifacts/handoff"/(t.NAME+".zip"))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (Exception, KeyboardInterrupt) as exc:
        traceback.print_exc()
        print(f"ERROR: {type(exc).__name__}: {exc}. Keep outputs; do not retrain or change thresholds.", file=sys.stderr)
        raise SystemExit(1)
