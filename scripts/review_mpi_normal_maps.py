"""Replay exactly five calibration normals and retain fixed-scale attribution maps."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys
import tempfile
import time
import zipfile

import numpy as np

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parents[1]/"src")]
from scripts import evaluate_mpi_pcb_normal_bank as ev
from scripts import export_mpi_normal_visual_review as visual
from scripts import mpi_score_validation_protocol as scores
from scripts import mpi_normal_map_cards as cards

fit = ev.fit
NAME = "mpi-normal-map-review-v1"
IDS = ["1671", "0616", "0705", "1365", "0290"]
VISUAL_SHA = "052030f33d2441dc683699d11ef0a371a768e64cf8b59fe1b1f4f97030870091"
SPATIAL_SHA = "8429a970864c1116d5f52b786057fc84c3e48f465b5c502715ab7f61d35e13dc"
RAW_SHA = "e8a3d34913ef3df07f4ba88e2eb6c9e02606d68c0149fe73e2501945f677d87c"
CONTROL_SHA = "d11fa13181036fc54761c8382d22be88a2032c096cdf5a1c9eb09dbbffe1afac"
FILES = ["case.json", "maps.npz", "card-1.png", "card-2.png", "focus.png"] + [r+".png" for r in fit.REGIONS]
POLICY = {"photo_ids": IDS, "tiles": 60, "raw_replay_tolerance": 1e-4,
          "decision_changes_allowed": 0, "map_shape": [12, 256, 256],
          "display_raw_range": [0, cards.DISPLAY_MAX], "black_band_radius": cards.BAND_RADIUS,
          "training": False, "threshold_refit": False, "test_decoding": False,
          "data_excluded": False, "automatic_pass_allowed": False}


def prepare(project, handoffs, visual_path):
    config = fit.load_config(project/"configs/mpi-pcb-normal-bank-pilot-v1.toml")
    prep = fit.load_preparation(handoffs/"mpi-pcb-normal-bank-preparation-v1.zip", config)
    records = ev.load_fit(handoffs/"mpi-pcb-normal-bank-fit-v1.zip", project, config, prep)
    selection, selection_sha = visual.load_config(project/"configs/mpi-normal-visual-review-v1.json")
    if fit.sha256_file(visual_path) != VISUAL_SHA:
        raise ValueError("Reviewed normal visual ZIP changed")
    with zipfile.ZipFile(visual_path) as z:
        if len(z.namelist()) != len(set(z.namelist())) or json.loads(z.read("selection.json")) != selection:
            raise ValueError("Normal visual ZIP inventory/selection changed")
    data = ev.checked_bundle(handoffs/"mpi-spatial-score-v1.zip", SPATIAL_SHA)
    with zipfile.ZipFile(handoffs/"mpi-spatial-score-v1.zip") as z:
        if visual.sha(z.read("raw-scores.json")) != RAW_SHA or visual.sha(z.read("control.json")) != CONTROL_SHA:
            raise ValueError("Pinned normal score payload differs")
    method = data["control.json"]["candidate_top1_q90"]
    if method["field"] != "top1" or [r["region"] for r in method["regions"]] != list(fit.REGIONS):
        raise ValueError("Frozen Q90 contract differs")
    entries = []
    for pid in IDS:
        photo = next(p for p in selection["photos"] if p["photo_id"] == pid)
        expected = next(p for p in data["raw-scores.json"]["calibration"] if p["photo_id"] == pid)
        if photo["cohort"] != "calibration" or expected["key"] != f"calibration/{pid}" or expected["label"] != 0:
            raise ValueError("Non-calibration input refused")
        rows = [r for r in prep["calibration_rows"] if r["photo_id"] == pid]
        if rows != photo["tiles"] or [r["member"] for r in expected["regions"]] != [r["member"] for r in rows]:
            raise ValueError("Normal source provenance differs")
        entries.append({"photo": photo, "expected": expected})
    paths = ["scripts/review_mpi_normal_maps.py", "scripts/mpi_normal_map_cards.py",
             "scripts/export_mpi_normal_visual_review.py", "scripts/mpi_score_validation_protocol.py",
             "scripts/mpi_score_probe_protocol.py", "scripts/evaluate_mpi_pcb_normal_bank.py"]
    identity = {"policy": POLICY, "visual_zip_sha256": VISUAL_SHA, "spatial_zip_sha256": SPATIAL_SHA,
                "checkpoint_sha256": ev.CHECKPOINT_SHA256, "selection_sha256": selection_sha,
                "fit_execution": fit.execution_identity(project, config), "entries": entries, "method": method,
                "source_files_lf_sha256": {p: visual.sha((project/p).read_bytes().replace(b"\r\n", b"\n")) for p in paths}}
    return config, records, entries, method, identity


def images_for(source, entry):
    images = []
    try:
        for tile in entry["photo"]["tiles"]:
            name = f"photos/{entry['photo']['photo_id']}/{tile['region']}.png"
            info = source.getinfo(name)
            if info.is_dir() or info.file_size > 2*1024*1024:
                raise ValueError("Invalid selected normal member")
            images.append(visual.decode(source.read(info), tile))
        return images
    except BaseException:
        for image in images:
            image.close()
        raise


def describe(entry, arrays, native, images, method):
    if arrays.shape != (12, 256, 256) or arrays.dtype != np.float32 or not np.isfinite(arrays).all() or np.any(arrays < 0):
        raise ValueError("Invalid saved anomaly maps")
    if len(native) != 12 or not np.isfinite(native).all():
        raise ValueError("Invalid native scores")
    rows, regions = [], []
    for i, tile in enumerate(entry["photo"]["tiles"]):
        top1 = fit.candidate_image_score(native[i], arrays[i], fit.CANDIDATE)
        stats = cards.statistics(images[i], arrays[i])
        if abs(stats["top_mean"]-top1) > 1e-8:
            raise ValueError("Top-pixel attribution does not reproduce image score")
        row = {"region": tile["region"], "member": tile["member"], "top1": top1, "native_raw": float(native[i])}
        rows.append(row)
        normal = method["regions"][i]
        regions.append({**row, "normalized_score": (top1-normal["center"])/normal["scale"], "map": stats})
    sample = {"key": f"calibration/{entry['photo']['photo_id']}", "photo_id": entry["photo"]["photo_id"], "label": 0, "regions": rows}
    delta = scores.replay(sample, entry["expected"], {"q90": method})
    return {"photo_id": sample["photo_id"], "sample": sample, "regions": regions,
            "decision": scores.score(sample, method), "threshold": method["threshold"],
            "maximum_absolute_replay_difference": delta, "automatic_pass_allowed": False,
            "automatic_cause_assigned": False}


def create_case(source, entry, runtime, method, stage):
    images = images_for(source, entry)
    arrays, native, latencies = [], [], []
    try:
        for tile, image in zip(entry["photo"]["tiles"], images, strict=True):
            runtime.synchronize()
            started = time.perf_counter()
            raw, values = runtime.predict(image)
            runtime.synchronize()
            latencies.append((time.perf_counter()-started)*1000)
            arrays.append(np.asarray(values, dtype=np.float32).copy())
            native.append(float(raw))
        arrays = np.stack(arrays)
        case = describe(entry, arrays, native, images, method)  # Fail before any completed case publication.
        case["timing"] = {"tile_predict_ms": latencies, "scope": "preprocessing, GPU inference, map transfer; includes first-call warmup, excludes model load and image decode; not a service benchmark"}
        fit.write_json(stage/"case.json", case)
        with (stage/"maps.npz").open("xb") as f:
            np.savez_compressed(f, anomaly_maps=arrays, native_raw=np.asarray(native, dtype=np.float64), regions=np.array(fit.REGIONS))
        for tile in entry["photo"]["tiles"]:
            name = f"photos/{entry['photo']['photo_id']}/{tile['region']}.png"
            visual.write(stage/(tile["region"]+".png"), source.read(name))
        for page in range(2):
            with cards.render(case, images, arrays, list(range(page*6, (page+1)*6))) as card:
                card.save(stage/f"card-{page+1}.png")
        winner = list(fit.REGIONS).index(case["decision"]["winning_region"])
        with cards.render(case, images, arrays, [winner]) as card:
            card.save(stage/"focus.png")
        return case
    finally:
        for image in images:
            image.close()


def read_case(root, signature, entry, method, environment):
    pid = entry["photo"]["photo_id"]
    receipt = fit.read_phase(root/"cases", pid, signature, FILES)
    path = root/"cases"/pid
    if receipt["entry_sha256"] != fit.digest(entry) or receipt["runtime"] != environment:
        raise ValueError("Cached case identity/runtime differs")
    if {p.name for p in path.iterdir()} != set(FILES) | {"receipt.json"}:
        raise ValueError("Unexpected cached case inventory")
    case = json.loads((path/"case.json").read_bytes())
    with np.load(path/"maps.npz", allow_pickle=False) as saved:
        if set(saved.files) != {"anomaly_maps", "native_raw", "regions"} or saved["regions"].tolist() != list(fit.REGIONS):
            raise ValueError("Cached map region inventory differs")
        arrays, native = saved["anomaly_maps"], saved["native_raw"]
    images = []
    try:
        for tile in entry["photo"]["tiles"]:
            images.append(visual.decode((path/(tile["region"]+".png")).read_bytes(), tile))
        rebuilt = describe(entry, arrays, native, images, method)
        if {k: v for k, v in case.items() if k != "timing"} != rebuilt:
            raise ValueError("Cached attribution or replay differs from saved maps")
        timing = case["timing"]["tile_predict_ms"]
        if len(timing) != 12 or not np.isfinite(timing).all() or min(timing) < 0:
            raise ValueError("Cached timing invalid")
    finally:
        for image in images:
            image.close()
    return case


def execute(root, source, checkpoint, config, records, entries, method, identity, handoff, runtime_factory=fit.TorchRuntime):
    if [e["photo"]["photo_id"] for e in entries] != IDS or any(e["photo"]["cohort"] != "calibration" for e in entries):
        raise ValueError("Exactly five frozen calibration normals required")
    signature = fit.digest(identity)
    environment = records["reload.json"]["receipt"]["runtime"]
    runtime = None
    with fit.locked_run(root):
        if not (root/"identity").exists():
            if (root/"cases").exists() or (root/"result").exists():
                raise ValueError("Output lacks input identity; preserved")
            with fit.phase_stage(root, "identity") as stage:
                fit.write_json(stage/"plan.json", identity)
                fit.seal_phase(stage, "identity", signature, {}, ["plan.json"])
        fit.read_phase(root, "identity", signature, ["plan.json"])
        if json.loads((root/"identity/plan.json").read_bytes()) != identity:
            raise ValueError("Existing plan differs")
        (root/"cases").mkdir(exist_ok=True)
        cases = {}
        for entry in entries:
            pid = entry["photo"]["photo_id"]
            if (root/"cases"/pid).exists():
                cases[pid] = read_case(root, signature, entry, method, environment)
        if (root/"result").exists() and len(cases) != 5:
            raise ValueError("Published result is missing completed cases")
        attempt = Path(tempfile.mkdtemp(prefix="attempt-", dir=root)) if len(cases) < 5 else None
        try:
            if len(cases) < 5:
                runtime = runtime_factory(config)
                runtime.load(checkpoint, records["fit.json"]["checkpoint_context"])
                if runtime.environment() != environment:
                    raise ValueError("Runtime differs from the frozen model reload")
            for entry in entries:
                pid = entry["photo"]["photo_id"]
                if pid not in cases:
                    with fit.phase_stage(root/"cases", pid) as stage:
                        create_case(source, entry, runtime, method, stage)
                        fit.seal_phase(stage, pid, signature, {"entry_sha256": fit.digest(entry), "runtime": environment}, FILES)
                    cases[pid] = read_case(root, signature, entry, method, environment)
                print(f"Normal map review {len(cases)}/5: {pid}", flush=True)
            ordered = [cases[pid] for pid in IDS]
            report = {"status": "normal_map_review_complete_exploratory", "policy": POLICY,
                      "execution_sha256": signature, "runtime": environment, "normal_photos": 5, "normal_tiles": 60,
                      "maximum_absolute_replay_difference": max(c["maximum_absolute_replay_difference"] for c in ordered),
                      "test_images_decoded": 0, "test_masks_decoded": 0, "training_performed": False,
                      "thresholds_changed": False, "new_performance_evaluation": False,
                      "automatic_pass_allowed": False, "automatic_cause_assigned": False,
                      "rights_note": "Private review only; no public redistribution or Git/Notion image upload."}
            payloads = {"report.json": report, "cases.json": ordered, "plan.json": identity, "method.json": method}
            result = root/"result"
            if not result.exists():
                with fit.phase_stage(root, "result") as stage:
                    for name, value in payloads.items():
                        fit.write_json(stage/name, value)
                    for pid in IDS:
                        shutil.copytree(root/"cases"/pid, stage/"cases"/pid)
                    visual.write(stage/"index.html", cards.index(ordered).encode("utf-8"))
                    visual.seal(stage, identity)
            visual.verify(result, identity)
            for name, value in payloads.items():
                if json.loads((result/name).read_bytes()) != value:
                    raise ValueError("Published result differs")
            visual.bundle(result, handoff)
            return report
        except Exception as exc:
            if attempt is not None:
                fit.write_json(attempt/"failure.json", {"error": f"{type(exc).__name__}: {exc}", "completed_photo_ids": list(cases)})
            raise
        finally:
            if runtime is not None:
                runtime.release()


def run(project, visual_path=None):
    project = Path(project).resolve()
    handoffs = project/"artifacts/handoff"
    visual_path = Path(visual_path) if visual_path else handoffs/"mpi-normal-visual-review-v1.zip"
    config, records, entries, method, identity = prepare(project, handoffs, visual_path)
    checkpoint = project/"artifacts/mpi-pcb-normal-bank-pilot-v1/fit/model.pt"
    if fit.sha256_file(checkpoint) != ev.CHECKPOINT_SHA256:
        raise ValueError("Reviewed checkpoint changed")
    handoff = handoffs/f"{NAME}.zip"
    with zipfile.ZipFile(visual_path) as source:
        report = execute(project/"artifacts/analyses"/NAME, source, checkpoint, config, records, entries, method, identity, handoff)
    return {**report, "handoff": str(handoff), "handoff_sha256": fit.sha256_file(handoff)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--visual-bundle", type=Path)
    args = parser.parse_args(argv)
    try:
        print(json.dumps(run(args.project, args.visual_bundle), ensure_ascii=False, indent=2))
        return 0
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}. Existing data/results preserved.", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
