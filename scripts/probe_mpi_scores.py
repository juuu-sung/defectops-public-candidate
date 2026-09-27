"""Replay one saved MPI model; compare two frozen score-only ablations."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
import time
import zipfile
from pathlib import Path

import numpy as np

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parents[1] / "src")]
from scripts import mpi_score_probe_protocol as protocol
from scripts import mpi_score_probe_cards as cards

ev, fit = protocol.ev, protocol.fit


def normal_photos(ids, rows, prefix):
    if [(r["photo_id"], r["region"]) for r in rows] != [(p, r) for p in ids for r in fit.REGIONS]:
        raise ValueError("Normal row inventory/order differs")
    return [{"key": f"{prefix}/{p}", "photo_id": p, "label": 0, "regions": rows[i*12:(i+1)*12]}
            for i, p in enumerate(ids)]


def sample_files(has_cards):
    return ["sample.json"] + (["card-1.jpg", "card-2.jpg", "location.json"] if has_cards else [])


def read_sample(root, name, signature, photo, reason, expected):
    receipt = fit.read_phase(root, name, signature, sample_files(reason is not None))
    if receipt["input_photo_sha256"] != fit.digest(photo) or receipt["card_reason"] != reason:
        raise ValueError("Cached sample input/selection changed")
    sample = json.loads((root / name / "sample.json").read_bytes())
    protocol.validate_sample(sample, photo)
    if expected is not None:
        protocol.replay(sample, expected)
    return sample, receipt


def create_sample(source, photo, runtime, recovered, reason, expected, stage, normal):
    images, arrays, rows, timings = [], [], [], []
    started = time.perf_counter()
    try:
        for row in photo["regions"]:
            runtime.synchronize()
            before = time.perf_counter()
            image = fit.checked_image(source, row) if normal else ev.read_tile(source, row, recovered)
            images.append(image)
            raw, values = runtime.predict(image)
            values = np.asarray(values, dtype=np.float32)
            if values.shape != (256, 256) or not np.isfinite(values).all() or np.any(values < 0):
                raise ValueError("Invalid model map")
            top1 = fit.candidate_image_score(raw, values, fit.CANDIDATE)
            rows.append({k: row[k] for k in ("region", "member")} | {"top1": top1, "native_raw": float(raw)})
            arrays.append(values)
            runtime.synchronize()
            timings.append((time.perf_counter() - before) * 1000)
        scoring_ms = (time.perf_counter() - started) * 1000
        sample = {k: photo[k] for k in ("key", "photo_id", "label")} | {"regions": rows}
        protocol.validate_sample(sample, photo)
        sample["baseline_maximum_absolute_replay_difference"] = protocol.replay(sample, expected) if expected is not None else None
        sample["timing"] = {"tile_ms": timings, "photo_scoring_ms": scoring_ms}
        if reason is not None:
            masks = [np.zeros((256, 256), dtype=bool) for _ in rows] if normal else ev.read_masks(source, photo)
            score = max(r["top1"] for r in rows)
            baseline = {"score": score, "threshold": ev.THRESHOLD,
                        "decision": "REVIEW_REQUIRED" if score >= ev.THRESHOLD else "BELOW_THRESHOLD_NOT_AUTO_PASS",
                        "regions": [{"region": r["region"], "score": r["top1"]} for r in rows]}
            location = ev.maps.describe_photo(baseline, arrays, masks, photo["key"] != "defect/0006")
            fit.write_json(stage / "location.json", location)
            for page in range(2):
                cards.render(sample, images, arrays, masks, page, reason).save(stage / f"card-{page+1}.jpg", quality=90)
        sample["timing"]["total_with_optional_cards_ms"] = (time.perf_counter() - started) * 1000
        fit.write_json(stage / "sample.json", sample)
    finally:
        for image in images:
            image.close()


def export(result, handoff):
    names = sorted(p.relative_to(result).as_posix() for p in result.rglob("*") if p.is_file() and p.name != "receipt.json")
    if handoff.exists():
        with zipfile.ZipFile(handoff) as archive:
            if sorted(archive.namelist()) != names or any(archive.read(n) != (result/n).read_bytes() for n in names):
                raise ValueError("Existing score-probe handoff differs; preserved")
        return
    handoff.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".mpi-score-probe-", suffix=".zip", dir=handoff.parent)
    os.close(fd)
    try:
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for name in names:
                archive.write(result / name, name)
        os.link(temporary, handoff)
    finally:
        Path(temporary).unlink(missing_ok=True)


def execute(root, source, checkpoint, recovered_path, config, preparation, records, previous,
            baseline, development, selection, identity, git, handoff, runtime_factory=fit.TorchRuntime):
    signature = fit.digest(identity)
    audit = {p["key"]: p for p in previous["test-audit.json"]["photos"]}
    cohorts = {
        "development": normal_photos(development["photo_ids"], development["rows"], "development"),
        "calibration": normal_photos(preparation["calibration_photo_ids"], preparation["calibration_rows"], "calibration"),
        "test": [audit[p["key"]] for p in baseline],
    }
    expected = {f"calibration/{p['photo_id']}": p for p in records["reload.json"]["photos"]} | {p["key"]: p for p in baseline}
    with fit.locked_run(root):
        if not (root / "identity").exists():
            if any((root / n).exists() for n in (*cohorts, "contract", "result")):
                raise ValueError("Outputs without frozen input identity; preserved")
            with fit.phase_stage(root, "identity") as stage:
                fit.write_json(stage / "plan.json", {"identity": identity, "git": git, "development": development,
                                                     "card_selection": selection})
                fit.seal_phase(stage, "identity", signature, {}, ["plan.json"])
        fit.read_phase(root, "identity", signature, ["plan.json"])
        cached = {}
        for group, photos in cohorts.items():
            path = root / group
            path.mkdir(exist_ok=True)
            cached[group] = {}
            for i, photo in enumerate(photos):
                name = f"photo-{i+1:03d}"
                if (path / name).exists():
                    cached[group][i] = read_sample(path, name, signature, photo, selection.get(photo["key"]), expected.get(photo["key"]))
        if (root / "contract").exists():
            fit.read_phase(root, "contract", signature, ["calibration.json"])
            if any(len(cached[g]) != len(cohorts[g]) for g in ("development", "calibration")):
                raise ValueError("Frozen contract has missing normal samples")
            restored = protocol.calibration_contract(
                [cached["development"][i][0] for i in range(len(cohorts["development"]))],
                [cached["calibration"][i][0] for i in range(len(cohorts["calibration"]))])
            if restored != json.loads((root / "contract/calibration.json").read_bytes()):
                raise ValueError("Frozen contract does not reproduce from normal scores")
        if (root / "result").exists():
            receipt = json.loads((root / "result/receipt.json").read_bytes())
            fit.read_phase(root, "result", signature, list(receipt["files"]))
            if any(len(cached[k]) != len(v) for k, v in cohorts.items()):
                raise ValueError("Completed probe is missing sample cache")
            if not (root / "contract").exists():
                raise ValueError("Completed probe missing normal-only contract")
            export(root / "result", handoff)
            return json.loads((root / "result/comparison.json").read_bytes())
        attempt = Path(tempfile.mkdtemp(prefix="attempt-", dir=root))
        runtime = None
        try:
            probe_sha = None
            if any(len(cached[k]) < len(v) for k, v in cohorts.items()):
                print("Loading existing MPI checkpoint. No model fitting or bank expansion.", flush=True)
                runtime = runtime_factory(config)
                before = time.perf_counter()
                runtime.load(checkpoint, records["fit.json"]["checkpoint_context"])
                load_seconds = time.perf_counter() - before
                if runtime.environment() != records["reload.json"]["receipt"]["runtime"]:
                    raise ValueError("Runtime environment differs from reviewed reload")
                probe = ev.normal_probe(source, preparation, records["reload.json"]["photos"], runtime)
                probe.update(process_id=os.getpid(), runtime=runtime.environment(), model_setup_seconds=load_seconds)
                fit.write_json(attempt / "normal-probe.json", probe)
                if not probe["passed"]:
                    raise ValueError(f"Normal probe blocked: {probe['reasons']}")
                probe_sha = fit.sha256_file(attempt / "normal-probe.json")
            samples, contract = {}, None
            recovered = None
            for group, photos in cohorts.items():
                if group == "test":
                    # Freeze both candidate thresholds using NORMALS before opening test payloads.
                    contract = protocol.calibration_contract(samples["development"], samples["calibration"])
                    if not (root / "contract").exists():
                        with fit.phase_stage(root, "contract") as stage:
                            fit.write_json(stage / "calibration.json", contract)
                            fit.seal_phase(stage, "contract", signature, {}, ["calibration.json"])
                    fit.read_phase(root, "contract", signature, ["calibration.json"])
                    if json.loads((root / "contract/calibration.json").read_bytes()) != contract:
                        raise ValueError("Frozen normal-only scoring contract differs")
                    if len(cached[group]) < len(photos):
                        recovered = ev.recovered_bytes(recovered_path, previous)
                print(f"{group}: {len(photos)} photos; verified reusable {len(cached[group])}.", flush=True)
                for i, photo in enumerate(photos):
                    if i not in cached[group]:
                        name, reason = f"photo-{i+1:03d}", selection.get(photo["key"])
                        with fit.phase_stage(root / group, name) as stage:
                            create_sample(source, photo, runtime, recovered, reason, expected.get(photo["key"]), stage, group != "test")
                            fit.seal_phase(stage, name, signature, {"input_photo_sha256": fit.digest(photo),
                                "card_reason": reason, "normal_probe_sha256": probe_sha, "process_id": os.getpid()}, sample_files(reason is not None))
                        cached[group][i] = read_sample(root / group, name, signature, photo, reason, expected.get(photo["key"]))
                    if (i+1) % 10 == 0 or i+1 == len(photos):
                        print(f"{group} {i+1}/{len(photos)} original photos", flush=True)
                samples[group] = [cached[group][i][0] for i in range(len(photos))]
                if group == "development":
                    protocol.fit_normalization(samples[group])  # Early stop for unusable scale.
            report, scores = protocol.comparison(samples["test"], contract, baseline)
            receipts = [receipt for group in cached.values() for _, receipt in group.values()]
            hashes = {r["normal_probe_sha256"] for r in receipts}
            probes = [json.loads(p.read_bytes()) for p in sorted(root.glob("attempt-*/normal-probe.json")) if fit.sha256_file(p) in hashes]
            if len(probes) != len(hashes) or not all(p["passed"] for p in probes):
                raise ValueError("Cached samples missing passing normal probe")
            if runtime is not None:
                fit.write_json(attempt / "runtime.json", {"process_id": os.getpid(), "memory": runtime.memory()})
            report.update(status="score_probe_completed_exploratory", policy=protocol.POLICY,
                execution_sha256=signature, checkpoint_sha256=ev.CHECKPOINT_SHA256,
                baseline_maximum_absolute_replay_difference=max(s["baseline_maximum_absolute_replay_difference"] for s in samples["calibration"] + samples["test"]),
                latency_ms={g: ev.latency_summary([s["timing"]["photo_scoring_ms"] for s in ps]) for g, ps in samples.items()},
                latency_scope="Both scores from ONE shared inference; checked decode/inference/transfer/scoring. Excludes cards/model load; not separate candidate timing or HTTP.",
                mixed_attempts=len(hashes) > 1, normal_probe_attempts=probes,
                runtime_attempts=[json.loads(p.read_bytes()) for p in sorted(root.glob("attempt-*/runtime.json"))],
                interrupted_attempts=[json.loads(p.read_bytes()) for p in sorted(root.glob("attempt-*/failure.json"))],
                card_original_photos=len(selection), normal_development_photos=len(samples["development"]),
                normal_calibration_photos=len(samples["calibration"]),
                location_note="Only selected review cards; no new full localization metric. defect/0006 excluded from location, not classification.",
                original_baseline_threshold_changed=False, automatic_pass_allowed=False, confirmatory_claim_allowed=False,
                training_performed=False, service_model_changed=False)
            with fit.phase_stage(root, "result") as stage:
                payload = {"comparison.json": report, "scores.json": scores, "raw-scores.json": samples,
                           "calibration.json": contract, "plan.json": json.loads((root / "identity/plan.json").read_bytes()),
                           "baseline-scores.json": baseline}
                for name, value in payload.items():
                    fit.write_json(stage / name, value)
                (stage / "cards").mkdir()
                entries, locations = [], {}
                for group, photos in cohorts.items():
                    for i, photo in enumerate(photos):
                        if photo["key"] not in selection:
                            continue
                        source_dir = root / group / f"photo-{i+1:03d}"
                        prefix = photo["key"].replace("/", "-")
                        names = [f"cards/{prefix}-{page}.jpg" for page in (1, 2)]
                        for page, name in enumerate(names, 1):
                            os.link(source_dir / f"card-{page}.jpg", stage / name)
                        locations[photo["key"]] = json.loads((source_dir / "location.json").read_bytes())
                        entries.append((photo["key"], selection[photo["key"]], names))
                fit.write_json(stage / "card-locations.json", locations)
                with (stage / "index.html").open("x", encoding="utf-8") as stream:
                    stream.write(cards.index(entries, report))
                names = sorted(p.relative_to(stage).as_posix() for p in stage.rglob("*") if p.is_file())
                fit.write_json(stage / "handoff-manifest.json", {n: fit.sha256_file(stage/n) for n in names})
                fit.seal_phase(stage, "result", signature, {}, names + ["handoff-manifest.json"])
            export(root / "result", handoff)
            return report
        except BaseException as exc:
            fit.write_json(attempt / "failure.json", {"error": f"{type(exc).__name__}: {exc}",
                "completed_photos": {k: len(v) for k, v in cached.items()}, "process_id": os.getpid(),
                "automatic_pass_allowed": False, "note": "Incomplete experiment, not a valid comparison. All old data/model/results preserved."})
            print(f"Failure details: {attempt / 'failure.json'}", file=sys.stderr)
            raise
        finally:
            if runtime is not None:
                runtime.release()


def run(project, archive):
    handoffs = project / "artifacts/handoff"
    config = fit.load_config(project / "configs/mpi-pcb-normal-bank-pilot-v1.toml")
    preparation = fit.load_preparation(handoffs / "mpi-pcb-normal-bank-preparation-v1.zip", config)
    records = ev.load_fit(handoffs / "mpi-pcb-normal-bank-fit-v1.zip", project, config, preparation)
    previous, _ = ev.load_prior(project, handoffs)
    reviewed = ev.checked_bundle(handoffs / "mpi-pcb-normal-bank-evaluation-v1.zip", protocol.EVALUATION_SHA256)
    baseline = reviewed["scores.json"]
    if ([p["key"] for p in baseline] != previous["cohort-amendment.json"]["photo_keys"]
            or len(baseline) != 109 or sum(p["label"] for p in baseline) != 55
            or reviewed["evaluation.json"]["checkpoint_sha256"] != ev.CHECKPOINT_SHA256):
        raise ValueError("Reviewed baseline cohort/checkpoint differs")
    fingerprints = ev.checked_bundle(handoffs / "mpi-pcb-normal-bank-preparation-v1.zip", fit.PREPARATION_SHA256)["normal-fingerprints.json"]
    development = protocol.select_development(fingerprints, preparation, previous["test-audit.json"]["photos"], protocol.POLICY["normal_development_photos"])
    selection = protocol.card_selection(records["reload.json"]["photos"], baseline)
    checkpoint = project / "artifacts/mpi-pcb-normal-bank-pilot-v1/fit/model.pt"
    if fit.sha256_file(checkpoint) != ev.CHECKPOINT_SHA256 or fit.sha256_file(archive) != fit.EXPECTED_ARCHIVE_SHA256:
        raise ValueError("Saved checkpoint or source archive checksum differs")
    implementation = ev.run_identity(project)
    # New module names intentionally sit outside the legacy *mpi_pcb*.py glob:
    # adding this experiment must not invalidate existing evaluation cache identity.
    for name in ("probe_mpi_scores.py", "mpi_score_probe_protocol.py", "mpi_score_probe_cards.py"):
        p = project / "scripts" / name
        implementation["source_files_lf_sha256"][f"scripts/{name}"] = hashlib.sha256(p.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
    identity = {"policy": protocol.POLICY, "implementation": implementation, "baseline_zip_sha256": protocol.EVALUATION_SHA256,
                "development_sha256": fit.digest(development), "card_selection_sha256": fit.digest(selection)}
    with zipfile.ZipFile(archive) as source:
        return execute(project / "artifacts/analyses" / protocol.NAME, source, checkpoint,
            project / "artifacts/evaluations/mpi-pcb-external-evaluation-v2/recovery/grid1-3-defect-0054.png",
            config, preparation, records, previous, baseline, development, selection, identity, fit.git_state(project),
            handoffs / f"{protocol.NAME}.zip")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--project", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args(argv)
    try:
        report = run(args.project.resolve(), args.archive.resolve())
        print(json.dumps({k: report[k] for k in ("status", "models", "card_original_photos", "baseline_maximum_absolute_replay_difference")}, indent=2, allow_nan=False))
        return 0
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
