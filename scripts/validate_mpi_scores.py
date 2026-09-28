"""Replay frozen MPI scorers, then audit 256 unused normals without recalibration."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import zipfile

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parents[1] / "src")]
from scripts import mpi_score_validation_protocol as p
from scripts import probe_mpi_scores as probe

fit, ev = p.fit, p.ev
FREEZE_PATH = "configs/mpi-score-validation-v1.json"


def prepare(project, handoffs):
    config = fit.load_config(project / "configs/mpi-pcb-normal-bank-pilot-v1.toml")
    preparation = fit.load_preparation(handoffs / "mpi-pcb-normal-bank-preparation-v1.zip", config)
    bundle = ev.checked_bundle(handoffs / "mpi-pcb-score-probe-v1.zip", p.PROBE_SHA256)
    prior = ev.checked_bundle(handoffs / "mpi-pcb-external-evaluation-v2.zip", ev.maps.EVALUATION_SHA256)
    raw, development = bundle["raw-scores.json"], bundle["plan.json"]["development"]
    with zipfile.ZipFile(handoffs / "mpi-pcb-score-probe-v1.zip") as z:
        if hashlib.sha256(z.read("raw-scores.json")).hexdigest() != p.RAW_SHA256:
            raise ValueError("Reviewed raw score identity differs")
    implementation = bundle["plan.json"]["identity"]["implementation"]
    for name, expected in implementation["source_files_lf_sha256"].items():
        if hashlib.sha256((project/name).read_bytes().replace(b"\r\n", b"\n")).hexdigest() != expected:
            raise ValueError(f"Reviewed inference source changed: {name}")
    for group, count in (("development",128),("calibration",169),("test",109)):
        if len(raw[group]) != count:
            raise ValueError("Reviewed score counts differ")
        p.validate_samples(raw[group], group if group != "test" else None)
    if sum(s["label"] for s in raw["test"]) != 55:
        raise ValueError("Observed test labels differ")
    fingerprints = ev.checked_bundle(handoffs / "mpi-pcb-normal-bank-preparation-v1.zip", fit.PREPARATION_SHA256)["normal-fingerprints.json"]
    audit = {s["key"]: s for s in prior["test-audit.json"]["photos"]}
    cohorts = {
        "development": probe.normal_photos(development["photo_ids"],development["rows"],"development"),
        "calibration": probe.normal_photos(preparation["calibration_photo_ids"],preparation["calibration_rows"],"calibration"),
        "test": [audit[s["key"]] for s in raw["test"]],
    }
    for group, photos in cohorts.items():
        for sample, photo in zip(raw[group],photos,strict=True):
            p.old.validate_sample(sample,photo)
    selected = p.select_normals(fingerprints, preparation, development, list(audit.values()), p.POLICY["normal_validation_count"])
    cohorts["validation"] = probe.normal_photos(selected["photo_ids"],selected["rows"],"validation")
    contract = p.fit_contract(raw["development"],raw["calibration"])
    frozen = {"policy": p.POLICY, "source_probe_sha256": p.PROBE_SHA256, "source_raw_sha256": p.RAW_SHA256,
              "checkpoint_sha256": ev.CHECKPOINT_SHA256, "contract": contract,
              "normal_validation": {k:v for k,v in selected.items() if k != "rows"},
              "observed_test_reference": p.observed_metrics(raw["test"],contract)}
    return frozen, cohorts, raw, config, preparation, prior


def publish(root, name, signature, payloads, metadata=None):
    if not (root/name).exists():
        with fit.phase_stage(root,name) as stage:
            for filename,value in payloads.items():
                fit.write_json(stage/filename,value)
            fit.seal_phase(stage,name,signature,metadata or {},list(payloads))
    receipt = fit.read_phase(root,name,signature,list(payloads))
    for filename,value in payloads.items():
        if json.loads((root/name/filename).read_bytes()) != value:
            raise ValueError(f"Frozen phase content differs: {name}/{filename}")
    return receipt


def execute(root, source, checkpoint, recovered_path, config, preparation, records, previous,
            cohorts, reference, frozen, identity, git, handoff, runtime_factory=fit.TorchRuntime):
    signature = fit.digest(identity)
    contract = frozen["contract"]
    with fit.locked_run(root):
        if not (root/"identity").exists() and any((root/n).exists() for n in (*cohorts,"replay","result")):
            raise ValueError("Outputs without frozen identity; preserved")
        # Capture Git state on the first invocation; source hashes bind all resumptions.
        initial_git = json.loads((root/"identity/plan.json").read_bytes())["git"] if (root/"identity").exists() else git
        publish(root,"identity",signature,{"plan.json":{"identity":identity,"git":initial_git,"freeze":frozen}})
        samples = {g:[] for g in cohorts}
        cached = {}
        for group,photos in cohorts.items():
            (root/group).mkdir(exist_ok=True)
            for i,photo in enumerate(photos):
                name = f"photo-{i+1:03d}"
                if (root/group/name).exists():
                    sample,receipt = probe.read_sample(root/group,name,signature,photo,None,None)
                    probe_file = root/receipt["probe_path"]
                    if fit.sha256_file(probe_file) != receipt["normal_probe_sha256"]:
                        raise ValueError("Sample runtime probe evidence differs")
                    gate = json.loads(probe_file.read_bytes())
                    if not gate["passed"] or gate["runtime"] != records["reload.json"]["receipt"]["runtime"]:
                        raise ValueError("Sample runtime gate was not passed")
                    if group != "validation":
                        p.replay(sample,reference[group][i],contract)
                    cached[group,i] = sample
        replay_complete = all((g,i) in cached for g in reference for i in range(len(cohorts[g])))
        if (root/"replay").exists() and not replay_complete:
            raise ValueError("Replay receipt has missing sample evidence")
        if any(g=="validation" for g,i in cached) and not (root/"replay").exists():
            raise ValueError("Validation samples exist before replay gate")
        if (root/"result").exists() and len(cached) != sum(map(len,cohorts.values())):
            raise ValueError("Completed result has missing samples")
        attempt = Path(tempfile.mkdtemp(prefix="attempt-",dir=root))
        runtime = None
        try:
            probe_path = None
            if len(cached) != sum(map(len,cohorts.values())):
                runtime = runtime_factory(config)
                runtime.load(checkpoint,records["fit.json"]["checkpoint_context"])
                if runtime.environment() != records["reload.json"]["receipt"]["runtime"]:
                    raise ValueError("Runtime environment differs from reviewed reload")
                gate = ev.normal_probe(source,preparation,records["reload.json"]["photos"],runtime)
                gate.update(runtime=runtime.environment(),process_id=os.getpid())
                fit.write_json(attempt/"normal-probe.json",gate)
                if not gate["passed"]:
                    raise ValueError(f"Normal probe blocked: {gate['reasons']}")
                probe_path = (attempt/"normal-probe.json").relative_to(root).as_posix()
            maximum_delta = 0.
            recovered = None
            for group,photos in cohorts.items():
                if group == "validation":
                    # This gate is published before any new validation image is decoded.
                    metrics = p.observed_metrics(samples["test"],contract)
                    if metrics != frozen["observed_test_reference"]:
                        raise ValueError("Observed-test decisions differ from frozen reference")
                    publish(root,"replay",signature,{"replay.json":{
                        "passed":True,"photos":sum(len(samples[g]) for g in reference),
                        "maximum_absolute_raw_difference":maximum_delta,"observed_test":metrics,
                        "thresholds_refitted":False,"new_independent_test":False}})
                if group == "test" and any((group,i) not in cached for i in range(len(photos))):
                    recovered = ev.recovered_bytes(recovered_path,previous)
                print(f"{group}: {len(photos)} photos; cached {sum((group,i) in cached for i in range(len(photos)))}",flush=True)
                for i,photo in enumerate(photos):
                    name = f"photo-{i+1:03d}"
                    if (group,i) not in cached:
                        with fit.phase_stage(root/group,name) as stage:
                            probe.create_sample(source,photo,runtime,recovered,None,None,stage,group!="test")
                            sample = json.loads((stage/"sample.json").read_bytes())
                            if group != "validation":
                                p.replay(sample,reference[group][i],contract)
                            fit.seal_phase(stage,name,signature,{"input_photo_sha256":fit.digest(photo),"card_reason":None,
                                "probe_path":probe_path,"normal_probe_sha256":fit.sha256_file(root/probe_path)},["sample.json"])
                        cached[group,i] = probe.read_sample(root/group,name,signature,photo,None,None)[0]
                    sample = cached[group,i]
                    if group != "validation":
                        maximum_delta = max(maximum_delta,p.replay(sample,reference[group][i],contract))
                    samples[group].append(sample)
                    if (i+1)%10==0 or i+1==len(photos):
                        print(f"{group} {i+1}/{len(photos)}",flush=True)
            report = {"status":"frozen_score_replay_and_unused_normal_audit_complete", "policy":p.POLICY,
                      "replay":json.loads((root/"replay/replay.json").read_bytes()),
                      "normal_validation":p.normal_summary(samples["validation"],contract)}
            payloads = {"report.json":report,"raw-scores.json":samples,"freeze.json":frozen,
                        "plan.json":json.loads((root/"identity/plan.json").read_bytes()),
                        "runtime-probes.json":{r:json.loads((root/r).read_bytes()) for r in sorted({
                            json.loads((root/g/f"photo-{i+1:03d}/receipt.json").read_bytes())["probe_path"]
                            for g,photos in cohorts.items() for i in range(len(photos))})}}
            # Hash canonical bytes, exactly as fit.write_json serializes them.
            payloads["handoff-manifest.json"] = {n:hashlib.sha256(fit.canonical(v)).hexdigest() for n,v in payloads.items()}
            publish(root,"result",signature,payloads)
            probe.export(root/"result",handoff)
            return report
        except Exception as exc:
            fit.write_json(attempt/"failure.json",{"error":f"{type(exc).__name__}: {exc}","completed_samples":len(cached),
                                                    "automatic_pass_allowed":False})
            print(f"Failure evidence: {attempt/'failure.json'}",file=sys.stderr)
            raise
        finally:
            if runtime is not None:
                runtime.release()


def run(project, archive, handoffs):
    frozen,cohorts,reference,config,preparation,previous = prepare(project,handoffs)
    if json.loads((project/FREEZE_PATH).read_bytes()) != frozen:
        raise ValueError("Checked-in frozen contract/cohort differs; do not regenerate after seeing validation")
    records = ev.load_fit(handoffs/"mpi-pcb-normal-bank-fit-v1.zip",project,config,preparation)
    checkpoint = project/"artifacts/mpi-pcb-normal-bank-pilot-v1/fit/model.pt"
    if fit.sha256_file(checkpoint) != ev.CHECKPOINT_SHA256 or fit.sha256_file(archive) != fit.EXPECTED_ARCHIVE_SHA256:
        raise ValueError("Checkpoint or archive identity differs")
    implementation = ev.run_identity(project)
    for filename in ("probe_mpi_scores.py","mpi_score_probe_protocol.py","mpi_score_probe_cards.py",
                     "mpi_score_validation_protocol.py","validate_mpi_scores.py"):
        implementation["source_files_lf_sha256"][f"scripts/{filename}"] = hashlib.sha256((project/"scripts"/filename).read_bytes().replace(b"\r\n",b"\n")).hexdigest()
    identity = {"freeze_sha256":fit.digest(frozen),"implementation":implementation,
                "cohorts_sha256":fit.digest(cohorts)}
    with zipfile.ZipFile(archive) as source:
        return execute(project/"artifacts/analyses"/p.NAME,source,checkpoint,
            project/"artifacts/evaluations/mpi-pcb-external-evaluation-v2/recovery/grid1-3-defect-0054.png",
            config,preparation,records,previous,cohorts,reference,frozen,identity,fit.git_state(project),
            project/"artifacts/handoff"/f"{p.NAME}.zip")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project",type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument("--handoffs",type=Path)
    parser.add_argument("--archive",type=Path)
    parser.add_argument("--freeze-only",type=Path,help="CPU-only: write NEW contract to this path, never overwrite")
    args = parser.parse_args(argv)
    project = args.project.resolve()
    handoffs = (args.handoffs or project/"artifacts/handoff").resolve()
    try:
        if args.freeze_only:
            frozen,*_ = prepare(project,handoffs)
            fit.write_json(args.freeze_only,frozen)
            print(json.dumps({"freeze_sha256":fit.digest(frozen),"reference":frozen["observed_test_reference"]},indent=2))
        else:
            if not args.archive:
                parser.error("--archive is required for GPU replay")
            report = run(project,args.archive.resolve(),handoffs)
            print(json.dumps({"status":report["status"],"normal_validation":report["normal_validation"]["methods"]},indent=2))
        return 0
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}",file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
