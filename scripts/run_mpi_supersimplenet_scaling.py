"""Freeze nested normal cohorts, then three sequential fresh fits. Never evaluate defects."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/"src")]
from scripts import mpi_supersimplenet_scaling_protocol as p
from scripts.mpi_supersimplenet_scaling_runtime import Runtime, Monitor, environment
from scripts import run_mpi_supersimplenet_fit as engine
from scripts.run_mpi_efficientad_probe import publish
from scripts.mpi_efficientad_probe_assets import download

fit = p.fit


def pack(path, payload):
    raw = {name: fit.canonical(value) for name, value in payload.items()}
    raw["handoff-manifest.json"] = fit.canonical({k: hashlib.sha256(v).hexdigest() for k,v in raw.items()})
    if path.exists():
        with zipfile.ZipFile(path) as z:
            if (len(z.namelist()) != len(raw) or set(z.namelist()) != set(raw)
                    or any(z.read(k) != v for k,v in raw.items())):
                raise ValueError("Existing scaling handoff differs; preserved")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".scaling-export-", dir=path.parent))/path.name
    with zipfile.ZipFile(stage, "x", zipfile.ZIP_DEFLATED) as z:
        for k,v in raw.items(): z.writestr(k,v)
    stage.rename(path)


def normal_preflight(root, archive, plan):
    """Persist one verified original photo at a time; no inference or test access."""
    signature = fit.digest(plan)
    groups = {"training256": plan["expanded_training"],
              **{g: plan["original_cohorts"][g] for g in ("development", "calibration")}}
    receipt_root = root/"normal-integrity"
    receipt_root.mkdir(exist_ok=True)
    expected = {f"{role}-{pid}" for role, group in groups.items() for pid in group["photo_ids"]}
    if any(d.name not in expected and not d.name.startswith(".") for d in receipt_root.iterdir()):
        raise ValueError("Unexpected normal preflight output")
    receipts = []
    with zipfile.ZipFile(archive) as z:
        names = Counter(z.namelist())
        if any(names[r["member"]] != 1 for group in groups.values() for r in group["rows"]):
            raise ValueError("Missing or duplicate registered normal member")
        for role, group in groups.items():
            for i,pid in enumerate(group["photo_ids"]):
                rows = group["rows"][i*12:(i+1)*12]
                name = f"{role}-{pid}"
                record = {"role":role, "photo_id":pid, "rows_sha256":fit.digest(rows),
                          "normal_tiles_checked":12, "test_images_decoded":0}
                if not (receipt_root/name).exists():
                    for row in rows:
                        with fit.checked_image(z,row): pass
                publish(receipt_root,name,signature,record)
                receipts.append(record)
                if (i+1)%32 == 0 or i+1 == len(group["photo_ids"]):
                    print(f"normal integrity {role} {i+1}/{len(group['photo_ids'])}",flush=True)
    summary = {"status":"normal_scaling_cohorts_frozen_and_verified", "plan_sha256":signature,
        "original_training_photos":64, "expanded_training_photos":256, "additional_training_photos":192,
        "normal_photos_verified":553, "normal_tiles_verified":6636, "receipts_sha256":fit.digest(receipts),
        "test_images_decoded":0, "test_masks_decoded":0, "model_scores_computed":0,
        "training_performed":False, "automatic_pass_allowed":False}
    publish(root,"prepared",signature,summary)
    return summary, receipts


def verify_prepared(root, plan):
    signature = fit.digest(plan)
    fit.read_phase(root,"identity",signature,["value.json"])
    if json.loads((root/"identity/value.json").read_bytes()) != plan:
        raise ValueError("Frozen scaling plan differs")
    fit.read_phase(root,"prepared",signature,["value.json"])
    for role, group in {"training256":plan["expanded_training"],
                       **{g:plan["original_cohorts"][g] for g in ("development","calibration")}}.items():
        for i,pid in enumerate(group["photo_ids"]):
            name = f"{role}-{pid}"
            fit.read_phase(root/"normal-integrity",name,signature,["value.json"])
            record = json.loads((root/"normal-integrity"/name/"value.json").read_bytes())
            if record["rows_sha256"] != fit.digest(group["rows"][i*12:(i+1)*12]):
                raise ValueError("Normal preflight receipt differs")


def run_arm(project, archive, plan, arm):
    root = project/"artifacts"/p.NAME
    verify_prepared(root,plan)
    protocol, identity = p.Protocol(arm), p.arm_identity(plan,arm)
    if environment() != plan["environment"]:
        raise ValueError("Environment differs from reviewed formal fit; no upgrade")
    arm_root = root/arm
    if not (arm_root/"result").exists() and shutil.disk_usage(project).free < 20*1024**3:
        raise ValueError("Need at least 20 GiB free before an unfinished arm")
    assets = project/"artifacts/research-assets/supersimplenet-v1"
    with fit.locked_run(assets):
        weights = download(assets,plan["weights"],False)
    return engine.execute(arm_root,archive,identity,weights,
        project/"artifacts/handoff"/(p.NAME+"-"+arm+".zip"),
        runtime_factory=lambda w:Runtime(w,protocol), monitor_factory=Monitor, protocol=protocol)


def collect(project,plan):
    root = project/"artifacts"/p.NAME
    payload = {"plan.json":plan, "normal-preflight.json":json.loads((root/"prepared/value.json").read_bytes())}
    reports = {}
    for arm in p.ARMS:
        identity = p.arm_identity(plan,arm)
        fit.read_phase(root/arm,"result",fit.digest(identity),["value.json"])
        expected = json.loads((root/arm/"result/value.json").read_bytes())
        path = project/"artifacts/handoff"/(p.NAME+"-"+arm+".zip")
        with zipfile.ZipFile(path) as z:
            manifest = json.loads(z.read("handoff-manifest.json"))
            if len(z.namelist()) != len(manifest)+1 or set(z.namelist()) != set(manifest)|{"handoff-manifest.json"}:
                raise ValueError("Arm handoff inventory differs")
            for name,digest in manifest.items():
                raw = z.read(name)
                if hashlib.sha256(raw).hexdigest() != digest:
                    raise ValueError("Arm handoff checksum differs")
                payload[arm+"/"+name] = json.loads(raw)
        reports[arm] = payload[arm+"/report.json"]
        if reports[arm] != expected or reports[arm]["identity_sha256"] != fit.digest(identity):
            raise ValueError("Arm report provenance differs")
        if (reports[arm]["completed_updates"] != identity["policy"]["updates"]
                or not reports[arm]["normal_reload_passed"]):
            raise ValueError("Incomplete arm cannot be published as completed")
    payload["report.json"] = {"status":"three_normal_only_scaling_fits_completed_ready_for_review",
        "arms":{a:{k:r[k] for k in ("training_photos","completed_epochs","completed_updates",
                 "cumulative_compute_seconds","threshold","normal_reload_passed","memory")} for a,r in reports.items()},
        "accuracy_measured":False, "test_images_decoded":0, "test_masks_decoded":0,
        "service_model_changed":False, "automatic_pass_allowed":False,
        "next_step":"review training and reload evidence, then separately freeze exploratory evaluation"}
    output = project/"artifacts/handoff"/(p.NAME+".zip")
    pack(output,payload)
    return payload["report.json"]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project",type=Path,default=ROOT)
    parser.add_argument("--archive",type=Path,required=True)
    parser.add_argument("--prepare-only",action="store_true")
    parser.add_argument("--worker",choices=p.ARMS,help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    project,archive = args.project.resolve(),args.archive.resolve()
    if args.worker and args.prepare_only:
        raise ValueError("Preparation and worker modes are mutually exclusive")
    plan = p.prepare(project,project/"artifacts/handoff")
    if fit.sha256_file(archive) != plan["archive_sha256"]:
        raise ValueError("Original archive differs; preserve source and outputs")
    if args.worker:
        print(json.dumps(run_arm(project,archive,plan,args.worker),indent=2))
        return 0
    root = project/"artifacts"/p.NAME
    with fit.locked_run(root):
        if not (root/"identity").exists() and any(f.name != ".run.lock" for f in root.iterdir()):
            raise ValueError("Unidentified output; preserved")
        publish(root,"identity",fit.digest(plan),plan)
        summary,receipts = normal_preflight(root,archive,plan)
        pack(project/"artifacts/handoff"/(p.NAME+"-preparation.zip"),
             {"plan.json":plan,"report.json":summary,"normal-integrity.json":receipts})
        if args.prepare_only:
            print(json.dumps(summary,indent=2)); return 0
        if not any((root/arm/"step-000000").exists() for arm in p.ARMS) and shutil.disk_usage(project).free < 100*1024**3:
            raise ValueError("Need 100 GiB free for three retained checkpoint series; no automatic deletion")
        for arm in p.ARMS:
            print(f"Starting/resuming {arm} in a separate GPU process; no time cap, no defect gate.",flush=True)
            command = [sys.executable,str(Path(__file__).resolve()),"--project",str(project),
                       "--archive",str(archive),"--worker",arm]
            child = subprocess.Popen(command)
            try:
                code = child.wait()
            except KeyboardInterrupt:
                # Windows console Ctrl+C also reaches the worker. Keep the suite
                # lock until its checkpoint/attempt receipt has been flushed.
                print("Waiting for worker interruption receipt; do not force-kill.",flush=True)
                child.wait()
                raise
            if code != 0:
                raise RuntimeError(f"Arm {arm} stopped with exit {code}; keep files and review the error")
        print(json.dumps(collect(project,plan),indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (Exception,KeyboardInterrupt) as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}. Preserve outputs; no reroll, no test evaluation.",file=sys.stderr)
        raise SystemExit(1)
