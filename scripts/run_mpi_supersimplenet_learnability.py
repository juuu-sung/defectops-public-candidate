"""Fixed 500-update normal/synthetic learning diagnostic, never service training."""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile
import time
import zipfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT),str(ROOT/"src")]
from scripts import mpi_supersimplenet_learnability_protocol as p
from scripts.mpi_supersimplenet_learnability_runtime import Runtime, environment
from scripts.run_mpi_supersimplenet_evaluation import Monitor
from scripts.run_mpi_efficientad_probe import publish, attempts

fit = p.fit


def export(root, handoff, identity, report):
    signature = fit.digest(identity)
    names = ["plan.json","report.json","attempts.json"]
    names += [f"step-{step:04d}/{n}" for step in p.POLICY["snapshots"]
              for n in ("arrays.npz","progress.json","receipt.json")]
    if not (root/"bundle").exists():
        with fit.phase_stage(root,"bundle") as stage:
            fit.write_json(stage/"plan.json",identity)
            fit.write_json(stage/"report.json",report)
            fit.write_json(stage/"attempts.json",attempts(root))
            for name in names[3:]:
                (stage/name).parent.mkdir(parents=True,exist_ok=True)
                shutil.copyfile(root/name,stage/name)
            fit.write_json(stage/"handoff-manifest.json",{n:fit.sha256_file(stage/n) for n in names})
            fit.seal_phase(stage,"bundle",signature,{},names+["handoff-manifest.json"])
    names += ["handoff-manifest.json"]
    fit.read_phase(root,"bundle",signature,names)
    if handoff.exists():
        with zipfile.ZipFile(handoff) as z:
            if len(z.namelist()) != len(names) or set(z.namelist()) != set(names):
                raise ValueError("Existing handoff inventory differs; preserved")
            if any(hashlib.sha256(z.read(n)).hexdigest() != fit.sha256_file(root/"bundle"/n) for n in names):
                raise ValueError("Existing handoff differs; preserved")
        return
    handoff.parent.mkdir(parents=True,exist_ok=True)
    staged = Path(tempfile.mkdtemp(prefix=".learnability-export-",dir=handoff.parent))/handoff.name
    with zipfile.ZipFile(staged,"x",zipfile.ZIP_DEFLATED) as z:
        for name in names: z.write(root/"bundle"/name,name)
    staged.rename(handoff)


def verify_reload(runtime, root, step, signature, images, expected):
    runtime.load(root/f"step-{step:04d}/model.pt",signature,step)
    arrays,checks,_ = runtime.snapshot(images)
    with np.load(root/f"step-{step:04d}/arrays.npz",allow_pickle=False) as z:
        if not np.array_equal(arrays["masks"],z["masks"]) or not np.array_equal(arrays["labels"],z["labels"]):
            raise ValueError("Reload synthetic targets differ")
        delta = max(float(np.max(np.abs(arrays[n]-z[n]))) for n in z.files if n not in ("masks","labels"))
    if delta > p.POLICY["reload_tolerance"] or runtime.fingerprints() != expected["fingerprints"]:
        raise ValueError("Reload parameters or raw outputs differ")
    if not checks["state_unchanged"]: raise ValueError("Snapshot changed model state")
    return delta


def execute(root, archive, fit_root, identity, records, handoff, runtime_factory=Runtime, monitor_factory=Monitor):
    signature = fit.digest(identity)
    original = fit_root/"step-028800/model.pt"
    if fit.sha256_file(original) != identity["checkpoint_sha256"]:
        raise ValueError("Original checkpoint changed")
    with fit.locked_run(root):
        if not (root/"identity").exists() and any(x.name != ".run.lock" for x in root.iterdir()):
            raise ValueError("Output without identity; preserve and review")
        publish(root,"identity",signature,identity)
        previous = attempts(root)
        prior_seconds = sum(a["seconds"] for a in previous)
        prior_updates = sum(a["attempted_updates"] for a in previous)
        trajectory = p.read_checkpoints(root,signature)
        if (root/"result").exists():
            report = p.e.value(root,"result",signature)
            if len(trajectory) != len(p.POLICY["snapshots"]): raise ValueError("Completed result missing checkpoints")
            export(root,handoff,identity,report)
            return report
        if prior_seconds >= p.POLICY["resource_plan"]["comparison_seconds"]:
            raise RuntimeError("Cumulative learning diagnostic time exhausted")
        attempt = Path(tempfile.mkdtemp(prefix="attempt-",dir=root))
        fit.write_json(attempt/"started.json",{"identity":signature,"unix_seconds":time.time()})
        started = time.perf_counter(); runtime = None
        attempted = 0; error = None; interrupted = False; reload_delta = None
        monitor = monitor_factory(None,prior_seconds,started,p.POLICY)
        try:
            with monitor, zipfile.ZipFile(archive) as source, ExitStack() as stack:
                images = [stack.enter_context(fit.checked_image(source,row)) for row in identity["job"]["rows"]]
                runtime = runtime_factory(original,identity["fit_signature"])
                monitor.runtime = runtime
                baseline = records["report.json"]["final_fingerprints"]
                if runtime.fingerprints() != baseline: raise ValueError("Copied model differs from original")
                losses = trajectory[-1]["losses"].copy() if trajectory else []
                completed = trajectory[-1]["step"] if trajectory else 0
                if trajectory:
                    verify_reload(runtime,root,completed,signature,images,trajectory[-1])
                target = trajectory[0]["target_sha256"] if trajectory else None
                for stop in p.POLICY["snapshots"]:
                    if trajectory and stop <= completed: continue
                    while len(losses) < stop:
                        monitor.check()
                        if prior_updates+attempted >= p.POLICY["maximum_attempted_updates"]:
                            raise RuntimeError("Attempted-update budget exhausted; no automatic extension")
                        attempted += 1
                        losses.append(runtime.step(images,len(losses),target))
                        if len(losses)%25 == 0:
                            print(f"Fixed synthetic learning {len(losses)}/{p.POLICY['updates']}; loss={losses[-1]['loss']:.6f}",flush=True)
                    arrays,checks,stats = runtime.snapshot(images,measure=stop in (0,p.POLICY["updates"]))
                    if not checks["state_unchanged"]: raise ValueError("Snapshot changed state")
                    current_target = p.target_digest(arrays)
                    if target is None: target = current_target
                    if target != current_target: raise ValueError("Fixed target changed")
                    fingerprints = runtime.fingerprints()
                    if fingerprints["backbone"] != baseline["backbone"]:
                        raise ValueError("Frozen backbone changed")
                    progress = {"step":stop,"losses":losses.copy(),"target_sha256":target,
                        "metrics":p.q.summarize_arrays(arrays),"checks":checks,
                        "feature_noise_measurements":stats,"fingerprints":fingerprints}
                    monitor.check()
                    phase = f"step-{stop:04d}"
                    with fit.phase_stage(root,phase) as stage:
                        runtime.save(stage/"model.pt",signature,stop)
                        np.savez_compressed(stage/"arrays.npz",**arrays)
                        fit.write_json(stage/"progress.json",progress)
                        monitor.check()
                        fit.seal_phase(stage,phase,signature,{},p.FILES)
                    trajectory.append(progress)
                    print(f"Saved step {stop}: train/eval maps, scores, losses and optimizer",flush=True)
                reload_delta = verify_reload(runtime,root,p.POLICY["updates"],signature,images,trajectory[-1])
                monitor.check()
                if fit.sha256_file(original) != identity["checkpoint_sha256"]:
                    raise ValueError("Original checkpoint file changed")
        except BaseException as exc:
            error = f"{type(exc).__name__}: {exc}"
            interrupted = isinstance(exc,KeyboardInterrupt)
            raise
        finally:
            elapsed = time.perf_counter()-started
            fit.write_json(attempt/"finished.json",{"seconds":elapsed,"attempted_updates":attempted,
                "status":"interrupted" if interrupted else ("failed" if error else "completed"),
                "error":error,"memory":monitor.snapshot()})
            monitor.runtime = None
            if runtime is not None: runtime.release()
        report = {"status":"fixed_synthetic_learnability_completed", "completed_updates":p.POLICY["updates"],
            "attempted_updates_including_replayed_work":prior_updates+attempted,
            "cumulative_compute_seconds":prior_seconds+elapsed,
            "trajectory":[{"step":x["step"],"metrics":x["metrics"],"loss_components":x["checks"]["loss_components"]} for x in trajectory],
            "original_checkpoint_unchanged":True,"backbone_unchanged":True,
            "final_reload_max_abs_difference":reload_delta,
            "trainable_components_changed":{k:trajectory[-1]["fingerprints"][k] != trajectory[0]["fingerprints"][k]
                for k in trajectory[0]["fingerprints"] if k != "backbone"},
            "test_images_decoded":0,"test_masks_decoded":0,"threshold_changed":False,
            "service_model_changed":False,"automatic_pass_allowed":False,
            "interpretation":"Repeated same eight training tiles and fixed synthetic task. Not independent accuracy; no automatic model selection."}
        publish(root,"result",signature,report)
        export(root,handoff,identity,report)
        return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project",type=Path,default=ROOT)
    parser.add_argument("--archive",type=Path,required=True)
    args = parser.parse_args(argv); project = args.project.resolve()
    if fit.sha256_file(args.archive) != fit.EXPECTED_ARCHIVE_SHA256: raise ValueError("Original archive changed")
    if shutil.disk_usage(project).free < p.POLICY["free_disk_GiB"]*1024**3: raise ValueError("Need 8 GiB free disk")
    fit_root = project/"artifacts"/p.q.p.NAME
    if not fit_root.is_dir(): raise ValueError("Existing formal fit folder required; do not retrain")
    print("500-update fixed-task diagnosis on a COPY; original model/threshold/service unchanged.",flush=True)
    with fit.locked_run(fit_root):
        records,identity = p.prepare(project,project/"artifacts/handoff",fit_root)
        current = environment()
        for name in ("torch","torchvision","anomalib","timm","supersimplenet_sources"):
            if current[name] != identity["environment"][name]: raise ValueError(f"Runtime environment changed: {name}")
        identity["runtime_environment"] = current
        report = execute(project/"artifacts/analyses"/p.NAME,args.archive,fit_root,identity,records,
                         project/"artifacts/handoff"/(p.NAME+".zip"))
    print(json.dumps({k:v for k,v in report.items() if k != "trajectory"},ensure_ascii=False,indent=2))
    return 0


if __name__ == "__main__":
    try: raise SystemExit(main())
    except (Exception,KeyboardInterrupt) as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}. Preserve outputs; do not change seed, noise, steps or thresholds.",file=sys.stderr)
        raise SystemExit(1)
