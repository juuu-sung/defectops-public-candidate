"""Normal-only changing-synthetic learning and matched read-only fixed-model control."""
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
from scripts import mpi_supersimplenet_varying_protocol as p
from scripts.mpi_supersimplenet_varying_runtime import Runtime, environment
from scripts.run_mpi_supersimplenet_evaluation import Monitor
from scripts.run_mpi_efficientad_probe import publish, attempts

fit = p.fit


def evaluate(runtime, images, monitor):
    evaluations,flat = {},{}
    before = runtime.fingerprints()
    for seed in p.POLICY["evaluation_seeds"]:
        monitor.check()
        arrays,checks = runtime.evaluate_seed(images,seed)
        if not checks["state_unchanged"]: raise ValueError("Evaluation changed model")
        evaluations[str(seed)] = {**p.summarize(arrays),"checks":checks}
        flat.update({f"s{seed}__{k}":v for k,v in arrays.items()})
    if runtime.fingerprints() != before: raise ValueError("Evaluation changed parameters/buffers")
    monitor.check()
    return evaluations,flat


def compare_arrays(actual, expected):
    if set(actual) != set(expected): raise ValueError("Replay array inventory differs")
    delta = 0.
    for name,a in actual.items():
        b = expected[name]
        if a.shape != b.shape or a.dtype != b.dtype: raise ValueError("Replay array shape/dtype differs")
        if a.dtype == np.bool_:
            if not np.array_equal(a,b): raise ValueError("Replay targets differ")
        else:
            if not np.isfinite(a).all() or not np.isfinite(b).all(): raise ValueError("Nonfinite replay")
            delta = max(delta,float(np.max(np.abs(a-b))))
    if delta > p.POLICY["reload_tolerance"]: raise ValueError("Replay raw outputs differ")
    return delta


def check_targets(evaluations, control):
    for seed in p.POLICY["evaluation_seeds"]:
        if evaluations[str(seed)]["target_sha256"] != control["evaluations"][str(seed)]["target_sha256"]:
            raise ValueError("Models were not evaluated on identical synthetic target locations")


def reload_verify(runtime, root, progress, signature, images, monitor):
    step = progress["step"]
    runtime.load(root/f"step-{step:04d}/model.pt",signature,step)
    _,actual = evaluate(runtime,images,monitor)
    with np.load(root/f"step-{step:04d}/arrays.npz",allow_pickle=False) as z:
        delta = compare_arrays(actual,{k:z[k] for k in z.files})
    if runtime.fingerprints() != progress["fingerprints"]: raise ValueError("Reload fingerprints differ")
    return delta


def export(root, handoff, identity, report):
    signature = fit.digest(identity)
    phases = ["fixed-control"]+[f"step-{s:04d}" for s in p.POLICY["snapshots"]]
    names = ["plan.json","report.json","attempts.json"]
    names += [f"{phase}/{n}" for phase in phases for n in p.EVIDENCE_FILES+["receipt.json"]]
    if not (root/"bundle").exists():
        with fit.phase_stage(root,"bundle") as stage:
            fit.write_json(stage/"plan.json",identity); fit.write_json(stage/"report.json",report)
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
                raise ValueError("Existing handoff inventory differs")
            if any(hashlib.sha256(z.read(n)).hexdigest()!=fit.sha256_file(root/"bundle"/n) for n in names):
                raise ValueError("Existing handoff differs; preserve it")
        return
    handoff.parent.mkdir(parents=True,exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".varying-export-",dir=handoff.parent))/handoff.name
    with zipfile.ZipFile(stage,"x",zipfile.ZIP_DEFLATED) as z:
        for name in names: z.write(root/"bundle"/name,name)
    stage.rename(handoff)


def execute(root, archive, fit_root, fixed_root, identity, records, references, handoff,
            runtime_factory=Runtime, monitor_factory=Monitor):
    signature = fit.digest(identity)
    original = fit_root/"step-028800/model.pt"
    fixed = fixed_root/"step-0500/model.pt"
    def check_originals():
        if (fit.sha256_file(original)!=identity["checkpoint_sha256"] or
                fit.sha256_file(fixed)!=identity["fixed_control_checkpoint_sha256"]):
            raise ValueError("Original or fixed-control checkpoint changed")
    check_originals()
    with fit.locked_run(root):
        if not (root/"identity").exists() and any(x.name!=".run.lock" for x in root.iterdir()):
            raise ValueError("Output without identity; preserve and review")
        publish(root,"identity",signature,identity)
        previous = attempts(root)
        prior_seconds = sum(a["seconds"] for a in previous)
        prior_updates = sum(a["attempted_updates"] for a in previous)
        control = p.read_phase(root,"fixed-control",signature)[0] if (root/"fixed-control").exists() else None
        trajectory = p.read_checkpoints(root,signature)
        if trajectory and control is None: raise ValueError("Missing fixed control")
        for progress in trajectory: check_targets(progress["evaluations"],control)
        if (root/"result").exists():
            report = p.old.e.value(root,"result",signature)
            if len(trajectory)!=len(p.POLICY["snapshots"]): raise ValueError("Missing completed checkpoints")
            export(root,handoff,identity,report); return report
        if prior_seconds >= p.POLICY["resource_plan"]["comparison_seconds"]:
            raise RuntimeError("Cumulative time exhausted")
        attempt = Path(tempfile.mkdtemp(prefix="attempt-",dir=root))
        fit.write_json(attempt/"started.json",{"identity":signature,"unix_seconds":time.time()})
        started = time.perf_counter(); runtime = None; error = None; interrupted = False; attempted = 0
        monitor = monitor_factory(None,prior_seconds,started,p.POLICY)
        try:
            with monitor,zipfile.ZipFile(archive) as source,ExitStack() as stack:
                images = [stack.enter_context(fit.checked_image(source,r)) for r in identity["job"]["rows"]]
                baseline = records["report.json"]["final_fingerprints"]
                if control is None:
                    print("1/3 Read-only fixed500 control on reference42 + four held-out synthetic seeds",flush=True)
                    runtime = runtime_factory(original,identity["fit_signature"]); monitor.runtime=runtime
                    runtime.load_fixed_control(fixed,identity["fixed_control_signature"])
                    if runtime.fingerprints()!=identity["fixed_control_fingerprints"]:
                        raise ValueError("Loaded fixed control differs")
                    evaluations,flat = evaluate(runtime,images,monitor)
                    replay = compare_arrays(p.arrays_for(flat,42),references[500])
                    control = {"step":500,"losses":[],"optimizer_updates":0,"evaluations":evaluations,
                               "fingerprints":runtime.fingerprints(),"previous_reference_max_abs_difference":replay}
                    with fit.phase_stage(root,"fixed-control") as stage:
                        np.savez_compressed(stage/"arrays.npz",**flat); fit.write_json(stage/"progress.json",control)
                        fit.seal_phase(stage,"fixed-control",signature,{},p.EVIDENCE_FILES)
                    monitor.check(); monitor.runtime=None; runtime.release(); runtime=None
                print("2/3 Fresh copy of original final model; eight fixed tiles, 500 varying-seed updates",flush=True)
                runtime = runtime_factory(original,identity["fit_signature"]); monitor.runtime=runtime
                if runtime.fingerprints()!=baseline: raise ValueError("New arm did not start from original fit")
                losses = trajectory[-1]["losses"].copy() if trajectory else []
                completed = trajectory[-1]["step"] if trajectory else -1
                if trajectory: reload_verify(runtime,root,trajectory[-1],signature,images,monitor)
                for stop in p.POLICY["snapshots"]:
                    if stop <= completed: continue
                    while len(losses)<stop:
                        monitor.check()
                        if prior_updates+attempted>=p.POLICY["maximum_attempted_updates"]:
                            raise RuntimeError("Attempted-update limit; no automatic extension")
                        attempted+=1
                        losses.append(runtime.step(images,len(losses)))
                        if len(losses)%25==0:
                            print(f"Varying synthetic learning {len(losses)}/500; loss={losses[-1]['loss']:.6f}",flush=True)
                    evaluations,flat = evaluate(runtime,images,monitor)
                    check_targets(evaluations,control)
                    replay = compare_arrays(p.arrays_for(flat,42),references[0]) if stop==0 else None
                    fingerprints = runtime.fingerprints()
                    if fingerprints["backbone"]!=baseline["backbone"]: raise ValueError("Backbone changed")
                    progress = {"step":stop,"losses":losses.copy(),"evaluations":evaluations,
                                "fingerprints":fingerprints,"previous_reference_max_abs_difference":replay}
                    phase = f"step-{stop:04d}"
                    with fit.phase_stage(root,phase) as stage:
                        runtime.save(stage/"model.pt",signature,stop)
                        np.savez_compressed(stage/"arrays.npz",**flat); fit.write_json(stage/"progress.json",progress)
                        monitor.check(); fit.seal_phase(stage,phase,signature,{},p.FILES)
                    trajectory.append(progress)
                    print(f"Saved step {stop}: four held-out seeds + reference, including paired clean maps",flush=True)
                print("3/3 Reloading final model and reproducing all five evaluation seeds",flush=True)
                reload_delta = reload_verify(runtime,root,trajectory[-1],signature,images,monitor)
                monitor.check(); check_originals()
        except BaseException as exc:
            error=f"{type(exc).__name__}: {exc}"; interrupted=isinstance(exc,KeyboardInterrupt)
            raise
        finally:
            elapsed=time.perf_counter()-started
            fit.write_json(attempt/"finished.json",{"seconds":elapsed,"attempted_updates":attempted,
                "status":"interrupted" if interrupted else ("failed" if error else "completed"),
                "error":error,"memory":monitor.snapshot()})
            monitor.runtime=None
            if runtime is not None: runtime.release()
        report = {"status":"varying_synthetic_diagnostic_completed","completed_updates":p.POLICY["updates"],
            "attempted_updates_including_replayed_work":prior_updates+attempted,"cumulative_compute_seconds":prior_seconds+elapsed,
            "fixed_control_heldout":p.aggregate(control["evaluations"]),
            "trajectory":[{"step":x["step"],"heldout":p.aggregate(x["evaluations"]),
                           "reference42":x["evaluations"]["42"]} for x in trajectory],
            "unique_training_target_hashes":len({r["target_sha256"] for r in trajectory[-1]["losses"]}),
            "final_reload_max_abs_difference":reload_delta,"original_checkpoints_unchanged":True,
            "test_images_decoded":0,"test_masks_decoded":0,"threshold_changed":False,"service_model_changed":False,
            "automatic_pass_allowed":False,"automatic_model_selection":False,
            "note":"Held-out synthetic seeds on the same eight training tiles; not unseen photos or actual defect accuracy."}
        publish(root,"result",signature,report); export(root,handoff,identity,report)
        return report


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project",type=Path,default=ROOT)
    parser.add_argument("--archive",type=Path,required=True)
    args=parser.parse_args(argv); project=args.project.resolve()
    if fit.sha256_file(args.archive)!=fit.EXPECTED_ARCHIVE_SHA256: raise ValueError("Original archive changed")
    if shutil.disk_usage(project).free<p.POLICY["free_disk_GiB"]*1024**3: raise ValueError("Need 8 GiB free disk")
    fit_root=project/"artifacts"/p.q.p.NAME
    fixed_root=project/"artifacts/analyses"/p.old.NAME
    if not fit_root.is_dir() or not fixed_root.is_dir(): raise ValueError("Original fit and prior fixed500 folders required")
    with fit.locked_run(fit_root),fit.locked_run(fixed_root):
        records,identity,references=p.prepare(project,project/"artifacts/handoff",fit_root,fixed_root)
        current=environment()
        for name in ("torch","torchvision","anomalib","timm","supersimplenet_sources"):
            if current[name]!=identity["environment"][name]: raise ValueError(f"Runtime environment changed: {name}")
        identity["runtime_environment"]=current
        report=execute(project/"artifacts/analyses"/p.NAME,args.archive,fit_root,fixed_root,identity,records,references,
                       project/"artifacts/handoff"/(p.NAME+".zip"))
    print(json.dumps({k:v for k,v in report.items() if k not in ("trajectory","fixed_control_heldout")},ensure_ascii=False,indent=2))
    return 0


if __name__=="__main__":
    try: raise SystemExit(main())
    except (Exception,KeyboardInterrupt) as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}. Preserve outputs; no seed reroll, retraining or threshold changes.",file=sys.stderr)
        raise SystemExit(1)
