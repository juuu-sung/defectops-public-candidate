"""Normal-only frozen-model functional diagnosis with resumable per-batch evidence."""
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
sys.path[:0] = [str(ROOT), str(ROOT/"src")]
from scripts import mpi_supersimplenet_synthetic_protocol as q
from scripts.mpi_supersimplenet_synthetic_runtime import Runtime, environment
from scripts.run_mpi_supersimplenet_evaluation import Monitor
from scripts.run_mpi_efficientad_probe import publish, attempts

fit = q.fit


def export(root, handoff, identity, report):
    signature = fit.digest(identity)
    names = ["plan.json", "report.json", "attempts.json"]
    names += [f"trials/{j['name']}/{n}" for j in identity["jobs"] for n in q.FILES+["receipt.json"]]
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
                raise ValueError("Existing handoff inventory differs; preserved")
            if any(hashlib.sha256(z.read(n)).hexdigest()!=fit.sha256_file(root/"bundle"/n) for n in names):
                raise ValueError("Existing handoff changed; preserved")
        return
    handoff.parent.mkdir(parents=True,exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".synthetic-export-",dir=handoff.parent))/handoff.name
    with zipfile.ZipFile(stage,"x",zipfile.ZIP_DEFLATED) as z:
        for n in names: z.write(root/"bundle"/n,n)
    stage.rename(handoff)


def execute(root, archive, fit_root, identity, records, handoff, runtime_factory=Runtime, monitor_factory=Monitor):
    signature = fit.digest(identity)
    checkpoint = fit_root/"step-028800/model.pt"
    if fit.sha256_file(checkpoint) != identity["checkpoint_sha256"]:
        raise ValueError("Checkpoint changed before diagnostic")
    with fit.locked_run(root):
        if not (root/"identity").exists() and any(x.name!=".run.lock" for x in root.iterdir()):
            raise ValueError("Output without identity; preserve and review")
        publish(root,"identity",signature,identity)
        prior_attempts = attempts(root)
        prior = sum(a["seconds"] for a in prior_attempts)
        prior_trials = sum(a["attempted_trials"] for a in prior_attempts)
        trials_root = root/"trials"; trials_root.mkdir(exist_ok=True)
        completed = {j["name"]:q.read_trial(trials_root,j,signature) for j in identity["jobs"] if (trials_root/j["name"]).exists()}
        if (root/"result").exists():
            result = q.e.value(root,"result",signature)
            if len(completed)!=len(identity["jobs"]): raise ValueError("Completed run is missing trials")
            export(root,handoff,identity,result)
            return result
        if prior >= q.POLICY["resource_plan"]["comparison_seconds"]:
            raise RuntimeError("Cumulative diagnostic time budget exhausted")
        attempt = Path(tempfile.mkdtemp(prefix="attempt-",dir=root))
        fit.write_json(attempt/"started.json",{"identity":signature,"unix_seconds":time.time()})
        started = time.perf_counter(); runtime = None; attempted = 0; error = None; interrupted = False
        monitor = monitor_factory(None,prior,started,q.POLICY)
        try:
            with monitor, zipfile.ZipFile(archive) as source:
                runtime = runtime_factory(checkpoint,identity["fit_signature"]); monitor.runtime = runtime
                if runtime.fingerprints()!=records["report.json"]["final_fingerprints"]:
                    raise ValueError("Loaded model differs from fit")
                for job in identity["jobs"]:
                    if job["name"] in completed: continue
                    monitor.check()
                    if prior_trials+attempted >= q.POLICY["maximum_attempted_trials"]:
                        raise RuntimeError("Diagnostic attempted-trial limit exhausted; no reroll")
                    attempted += 1
                    with ExitStack() as stack:
                        images = [stack.enter_context(fit.checked_image(source,row)) for row in job["rows"]]
                        before = runtime.predict(images[0])
                        replay = None
                        if job["group"]=="development":
                            index = job["row_start"]
                            expected = np.load(fit_root/"development"/f"photo-{index//12:03d}/predictions.npy",allow_pickle=False)[index%12]
                            replay = float(np.max(np.abs(before-expected)))
                            if replay > 1e-4: raise ValueError("Saved normal prediction replay differs")
                        arrays,checks = runtime.trial(images,job["seed"])
                        if not checks["state_unchanged"]:
                            raise ValueError("Synthetic trial changed model state")
                        after = runtime.predict(images[0])
                        delta = float(np.max(np.abs(before-after)))
                        if delta > 1e-4: raise ValueError("Normal prediction changed after temporary train mode")
                        checks.update(normal_before_after_max_abs_difference=delta, saved_normal_replay_max_abs_difference=replay)
                    result = {"job":job["name"],"metrics":q.summarize_arrays(arrays),"checks":checks}
                    monitor.check()
                    with fit.phase_stage(trials_root,job["name"]) as stage:
                        np.savez_compressed(stage/"arrays.npz",**arrays)
                        fit.write_json(stage/"result.json",result)
                        fit.seal_phase(stage,job["name"],signature,{"job_sha256":fit.digest(job)},q.FILES)
                    completed[job["name"]]=result
                    print(f"Frozen synthetic check {len(completed)}/{len(identity['jobs'])}: {job['name']} saved",flush=True)
                if runtime.fingerprints()!=records["report.json"]["final_fingerprints"]:
                    raise ValueError("Final parameters/buffers changed")
                monitor.check()
                if fit.sha256_file(checkpoint)!=identity["checkpoint_sha256"]:
                    raise ValueError("Checkpoint file changed")
        except BaseException as exc:
            error = f"{type(exc).__name__}: {exc}"
            interrupted = isinstance(exc, KeyboardInterrupt)
            raise
        finally:
            elapsed=time.perf_counter()-started
            fit.write_json(attempt/"finished.json",{"seconds":elapsed,"attempted_trials":attempted,
                "error":error,"status":"interrupted" if interrupted else ("failed" if error else "completed"), "memory":monitor.snapshot()})
            monitor.runtime=None
            if runtime is not None: runtime.release()
        results=[completed[j["name"]] for j in identity["jobs"]]
        report={"status":"frozen_synthetic_function_check_completed", "completed_trials":len(results),
            "groups":q.aggregate(results,identity["jobs"]),"cumulative_compute_seconds":prior+elapsed,
            "optimizer_updates":0,"test_images_decoded":0,"test_masks_decoded":0,
            "checkpoint_unchanged":True,"threshold_changed":False,"service_model_changed":False,
            "automatic_pass_allowed":False,"causal_conclusion_automatic":False,
            "note":"Repeated synthetic presentations are correlated; no independent defect accuracy or service approval."}
        publish(root,"result",signature,report)
        export(root,handoff,identity,report)
        return report


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project",type=Path,default=ROOT)
    parser.add_argument("--archive",type=Path,required=True)
    args=parser.parse_args(argv); project=args.project.resolve()
    if fit.sha256_file(args.archive)!=fit.EXPECTED_ARCHIVE_SHA256: raise ValueError("Original archive changed")
    if shutil.disk_usage(project).free < q.POLICY["free_disk_GiB"]*1024**3: raise ValueError("Need 2 GiB free disk")
    fit_root=project/"artifacts"/q.p.NAME
    if not fit_root.is_dir(): raise ValueError("Existing formal fit folder required; do not retrain")
    print("Validating fixed fit and normal-only inputs; no test image access.",flush=True)
    with fit.locked_run(fit_root):
        records,identity=q.prepare(project,project/"artifacts/handoff",fit_root)
        current=environment()
        for name in ("torch","torchvision","anomalib","timm","supersimplenet_sources"):
            if current[name]!=identity["environment"][name]: raise ValueError(f"Runtime environment changed: {name}")
        identity["runtime_environment"]=current
        report=execute(project/"artifacts/analyses"/q.NAME,args.archive,fit_root,identity,records,
                       project/"artifacts/handoff"/(q.NAME+".zip"))
    print(json.dumps(report,ensure_ascii=False,indent=2))
    return 0


if __name__=="__main__":
    try: raise SystemExit(main())
    except (Exception,KeyboardInterrupt) as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}. Preserve all outputs; do not retrain or change thresholds.",file=sys.stderr)
        raise SystemExit(1)
