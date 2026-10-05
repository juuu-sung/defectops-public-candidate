"""Read-only normal/anomaly gradient attribution on fixed synthetic examples."""
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
from scripts import mpi_supersimplenet_gradient_protocol as p
from scripts.mpi_supersimplenet_gradient_runtime import Runtime, environment
from scripts.run_mpi_efficientad_probe import publish, attempts
from scripts.run_mpi_supersimplenet_evaluation import Monitor

fit = p.fit


def export(root, handoff, identity, report):
    names = ["plan.json","report.json","attempts.json"]
    names += [f"{j['name']}/{n}" for j in p.jobs() for n in ("result.json","arrays.npz","receipt.json")]
    signature = fit.digest(identity)
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
            if (len(z.namelist()) != len(names) or set(z.namelist()) != set(names) or
                    any(hashlib.sha256(z.read(n)).hexdigest()!=fit.sha256_file(root/"bundle"/n) for n in names)):
                raise ValueError("Existing handoff differs; preserve it")
        return
    handoff.parent.mkdir(parents=True,exist_ok=True)
    stage=Path(tempfile.mkdtemp(prefix=".gradient-export-",dir=handoff.parent))/handoff.name
    with zipfile.ZipFile(stage,"x",zipfile.ZIP_DEFLATED) as z:
        for name in names: z.write(root/"bundle"/name,name)
    stage.rename(handoff)


def execute(root, archive, paths, identity, references, handoff, runtime_factory=Runtime, monitor_factory=Monitor):
    signature = fit.digest(identity)
    hashes = {"original":identity["checkpoint_sha256"],"varying015":identity["baseline_checkpoint_sha256"],
              "varying060":identity["strong_checkpoint_sha256"]}
    def check_inputs():
        if any(fit.sha256_file(paths[k])!=h for k,h in hashes.items()):
            raise ValueError("Input model changed")
    check_inputs()
    with fit.locked_run(root):
        if not (root/"identity").exists() and any(x.name!=".run.lock" for x in root.iterdir()):
            raise ValueError("Output without identity; preserve it")
        publish(root,"identity",signature,identity)
        previous = attempts(root)
        results = {j["name"]:p.read_job(root,j,signature) for j in p.jobs() if (root/j["name"]).exists()}
        if (root/"result").exists():
            if len(results)!=len(p.jobs()): raise ValueError("Incomplete completed report")
            report=p.old.v.old.e.value(root,"result",signature)
            export(root,handoff,identity,report); return report
        prior=sum(a["seconds"] for a in previous); prior_jobs=sum(a["attempted_jobs"] for a in previous)
        if prior>=p.POLICY["resource_plan"]["comparison_seconds"]: raise RuntimeError("Cumulative time exhausted")
        attempt=Path(tempfile.mkdtemp(prefix="attempt-",dir=root))
        fit.write_json(attempt/"started.json",{"identity":signature,"unix_seconds":time.time()})
        started=time.perf_counter(); runtime=None; error=None; interrupted=False; attempted=0
        monitor=monitor_factory(None,prior,started,p.POLICY)
        try:
            with monitor,zipfile.ZipFile(archive) as source,ExitStack() as stack:
                images=[stack.enter_context(fit.checked_image(source,r)) for r in identity["job"]["rows"]]
                for state in p.STATES:
                    pending=[j for j in p.jobs() if j["state"]==state and j["name"] not in results]
                    if not pending: continue
                    runtime=runtime_factory(paths["original"],identity["fit_signature"]); monitor.runtime=runtime
                    if state=="varying015": runtime.load_baseline(paths[state],identity["baseline_signature"])
                    if state=="varying060": runtime.load_arm(paths[state],identity["noise_signature"],500,"std060")
                    if runtime.fingerprints()!=identity["state_fingerprints"][state]: raise ValueError("Loaded state differs")
                    for job in pending:
                        monitor.check()
                        if prior_jobs+attempted>=p.POLICY["maximum_attempted_jobs"]: raise RuntimeError("Attempt budget exceeded")
                        attempted+=1
                        expected=p.old.v.arrays_for(p.old.arrays_for(references[state],job["strength"]),job["seed"])
                        result,arrays=runtime.probe(images,job,expected)
                        if runtime.fingerprints()!=identity["state_fingerprints"][state]: raise ValueError("Probe changed model")
                        monitor.check()
                        with fit.phase_stage(root,job["name"]) as stage:
                            fit.write_json(stage/"result.json",result); np.savez_compressed(stage/"arrays.npz",**arrays)
                            fit.seal_phase(stage,job["name"],signature,{},["result.json","arrays.npz"])
                        results[job["name"]]=p.read_job(root,job,signature)
                        print(f"Gradient audit {len(results)}/{len(p.jobs())}: {job['name']}; no updates",flush=True)
                    monitor.check(); monitor.runtime=None; runtime.release(); runtime=None
                check_inputs()
        except BaseException as exc:
            error=f"{type(exc).__name__}: {exc}"; interrupted=isinstance(exc,KeyboardInterrupt)
            raise
        finally:
            elapsed=time.perf_counter()-started
            fit.write_json(attempt/"finished.json",{"seconds":elapsed,"attempted_jobs":attempted,"memory":monitor.snapshot(),
                "status":"interrupted" if interrupted else ("failed" if error else "completed"),"error":error})
            monitor.runtime=None
            if runtime is not None: runtime.release()
        report={"status":"gradient_audit_completed","completed_jobs":len(results),"optimizer_updates":0,
            "cumulative_compute_seconds":prior+elapsed,"attempted_jobs_including_replay":prior_jobs+attempted,
            "input_models_unchanged":True,"test_images_decoded":0,"test_masks_decoded":0,
            "threshold_changed":False,"service_model_changed":False,"automatic_pass_allowed":False,
            "rows":[{"job":j, "loss_terms":results[j['name']]['loss_terms'], "groups":
                {k:v['summary'] for k,v in results[j['name']]['groups'].items()}} for j in p.jobs()],
            "note":"Raw gradients, not AdamW updates; opposition alone is not causal proof. No reset or retraining."}
        publish(root,"result",signature,report); export(root,handoff,identity,report)
        return report


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project",type=Path,default=ROOT); parser.add_argument("--archive",type=Path,required=True)
    args=parser.parse_args(argv); project=args.project.resolve()
    if fit.sha256_file(args.archive)!=fit.EXPECTED_ARCHIVE_SHA256: raise ValueError("Archive differs")
    if shutil.disk_usage(project).free<2*1024**3: raise ValueError("Need 2 GiB free disk")
    formal=project/"artifacts"/p.old.v.q.p.NAME
    fixed=project/"artifacts/analyses"/p.old.v.old.NAME
    varying=project/"artifacts/analyses"/p.old.v.NAME
    noise=project/"artifacts/analyses"/p.old.NAME
    with ExitStack() as stack:
        for path in (formal,fixed,varying,noise):
            if not path.is_dir(): raise ValueError(f"Preserved input folder required: {path}")
            stack.enter_context(fit.locked_run(path))
        identity,references=p.prepare(project,formal,fixed,varying,noise)
        current=environment()
        for key in ("torch","torchvision","anomalib","timm","supersimplenet_sources"):
            if current[key]!=identity['environment'][key]: raise ValueError(f"Environment changed: {key}")
        identity['runtime_environment']=current
        paths={"original":formal/"step-028800/model.pt","varying015":varying/"step-0500/model.pt",
               "varying060":noise/"std060-step-0500/model.pt"}
        report=execute(project/"artifacts/analyses"/p.NAME,args.archive,paths,identity,references,
                       project/"artifacts/handoff"/(p.NAME+".zip"))
    print(json.dumps({k:v for k,v in report.items() if k!='rows'},indent=2)); return 0


if __name__=="__main__":
    try: raise SystemExit(main())
    except (Exception,KeyboardInterrupt) as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}. Preserve all files; do not retrain or change thresholds.",file=sys.stderr)
        raise SystemExit(1)
