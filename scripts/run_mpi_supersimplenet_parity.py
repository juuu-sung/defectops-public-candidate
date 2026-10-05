"""Compare implementations on saved step0/final copies. Never save updated weights."""
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
import traceback
import zipfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/"src")]
from scripts import mpi_supersimplenet_parity_protocol as p
from scripts.mpi_supersimplenet_parity_runtime import Runtime, environment
from scripts.run_mpi_supersimplenet_evaluation import Monitor
from scripts.run_mpi_efficientad_probe import publish, attempts

fit = p.fit
FILES = ["result.json", "arrays.npz"]


def read_job(root, job, signature):
    receipt = fit.read_phase(root, job["name"], signature, FILES)
    result = json.loads((root/job["name"]/"result.json").read_bytes())
    if receipt["job"] != job or result["job"] != job or result["model_state_restored"] is not True:
        raise ValueError("Cached job identity/state differs")
    with np.load(root/job["name"]/"arrays.npz", allow_pickle=False) as z:
        for k in ("maps", "scores"):
            if result["checks"][k] != p.compare(z["official_"+k],z["reference_"+k],"outputs"):
                raise ValueError("Cached output metrics differ")
    return result


def check_inputs(fit_root, identity):
    for step, record in identity["checkpoints"].items():
        if fit.sha256_file(fit_root/f"step-{int(step):06d}/model.pt") != record["sha256"]:
            raise ValueError("Saved checkpoint changed")
    normal = json.loads((fit_root/"calibrated/value.json").read_bytes())
    if fit.digest(normal["contract"]) != identity["contract_sha256"]:
        raise ValueError("Original threshold contract changed")


def export(root, handoff, identity, report):
    raw = {"plan.json":fit.canonical(identity), "report.json":fit.canonical(report),
           "attempts.json":fit.canonical(attempts(root))}
    for job in identity["jobs"]:
        read_job(root/"jobs",job,fit.digest(identity))
        for name in FILES+["receipt.json"]:
            key = f"jobs/{job['name']}/{name}"
            raw[key] = (root/key).read_bytes()
    raw["handoff-manifest.json"] = fit.canonical({k:hashlib.sha256(v).hexdigest() for k,v in raw.items()})
    if handoff.exists():
        with zipfile.ZipFile(handoff) as z:
            if len(z.namelist()) != len(raw) or set(z.namelist()) != set(raw) or any(z.read(k)!=v for k,v in raw.items()):
                raise ValueError("Existing handoff differs; preserved")
        return
    handoff.parent.mkdir(parents=True,exist_ok=True)
    staged = Path(tempfile.mkdtemp(prefix=".parity-export-",dir=handoff.parent))/handoff.name
    with zipfile.ZipFile(staged,"x",zipfile.ZIP_DEFLATED) as z:
        for k,v in raw.items(): z.writestr(k,v)
    staged.rename(handoff)


def execute(root, archive, fit_root, identity, handoff, runtime_factory=Runtime, monitor_factory=Monitor):
    signature = fit.digest(identity)
    check_inputs(fit_root,identity)
    with fit.locked_run(root):
        if not (root/"identity").exists() and any(x.name!=".run.lock" for x in root.iterdir()):
            raise ValueError("Output exists without identity; preserved")
        publish(root,"identity",signature,identity)
        previous = attempts(root)
        done = {}
        job_root = root/"jobs"; job_root.mkdir(exist_ok=True)
        expected_names = {j["name"] for j in identity["jobs"]}
        if any(x.name not in expected_names and not x.name.startswith(".") for x in job_root.iterdir()):
            raise ValueError("Unexpected completed job")
        for job in identity["jobs"]:
            if (job_root/job["name"]).exists(): done[job["name"]] = read_job(job_root,job,signature)
        if (root/"result").exists():
            report = p.e.value(root,"result",signature)
            if len(done) != len(identity["jobs"]): raise ValueError("Completed audit lacks evidence")
            export(root,handoff,identity,report)
            return report
        prior = sum(a["seconds"] for a in previous)
        tried = sum(a["attempted_jobs"] for a in previous)
        if prior >= p.POLICY["resource_plan"]["comparison_seconds"]:
            raise RuntimeError("Audit time budget exhausted; no automatic extension")
        attempt = Path(tempfile.mkdtemp(prefix="attempt-",dir=root))
        fit.write_json(attempt/"started.json",{"identity":signature,"unix_seconds":time.time()})
        started = time.perf_counter(); runtime = None; count = 0; status = "failed"; error = None
        monitor = monitor_factory(None,prior,started,p.POLICY)
        try:
            with monitor, zipfile.ZipFile(archive) as z:
                names = z.namelist()
                if any(names.count(row["member"]) != 1 or "/train/good/" not in row["member"] for row in identity["rows"]):
                    raise ValueError("Missing/duplicate/non-normal input")
                runtime = runtime_factory(fit_root/"step-028800/model.pt",identity["fit_signature"])
                monitor.runtime = runtime
                for job in identity["jobs"]:
                    if job["name"] in done: continue
                    monitor.check()
                    if tried+count >= p.POLICY["maximum_attempted_jobs"]:
                        raise RuntimeError("Audit job cap exhausted; no reroll")
                    count += 1
                    print(f"Comparing {job['name']}: forward, loss, gradients, disposable AdamW update",flush=True)
                    with ExitStack() as stack:
                        images = [stack.enter_context(fit.checked_image(z,row)) for row in identity["rows"]]
                        result,arrays = runtime.compare_job(images,job,
                            fit_root/f"step-{job['step']:06d}/model.pt",identity["fit_signature"],
                            identity["checkpoints"][str(job["step"])]["fingerprints"])
                    monitor.check()
                    with fit.phase_stage(job_root,job["name"]) as stage:
                        fit.write_json(stage/"result.json",result)
                        np.savez_compressed(stage/"arrays.npz",**arrays)
                        fit.seal_phase(stage,job["name"],signature,{"job":job},FILES)
                    done[job["name"]] = read_job(job_root,job,signature)
                    print(f"Saved {job['name']}: {'matched' if result['passed'] else 'DIFFERENCE NEEDS REVIEW'}",flush=True)
                check_inputs(fit_root,identity)
                monitor.check()
                status = "completed"
        except BaseException as exc:
            error = f"{type(exc).__name__}: {exc}"
            status = "interrupted" if isinstance(exc,KeyboardInterrupt) else "failed"
            raise
        finally:
            fit.write_json(attempt/"finished.json",{"status":status,"error":error,"seconds":time.perf_counter()-started,
                "attempted_jobs":count,"memory":monitor.snapshot()})
            monitor.runtime = None
            if runtime is not None: runtime.release()
        passed = all(r["passed"] for r in done.values())
        report = {"status":"implementation_parity_matched" if passed else "implementation_parity_needs_review",
            "completed_jobs":len(done),"all_matched":passed,
            "jobs":{k:{"passed":v["passed"],"loss_values":v["loss_values"]} for k,v in done.items()},
            "original_checkpoints_unchanged":True,"threshold_changed":False,"service_model_changed":False,
            "test_images_decoded":0,"test_masks_decoded":0,"saved_updated_models":0,
            "completed_disposable_updates":2*len(done),"attempted_jobs_including_interrupted":tried+count,
            "cumulative_seconds":prior+time.perf_counter()-started,
            "automatic_pass_allowed":False,"later_phases_automatically_started":False,
            "scope":"Shared frozen backbone; independently transcribed adaptor/head/loss. Not full author reproduction or accuracy."}
        publish(root,"result",signature,report)
        export(root,handoff,identity,report)
        return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project",type=Path,default=ROOT)
    parser.add_argument("--archive",type=Path,required=True)
    args = parser.parse_args(argv); project = args.project.resolve()
    if fit.sha256_file(args.archive) != fit.EXPECTED_ARCHIVE_SHA256: raise ValueError("Original archive changed")
    if shutil.disk_usage(project).free < p.POLICY["free_disk_GiB"]*1024**3: raise ValueError("Need 2 GiB free disk")
    fit_root = project/"artifacts"/p.e.p.NAME
    if not fit_root.is_dir(): raise ValueError("Keep original formal fit folder; do not retrain")
    print("Validating preserved step0 and final checkpoints; no test access or new fit.",flush=True)
    with fit.locked_run(fit_root):
        identity = p.prepare(project,project/"artifacts/handoff",fit_root)
        current = environment()
        for key in ("torch","torchvision","anomalib","timm","supersimplenet_sources"):
            if current[key] != identity["environment"][key]: raise ValueError(f"Environment changed: {key}")
        identity["runtime_environment"] = current
        report = execute(project/"artifacts/analyses"/p.NAME,args.archive,fit_root,identity,
                         project/"artifacts/handoff"/(p.NAME+".zip"))
    print(json.dumps(report,ensure_ascii=False,indent=2))
    return 0


if __name__ == "__main__":
    try: raise SystemExit(main())
    except (Exception,KeyboardInterrupt) as exc:
        traceback.print_exc()
        print(f"ERROR: {type(exc).__name__}: {exc}. Keep all evidence; do not retrain, reroll or change thresholds.",file=sys.stderr)
        raise SystemExit(1)
