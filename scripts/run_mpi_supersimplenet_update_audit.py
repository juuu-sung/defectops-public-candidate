"""Isolate step-zero numerical differences. No elapsed-time cap or saved new models."""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import hashlib
import io
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
from scripts import mpi_supersimplenet_update_audit_protocol as p
from scripts.mpi_supersimplenet_update_audit_runtime import Runtime, environment
from scripts.run_mpi_supersimplenet_evaluation import Monitor as BaseMonitor
from scripts.run_mpi_supersimplenet_parity import check_inputs
from scripts.run_mpi_efficientad_probe import publish, attempts

fit = p.fit
FILES = ["result.json", "arrays.npz"]


class Monitor(BaseMonitor):
    def check(self):
        # Do not pass infinity to the old monitor: unlimited time is explicit policy.
        self.sample()
        if self.runtime is not None:
            memory = self.runtime.memory()
            self.allocated = max(self.allocated,memory["torch_cuda_peak_allocated_bytes"])
            self.reserved = max(self.reserved,memory["torch_cuda_peak_reserved_bytes"])
        if self.error: raise RuntimeError(f"Memory monitor failed: {self.error}")
        if (self.rss > self.limits["rss_GiB"]*1024**3
                or self.reserved > self.limits["selection_and_inference_reserved_GiB"]*1024**3):
            raise RuntimeError("Memory cap exceeded; no automatic shrinking; files preserved")


def read_job(root, job, signature):
    receipt = fit.read_phase(root,job["name"],signature,FILES)
    result = json.loads((root/job["name"]/"result.json").read_bytes())
    if receipt["job"] != job or result["job"] != job or result["model_state_restored"] is not True:
        raise ValueError("Cached job identity/state differs")
    with np.load(root/job["name"]/"arrays.npz",allow_pickle=False) as arrays:
        p.verify_arrays(result,arrays)
    return result


def replay_prior(prior_zip, job, result, arrays):
    with zipfile.ZipFile(prior_zip) as z:
        base = "jobs/"+job["name"]+"/"
        prior = json.loads(z.read(base+"result.json"))
        if prior["noise_sha256"] != result["noise_sha256"]:
            raise ValueError("Frozen synthetic noise differs from reviewed parity run")
        with np.load(io.BytesIO(z.read(base+"arrays.npz")),allow_pickle=False) as old:
            if any(not np.array_equal(old[k],arrays[k]) for k in ("masks","labels")):
                raise ValueError("Frozen synthetic targets changed")
            checks = {k:p.parity.compare(arrays["official_a_pre_train_synthetic_"+k],old["official_"+k],"outputs")
                      for k in ("maps","scores")}
            if not all(v["passed"] for v in checks.values()):
                raise ValueError("Original step-zero forward replay differs; preserve evidence")
            return checks


def export(root,handoff,identity,report):
    raw = {"plan.json":fit.canonical(identity),"report.json":fit.canonical(report),"attempts.json":fit.canonical(attempts(root))}
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
    staged = Path(tempfile.mkdtemp(prefix=".update-audit-",dir=handoff.parent))/handoff.name
    with zipfile.ZipFile(staged,"x",zipfile.ZIP_DEFLATED) as z:
        for k,v in raw.items(): z.writestr(k,v)
    staged.rename(handoff)


def execute(root,archive,fit_root,identity,handoff,prior_zip,runtime_factory=Runtime,monitor_factory=Monitor):
    signature = fit.digest(identity)
    check_inputs(fit_root,identity)
    if fit.sha256_file(prior_zip) != identity["prior_parity_zip_sha256"]:
        raise ValueError("Prior parity ZIP changed")
    with fit.locked_run(root):
        if not (root/"identity").exists() and any(x.name!=".run.lock" for x in root.iterdir()):
            raise ValueError("Output exists without identity; preserved")
        publish(root,"identity",signature,identity)
        previous = attempts(root)
        job_root = root/"jobs"; job_root.mkdir(exist_ok=True)
        if any(x.name not in {j['name'] for j in identity['jobs']} and not x.name.startswith('.') for x in job_root.iterdir()):
            raise ValueError("Unexpected completed job")
        done = {j["name"]:read_job(job_root,j,signature) for j in identity["jobs"] if (job_root/j["name"]).exists()}
        if (root/"result").exists():
            report = p.parity.e.value(root,"result",signature)
            if len(done) != len(identity["jobs"]): raise ValueError("Completed audit lacks jobs")
            export(root,handoff,identity,report)
            return report
        prior = sum(a["seconds"] for a in previous)
        tried = sum(a["attempted_jobs"] for a in previous)
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
                        raise RuntimeError("Fixed job attempt cap exhausted; no reroll")
                    count += 1
                    print(f"{count}/2 {job['name']}: four matched paths; no elapsed-time cap",flush=True)
                    with ExitStack() as stack:
                        images = [stack.enter_context(fit.checked_image(z,row)) for row in identity["rows"]]
                        result,arrays = runtime.compare_job(images,job,fit_root/"step-000000/model.pt",
                                                           identity["fit_signature"],identity["checkpoints"]["0"]["fingerprints"])
                    result["prior_forward_replay"] = replay_prior(prior_zip,job,result,arrays)
                    monitor.check()
                    with fit.phase_stage(job_root,job["name"]) as stage:
                        fit.write_json(stage/"result.json",result)
                        np.savez_compressed(stage/"arrays.npz",**arrays)
                        fit.seal_phase(stage,job["name"],signature,{"job":job},FILES)
                    done[job["name"]] = read_job(job_root,job,signature)
                    print(f"Saved comparisons: {result['pair_passed']}",flush=True)
                check_inputs(fit_root,identity)
                monitor.check(); status = "completed"
        except BaseException as exc:
            error = f"{type(exc).__name__}: {exc}"
            status = "interrupted" if isinstance(exc,KeyboardInterrupt) else "failed"
            raise
        finally:
            fit.write_json(attempt/"finished.json",{"status":status,"error":error,"seconds":time.perf_counter()-started,
                                                    "attempted_jobs":count,"memory":monitor.snapshot()})
            monitor.runtime = None
            if runtime is not None: runtime.release()
        report = {"status":"update_audit_completed_requires_interpretation","completed_jobs":len(done),
                  "pairs_all_jobs_matched":{pair:all(r["pair_passed"][pair] for r in done.values()) for pair in p.PAIRS},
                  "completed_disposable_updates":4*len(done),"attempted_jobs_including_interrupted":tried+count,
                  "cumulative_seconds":prior+time.perf_counter()-started,"elapsed_time_cap_seconds":None,
                  "original_checkpoints_unchanged":True,"saved_updated_models":0,
                  "threshold_changed":False,"service_model_changed":False,"test_images_decoded":0,
                  "test_masks_decoded":0,"automatic_pass_allowed":False,"later_phases_automatically_started":False,
                  "scope":"Step-zero algebra/numerical controls on eight tiles of one normal photo; not accuracy or generalization."}
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
    fit_root = project/"artifacts"/p.parity.e.p.NAME
    if not fit_root.is_dir(): raise ValueError("Original formal fit folder missing; do not retrain")
    handoffs = project/"artifacts/handoff"
    print("Preserving original fit and prior diagnostics. No time limit; memory safeguards remain.",flush=True)
    with fit.locked_run(fit_root):
        identity = p.prepare(project,handoffs,fit_root)
        current = environment()
        for key in ("torch","torchvision","anomalib","timm","supersimplenet_sources"):
            if current[key] != identity["environment"][key]: raise ValueError(f"Environment changed: {key}")
        identity["runtime_environment"] = current
        report = execute(project/"artifacts/analyses"/p.NAME,args.archive,fit_root,identity,
                         handoffs/(p.NAME+".zip"),handoffs/(p.parity.NAME+".zip"))
    print(json.dumps(report,ensure_ascii=False,indent=2))
    return 0


if __name__ == "__main__":
    try: raise SystemExit(main())
    except (Exception,KeyboardInterrupt) as exc:
        traceback.print_exc()
        print(f"ERROR: {type(exc).__name__}: {exc}. Preserve all files; do not retrain or change thresholds.",file=sys.stderr)
        raise SystemExit(1)
