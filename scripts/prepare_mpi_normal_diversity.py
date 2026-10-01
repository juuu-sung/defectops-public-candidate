"""Prepare a fixed 64-photo normal cohort. Never fit, calibrate, or score test images."""
from __future__ import annotations

import argparse
from collections import Counter
import importlib.metadata
import json
from pathlib import Path
import sys
import tempfile
import threading
import time
import zipfile

import numpy as np

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parents[1]/"src")]
from scripts import mpi_normal_diversity_protocol as protocol
from scripts import export_mpi_normal_visual_review as bundle_tools

fit = protocol.fit
NAME = "mpi-normal-diversity-preparation-v1"


class FeatureRuntime(fit.TorchRuntime):
    def means(self, image, compare_reference=False):
        torch = self.torch
        model = self.model.model
        if model.training or model.feature_extractor.training or model.tiler is not None or model.embedding_store:
            raise ValueError("Expected frozen eval feature extractor without tiling or accumulated embeddings")
        with torch.no_grad():
            tensor = fit._preprocess(image, self.config["image_size"], self.gpu)
            features = model.feature_extractor(tensor)
            pooled = {k: model.feature_pooler(v) for k, v in features.items()}
            embedding = model.generate_embedding(pooled)
            if tuple(embedding.shape) != (1, 768, 64, 64) or embedding.dtype != torch.float32:
                raise ValueError("Frozen fused feature shape/dtype differs")
            if not bool(torch.isfinite(embedding).all()):
                raise ValueError("Nonfinite frozen features")
            difference = 0.
            if compare_reference:
                # Anomalib's feature-only forward branch: no fit, optimizer, bank update, or nearest-neighbor search.
                # Set only the parent branch flag; all backbone/BN children stay in eval mode.
                try:
                    model.training = True
                    reference = model(tensor)
                    difference = float((reference-model.reshape_embedding(embedding)).abs().max().item())
                    if difference > 1e-4:
                        raise ValueError("Feature-only path differs from original fused embedding")
                finally:
                    model.training = False
                    model.embedding_store.clear()
            return embedding.mean(dim=(2, 3))[0].cpu().numpy().copy(), difference


class Monitor:
    """RSS/device-wide memory sampled at 100ms; allocator peaks checked at boundaries."""
    def __init__(self, runtime, policy, prior_seconds):
        import psutil
        self.process = psutil.Process()
        self.runtime, self.policy, self.prior_seconds = runtime, policy, prior_seconds
        self.started = time.perf_counter()
        self.rss = self.system_gpu = self.allocated = self.reserved = 0
        self.samples, self.error = 0, None
        self.sample_lock = threading.Lock()
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._loop, daemon=True)

    def _sample(self):
        with self.sample_lock:
            self.rss = max(self.rss, self.process.memory_info().rss)
            free, total = self.runtime.torch.cuda.mem_get_info()
            self.system_gpu = max(self.system_gpu, total-free)
            self.samples += 1

    def _loop(self):
        while not self.stop.wait(.1):
            try:
                self._sample()
            except Exception as exc:
                self.error = str(exc)
                return

    def __enter__(self):
        self._sample()
        self.thread.start()
        return self

    def check(self):
        self._sample()
        if self.error:
            raise RuntimeError(f"Memory monitoring failed: {self.error}")
        m = self.runtime.memory()
        self.allocated = max(self.allocated, m["torch_cuda_peak_allocated_bytes"])
        self.reserved = max(self.reserved, m["torch_cuda_peak_reserved_bytes"])
        p = self.policy["resource_plan"]
        if (self.rss > p["rss_GiB"]*1024**3 or self.reserved > p["selection_and_inference_reserved_GiB"]*1024**3
                or self.prior_seconds+time.perf_counter()-self.started > p["selection_seconds"]):
            raise RuntimeError("Selection resource budget exceeded; no automatic shrinking")

    def snapshot(self):
        return {"rss_sample_peak_bytes": self.rss, "device_used_sample_peak_bytes": self.system_gpu,
                "torch_cuda_peak_allocated_bytes": self.allocated, "torch_cuda_peak_reserved_bytes": self.reserved,
                "samples": self.samples, "sampling_interval_ms": 100,
                "scope": "sampled RSS/device used memory; allocator peaks after model load; not instantaneous system peaks"}

    def __exit__(self, *args):
        self.stop.set()
        self.thread.join(timeout=5)
        if self.thread.is_alive() and args[0] is None:
            raise RuntimeError("Memory sampler did not stop")
        if self.error and args[0] is None:
            raise RuntimeError(f"Memory monitoring failed: {self.error}")


def read_vector(root, pid, signature, rows, environment):
    receipt = fit.read_phase(root/"photos", pid, signature, ["descriptor.npy"])
    folder = root/"photos"/pid
    if (receipt["rows_sha256"] != fit.digest(rows) or receipt["environment"] != environment
            or {p.name for p in folder.iterdir()} != {"descriptor.npy", "receipt.json"}):
        raise ValueError("Cached descriptor provenance differs")
    value = np.load(folder/"descriptor.npy", allow_pickle=False)
    protocol.validate_vectors(value[None], 1)
    return value


def compute_vector(source, rows, runtime, monitor, reference=False):
    means, differences = [], []
    for row in rows:
        monitor.check()
        with fit.checked_image(source, row) as image:
            mean, difference = runtime.means(image, compare_reference=reference)
        means.append(mean)
        differences.append(difference)
    runtime.synchronize()
    monitor.check()
    return protocol.descriptor(np.stack(means)), max(differences)


def environment(runtime):
    return {"model_runtime": runtime.environment(),
            "libraries": {k: importlib.metadata.version(k) for k in ("numpy", "Pillow", "psutil")}}


def selection_payload(pool, vectors, identity, check):
    policy = identity["policy"]
    greedy = protocol.select(pool["ids"], vectors, policy["selection"]["photo_count"], check)
    ids = sorted(greedy, key=protocol.rank)
    baseline = identity["baseline_ids"]
    unchanged = set(ids) == set(baseline)
    selected_rows = [r for r in pool["rows"] if r["photo_id"] in set(ids)]
    return {"status": "selection_unchanged_stop" if unchanged else "normal_diversity_selected_ready_for_review",
            "greedy_photo_ids": greedy, "training_photo_ids": ids, "training_photo_ids_sha256": fit.digest(ids),
            "training_rows": selected_rows, "training_rows_sha256": fit.digest(selected_rows),
            "baseline_overlap_count": len(set(ids) & set(baseline)),
            "selected_descriptor_coverage": protocol.coverage(pool["ids"], vectors, ids, check),
            "baseline_descriptor_coverage": protocol.coverage(pool["ids"], vectors, baseline, check),
            "training_performed": False, "thresholds_changed": False, "model_scores_computed": 0,
            "test_images_decoded": 0, "test_masks_decoded": 0, "automatic_pass_allowed": False,
            "fit_authorized": False, "next_step": "stop_no_reroll" if unchanged else "review_selection_before_authorizing_fit"}


def read_attempts(root):
    elapsed = 0.
    for attempt in sorted(root.glob("attempt-*")):
        if not (attempt/"finished.json").exists():
            raise ValueError(f"Attempt ended without a cost receipt: {attempt}. Preserve it for review; do not reset the budget.")
        record = json.loads((attempt/"finished.json").read_bytes())
        seconds = record["seconds"]
        if not np.isfinite(seconds) or seconds < 0:
            raise ValueError("Invalid attempt elapsed time")
        elapsed += seconds
    return elapsed


def execute(root, source, checkpoint, config, records, pool, identity, handoff,
            runtime_factory=FeatureRuntime, monitor_factory=Monitor, environment_fn=environment):
    signature, policy = fit.digest(identity), identity["policy"]
    ids = pool["ids"]
    rows = {pid: pool["rows"][i*12:(i+1)*12] for i, pid in enumerate(ids)}
    if any([r["region"] for r in rs] != list(fit.REGIONS) or any(r["photo_id"] != pid for r in rs) for pid, rs in rows.items()):
        raise ValueError("Pool row order differs")
    names = Counter(source.namelist())
    if any(names[r["member"]] != 1 for r in pool["rows"]):
        raise ValueError("Missing or duplicate normal archive members")
    with fit.locked_run(root):
        if not (root/"identity").exists():
            if (root/"photos").exists() or (root/"result").exists():
                raise ValueError("Existing output has no identity")
            with fit.phase_stage(root, "identity") as stage:
                fit.write_json(stage/"plan.json", identity)
                fit.seal_phase(stage, "identity", signature, {}, ["plan.json"])
        fit.read_phase(root, "identity", signature, ["plan.json"])
        if json.loads((root/"identity/plan.json").read_bytes()) != identity:
            raise ValueError("Stored plan differs")
        (root/"photos").mkdir(exist_ok=True)
        if any(p.name not in ids and not p.name.startswith(".") for p in (root/"photos").iterdir()):
            raise ValueError("Unexpected cached normal photo; no silent reuse")
        cached_env = None
        if (root/"runtime").exists():
            fit.read_phase(root, "runtime", signature, ["environment.json"])
            cached_env = json.loads((root/"runtime/environment.json").read_bytes())
        vectors = {}
        for pid in ids:
            if (root/"photos"/pid).exists():
                if cached_env is None:
                    raise ValueError("Descriptors lack runtime provenance")
                vectors[pid] = read_vector(root, pid, signature, rows[pid], cached_env)
        result = root/"result"
        prior = read_attempts(root)
        if result.exists():
            if len(vectors) != len(ids):
                raise ValueError("Completed result lacks descriptors")
            bundle_tools.verify(result, identity)
            attempts = [json.loads(p.read_bytes()) for p in sorted(root.glob("attempt-*/finished.json"))]
            if not attempts or not any(a["outcome"] == "completed" for a in attempts):
                raise ValueError("Published result lacks successful execution cost receipt; review preserved outputs")
            x = np.stack([vectors[pid] for pid in ids])
            if not np.array_equal(np.load(result/"descriptors.npy", allow_pickle=False), x):
                raise ValueError("Published descriptors differ")
            if json.loads((result/"selection.json").read_bytes()) != selection_payload(pool, x, identity, lambda: None):
                raise ValueError("Published selection does not reproduce")
            bundle_tools.bundle(result, handoff)
            return json.loads((result/"report.json").read_bytes())
        if prior >= policy["resource_plan"]["selection_seconds"]:
            raise RuntimeError("Prior attempts exhausted selection time budget")
        attempt = Path(tempfile.mkdtemp(prefix="attempt-", dir=root))
        started, runtime, monitor, outcome, cost = time.perf_counter(), None, None, "interrupted", None
        try:
            runtime = runtime_factory(config)
            with monitor_factory(runtime, policy, prior+time.perf_counter()-started) as monitor:
                runtime.load(checkpoint, records["fit.json"]["checkpoint_context"])
                if runtime.environment() != records["reload.json"]["receipt"]["runtime"]:
                    raise ValueError("Runtime differs from frozen checkpoint reload")
                env = environment_fn(runtime)
                if cached_env is not None and env != cached_env:
                    raise ValueError("Descriptor cache runtime differs")
                if cached_env is None:
                    with fit.phase_stage(root, "runtime") as stage:
                        fit.write_json(stage/"environment.json", env)
                        fit.seal_phase(stage, "runtime", signature, {}, ["environment.json"])
                probe_ids = ids[:policy["selection"]["preflight_photo_count"]]
                probe_started, maximum_difference, reference_difference = time.perf_counter(), 0., 0.
                for pid in probe_ids:
                    a, d1 = compute_vector(source, rows[pid], runtime, monitor, reference=True)
                    b, d2 = compute_vector(source, rows[pid], runtime, monitor, reference=False)
                    delta = float(np.max(np.abs(a-b)))
                    if pid in vectors:
                        delta = max(delta, float(np.max(np.abs(a-vectors[pid]))))
                    maximum_difference = max(maximum_difference, delta)
                    reference_difference = max(reference_difference, d1, d2)
                    if delta > policy["selection"]["descriptor_replay_absolute_tolerance"]:
                        raise ValueError("Descriptor replay drift; no selection performed")
                    if pid not in vectors:
                        with fit.phase_stage(root/"photos", pid) as stage:
                            with (stage/"descriptor.npy").open("xb") as f:
                                np.save(f, a, allow_pickle=False)
                            fit.seal_phase(stage, pid, signature, {"rows_sha256": fit.digest(rows[pid]), "environment": env}, ["descriptor.npy"])
                        vectors[pid] = a
                probe_seconds = time.perf_counter()-probe_started
                # Conservative throughput estimate includes reference/replay overhead per probe photo.
                estimate = prior+time.perf_counter()-started + probe_seconds/len(probe_ids)*(len(ids)-len(vectors))
                probe = {"photo_ids": probe_ids, "maximum_descriptor_difference": maximum_difference,
                         "maximum_fused_feature_difference": reference_difference, "seconds": probe_seconds,
                         "projected_cumulative_seconds": estimate, "environment": env}
                fit.write_json(attempt/"preflight.json", probe)
                if estimate > policy["resource_plan"]["selection_seconds"]:
                    raise RuntimeError("Projected selection cost exceeds one-hour budget; no automatic reduction")
                print(f"Normal feature preflight passed; {len(vectors)}/{len(ids)} descriptors cached. No fit or anomaly scoring.", flush=True)
                for i, pid in enumerate(ids, 1):
                    if pid not in vectors:
                        with fit.phase_stage(root/"photos", pid) as stage:
                            vector, _ = compute_vector(source, rows[pid], runtime, monitor)
                            with (stage/"descriptor.npy").open("xb") as f:
                                np.save(f, vector, allow_pickle=False)
                            fit.seal_phase(stage, pid, signature, {"rows_sha256": fit.digest(rows[pid]), "environment": env}, ["descriptor.npy"])
                        vectors[pid] = read_vector(root, pid, signature, rows[pid], env)
                    if i % 25 == 0 or i == len(ids):
                        print(f"Normal photo descriptors {i}/{len(ids)}", flush=True)
                x = np.stack([vectors[pid] for pid in ids])
                selection = selection_payload(pool, x, identity, monitor.check)
                monitor.check()
                cost = monitor.snapshot()
                report = {"status": selection["status"], "candidate_photos": len(ids), "candidate_tiles": len(ids)*12,
                          "selected_photos": len(selection["training_photo_ids"]), "baseline_overlap_count": selection["baseline_overlap_count"],
                          "training_performed": False, "thresholds_changed": False, "model_scores_computed": 0,
                          "test_images_decoded": 0, "test_masks_decoded": 0, "automatic_pass_allowed": False,
                          "fit_authorized": False, "cumulative_compute_seconds": prior+time.perf_counter()-started,
                          "previous_attempt_cost_receipts": [json.loads(p.read_bytes()) for p in sorted(root.glob("attempt-*/finished.json"))],
                          "cost_scope": "previous attempts plus current runtime setup through selection; excludes metadata audit, ZIP hashing and final packaging",
                          "memory_this_attempt": cost, "environment": env,
                          "scope": "normal descriptor selection only; not training, performance evaluation, or production approval"}
                with fit.phase_stage(root, "result") as stage:
                    for name, value in {"report.json": report, "selection.json": selection, "pool.json": pool,
                                        "plan.json": identity, "preflight.json": probe}.items():
                        fit.write_json(stage/name, value)
                    with (stage/"descriptors.npy").open("xb") as f:
                        np.save(f, x, allow_pickle=False)
                    # Compact per-photo provenance, no original images or new model in the handoff.
                    fit.write_json(stage/"descriptor-receipts.json", {pid: json.loads((root/"photos"/pid/"receipt.json").read_bytes()) for pid in ids})
                    monitor.check()
                    bundle_tools.seal(stage, identity)
                outcome = "completed"
        except BaseException as exc:
            outcome = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            if monitor is not None:
                cost = monitor.snapshot()
            try:
                if runtime is not None:
                    runtime.release()
            finally:
                fit.write_json(attempt/"finished.json", {"seconds": time.perf_counter()-started, "outcome": outcome, "memory": cost})
        bundle_tools.verify(result, identity)
        handoff.parent.mkdir(parents=True, exist_ok=True)
        bundle_tools.bundle(result, handoff)
        return report


def run(project, archive):
    project, archive = Path(project).resolve(), Path(archive).resolve()
    policy, config, records, pool, identity = protocol.prepare(project, project/"artifacts/handoff")
    checkpoint = project/"artifacts/mpi-pcb-normal-bank-pilot-v1/fit/model.pt"
    print("Checking original ZIP and checkpoint hashes. Test payloads remain unopened.", flush=True)
    if (fit.sha256_file(archive) != policy["source_sha256"]["archive"]
            or fit.sha256_file(checkpoint) != policy["source_sha256"]["baseline_checkpoint"]):
        raise ValueError("Original archive or frozen checkpoint differs")
    with zipfile.ZipFile(archive) as source:
        return execute(project/"artifacts/experiments/mpi-normal-diversity-v1/prepare", source, checkpoint,
                       config, records, pool, identity, project/"artifacts/handoff"/f"{NAME}.zip")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--archive", type=Path, default=Path.home()/"Downloads/extracted.zip")
    args = parser.parse_args(argv)
    try:
        print(json.dumps(run(args.project, args.archive), ensure_ascii=False, indent=2))
        return 0
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}. Existing data/results preserved.", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
