"""CUDA synthetic probe: keep all projected candidates, chunk distance workspace."""
from __future__ import annotations

import gc
import hashlib
import importlib.metadata
import platform
import random
import time
from pathlib import Path

import numpy as np

from scripts import mpi_patchcore_scaling_probe as p


def environment():
    import torch
    import torchvision
    from anomalib.models.components.sampling import k_center_greedy
    from anomalib.models.components.dimensionality_reduction import random_projection
    found = {"torch": torch.__version__, "torchvision": torchvision.__version__,
             "anomalib": importlib.metadata.version("anomalib")}
    if found != {"torch": "2.8.0+cu128", "torchvision": "0.23.0+cu128", "anomalib": "2.2.0"}:
        raise ValueError(f"Expected frozen Windows GPU versions: {found}")
    if platform.system() != "Windows" or platform.python_version_tuple()[:2] != ("3", "12"):
        raise ValueError("Requires lab Windows Python 3.12; no CPU fallback")
    if not torch.cuda.is_available() or "RTX 5060" not in torch.cuda.get_device_name(0):
        raise ValueError("Expected lab RTX 5060 CUDA")
    if not 7*1024**3 < torch.cuda.get_device_properties(0).total_memory < 9*1024**3:
        raise ValueError("Expected the reviewed 8 GiB GPU")
    return {**found, "python": platform.python_version(), "numpy": np.__version__,
            "gpu": torch.cuda.get_device_name(0), "cuda": torch.version.cuda,
            "gpu_total_bytes": torch.cuda.get_device_properties(0).total_memory,
            "reference_sources": {m.__name__: hashlib.sha256(Path(m.__file__).read_bytes()).hexdigest()
                                  for m in (k_center_greedy, random_projection)}}


def update_chunked(torch, features, minimum, center_index, chunk, check=lambda: None):
    """Same rowwise L2 recurrence as KCenterGreedy; no candidate subsampling."""
    center = features[center_index]
    for start, end in p.slices(features.shape[0], chunk):
        check()
        distances = torch.linalg.norm(features[start:end]-center, ord=2, dim=1, keepdim=True)
        torch.minimum(minimum[start:end], distances, out=minimum[start:end])
        del distances
    # torch.max returns the first maximum, matching the reference's tie handling.
    index = torch.max(minimum.squeeze(1), dim=0).indices
    minimum[index, 0] = 0.
    return int(index.item())


class Runtime:
    def __init__(self):
        import torch
        self.torch = torch
        if torch.cuda.mem_get_info(0)[0] < p.POLICY["free_gpu_GiB"]*1024**3:
            raise RuntimeError("Need 6 GiB free GPU. Close other GPU jobs normally")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.cuda.reset_peak_memory_stats()

    def memory(self):
        return {"torch_cuda_peak_allocated_bytes": self.torch.cuda.max_memory_allocated(),
                "torch_cuda_peak_reserved_bytes": self.torch.cuda.max_memory_reserved()}

    def parity(self, check):
        torch = self.torch
        from anomalib.models.components.sampling.k_center_greedy import KCenterGreedy
        results = []
        with torch.no_grad():
            for seed in p.POLICY["parity_seeds"]:
                random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
                raw = torch.randn(p.POLICY["parity_rows"], 768, device="cuda")
                sampler = KCenterGreedy(raw, sampling_ratio=64/2048)
                sampler.model.fit(raw)
                features = sampler.model.transform(raw)
                if not torch.isfinite(features).all():
                    raise ValueError("Nonfinite reference projection")
                start = int(torch.randint(features.shape[0], (1,), device="cuda").item())
                sampler.features = features
                sampler.reset_distances()
                index, official, states = start, [], []
                for _ in range(p.POLICY["parity_selections"]):
                    check()
                    sampler.update_distances(index)
                    index = int(torch.as_tensor(sampler.get_new_idx()).item())
                    sampler.min_distances[index, 0] = 0.
                    official.append(index)
                    states.append(sampler.min_distances.clone())
                for chunk in p.POLICY["parity_chunks"]:
                    projected = torch.cat([sampler.model.transform(raw[a:b])
                                           for a, b in p.slices(len(raw), chunk)])
                    if not torch.isfinite(projected).all():
                        raise ValueError("Nonfinite chunked projection")
                    error = float((projected-features).abs().max().item())
                    if not np.isfinite(error) or error > p.POLICY["projection_absolute_tolerance"]:
                        raise ValueError("Chunked projection differs; do not relax tolerance")
                    # Check the recurrence independently from projection GEMM differences.
                    minimum = torch.full((len(features), 1), float("inf"), device="cuda")
                    idx, actual, distance_error = start, [], 0.
                    for i in range(len(official)):
                        idx = update_chunked(torch, features, minimum, idx, chunk, check)
                        actual.append(idx)
                        distance_error = max(distance_error, float((minimum-states[i]).abs().max().item()))
                    p.validate_indices(official, actual)
                    if distance_error > p.POLICY["distance_absolute_tolerance"]:
                        raise ValueError("Chunked minimum-distance state differs")
                    # End-to-end projected blocks must choose the same representatives too.
                    minimum.fill_(float("inf")); idx, end_to_end = start, []
                    for _ in official:
                        idx = update_chunked(torch, projected, minimum, idx, chunk, check)
                        end_to_end.append(idx)
                    p.validate_indices(official, end_to_end)
                    results.append({"seed": seed, "chunk_rows": chunk, "start_index": start,
                                    "selected_indices": actual, "exact_indices_match": True,
                                    "end_to_end_indices_match": True,
                                    "projection_max_absolute_difference": error,
                                    "distance_max_absolute_difference": distance_error})
                    del projected, minimum
                del sampler, raw, features, states
        self.release()
        return {"passed": True, "rows": 2048, "selections": 64, "cases": results,
                "scope": "installed official projection/update/get_new_idx on identical synthetic inputs; not a full fit equivalence proof"}

    def scale(self, check):
        torch = self.torch
        from anomalib.models.components.dimensionality_reduction import SparseRandomProjection
        b = p.budget(); n, dim, chunk = b["candidates"], b["projected_dimension"], p.POLICY["chunk_rows"]
        random.seed(42); np.random.seed(42); torch.manual_seed(42); torch.cuda.manual_seed_all(42)
        generator = np.random.default_rng(42)
        started = time.perf_counter()
        with torch.no_grad():
            # fit reads shape/device only. Expanded view has no full raw allocation.
            projector = SparseRandomProjection(eps=.9)
            projector.fit(torch.empty(1, 768, device="cuda").expand(n, 768))
            if projector.n_components != dim:
                raise ValueError("Installed projection dimension differs from locked budget")
            projection_hash = hashlib.sha256(projector.sparse_random_matrix.cpu().numpy().tobytes()).hexdigest()
            features = torch.empty(n, dim, dtype=torch.float32, device="cuda")
            for a, z in p.slices(n, chunk):
                check()
                cpu = generator.standard_normal((z-a, 768), dtype=np.float32)
                raw = torch.from_numpy(cpu).to("cuda")
                features[a:z].copy_(projector.transform(raw))
                del cpu, raw
            del projector
            torch.cuda.synchronize(); check()
            projection_seconds = time.perf_counter()-started
            minimum = torch.full((n, 1), float("inf"), dtype=torch.float32, device="cuda")
            idx = int(torch.randint(n, (1,), device="cuda").item())
            initial, chosen, measured = idx, [], []
            warmup = p.POLICY["full_size_warmup_selections"]
            total = warmup+p.POLICY["full_size_measured_selections"]
            for i in range(total):
                torch.cuda.synchronize(); before = time.perf_counter()
                idx = update_chunked(torch, features, minimum, idx, chunk, check)
                torch.cuda.synchronize(); elapsed = time.perf_counter()-before
                chosen.append(idx)
                if i >= warmup:
                    measured.append(elapsed)
                if (i+1) % 10 == 0 or i+1 == total:
                    print(f"Synthetic full-candidate selection {i+1}/{total}; no product images", flush=True)
            check()
            if not torch.isfinite(minimum).all() or float(minimum.min()) < 0 or len(set(chosen)) != len(chosen):
                raise ValueError("Invalid synthetic selection state")
            result = {"passed": True, "budget": b, "initial_index": initial,
                      "projection_sha256": projection_hash, "selected_indices": chosen,
                      "synthetic_projection_seconds": projection_seconds,
                      "measured_iteration_seconds": measured,
                      "iteration_mean_seconds": float(np.mean(measured)),
                      "iteration_p95_seconds": float(np.quantile(measured, .95, method="higher")),
                      "selection_only_linear_estimate_seconds": float(np.mean(measured)*b["representatives"]),
                      "estimate_scope": "Synthetic 100-iteration extrapolation; not a time cap or full-fit ETA. Excludes real extraction, disk cache, all remaining selections, calibration, reload and evaluation.",
                      "all_candidates_kept": True, "chunk_rows": chunk,
                      "full_training_executed": False}
            del minimum, features
        self.release()
        return result

    def release(self):
        gc.collect()
        self.torch.cuda.empty_cache()
