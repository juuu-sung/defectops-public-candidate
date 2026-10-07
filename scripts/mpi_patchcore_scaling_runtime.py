"""CPU feature cache, CUDA chunked projection/greedy selection, fixed-size bank."""
from __future__ import annotations

import gc
import json
import random
import time

import numpy as np

from scripts import mpi_patchcore_scaling_protocol as p
from scripts.mpi_patchcore_scaling_probe_runtime import environment, update_chunked

fit = p.fit


class Runtime(fit.TorchRuntime):
    def __init__(self, config):
        super().__init__(config)
        t = self.torch
        if t.cuda.mem_get_info()[0] < p.POLICY["free_gpu_GiB"]*1024**3:
            raise RuntimeError("Need 6 GiB free GPU; close other GPU jobs normally")
        t.backends.cuda.matmul.allow_tf32 = False
        # Match the historical PatchCore convolution default. Projection uses matmul=False.
        t.backends.cudnn.allow_tf32 = True
        t.backends.cudnn.benchmark = False
        t.backends.cudnn.deterministic = True
        t.cuda.reset_peak_memory_stats()

    def offline_model(self, checkpoint):
        self.release()
        torch, _, constructor, _, device = self.gpu
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        prefix = "model.feature_extractor."
        weights = {k[len(prefix):]: v for k, v in saved["state_dict"].items() if k.startswith(prefix)}
        del saved
        if not weights:
            raise ValueError("Frozen baseline backbone tensors missing")
        cfg = self.config
        model = constructor(backbone=cfg["backbone"], layers=cfg["layers"], pre_trained=False,
            num_neighbors=cfg["num_neighbors"], coreset_sampling_ratio=78643/3145728,
            pre_processor=False, post_processor=False, evaluator=False, visualizer=False)
        extractor = model.model.feature_extractor
        extractor.load_state_dict(weights, strict=True)
        if any(not torch.equal(v.cpu(), weights[k]) for k, v in extractor.state_dict().items()):
            raise ValueError("Backbone changed after load")
        del weights
        self.model = model.to(device)
        self.model.model.train()
        self.model.model.feature_extractor.eval()

    def extract(self, archive, rows, check):
        torch = self.torch
        indices = torch.tensor(fit.spatial_patch_indices(64, 2), dtype=torch.long, device="cuda")
        features = []
        with torch.no_grad():
            for row in rows:
                check()
                with fit.checked_image(archive, row) as image:
                    self.model.model(fit._preprocess(image, 256, self.gpu))
                if len(self.model.model.embedding_store) != 1:
                    raise ValueError("Unexpected embedding accumulation")
                embedding = self.model.model.embedding_store.pop()
                if embedding.shape != (4096, 768) or embedding.dtype != torch.float32:
                    raise ValueError("Frozen layer/feature shape differs")
                block = embedding.index_select(0, indices).cpu().numpy().copy()
                if not np.isfinite(block).all():
                    raise ValueError("Nonfinite extracted features")
                features.append(block)
                del embedding
        result = np.concatenate(features)
        if result.shape != (12288, 768):
            raise ValueError("Expected exactly 12 tiles per original photo")
        return result

    def project(self, root, signature, raw_paths, check):
        """Model is absent during projection and selection; cache every raw interval."""
        self.release()
        from anomalib.models.components.dimensionality_reduction import SparseRandomProjection
        torch, cfg = self.torch, p.POLICY
        root.mkdir(exist_ok=True)
        n, dim = cfg["candidates"], cfg["projected_dimension"]
        init = root/"initial"
        if not init.exists():
            random.seed(42); np.random.seed(42); torch.manual_seed(42); torch.cuda.manual_seed_all(42)
            projector = SparseRandomProjection(eps=.9)
            projector.fit(torch.empty(1, 768, device="cuda").expand(n, 768))
            if projector.n_components != dim:
                raise ValueError("Projection dimension differs")
            matrix = projector.sparse_random_matrix.cpu().numpy().copy()
            start = int(torch.randint(n, (1,), device="cuda").item())
            with fit.phase_stage(root, "initial") as stage:
                np.save(stage/"matrix.npy", matrix, allow_pickle=False)
                fit.write_json(stage/"state.json", {"start_index": start, "n": n, "dim": dim, "seed": 42})
                fit.seal_phase(stage, "initial", signature, {"start_index": start, "n": n, "dim": dim},
                               ["matrix.npy", "state.json"])
            del projector, matrix
        initial_receipt = fit.read_phase(root, "initial", signature, ["matrix.npy", "state.json"])
        state = json.loads((init/"state.json").read_bytes())
        if state["n"] != n or state["dim"] != dim or state["seed"] != 42 or not 0 <= state["start_index"] < n:
            raise ValueError("Projection initialization differs")
        matrix = p.array(init/"matrix.npy", (dim, 768))
        projector = SparseRandomProjection(eps=.9)
        projector.n_components = dim
        projector.sparse_random_matrix = torch.from_numpy(np.array(matrix)).to("cuda")
        del matrix
        paths, receipts = [], [initial_receipt]
        with torch.no_grad():
            for i, (a, b) in enumerate(p.probe.slices(n, cfg["chunk_rows"])):
                check()
                name = f"chunk-{i:03d}"
                if not (root/name).exists():
                    raw = p.raw_block(raw_paths, a, b)
                    block = projector.transform(torch.from_numpy(raw).to("cuda")).cpu().numpy()
                    if block.shape != (b-a, dim) or not np.isfinite(block).all():
                        raise ValueError("Invalid projected features")
                    with fit.phase_stage(root, name) as stage:
                        np.save(stage/"features.npy", block, allow_pickle=False)
                        fit.seal_phase(stage, name, signature, {"start": a, "end": b}, ["features.npy"])
                    del block, raw
                receipt = fit.read_phase(root, name, signature, ["features.npy"])
                if receipt["start"] != a or receipt["end"] != b:
                    raise ValueError("Projection row coverage differs")
                p.array(root/name/"features.npy", (b-a, dim))
                paths.append(root/name/"features.npy"); receipts.append(receipt)
                print(f"projected candidates {b}/{n}", flush=True)
        del projector
        self.release()
        return paths, state["start_index"], receipts

    def select(self, root, signature, paths, start, check):
        """Checkpoint invariant: pending center not yet folded into all distances."""
        torch, cfg = self.torch, p.POLICY
        n, dim, maximum = cfg["candidates"], cfg["projected_dimension"], cfg["representatives"]
        root.mkdir(exist_ok=True)
        steps = p.checkpoint_steps(root, signature, n, maximum, cfg["selection_checkpoint_interval"])
        if steps and steps[-1][1]["step"] == maximum:
            return steps[-1][1]["selected"], [s[2] for s in steps]
        features = minimum = None
        try:
            features = torch.empty((n, dim), dtype=torch.float32, device="cuda")
            for path, (a, b) in zip(paths, p.probe.slices(n, cfg["chunk_rows"]), strict=True):
                check()
                block = p.array(path, (b-a, dim))
                features[a:b].copy_(torch.from_numpy(np.array(block)).to("cuda"))
                del block
            if steps:
                name, progress, receipt = steps[-1]
                cpu = p.array(root/name/"minimum.npy", (n, 1), finite=False)
                minimum = torch.from_numpy(np.array(cpu)).to("cuda")
                del cpu
                selected, center, previous = list(progress["selected"]), progress["pending_center"], receipt["receipt_sha256"]
            else:
                minimum = torch.full((n, 1), float("inf"), dtype=torch.float32, device="cuda")
                selected, center, previous = [], start, None
            seen = set(selected)

            def save():
                nonlocal previous
                name = f"step-{len(selected):06d}"
                cpu = minimum.cpu().numpy().copy()
                progress = {"step": len(selected), "selected": list(selected), "pending_center": center,
                            "previous_receipt": previous}
                p.validate_selection(progress, cpu, n, maximum)
                with fit.phase_stage(root, name) as stage:
                    np.save(stage/"minimum.npy", cpu, allow_pickle=False)
                    fit.write_json(stage/"progress.json", progress)
                    receipt = fit.seal_phase(stage, name, signature, {}, ["minimum.npy", "progress.json"])
                # Verify the persisted state before continuing selection.
                read = p.array(root/name/"minimum.npy", (n, 1), finite=False)
                if not np.array_equal(read, cpu):
                    raise ValueError("Selection checkpoint reload differs")
                previous = receipt["receipt_sha256"]

            if not steps:
                save()
            began = time.perf_counter()
            with torch.no_grad():
                while len(selected) < maximum:
                    check()
                    center = update_chunked(torch, features, minimum, center, cfg["chunk_rows"])
                    if center in seen:
                        raise ValueError("Greedy selection repeated an index; no substitute selection")
                    selected.append(center); seen.add(center)
                    if len(selected) % cfg["selection_checkpoint_interval"] == 0 or len(selected) == maximum:
                        save()
                        print(f"representatives {len(selected)}/{maximum}; this attempt {time.perf_counter()-began:.1f}s", flush=True)
            check()
        finally:
            del features, minimum
            gc.collect(); torch.cuda.empty_cache()
        steps = p.checkpoint_steps(root, signature, n, maximum, cfg["selection_checkpoint_interval"])
        return selected, [s[2] for s in steps]

    def assemble(self, baseline, raw_paths, selected, check):
        torch = self.torch
        self.offline_model(baseline)
        ids = np.asarray(selected, dtype=np.int64)
        bank = np.empty((len(ids), 768), dtype=np.float32)
        filled = np.zeros(len(ids), dtype=bool)
        for i, path in enumerate(raw_paths):
            check()
            target = np.flatnonzero((ids >= i*12288) & (ids < (i+1)*12288))
            raw = p.array(path, (12288, 768))
            bank[target] = raw[ids[target]-i*12288]
            filled[target] = True
            del raw
        if not filled.all() or not np.isfinite(bank).all() or bank.shape != (78643, 768):
            raise ValueError("Selected raw-feature bank incomplete")
        self.model.model.memory_bank = torch.from_numpy(bank).to("cuda")
        self.model._is_fitted.fill_(True)
        self.model.model.eval()

    def load(self, path, context):
        # Do not reset allocator peaks after model loading: the load belongs to this attempt.
        self.release()
        self.model = fit._model(self.recipe, path, context, self.gpu)
