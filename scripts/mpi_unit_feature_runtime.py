"""Fixed-bank unit-length feature ablation; no backbone or coreset fitting."""
from __future__ import annotations

from scripts.trace_mpi_patches import TraceRuntime


def unit_rows(values, torch):
    if len(values.shape) != 2 or values.dtype != torch.float32 or not bool(torch.isfinite(values).all()):
        raise ValueError("Expected finite float32 patch vectors")
    norms = torch.linalg.vector_norm(values, dim=1, keepdim=True)
    if not bool(torch.isfinite(norms).all()) or bool((norms <= 1e-12).any()):
        raise ValueError("Zero/invalid feature norm; no silent fallback")
    return values / norms


def transform_core(core, torch):
    """Both native query and bank distances use the same transform, including k=9."""
    if getattr(core, "_defectops_unit_features", False):
        raise ValueError("Feature transform already installed")
    original = core.reshape_embedding
    dimensions = core.memory_bank.shape[1]
    transformed = torch.empty_like(core.memory_bank)
    for start in range(0, len(transformed), 1024):
        transformed[start:start+1024] = unit_rows(core.memory_bank[start:start+1024], torch)

    def reshape(embedding):
        rows = original(embedding)
        if rows.shape[1] != dimensions:
            raise ValueError("Query/bank feature dimensions differ")
        return unit_rows(rows, torch)

    core.memory_bank = transformed
    core.reshape_embedding = reshape
    core._defectops_unit_features = True


class UnitRuntime(TraceRuntime):
    def __init__(self, config):
        self.retained_allocated = self.retained_reserved = 0
        super().__init__(config)

    def reset_peak(self):
        # Native load() resets peaks; retain load and earlier-stage peaks for the cap.
        self.synchronize()
        self.retained_allocated = max(self.retained_allocated, self.torch.cuda.max_memory_allocated())
        self.retained_reserved = max(self.retained_reserved, self.torch.cuda.max_memory_reserved())
        super().reset_peak()

    def memory(self):
        result = super().memory()
        result["torch_cuda_peak_allocated_bytes"] = max(result["torch_cuda_peak_allocated_bytes"], self.retained_allocated)
        result["torch_cuda_peak_reserved_bytes"] = max(result["torch_cuda_peak_reserved_bytes"], self.retained_reserved)
        return result

    def activate(self):
        if self.model.model.training or self.model.model.feature_extractor.training:
            raise ValueError("Feature metric requires frozen eval-mode model")
        before = self.bank_fingerprint()
        with self.torch.no_grad():
            transform_core(self.model.model, self.torch)
        return {"original_bank": before, "unit_bank": self.bank_fingerprint(),
                "operation": "row_l2_unit_length_float32", "epsilon_rejection": 1e-12,
                "coreset_reselected": False, "backbone_trained": False}
