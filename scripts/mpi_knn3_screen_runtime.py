"""Raw-feature mean-of-three patch map, preserving native nearest-one replay."""
from contextlib import contextmanager

from scripts.mpi_unit_feature_runtime import UnitRuntime

BANK_SHA = "a7e6e28fbf6b01395f4f767edc41723407746a31da5e08e5809cbd842dea0615"


def dual_distances(distances):
    # min preserves the native lowest-index tie rule. topk's first index may not.
    nearest = distances.min(dim=1)
    mean_three = distances.topk(k=3, largest=False, dim=1).values.mean(dim=1)
    return nearest, mean_three


@contextmanager
def capture_mean_three(core, captured):
    original = core.nearest_neighbors
    had_override = "nearest_neighbors" in vars(core)
    override = vars(core).get("nearest_neighbors")

    def search(embedding, n_neighbors):
        if n_neighbors != 1:
            return original(embedding, n_neighbors)
        if tuple(embedding.shape) != (4096, 768):
            raise ValueError("Unexpected nearest-one query grid")
        nearest, mean_three = dual_distances(core.euclidean_dist(embedding, core.memory_bank))
        captured.append(mean_three)
        return nearest

    core.nearest_neighbors = search
    try:
        yield
    finally:
        if had_override:
            core.nearest_neighbors = override
        else:
            del core.nearest_neighbors


class MeanThreeRuntime(UnitRuntime):
    """Inherit load-peak accounting only; never call the unit-length activation."""

    def activate(self):
        core = self.model.model
        if core.training or core.feature_extractor.training or core.tiler is not None:
            raise ValueError("Requires frozen eval mode without tiling")
        before = self.bank_fingerprint()
        if before["float32_le_sha256"] != BANK_SHA:
            raise ValueError("Original raw-feature bank changed")
        return {"original_bank": before, "operation": "mean_of_three_raw_euclidean_patch_distances",
                "query_and_bank_normalized": False, "coreset_reselected": False,
                "native_image_score": "unchanged nearest1 and weighted9, auxiliary only"}

    def predict_pair(self, image):
        captured = []
        core = self.model.model
        with self.torch.no_grad(), capture_mean_three(core, captured):
            raw, baseline_map = super().predict(image)
            if len(captured) != 1:
                raise ValueError("Expected exactly one patch-distance call")
            candidate = core.anomaly_map_generator(captured[0].reshape(1, 1, 64, 64), (256, 256))
            candidate = candidate.detach().cpu().numpy().reshape(256, 256)
        return raw, baseline_map, candidate
