"""Use native Anomalib inference, temporarily selecting original bank rows."""
from contextlib import contextmanager

from scripts.trace_mpi_patches import TraceRuntime


@contextmanager
def selected_bank(core, indices, torch):
    original = core.memory_bank
    try:
        ids = torch.tensor(indices, dtype=torch.long, device=original.device)
        core.memory_bank = original.index_select(0, ids)
        yield
    finally:
        core.memory_bank = original


class RegionRuntime(TraceRuntime):
    def all_bank_hashes(self):
        count = int(self.model.model.memory_bank.shape[0])
        hashes = []
        for start in range(0, count, 1024):
            hashes.extend(self.bank_hashes(list(range(start, min(count,start+1024)))))
        return hashes

    def predict_region(self, image, indices):
        if len(indices) < self.config['num_neighbors']:
            raise ValueError('Region bank too small for native weighted score')
        with selected_bank(self.model.model, indices, self.torch):
            return self.predict(image)
