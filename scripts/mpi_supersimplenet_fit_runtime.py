"""Formal fit uses the proven CUDA boundary with separate checkpoint purpose."""
from scripts import mpi_supersimplenet_fit_protocol as p
from scripts.mpi_supersimplenet_runtime import Runtime as ProbeRuntime, environment


class Runtime(ProbeRuntime):
    CHECKPOINT_PURPOSE = p.NAME

    def step(self, images, index):
        rates = p.learning_rates(index)
        for group, rate in zip(self.optimizer.param_groups, rates, strict=True):
            group["lr"] = rate
        value = super().step(images, index)
        value["learning_rates"] = rates
        value["epoch"] = index // 96 + 1
        return value

    def predict(self, image):
        return super().predict([image])[0]

    def load(self, path, signature, step):
        super().load(path, signature, step)
        if [g["lr"] for g in self.optimizer.param_groups] != p.learning_rates(max(0, step-1)):
            raise ValueError("Restored learning rates differ from frozen epoch schedule")
