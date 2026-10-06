"""Same official loss and public forward as formal fit; schedule only is parameterized."""
from scripts.mpi_supersimplenet_runtime import Runtime as BaseRuntime, environment
from scripts.evaluate_mpi_normal_diversity import Monitor as BaseMonitor
from scripts import mpi_supersimplenet_scaling_protocol as p


class Runtime(BaseRuntime):
    def __init__(self, weights, protocol):
        self.protocol = protocol
        self.CHECKPOINT_PURPOSE = p.NAME+"-"+protocol.arm
        super().__init__(weights)

    def step(self, images, index):
        rates = self.protocol.learning_rates(index)
        for group, rate in zip(self.optimizer.param_groups, rates, strict=True):
            group["lr"] = rate
        result = super().step(images, index)
        result.update(learning_rates=rates, epoch=index//self.protocol.POLICY["updates_per_epoch"]+1)
        return result

    def predict(self, image):
        return super().predict([image])[0]

    def load(self, path, signature, step):
        super().load(path, signature, step)
        if [g["lr"] for g in self.optimizer.param_groups] != self.protocol.learning_rates(max(0, step-1)):
            raise ValueError("Checkpoint learning rate differs from registered arm")


class Monitor(BaseMonitor):
    def check(self):
        self.sample()
        if self.runtime is not None:
            memory = self.runtime.memory()
            self.allocated = max(self.allocated, memory["torch_cuda_peak_allocated_bytes"])
            self.reserved = max(self.reserved, memory["torch_cuda_peak_reserved_bytes"])
        if self.error:
            raise RuntimeError(f"Scaling memory monitor failed: {self.error}")
        if (self.rss > self.limits["rss_GiB"]*1024**3
                or self.reserved > self.limits["selection_and_inference_reserved_GiB"]*1024**3):
            raise RuntimeError("Scaling memory limit exceeded; no automatic shrinking")
