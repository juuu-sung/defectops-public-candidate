"""Official head reconstruction only; the previous varying runner stays immutable."""
from scripts import mpi_supersimplenet_head_reset_protocol as p
from scripts.mpi_supersimplenet_varying_runtime import Runtime as VaryingRuntime, environment
from scripts.mpi_supersimplenet_matched_eval_runtime import Runtime as MatchedRuntime


class Runtime(VaryingRuntime):
    def evaluate_seed(self, images, seed):
        if seed not in p.POLICY["evaluation_seeds"]:
            raise ValueError("Unregistered evaluation seed")
        self.optimizer.zero_grad(set_to_none=True)
        self.model.eval()
        flags = [(t, t.requires_grad) for t in self.model.parameters()]
        self.model.requires_grad_(False)
        try:
            return MatchedRuntime.trial(self, images, seed)
        finally:
            for tensor, flag in flags:
                tensor.requires_grad_(flag)

    def reset_head(self):
        if getattr(self, "control_only", False) or getattr(self, "reset_applied", False):
            raise ValueError("Read-only or already initialized; head reset forbidden")
        from anomalib.models.image.supersimplenet.torch_model import SegmentationDetectionModule
        torch, model = self.torch, self.model
        before = self.fingerprints()
        # Fresh constructor also resets bias and BN buffers, unlike apply(init_weights).
        # fork_rng prevents initialization from changing the training/evaluation draws.
        with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))), torch.device("cpu"):
            torch.manual_seed(p.POLICY["head_initialization_seed"])
            head = SegmentationDetectionModule(channel_dim=1536, stop_grad=True)
        expected = model.segdec.state_dict()
        actual = head.state_dict()
        if set(expected) != set(actual) or any(expected[k].shape != actual[k].shape for k in expected):
            raise ValueError("Official fresh head architecture differs")
        bn = [m for m in head.modules() if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)]
        fresh = all(bool((m.running_mean == 0).all() and (m.running_var == 1).all()
                         and (m.weight == 1).all() and (m.bias == 0).all()
                         and m.num_batches_tracked == 0) for m in bn)
        del self.optimizer
        model.segdec = head.to(device="cuda", dtype=torch.float32).requires_grad_(True)
        self.optimizer = self.new_optimizer()
        model.eval()
        record = {"seed": p.POLICY["head_initialization_seed"], "before": before,
                  "after": self.fingerprints(), "batchnorm_modules": len(bn),
                  "batchnorm_fresh": fresh, "optimizer_empty": not bool(self.optimizer.state)}
        p.validate_initialization(record, before)
        self.reset_applied = True
        return record

    def step(self, images, index):
        if getattr(self, "control_only", False):
            raise ValueError("Baseline is read-only; optimizer updates forbidden")
        if not getattr(self, "reset_applied", False):
            raise ValueError("Candidate head must be initialized or restored before training")
        return super().step(images, index)

    def load_baseline(self, path, signature):
        VaryingRuntime.load(self, path, signature, 500)
        self.control_only = True

    def save(self, path, signature, step):
        if not getattr(self, "reset_applied", False) or getattr(self, "control_only", False):
            raise ValueError("Only the initialized candidate can be saved")
        self.optimizer.zero_grad(set_to_none=True)
        self.synchronize()
        self.torch.save({"purpose": p.NAME, "identity": signature, "step": step,
                        "initialization_seed": p.POLICY["head_initialization_seed"],
                        "model": self.model.state_dict(), "optimizer": self.optimizer.state_dict()}, path)

    def load(self, path, signature, step):
        payload = self.torch.load(path, map_location="cpu", weights_only=True)
        if (payload["purpose"] != p.NAME or payload["identity"] != signature or payload["step"] != step
                or payload["initialization_seed"] != p.POLICY["head_initialization_seed"]):
            raise ValueError("Head-reset checkpoint identity differs")
        if getattr(self, "control_only", False):
            raise ValueError("Cannot replace read-only baseline")
        del self.optimizer
        self.torch.cuda.empty_cache()
        self.model.load_state_dict(payload["model"], strict=True)
        self.optimizer = self.new_optimizer()
        self.optimizer.load_state_dict(payload["optimizer"])
        params = [t for g in self.optimizer.param_groups for t in g["params"]]
        if (step == 0 and self.optimizer.state) or (step and (
                len(self.optimizer.state) != len(params)
                or any(int(s["step"].item()) != step for s in self.optimizer.state.values()))):
            raise ValueError("Optimizer restoration differs")
        self.reset_applied = True
        self.model.eval()
