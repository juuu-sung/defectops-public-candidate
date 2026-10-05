"""Official generator with bounded amplitude changes; all other learning stays fixed."""
from contextlib import contextmanager

from scripts import mpi_supersimplenet_noise_protocol as p
from scripts.mpi_supersimplenet_varying_runtime import Runtime as VaryingRuntime, environment


@contextmanager
def noise_level(generator, std):
    if std not in p.STRENGTHS.values() or generator.noise_mean != 0 or generator.threshold != .6:
        raise ValueError("Unregistered noise/Perlin configuration")
    old = generator.noise_std
    if old not in p.STRENGTHS.values():
        raise ValueError("Unexpected existing noise amplitude")
    generator.noise_std = std
    try:
        yield
    finally:
        generator.noise_std = old


class Runtime(VaryingRuntime):
    def step_at_strength(self, images, index, arm):
        if arm not in p.ARMS:
            raise ValueError("Baseline control cannot be trained")
        with noise_level(self.model.anomaly_generator, p.STRENGTHS[arm]):
            result = super().step(images, index)
        return {**result, "noise_std": p.STRENGTHS[arm]}

    def evaluate_strength(self, images, seed, strength):
        if strength not in p.STRENGTHS:
            raise ValueError("Unregistered evaluation amplitude")
        stats = {}
        def measure(module, args, out):
            clean, mixed, masks = args[0], out[0], out[1]
            noise_sq, clean_sq, count, outside = 0., 0., 0, 0.
            # No batch-sized double copy; detached reductions cannot affect learning.
            for i in range(len(mixed)):
                c = clean[i % len(clean)].detach()
                delta = mixed[i].detach() - c
                inside = masks[i, 0].bool()
                if inside.any():
                    noise_sq += float(delta[:, inside].double().square().sum())
                    clean_sq += float(c[:, inside].double().square().sum())
                    count += int(inside.sum()) * c.shape[0]
                if (~inside).any():
                    outside = max(outside, float(delta[:, ~inside].abs().max()))
            rms = (noise_sq/count)**.5 if count else None
            std = p.STRENGTHS[strength]
            if outside != 0 or (count and not .95*std <= rms <= 1.05*std):
                raise ValueError("Realized noise differs from requested amplitude")
            stats.update(masked_values=count, noise_rms=rms,
                matched_feature_rms=(clean_sq/count)**.5 if count else None,
                noise_to_feature_rms=(noise_sq/clean_sq)**.5 if clean_sq else None,
                outside_mask_max_abs=outside)
        with noise_level(self.model.anomaly_generator, p.STRENGTHS[strength]):
            handle = self.model.anomaly_generator.register_forward_hook(measure)
            try:
                arrays, checks = super().evaluate_seed(images, seed)
            finally:
                handle.remove()
        if not stats:
            raise ValueError("No official generator invocation measured")
        return arrays, {**checks, "noise_std": p.STRENGTHS[strength], "realized_noise": stats}

    def save_arm(self, path, signature, step, arm):
        self.optimizer.zero_grad(set_to_none=True)
        self.synchronize()
        self.torch.save({"purpose": p.NAME, "identity": signature, "step": step, "arm": arm,
            "training_noise_std": p.STRENGTHS[arm], "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict()}, path)

    def load_arm(self, path, signature, step, arm):
        payload = self.torch.load(path, map_location="cpu", weights_only=True)
        if (payload["purpose"] != p.NAME or payload["identity"] != signature or payload["step"] != step
                or payload["arm"] != arm or payload["training_noise_std"] != p.STRENGTHS[arm]):
            raise ValueError("Noise-arm checkpoint identity differs")
        del self.optimizer
        self.torch.cuda.empty_cache()
        self.model.load_state_dict(payload["model"], strict=True)
        self.optimizer = self.new_optimizer()
        self.optimizer.load_state_dict(payload["optimizer"])
        params = [t for g in self.optimizer.param_groups for t in g["params"]]
        if step and (len(self.optimizer.state) != len(params) or
                     any(int(s["step"].item()) != step for s in self.optimizer.state.values())):
            raise ValueError("Optimizer restoration differs")
        self.model.eval()

    def load_baseline(self, path, signature):
        super().load(path, signature, 500)
        self.control_only = True
