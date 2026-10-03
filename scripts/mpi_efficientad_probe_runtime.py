"""Real CUDA boundary for a 100-update normal-only EfficientAD-S resource probe."""
from __future__ import annotations

import hashlib
import importlib.metadata
import platform
import random
from pathlib import Path

import numpy as np
from PIL import Image


def environment():
    import torch
    import torchvision
    import psutil  # noqa: F401 -- fail before downloads if monitoring is unavailable
    versions = {"torch": torch.__version__, "torchvision": torchvision.__version__,
                "anomalib": importlib.metadata.version("anomalib")}
    if versions != {"torch": "2.8.0+cu128", "torchvision": "0.23.0+cu128", "anomalib": "2.2.0"}:
        raise ValueError(f"Expected reviewed Windows GPU versions, found {versions}")
    if platform.system() != "Windows" or platform.python_version_tuple()[:2] != ("3", "12"):
        raise ValueError("This probe requires the reviewed Windows Python 3.12 environment")
    if not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable; no CPU fallback")
    name = torch.cuda.get_device_name(0)
    if "RTX 5060" not in name or torch.cuda.get_device_properties(0).total_memory > 9 * 1024**3:
        raise ValueError("Expected RTX 5060 8GB")
    from anomalib.models.image.efficient_ad import torch_model
    return {**versions, "python": platform.python_version(), "gpu": name,
            "pillow": importlib.metadata.version("Pillow"), "numpy": np.__version__,
            "efficientad_source_sha256": hashlib.sha256(Path(torch_model.__file__).read_bytes()).hexdigest()}


class Runtime:
    def __init__(self, teacher, aux):
        import torch
        from torchvision import transforms
        from anomalib.models.image.efficient_ad.torch_model import EfficientAdModel, EfficientAdModelSize
        self.torch = torch
        if torch.cuda.mem_get_info()[0] < 6 * 1024**3:
            raise RuntimeError("Less than 6 GiB GPU memory free; close other GPU jobs normally")
        self.aux = aux
        self.Model, self.Size = EfficientAdModel, EfficientAdModelSize
        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.cuda.reset_peak_memory_stats()
        self.model = self.new_model()
        self.model.teacher.load_state_dict(torch.load(teacher, map_location="cpu", weights_only=True), strict=True)
        self.optimizer = self.new_optimizer()
        self.to_tensor = transforms.ToTensor()
        self.aux_transform = transforms.Compose([transforms.Resize((512, 512)),
            transforms.RandomGrayscale(p=.3), transforms.CenterCrop((256, 256)), transforms.ToTensor()])

    def new_model(self):
        model = self.Model(teacher_out_channels=384, model_size=self.Size.S,
                           padding=False, pad_maps=True).to("cuda")
        for module in (model.teacher, model.mean_std, model.quantiles):
            module.requires_grad_(False)
        return model

    def new_optimizer(self):
        return self.torch.optim.Adam(list(self.model.student.parameters()) + list(self.model.ae.parameters()),
                                     lr=1e-4, weight_decay=1e-5)

    def tensor(self, image):
        if image.mode != "RGB" or image.size != (256, 256):
            raise ValueError("Expected original RGB 256x256 tile")
        return self.to_tensor(image).unsqueeze(0).to("cuda")

    def statistics(self, images, check):
        torch = self.torch
        total = square = None
        count = 0
        self.model.eval()
        with torch.no_grad():
            for image in images:
                check()
                try:
                    y = self.model.teacher(self.tensor(image)).double()
                finally:
                    image.close()
                sums, squares = y.sum((0, 2, 3)), y.square().sum((0, 2, 3))
                total = sums if total is None else total + sums
                square = squares if square is None else square + squares
                count += y.shape[0] * y.shape[2] * y.shape[3]
                check()
            mean = total / count
            variance = square / count - mean.square()
            if not torch.isfinite(variance).all() or (variance <= 1e-12).any():
                raise ValueError("Invalid teacher channel variance; no silent epsilon correction")
            self.model.mean_std["mean"].copy_(mean.float()[None, :, None, None])
            self.model.mean_std["std"].copy_(variance.sqrt().float()[None, :, None, None])
        return {"teacher_channels": 384, "spatial_observations_per_channel": count,
                "statistics_scope": "96 probe train tiles only; not full-fit statistics"}

    def step(self, image, index):
        torch = self.torch
        self.model.train()
        self.model.teacher.eval()
        x = self.tensor(image)
        with Image.open(self.aux / f"aux-{index % 96:03d}.jpg") as aux:
            background = self.aux_transform(aux.convert("RGB")).unsqueeze(0).to("cuda")
        self.optimizer.zero_grad(set_to_none=True)
        components = self.model(x, background)
        if len(components) != 3 or not all(torch.isfinite(v).all() for v in components):
            raise ValueError("Invalid EfficientAD loss components")
        loss = sum(components)
        loss.backward()
        parameters = [p for g in self.optimizer.param_groups for p in g["params"]]
        if any(p.grad is None or not torch.isfinite(p.grad).all() for p in parameters):
            raise ValueError("Missing or nonfinite student/autoencoder gradient")
        grad_square = sum(float(p.grad.detach().double().square().sum()) for p in parameters)
        if not np.isfinite(grad_square) or grad_square <= 0:
            raise ValueError("Invalid gradient norm")
        self.optimizer.step()
        if any(not torch.isfinite(p).all() for p in parameters):
            raise ValueError("Nonfinite updated parameter")
        torch.cuda.synchronize()
        return {"step": index + 1, "loss_components": [float(v.detach()) for v in components],
                "gradient_l2": grad_square**.5}

    def predict(self, image):
        self.model.eval()
        with self.torch.no_grad():
            result = self.model(self.tensor(image), normalize=False).anomaly_map
        values = result.detach().cpu().numpy()
        if values.shape != (1, 1, 256, 256) or not np.isfinite(values).all() or (values < 0).any():
            raise ValueError("Invalid raw EfficientAD probe map")
        return values

    def save(self, path, signature, step):
        torch = self.torch
        torch.cuda.synchronize()
        torch.save({"purpose": "resource_probe_not_a_service_model", "identity": signature, "step": step,
                    "model": self.model.state_dict(), "optimizer": self.optimizer.state_dict(),
                    "cpu_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all(),
                    "python_rng": random.getstate()}, path)

    def load(self, path, signature, step):
        torch = self.torch
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if (payload["purpose"] != "resource_probe_not_a_service_model" or
                payload["identity"] != signature or payload["step"] != step):
            raise ValueError("Probe checkpoint identity/step mismatch")
        # Release the old optimizer/model before creating replacements on the 8GB GPU.
        del self.optimizer, self.model
        torch.cuda.empty_cache()
        self.model = self.new_model()
        self.model.load_state_dict(payload["model"], strict=True)
        self.optimizer = self.new_optimizer()
        self.optimizer.load_state_dict(payload["optimizer"])
        torch.set_rng_state(payload["cpu_rng"])
        torch.cuda.set_rng_state_all(payload["cuda_rng"])
        random.setstate(payload["python_rng"])
        if (not torch.equal(torch.get_rng_state(), payload["cpu_rng"])
                or any(not torch.equal(x, y) for x, y in zip(torch.cuda.get_rng_state_all(), payload["cuda_rng"], strict=True))
                or random.getstate() != payload["python_rng"]):
            raise ValueError("Checkpoint RNG state was not restored")
        if step and any(int(s["step"].item()) != step for s in self.optimizer.state.values()):
            raise ValueError("Optimizer step was not restored")

    def memory(self):
        return {"torch_cuda_peak_allocated_bytes": self.torch.cuda.max_memory_allocated(),
                "torch_cuda_peak_reserved_bytes": self.torch.cuda.max_memory_reserved()}
