"""CUDA boundary for the pinned SuperSimpleNet normal-only probe."""
from __future__ import annotations

import hashlib
import importlib.metadata
import random
from pathlib import Path
from unittest.mock import patch

import numpy as np

from scripts import mpi_supersimplenet_protocol as p
from scripts.mpi_efficientad_probe_runtime import environment as base_environment


def environment():
    result = base_environment()
    import anomalib
    versions = {name: importlib.metadata.version(name) for name in ("timm", "safetensors", "psutil")}
    if versions["timm"] != "1.0.20":
        raise ValueError("Expected timm 1.0.20; do not automatically upgrade the environment")
    result.pop("efficientad_source_sha256")
    result.update(versions)
    result["supersimplenet_sources"] = p.verify_sources(Path(anomalib.__file__).parent)
    return result


def state_digest(module):
    """Hash tensor contents, not a nondeterministic torch.save container."""
    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        array = tensor.detach().cpu().contiguous().numpy()
        digest.update(name.encode())
        digest.update(str((array.dtype.str, array.shape)).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


class Runtime:
    def __init__(self, weights):
        import torch
        from torchvision import transforms
        from anomalib.models.image.supersimplenet.loss import SSNLoss
        self.torch, self.weights = torch, weights
        if torch.cuda.mem_get_info()[0] < 6 * 1024**3:
            raise RuntimeError("Less than 6 GiB GPU free; close other GPU jobs normally")
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
        self.loss = SSNLoss()
        self.optimizer = self.new_optimizer()
        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(
            mean=[.485, .456, .406], std=[.229, .224, .225])])

    def new_model(self):
        from safetensors.torch import load_file
        from anomalib.models.components import TimmFeatureExtractor
        from anomalib.models.image.supersimplenet import torch_model

        def offline_extractor(*args, **kwargs):
            # Only initialization is adapted: no implicit HF 'main' downloads on reload.
            return TimmFeatureExtractor(*args, **kwargs, pre_trained=False)

        with patch.object(torch_model, "TimmFeatureExtractor", offline_extractor):
            model = torch_model.SupersimplenetModel(perlin_threshold=.6,
                backbone=p.POLICY["backbone"], layers=["layer2", "layer3"], stop_grad=True)
        extractor = model.feature_extractor.feature_extractor.feature_extractor
        state = load_file(str(self.weights), device="cpu")
        expected = extractor.state_dict()
        missing = set(expected) - set(state)
        unused = set(state) - set(expected)
        if missing or any(not key.startswith(("layer4.", "fc.")) for key in unused):
            raise ValueError(f"Pinned backbone state mismatch: missing={sorted(missing)}, unused={sorted(unused)}")
        extractor.load_state_dict({key: state[key] for key in expected}, strict=True)
        model.feature_extractor.requires_grad_(False)
        model.feature_extractor.eval()
        return model.to("cuda")

    def new_optimizer(self):
        return self.torch.optim.AdamW([
            {"params": self.model.adaptor.parameters(), "lr": .0001, "weight_decay": .01},
            {"params": self.model.segdec.parameters(), "lr": .0002, "weight_decay": .00001},
        ])

    def tensor(self, images):
        if not images or any(im.mode != "RGB" or im.size != (256, 256) for im in images):
            raise ValueError("Expected original RGB 256x256 normal tiles")
        return self.torch.stack([self.transform(im) for im in images]).to("cuda")

    def step(self, images, index):
        torch = self.torch
        if len(images) != 8:
            raise ValueError("Batch8 required; no automatic shrinking")
        self.model.train()
        self.model.feature_extractor.eval()
        x = self.tensor(images)
        # Source normals only. These zeros are not read from any ground-truth file.
        masks = torch.zeros((8, 256, 256), device="cuda")
        labels = torch.zeros(8, device="cuda")
        self.optimizer.zero_grad(set_to_none=True)
        maps, scores, synthetic_masks, synthetic_labels = self.model(x, masks=masks, labels=labels)
        if (maps.shape != (16, 1, 64, 64) or scores.shape != (16,)
                or synthetic_masks.shape != maps.shape or synthetic_labels.shape != scores.shape
                or not all(torch.isfinite(t).all() for t in (maps, scores, synthetic_masks, synthetic_labels))
                or not ((synthetic_masks == 0) | (synthetic_masks == 1)).all()
                or not torch.equal(synthetic_labels.bool(), synthetic_masks.flatten(1).any(1))):
            raise ValueError("Unexpected effective batch16 synthetic outputs")
        loss = self.loss(pred_map=maps, pred_score=scores,
                         target_mask=synthetic_masks, target_label=synthetic_labels)
        if loss.ndim != 0 or not torch.isfinite(loss):
            raise ValueError("Nonfinite/scalar loss required")
        loss.backward()
        gradient_norms = {}
        for name, module in (("adaptor", self.model.adaptor), ("segmentation", self.model.segdec.seg_head),
                             ("classification_conv", self.model.segdec.cls_conv),
                             ("classification_fc", self.model.segdec.cls_fc)):
            parameters = list(module.parameters())
            if any(t.grad is None or not torch.isfinite(t.grad).all() for t in parameters):
                raise ValueError(f"Missing/nonfinite {name} gradients")
            norm = sum(float(t.grad.detach().double().square().sum()) for t in parameters)**.5
            if not np.isfinite(norm) or norm <= 0:
                raise ValueError(f"Zero/nonfinite {name} gradient")
            gradient_norms[name] = norm
        if any(t.grad is not None for t in self.model.feature_extractor.parameters()):
            raise ValueError("Frozen backbone received gradients")
        self.optimizer.step()
        if any(not torch.isfinite(t).all() for g in self.optimizer.param_groups for t in g["params"]):
            raise ValueError("Nonfinite updated weights")
        torch.cuda.synchronize()
        return {"step": index+1, "loss": float(loss.detach()), "gradient_l2": gradient_norms,
                "input_batch": 8, "effective_feature_batch": 16,
                "synthetic_anomalous_samples": int(synthetic_labels.sum()),
                "synthetic_anomalous_fraction": float(synthetic_masks.mean())}

    def predict(self, images):
        self.model.eval()
        with self.torch.no_grad():
            result = self.model(self.tensor(images))
        return p.validate_prediction(result.pred_score.detach().cpu().numpy(),
                                     result.anomaly_map.detach().cpu().numpy(), len(images))

    def fingerprints(self):
        return {"backbone": state_digest(self.model.feature_extractor),
                "adaptor": state_digest(self.model.adaptor),
                "segmentation": state_digest(self.model.segdec.seg_head),
                "classification": state_digest(self.model.segdec.cls_conv) + state_digest(self.model.segdec.cls_fc)}

    def save(self, path, signature, step):
        torch = self.torch
        np_state = np.random.get_state()
        torch.cuda.synchronize()
        torch.save({"purpose": p.NAME, "identity": signature, "step": step,
                    "model": self.model.state_dict(), "optimizer": self.optimizer.state_dict(),
                    "cpu_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all(),
                    "python_rng": random.getstate(),
                    "numpy_rng": [np_state[0], np_state[1].tolist(), *np_state[2:]]}, path)

    def load(self, path, signature, step):
        torch = self.torch
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if payload["purpose"] != p.NAME or payload["identity"] != signature or payload["step"] != step:
            raise ValueError("Checkpoint identity differs")
        del self.optimizer, self.model
        torch.cuda.empty_cache()
        self.model = self.new_model()
        self.model.load_state_dict(payload["model"], strict=True)
        self.optimizer = self.new_optimizer()
        self.optimizer.load_state_dict(payload["optimizer"])
        torch.set_rng_state(payload["cpu_rng"])
        torch.cuda.set_rng_state_all(payload["cuda_rng"])
        random.setstate(payload["python_rng"])
        state = payload["numpy_rng"]
        np.random.set_state((state[0], np.array(state[1], dtype=np.uint32), *state[2:]))
        current = np.random.get_state()
        if (not torch.equal(torch.get_rng_state(), payload["cpu_rng"])
                or any(not torch.equal(a, b) for a, b in zip(torch.cuda.get_rng_state_all(), payload["cuda_rng"], strict=True))
                or random.getstate() != payload["python_rng"]
                or current[0] != state[0] or current[1].tolist() != state[1] or list(current[2:]) != state[2:]):
            raise ValueError("RNG state restoration failed")
        parameters = [t for g in self.optimizer.param_groups for t in g["params"]]
        if step and (len(self.optimizer.state) != len(parameters) or
                     any(int(s["step"].item()) != step for s in self.optimizer.state.values())):
            raise ValueError("Optimizer state restoration failed")

    def memory(self):
        return {"torch_cuda_peak_allocated_bytes": self.torch.cuda.max_memory_allocated(),
                "torch_cuda_peak_reserved_bytes": self.torch.cuda.max_memory_reserved()}
