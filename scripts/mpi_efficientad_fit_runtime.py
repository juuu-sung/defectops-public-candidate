"""Formal EfficientAD CUDA boundary and pinned full Imagenette train preparation."""
from __future__ import annotations

import hashlib
import io
import json
from pathlib import PurePosixPath
import random
import tarfile

import numpy as np
from PIL import Image
from scripts import mpi_efficientad_probe_assets as assets
from scripts.mpi_efficientad_probe_runtime import Runtime as ProbeRuntime, environment
from scripts import mpi_efficientad_fit_protocol as p

fit = p.fit


def train_members(members):
    names, seen = [], set()
    for m in members:
        path = PurePosixPath(m.name)
        if path.is_absolute() or ".." in path.parts or "\\" in m.name or m.issym() or m.islnk():
            raise ValueError("Unsafe Imagenette member")
        if m.name in seen:
            raise ValueError("Duplicate Imagenette member")
        seen.add(m.name)
        if (len(path.parts) == 4 and path.parts[:2] == ("imagenette2", "train")
                and path.suffix.lower() in (".jpg", ".jpeg") and not path.name.startswith(".")):
            if not m.isfile() or not 0 < m.size <= 16 * 1024**2:
                raise ValueError("Invalid Imagenette image")
            names.append(m.name)
    if len(names) != 9469:
        raise ValueError("Expected full 9469-image Imagenette training pool")
    return sorted(names)


def prepare_assets(root):
    # Assets were acquired with probe approval. No network or validation-image access.
    teacher_root, probe_manifest, _ = assets.prepare(root, False)
    signature = fit.digest({"sources": assets.ASSETS, "train_count": 9469, "version": 1})
    files = ["manifest.json"] + [f"aux-{i:05d}.jpg" for i in range(9469)]
    phase = "formal-train-v1"
    if (root / phase).exists():
        fit.read_phase(root, phase, signature, files)
    else:
        archive = root / assets.ASSETS["imagenette"]["filename"]
        with tarfile.open(archive, "r:gz") as t:
            selected = train_members(t.getmembers())
        indices = {n: i for i, n in enumerate(selected)}
        with fit.phase_stage(root, phase) as stage:
            rows = []
            with tarfile.open(archive, "r|gz") as t:
                for m in t:
                    if m.name not in indices:
                        continue
                    raw = t.extractfile(m).read()
                    with Image.open(io.BytesIO(raw)) as image:
                        if image.format != "JPEG":
                            raise ValueError("Expected Imagenette JPEG")
                        image.load()
                    i = indices[m.name]
                    filename = f"aux-{i:05d}.jpg"
                    (stage / filename).write_bytes(raw)
                    rows.append({"index": i, "source": m.name, "file": filename,
                                 "sha256": hashlib.sha256(raw).hexdigest()})
                    if len(rows) % 1000 == 0:
                        print(f"Preparing auxiliary train images {len(rows)}/9469", flush=True)
            if len(rows) != 9469:
                raise ValueError("Incomplete Imagenette train preparation")
            fit.write_json(stage / "manifest.json", {"sources": assets.ASSETS,
                "teacher_sha256": probe_manifest["teacher_sha256"],
                "auxiliary": sorted(rows, key=lambda r: r["index"]),
                "scope": "research only; no images or weights in handoff; no val use"})
            fit.seal_phase(stage, phase, signature, {}, files)
    manifest = json.loads((root / phase / "manifest.json").read_bytes())
    if manifest["teacher_sha256"] != fit.sha256_file(teacher_root / "teacher.pth"):
        raise ValueError("Formal teacher differs from probe teacher")
    return teacher_root / "teacher.pth", root / phase, manifest


class Runtime(ProbeRuntime):
    def statistics(self, images, check):
        result = super().statistics(images, check)
        if result["spatial_observations_per_channel"] != 768 * 56 * 56:
            raise ValueError("Full normal teacher statistics incomplete")
        result["statistics_scope"] = "all_768_formal_train_tiles_no_calibration"
        return result

    def step(self, image, index):
        torch = self.torch
        self.model.train()
        self.model.teacher.eval()
        rate = p.learning_rate(index)
        for group in self.optimizer.param_groups:
            group["lr"] = rate
        auxiliary_index = p.sample_index(index, 9469, "auxiliary")
        with Image.open(self.aux / f"aux-{auxiliary_index:05d}.jpg") as aux:
            background = self.aux_transform(aux.convert("RGB")).unsqueeze(0).to("cuda")
        self.optimizer.zero_grad(set_to_none=True)
        components = self.model(self.tensor(image), background)
        if len(components) != 3 or not all(torch.isfinite(v).all() for v in components):
            raise ValueError("Nonfinite EfficientAD loss")
        sum(components).backward()
        params = [v for g in self.optimizer.param_groups for v in g["params"]]
        if any(v.grad is None or not torch.isfinite(v.grad).all() for v in params):
            raise ValueError("Missing/nonfinite gradient")
        norm = sum(float(v.grad.detach().double().square().sum()) for v in params)**.5
        if not np.isfinite(norm) or norm <= 0:
            raise ValueError("Invalid gradient norm")
        self.optimizer.step()
        if any(not torch.isfinite(v).all() for v in params):
            raise ValueError("Nonfinite updated weights")
        torch.cuda.synchronize()
        return {"step": index + 1, "lr": rate, "auxiliary_index": auxiliary_index,
                "loss_components": [float(v.detach()) for v in components], "gradient_l2": norm}

    def predict(self, image):
        self.model.eval()
        with self.torch.no_grad():
            maps = self.model.get_maps(self.tensor(image), normalize=False)
        result = np.stack([v.detach().cpu().numpy()[0, 0] for v in maps])
        return p.validate_maps(result)

    def save(self, path, signature, step):
        torch = self.torch
        torch.cuda.synchronize()
        numpy_state = np.random.get_state()
        torch.save({"purpose": "formal_efficientad_research_not_service", "identity": signature,
                    "step": step, "model": self.model.state_dict(), "optimizer": self.optimizer.state_dict(),
                    "cpu_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all(),
                    "python_rng": random.getstate(),
                    "numpy_rng": [numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:]]}, path)

    def load(self, path, signature, step):
        torch = self.torch
        data = torch.load(path, map_location="cpu", weights_only=True)
        if (data["purpose"] != "formal_efficientad_research_not_service"
                or data["identity"] != signature or data["step"] != step):
            raise ValueError("Formal checkpoint identity differs; probe checkpoint forbidden")
        del self.optimizer, self.model
        torch.cuda.empty_cache()
        self.model = self.new_model()
        self.model.load_state_dict(data["model"], strict=True)
        self.optimizer = self.new_optimizer()
        self.optimizer.load_state_dict(data["optimizer"])
        torch.set_rng_state(data["cpu_rng"])
        torch.cuda.set_rng_state_all(data["cuda_rng"])
        random.setstate(data["python_rng"])
        ns = data["numpy_rng"]
        np.random.set_state((ns[0], np.array(ns[1], dtype=np.uint32), *ns[2:]))
        if (not torch.equal(torch.get_rng_state(), data["cpu_rng"])
                or any(not torch.equal(a, b) for a, b in zip(torch.cuda.get_rng_state_all(), data["cuda_rng"], strict=True))
                or random.getstate() != data["python_rng"]
                or np.random.get_state()[1].tolist() != ns[1]):
            raise ValueError("RNG restore mismatch")
        if step and (not self.optimizer.state or any(int(s["step"]) != step for s in self.optimizer.state.values())):
            raise ValueError("Optimizer restore mismatch")
        expected_lr = p.learning_rate(max(0, step-1))
        if any(g["lr"] != expected_lr for g in self.optimizer.param_groups):
            raise ValueError("Restored learning rate mismatch")
