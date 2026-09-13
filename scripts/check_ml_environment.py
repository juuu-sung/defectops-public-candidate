#!/usr/bin/env python3
"""GPU smoke test: synthetic tensors only; no dataset or pretrained weights."""

import argparse
import importlib.metadata
import json
import platform
import random
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = {"status": "failed", "python": platform.python_version(),
              "executable": sys.executable, "os": platform.platform(),
              "purpose": "Synthetic GPU operations and PaDiM features; not model training/evaluation."}
    stage = "import torch/torchvision"
    try:
        import torch
        import torchvision

        report["versions"] = {"torch": torch.__version__, "torchvision": torchvision.__version__,
                              "anomalib": importlib.metadata.version("anomalib")}
        report["torch_cuda_runtime"] = torch.version.cuda
        report["cuda_available"] = torch.cuda.is_available()
        if not report["cuda_available"]:
            raise RuntimeError("CUDA unavailable. Check the selected venv and CUDA wheel.")
        report["gpu"] = torch.cuda.get_device_name(0)
        report["compute_capability"] = list(torch.cuda.get_device_capability(0))
        stage = "CUDA matrix multiplication"
        with torch.no_grad():
            x = torch.ones((64, 64), device="cuda")
            y = x @ x
            torch.cuda.synchronize()
            if not bool(torch.all(y == 64).item()):
                raise RuntimeError("Unexpected CUDA matrix multiplication result")
        report["matrix_multiply"] = "passed"
        stage = "torchvision CUDA operator"
        boxes = torch.tensor([[0., 0., 10., 10.], [0., 0., 10., 10.]], device="cuda")
        scores = torch.tensor([.9, .8], device="cuda")
        selected = torchvision.ops.nms(boxes, scores, .5)
        torch.cuda.synchronize()
        if selected.tolist() != [0]:
            raise RuntimeError("Unexpected torchvision CUDA NMS result")
        report["torchvision_cuda_operator"] = "passed"
        stage = "Anomalib PaDiM feature extraction"
        from anomalib.models import Padim

        random.seed(42)
        torch.manual_seed(42)
        # Actual training must use pretrained weights and the frozen manifest.
        model = Padim(backbone="resnet18", pre_trained=False, n_features=100,
                      pre_processor=False, post_processor=False, evaluator=False, visualizer=False).to("cuda")
        model.model.train()
        with torch.no_grad():
            embedding = model.model(torch.zeros((1, 3, 256, 256), device="cuda"))
        torch.cuda.synchronize()
        if not isinstance(embedding, torch.Tensor) or not bool(torch.isfinite(embedding).all().item()):
            raise RuntimeError("PaDiM did not return finite feature tensors")
        report["padim_embedding_shape"] = list(embedding.shape)
        report["status"] = "passed"
    except Exception as exc:
        report["failed_stage"] = stage
        report["error"] = f"{type(exc).__name__}: {exc}"
    output = json.dumps(report, indent=2, ensure_ascii=True)
    print(output)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output + "\n", encoding="utf-8")
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
