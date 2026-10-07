"""Inference-only scaling loader. Does not load an optimizer or download weights."""
from unittest.mock import patch

from scripts import mpi_supersimplenet_scaling_protocol as s
from scripts.mpi_supersimplenet_evaluation_runtime import Runtime as Base, environment
from scripts.run_mpi_supersimplenet_evaluation import Monitor as SampleMonitor
from scripts.mpi_supersimplenet_scaling_runtime import Monitor as NoTimeMonitor


class Monitor(SampleMonitor):
    check = NoTimeMonitor.check


class Runtime(Base):
    def __init__(self, checkpoint, signature, arm):
        import torch
        from torchvision import transforms
        from anomalib.models.components import TimmFeatureExtractor
        from anomalib.models.image.supersimplenet import torch_model
        self.torch = torch
        if torch.cuda.mem_get_info()[0] < 6*1024**3:
            raise RuntimeError("Need 6 GiB free GPU; close other GPU jobs normally")
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.cuda.reset_peak_memory_stats()
        policy = s.Protocol(arm).POLICY
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if (state["purpose"] != s.NAME+"-"+arm or state["step"] != policy["updates"]
                or state["identity"] != signature):
            raise ValueError("Not the frozen final scaling checkpoint")
        def offline(*args, **kwargs):
            return TimmFeatureExtractor(*args, **kwargs, pre_trained=False)
        with patch.object(torch_model, "TimmFeatureExtractor", offline):
            model = torch_model.SupersimplenetModel(perlin_threshold=.6,
                backbone=policy["backbone"], layers=["layer2", "layer3"], stop_grad=True)
        model.load_state_dict(state["model"], strict=True)
        del state
        self.model = model.requires_grad_(False).eval().to("cuda")
        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(
            mean=[.485, .456, .406], std=[.229, .224, .225])])
