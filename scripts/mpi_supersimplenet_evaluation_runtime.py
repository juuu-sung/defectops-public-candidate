"""Inference-only final SuperSimpleNet loader; no optimizer or network downloads."""
from unittest.mock import patch

from scripts import mpi_supersimplenet_fit_protocol as p
from scripts.mpi_supersimplenet_runtime import Runtime as ProvenRuntime, environment


class Runtime:
    tensor = ProvenRuntime.tensor
    memory = ProvenRuntime.memory
    fingerprints = ProvenRuntime.fingerprints

    def __init__(self, checkpoint, signature):
        import torch
        from torchvision import transforms
        from anomalib.models.components import TimmFeatureExtractor
        from anomalib.models.image.supersimplenet import torch_model
        self.torch = torch
        if torch.cuda.mem_get_info()[0] < 6 * 1024**3:
            raise RuntimeError("Need 6 GiB free GPU; close other GPU jobs normally")
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.cuda.reset_peak_memory_stats()
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if state["purpose"] != p.NAME or state["step"] != 28800 or state["identity"] != signature:
            raise ValueError("Not the frozen final 300-epoch SuperSimpleNet checkpoint")

        def offline_extractor(*args, **kwargs):
            return TimmFeatureExtractor(*args, **kwargs, pre_trained=False)

        with patch.object(torch_model, "TimmFeatureExtractor", offline_extractor):
            model = torch_model.SupersimplenetModel(perlin_threshold=.6,
                backbone=p.POLICY["backbone"], layers=["layer2", "layer3"], stop_grad=True)
        model.load_state_dict(state["model"], strict=True)
        del state
        self.model = model.requires_grad_(False).eval().to("cuda")
        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(
            mean=[.485, .456, .406], std=[.229, .224, .225])])

    def predict(self, image):
        # Reuse exactly the signed-logit/raw-map extraction verified during fit.
        return ProvenRuntime.predict(self, [image])[0]

    def synchronize(self):
        self.torch.cuda.synchronize()

    def release(self):
        self.synchronize()
        del self.model
        self.torch.cuda.empty_cache()
