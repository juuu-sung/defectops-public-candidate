"""Inference-only loader: no optimizer, downloads, teacher replacement or fitting."""
from scripts.mpi_efficientad_probe_runtime import environment


class Runtime:
    def __init__(self, checkpoint, signature):
        import torch
        from torchvision.transforms import ToTensor
        from anomalib.models.image.efficient_ad.torch_model import EfficientAdModel, EfficientAdModelSize
        self.torch = torch
        if torch.cuda.mem_get_info()[0] < 6 * 1024**3:
            raise RuntimeError("Need 6 GiB free GPU; close other GPU jobs normally")
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.cuda.reset_peak_memory_stats()
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if (state["purpose"] != "formal_efficientad_research_not_service"
                or state["step"] != 70000 or state["identity"] != signature):
            raise ValueError("Not the frozen final EfficientAD checkpoint")
        self.model = EfficientAdModel(teacher_out_channels=384, model_size=EfficientAdModelSize.S,
                                     padding=False, pad_maps=True)
        self.model.load_state_dict(state["model"], strict=True)
        del state
        self.model.requires_grad_(False).eval().to("cuda")
        self.to_tensor = ToTensor()

    def predict(self, image):
        import numpy as np
        if image.mode != "RGB" or image.size != (256, 256):
            raise ValueError("Expected original RGB 256x256 tile")
        with self.torch.inference_mode():
            maps = self.model.get_maps(self.to_tensor(image).unsqueeze(0).to("cuda"), normalize=False)
        return np.stack([a.cpu().numpy()[0, 0] for a in maps])

    def synchronize(self):
        self.torch.cuda.synchronize()

    def memory(self):
        return {"torch_cuda_peak_allocated_bytes": self.torch.cuda.max_memory_allocated(),
                "torch_cuda_peak_reserved_bytes": self.torch.cuda.max_memory_reserved()}

    def release(self):
        self.synchronize()
        del self.model
        self.torch.cuda.empty_cache()
