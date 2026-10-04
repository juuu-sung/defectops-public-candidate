"""Inference only, restricted to three authenticated historical checkpoints."""
import hashlib
import numpy as np
from scripts.mpi_efficientad_evaluation_runtime import Runtime as FinalRuntime, environment
from scripts.mpi_efficientad_probe_runtime import Runtime as TrainingRuntime
from scripts.mpi_efficientad_trajectory import STEPS


class Runtime(FinalRuntime):
    def __init__(self, checkpoint, signature, step):
        import torch
        from torchvision.transforms import ToTensor
        from anomalib.models.image.efficient_ad.torch_model import EfficientAdModel, EfficientAdModelSize
        if step not in STEPS:
            raise ValueError("Unplanned checkpoint")
        if torch.cuda.mem_get_info()[0] < 6*1024**3:
            raise RuntimeError("Need 6 GiB free GPU; close other GPU jobs normally")
        self.torch = torch
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.cuda.reset_peak_memory_stats()
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if (state["purpose"] != "formal_efficientad_research_not_service" or
                state["identity"] != signature or state["step"] != step):
            raise ValueError("Checkpoint identity/step differs")
        self.fixed_fingerprints = {}
        for prefix in ("teacher.", "mean_std."):
            digest = hashlib.sha256()
            for key, value in sorted(state["model"].items()):
                if key.startswith(prefix):
                    digest.update(key.encode())
                    digest.update(value.contiguous().numpy().tobytes())
            self.fixed_fingerprints[prefix] = digest.hexdigest()
        self.model = EfficientAdModel(teacher_out_channels=384, model_size=EfficientAdModelSize.S,
                                     padding=False, pad_maps=True)
        self.model.load_state_dict(state["model"], strict=True)
        del state
        self.model.requires_grad_(False).eval().to("cuda")
        self.to_tensor = ToTensor()

    def input_audit(self, image):
        # Invoke the unchanged historical training input method without an optimizer/fit.
        training = TrainingRuntime.tensor(self, image)
        evaluation = self.to_tensor(image).unsqueeze(0).to("cuda")
        expected = np.asarray(image).transpose(2, 0, 1)[None].astype(np.float32)/255
        actual = evaluation.cpu().numpy()
        if not self.torch.equal(training, evaluation) or not np.array_equal(actual, expected):
            raise ValueError("Training/evaluation RGB tensor mismatch")
        return {"mode": image.mode, "size": list(image.size), "dtype": str(actual.dtype),
                "shape": list(actual.shape), "range": [float(actual.min()), float(actual.max())],
                "training_evaluation_maximum_difference": 0.,
                "rgb_divide_255_maximum_difference": 0.,
                "input_tensor_sha256": hashlib.sha256(actual.tobytes()).hexdigest(),
                "fixed_teacher_and_statistics_sha256": self.fixed_fingerprints}
