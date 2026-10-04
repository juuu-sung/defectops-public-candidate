"""Frozen Phase 0/1 contract. No calibration, test access or formal fit entry point."""
from __future__ import annotations

import hashlib
import numpy as np

NAME = "mpi-supersimplenet-probe-v1"
POLICY = {
    "purpose": "normal_only_resource_probe_not_formal_training",
    "seed": 42, "image_size": 256, "batch_size": 8,
    "normal_photos": 8, "tiles": 96, "optimizer_updates": 100,
    "checkpoint_interval": 25, "backbone": "wide_resnet50_2.tv_in1k",
    "layers": ["layer2", "layer3"], "perlin_threshold": .6,
    "noise_std": .015, "stop_grad": True, "precision": "float32",
    "normalization": "RGB/255, ImageNet mean/std exactly once; no crop/augmentation",
    "adaptor_lr": .0001, "adaptor_weight_decay": .01,
    "heads_lr": .0002, "heads_weight_decay": .00001,
    "optimizer": "AdamW", "lr_schedule": "constant_probe_only",
    "native_score": "signed classification logit; not map maximum or probability",
    "reload_absolute_tolerance": 1e-4,
    "resource_plan": {"comparison_seconds": 900, "rss_GiB": 20,
                      "selection_and_inference_reserved_GiB": 6},
    "full_training_authorized": False, "automatic_pass_allowed": False,
}

# Full SHA from the public timm repository LFS metadata; revision is immutable.
WEIGHTS = {
    "url": "https://huggingface.co/timm/wide_resnet50_2.tv_in1k/resolve/343c5fa59ca8a131e0cdf42144a641b36a0387c2/model.safetensors",
    "filename": "wide_resnet50_2.tv_in1k.safetensors",
    "sha256": "df6fb6c4824769769de18e14088475fd6ee94236849aa4e5d8022ba9d9a9a16c",
    "maximum_bytes": 300 * 1024**2,
}

# Anomalib v2.2.0 tag; normalize CRLF so Windows source installs are accepted.
SOURCE_HASHES = {
    "models/image/supersimplenet/torch_model.py": "cd18005768fae180289e6e79c62dd49ad1e3ff9139bbf8a13bba082388661b01",
    "models/image/supersimplenet/anomaly_generator.py": "39afccdec5e354c49e827a005f1c856eeb5120a968922c5b0ba32ccfce24ff63",
    "models/image/supersimplenet/loss.py": "fa3381a2cf4847fe42a792a74fbb774421602389f585f0f67204159ccd6c2c21",
    "models/components/feature_extractors/timm.py": "018f16cbba5fb85e29a286a00f22fc50685517eed9014f32acbc24119600949c",
    "data/utils/generators/perlin.py": "f71a1c9dba3e658e7176fc563fbb8b12b6ace8db9b6dc43bb02328177aacc153",
}


def verify_sources(root):
    found = {name: hashlib.sha256((root / name).read_bytes().replace(b"\r\n", b"\n")).hexdigest()
             for name in SOURCE_HASHES}
    if found != SOURCE_HASHES:
        raise ValueError("Anomalib implementation differs from reviewed v2.2.0 source")
    return found


def batch_indices(index):
    """Independent epoch permutation: replay order without consuming model RNG."""
    if not 0 <= index < 100:
        raise ValueError("Only 100 probe updates authorized")
    epoch, batch = divmod(index, 12)
    order = np.random.default_rng(42 + epoch).permutation(96)
    return order[batch*8:(batch+1)*8].tolist()


def validate_prediction(scores, maps, count):
    # Official classification head squeezes batch1 to a scalar. Preserve signed logits.
    scores, maps = np.asarray(scores), np.asarray(maps)
    if scores.shape not in ((count,), ()) or (scores.ndim == 0 and count != 1):
        raise ValueError("Unexpected native score shape")
    if maps.shape != (count, 1, 256, 256):
        raise ValueError("Unexpected native map shape")
    if not np.isfinite(scores).all() or not np.isfinite(maps).all():
        raise ValueError("Nonfinite native prediction")
    # A single numeric reference file includes BOTH heads, not only the heatmap.
    return np.concatenate((scores.reshape(count, 1), maps.reshape(count, -1)), axis=1)


def replay_difference(actual, expected):
    actual, expected = np.asarray(actual), np.asarray(expected)
    if actual.shape != expected.shape or actual.shape != (1, 65537):
        raise ValueError("Reload reference shape differs")
    if not np.isfinite(actual).all() or not np.isfinite(expected).all():
        raise ValueError("Nonfinite reload reference")
    delta = float(np.max(np.abs(actual - expected)))
    if delta > POLICY["reload_absolute_tolerance"]:
        raise ValueError(f"Reload map/score difference exceeds tolerance: {delta}")
    return {"maximum_raw_score_difference": float(abs(actual[0, 0]-expected[0, 0])),
            "maximum_raw_map_difference": float(np.max(np.abs(actual[:, 1:]-expected[:, 1:])))}
