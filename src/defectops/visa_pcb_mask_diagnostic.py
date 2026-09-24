"""Read-only diagnosis of the first VisA PCB anomaly mask after a stopped audit."""

from __future__ import annotations

import io
import tarfile
from pathlib import Path

import numpy as np
from PIL import Image


MASK_MEMBER = "pcb1/Data/Masks/Anomaly/000.png"


def inspect_mask(mask: Image.Image) -> dict[str, object]:
    """Compare the locked parser with VisA's documented nonzero-first binarization."""
    raw = np.asarray(mask)
    if raw.ndim != 2:
        raise ValueError(f"Expected a single-channel class-ID mask, got mode {mask.mode} and shape {raw.shape}")
    old = mask.convert("L").resize((256, 256), Image.Resampling.NEAREST)
    binary = Image.fromarray(np.where(raw != 0, 255, 0).astype(np.uint8))
    corrected = binary.resize((256, 256), Image.Resampling.NEAREST)
    values, counts = np.unique(raw, return_counts=True)
    return {
        "mask_mode": mask.mode,
        "original_size": list(mask.size),
        "original_nonzero_pixels": int(np.count_nonzero(raw)),
        "original_pixels_at_least_128": int(np.count_nonzero(raw >= 128)),
        "original_value_counts": [{"value": int(value), "count": int(count)}
                                  for value, count in zip(values[:20], counts[:20])],
        "original_distinct_values": int(len(values)),
        "current_parser_resized_pixels_at_least_128": int(np.count_nonzero(np.asarray(old) >= 128)),
        "nonzero_first_resized_pixels_at_least_128": int(np.count_nonzero(np.asarray(corrected) >= 128)),
    }


def diagnose_archive(archive_path: Path) -> dict[str, object]:
    with tarfile.open(archive_path, "r:") as archive:
        member = archive.getmember(MASK_MEMBER)
        if not member.isfile():
            raise ValueError(f"Not a file: {MASK_MEMBER}")
        stream = archive.extractfile(member)
        if stream is None:
            raise ValueError(f"Cannot read: {MASK_MEMBER}")
        with Image.open(io.BytesIO(stream.read())) as mask:
            if mask.format != "PNG":
                raise ValueError(f"Not a PNG: {MASK_MEMBER}")
            mask.load()
            return {"status": "read_only_mask_diagnostic", "member": MASK_MEMBER,
                    "archive_modified": False, "test_scores_computed": 0, **inspect_mask(mask)}
