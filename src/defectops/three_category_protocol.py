"""Validate the preregistered three-category comparison contract."""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any


EXPECTED_REGISTRATION_STATUS = (
    "registered_before_cable_and_metal_nut_test_observed"
)
EXPECTED_EVIDENCE_STATUS = (
    "mixed_transistor_observed_additional_categories_preregistered"
)


def load_three_category_protocol(path: Path) -> dict[str, Any]:
    """Load the immutable comparison plan and reject silent scope or rule changes."""
    config = tomllib.loads(path.read_text(encoding="utf-8"))
    input_config = config.get("input", {})
    padim = config.get("padim", {})
    patchcore = config.get("patchcore", {})
    calibration = config.get("calibration", {})
    evaluation = config.get("evaluation", {})
    selection = config.get("selection", {})
    checks = {
        "schema_version": config.get("schema_version") == 1,
        "protocol": config.get("protocol") == "A-normal-only-v1",
        "registration_status": config.get("registration_status")
        == EXPECTED_REGISTRATION_STATUS,
        "evidence_status": config.get("evidence_status") == EXPECTED_EVIDENCE_STATUS,
        "categories": config.get("categories")
        == ["transistor", "cable", "metal_nut"],
        "observed_categories": config.get("observed_test_categories")
        == ["transistor"],
        "untouched_categories": config.get("untouched_test_categories")
        == ["cable", "metal_nut"],
        "split_seed": config.get("split_seed") == 42,
        "calibration_fraction": config.get("calibration_fraction") == 0.20,
        "model_seeds": config.get("model_seeds") == [42, 43, 44],
        "modeling_unit": config.get("modeling_unit")
        == "separate_model_and_threshold_per_category",
        "input": input_config
        == {
            "source_encoding": "png",
            "source_color": "rgb",
            "resize": "stretch_to_square_bilinear_antialias",
            "image_size": 256,
            "normalization": "imagenet_mean_std",
            "augmentation": "none",
        },
        "padim": padim
        == {
            "backbone": "resnet18",
            "layers": ["layer1", "layer2", "layer3"],
            "n_features": 100,
        },
        "patchcore": patchcore
        == {
            "backbone": "wide_resnet50_2",
            "layers": ["layer2", "layer3"],
            "coreset_sampling_ratio": 0.10,
            "num_neighbors": 9,
        },
        "calibration": calibration
        == {
            "image_quantile": 0.95,
            "pixel_quantile": 0.995,
            "quantile_method": "higher",
            "score_space": "raw",
            "decision_rule": "score >= threshold",
            "morphology": "none",
        },
        "evaluation_metrics": evaluation.get("image_metrics")
        == ["auroc", "average_precision", "fixed_threshold_confusion"]
        and evaluation.get("pixel_metrics")
        == ["auroc", "aupro", "fixed_threshold_confusion"]
        and evaluation.get("aupro_fpr_limit") == 0.30
        and evaluation.get("bootstrap_unit") == "image_stratified"
        and evaluation.get("reporting")
        == "category_and_seed_values_plus_macro_mean_sample_std",
        "selection": selection
        == {
            "scope": "portfolio_candidate_not_unbiased_deployment_estimate",
            "primary_rule": "lowest_macro_escape_rate_at_frozen_threshold",
            "tie_breakers": [
                "highest_macro_image_auroc",
                "lowest_macro_overkill_rate",
                "lowest_warm_p95_latency",
            ],
            "automatic_pass_allowed": False,
            "post_test_tuning_requires_new_untouched_data": True,
        },
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError("Three-category protocol mismatch: " + ", ".join(failed))
    return config
