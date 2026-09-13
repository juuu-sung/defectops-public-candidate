"""Deterministic binary and segmentation metrics for frozen anomaly evaluation."""

from __future__ import annotations

import math
from collections import deque
from statistics import NormalDist
from typing import Any

import numpy as np


def safe_ratio(numerator: int, denominator: int) -> float | None:
    """Return a ratio, or ``None`` when its denominator is zero."""
    return numerator / denominator if denominator else None


def confusion_counts(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, int]:
    """Calculate binary confusion counts using ``score >= threshold``."""
    labels = np.asarray(labels, dtype=np.bool_).reshape(-1)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    if labels.size != scores.size or labels.size == 0:
        raise ValueError("labels and scores must have the same non-zero length")
    if not np.isfinite(scores).all() or not math.isfinite(threshold):
        raise ValueError("scores and threshold must be finite")
    predicted = scores >= threshold
    return {
        "tp": int(np.count_nonzero(predicted & labels)),
        "fp": int(np.count_nonzero(predicted & ~labels)),
        "tn": int(np.count_nonzero(~predicted & ~labels)),
        "fn": int(np.count_nonzero(~predicted & labels)),
    }


def rates_from_confusion(counts: dict[str, int]) -> dict[str, float | None]:
    """Return fixed-threshold rates without inventing values for zero denominators."""
    tp, fp, tn, fn = (counts[key] for key in ("tp", "fp", "tn", "fn"))
    precision = safe_ratio(tp, tp + fp)
    recall = safe_ratio(tp, tp + fn)
    return {
        "precision": precision,
        "recall": recall,
        "f1": None if precision is None or recall is None or precision + recall == 0 else (
            2 * precision * recall / (precision + recall)
        ),
        "escape_rate": safe_ratio(fn, tp + fn),
        "overkill_rate": safe_ratio(fp, tn + fp),
        "review_rate": safe_ratio(tp + fp, tp + fp + tn + fn),
    }


def wilson_interval(successes: int, total: int, confidence_level: float) -> list[float] | None:
    """Return a Wilson score interval for an image-level binomial proportion."""
    if successes < 0 or total < 0 or successes > total or not 0 < confidence_level < 1:
        raise ValueError("invalid Wilson interval inputs")
    if total == 0:
        return None
    z_value = NormalDist().inv_cdf(0.5 + confidence_level / 2.0)
    proportion = successes / total
    denominator = 1.0 + z_value**2 / total
    center = (proportion + z_value**2 / (2 * total)) / denominator
    half_width = (
        z_value
        * math.sqrt(proportion * (1.0 - proportion) / total + z_value**2 / (4 * total**2))
        / denominator
    )
    return [max(0.0, center - half_width), min(1.0, center + half_width)]


def binary_auroc(labels: np.ndarray, scores: np.ndarray) -> float | None:
    """Calculate exact binary AUROC with average ranks for tied scores."""
    labels = np.asarray(labels, dtype=np.bool_).reshape(-1)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    if labels.size != scores.size or labels.size == 0:
        raise ValueError("labels and scores must have the same non-zero length")
    if not np.isfinite(scores).all():
        raise ValueError("scores must be finite")
    positives = int(np.count_nonzero(labels))
    negatives = labels.size - positives
    if positives == 0 or negatives == 0:
        return None

    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    sorted_labels = labels[order]
    boundaries = np.flatnonzero(np.r_[True, sorted_scores[1:] != sorted_scores[:-1], True])
    starts = boundaries[:-1]
    ends = boundaries[1:]
    positive_prefix = np.r_[0, np.cumsum(sorted_labels, dtype=np.int64)]
    positive_counts = positive_prefix[ends] - positive_prefix[starts]
    average_ranks = (starts + 1 + ends) / 2.0
    positive_rank_sum = float(np.sum(average_ranks * positive_counts, dtype=np.float64))
    return (positive_rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)


def binary_average_precision(labels: np.ndarray, scores: np.ndarray) -> float | None:
    """Calculate non-interpolated binary average precision with score ties grouped."""
    labels = np.asarray(labels, dtype=np.bool_).reshape(-1)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    if labels.size != scores.size or labels.size == 0:
        raise ValueError("labels and scores must have the same non-zero length")
    if not np.isfinite(scores).all():
        raise ValueError("scores must be finite")
    positives = int(np.count_nonzero(labels))
    if positives == 0:
        return None

    order = np.argsort(-scores, kind="mergesort")
    sorted_scores = scores[order]
    sorted_labels = labels[order]
    group_ends = np.r_[np.flatnonzero(sorted_scores[1:] != sorted_scores[:-1]) + 1, labels.size]
    cumulative_tp = np.cumsum(sorted_labels, dtype=np.int64)[group_ends - 1]
    precision = cumulative_tp / group_ends
    recall = cumulative_tp / positives
    recall_increments = np.diff(np.r_[0.0, recall])
    return float(np.sum(recall_increments * precision, dtype=np.float64))


def stratified_bootstrap_interval(
    labels: np.ndarray,
    scores: np.ndarray,
    metric,
    *,
    seed: int,
    replicates: int,
    confidence_level: float,
) -> list[float] | None:
    """Return a deterministic percentile interval from image-stratified bootstrap samples."""
    labels = np.asarray(labels, dtype=np.bool_).reshape(-1)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    if labels.size != scores.size or replicates <= 0 or not 0 < confidence_level < 1:
        raise ValueError("invalid bootstrap inputs")
    positive_indices = np.flatnonzero(labels)
    negative_indices = np.flatnonzero(~labels)
    if positive_indices.size == 0 or negative_indices.size == 0:
        return None
    generator = np.random.default_rng(seed)
    values = np.empty(replicates, dtype=np.float64)
    for index in range(replicates):
        sample = np.r_[
            generator.choice(positive_indices, positive_indices.size, replace=True),
            generator.choice(negative_indices, negative_indices.size, replace=True),
        ]
        value = metric(labels[sample], scores[sample])
        if value is None:
            raise RuntimeError("stratified bootstrap unexpectedly produced a single-class sample")
        values[index] = value
    alpha = (1.0 - confidence_level) / 2.0
    return [
        float(np.quantile(values, alpha, method="linear")),
        float(np.quantile(values, 1.0 - alpha, method="linear")),
    ]


def connected_components_4(mask: np.ndarray) -> list[dict[str, Any]]:
    """Return 4-connected foreground components and exclusive XYXY boxes."""
    mask = np.asarray(mask, dtype=np.bool_)
    if mask.ndim != 2:
        raise ValueError("connected-component masks must be two-dimensional")
    height, width = mask.shape
    visited = np.zeros_like(mask)
    components: list[dict[str, Any]] = []
    for flat_index in np.flatnonzero(mask):
        y, x = divmod(int(flat_index), width)
        if visited[y, x]:
            continue
        queue = deque([(y, x)])
        visited[y, x] = True
        pixels: list[int] = []
        x_min = x_max = x
        y_min = y_max = y
        while queue:
            current_y, current_x = queue.popleft()
            pixels.append(current_y * width + current_x)
            x_min = min(x_min, current_x)
            x_max = max(x_max, current_x)
            y_min = min(y_min, current_y)
            y_max = max(y_max, current_y)
            for neighbor_y, neighbor_x in (
                (current_y - 1, current_x),
                (current_y + 1, current_x),
                (current_y, current_x - 1),
                (current_y, current_x + 1),
            ):
                if (
                    0 <= neighbor_y < height
                    and 0 <= neighbor_x < width
                    and mask[neighbor_y, neighbor_x]
                    and not visited[neighbor_y, neighbor_x]
                ):
                    visited[neighbor_y, neighbor_x] = True
                    queue.append((neighbor_y, neighbor_x))
        components.append(
            {
                "pixels": np.asarray(pixels, dtype=np.int64),
                "area": len(pixels),
                "bbox_xyxy_exclusive": [x_min, y_min, x_max + 1, y_max + 1],
            }
        )
    components.sort(key=lambda component: (-component["area"], component["bbox_xyxy_exclusive"]))
    return components


def aupro_fpr_grid(
    anomaly_maps: np.ndarray,
    ground_truth_masks: np.ndarray,
    *,
    fpr_limit: float,
    fpr_points: int,
) -> dict[str, float | int]:
    """Calculate normalized AUPRO on a deterministic background-FPR grid."""
    maps = np.asarray(anomaly_maps, dtype=np.float32)
    masks = np.asarray(ground_truth_masks, dtype=np.bool_)
    if maps.shape != masks.shape or maps.ndim != 3 or not np.isfinite(maps).all():
        raise ValueError("maps and masks must be finite arrays with matching N,H,W shapes")
    if not 0 < fpr_limit < 1 or fpr_points < 2:
        raise ValueError("invalid AUPRO grid settings")
    background_scores = np.sort(maps[~masks].astype(np.float64, copy=False))
    if background_scores.size == 0:
        raise ValueError("AUPRO requires background pixels")

    region_scores: list[np.ndarray] = []
    for anomaly_map, mask in zip(maps, masks, strict=True):
        flat_map = anomaly_map.reshape(-1)
        for component in connected_components_4(mask):
            region_scores.append(flat_map[component["pixels"]].astype(np.float64, copy=False))
    if not region_scores:
        raise ValueError("AUPRO requires at least one anomalous region")

    requested_fpr = np.linspace(0.0, fpr_limit, fpr_points, dtype=np.float64)
    thresholds = np.empty_like(requested_fpr)
    thresholds[0] = np.nextafter(background_scores[-1], math.inf)
    for index, target_fpr in enumerate(requested_fpr[1:], start=1):
        selected_background = max(1, math.ceil(target_fpr * background_scores.size))
        thresholds[index] = background_scores[-selected_background]

    observed_fpr = np.asarray(
        [np.count_nonzero(background_scores >= threshold) / background_scores.size for threshold in thresholds],
        dtype=np.float64,
    )
    pro = np.asarray(
        [
            np.mean([np.count_nonzero(scores >= threshold) / scores.size for scores in region_scores])
            for threshold in thresholds
        ],
        dtype=np.float64,
    )

    unique_fpr: list[float] = []
    unique_pro: list[float] = []
    for fpr, overlap in zip(observed_fpr, pro, strict=True):
        if unique_fpr and math.isclose(fpr, unique_fpr[-1], rel_tol=0.0, abs_tol=1e-15):
            unique_pro[-1] = max(unique_pro[-1], float(overlap))
        else:
            unique_fpr.append(float(fpr))
            unique_pro.append(float(overlap))
    if unique_fpr[-1] < fpr_limit:
        raise RuntimeError("AUPRO grid did not reach the configured FPR limit")
    if unique_fpr[-1] > fpr_limit:
        right = len(unique_fpr) - 1
        while right > 0 and unique_fpr[right - 1] >= fpr_limit:
            right -= 1
        left = max(0, right - 1)
        if unique_fpr[right] == unique_fpr[left]:
            endpoint_pro = unique_pro[right]
        else:
            fraction = (fpr_limit - unique_fpr[left]) / (unique_fpr[right] - unique_fpr[left])
            endpoint_pro = unique_pro[left] + fraction * (unique_pro[right] - unique_pro[left])
        unique_fpr = unique_fpr[:right] + [fpr_limit]
        unique_pro = unique_pro[:right] + [endpoint_pro]

    area = float(np.trapezoid(np.asarray(unique_pro), np.asarray(unique_fpr)))
    return {
        "value": area / fpr_limit,
        "fpr_limit": fpr_limit,
        "fpr_points_requested": fpr_points,
        "fpr_points_observed": len(unique_fpr),
        "region_count": len(region_scores),
    }
