"""Shared prediction utilities for floor, normalize, and static overrides.

Extracted from predictor.py and predictor_v2.py to eliminate duplication.
All prediction modules should import these instead of maintaining copies.
"""

from __future__ import annotations

import numpy as np

from src.constants import (
    NUM_PREDICTION_CLASSES,
    OBS_CONFIDENCE_K,
    PROBABILITY_FLOOR,
    STATIC_TERRAIN_CONFIDENCE,
)
from src.terrain import InternalTerrain


def floor_and_normalize(
    tensor: np.ndarray,
    floor: float = PROBABILITY_FLOOR,
) -> np.ndarray:
    """Apply probability floor and renormalize each cell to sum to 1.

    Prevents infinite KL divergence from zero probabilities.

    Args:
        tensor: H x W x C probability tensor.
        floor: Minimum probability value per class.

    Returns:
        Tensor with all values >= floor, each cell summing to 1.0.
    """
    safe = np.maximum(tensor, floor)
    return safe / safe.sum(axis=-1, keepdims=True)


def apply_static_overrides(
    tensor: np.ndarray,
    grid: np.ndarray,
) -> np.ndarray:
    """Override ocean and mountain cells with near-certain probabilities.

    Ocean cells get [~1, 0, 0, 0, 0, 0] (class 0 = Empty).
    Mountain cells get [0, 0, 0, 0, 0, ~1] (class 5 = Mountain).
    Residual probability is spread evenly across other classes.

    Args:
        tensor: H x W x C probability tensor.
        grid: H x W grid of InternalTerrain values.

    Returns:
        Tensor with static terrain cells overridden.
    """
    result = tensor.copy()
    residual = 1.0 - STATIC_TERRAIN_CONFIDENCE
    per_class = residual / (NUM_PREDICTION_CLASSES - 1)

    static_pairs = [
        (InternalTerrain.OCEAN, 0),
        (InternalTerrain.MOUNTAIN, 5),
    ]
    for terrain, cls_idx in static_pairs:
        mask = grid == terrain
        if mask.any():
            result[mask] = per_class
            result[mask, cls_idx] = STATIC_TERRAIN_CONFIDENCE
    return result


def blend_observations(
    tensor: np.ndarray,
    obs_store: object,
    seed_idx: int,
    max_weight: float,
    k: float = OBS_CONFIDENCE_K,
) -> np.ndarray:
    """Blend observation data into tensor with count-scaled weights.

    Observation weight scales with count: w = max_weight * count/(count+k).
    With 1 observation and k=5, w = max_weight * 0.17 -- prior dominates.
    With 10 observations and k=5, w = max_weight * 0.67 -- obs dominates.

    Args:
        tensor: H x W x C base probability tensor.
        obs_store: Observation store with get_observed_probs,
            get_coverage_mask, and observation_count methods.
        seed_idx: Which seed to blend observations for.
        max_weight: Maximum observation weight at infinite count.
        k: Confidence scaling constant (higher = slower ramp-up).

    Returns:
        Tensor with observed cells blended toward observation values.
    """
    obs_probs = obs_store.get_observed_probs(seed_idx)
    coverage = obs_store.get_coverage_mask(seed_idx)
    obs_counts = obs_store.observation_count(seed_idx)
    observed = coverage & ~np.isnan(obs_probs[:, :, 0])

    if not observed.any():
        return tensor

    counts = obs_counts[observed].astype(np.float64)
    w_obs = (max_weight * counts / (counts + k))[:, np.newaxis]
    result = tensor.copy()
    result[observed] = w_obs * obs_probs[observed] + (1.0 - w_obs) * tensor[observed]
    return result
