"""Online calibration using probe observation data.

After querying settlement-focused cells, computes per-terrain-type
error corrections to calibrate predictions for the current round.
"""

from __future__ import annotations

import logging

import numpy as np

from src.constants import (
    ONLINE_CALIBRATION_MAX_ADJUST,
    ONLINE_CALIBRATION_MIN_CELLS,
    PROBABILITY_FLOOR,
    REGIME_AGGRESSIVE,
    REGIME_AGGRESSIVE_THRESHOLD,
    REGIME_COLLAPSE,
    REGIME_COLLAPSE_THRESHOLD,
    REGIME_SURVIVE,
)
from src.terrain import InternalTerrain

logger = logging.getLogger(__name__)

NUM_TERRAIN_TYPES = 7


def online_calibrate(
    predictions: np.ndarray,
    grid: np.ndarray,
    obs_probs: np.ndarray,
    coverage_mask: np.ndarray,
) -> np.ndarray:
    """Calibrate predictions using observed probe data.

    For each observed cell, computes error = observed - predicted.
    Averages these errors per initial terrain type and applies as
    a global correction to all cells of that type.

    Args:
        predictions: H x W x 6 prediction tensor.
        grid: H x W array of InternalTerrain values.
        obs_probs: H x W x 6 observed probability tensor (NaN if unobserved).
        coverage_mask: H x W boolean mask of observed cells.

    Returns:
        H x W x 6 calibrated prediction tensor.
    """
    observed = coverage_mask & ~np.isnan(obs_probs[:, :, 0])
    num_observed = int(observed.sum())

    if num_observed < ONLINE_CALIBRATION_MIN_CELLS:
        logger.info(
            "Too few observed cells (%d) for calibration, skipping",
            num_observed,
        )
        return predictions

    corrections = _compute_terrain_corrections(predictions, grid, obs_probs, observed)
    result = _apply_terrain_corrections(predictions, grid, corrections)
    logger.info("Online calibration applied from %d observed cells", num_observed)
    return result


def detect_regime(
    obs_probs: np.ndarray,
    coverage_mask: np.ndarray,
    grid: np.ndarray,
) -> str:
    """Detect round regime from observation data.

    Examines settlement probability across observed non-static cells
    to classify the round as survive, collapse, or aggressive.

    Args:
        obs_probs: H x W x 6 observed probabilities.
        coverage_mask: H x W boolean mask.
        grid: H x W initial terrain grid.

    Returns:
        One of REGIME_SURVIVE, REGIME_COLLAPSE, REGIME_AGGRESSIVE.
    """
    observed = coverage_mask & ~np.isnan(obs_probs[:, :, 0])
    static = _static_mask(grid)
    dynamic_observed = observed & ~static

    if not dynamic_observed.any():
        return REGIME_SURVIVE

    settlement_prob = obs_probs[dynamic_observed, 1].mean()
    port_prob = obs_probs[dynamic_observed, 2].mean()
    combined = settlement_prob + port_prob

    if combined < REGIME_COLLAPSE_THRESHOLD:
        regime = REGIME_COLLAPSE
    elif combined > REGIME_AGGRESSIVE_THRESHOLD:
        regime = REGIME_AGGRESSIVE
    else:
        regime = REGIME_SURVIVE

    logger.info(
        "Detected regime: %s (settlement+port prob=%.3f)",
        regime,
        combined,
    )
    return regime


def _compute_terrain_corrections(
    predictions: np.ndarray,
    grid: np.ndarray,
    obs_probs: np.ndarray,
    observed: np.ndarray,
) -> dict[int, np.ndarray]:
    """Compute avg error per terrain type from observed cells."""
    corrections: dict[int, np.ndarray] = {}

    for t_val in range(NUM_TERRAIN_TYPES):
        combined = observed & (grid == t_val)
        n = int(combined.sum())
        if n < ONLINE_CALIBRATION_MIN_CELLS:
            continue
        error = obs_probs[combined] - predictions[combined]
        avg_error = error.mean(axis=0)
        avg_error = np.clip(
            avg_error,
            -ONLINE_CALIBRATION_MAX_ADJUST,
            ONLINE_CALIBRATION_MAX_ADJUST,
        )
        if np.abs(avg_error).max() > 0.001:
            corrections[t_val] = avg_error

    return corrections


def _apply_terrain_corrections(
    predictions: np.ndarray,
    grid: np.ndarray,
    corrections: dict[int, np.ndarray],
) -> np.ndarray:
    """Apply per-terrain correction to all cells of that type."""
    if not corrections:
        return predictions
    result = predictions.copy()
    for t_val, correction in corrections.items():
        mask = grid == t_val
        if mask.any():
            result[mask] += correction
    # Floor and renormalize
    result = np.maximum(result, PROBABILITY_FLOOR)
    result = result / result.sum(axis=-1, keepdims=True)
    return result


def _static_mask(grid: np.ndarray) -> np.ndarray:
    """Return mask of static terrain (ocean, mountain)."""
    return (grid == InternalTerrain.OCEAN) | (grid == InternalTerrain.MOUNTAIN)
