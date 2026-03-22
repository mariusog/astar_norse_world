"""Soft regime blending based on post-observation evidence.

Instead of binary regime detection (survive vs collapse), estimates
a continuous confidence score from observed settlement survival rates
and blends predictions from both regime priors accordingly.

Survive rounds: ~33% of initial settlement cells remain settlement/port.
Collapse rounds: ~0% of initial settlement cells remain settlement/port.
"""

from __future__ import annotations

import logging

import numpy as np

from src.observation import ObservationStore
from src.terrain import Terrain

logger = logging.getLogger(__name__)

# Threshold for normalizing settlement survival rate to confidence.
# Survive rounds have ~33% survival; dividing by 0.25 maps that to ~1.0.
SETTLEMENT_SURVIVAL_NORMALIZER = 0.25

# Prediction classes considered "settlement-like" (Settlement, Port)
SETTLEMENT_CLASSES = frozenset({int(Terrain.SETTLEMENT), int(Terrain.PORT)})

# InternalTerrain values for initial settlement/port cells
INITIAL_SETTLEMENT_CODE = 2  # InternalTerrain.SETTLEMENT
INITIAL_PORT_CODE = 3  # InternalTerrain.PORT


def estimate_regime_confidence(
    grids: list[np.ndarray],
    obs_stores: list[ObservationStore],
) -> float:
    """Estimate survive confidence from observed settlement survival.

    Counts how many initial settlement/port cells were observed as
    still being settlement/port (prediction class 1 or 2).

    Args:
        grids: Initial grids per seed (InternalTerrain values).
        obs_stores: Observation stores per seed.

    Returns:
        Float in [0.0, 1.0]. 1.0 = certainly survive, 0.0 = certainly collapse.
    """
    total_checked = 0
    total_survived = 0

    for seed_idx, grid in enumerate(grids):
        store = obs_stores[seed_idx]
        checked, survived = _count_settlement_survival(grid, store, seed_idx)
        total_checked += checked
        total_survived += survived

    if total_checked == 0:
        logger.warning("No settlement cells observed; defaulting confidence=0.6")
        return 0.6

    survival_rate = total_survived / total_checked
    confidence = min(1.0, survival_rate / SETTLEMENT_SURVIVAL_NORMALIZER)
    logger.info(
        "Regime confidence: %.2f (rate=%.3f, checked=%d, survived=%d)",
        confidence,
        survival_rate,
        total_checked,
        total_survived,
    )
    return confidence


def _count_settlement_survival(
    grid: np.ndarray,
    store: ObservationStore,
    seed_idx: int,
) -> tuple[int, int]:
    """Count observed settlement cells and how many survived.

    Args:
        grid: H x W InternalTerrain grid.
        store: Observation store for this seed.
        seed_idx: Seed index in the store.

    Returns:
        Tuple of (checked_count, survived_count).
    """
    coverage = store.get_coverage_mask(seed_idx)
    initial_settlement = _is_initial_settlement(grid)
    observable = initial_settlement & coverage
    checked = int(observable.sum())
    if checked == 0:
        return 0, 0

    obs_probs = store.get_observed_probs(seed_idx)
    survived = _count_observed_settlements(obs_probs, observable)
    return checked, survived


def _is_initial_settlement(grid: np.ndarray) -> np.ndarray:
    """Return boolean mask of cells that were initially settlement or port."""
    return (grid == INITIAL_SETTLEMENT_CODE) | (grid == INITIAL_PORT_CODE)


def _count_observed_settlements(
    obs_probs: np.ndarray,
    mask: np.ndarray,
) -> int:
    """Count cells in mask where the most likely class is settlement/port."""
    argmax_classes = np.argmax(obs_probs[mask], axis=1)
    settlement_mask = np.isin(argmax_classes, list(SETTLEMENT_CLASSES))
    return int(settlement_mask.sum())


def soft_blend_predictions(
    survive_pred: np.ndarray,
    collapse_pred: np.ndarray,
    confidence: float,
    coverage_mask: np.ndarray,
) -> np.ndarray:
    """Blend survive and collapse predictions based on regime confidence.

    Observed cells keep the survive prediction (observations already
    reflect reality). Unobserved cells get soft-blended between regimes.

    Args:
        survive_pred: H x W x C tensor from survive priors.
        collapse_pred: H x W x C tensor from collapse priors.
        confidence: Survive confidence in [0.0, 1.0].
        coverage_mask: H x W bool, True where observed.

    Returns:
        H x W x C blended prediction tensor.
    """
    result = survive_pred.copy()
    unobserved = ~coverage_mask
    result[unobserved] = (
        confidence * survive_pred[unobserved] + (1.0 - confidence) * collapse_pred[unobserved]
    )
    return result


def build_regime_predictions(
    grid: np.ndarray,
    survive_priors: np.ndarray,
    collapse_priors: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Build H x W x C prediction tensors from per-terrain-type priors.

    Each prior maps InternalTerrain -> probability vector of shape (C,).
    Looks up each cell's initial terrain type and assigns the prior.

    Args:
        grid: H x W InternalTerrain grid.
        survive_priors: Shape (num_types, C) survive regime priors.
        collapse_priors: Shape (num_types, C) collapse regime priors.

    Returns:
        Tuple of (survive_pred, collapse_pred), each H x W x C.
    """
    grid_clipped = np.clip(grid.astype(np.int32), 0, survive_priors.shape[0] - 1)
    survive_pred = survive_priors[grid_clipped].copy()
    collapse_pred = collapse_priors[grid_clipped].copy()
    return survive_pred, collapse_pred


def score_soft_blend(
    grid: np.ndarray,
    obs_store: ObservationStore,
    seed_idx: int,
    survive_pred: np.ndarray,
    collapse_pred: np.ndarray,
    confidence: float,
) -> np.ndarray:
    """Convenience: blend predictions for a single seed.

    Args:
        grid: H x W InternalTerrain grid.
        obs_store: Observation store.
        seed_idx: Seed index.
        survive_pred: H x W x C survive predictions.
        collapse_pred: H x W x C collapse predictions.
        confidence: Regime confidence from estimate_regime_confidence.

    Returns:
        H x W x C blended prediction tensor.
    """
    coverage = obs_store.get_coverage_mask(seed_idx)
    return soft_blend_predictions(survive_pred, collapse_pred, confidence, coverage)
