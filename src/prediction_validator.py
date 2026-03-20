"""Pre-submission validation for prediction arrays.

Catches common bugs that caused bad scores in past rounds:
- R2: Laplace smoothing made predictions near-uniform
- R5: Regime detection misclassified survive as collapse
- R6: unified_priors returned uniform 1/6 priors due to wrong data path
"""

from __future__ import annotations

import logging

import numpy as np

from src.constants import (
    NUM_PREDICTION_CLASSES,
    PROBABILITY_FLOOR,
)
from src.terrain import InternalTerrain, Terrain

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Validation thresholds (referenced from constants where possible)
# ---------------------------------------------------------------------------

NORMALIZATION_TOLERANCE = 0.01
MIN_MAX_PROB_THRESHOLD = 0.3
MIN_CONFIDENT_CELL_FRACTION = 0.90
STATIC_TERRAIN_MIN_PROB = 0.9


def validate_predictions(
    predictions: list[np.ndarray],
    grids: list[np.ndarray],
) -> list[str]:
    """Run sanity checks on predictions before submission.

    Args:
        predictions: list of 5 H x W x 6 tensors (one per seed).
        grids: list of 5 H x W initial grids (InternalTerrain values).

    Returns:
        List of error messages. Empty list means all checks passed.
    """
    errors: list[str] = []
    errors.extend(_check_count_and_shapes(predictions, grids))
    if errors:
        return errors

    for i, (pred, grid) in enumerate(zip(predictions, grids, strict=True)):
        errors.extend(_check_single_prediction(pred, grid, seed_idx=i))

    errors.extend(_check_non_trivial(predictions))
    return errors


def _check_count_and_shapes(
    predictions: list[np.ndarray],
    grids: list[np.ndarray],
) -> list[str]:
    """Validate list lengths and array shapes."""
    errors: list[str] = []
    if len(predictions) != len(grids):
        errors.append(f"Prediction count ({len(predictions)}) != grid count ({len(grids)})")
        return errors

    for i, pred in enumerate(predictions):
        if pred.ndim != 3 or pred.shape[2] != NUM_PREDICTION_CLASSES:
            errors.append(
                f"Seed {i}: shape {pred.shape}, expected (H, W, {NUM_PREDICTION_CLASSES})"
            )
        elif i < len(grids):
            h, w = grids[i].shape[:2]
            if pred.shape[0] != h or pred.shape[1] != w:
                errors.append(f"Seed {i}: pred shape {pred.shape[:2]} != grid shape ({h}, {w})")
    return errors


def _check_single_prediction(
    pred: np.ndarray,
    grid: np.ndarray,
    seed_idx: int,
) -> list[str]:
    """Run per-seed checks on one prediction tensor."""
    errors: list[str] = []
    errors.extend(_check_normalization(pred, seed_idx))
    errors.extend(_check_no_uniform(pred, seed_idx))
    errors.extend(_check_probability_floor(pred, seed_idx))
    errors.extend(_check_static_terrain(pred, grid, seed_idx))
    return errors


def _check_normalization(pred: np.ndarray, seed_idx: int) -> list[str]:
    """Each cell's probabilities must sum to 1.0 within tolerance."""
    sums = pred.sum(axis=2)
    bad_mask = np.abs(sums - 1.0) > NORMALIZATION_TOLERANCE
    bad_count = int(bad_mask.sum())
    if bad_count > 0:
        worst = float(np.max(np.abs(sums - 1.0)))
        return [f"Seed {seed_idx}: {bad_count} cells not normalized (worst deviation: {worst:.4f})"]
    return []


def _check_no_uniform(pred: np.ndarray, seed_idx: int) -> list[str]:
    """At least 90% of cells should have max prob > 0.3."""
    max_probs = pred.max(axis=2)
    confident = float((max_probs > MIN_MAX_PROB_THRESHOLD).mean())
    if confident < MIN_CONFIDENT_CELL_FRACTION:
        return [
            f"Seed {seed_idx}: only {confident:.1%} cells have max prob > "
            f"{MIN_MAX_PROB_THRESHOLD} (need {MIN_CONFIDENT_CELL_FRACTION:.0%}). "
            f"Predictions may be near-uniform."
        ]
    return []


def _check_probability_floor(pred: np.ndarray, seed_idx: int) -> list[str]:
    """No probability value should be below the floor minus tolerance."""
    floor_threshold = PROBABILITY_FLOOR - 0.001
    below = float((pred < floor_threshold).sum())
    if below > 0:
        min_val = float(pred.min())
        return [
            f"Seed {seed_idx}: {int(below)} values below floor "
            f"{PROBABILITY_FLOOR} (min: {min_val:.6f})"
        ]
    return []


def _check_static_terrain(
    pred: np.ndarray,
    grid: np.ndarray,
    seed_idx: int,
) -> list[str]:
    """Ocean cells should predict class 0, mountain cells class 5."""
    errors: list[str] = []
    ocean_mask = grid == InternalTerrain.OCEAN
    mountain_mask = grid == InternalTerrain.MOUNTAIN

    if ocean_mask.any():
        ocean_probs = pred[ocean_mask, Terrain.EMPTY]
        bad_ocean = int((ocean_probs < STATIC_TERRAIN_MIN_PROB).sum())
        if bad_ocean > 0:
            return [
                f"Seed {seed_idx}: {bad_ocean} ocean cells have "
                f"class 0 prob < {STATIC_TERRAIN_MIN_PROB}"
            ]

    if mountain_mask.any():
        mtn_probs = pred[mountain_mask, Terrain.MOUNTAIN]
        bad_mtn = int((mtn_probs < STATIC_TERRAIN_MIN_PROB).sum())
        if bad_mtn > 0:
            errors.append(
                f"Seed {seed_idx}: {bad_mtn} mountain cells have "
                f"class 5 prob < {STATIC_TERRAIN_MIN_PROB}"
            )
    return errors


def _check_non_trivial(predictions: list[np.ndarray]) -> list[str]:
    """Predictions should differ across seeds (not identical copies)."""
    if len(predictions) < 2:
        return []
    ref = predictions[0]
    all_identical = all(np.allclose(ref, p, atol=1e-8) for p in predictions[1:])
    if all_identical:
        return ["All seed predictions are identical -- likely a bug"]
    return []
