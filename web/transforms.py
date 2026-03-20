"""Parametric post-processing transforms for prediction tensors.

Each transform takes an H x W x C prediction tensor (and optional params),
modifies it, and returns a valid probability distribution. Transforms are
composable: apply them in sequence to build complex strategies from a base.
"""

from __future__ import annotations

import logging

import numpy as np

from src.constants import PROBABILITY_FLOOR

logger = logging.getLogger(__name__)


def floor_and_normalize(pred: np.ndarray) -> np.ndarray:
    """Apply probability floor and renormalize to sum to 1."""
    safe = np.maximum(pred, PROBABILITY_FLOOR)
    return safe / safe.sum(axis=-1, keepdims=True)


def temperature_scale(pred: np.ndarray, temperature: float) -> np.ndarray:
    """Scale logits by temperature. T<1 sharpens, T>1 smooths.

    Args:
        pred: H x W x C probability tensor.
        temperature: Scaling factor. Must be positive.

    Returns:
        H x W x C renormalized tensor.
    """
    logits = np.log(np.maximum(pred, 1e-10))
    scaled = logits / max(temperature, 1e-6)
    shifted = scaled - scaled.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return floor_and_normalize(exp / exp.sum(axis=-1, keepdims=True))


def power_transform(pred: np.ndarray, power: float) -> np.ndarray:
    """Raise probabilities to a power and renormalize.

    power > 1 sharpens (high probs get higher), power < 1 smooths.

    Args:
        pred: H x W x C probability tensor.
        power: Exponent to apply.

    Returns:
        H x W x C renormalized tensor.
    """
    powered = np.power(np.maximum(pred, 1e-10), power)
    return floor_and_normalize(powered / powered.sum(axis=-1, keepdims=True))


def spatial_smooth(pred: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian blur each class channel independently.

    Args:
        pred: H x W x C probability tensor.
        sigma: Standard deviation for Gaussian kernel.

    Returns:
        H x W x C renormalized tensor.
    """
    from scipy.ndimage import gaussian_filter

    result = np.zeros_like(pred)
    for c in range(pred.shape[-1]):
        result[:, :, c] = gaussian_filter(pred[:, :, c], sigma=sigma)
    return floor_and_normalize(result)


def settlement_boost(
    pred: np.ndarray,
    grid: np.ndarray,
    factor: float,
) -> np.ndarray:
    """Boost settlement probability for cells near initial settlements.

    Args:
        pred: H x W x C probability tensor.
        grid: H x W InternalTerrain grid.
        factor: Boost magnitude.

    Returns:
        H x W x C renormalized tensor.
    """
    from src.features import compute_settlement_distance

    dist = compute_settlement_distance(grid)
    boost = np.zeros_like(pred)
    boost[:, :, 1] = factor * np.exp(-dist / 3.0)
    result = pred + boost
    return floor_and_normalize(result)


def collapse_shift(pred: np.ndarray, threshold: float) -> np.ndarray:
    """Shift settlement probability to empty for cells below threshold.

    Args:
        pred: H x W x C probability tensor.
        threshold: Minimum settlement probability to keep.

    Returns:
        H x W x C renormalized tensor.
    """
    result = pred.copy()
    low_settle = result[:, :, 1] < threshold
    result[low_settle, 0] += result[low_settle, 1]
    result[low_settle, 1] = PROBABILITY_FLOOR
    return floor_and_normalize(result)


def inland_power(
    pred: np.ndarray,
    grid: np.ndarray,
    power: float,
) -> np.ndarray:
    """Apply power transform only to non-coastal cells.

    Args:
        pred: H x W x C probability tensor.
        grid: H x W InternalTerrain grid.
        power: Exponent for inland cells.

    Returns:
        H x W x C renormalized tensor.
    """
    from src.features import compute_settlement_distance

    dist = compute_settlement_distance(grid)
    result = pred.copy()
    inland = dist > 5
    if inland.any():
        powered = np.power(np.maximum(result[inland], 1e-10), power)
        result[inland] = powered / powered.sum(axis=-1, keepdims=True)
    return floor_and_normalize(result)


def port_smooth(
    pred: np.ndarray,
    grid: np.ndarray,
    weight: float,
) -> np.ndarray:
    """Boost port class probability near initial port cells.

    Args:
        pred: H x W x C probability tensor.
        grid: H x W InternalTerrain grid.
        weight: Boost magnitude for port class.

    Returns:
        H x W x C renormalized tensor.
    """
    from src.terrain import InternalTerrain

    port_mask = grid == InternalTerrain.PORT
    if not port_mask.any():
        return pred
    result = pred.copy()
    result[port_mask, 2] += weight
    return floor_and_normalize(result)


def class_bias(pred: np.ndarray, class_idx: int, delta: float) -> np.ndarray:
    """Add a fixed delta to one class across all cells.

    Args:
        pred: H x W x C probability tensor.
        class_idx: Which class to boost (0-5).
        delta: Amount to add (can be negative).

    Returns:
        H x W x C renormalized tensor.
    """
    result = pred.copy()
    result[:, :, class_idx] += delta
    result = np.maximum(result, PROBABILITY_FLOOR)
    return floor_and_normalize(result)


# ---------------------------------------------------------------------------
# Transform registry: maps name -> callable
# ---------------------------------------------------------------------------

TRANSFORMS: dict[str, callable] = {
    "temperature_scale": temperature_scale,
    "power_transform": power_transform,
    "spatial_smooth": spatial_smooth,
    "settlement_boost": settlement_boost,
    "collapse_shift": collapse_shift,
    "inland_power": inland_power,
    "port_smooth": port_smooth,
    "class_bias": class_bias,
}

# Transforms that need the grid as second argument
GRID_TRANSFORMS: set[str] = {
    "settlement_boost",
    "inland_power",
    "port_smooth",
}


def apply_transform(
    name: str,
    pred: np.ndarray,
    grid: np.ndarray | None,
    params: dict,
) -> np.ndarray:
    """Apply a named transform with parameters.

    Args:
        name: Transform name from TRANSFORMS registry.
        pred: H x W x C prediction tensor.
        grid: H x W grid (required for grid-aware transforms).
        params: Keyword arguments for the transform function.

    Returns:
        Transformed H x W x C tensor.

    Raises:
        KeyError: If transform name not found.
    """
    fn = TRANSFORMS[name]
    if name in GRID_TRANSFORMS:
        if grid is None:
            msg = f"Transform {name} requires grid argument"
            raise ValueError(msg)
        return fn(pred, grid, **params)
    return fn(pred, **params)


def apply_transform_chain(
    pred: np.ndarray,
    grid: np.ndarray | None,
    transforms: list[tuple[str, dict]],
) -> np.ndarray:
    """Apply a sequence of transforms to a prediction tensor.

    Args:
        pred: H x W x C base prediction.
        grid: H x W grid for grid-aware transforms.
        transforms: List of (transform_name, params_dict) tuples.

    Returns:
        Final H x W x C tensor after all transforms.
    """
    result = pred
    for name, params in transforms:
        result = apply_transform(name, result, grid, params)
    return result
