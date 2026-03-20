"""Multi-round terrain prior builder.

Aggregates ground truth data across historical rounds to build
per-terrain-type probability priors. Recent rounds are weighted
more heavily via exponential decay.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from src.constants import NUM_PREDICTION_CLASSES, NUM_SEEDS, PROBABILITY_FLOOR
from src.terrain import InternalTerrain

logger = logging.getLogger(__name__)

# Exponential decay rate for round weighting (higher = more recent bias)
ROUND_DECAY_RATE = 0.3

# Number of InternalTerrain types (0..6 inclusive)
NUM_INTERNAL_TYPES = 7


def build_terrain_priors(
    data_dir: str | Path,
) -> np.ndarray:
    """Scan all rounds and build GT distribution per initial terrain type.

    For each InternalTerrain type, collects all matching cells' ground
    truth outcome vectors across all rounds and seeds. Recent rounds
    are weighted with exponential decay.

    Args:
        data_dir: Base data directory containing rounds/ subdirectory.

    Returns:
        Array of shape (NUM_INTERNAL_TYPES, NUM_PREDICTION_CLASSES)
        where each row is a normalized probability distribution over
        prediction classes for that initial terrain type.
    """
    data_path = Path(data_dir)
    rounds_dir = data_path / "rounds"

    if not rounds_dir.exists():
        logger.warning("No rounds directory at %s", rounds_dir)
        return _uniform_priors()

    round_ids = _discover_round_ids(rounds_dir)
    if not round_ids:
        logger.warning("No rounds found in %s", rounds_dir)
        return _uniform_priors()

    max_round = max(round_ids)
    counts = np.zeros((NUM_INTERNAL_TYPES, NUM_PREDICTION_CLASSES), dtype=np.float64)

    for round_id in sorted(round_ids):
        weight = _compute_round_weight(round_id, max_round)
        _accumulate_round(rounds_dir, round_id, weight, counts)

    return _normalize_priors(counts)


def _discover_round_ids(rounds_dir: Path) -> list[int]:
    """Find all round IDs that have data directories."""
    ids = []
    for child in rounds_dir.iterdir():
        if child.is_dir() and child.name.isdigit():
            ids.append(int(child.name))
    return ids


def _compute_round_weight(round_id: int, max_round: int) -> float:
    """Exponential decay weight favoring recent rounds."""
    age = max_round - round_id
    return float(np.exp(-ROUND_DECAY_RATE * age))


def _accumulate_round(
    rounds_dir: Path,
    round_id: int,
    weight: float,
    counts: np.ndarray,
) -> None:
    """Add one round's data to the accumulated counts."""
    round_dir = rounds_dir / str(round_id)

    for seed_idx in range(NUM_SEEDS):
        seed_dir = round_dir / f"seed_{seed_idx}"
        initial_path = seed_dir / "initial_grid.npy"
        gt_path = seed_dir / "ground_truth.npy"

        if not initial_path.exists() or not gt_path.exists():
            continue

        initial = np.load(initial_path)
        gt = np.load(gt_path)

        if initial.shape != gt.shape:
            logger.warning(
                "Shape mismatch round %d seed %d: %s vs %s",
                round_id,
                seed_idx,
                initial.shape,
                gt.shape,
            )
            continue

        _accumulate_seed(initial, gt, weight, counts)
        logger.debug("Round %d seed %d: weight=%.3f", round_id, seed_idx, weight)


def _accumulate_seed(
    initial: np.ndarray,
    gt: np.ndarray,
    weight: float,
    counts: np.ndarray,
) -> None:
    """Accumulate weighted counts from one seed's initial/GT pair."""
    pred_map = np.vectorize(_to_prediction_class)(gt)
    for init_type in range(NUM_INTERNAL_TYPES):
        mask = initial == init_type
        if not mask.any():
            continue
        pred_vals = pred_map[mask]
        for pc in range(NUM_PREDICTION_CLASSES):
            counts[init_type, pc] += weight * int((pred_vals == pc).sum())


def _to_prediction_class(internal_val: int) -> int:
    """Map InternalTerrain value to prediction class index."""
    try:
        return int(InternalTerrain(internal_val).to_prediction_class())
    except ValueError:
        return 0  # Default to EMPTY for unknown values


def _normalize_priors(counts: np.ndarray) -> np.ndarray:
    """Normalize counts to probabilities with floor.

    Applies iterative floor-and-renormalize to ensure all values
    stay at or above PROBABILITY_FLOOR after normalization.
    """
    priors = counts.copy()
    for i in range(NUM_INTERNAL_TYPES):
        row_sum = priors[i].sum()
        if row_sum > 0:
            priors[i] /= row_sum
        else:
            priors[i] = 1.0 / NUM_PREDICTION_CLASSES

    # Apply floor: redistribute mass from above-floor to below-floor
    for i in range(NUM_INTERNAL_TYPES):
        priors[i] = _apply_floor_to_row(priors[i])
    return priors


def _apply_floor_to_row(row: np.ndarray) -> np.ndarray:
    """Apply probability floor to a single distribution row.

    Clamps values below floor and redistributes the deficit
    proportionally from above-floor values.
    """
    n = len(row)
    floor = PROBABILITY_FLOOR
    result = row.copy()

    # Iterative clamping (converges in 1-2 passes)
    for _ in range(n):
        below = result < floor
        if not below.any():
            break
        deficit = (floor - result[below]).sum()
        result[below] = floor
        above = ~below
        above_sum = result[above].sum()
        if above_sum > deficit:
            result[above] -= deficit * (result[above] / above_sum)
        else:
            result[:] = 1.0 / n
            break

    # Final safety normalization
    result /= result.sum()
    return result


def _uniform_priors() -> np.ndarray:
    """Return uniform priors when no data is available."""
    return np.full(
        (NUM_INTERNAL_TYPES, NUM_PREDICTION_CLASSES),
        1.0 / NUM_PREDICTION_CLASSES,
        dtype=np.float64,
    )


def save_priors(priors: np.ndarray, path: str | Path) -> None:
    """Persist priors array to disk.

    Args:
        priors: Shape (NUM_INTERNAL_TYPES, NUM_PREDICTION_CLASSES).
        path: File path (saved as .npy).
    """
    np.save(path, priors)
    logger.info("Saved priors to %s", path)


def load_priors(path: str | Path) -> np.ndarray:
    """Load priors array from disk.

    Args:
        path: Path to .npy file.

    Returns:
        Priors array of shape (NUM_INTERNAL_TYPES, NUM_PREDICTION_CLASSES).
    """
    priors = np.load(path)
    logger.info("Loaded priors from %s, shape %s", path, priors.shape)
    return priors


def build_prior_prediction(
    grid: np.ndarray,
    priors: np.ndarray,
) -> np.ndarray:
    """Apply terrain priors to a grid to produce prediction tensor.

    For each cell, looks up the prior distribution for its initial
    terrain type. Applies probability floor and renormalizes.

    Args:
        grid: H x W array of InternalTerrain values.
        priors: Shape (NUM_INTERNAL_TYPES, NUM_PREDICTION_CLASSES).

    Returns:
        H x W x NUM_PREDICTION_CLASSES probability tensor.
    """
    grid_clipped = np.clip(grid.astype(np.int32), 0, priors.shape[0] - 1)
    prediction = priors[grid_clipped].copy()

    # Apply floor per cell using vectorized reshape
    h, w, c = prediction.shape
    flat = prediction.reshape(-1, c)
    for i in range(flat.shape[0]):
        flat[i] = _apply_floor_to_row(flat[i])

    return prediction
