"""Error-correcting post-processing transforms for predictions.

Applies per-(terrain, distance_bin) correction factors learned from
historical ground truth to fix systematic prediction biases.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

from src.constants import NUM_PREDICTION_CLASSES, PROBABILITY_FLOOR
from src.features import compute_settlement_distance
from src.terrain import SERVER_TO_INTERNAL

logger = logging.getLogger(__name__)

# Distance bin edges: [0,2), [2,5), [5,10), [10,inf)
DIST_BIN_EDGES = [0, 2, 5, 10]
NUM_DIST_BINS = len(DIST_BIN_EDGES)
NUM_TERRAIN_TYPES = 7

# Maximum correction magnitude per class to avoid over-correction
MAX_CORRECTION_MAG = 0.15


def error_correction(
    pred: np.ndarray,
    grid: np.ndarray,
    corrections: dict[tuple[int, int], np.ndarray],
) -> np.ndarray:
    """Apply per-(terrain, distance_bin) correction factors.

    For each cell, looks up the correction vector based on its initial
    terrain type and distance bin, adds it to the prediction, then
    renormalizes.

    Args:
        pred: H x W x 6 prediction tensor.
        grid: H x W array of InternalTerrain values.
        corrections: Mapping (terrain_type, dist_bin) -> 6-element
            correction vector (avg GT - avg prediction errors).

    Returns:
        H x W x 6 corrected and renormalized prediction tensor.
    """
    if not corrections:
        return pred
    result = pred.copy()
    dist_map = compute_settlement_distance(grid)
    dist_bins = _assign_distance_bins(dist_map)
    _apply_corrections(result, grid, dist_bins, corrections)
    return _floor_and_normalize(result)


def compute_error_corrections(
    data_dir: str | Path,
    exclude_rounds: set[int] | None = None,
) -> dict[tuple[int, int], np.ndarray]:
    """Compute avg(GT - prediction) per (terrain, distance_bin).

    Trains a simple prior predictor on non-excluded rounds and measures
    the systematic residual for each terrain/distance bin combination.

    Args:
        data_dir: Directory containing round subdirectories.
        exclude_rounds: Round numbers to exclude (for LOO backtest).

    Returns:
        Dict mapping (terrain_type, dist_bin_idx) -> 6-element correction.
    """
    data_path = Path(data_dir)
    exclude = exclude_rounds or set()
    accum = _init_correction_accum()
    priors = _build_simple_priors(data_path, exclude)

    for round_dir in sorted(data_path.iterdir()):
        if not round_dir.is_dir():
            continue
        _accumulate_round_errors(round_dir, exclude, priors, accum)

    return _finalize_corrections(accum)


def _assign_distance_bins(dist_map: np.ndarray) -> np.ndarray:
    """Assign each cell to a distance bin index."""
    bins = np.full_like(dist_map, NUM_DIST_BINS - 1, dtype=np.int32)
    for i in range(len(DIST_BIN_EDGES) - 1, -1, -1):
        bins[dist_map >= DIST_BIN_EDGES[i]] = i
    # Re-assign using proper binning
    bins = np.digitize(dist_map, DIST_BIN_EDGES) - 1
    bins = np.clip(bins, 0, NUM_DIST_BINS - 1)
    return bins


def _apply_corrections(
    result: np.ndarray,
    grid: np.ndarray,
    dist_bins: np.ndarray,
    corrections: dict[tuple[int, int], np.ndarray],
) -> None:
    """Apply correction vectors in-place per terrain/distance combo."""
    for terrain_type in range(NUM_TERRAIN_TYPES):
        t_mask = grid == terrain_type
        if not t_mask.any():
            continue
        for bin_idx in range(NUM_DIST_BINS):
            key = (terrain_type, bin_idx)
            if key not in corrections:
                continue
            combined = t_mask & (dist_bins == bin_idx)
            if not combined.any():
                continue
            result[combined] += corrections[key]


def _init_correction_accum() -> dict[tuple[int, int], dict[str, Any]]:
    """Create empty accumulator for correction vectors."""
    accum: dict[tuple[int, int], dict[str, Any]] = {}
    for t in range(NUM_TERRAIN_TYPES):
        for d in range(NUM_DIST_BINS):
            accum[(t, d)] = {
                "error_sum": np.zeros(NUM_PREDICTION_CLASSES),
                "count": 0,
            }
    return accum


def _build_simple_priors(
    data_path: Path,
    exclude: set[int],
) -> dict[int, np.ndarray]:
    """Build flat terrain priors from non-excluded rounds."""
    sums: dict[int, np.ndarray] = {
        t: np.zeros(NUM_PREDICTION_CLASSES) for t in range(NUM_TERRAIN_TYPES)
    }
    counts: dict[int, int] = {t: 0 for t in range(NUM_TERRAIN_TYPES)}

    for round_dir in sorted(data_path.iterdir()):
        if not round_dir.is_dir():
            continue
        rjson = round_dir / "round.json"
        if not rjson.exists():
            continue
        with open(rjson) as f:
            meta = json.load(f)
        if meta.get("round_number") in exclude:
            continue
        _accumulate_priors_from_round(round_dir, meta, sums, counts)

    priors: dict[int, np.ndarray] = {}
    uniform = np.full(NUM_PREDICTION_CLASSES, 1.0 / NUM_PREDICTION_CLASSES)
    for t in range(NUM_TERRAIN_TYPES):
        if counts[t] > 0:
            priors[t] = sums[t] / counts[t]
        else:
            priors[t] = uniform.copy()
    return priors


def _accumulate_priors_from_round(
    round_dir: Path,
    meta: dict[str, Any],
    sums: dict[int, np.ndarray],
    counts: dict[int, int],
) -> None:
    """Add prior data from one round."""
    states = meta.get("initial_states", [])
    for seed_idx in range(len(states)):
        gt_path = round_dir / f"seed_{seed_idx}" / "ground_truth.npy"
        if not gt_path.exists():
            continue
        gt = np.load(gt_path)
        grid_raw = np.array(states[seed_idx]["grid"])
        internal = _server_grid_to_internal(grid_raw)
        for t in range(NUM_TERRAIN_TYPES):
            mask = internal == t
            n = int(mask.sum())
            if n > 0:
                sums[t] += gt[mask].sum(axis=0)
                counts[t] += n


def _accumulate_round_errors(
    round_dir: Path,
    exclude: set[int],
    priors: dict[int, np.ndarray],
    accum: dict[tuple[int, int], dict[str, Any]],
) -> None:
    """Compute prediction errors for one round and accumulate."""
    rjson = round_dir / "round.json"
    if not rjson.exists():
        return
    with open(rjson) as f:
        meta = json.load(f)
    rn = meta.get("round_number")
    if rn in exclude:
        return
    states = meta.get("initial_states", [])
    for seed_idx in range(len(states)):
        _accumulate_seed_errors(round_dir, states[seed_idx], seed_idx, priors, accum)


def _accumulate_seed_errors(
    round_dir: Path,
    state: dict[str, Any],
    seed_idx: int,
    priors: dict[int, np.ndarray],
    accum: dict[tuple[int, int], dict[str, Any]],
) -> None:
    """Compute errors for one seed and add to accumulator."""
    gt_path = round_dir / f"seed_{seed_idx}" / "ground_truth.npy"
    if not gt_path.exists():
        return
    gt = np.load(gt_path)
    grid_raw = np.array(state["grid"])
    internal = _server_grid_to_internal(grid_raw)
    dist_map = compute_settlement_distance(internal)
    dist_bins = _assign_distance_bins(dist_map)

    # Build prior-based prediction for this seed
    h, w = internal.shape
    pred = np.zeros((h, w, NUM_PREDICTION_CLASSES), dtype=np.float64)
    for t in range(NUM_TERRAIN_TYPES):
        mask = internal == t
        if mask.any():
            pred[mask] = priors[t]

    # Accumulate error = GT - prediction
    for t in range(NUM_TERRAIN_TYPES):
        t_mask = internal == t
        if not t_mask.any():
            continue
        for d in range(NUM_DIST_BINS):
            combined = t_mask & (dist_bins == d)
            n = int(combined.sum())
            if n > 0:
                error = gt[combined] - pred[combined]
                accum[(t, d)]["error_sum"] += error.sum(axis=0)
                accum[(t, d)]["count"] += n


def _finalize_corrections(
    accum: dict[tuple[int, int], dict[str, Any]],
) -> dict[tuple[int, int], np.ndarray]:
    """Convert accumulated errors to clipped correction vectors."""
    corrections: dict[tuple[int, int], np.ndarray] = {}
    for key, data in accum.items():
        if data["count"] == 0:
            continue
        avg_error = data["error_sum"] / data["count"]
        # Clip to prevent over-correction
        avg_error = np.clip(avg_error, -MAX_CORRECTION_MAG, MAX_CORRECTION_MAG)
        # Only store non-trivial corrections
        if np.abs(avg_error).max() > 0.001:
            corrections[key] = avg_error
    return corrections


def _floor_and_normalize(tensor: np.ndarray) -> np.ndarray:
    """Apply probability floor and renormalize."""
    safe = np.maximum(tensor, PROBABILITY_FLOOR)
    return safe / safe.sum(axis=-1, keepdims=True)


def _server_grid_to_internal(grid_raw: np.ndarray) -> np.ndarray:
    """Convert server grid codes to InternalTerrain values."""
    return np.vectorize(lambda v: SERVER_TO_INTERNAL.get(v, 1))(grid_raw).astype(np.int8)


def apply_floor_to_row(row: np.ndarray) -> np.ndarray:
    """Apply probability floor to a single distribution row.

    Clamps values below floor and redistributes the deficit
    proportionally from above-floor values.
    """
    n = len(row)
    floor = PROBABILITY_FLOOR
    result = row.copy()

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

    result /= result.sum()
    return result
