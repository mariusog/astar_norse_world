"""Feature-based per-cell predictor using terrain, distance, and density.

Replaces flat terrain priors with a (terrain_type, distance_bin,
settlement_density_bin) lookup table learned from historical data.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from src.constants import (
    DIST_BIN_EDGES,
    FEATURE_PROB_FLOOR,
    NUM_PREDICTION_CLASSES,
    SETTLEMENT_DENSITY_MAX_BIN,
    SETTLEMENT_DENSITY_WINDOW,
)
from src.features import compute_settlement_density, compute_settlement_distance
from src.terrain import InternalTerrain

logger = logging.getLogger(__name__)

# Type alias for the lookup table
FeatureLookup = dict[tuple[int, int, int], np.ndarray]

# Static overrides for immutable terrain
_OCEAN_PRIOR = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
_MOUNTAIN_PRIOR = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0])


def build_feature_lookup(
    data_dir: str | Path,
    regime_weights: dict[int, float] | None = None,
) -> FeatureLookup:
    """Build lookup table from historical round data.

    Args:
        data_dir: Path to data/rounds directory.
        regime_weights: Optional round_number -> weight mapping.

    Returns:
        Dict mapping (terrain, dist_bin, density_bin) -> prob vector.
    """
    data_path = Path(data_dir)
    accum: dict[tuple[int, int, int], dict[str, object]] = {}
    for round_dir in sorted(data_path.iterdir()):
        if not round_dir.is_dir():
            continue
        _process_round_dir(round_dir, accum, regime_weights)
    return _finalize_lookup(accum)


def predict_from_features(
    grid: np.ndarray,
    lookup: FeatureLookup,
) -> np.ndarray:
    """Predict probability tensor using feature lookup.

    Args:
        grid: H x W array of InternalTerrain values.
        lookup: Feature lookup table from build_feature_lookup.

    Returns:
        H x W x 6 probability tensor.
    """
    height, width = grid.shape
    dist_map = compute_settlement_distance(grid)
    density_map = compute_settlement_density(grid, window=SETTLEMENT_DENSITY_WINDOW)
    dist_bins = _digitize_distances(dist_map)
    density_bins = _digitize_density(density_map)
    tensor = np.full(
        (height, width, NUM_PREDICTION_CLASSES),
        1.0 / NUM_PREDICTION_CLASSES,
        dtype=np.float64,
    )
    _fill_tensor(tensor, grid, dist_bins, density_bins, lookup)
    _apply_static_overrides(tensor, grid)
    _floor_and_normalize(tensor)
    return tensor


def _process_round_dir(
    round_dir: Path,
    accum: dict[tuple[int, int, int], dict[str, object]],
    regime_weights: dict[int, float] | None,
) -> None:
    """Process all seeds in a round directory."""
    rjson = round_dir / "round.json"
    if not rjson.exists():
        return
    with open(rjson) as f:
        rd = json.load(f)
    round_num = rd.get("round_number", 0)
    weight = 1.0
    if regime_weights is not None:
        weight = regime_weights.get(round_num, 1.0)
    for seed_idx in range(len(rd.get("initial_states", []))):
        _process_seed(round_dir, rd, seed_idx, weight, accum)


def _process_seed(
    round_dir: Path,
    rd: dict,
    seed_idx: int,
    weight: float,
    accum: dict[tuple[int, int, int], dict[str, object]],
) -> None:
    """Process one seed: compute features and accumulate GT."""
    gt_path = round_dir / f"seed_{seed_idx}" / "ground_truth.npy"
    ig_path = round_dir / f"seed_{seed_idx}" / "initial_grid.npy"
    if not gt_path.exists() or not ig_path.exists():
        return
    gt = np.load(gt_path)
    grid = np.load(ig_path)
    dist_map = compute_settlement_distance(grid)
    density_map = compute_settlement_density(grid, window=SETTLEMENT_DENSITY_WINDOW)
    dist_bins = _digitize_distances(dist_map)
    density_bins = _digitize_density(density_map)
    _accumulate(grid, gt, dist_bins, density_bins, weight, accum)


def _digitize_distances(dist_map: np.ndarray) -> np.ndarray:
    """Bin settlement distances using DIST_BIN_EDGES."""
    return np.digitize(dist_map, DIST_BIN_EDGES[1:]).astype(np.int32)


def _digitize_density(density_map: np.ndarray) -> np.ndarray:
    """Bin density values, capping at SETTLEMENT_DENSITY_MAX_BIN."""
    return np.minimum(density_map, SETTLEMENT_DENSITY_MAX_BIN).astype(np.int32)


def _accumulate(
    grid: np.ndarray,
    gt: np.ndarray,
    dist_bins: np.ndarray,
    density_bins: np.ndarray,
    weight: float,
    accum: dict[tuple[int, int, int], dict[str, object]],
) -> None:
    """Accumulate weighted GT vectors by feature key."""
    height, width = grid.shape
    for y in range(height):
        for x in range(width):
            key = (int(grid[y, x]), int(dist_bins[y, x]), int(density_bins[y, x]))
            if key not in accum:
                accum[key] = {
                    "sum": np.zeros(NUM_PREDICTION_CLASSES),
                    "weight": 0.0,
                }
            accum[key]["sum"] += gt[y, x] * weight  # type: ignore[union-attr]
            accum[key]["weight"] += weight  # type: ignore[operator]


def _finalize_lookup(
    accum: dict[tuple[int, int, int], dict[str, object]],
) -> FeatureLookup:
    """Normalize accumulated sums to probability vectors."""
    lookup: FeatureLookup = {}
    for key, data in accum.items():
        w = data["weight"]
        if w > 0:  # type: ignore[operator]
            vec = data["sum"] / w  # type: ignore[operator]
            lookup[key] = vec
    logger.info("Feature lookup: %d unique keys", len(lookup))
    return lookup


def _fill_tensor(
    tensor: np.ndarray,
    grid: np.ndarray,
    dist_bins: np.ndarray,
    density_bins: np.ndarray,
    lookup: FeatureLookup,
) -> None:
    """Fill tensor from lookup with cascading fallback."""
    height, width = grid.shape
    for y in range(height):
        for x in range(width):
            vec = _lookup_with_fallback(
                int(grid[y, x]),
                int(dist_bins[y, x]),
                int(density_bins[y, x]),
                lookup,
            )
            if vec is not None:
                tensor[y, x] = vec


def _lookup_with_fallback(
    terrain: int,
    dist_bin: int,
    density_bin: int,
    lookup: FeatureLookup,
) -> np.ndarray | None:
    """Look up with fallback: exact -> drop density -> drop distance."""
    exact = lookup.get((terrain, dist_bin, density_bin))
    if exact is not None:
        return exact
    no_density = lookup.get((terrain, dist_bin, -1))
    if no_density is not None:
        return no_density
    # Aggregate over all density bins for this terrain+dist
    return _aggregate_fallback_density(terrain, dist_bin, lookup)


def _aggregate_fallback_density(
    terrain: int,
    dist_bin: int,
    lookup: FeatureLookup,
) -> np.ndarray | None:
    """Average across density bins for given terrain and dist_bin."""
    total = np.zeros(NUM_PREDICTION_CLASSES)
    count = 0
    for (t, d, _den), vec in lookup.items():
        if t == terrain and d == dist_bin:
            total += vec
            count += 1
    if count == 0:
        return _aggregate_fallback_dist(terrain, lookup)
    return total / count


def _aggregate_fallback_dist(
    terrain: int,
    lookup: FeatureLookup,
) -> np.ndarray | None:
    """Average across all bins for given terrain type."""
    total = np.zeros(NUM_PREDICTION_CLASSES)
    count = 0
    for (t, _d, _den), vec in lookup.items():
        if t == terrain:
            total += vec
            count += 1
    if count == 0:
        return None
    return total / count


def _apply_static_overrides(
    tensor: np.ndarray,
    grid: np.ndarray,
) -> None:
    """Override ocean and mountain cells with deterministic priors."""
    ocean_mask = grid == InternalTerrain.OCEAN
    mountain_mask = grid == InternalTerrain.MOUNTAIN
    tensor[ocean_mask] = _OCEAN_PRIOR
    tensor[mountain_mask] = _MOUNTAIN_PRIOR


def _floor_and_normalize(tensor: np.ndarray) -> None:
    """Apply probability floor and renormalize each cell."""
    tensor[:] = np.maximum(tensor, FEATURE_PROB_FLOOR)
    sums = tensor.sum(axis=2, keepdims=True)
    tensor[:] = tensor / sums


class FeatureGridPredictor:
    """Adapter wrapping feature lookup as GridPredictor."""

    def __init__(self, lookup: FeatureLookup) -> None:
        self._lookup = lookup

    def predict_grid(self, grid: np.ndarray) -> np.ndarray:
        """Predict H x W x 6 probability tensor."""
        return predict_from_features(grid, self._lookup)
