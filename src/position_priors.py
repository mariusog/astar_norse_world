"""Position-aware priors using settlement distance model.

For each terrain type and distance band from settlements, computes
learned probability vectors from historical data. Blends these
distance-conditioned priors with flat terrain priors.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

from src.constants import NUM_PREDICTION_CLASSES
from src.terrain import SERVER_TO_INTERNAL, InternalTerrain

logger = logging.getLogger(__name__)

MAX_DISTANCE_BAND = 5
DISTANCE_MODEL_WEIGHT = 0.3


class SettlementDistanceModel:
    """Learned model: P(class | terrain_type, distance_to_settlement)."""

    def __init__(
        self,
        distance_priors: dict[int, dict[int, list[float]]],
    ) -> None:
        self._priors = distance_priors

    def get_prior(
        self,
        terrain_type: int,
        distance: int,
    ) -> np.ndarray | None:
        """Look up distance-conditioned prior, or None if unavailable."""
        clamped = min(distance, MAX_DISTANCE_BAND)
        terrain_priors = self._priors.get(terrain_type)
        if terrain_priors is None:
            return None
        vec = terrain_priors.get(clamped)
        if vec is None:
            return None
        return np.array(vec, dtype=np.float64)

    @property
    def terrain_types(self) -> list[int]:
        """Return terrain types that have distance data."""
        return list(self._priors.keys())


def predict_from_position(
    grid: np.ndarray,
    settlements: list[dict[str, Any]],
    base_priors: dict[int, list[float]],
    distance_model: SettlementDistanceModel,
    blend_weight: float = DISTANCE_MODEL_WEIGHT,
) -> np.ndarray:
    """Produce H x W x 6 tensor blending flat priors with distance model."""
    height, width = grid.shape
    tensor = _apply_flat_priors(grid, base_priors, height, width)
    dist_map = _compute_distances(grid, settlements, height, width)
    return _blend_distance_model(
        tensor,
        grid,
        dist_map,
        distance_model,
        blend_weight,
    )


def build_distance_model(
    data_dir: str | Path,
) -> SettlementDistanceModel:
    """Build distance model from captured round data."""
    data_path = Path(data_dir)
    accum = _init_accumulator()
    for round_dir in sorted(data_path.iterdir()):
        if round_dir.is_dir():
            _process_round(round_dir, accum)
    priors = _finalize_distance_priors(accum)
    logger.info("Built distance model: %d terrain types", len(priors))
    return SettlementDistanceModel(priors)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _init_accumulator() -> dict[int, dict[int, dict[str, Any]]]:
    """Create empty accumulator for distance-conditioned priors."""
    accum: dict[int, dict[int, dict[str, Any]]] = {}
    for t in range(7):
        accum[t] = {}
        for d in range(1, MAX_DISTANCE_BAND + 1):
            accum[t][d] = {
                "sum": np.zeros(NUM_PREDICTION_CLASSES),
                "count": 0,
            }
    return accum


def _process_round(
    round_dir: Path,
    accum: dict[int, dict[int, dict[str, Any]]],
) -> None:
    """Process all seeds in one round directory."""
    rjson = round_dir / "round.json"
    if not rjson.exists():
        return
    with open(rjson) as f:
        rd = json.load(f)
    states = rd.get("initial_states", [])
    for seed_idx in range(len(states)):
        _process_seed(round_dir, rd, seed_idx, accum)


def _process_seed(
    round_dir: Path,
    rd: dict[str, Any],
    seed_idx: int,
    accum: dict[int, dict[int, dict[str, Any]]],
) -> None:
    """Process one seed's ground truth for distance accumulation."""
    gt_path = round_dir / f"seed_{seed_idx}" / "ground_truth.npy"
    if not gt_path.exists():
        return
    gt = np.load(gt_path)
    grid_raw = np.array(rd["initial_states"][seed_idx]["grid"])
    internal = np.vectorize(lambda v: SERVER_TO_INTERNAL.get(v, 1))(grid_raw)
    height, width = internal.shape
    positions = _extract_settlements(rd["initial_states"][seed_idx], internal)
    dist_map = _compute_distances_from_positions(positions, height, width)
    _accumulate_distance_data(internal, gt, dist_map, accum)


def _extract_settlements(
    state: dict[str, Any],
    grid: np.ndarray,
) -> list[tuple[int, int]]:
    """Get settlement positions from state and grid."""
    positions = [(s["x"], s["y"]) for s in state.get("settlements", [])]
    for row, col in np.argwhere(grid == int(InternalTerrain.SETTLEMENT)):
        positions.append((col, row))
    return positions


def _compute_distances_from_positions(
    positions: list[tuple[int, int]],
    height: int,
    width: int,
) -> np.ndarray:
    """Compute Manhattan distance map from settlement positions."""
    rows, cols = np.mgrid[0:height, 0:width]
    dist_map = np.full((height, width), height + width, dtype=np.float64)
    for sx, sy in positions:
        d = np.abs(cols - sx) + np.abs(rows - sy)
        dist_map = np.minimum(dist_map, d.astype(np.float64))
    return dist_map


def _accumulate_distance_data(
    grid: np.ndarray,
    gt: np.ndarray,
    dist_map: np.ndarray,
    accum: dict[int, dict[int, dict[str, Any]]],
) -> None:
    """Accumulate ground truth by terrain type and distance band."""
    for d in range(1, MAX_DISTANCE_BAND + 1):
        d_mask = (dist_map >= d - 0.5) & (dist_map < d + 0.5)
        for t_val in range(7):
            combined = d_mask & (grid == t_val)
            n = int(combined.sum())
            if n > 0:
                accum[t_val][d]["sum"] += gt[combined].sum(axis=0)
                accum[t_val][d]["count"] += n


def _finalize_distance_priors(
    accum: dict[int, dict[int, dict[str, Any]]],
) -> dict[int, dict[int, list[float]]]:
    """Convert accumulated counts to normalized probability vectors."""
    priors: dict[int, dict[int, list[float]]] = {}
    for t_val in range(7):
        t_priors: dict[int, list[float]] = {}
        for d in range(1, MAX_DISTANCE_BAND + 1):
            count = accum[t_val][d]["count"]
            if count > 0:
                t_priors[d] = (accum[t_val][d]["sum"] / count).tolist()
        if t_priors:
            priors[t_val] = t_priors
    return priors


def _apply_flat_priors(
    grid: np.ndarray,
    base_priors: dict[int, list[float]],
    height: int,
    width: int,
) -> np.ndarray:
    """Create base tensor from flat terrain priors."""
    tensor = np.zeros(
        (height, width, NUM_PREDICTION_CLASSES),
        dtype=np.float64,
    )
    for terrain_val, prior_vec in base_priors.items():
        mask = grid == terrain_val
        if mask.any():
            tensor[mask] = prior_vec
    return tensor


def _compute_distances(
    grid: np.ndarray,
    settlements: list[dict[str, Any]],
    height: int,
    width: int,
) -> np.ndarray:
    """Compute distance map from settlements list and grid."""
    positions = [(s["x"], s["y"]) for s in settlements]
    for row, col in np.argwhere(grid == int(InternalTerrain.SETTLEMENT)):
        positions.append((col, row))
    return _compute_distances_from_positions(positions, height, width)


def _blend_distance_model(
    tensor: np.ndarray,
    grid: np.ndarray,
    dist_map: np.ndarray,
    model: SettlementDistanceModel,
    weight: float,
) -> np.ndarray:
    """Blend distance-conditioned priors into the base tensor."""
    result = tensor.copy()
    height, width = grid.shape
    for row in range(height):
        for col in range(width):
            _blend_cell(result, grid, dist_map, model, weight, row, col)
    return result


def _blend_cell(
    result: np.ndarray,
    grid: np.ndarray,
    dist_map: np.ndarray,
    model: SettlementDistanceModel,
    weight: float,
    row: int,
    col: int,
) -> None:
    """Blend distance prior for a single cell in-place."""
    dist = round(dist_map[row, col])
    if dist < 1 or dist > MAX_DISTANCE_BAND:
        return
    t_val = int(grid[row, col])
    dist_prior = model.get_prior(t_val, dist)
    if dist_prior is None:
        return
    result[row, col] = (1.0 - weight) * result[row, col] + weight * dist_prior
