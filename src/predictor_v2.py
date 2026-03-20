"""Prior-based predictor using terrain priors and settlement distance.

Replaces Monte Carlo simulation with precomputed terrain type priors
and settlement distance adjustments. Runs in <100ms for 40x40 grids.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

from src.constants import (
    NUM_PREDICTION_CLASSES,
    OBS_CONFIDENCE_K,
    PROBABILITY_FLOOR,
    STATIC_TERRAIN_CONFIDENCE,
)
from src.features import compute_settlement_distance
from src.terrain import InternalTerrain

logger = logging.getLogger(__name__)

_MAX_OBS_WEIGHT = 0.8
_MAX_SETTLEMENT_DISTANCE = 5
_DISTANCE_BLEND_WEIGHT = 0.4

# Flat terrain priors from aggregate R1+R2 ground truth analysis
DEFAULT_TERRAIN_PRIORS: dict[int, list[float]] = {
    0: [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    1: [0.756, 0.173, 0.015, 0.016, 0.041, 0.0],
    2: [0.377, 0.410, 0.006, 0.033, 0.174, 0.0],
    3: [0.376, 0.121, 0.300, 0.029, 0.174, 0.0],
    4: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    5: [0.089, 0.177, 0.014, 0.015, 0.705, 0.0],
    6: [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
}

_SERVER_TO_INTERNAL = {0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6, 10: 0, 11: 1}


class PriorPredictor:
    """Predict using terrain priors + distance, no MC simulation."""

    def __init__(
        self,
        grid: np.ndarray,
        settlements: list[dict[str, Any]],
        observation_store: Any | None = None,
        priors: dict[int, list[float]] | None = None,
    ) -> None:
        self._grid = grid
        self._settlements = settlements
        self._obs_store = observation_store
        self._priors = priors or DEFAULT_TERRAIN_PRIORS
        self._height, self._width = grid.shape
        self._dist_map = compute_settlement_distance(grid)

    def predict(self, seed_index: int = 0) -> np.ndarray:
        """Build H x W x 6 probability tensor from priors."""
        tensor = self._apply_base_priors()
        tensor = self._adjust_for_distance(tensor)
        tensor = self._blend_observations(tensor, seed_index)
        tensor = _apply_static_overrides(tensor, self._grid)
        tensor = _floor_and_normalize(tensor)
        return tensor

    def _apply_base_priors(self) -> np.ndarray:
        """Create base tensor from terrain type priors."""
        tensor = np.zeros(
            (self._height, self._width, NUM_PREDICTION_CLASSES),
            dtype=np.float64,
        )
        for terrain_val, prior_vec in self._priors.items():
            mask = self._grid == terrain_val
            if mask.any():
                tensor[mask] = prior_vec
        return tensor

    def _adjust_for_distance(self, tensor: np.ndarray) -> np.ndarray:
        """Adjust probabilities based on settlement proximity."""
        near = self._dist_map <= _MAX_SETTLEMENT_DISTANCE
        static = _static_terrain_mask(self._grid)
        adjustable = near & ~static
        if not adjustable.any():
            return tensor
        return _apply_distance_blend(
            tensor,
            adjustable,
            self._dist_map[adjustable],
        )

    def _blend_observations(
        self,
        tensor: np.ndarray,
        seed_index: int,
    ) -> np.ndarray:
        """Blend observation data with count-scaled weights."""
        if self._obs_store is None:
            return tensor
        obs_probs = self._obs_store.get_observed_probs(seed_index)
        coverage = self._obs_store.get_coverage_mask(seed_index)
        obs_counts = self._obs_store.observation_count(seed_index)
        observed = coverage & ~np.isnan(obs_probs[:, :, 0])
        if not observed.any():
            return tensor
        counts = obs_counts[observed].astype(np.float64)
        w_obs = _MAX_OBS_WEIGHT * counts / (counts + OBS_CONFIDENCE_K)
        w_obs = w_obs[:, np.newaxis]
        result = tensor.copy()
        blended = w_obs * obs_probs[observed] + (1.0 - w_obs) * tensor[observed]
        result[observed] = blended
        return result

    def _get_coverage_pct(self, seed_index: int) -> float:
        """Return observation coverage as a percentage."""
        if self._obs_store is None:
            return 0.0
        return self._obs_store.coverage_fraction(seed_index) * 100


def _apply_distance_blend(
    tensor: np.ndarray,
    mask: np.ndarray,
    distances: np.ndarray,
) -> np.ndarray:
    """Blend settlement-proximity adjustment into the tensor."""
    result = tensor.copy()
    base = result[mask]
    safe_dist = np.maximum(distances, 0.5)
    factor = np.clip(1.0 / safe_dist, 0.0, 1.0)
    w = (_DISTANCE_BLEND_WEIGHT * factor)[:, np.newaxis]
    settle_boost = np.zeros_like(base)
    settle_boost[:, 0] = 0.5
    settle_boost[:, 1] = 0.3
    settle_boost[:, 4] = 0.15
    settle_boost[:, 3] = 0.03
    settle_boost[:, 2] = 0.02
    result[mask] = (1.0 - w) * base + w * settle_boost
    return result


def _static_terrain_mask(grid: np.ndarray) -> np.ndarray:
    """Return mask of cells that are static (ocean, mountain)."""
    ocean = grid == InternalTerrain.OCEAN
    mountain = grid == InternalTerrain.MOUNTAIN
    return ocean | mountain


def _apply_static_overrides(
    tensor: np.ndarray,
    grid: np.ndarray,
) -> np.ndarray:
    """Override ocean and mountain cells with near-certain probs."""
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


def _floor_and_normalize(tensor: np.ndarray) -> np.ndarray:
    """Apply probability floor and renormalize rows to sum to 1."""
    safe = np.maximum(tensor, PROBABILITY_FLOOR)
    return safe / safe.sum(axis=2, keepdims=True)


# ---------------------------------------------------------------------------
# Prior builder from historical data
# ---------------------------------------------------------------------------


def build_priors_from_rounds(
    data_dir: str | Path = "data/rounds",
) -> dict[int, list[float]]:
    """Build terrain priors from captured round data."""
    data_path = Path(data_dir)
    accum: dict[int, dict[str, Any]] = {}
    for t in range(7):
        accum[t] = {"sum": np.zeros(NUM_PREDICTION_CLASSES), "count": 0}
    for round_dir in sorted(data_path.iterdir()):
        if round_dir.is_dir():
            _accumulate_round(round_dir, accum)
    return _finalize_priors(accum)


def _accumulate_round(
    round_dir: Path,
    accum: dict[int, dict[str, Any]],
) -> None:
    """Accumulate GT distributions from one round directory."""
    rjson = round_dir / "round.json"
    if not rjson.exists():
        return
    with open(rjson) as f:
        rd = json.load(f)
    states = rd.get("initial_states", [])
    for seed_idx in range(len(states)):
        _accumulate_seed(round_dir, rd, seed_idx, accum)


def _accumulate_seed(
    round_dir: Path,
    rd: dict[str, Any],
    seed_idx: int,
    accum: dict[int, dict[str, Any]],
) -> None:
    """Accumulate GT data from one seed within a round."""
    gt_path = round_dir / f"seed_{seed_idx}" / "ground_truth.npy"
    if not gt_path.exists():
        return
    gt = np.load(gt_path)
    grid_raw = np.array(rd["initial_states"][seed_idx]["grid"])
    internal = np.vectorize(lambda v: _SERVER_TO_INTERNAL.get(v, 1))(grid_raw)
    for t_val in range(7):
        mask = internal == t_val
        n = int(mask.sum())
        if n > 0:
            accum[t_val]["sum"] += gt[mask].sum(axis=0)
            accum[t_val]["count"] += n


def _finalize_priors(
    accum: dict[int, dict[str, Any]],
) -> dict[int, list[float]]:
    """Convert accumulated counts to normalized probability vectors."""
    uniform = [1.0 / NUM_PREDICTION_CLASSES] * NUM_PREDICTION_CLASSES
    priors: dict[int, list[float]] = {}
    for t_val in range(7):
        count = accum[t_val]["count"]
        if count > 0:
            priors[t_val] = (accum[t_val]["sum"] / count).tolist()
        else:
            priors[t_val] = uniform
    return priors
