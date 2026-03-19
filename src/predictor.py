"""Prediction tensor generator combining Monte Carlo sim and observations.

Blends local simulation output with server observations to produce
a final H x W x 6 probability tensor for submission.
"""

from __future__ import annotations

import logging

import numpy as np

from src.constants import (
    DEFAULT_MC_RUNS,
    NUM_PREDICTION_CLASSES,
    OBSERVATION_WEIGHT,
    PROBABILITY_FLOOR,
    SIMULATION_WEIGHT,
    STATIC_TERRAIN_CONFIDENCE,
)
from src.observation import ObservationStore
from src.runner import run_monte_carlo_from_state
from src.settlement import Settlement
from src.terrain import InternalTerrain

logger = logging.getLogger(__name__)


class Predictor:
    """Produce probability tensors by blending sim and observations.

    For observed cells, blends observation probs with sim probs using
    configurable weights (default 0.8 obs / 0.2 sim). For unobserved
    cells, uses sim output only. Static terrain (mountains, ocean)
    gets near-certain probability.

    Args:
        initial_grid: H x W initial terrain grid.
        settlements: List of initial settlements.
        observation_store: Accumulated server observations.
        obs_weight: Weight for server observations in blended cells.
        sim_weight: Weight for local simulation in blended cells.
    """

    def __init__(
        self,
        initial_grid: np.ndarray,
        settlements: list[Settlement],
        observation_store: ObservationStore,
        obs_weight: float = OBSERVATION_WEIGHT,
        sim_weight: float = SIMULATION_WEIGHT,
    ) -> None:
        self._grid = initial_grid
        self._settlements = settlements
        self._obs_store = observation_store
        self._obs_weight = obs_weight
        self._sim_weight = sim_weight
        self._height, self._width = initial_grid.shape

    def predict(
        self,
        seed_index: int,
        num_mc_runs: int = DEFAULT_MC_RUNS,
        map_seed: int = 0,
    ) -> np.ndarray:
        """Produce H x W x 6 probability tensor for one seed.

        Steps:
        1. Run Monte Carlo simulation for prior probabilities
        2. Get observed probabilities from ObservationStore
        3. Blend observed and simulated
        4. Apply static terrain certainty
        5. Apply probability floor and renormalize

        Args:
            seed_index: Which seed to predict.
            num_mc_runs: Number of Monte Carlo runs.
            map_seed: Seed for map generation in Monte Carlo.

        Returns:
            H x W x 6 probability tensor, each cell sums to 1.0.
        """
        sim_probs = self._run_simulation(map_seed, num_mc_runs)
        obs_probs = self._obs_store.get_observed_probs(seed_index)
        coverage = self._obs_store.get_coverage_mask(seed_index)

        blended = _blend_probabilities(
            sim_probs,
            obs_probs,
            coverage,
            self._obs_weight,
            self._sim_weight,
        )
        blended = _apply_static_terrain(blended, self._grid)
        blended = _apply_floor_and_normalize(blended)

        logger.info(
            "Prediction for seed %d: %d MC runs, %.0f%% observed",
            seed_index,
            num_mc_runs,
            float(coverage.sum()) / (self._height * self._width) * 100,
        )
        return blended

    def _run_simulation(
        self,
        map_seed: int,
        num_mc_runs: int,
    ) -> np.ndarray:
        """Run Monte Carlo simulation from the actual initial state."""
        return run_monte_carlo_from_state(
            grid=self._grid,
            settlements=self._settlements,
            num_runs=num_mc_runs,
        )

    def predict_from_sim(
        self,
        sim_probs: np.ndarray,
        seed_index: int,
    ) -> np.ndarray:
        """Produce prediction using pre-computed sim probabilities.

        Use this when you already have simulation results and want
        to avoid re-running Monte Carlo.

        Args:
            sim_probs: H x W x 6 pre-computed simulation probs.
            seed_index: Which seed to predict.

        Returns:
            H x W x 6 probability tensor.
        """
        obs_probs = self._obs_store.get_observed_probs(seed_index)
        coverage = self._obs_store.get_coverage_mask(seed_index)

        blended = _blend_probabilities(
            sim_probs,
            obs_probs,
            coverage,
            self._obs_weight,
            self._sim_weight,
        )
        blended = _apply_static_terrain(blended, self._grid)
        return _apply_floor_and_normalize(blended)


# ---------------------------------------------------------------------------
# Blending helpers
# ---------------------------------------------------------------------------


def _blend_probabilities(
    sim_probs: np.ndarray,
    obs_probs: np.ndarray,
    coverage_mask: np.ndarray,
    obs_weight: float,
    sim_weight: float,
) -> np.ndarray:
    """Blend simulation and observation probabilities.

    Observed cells: weighted average of obs and sim.
    Unobserved cells: sim only.
    """
    result = sim_probs.copy()
    observed = coverage_mask & ~np.isnan(obs_probs[:, :, 0])

    if observed.any():
        result[observed] = obs_weight * obs_probs[observed] + sim_weight * sim_probs[observed]

    return result


def _apply_static_terrain(
    probs: np.ndarray,
    initial_grid: np.ndarray,
) -> np.ndarray:
    """Override probabilities for terrain that cannot change.

    Mountains and ocean are static -- they stay the same after
    simulation with near-certain probability.
    """
    result = probs.copy()
    residual = 1.0 - STATIC_TERRAIN_CONFIDENCE
    per_class_residual = residual / (NUM_PREDICTION_CLASSES - 1)

    result = _set_static_class(
        result,
        initial_grid,
        InternalTerrain.MOUNTAIN,
        target_class=5,
        per_class_residual=per_class_residual,
    )
    result = _set_static_class(
        result,
        initial_grid,
        InternalTerrain.OCEAN,
        target_class=0,
        per_class_residual=per_class_residual,
    )
    return result


def _set_static_class(
    probs: np.ndarray,
    grid: np.ndarray,
    terrain: InternalTerrain,
    target_class: int,
    per_class_residual: float,
) -> np.ndarray:
    """Set near-certain probability for a static terrain type."""
    mask = grid == terrain
    if not mask.any():
        return probs

    probs[mask] = per_class_residual
    probs[mask, target_class] = STATIC_TERRAIN_CONFIDENCE
    return probs


def _apply_floor_and_normalize(probs: np.ndarray) -> np.ndarray:
    """Apply probability floor and renormalize each cell to sum to 1.

    Prevents infinite KL divergence from zero probabilities.
    """
    safe = np.maximum(probs, PROBABILITY_FLOOR)
    sums = safe.sum(axis=2, keepdims=True)
    return safe / sums
