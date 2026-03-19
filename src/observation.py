"""Observation aggregator for viewport query results.

Accumulates stochastic server observations per seed and converts
frequency counts into per-cell probability estimates with Laplace
smoothing.
"""

from __future__ import annotations

import logging

import numpy as np

from src.constants import NUM_PREDICTION_CLASSES

logger = logging.getLogger(__name__)


class ObservationStore:
    """Accumulate viewport observations and compute probability tensors.

    Each query returns a terrain grid for a viewport region. Since
    simulations are stochastic, the same cell may yield different
    terrain types across queries. This store merges overlapping
    observations using frequency counting and Laplace smoothing.

    Args:
        height: Map height in cells.
        width: Map width in cells.
        num_seeds: Number of seeds to track.
    """

    def __init__(self, height: int, width: int, num_seeds: int) -> None:
        self._height = height
        self._width = width
        self._num_seeds = num_seeds
        # Per-seed frequency counts: shape (H, W, num_classes)
        self._counts: dict[int, np.ndarray] = {}
        # Per-seed observation counts: shape (H, W)
        self._obs_counts: dict[int, np.ndarray] = {}

    def add_observation(
        self,
        seed_index: int,
        viewport_x: int,
        viewport_y: int,
        grid_patch: np.ndarray,
    ) -> None:
        """Record one viewport observation from a query result.

        Args:
            seed_index: Which seed this observation is for.
            viewport_x: Left column of the viewport.
            viewport_y: Top row of the viewport.
            grid_patch: 2D array of terrain codes, shape (vh, vw).
        """
        self._ensure_seed(seed_index)
        vh, vw = grid_patch.shape
        counts = self._counts[seed_index]
        obs = self._obs_counts[seed_index]

        for row in range(vh):
            for col in range(vw):
                gy = viewport_y + row
                gx = viewport_x + col
                if not self._in_bounds(gx, gy):
                    continue
                terrain_code = int(grid_patch[row, col])
                if 0 <= terrain_code < NUM_PREDICTION_CLASSES:
                    counts[gy, gx, terrain_code] += 1
                    obs[gy, gx] += 1

        logger.info(
            "Added observation for seed %d: viewport (%d,%d) size %dx%d",
            seed_index,
            viewport_x,
            viewport_y,
            vw,
            vh,
        )

    def get_observed_probs(self, seed_index: int) -> np.ndarray:
        """Return H x W x 6 probability tensor from observations.

        Observed cells use frequency counts with Laplace (add-1)
        smoothing. Unobserved cells are filled with NaN.

        Args:
            seed_index: Which seed to compute probabilities for.

        Returns:
            Array of shape (H, W, 6). NaN where unobserved.
        """
        probs = np.full(
            (self._height, self._width, NUM_PREDICTION_CLASSES),
            np.nan,
        )
        if seed_index not in self._counts:
            return probs

        counts = self._counts[seed_index]
        obs = self._obs_counts[seed_index]
        observed = obs > 0

        # Laplace smoothing: add 1 to each class count
        smoothed = counts[observed] + 1.0
        totals = smoothed.sum(axis=1, keepdims=True)
        probs[observed] = smoothed / totals

        return probs

    def get_coverage_mask(self, seed_index: int) -> np.ndarray:
        """Return H x W boolean mask of observed cells.

        Args:
            seed_index: Which seed to check coverage for.

        Returns:
            Boolean array of shape (H, W). True where observed.
        """
        if seed_index not in self._obs_counts:
            return np.zeros((self._height, self._width), dtype=bool)
        return self._obs_counts[seed_index] > 0

    def observation_count(self, seed_index: int) -> np.ndarray:
        """Return H x W count of observations per cell.

        Args:
            seed_index: Which seed to count for.

        Returns:
            Integer array of shape (H, W).
        """
        if seed_index not in self._obs_counts:
            return np.zeros((self._height, self._width), dtype=np.int32)
        return self._obs_counts[seed_index].copy()

    def total_observations(self, seed_index: int) -> int:
        """Return total number of query observations for this seed."""
        if seed_index not in self._obs_counts:
            return 0
        obs = self._obs_counts[seed_index]
        max_count = obs.max() if obs.size > 0 else 0
        return int(max_count)

    def coverage_fraction(self, seed_index: int) -> float:
        """Return fraction of map cells that have been observed."""
        mask = self.get_coverage_mask(seed_index)
        total_cells = self._height * self._width
        if total_cells == 0:
            return 0.0
        return float(mask.sum()) / total_cells

    # -- Internal helpers ---------------------------------------------------

    def _ensure_seed(self, seed_index: int) -> None:
        """Initialize storage for a seed if not already present."""
        if seed_index not in self._counts:
            self._counts[seed_index] = np.zeros(
                (self._height, self._width, NUM_PREDICTION_CLASSES),
                dtype=np.int32,
            )
            self._obs_counts[seed_index] = np.zeros(
                (self._height, self._width),
                dtype=np.int32,
            )

    def _in_bounds(self, x: int, y: int) -> bool:
        """Check if (x, y) is within the map grid."""
        return 0 <= x < self._width and 0 <= y < self._height
