"""Observation aggregator for viewport query results.

Accumulates stochastic server observations per seed and converts
frequency counts into per-cell probability estimates with Laplace
smoothing.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from src.constants import LAPLACE_ALPHA, NUM_PREDICTION_CLASSES
from src.terrain import SERVER_TO_PRED_CLASS

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
                Accepts both server codes (0-5, 10, 11) and prediction
                class indices (0-5). Server codes 10/11 map to class 0 (Empty).
        """
        self._ensure_seed(seed_index)
        vh, vw = grid_patch.shape
        self._accumulate_patch(
            seed_index,
            viewport_x,
            viewport_y,
            grid_patch,
            vh,
            vw,
        )
        logger.info(
            "Added observation for seed %d: viewport (%d,%d) size %dx%d",
            seed_index,
            viewport_x,
            viewport_y,
            vw,
            vh,
        )

    def _accumulate_patch(
        self,
        seed_index: int,
        viewport_x: int,
        viewport_y: int,
        grid_patch: np.ndarray,
        vh: int,
        vw: int,
    ) -> None:
        """Add terrain counts from a viewport patch into the store."""
        counts = self._counts[seed_index]
        obs = self._obs_counts[seed_index]
        for row in range(vh):
            for col in range(vw):
                gy = viewport_y + row
                gx = viewport_x + col
                if not self._in_bounds(gx, gy):
                    continue
                raw_code = int(grid_patch[row, col])
                pred_class = SERVER_TO_PRED_CLASS.get(raw_code, -1)
                if 0 <= pred_class < NUM_PREDICTION_CLASSES:
                    counts[gy, gx, pred_class] += 1
                    obs[gy, gx] += 1

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

        # Laplace smoothing with small alpha to keep single observations sharp
        smoothed = counts[observed] + LAPLACE_ALPHA
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

    def max_cell_observations(self, seed_index: int) -> int:
        """Return the maximum observation count for any single cell."""
        if seed_index not in self._obs_counts:
            return 0
        obs = self._obs_counts[seed_index]
        return int(obs.max()) if obs.size > 0 else 0

    def coverage_fraction(self, seed_index: int) -> float:
        """Return fraction of map cells that have been observed."""
        mask = self.get_coverage_mask(seed_index)
        total_cells = self._height * self._width
        if total_cells == 0:
            return 0.0
        return float(mask.sum()) / total_cells

    # -- Persistence -------------------------------------------------------

    def save_to_disk(self, path: str | Path) -> None:
        """Save observation data to a .npz file for cross-process persistence.

        Args:
            path: File path to write (will be overwritten if exists).
        """
        path = Path(path)
        arrays: dict[str, np.ndarray] = {
            "_meta": np.array([self._height, self._width, self._num_seeds], dtype=np.int32),
        }
        for si in self._counts:
            arrays[f"counts_{si}"] = self._counts[si]
            arrays[f"obs_{si}"] = self._obs_counts[si]
        np.savez_compressed(path, **arrays)
        logger.info("Saved observations to %s (%d seeds)", path, len(self._counts))

    @classmethod
    def load_from_disk(cls, path: str | Path) -> ObservationStore:
        """Restore an ObservationStore from a .npz file.

        Args:
            path: File path previously written by save_to_disk().

        Returns:
            Restored ObservationStore with all accumulated counts.
        """
        path = Path(path)
        data = np.load(path)
        meta = data["_meta"]
        height, width, num_seeds = int(meta[0]), int(meta[1]), int(meta[2])
        store = cls(height, width, num_seeds)
        expected_counts = (height, width, NUM_PREDICTION_CLASSES)
        expected_obs = (height, width)
        for key in data.files:
            if key.startswith("counts_"):
                si = int(key.split("_", 1)[1])
                arr = data[key]
                if arr.shape != expected_counts:
                    msg = f"counts shape {arr.shape} != expected {expected_counts}"
                    raise ValueError(msg)
                store._counts[si] = arr
            elif key.startswith("obs_"):
                si = int(key.split("_", 1)[1])
                arr = data[key]
                if arr.shape != expected_obs:
                    msg = f"obs shape {arr.shape} != expected {expected_obs}"
                    raise ValueError(msg)
                store._obs_counts[si] = arr
        logger.info("Loaded observations from %s (%d seeds)", path, len(store._counts))
        return store

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
