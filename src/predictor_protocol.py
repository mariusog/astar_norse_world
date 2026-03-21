"""Common predictor protocol for terrain probability prediction."""

from __future__ import annotations

from typing import Protocol

import numpy as np


class GridPredictor(Protocol):
    """Protocol for predicting terrain probability tensors from a grid."""

    def predict_grid(self, grid: np.ndarray) -> np.ndarray:
        """Predict H x W x 6 probability tensor from terrain grid.

        Args:
            grid: H x W array of InternalTerrain values.

        Returns:
            H x W x 6 probability tensor, each cell sums to ~1.0.
        """
        ...
