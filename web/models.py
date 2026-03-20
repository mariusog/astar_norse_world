"""Model registry for prediction strategies.

Each strategy wraps a predict_fn(grid, data_dir) -> H x W x 6 tensor.
Strategies are registered at import time and can be evaluated by name.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from src.constants import (
    NUM_PREDICTION_CLASSES,
    PROBABILITY_FLOOR,
    STATIC_TERRAIN_CONFIDENCE,
)
from src.terrain import InternalTerrain

logger = logging.getLogger(__name__)


@dataclass
class Strategy:
    """A named prediction strategy with its callable."""

    name: str
    description: str
    predict_fn: Callable[[np.ndarray, str], np.ndarray]


STRATEGIES: dict[str, Strategy] = {}


def register(
    name: str,
    description: str,
    fn: Callable[[np.ndarray, str], np.ndarray],
) -> None:
    """Register a prediction strategy by name."""
    STRATEGIES[name] = Strategy(
        name=name,
        description=description,
        predict_fn=fn,
    )


def list_strategies() -> list[Strategy]:
    """Return all registered strategies."""
    return list(STRATEGIES.values())


def get_strategy(name: str) -> Strategy:
    """Look up a strategy by name. Raises KeyError if not found."""
    return STRATEGIES[name]


def apply_static_and_normalize(
    tensor: np.ndarray,
    grid: np.ndarray,
) -> np.ndarray:
    """Apply static terrain overrides, floor, and normalize."""
    result = tensor.copy()
    residual = 1.0 - STATIC_TERRAIN_CONFIDENCE
    per_class = residual / (NUM_PREDICTION_CLASSES - 1)
    for terrain, cls_idx in [
        (InternalTerrain.OCEAN, 0),
        (InternalTerrain.MOUNTAIN, 5),
    ]:
        mask = grid == terrain
        if mask.any():
            result[mask] = per_class
            result[mask, cls_idx] = STATIC_TERRAIN_CONFIDENCE
    safe = np.maximum(result, PROBABILITY_FLOOR)
    return safe / safe.sum(axis=2, keepdims=True)


# ---------------------------------------------------------------------------
# Register all strategies (imported from web.strategies)
# ---------------------------------------------------------------------------

from web.strategies import (  # noqa: E402
    predict_distance_priors,
    predict_feature_lookup,
    predict_flat_priors,
    predict_xgboost,
    predict_xgboost_collapse,
    predict_xgboost_survive,
)

register(
    "flat_priors",
    "Flat terrain-type priors from historical GT",
    predict_flat_priors,
)
register(
    "distance_priors",
    "Flat priors + settlement distance model",
    predict_distance_priors,
)
register(
    "feature_lookup",
    "Feature-conditioned lookup table",
    predict_feature_lookup,
)
register("xgboost", "XGBoost trained on all rounds", predict_xgboost)
register(
    "xgboost_survive",
    "XGBoost excluding collapse rounds (3,4)",
    predict_xgboost_survive,
)
register(
    "xgboost_collapse",
    "XGBoost excluding survive rounds (1,2,5)",
    predict_xgboost_collapse,
)
