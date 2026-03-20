"""Regime-adaptive feature model for terrain prediction.

Adjusts historical round weights based on detected regime
(survive, aggressive, collapse) to produce tuned feature lookups.
"""

from __future__ import annotations

import logging
from pathlib import Path

from src.feature_predictor import FeatureLookup, build_feature_lookup

logger = logging.getLogger(__name__)

# Regime weight definitions: round_number -> weight multiplier
# Rounds weighted 2x are more similar to the target regime
_REGIME_WEIGHTS: dict[str, dict[int, float]] = {
    "survive": {1: 2.0, 2: 2.0, 3: 1.0, 4: 1.0, 5: 2.0, 6: 2.0},
    "aggressive": {1: 2.0, 2: 2.0, 3: 1.0, 4: 1.0, 5: 2.0, 6: 2.0},
    "collapse": {1: 1.0, 2: 1.0, 3: 2.0, 4: 2.0, 5: 1.0, 6: 1.0},
}

# Default data directory relative to project root
_DEFAULT_DATA_DIR = "data/rounds"


def build_adaptive_feature_lookup(
    regime: str,
    data_dir: str | Path = _DEFAULT_DATA_DIR,
) -> FeatureLookup:
    """Build feature lookup with regime-specific round weighting.

    Args:
        regime: One of 'survive', 'aggressive', 'collapse'.
        data_dir: Path to historical round data directory.

    Returns:
        Feature lookup table with regime-tuned weights.
    """
    weights = _get_regime_weights(regime)
    logger.info("Building adaptive lookup: regime=%s, weights=%s", regime, weights)
    return build_feature_lookup(data_dir, regime_weights=weights)


def _get_regime_weights(regime: str) -> dict[int, float]:
    """Get round weights for a regime, defaulting to uniform."""
    weights = _REGIME_WEIGHTS.get(regime)
    if weights is None:
        logger.warning("Unknown regime '%s', using uniform weights", regime)
        return {}
    return weights
