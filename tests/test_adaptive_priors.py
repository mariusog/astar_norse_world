"""Tests for regime-adaptive feature model."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.adaptive_priors import (
    _REGIME_WEIGHTS,
    _get_regime_weights,
    build_adaptive_feature_lookup,
)
from src.terrain import InternalTerrain


@pytest.fixture
def mock_data_dir(tmp_path: Path) -> Path:
    """Create mock data with two rounds for regime testing."""
    for rnum in (1, 3):
        round_dir = tmp_path / f"round_{rnum}"
        round_dir.mkdir()
        grid = np.full((5, 5), InternalTerrain.PLAINS, dtype=np.int8)
        grid[0, :] = InternalTerrain.OCEAN
        grid[2, 2] = InternalTerrain.SETTLEMENT
        gt = np.full((5, 5, 6), 1.0 / 6, dtype=np.float64)
        gt[0, :] = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        seed_dir = round_dir / "seed_0"
        seed_dir.mkdir()
        np.save(seed_dir / "ground_truth.npy", gt)
        np.save(seed_dir / "initial_grid.npy", grid)
        rd = {
            "round_number": rnum,
            "initial_states": [{"grid": grid.tolist()}],
        }
        with open(round_dir / "round.json", "w") as f:
            json.dump(rd, f)
    return tmp_path


class TestGetRegimeWeights:
    """Tests for regime weight lookup."""

    def test_survive_weights(self) -> None:
        weights = _get_regime_weights("survive")
        assert weights[1] == 2.0
        assert weights[3] == 1.0

    def test_collapse_weights(self) -> None:
        weights = _get_regime_weights("collapse")
        assert weights[3] == 2.0
        assert weights[1] == 1.0

    def test_aggressive_weights(self) -> None:
        weights = _get_regime_weights("aggressive")
        assert weights[1] == 2.0

    def test_unknown_regime_returns_empty(self) -> None:
        weights = _get_regime_weights("unknown")
        assert weights == {}


class TestBuildAdaptiveFeatureLookup:
    """Tests for adaptive feature lookup construction."""

    def test_survive_produces_lookup(self, mock_data_dir: Path) -> None:
        lookup = build_adaptive_feature_lookup("survive", mock_data_dir)
        assert len(lookup) > 0

    def test_collapse_produces_lookup(self, mock_data_dir: Path) -> None:
        lookup = build_adaptive_feature_lookup("collapse", mock_data_dir)
        assert len(lookup) > 0

    def test_unknown_regime_still_works(self, mock_data_dir: Path) -> None:
        lookup = build_adaptive_feature_lookup("unknown", mock_data_dir)
        assert len(lookup) > 0

    def test_all_values_are_prob_vectors(self, mock_data_dir: Path) -> None:
        lookup = build_adaptive_feature_lookup("survive", mock_data_dir)
        for vec in lookup.values():
            assert vec.shape == (6,)
            assert vec.sum() == pytest.approx(1.0, abs=0.05)

    def test_regime_weights_defined(self) -> None:
        """All three regimes have weight dictionaries."""
        assert "survive" in _REGIME_WEIGHTS
        assert "aggressive" in _REGIME_WEIGHTS
        assert "collapse" in _REGIME_WEIGHTS
