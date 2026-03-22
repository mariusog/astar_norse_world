"""Tests for dynamic cell classifier."""

from __future__ import annotations

import numpy as np

from src.cell_classifier import (
    classify_cells,
    classify_static_confident,
    dynamic_fraction,
)
from src.terrain import InternalTerrain


def _ocean_grid() -> np.ndarray:
    """8x8 all-ocean grid."""
    return np.full((8, 8), InternalTerrain.OCEAN, dtype=np.int8)


def _mixed_grid() -> np.ndarray:
    """10x10 grid with ocean border, settlement at center, varied terrain."""
    g = np.full((10, 10), InternalTerrain.PLAINS, dtype=np.int8)
    g[0, :] = InternalTerrain.OCEAN
    g[-1, :] = InternalTerrain.OCEAN
    g[:, 0] = InternalTerrain.OCEAN
    g[:, -1] = InternalTerrain.OCEAN
    g[5, 5] = InternalTerrain.SETTLEMENT
    g[3, 3] = InternalTerrain.FOREST
    g[4, 4] = InternalTerrain.FOREST
    g[1, 1] = InternalTerrain.MOUNTAIN
    g[1, 8] = InternalTerrain.MOUNTAIN
    return g


# ---------------------------------------------------------------------------
# Tests: classify_cells
# ---------------------------------------------------------------------------


class TestClassifyCells:
    def test_returns_correct_shape(self) -> None:
        g = _mixed_grid()
        mask = classify_cells(g)
        assert mask.shape == (10, 10)
        assert mask.dtype == bool

    def test_ocean_is_not_dynamic(self) -> None:
        g = _ocean_grid()
        mask = classify_cells(g)
        assert not mask.any()

    def test_settlement_is_dynamic(self) -> None:
        g = _mixed_grid()
        mask = classify_cells(g)
        assert mask[5, 5] is np.True_

    def test_mountain_is_not_dynamic(self) -> None:
        g = _mixed_grid()
        mask = classify_cells(g)
        assert mask[1, 1] is np.False_
        assert mask[1, 8] is np.False_

    def test_plains_near_settlement_is_dynamic(self) -> None:
        g = _mixed_grid()
        mask = classify_cells(g)
        # Cell (5,4) is plains, 1 cell from settlement at (5,5)
        assert mask[5, 4] is np.True_

    def test_forest_near_settlement_is_dynamic(self) -> None:
        g = _mixed_grid()
        mask = classify_cells(g)
        # Forest at (4,4), settlement at (5,5) - distance 2
        assert mask[4, 4] is np.True_

    def test_far_plains_is_not_dynamic(self) -> None:
        g = np.full((20, 20), InternalTerrain.PLAINS, dtype=np.int8)
        g[0, :] = InternalTerrain.OCEAN
        g[-1, :] = InternalTerrain.OCEAN
        g[:, 0] = InternalTerrain.OCEAN
        g[:, -1] = InternalTerrain.OCEAN
        g[5, 5] = InternalTerrain.SETTLEMENT
        mask = classify_cells(g)
        # Cell (15,15) is far from settlement, should not be dynamic
        assert mask[15, 15] is np.False_


# ---------------------------------------------------------------------------
# Tests: classify_static_confident
# ---------------------------------------------------------------------------


class TestClassifyStaticConfident:
    def test_ocean_is_static(self) -> None:
        g = _mixed_grid()
        static = classify_static_confident(g)
        assert static[0, 0] is np.True_

    def test_mountain_is_static(self) -> None:
        g = _mixed_grid()
        static = classify_static_confident(g)
        assert static[1, 1] is np.True_

    def test_settlement_not_static(self) -> None:
        g = _mixed_grid()
        static = classify_static_confident(g)
        assert static[5, 5] is np.False_

    def test_plains_not_static(self) -> None:
        g = _mixed_grid()
        static = classify_static_confident(g)
        assert static[5, 4] is np.False_


# ---------------------------------------------------------------------------
# Tests: dynamic_fraction
# ---------------------------------------------------------------------------


class TestDynamicFraction:
    def test_all_ocean_returns_zero(self) -> None:
        g = _ocean_grid()
        assert dynamic_fraction(g) == 0.0

    def test_returns_between_zero_and_one(self) -> None:
        g = _mixed_grid()
        frac = dynamic_fraction(g)
        assert 0.0 < frac < 1.0

    def test_real_data_nonzero(self) -> None:
        """Check dynamic fraction is nonzero on a generated map."""
        try:
            from src.map_generator import generate_map

            grid, _ = generate_map(seed=42, width=40, height=40)
        except ImportError:
            return  # Skip if map_generator not available
        frac = dynamic_fraction(grid)
        # Generated maps vary; just verify it's nonzero and under 100%
        assert frac > 0.0
        assert frac < 1.0
