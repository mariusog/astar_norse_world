"""Tests for observation-focused query planner v2."""

from __future__ import annotations

import numpy as np

from src.cell_classifier import classify_cells
from src.query_planner_v2 import VP_SIZE, plan_queries
from src.query_strategy import Viewport
from src.terrain import InternalTerrain


def _competition_grid() -> np.ndarray:
    """40x40 grid mimicking competition map structure."""
    g = np.full((40, 40), InternalTerrain.PLAINS, dtype=np.int8)
    # Ocean border
    g[0, :] = InternalTerrain.OCEAN
    g[-1, :] = InternalTerrain.OCEAN
    g[:, 0] = InternalTerrain.OCEAN
    g[:, -1] = InternalTerrain.OCEAN
    # Fjord
    g[1:10, 5] = InternalTerrain.OCEAN
    # Mountains
    g[15, 10:20] = InternalTerrain.MOUNTAIN
    # Settlements
    g[5, 10] = InternalTerrain.SETTLEMENT
    g[10, 25] = InternalTerrain.SETTLEMENT
    g[20, 15] = InternalTerrain.SETTLEMENT
    g[30, 30] = InternalTerrain.SETTLEMENT
    g[25, 5] = InternalTerrain.PORT
    # Forest
    g[5:8, 12:15] = InternalTerrain.FOREST
    g[28:32, 28:32] = InternalTerrain.FOREST
    return g


def _small_grid() -> np.ndarray:
    """10x10 simple grid with one settlement."""
    g = np.full((10, 10), InternalTerrain.PLAINS, dtype=np.int8)
    g[0, :] = InternalTerrain.OCEAN
    g[-1, :] = InternalTerrain.OCEAN
    g[:, 0] = InternalTerrain.OCEAN
    g[:, -1] = InternalTerrain.OCEAN
    g[5, 5] = InternalTerrain.SETTLEMENT
    return g


# ---------------------------------------------------------------------------
# Tests: plan_queries
# ---------------------------------------------------------------------------


class TestPlanQueries:
    def test_returns_correct_num_seeds(self) -> None:
        g = _competition_grid()
        dm = classify_cells(g)
        result = plan_queries(g, dm, budget=50, num_seeds=5)
        assert len(result) == 5

    def test_respects_budget(self) -> None:
        g = _competition_grid()
        dm = classify_cells(g)
        result = plan_queries(g, dm, budget=50, num_seeds=5)
        total_queries = sum(len(vps) for vps in result)
        assert total_queries <= 50

    def test_queries_per_seed_respects_budget(self) -> None:
        g = _competition_grid()
        dm = classify_cells(g)
        result = plan_queries(g, dm, budget=50, num_seeds=5)
        for seed_vps in result:
            assert len(seed_vps) <= 10

    def test_all_viewports_are_max_size_for_large_grid(self) -> None:
        g = _competition_grid()
        dm = classify_cells(g)
        result = plan_queries(g, dm, budget=50, num_seeds=5)
        for seed_vps in result:
            for vp in seed_vps:
                assert vp.viewport_w == VP_SIZE
                assert vp.viewport_h == VP_SIZE

    def test_viewports_within_bounds(self) -> None:
        g = _competition_grid()
        dm = classify_cells(g)
        h, w = g.shape
        result = plan_queries(g, dm, budget=50, num_seeds=5)
        for seed_vps in result:
            for vp in seed_vps:
                assert vp.viewport_x >= 0
                assert vp.viewport_y >= 0
                assert vp.viewport_x + vp.viewport_w <= w
                assert vp.viewport_y + vp.viewport_h <= h

    def test_seed_index_matches(self) -> None:
        g = _competition_grid()
        dm = classify_cells(g)
        result = plan_queries(g, dm, budget=50, num_seeds=5)
        for seed_idx, seed_vps in enumerate(result):
            for vp in seed_vps:
                assert vp.seed_index == seed_idx

    def test_viewports_are_viewport_type(self) -> None:
        g = _competition_grid()
        dm = classify_cells(g)
        result = plan_queries(g, dm, budget=50, num_seeds=5)
        for seed_vps in result:
            for vp in seed_vps:
                assert isinstance(vp, Viewport)

    def test_covers_dynamic_cells(self) -> None:
        g = _competition_grid()
        dm = classify_cells(g)
        h, w = g.shape
        result = plan_queries(g, dm, budget=50, num_seeds=5)
        # Check coverage on seed 0
        covered = np.zeros((h, w), dtype=bool)
        for vp in result[0]:
            x1 = min(vp.viewport_x + vp.viewport_w, w)
            y1 = min(vp.viewport_y + vp.viewport_h, h)
            covered[vp.viewport_y : y1, vp.viewport_x : x1] = True
        dyn_covered = (dm & covered).sum()
        dyn_total = dm.sum()
        frac = dyn_covered / dyn_total if dyn_total > 0 else 1.0
        # Should cover majority of dynamic cells with 10 queries
        assert frac > 0.5

    def test_small_grid_works(self) -> None:
        g = _small_grid()
        dm = classify_cells(g)
        result = plan_queries(g, dm, budget=10, num_seeds=2)
        assert len(result) == 2
        for seed_vps in result:
            for vp in seed_vps:
                assert vp.viewport_x + vp.viewport_w <= 10
                assert vp.viewport_y + vp.viewport_h <= 10

    def test_no_dynamic_cells_still_returns_viewports(self) -> None:
        g = np.full((20, 20), InternalTerrain.OCEAN, dtype=np.int8)
        dm = np.zeros((20, 20), dtype=bool)
        result = plan_queries(g, dm, budget=10, num_seeds=2)
        assert len(result) == 2
        # Should still produce fallback viewports for coverage
        for seed_vps in result:
            for vp in seed_vps:
                assert vp.viewport_w == min(VP_SIZE, 20)

    def test_budget_one_per_seed(self) -> None:
        g = _competition_grid()
        dm = classify_cells(g)
        result = plan_queries(g, dm, budget=5, num_seeds=5)
        for seed_vps in result:
            assert len(seed_vps) <= 1
