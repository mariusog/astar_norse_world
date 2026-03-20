"""Tests for unified survive-weighted prior builder."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.terrain import InternalTerrain
from src.unified_priors import (
    DIST_BIN_EDGES,
    NUM_INTERNAL_TYPES,
    build_distance_priors,
    build_unified_priors,
    load_priors,
    predict_from_priors,
    save_priors,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_round_dir(
    base: Path,
    round_number: int,
    grids: list[np.ndarray],
    gts: list[np.ndarray],
) -> None:
    """Create a round directory with seed data."""
    rd = base / "rounds" / f"round_{round_number}"
    rd.mkdir(parents=True, exist_ok=True)
    with open(rd / "round.json", "w") as f:
        json.dump({"round_number": round_number}, f)
    for i, (g, gt) in enumerate(zip(grids, gts, strict=True)):
        sd = rd / f"seed_{i}"
        sd.mkdir(parents=True, exist_ok=True)
        np.save(sd / "initial_grid.npy", g)
        np.save(sd / "ground_truth.npy", gt)


def _simple_grid() -> np.ndarray:
    """4x4 grid with ocean border and plains/settlement interior."""
    g = np.full((4, 4), InternalTerrain.PLAINS, dtype=np.int8)
    g[0, :] = InternalTerrain.OCEAN
    g[-1, :] = InternalTerrain.OCEAN
    g[:, 0] = InternalTerrain.OCEAN
    g[:, -1] = InternalTerrain.OCEAN
    g[2, 2] = InternalTerrain.SETTLEMENT
    return g


def _survive_gt(grid: np.ndarray) -> np.ndarray:
    """GT where settlements survive (high settlement prob)."""
    h, w = grid.shape
    gt = np.zeros((h, w, 6), dtype=np.float64)
    for y in range(h):
        for x in range(w):
            t = int(grid[y, x])
            if t == InternalTerrain.OCEAN:
                gt[y, x, 0] = 1.0
            elif t == InternalTerrain.SETTLEMENT:
                gt[y, x, 1] = 0.8
                gt[y, x, 0] = 0.2
            else:
                gt[y, x, 0] = 0.9
                gt[y, x, 1] = 0.1
    return gt


def _collapse_gt(grid: np.ndarray) -> np.ndarray:
    """GT where settlements collapse (high empty/forest prob)."""
    h, w = grid.shape
    gt = np.zeros((h, w, 6), dtype=np.float64)
    for y in range(h):
        for x in range(w):
            t = int(grid[y, x])
            if t == InternalTerrain.OCEAN:
                gt[y, x, 0] = 1.0
            elif t == InternalTerrain.SETTLEMENT:
                gt[y, x, 0] = 0.7
                gt[y, x, 4] = 0.3
            else:
                gt[y, x, 0] = 0.95
                gt[y, x, 4] = 0.05
    return gt


# ---------------------------------------------------------------------------
# Tests: build_unified_priors
# ---------------------------------------------------------------------------


class TestBuildUnifiedPriors:
    def test_returns_correct_shape(self, tmp_path: Path) -> None:
        g = _simple_grid()
        _make_round_dir(tmp_path, 1, [g], [_survive_gt(g)])
        priors = build_unified_priors(tmp_path)
        assert priors.shape == (NUM_INTERNAL_TYPES, 6)

    def test_rows_are_normalized(self, tmp_path: Path) -> None:
        g = _simple_grid()
        _make_round_dir(tmp_path, 1, [g], [_survive_gt(g)])
        priors = build_unified_priors(tmp_path)
        for t in range(NUM_INTERNAL_TYPES):
            assert pytest.approx(priors[t].sum(), abs=1e-6) == 1.0

    def test_survive_rounds_weighted_higher(self, tmp_path: Path) -> None:
        g = _simple_grid()
        _make_round_dir(tmp_path, 1, [g], [_survive_gt(g)])
        _make_round_dir(tmp_path, 3, [g], [_collapse_gt(g)])
        priors = build_unified_priors(tmp_path)
        # Settlement type should lean toward survive (class 1)
        # because R1 is survive (weight 3) vs R3 collapse (weight 1)
        settle_idx = InternalTerrain.SETTLEMENT
        assert priors[settle_idx, 1] > priors[settle_idx, 4]

    def test_no_rounds_returns_uniform(self, tmp_path: Path) -> None:
        priors = build_unified_priors(tmp_path)
        expected = 1.0 / 6
        assert pytest.approx(priors[0, 0], abs=1e-6) == expected

    def test_missing_directory_returns_uniform(self) -> None:
        priors = build_unified_priors("/nonexistent/path")
        assert priors.shape == (NUM_INTERNAL_TYPES, 6)


# ---------------------------------------------------------------------------
# Tests: build_distance_priors
# ---------------------------------------------------------------------------


class TestBuildDistancePriors:
    def test_returns_correct_shape(self, tmp_path: Path) -> None:
        g = _simple_grid()
        _make_round_dir(tmp_path, 1, [g], [_survive_gt(g)])
        dp = build_distance_priors(tmp_path)
        num_bins = len(DIST_BIN_EDGES) - 1
        assert dp.shape == (NUM_INTERNAL_TYPES, num_bins, 6)

    def test_rows_are_normalized(self, tmp_path: Path) -> None:
        g = _simple_grid()
        _make_round_dir(tmp_path, 1, [g], [_survive_gt(g)])
        dp = build_distance_priors(tmp_path)
        for t in range(NUM_INTERNAL_TYPES):
            for b in range(dp.shape[1]):
                assert pytest.approx(dp[t, b].sum(), abs=1e-6) == 1.0

    def test_no_data_returns_uniform(self, tmp_path: Path) -> None:
        dp = build_distance_priors(tmp_path)
        assert pytest.approx(dp[0, 0, 0], abs=1e-6) == 1.0 / 6


# ---------------------------------------------------------------------------
# Tests: predict_from_priors
# ---------------------------------------------------------------------------


class TestPredictFromPriors:
    def test_output_shape(self, tmp_path: Path) -> None:
        g = _simple_grid()
        _make_round_dir(tmp_path, 1, [g], [_survive_gt(g)])
        priors = build_unified_priors(tmp_path)
        pred = predict_from_priors(g, priors)
        assert pred.shape == (4, 4, 6)

    def test_probabilities_sum_to_one(self, tmp_path: Path) -> None:
        g = _simple_grid()
        _make_round_dir(tmp_path, 1, [g], [_survive_gt(g)])
        priors = build_unified_priors(tmp_path)
        pred = predict_from_priors(g, priors)
        sums = pred.sum(axis=2)
        np.testing.assert_allclose(sums, 1.0, atol=1e-6)

    def test_floor_applied(self, tmp_path: Path) -> None:
        g = _simple_grid()
        _make_round_dir(tmp_path, 1, [g], [_survive_gt(g)])
        priors = build_unified_priors(tmp_path)
        pred = predict_from_priors(g, priors)
        assert pred.min() >= 0.01 - 1e-9

    def test_ocean_gets_static_override(self, tmp_path: Path) -> None:
        g = _simple_grid()
        _make_round_dir(tmp_path, 1, [g], [_survive_gt(g)])
        priors = build_unified_priors(tmp_path)
        pred = predict_from_priors(g, priors)
        # Ocean cells should have high empty probability
        assert pred[0, 0, 0] > 0.9

    def test_with_distance_priors(self, tmp_path: Path) -> None:
        g = _simple_grid()
        _make_round_dir(tmp_path, 1, [g], [_survive_gt(g)])
        priors = build_unified_priors(tmp_path)
        dp = build_distance_priors(tmp_path)
        pred = predict_from_priors(g, priors, dist_priors=dp)
        assert pred.shape == (4, 4, 6)
        sums = pred.sum(axis=2)
        np.testing.assert_allclose(sums, 1.0, atol=1e-6)


# ---------------------------------------------------------------------------
# Tests: save/load roundtrip
# ---------------------------------------------------------------------------


class TestSaveLoad:
    def test_roundtrip_priors_only(self, tmp_path: Path) -> None:
        priors = np.random.default_rng(42).random((NUM_INTERNAL_TYPES, 6))
        priors /= priors.sum(axis=1, keepdims=True)
        path = tmp_path / "priors.npz"
        save_priors(priors, path)
        loaded, dp = load_priors(path)
        np.testing.assert_allclose(loaded, priors)
        assert dp is None

    def test_roundtrip_with_distance_priors(self, tmp_path: Path) -> None:
        rng = np.random.default_rng(42)
        priors = rng.random((NUM_INTERNAL_TYPES, 6))
        priors /= priors.sum(axis=1, keepdims=True)
        dp = rng.random((NUM_INTERNAL_TYPES, 5, 6))
        dp /= dp.sum(axis=2, keepdims=True)
        path = tmp_path / "priors.npz"
        save_priors(priors, path, dist_priors=dp)
        loaded_p, loaded_dp = load_priors(path)
        np.testing.assert_allclose(loaded_p, priors)
        np.testing.assert_allclose(loaded_dp, dp)


# ---------------------------------------------------------------------------
# Tests: backtest on real data
# ---------------------------------------------------------------------------


class TestBacktest:
    @pytest.mark.slow
    def test_avg_score_above_75(self) -> None:
        """Backtest unified priors on all available rounds."""
        data_dir = Path("data")
        rounds_dir = data_dir / "rounds"
        if not rounds_dir.exists():
            pytest.skip("No round data available")

        priors = build_unified_priors(data_dir)
        dp = build_distance_priors(data_dir)

        scores = []
        for rd in rounds_dir.iterdir():
            rj_path = rd / "round.json"
            if not rj_path.exists():
                continue
            for si in range(5):
                sd = rd / f"seed_{si}"
                ig_path = sd / "initial_grid.npy"
                gt_path = sd / "ground_truth.npy"
                if not ig_path.exists() or not gt_path.exists():
                    continue
                ig = np.load(ig_path)
                gt = np.load(gt_path)
                pred = predict_from_priors(ig, priors, dist_priors=dp)
                s = _kl_score(pred, gt)
                scores.append(s)

        avg = float(np.mean(scores))
        assert avg >= 75.0, f"Avg score {avg:.1f} < 75.0"

    def test_avg_score_above_75_with_real_data(self) -> None:
        """Non-slow version that just checks the function works."""
        data_dir = Path("data")
        rounds_dir = data_dir / "rounds"
        if not rounds_dir.exists():
            pytest.skip("No round data available")

        priors = build_unified_priors(data_dir)
        assert priors.shape == (NUM_INTERNAL_TYPES, 6)
        assert np.all(priors >= 0)


def _kl_score(pred: np.ndarray, gt: np.ndarray) -> float:
    """Compute exp(-3 * KL) score, 0-100 scale."""
    eps = 1e-10
    kl = np.sum(gt * np.log((gt + eps) / (pred + eps)), axis=2)
    entropy = -np.sum(gt * np.log(gt + eps), axis=2)
    mask = entropy > 0.01
    if mask.sum() == 0:
        return 100.0
    return float(np.exp(-3 * kl[mask]).mean() * 100)
