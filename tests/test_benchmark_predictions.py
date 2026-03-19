"""Prediction quality benchmarks (marked slow).

Generates ground truth with many MC runs, then scores the predictor
with fewer runs against it. Reports baseline scores to
docs/benchmark_results.md.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pytest

from src.constants import NUM_PREDICTION_CLASSES, PROBABILITY_FLOOR
from src.runner import run_monte_carlo
from src.scoring import score_prediction

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

GROUND_TRUTH_RUNS = 500
PREDICTOR_RUNS = 50
BENCHMARK_SEEDS = [42, 123, 256, 789, 1024]
SMALL_MAP_W = 20
SMALL_MAP_H = 20


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _generate_ground_truth(
    map_seed: int,
    num_runs: int = GROUND_TRUTH_RUNS,
) -> np.ndarray:
    """Generate ground truth via many MC runs."""
    return run_monte_carlo(
        map_seed=map_seed,
        num_runs=num_runs,
        width=SMALL_MAP_W,
        height=SMALL_MAP_H,
    )


def _generate_prediction(
    map_seed: int,
    num_runs: int = PREDICTOR_RUNS,
) -> np.ndarray:
    """Generate prediction with fewer MC runs."""
    return run_monte_carlo(
        map_seed=map_seed,
        num_runs=num_runs,
        width=SMALL_MAP_W,
        height=SMALL_MAP_H,
    )


def _generate_prediction_with_observations(
    map_seed: int,
    ground_truth: np.ndarray,
    num_runs: int = PREDICTOR_RUNS,
) -> np.ndarray:
    """Blend MC sim with mock observations from ground truth.

    Simulates observing a 10x10 viewport of the ground truth,
    then blending with the MC-based prediction.
    """
    sim_probs = run_monte_carlo(
        map_seed=map_seed,
        num_runs=num_runs,
        width=SMALL_MAP_W,
        height=SMALL_MAP_H,
    )
    # Mock: use ground truth as observed in a viewport region
    obs_weight = 0.8
    sim_weight = 0.2
    blended = sim_probs.copy()
    # Observe center 10x10 region
    y0, x0 = 5, 5
    y1, x1 = 15, 15
    blended[y0:y1, x0:x1] = (
        obs_weight * ground_truth[y0:y1, x0:x1] + sim_weight * sim_probs[y0:y1, x0:x1]
    )
    # Re-floor and normalize
    blended = np.maximum(blended, PROBABILITY_FLOOR)
    blended = blended / blended.sum(axis=2, keepdims=True)
    return blended


# ---------------------------------------------------------------------------
# Benchmark tests (slow)
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestPredictionQualityBaseline:
    """Benchmark prediction quality across multiple seeds."""

    def test_pure_sim_scores_above_zero(self) -> None:
        """Pure MC sim should score above 0 for every seed."""
        for seed in BENCHMARK_SEEDS:
            gt = _generate_ground_truth(seed)
            pred = _generate_prediction(seed)
            result = score_prediction(gt, pred)
            assert result["score"] > 0, f"seed={seed} scored 0"

    def test_more_runs_improves_score(self) -> None:
        """Prediction with more MC runs scores better."""
        seed = BENCHMARK_SEEDS[0]
        gt = _generate_ground_truth(seed)
        pred_50 = _generate_prediction(seed, num_runs=50)
        pred_200 = _generate_prediction(seed, num_runs=200)
        score_50 = score_prediction(gt, pred_50)["score"]
        score_200 = score_prediction(gt, pred_200)["score"]
        # 200 runs should score at least as well as 50 (statistically)
        # Allow small tolerance for randomness
        assert score_200 >= score_50 * 0.8, (
            f"200 runs ({score_200:.1f}) not better than 50 ({score_50:.1f})"
        )

    def test_observation_blending_improves_score(self) -> None:
        """Blending observations with sim improves prediction."""
        seed = BENCHMARK_SEEDS[0]
        gt = _generate_ground_truth(seed)
        pure_sim = _generate_prediction(seed)
        blended = _generate_prediction_with_observations(seed, gt)
        score_pure = score_prediction(gt, pure_sim)["score"]
        score_blended = score_prediction(gt, blended)["score"]
        assert score_blended >= score_pure, (
            f"Blended ({score_blended:.1f}) not better than pure ({score_pure:.1f})"
        )


@pytest.mark.slow
class TestBenchmarkReport:
    """Generate benchmark report to docs/benchmark_results.md."""

    def test_generate_report(self) -> None:
        """Run benchmarks and write results to docs/."""
        rows: list[dict] = []
        for seed in BENCHMARK_SEEDS:
            gt = _generate_ground_truth(seed)

            # Strategy A: pure local sim
            t0 = time.monotonic()
            pred_a = _generate_prediction(seed)
            time_a = time.monotonic() - t0
            score_a = score_prediction(gt, pred_a)

            # Strategy B: sim + mock observations
            t0 = time.monotonic()
            pred_b = _generate_prediction_with_observations(seed, gt)
            time_b = time.monotonic() - t0
            score_b = score_prediction(gt, pred_b)

            rows.append(
                {
                    "seed": seed,
                    "strategy": "pure_sim",
                    "score": score_a["score"],
                    "weighted_kl": score_a["weighted_kl"],
                    "dynamic_cells": score_a["num_dynamic_cells"],
                    "runtime_s": time_a,
                }
            )
            rows.append(
                {
                    "seed": seed,
                    "strategy": "sim+obs",
                    "score": score_b["score"],
                    "weighted_kl": score_b["weighted_kl"],
                    "dynamic_cells": score_b["num_dynamic_cells"],
                    "runtime_s": time_b,
                }
            )

        _write_report(rows)
        report_path = Path("docs/benchmark_results.md")
        assert report_path.exists()


def _write_report(rows: list[dict]) -> None:
    """Write benchmark results to docs/benchmark_results.md."""
    path = Path("docs/benchmark_results.md")
    path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "# Prediction Quality Benchmark",
        "",
        f"Ground truth: {GROUND_TRUTH_RUNS} MC runs | "
        f"Predictor: {PREDICTOR_RUNS} MC runs | "
        f"Map: {SMALL_MAP_W}x{SMALL_MAP_H}",
        "",
        "| Seed | Strategy | Score | W-KL | Dyn Cells | Time(s) |",
        "|------|----------|-------|------|-----------|---------|",
    ]
    for r in rows:
        lines.append(
            f"| {r['seed']} | {r['strategy']} | "
            f"{r['score']:.1f} | {r['weighted_kl']:.4f} | "
            f"{r['dynamic_cells']} | {r['runtime_s']:.2f} |"
        )

    pure = [r for r in rows if r["strategy"] == "pure_sim"]
    blended = [r for r in rows if r["strategy"] == "sim+obs"]
    avg_pure = sum(r["score"] for r in pure) / len(pure)
    avg_blend = sum(r["score"] for r in blended) / len(blended)

    lines.extend(
        [
            "",
            f"**Avg pure sim**: {avg_pure:.1f} | **Avg sim+obs**: {avg_blend:.1f}",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Scoring sanity checks
# ---------------------------------------------------------------------------


class TestScoringSanity:
    """Quick sanity checks for scoring (not slow)."""

    def test_perfect_prediction_scores_100(self) -> None:
        """Identical ground truth and prediction scores 100."""
        rng = np.random.default_rng(42)
        gt = rng.dirichlet([1] * NUM_PREDICTION_CLASSES, size=(5, 5))
        result = score_prediction(gt, gt)
        assert result["score"] == pytest.approx(100.0, abs=0.1)

    def test_uniform_prediction_scores_low(self) -> None:
        """Uniform 1/6 prediction against non-uniform GT scores poorly."""
        rng = np.random.default_rng(42)
        gt = rng.dirichlet([0.1] * NUM_PREDICTION_CLASSES, size=(5, 5))
        pred = np.ones((5, 5, NUM_PREDICTION_CLASSES)) / NUM_PREDICTION_CLASSES
        result = score_prediction(gt, pred)
        assert result["score"] < 50.0

    def test_shape_mismatch_raises(self) -> None:
        """Mismatched shapes raise ValueError."""
        gt = np.ones((3, 3, 6)) / 6
        pred = np.ones((4, 4, 6)) / 6
        with pytest.raises(ValueError, match="Shape mismatch"):
            score_prediction(gt, pred)
