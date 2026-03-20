"""Backtest prediction pipeline against historical rounds."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np

from src.constants import (
    LAPLACE_ALPHA,
    NUM_PREDICTION_CLASSES,
    OBSERVATION_WEIGHT,
    PROBABILITY_FLOOR,
    SIMULATION_WEIGHT,
)
from src.features import compute_settlement_distance
from src.scoring import score_prediction

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "rounds"
DOCS_DIR = PROJECT_ROOT / "docs"
DYNAMIC_DIST_THRESHOLD = 3
NUM_INTERNAL_TYPES = 7


def discover_rounds() -> list[dict[str, Any]]:
    """Find all round directories with metadata, sorted by round_number."""
    rounds: list[dict[str, Any]] = []
    if not DATA_DIR.exists():
        return rounds
    for rd in sorted(DATA_DIR.iterdir()):
        mp = rd / "round.json"
        if mp.exists():
            with open(mp) as f:
                meta = json.load(f)
            meta["_dir"] = rd
            rounds.append(meta)
    rounds.sort(key=lambda r: r["round_number"])
    return rounds


def load_seed_data(round_dir: Path) -> list[tuple[np.ndarray, np.ndarray]]:
    """Load (initial_grid, ground_truth) pairs for all seeds."""
    pairs: list[tuple[np.ndarray, np.ndarray]] = []
    for sd in sorted(round_dir.glob("seed_*")):
        gp, tp = sd / "initial_grid.npy", sd / "ground_truth.npy"
        if gp.exists() and tp.exists():
            pairs.append((np.load(gp), np.load(tp)))
    return pairs


def build_terrain_priors(
    rounds: list[dict[str, Any]],
    exclude_round_num: int | None = None,
) -> np.ndarray:
    """Build per-terrain-type priors, optionally excluding one round (LOO)."""
    counts = np.zeros((NUM_INTERNAL_TYPES, NUM_PREDICTION_CLASSES), dtype=np.float64)
    for r in rounds:
        if exclude_round_num is not None and r["round_number"] == exclude_round_num:
            continue
        for grid, gt in load_seed_data(r["_dir"]):
            for t in range(NUM_INTERNAL_TYPES):
                mask = grid == t
                if mask.any():
                    counts[t] += gt[mask].sum(axis=0)
    # Normalize with floor
    for i in range(NUM_INTERNAL_TYPES):
        s = counts[i].sum()
        counts[i] = counts[i] / s if s > 0 else 1.0 / NUM_PREDICTION_CLASSES
        counts[i] = np.maximum(counts[i], PROBABILITY_FLOOR)
        counts[i] /= counts[i].sum()
    return counts


def find_dynamic_cells(grid: np.ndarray) -> list[tuple[int, int]]:
    """Identify cells near settlements (within DYNAMIC_DIST_THRESHOLD)."""
    dist = compute_settlement_distance(grid)
    h, w = grid.shape
    return [(y, x) for y in range(h) for x in range(w) if dist[y, x] <= DYNAMIC_DIST_THRESHOLD]


def simulate_observations(
    gt: np.ndarray,
    dynamic_cells: list[tuple[int, int]],
    obs_per_cell: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Simulate observation counts by sampling from GT probabilities."""
    h, w, c = gt.shape
    counts = np.zeros((h, w, c), dtype=np.int32)
    for y, x in dynamic_cells:
        probs = gt[y, x]
        ps = probs.sum()
        if ps < 1e-12:
            continue
        safe_p = probs / ps
        for _ in range(obs_per_cell):
            counts[y, x, rng.choice(c, p=safe_p)] += 1
    return counts


def blend_with_prior(prior: np.ndarray, obs_counts: np.ndarray) -> np.ndarray:
    """Blend Laplace-smoothed observation frequencies into prior."""
    result = prior.copy()
    total_obs = obs_counts.sum(axis=-1)
    mask = total_obs > 0
    if not mask.any():
        return result
    smoothed = obs_counts[mask].astype(np.float64) + LAPLACE_ALPHA
    obs_probs = smoothed / smoothed.sum(axis=-1, keepdims=True)
    result[mask] = OBSERVATION_WEIGHT * obs_probs + SIMULATION_WEIGHT * prior[mask]
    result[mask] /= result[mask].sum(axis=-1, keepdims=True)
    return result


def apply_floor(pred: np.ndarray) -> np.ndarray:
    """Apply probability floor and renormalize."""
    safe = np.maximum(pred, PROBABILITY_FLOOR)
    return safe / safe.sum(axis=-1, keepdims=True)


def backtest_seed(
    grid: np.ndarray,
    gt: np.ndarray,
    priors: np.ndarray,
    obs_per_cell: int,
    rng: np.random.Generator,
) -> dict[str, float | int]:
    """Run backtest for a single seed."""
    pred = priors[np.clip(grid.astype(np.int32), 0, priors.shape[0] - 1)].copy()
    if obs_per_cell > 0:
        dyn = find_dynamic_cells(grid)
        obs = simulate_observations(gt, dyn, obs_per_cell, rng)
        pred = blend_with_prior(pred, obs)
    return score_prediction(gt, apply_floor(pred))


def backtest_round(
    rinfo: dict[str, Any],
    all_rounds: list[dict[str, Any]],
    obs_per_cell: int,
    loo: bool,
    seed: int,
) -> dict[str, Any]:
    """Backtest one round across all its seeds."""
    rnum = rinfo["round_number"]
    priors = build_terrain_priors(all_rounds, rnum if loo else None)
    rng = np.random.default_rng(seed)
    scores = [
        backtest_seed(g, gt, priors, obs_per_cell, rng)["score"]
        for g, gt in load_seed_data(rinfo["_dir"])
    ]
    return {
        "round_number": rnum,
        "avg_score": float(np.mean(scores)) if scores else 0.0,
        "seed_scores": scores,
        "num_seeds": len(scores),
    }


def run_backtest(loo: bool, obs_per_cell: int, seed: int) -> list[dict[str, Any]]:
    """Run full backtest across all rounds."""
    rounds = discover_rounds()
    if not rounds:
        logger.error("No round data found in %s", DATA_DIR)
        return []
    logger.info(
        "Backtesting %d rounds (mode=%s, obs/cell=%d, seed=%d)",
        len(rounds),
        "LOO" if loo else "full",
        obs_per_cell,
        seed,
    )
    results: list[dict[str, Any]] = []
    for r in rounds:
        res = backtest_round(r, rounds, obs_per_cell, loo, seed)
        results.append(res)
        logger.info("R%d: avg=%.1f", res["round_number"], res["avg_score"])
    return results


def generate_report(
    results: list[dict[str, Any]],
    mode: str,
    obs_per_cell: int,
    seed: int,
) -> str:
    """Generate Tier 1 markdown summary report."""
    lines = [
        "# Backtest Results",
        "",
        f"**Mode**: {mode} | **Obs/cell**: {obs_per_cell} | **Seed**: {seed}",
        f"**Rounds**: {len(results)}",
        "",
        "## Per-Round Scores",
        "",
        "| Round | Avg Score | Seeds | Min | Max |",
        "|-------|-----------|-------|-----|-----|",
    ]
    for r in results:
        ss = r["seed_scores"]
        mn, mx = (min(ss), max(ss)) if ss else (0, 0)
        lines.append(
            f"| R{r['round_number']} | {r['avg_score']:.1f} "
            f"| {r['num_seeds']} | {mn:.1f} | {mx:.1f} |"
        )
    avg = float(np.mean([r["avg_score"] for r in results])) if results else 0.0
    lines.extend(["", f"**Overall average**: {avg:.1f}/100", "", "## Key Findings", ""])
    if results:
        best = max(results, key=lambda r: r["avg_score"])
        worst = min(results, key=lambda r: r["avg_score"])
        lines.append(f"- Best round: R{best['round_number']} ({best['avg_score']:.1f})")
        lines.append(f"- Worst round: R{worst['round_number']} ({worst['avg_score']:.1f})")
    return "\n".join(lines)


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Backtest prediction pipeline")
    parser.add_argument("--loo", action="store_true", help="Leave-one-out mode")
    parser.add_argument("--full", action="store_true", help="Full data mode")
    parser.add_argument("--obs-per-cell", type=int, default=3, help="Obs per dynamic cell")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    loo = args.loo or not args.full
    results = run_backtest(loo, args.obs_per_cell, args.seed)
    if not results:
        sys.exit(1)

    mode = "LOO" if loo else "full"
    report = generate_report(results, mode, args.obs_per_cell, args.seed)
    output_path = DOCS_DIR / "backtest_results.md"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(report)
    logger.info("Report written to %s", output_path)


if __name__ == "__main__":
    main()
