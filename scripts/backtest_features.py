"""Strict leave-one-out backtest for terrain prior and feature models.

Prevents data leakage by building priors ONLY from non-target rounds.
Supports --flat (terrain-type LOO baseline) and --features (feature-enhanced).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np

from src.constants import NUM_INTERNAL_TYPES, NUM_PREDICTION_CLASSES
from src.scoring import score_prediction
from src.transforms import apply_floor_to_row

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "rounds"
DOCS_DIR = PROJECT_ROOT / "docs"


def discover_rounds() -> list[dict[str, Any]]:
    """Find all round directories, sorted by round_number."""
    rounds: list[dict[str, Any]] = []
    if not DATA_DIR.exists():
        return rounds
    for round_dir in sorted(DATA_DIR.iterdir()):
        meta_path = round_dir / "round.json"
        if not meta_path.exists():
            continue
        with open(meta_path) as f:
            meta = json.load(f)
        meta["_dir"] = round_dir
        rounds.append(meta)
    rounds.sort(key=lambda r: r["round_number"])
    return rounds


def load_seed_data(round_dir: Path) -> list[dict[str, np.ndarray]]:
    """Load initial grids and ground truths for all seeds in a round."""
    seeds = []
    for sd in sorted(round_dir.glob("seed_*")):
        init_path = sd / "initial_grid.npy"
        gt_path = sd / "ground_truth.npy"
        if init_path.exists() and gt_path.exists():
            seeds.append(
                {
                    "initial": np.load(init_path),
                    "gt": np.load(gt_path),
                }
            )
    return seeds


def accumulate_soft_gt(
    initial: np.ndarray,
    gt: np.ndarray,
    counts: np.ndarray,
    totals: np.ndarray,
) -> None:
    """Accumulate soft GT distributions by initial terrain type.

    Args:
        initial: (H, W) InternalTerrain grid.
        gt: (H, W, 6) probability distribution ground truth.
        counts: (NUM_INTERNAL_TYPES, 6) running sum of GT probabilities.
        totals: (NUM_INTERNAL_TYPES,) running count of cells per type.
    """
    for init_type in range(NUM_INTERNAL_TYPES):
        mask = initial == init_type
        n = int(mask.sum())
        if n == 0:
            continue
        counts[init_type] += gt[mask].sum(axis=0)
        totals[init_type] += n


def build_loo_terrain_priors(
    all_rounds: list[dict[str, Any]],
    exclude_idx: int,
) -> np.ndarray:
    """Build terrain-type priors from all rounds EXCEPT exclude_idx.

    Returns (NUM_INTERNAL_TYPES, 6) normalized probability distributions.
    """
    counts = np.zeros((NUM_INTERNAL_TYPES, NUM_PREDICTION_CLASSES), dtype=np.float64)
    totals = np.zeros(NUM_INTERNAL_TYPES, dtype=np.float64)
    for i, rnd in enumerate(all_rounds):
        if i == exclude_idx:
            continue
        for seed_data in load_seed_data(rnd["_dir"]):
            accumulate_soft_gt(seed_data["initial"], seed_data["gt"], counts, totals)
    return _normalize_counts(counts, totals)


def _normalize_counts(
    counts: np.ndarray,
    totals: np.ndarray,
) -> np.ndarray:
    """Normalize accumulated counts to probability distributions."""
    priors = np.full(
        (NUM_INTERNAL_TYPES, NUM_PREDICTION_CLASSES),
        1.0 / NUM_PREDICTION_CLASSES,
        dtype=np.float64,
    )
    for i in range(NUM_INTERNAL_TYPES):
        if totals[i] > 0:
            priors[i] = counts[i] / totals[i]
        priors[i] = apply_floor_to_row(priors[i])
    return priors


def predict_from_priors(
    initial_grid: np.ndarray,
    priors: np.ndarray,
) -> np.ndarray:
    """Apply terrain priors to initial grid, returning (H, W, 6) tensor."""
    idx = np.clip(initial_grid.astype(np.int32), 0, priors.shape[0] - 1)
    return priors[idx].copy()


def score_one_seed(gt: np.ndarray, prediction: np.ndarray) -> float:
    """Score a single seed prediction against ground truth."""
    return score_prediction(gt, prediction)["score"]


def run_loo_backtest(
    all_rounds: list[dict[str, Any]],
    mode: str,
) -> list[dict[str, Any]]:
    """Run LOO backtest across all rounds for the given mode.

    Args:
        all_rounds: List of round metadata dicts.
        mode: 'flat' for terrain-type LOO priors, 'features' for enhanced.
    """
    results = []
    for target_idx, target_round in enumerate(all_rounds):
        rnum = target_round["round_number"]
        priors = _build_priors_for_mode(all_rounds, target_idx, mode)
        seeds = load_seed_data(target_round["_dir"])
        seed_scores = [
            score_one_seed(sd["gt"], predict_from_priors(sd["initial"], priors)) for sd in seeds
        ]
        avg = float(np.mean(seed_scores)) if seed_scores else 0.0
        results.append({"round": rnum, "avg": avg, "seeds": seed_scores})
        logger.info("R%d [%s]: avg=%.1f", rnum, mode, avg)
    return results


def _build_priors_for_mode(
    all_rounds: list[dict[str, Any]],
    target_idx: int,
    mode: str,
) -> np.ndarray:
    """Build priors based on mode. Features mode defers to feature_predictor."""
    if mode == "features":
        return _build_feature_priors(all_rounds, target_idx)
    return build_loo_terrain_priors(all_rounds, target_idx)


def _build_feature_priors(
    all_rounds: list[dict[str, Any]],
    target_idx: int,
) -> np.ndarray:
    """Build feature-enhanced priors (delegates to feature_predictor)."""
    from src.feature_predictor import build_feature_priors

    training = [r for i, r in enumerate(all_rounds) if i != target_idx]
    return build_feature_priors(training)


def format_report(
    flat_results: list[dict[str, Any]],
    feature_results: list[dict[str, Any]] | None,
) -> str:
    """Generate markdown report (under 40 lines)."""
    lines = [
        "# Feature Backtest (strict LOO)",
        "",
        "Each round scored using priors built from OTHER rounds only.",
        "",
    ]
    lines.extend(_format_table("Flat Terrain Priors (baseline)", flat_results))
    if feature_results:
        lines.append("")
        lines.extend(_format_table("Feature Priors", feature_results))
    lines.append("")
    lines.extend(_format_summary(flat_results, feature_results))
    return "\n".join(lines)


def _format_table(
    title: str,
    results: list[dict[str, Any]],
) -> list[str]:
    """Format one mode's results as a markdown table."""
    n_seeds = len(results[0]["seeds"]) if results else 0
    header = "| Round | Avg | " + " | ".join(f"S{i}" for i in range(n_seeds)) + " |"
    sep = "|" + "------|" * (2 + n_seeds)
    lines = [f"## {title}", "", header, sep]
    for r in results:
        cells = " | ".join(f"{s:.1f}" for s in r["seeds"])
        lines.append(f"| R{r['round']} | {r['avg']:.1f} | {cells} |")
    overall = float(np.mean([r["avg"] for r in results]))
    pad = " | ".join([""] * n_seeds)
    lines.append(f"| **Overall** | **{overall:.1f}** | {pad} |")
    return lines


def _format_summary(
    flat: list[dict[str, Any]],
    feat: list[dict[str, Any]] | None,
) -> list[str]:
    """Generate summary section."""
    flat_avg = float(np.mean([r["avg"] for r in flat]))
    lines = ["## Summary", "", f"- Flat terrain LOO avg: **{flat_avg:.1f}**"]
    if feat:
        feat_avg = float(np.mean([r["avg"] for r in feat]))
        delta = feat_avg - flat_avg
        lines.append(f"- Feature LOO avg: **{feat_avg:.1f}** (delta: {delta:+.1f})")
    return lines


def write_report(report: str) -> Path:
    """Write report to docs/feature_backtest.md."""
    out = DOCS_DIR / "feature_backtest.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        f.write(report)
    logger.info("Report written to %s", out)
    return out


def main() -> None:
    """Run the LOO backtest."""
    parser = argparse.ArgumentParser(description="LOO backtest for feature model")
    parser.add_argument("--flat", action="store_true", help="Terrain-type LOO priors (baseline)")
    parser.add_argument("--features", action="store_true", help="Feature-enhanced priors")
    args = parser.parse_args()

    if not args.flat and not args.features:
        args.flat = True

    rounds = discover_rounds()
    if len(rounds) < 2:
        logger.error("Need at least 2 rounds for LOO, found %d", len(rounds))
        sys.exit(1)
    logger.info("Found %d rounds for LOO backtest", len(rounds))

    flat_results = run_loo_backtest(rounds, mode="flat")
    feature_results = None

    if args.features:
        try:
            from src.feature_predictor import build_feature_priors  # noqa: F401

            feature_results = run_loo_backtest(rounds, mode="features")
        except ImportError:
            logger.warning("src/feature_predictor.py not found -- skipping features mode")

    report = format_report(flat_results, feature_results)
    write_report(report)


if __name__ == "__main__":
    main()
