"""Round-over-round analysis: terrain priors, stability, and rolling backtests."""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np

from src.constants import (
    NUM_PREDICTION_CLASSES,
    PROBABILITY_FLOOR,
    SCORE_DECAY_RATE,
    SCORE_ENTROPY_THRESHOLD,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "rounds"
DOCS_DIR = PROJECT_ROOT / "docs"

TERRAIN_NAMES = ["Empty", "Settlement", "Port", "Ruin", "Forest", "Mountain"]


def discover_rounds() -> list[dict[str, Any]]:
    """Find all round directories and load metadata, sorted by round_number."""
    rounds = []
    if not DATA_DIR.exists():
        logger.warning("Data directory %s does not exist", DATA_DIR)
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


def load_ground_truths(round_dir: Path) -> list[np.ndarray]:
    """Load all seed ground truth tensors for a round."""
    truths = []
    seed_dirs = sorted(round_dir.glob("seed_*"))
    for sd in seed_dirs:
        gt_path = sd / "ground_truth.npy"
        if gt_path.exists():
            truths.append(np.load(gt_path))
    return truths


def compute_terrain_priors(ground_truths: list[np.ndarray]) -> np.ndarray:
    """Mean probability vector per terrain class across seeds. Shape (6,)."""
    if not ground_truths:
        return np.zeros(NUM_PREDICTION_CLASSES)
    return np.stack(ground_truths, axis=0).mean(axis=(0, 1, 2))


def compute_per_cell_priors(ground_truths: list[np.ndarray]) -> np.ndarray:
    """Mean probability tensor averaged over seeds. Shape (H, W, 6)."""
    if not ground_truths:
        return np.array([])
    return np.stack(ground_truths, axis=0).mean(axis=0)


def kl_divergence_per_cell(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """KL(p || q) per cell. p and q are (H, W, 6)."""
    q_safe = np.maximum(q, PROBABILITY_FLOOR)
    p_safe = np.maximum(p, PROBABILITY_FLOOR)
    # Re-normalize after flooring
    q_safe = q_safe / q_safe.sum(axis=-1, keepdims=True)
    p_safe = p_safe / p_safe.sum(axis=-1, keepdims=True)
    return (p_safe * np.log(p_safe / q_safe)).sum(axis=-1)


def entropy_per_cell(p: np.ndarray) -> np.ndarray:
    """Shannon entropy per cell. p is (H, W, 6)."""
    p_safe = np.maximum(p, PROBABILITY_FLOOR)
    p_safe = p_safe / p_safe.sum(axis=-1, keepdims=True)
    return -(p_safe * np.log(p_safe)).sum(axis=-1)


def score_prediction(ground_truth: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    """Score a prediction against ground truth using competition formula."""
    ent = entropy_per_cell(ground_truth)
    dynamic_mask = ent > SCORE_ENTROPY_THRESHOLD
    num_dynamic = int(dynamic_mask.sum())
    if num_dynamic == 0:
        return {"score": 100.0, "weighted_kl": 0.0, "num_dynamic_cells": 0}
    kl = kl_divergence_per_cell(ground_truth, prediction)
    weight_sum = ent[dynamic_mask].sum()
    mean_wkl = (ent[dynamic_mask] * kl[dynamic_mask]).sum() / max(weight_sum, 1e-12)
    score = 100.0 * np.exp(-SCORE_DECAY_RATE * mean_wkl)
    return {"score": float(score), "weighted_kl": float(mean_wkl), "num_dynamic_cells": num_dynamic}


def compute_round_priors(
    rounds: list[dict[str, Any]],
) -> list[tuple[int, np.ndarray, np.ndarray]]:
    """Return (round_number, global_prior(6,), cell_prior(H,W,6)) per round."""
    results = []
    for r in rounds:
        gts = load_ground_truths(r["_dir"])
        if gts:
            results.append(
                (r["round_number"], compute_terrain_priors(gts), compute_per_cell_priors(gts))
            )
    return results


def compute_stability(
    round_priors: list[tuple[int, np.ndarray, np.ndarray]],
) -> np.ndarray:
    """Std dev of global priors across rounds. Returns shape (6,)."""
    if len(round_priors) < 2:
        return np.zeros(NUM_PREDICTION_CLASSES)
    priors = np.stack([rp[1] for rp in round_priors], axis=0)
    return priors.std(axis=0)


def rolling_backtest(
    rounds: list[dict[str, Any]],
    round_priors: list[tuple[int, np.ndarray, np.ndarray]],
) -> list[dict[str, Any]]:
    """Use round N-1 cell priors to predict round N; score each seed."""
    results = []
    prior_map = {rn: tensor for rn, _, tensor in round_priors}
    round_map = {r["round_number"]: r for r in rounds}
    for i in range(1, len(round_priors)):
        prev_rn, curr_rn = round_priors[i - 1][0], round_priors[i][0]
        gts = load_ground_truths(round_map[curr_rn]["_dir"])
        seed_scores = [score_prediction(gt, prior_map[prev_rn]) for gt in gts]
        mean_score = np.mean([s["score"] for s in seed_scores])
        results.append(
            {
                "predict_from": prev_rn,
                "predict_for": curr_rn,
                "mean_score": float(mean_score),
                "seed_scores": seed_scores,
            }
        )
    return results


def generate_report(
    round_priors: list[tuple[int, np.ndarray, np.ndarray]],
    stability: np.ndarray,
    backtest_results: list[dict[str, Any]],
) -> str:
    """Generate Tier 1 markdown report (under 40 lines)."""
    prior_rows = [(f"R{rn}", p) for rn, p, _ in round_priors]
    lines = [
        "# Round-over-Round Analysis",
        "",
        f"**Rounds analyzed**: {len(round_priors)}",
        "**Generated**: auto",
        "",
        "## Global Terrain Priors (mean probability per class)",
        "",
        _format_terrain_table("Round", prior_rows),
        "",
        "## Prior Stability (std dev across rounds)",
        "",
        _format_terrain_table("Metric", [("Std Dev", stability)]),
        "",
        "## Rolling Backtest (R(N-1) priors -> R(N) score)",
        "",
        _format_backtest_table(backtest_results),
        "",
        "## Key Findings",
        "",
    ]
    lines.extend(_key_findings(stability, backtest_results))
    return "\n".join(lines)


def _format_terrain_table(
    label_col: str,
    rows_data: list[tuple[str, np.ndarray]],
) -> str:
    """Format a table with terrain columns as markdown."""
    header = f"| {label_col} | " + " | ".join(TERRAIN_NAMES) + " |"
    sep = "|" + "-------|" * (len(TERRAIN_NAMES) + 1)
    rows = [header, sep]
    for label, vals in rows_data:
        cells = " | ".join(f"{v:.4f}" for v in vals)
        rows.append(f"| {label} | {cells} |")
    return "\n".join(rows)


def _format_backtest_table(results: list[dict[str, Any]]) -> str:
    """Format backtest results as markdown table."""
    if not results:
        return "No backtest possible (need 2+ rounds)."
    header = "| From | To | Mean Score | Min Seed | Max Seed |"
    sep = "|------|------|-----------|----------|----------|"
    rows = [header, sep]
    for r in results:
        ss = [s["score"] for s in r["seed_scores"]]
        rows.append(
            f"| R{r['predict_from']} | R{r['predict_for']} "
            f"| {r['mean_score']:.1f} | {min(ss):.1f} | {max(ss):.1f} |"
        )
    return "\n".join(rows)


def _key_findings(
    stability: np.ndarray,
    backtest_results: list[dict[str, Any]],
) -> list[str]:
    """Generate bullet-point findings."""
    findings = []
    most_stable = TERRAIN_NAMES[int(np.argmin(stability))]
    least_stable = TERRAIN_NAMES[int(np.argmax(stability))]
    findings.append(f"- **Most stable**: {most_stable} (std={stability.min():.4f})")
    findings.append(f"- **Least stable**: {least_stable} (std={stability.max():.4f})")
    if backtest_results:
        avg = np.mean([r["mean_score"] for r in backtest_results])
        findings.append(f"- **Avg rolling backtest score**: {avg:.1f}/100")
    return findings


def main() -> None:
    """Run the round-over-round analysis pipeline."""
    rounds = discover_rounds()
    logger.info("Found %d rounds", len(rounds))
    if not rounds:
        logger.error("No round data found in %s", DATA_DIR)
        sys.exit(1)

    round_priors = compute_round_priors(rounds)
    stability = compute_stability(round_priors)
    backtest_results = rolling_backtest(rounds, round_priors)

    report = generate_report(round_priors, stability, backtest_results)
    output_path = DOCS_DIR / "round_analysis.md"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(report)
    logger.info("Report written to %s", output_path)

    for bt in backtest_results:
        logger.info(
            "Backtest R%d->R%d: score=%.1f", bt["predict_from"], bt["predict_for"], bt["mean_score"]
        )


if __name__ == "__main__":
    main()
