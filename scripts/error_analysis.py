"""Per-round error breakdown: which terrain types and distance bins lose the most points.

Usage:
    python -m scripts.error_analysis 2>&1 | tail -80
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.constants import PROBABILITY_FLOOR, SCORE_ENTROPY_THRESHOLD
from src.features import compute_settlement_distance
from src.ml_predictor import build_training_data, predict_grid, train_model
from src.scoring import entropy, kl_divergence

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "rounds"

_ROUND_REGIME: dict[int, str] = {
    1: "survive",
    2: "survive",
    3: "deep_collapse",
    4: "partial_collapse",
    5: "survive",
    6: "aggressive",
    7: "aggressive",
    8: "deep_collapse",
    9: "partial_collapse",
    10: "deep_collapse",
    11: "aggressive",
    12: "aggressive",
    13: "partial_collapse",
    14: "aggressive",
    15: "aggressive",
    16: "survive",
    17: "aggressive",
    18: "aggressive",
    19: "deep_collapse",
    20: "survive",
    21: "survive",
}

_REGIME_INCLUDE: dict[str, set[int]] = {
    "survive": {1, 2, 4, 5, 9, 13, 16, 20, 21},
    "aggressive": {6, 7, 11, 12, 14, 15, 17, 18},
    "deep_collapse": {3, 4, 8, 9, 10, 13, 19},
    "partial_collapse": {1, 2, 4, 5, 9, 13, 16, 20, 21},
}

TERRAIN_NAMES = ["Ocean", "Plains", "Settlement", "Port", "Ruin", "Forest", "Mountain"]
DIST_BINS = [(0, 2), (2, 5), (5, 10), (10, 999)]
DIST_LABELS = ["0-2", "2-5", "5-10", "10+"]


def _discover_rounds() -> list[dict]:
    rounds = []
    for rd in sorted(DATA_DIR.iterdir()):
        rj = rd / "round.json"
        if rj.exists():
            meta = json.loads(rj.read_text())
            meta["_dir"] = rd
            rounds.append(meta)
    rounds.sort(key=lambda r: r["round_number"])
    return rounds


def main() -> None:
    rounds = _discover_rounds()

    # Aggregate error by (regime, terrain, distance_bin)
    regime_terrain_dist_loss: dict[str, np.ndarray] = {}  # regime -> (7, 4) weighted KL
    regime_terrain_dist_weight: dict[str, np.ndarray] = {}  # regime -> (7, 4) entropy sum

    worst_rounds: list[tuple[int, str, float, dict]] = []

    for target in rounds:
        rn = target["round_number"]
        rd = Path(target["_dir"])
        regime = _ROUND_REGIME.get(rn, "survive")

        # Train model (LOO)
        if regime in ("survive", "partial_collapse"):
            exclude = {rn}
        else:
            include = _REGIME_INCLUDE.get(regime)
            exclude = (set(range(1, 100)) - include) | {rn} if include else {rn}
        x, y = build_training_data(str(DATA_DIR), exclude_round_numbers=exclude)
        if len(x) == 0:
            x, y = build_training_data(str(DATA_DIR), exclude_round_numbers={rn})
        model = train_model(x, y, seed=42)

        # Score first seed only (representative)
        gt_p = rd / "seed_0" / "ground_truth.npy"
        gr_p = rd / "seed_0" / "initial_grid.npy"
        if not gt_p.exists():
            continue
        gt = np.load(gt_p)
        grid = np.load(gr_p)
        pred = predict_grid(grid, model)

        # Apply floor+norm like pipeline
        pred = np.maximum(pred, PROBABILITY_FLOOR)
        pred = pred / pred.sum(axis=-1, keepdims=True)

        cell_ent = entropy(gt)
        cell_kl = kl_divergence(gt, pred)
        dynamic = cell_ent >= SCORE_ENTROPY_THRESHOLD
        settle_dist = compute_settlement_distance(grid)

        # Per-terrain, per-distance breakdown
        terrain_loss = {}
        total_ent = cell_ent[dynamic].sum()

        for t in range(7):
            for di, (dlo, dhi) in enumerate(DIST_BINS):
                mask = dynamic & (grid == t) & (settle_dist >= dlo) & (settle_dist < dhi)
                if mask.sum() == 0:
                    continue
                ent_sum = cell_ent[mask].sum()
                kl_sum = (cell_ent[mask] * cell_kl[mask]).sum()
                pct = 100.0 * ent_sum / total_ent
                terrain_loss[(t, di)] = {
                    "ent_sum": ent_sum,
                    "kl_sum": kl_sum,
                    "pct_weight": pct,
                    "mean_kl": cell_kl[mask].mean(),
                    "count": int(mask.sum()),
                }

                if regime not in regime_terrain_dist_loss:
                    regime_terrain_dist_loss[regime] = np.zeros((7, 4))
                    regime_terrain_dist_weight[regime] = np.zeros((7, 4))
                regime_terrain_dist_loss[regime][t, di] += kl_sum
                regime_terrain_dist_weight[regime][t, di] += ent_sum

        # Compute round score
        weighted_kl = (cell_ent[dynamic] * cell_kl[dynamic]).sum() / cell_ent[dynamic].sum()
        score = 100.0 * np.exp(-3 * weighted_kl)

        # Find top error sources
        sorted_errs = sorted(terrain_loss.items(), key=lambda x: x[1]["kl_sum"], reverse=True)
        top3 = [
            (TERRAIN_NAMES[t], DIST_LABELS[di], v["kl_sum"], v["mean_kl"], v["count"])
            for (t, di), v in sorted_errs[:3]
        ]
        worst_rounds.append((rn, regime, score, top3))

    # Print per-round analysis
    print("=" * 90)
    header = "Top error sources (terrain @ dist: weighted_kl, mean_kl, n_cells)"
    print(f"{'Round':>5} {'Regime':<16} {'Score':>6}  {header}")
    print("-" * 90)
    worst_rounds.sort(key=lambda x: x[2])
    for rn, regime, score, top3 in worst_rounds:
        top_str = " | ".join(
            f"{t}@{d}: wkl={wkl:.3f} mkl={mkl:.3f} n={n}" for t, d, wkl, mkl, n in top3
        )
        print(f"R{rn:>3}  {regime:<16} {score:>6.1f}  {top_str}")

    # Aggregate by regime
    print()
    print("=" * 90)
    print("AGGREGATE ERROR BY REGIME x TERRAIN x DISTANCE")
    print("=" * 90)
    for regime in sorted(regime_terrain_dist_loss):
        loss = regime_terrain_dist_loss[regime]
        weight = regime_terrain_dist_weight[regime]
        total_loss = loss.sum()
        print(f"\n--- {regime} (total weighted KL = {total_loss:.3f}) ---")
        print(f"{'Terrain':<12} {'Dist':>6} {'%Loss':>7} {'MeanKL':>8} {'%Weight':>8}")
        entries = []
        for t in range(7):
            for di in range(4):
                if weight[t, di] > 0:
                    pct_loss = 100 * loss[t, di] / total_loss
                    mean_kl = loss[t, di] / weight[t, di]
                    pct_wt = 100 * weight[t, di] / weight.sum()
                    entries.append((t, di, pct_loss, mean_kl, pct_wt))
        entries.sort(key=lambda x: x[2], reverse=True)
        for t, di, pct_loss, mean_kl, pct_wt in entries[:8]:
            name = TERRAIN_NAMES[t]
            dist = DIST_LABELS[di]
            print(f"{name:<12} {dist:>6} {pct_loss:>6.1f}% {mean_kl:>8.4f} {pct_wt:>7.1f}%")


if __name__ == "__main__":
    main()
