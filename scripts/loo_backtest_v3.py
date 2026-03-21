"""LOO backtest using the submit_v3 pipeline (offline, no API calls).

For each round:
  1. Exclude it from training data
  2. Infer regime from ground truth (proxy for probe result)
  3. Train XGBoost on regime-matched remaining rounds
  4. Build flat priors from regime-matched remaining rounds
  5. Ensemble, power-transform, regime transforms (no observation blending)
  6. Score against ground truth

Usage:
    python -m scripts.loo_backtest_v3 2>&1 | tail -40
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.constants import NUM_PREDICTION_CLASSES, PROBABILITY_FLOOR
from src.ml_predictor import build_training_data, predict_grid, train_model
from src.scoring import score_prediction
from src.terrain import InternalTerrain

logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "rounds"

# Mirror from submit_v3
_REGIME_INCLUDE: dict[str, set[int]] = {
    "survive": {1, 2, 4, 5, 9},
    "aggressive": {6, 7},
    "deep_collapse": {3, 4, 8, 9, 10},
    "partial_collapse": {1, 2, 4, 5, 9},
}

_REGIME_ENSEMBLE: dict[str, float] = {
    "survive": 0.9,
    "aggressive": 0.4,
    "deep_collapse": 0.7,
    "partial_collapse": 0.9,
}

_REGIME_POWER: dict[str, float] = {
    "survive": 0.9,
    "aggressive": 0.8,
    "deep_collapse": 1.0,
    "partial_collapse": 1.05,
}

_REGIME_TRANSFORMS: dict[str, list[tuple[str, dict]]] = {
    "survive": [("spatial_smooth", {"sigma": 0.3})],
    "aggressive": [],
    "deep_collapse": [("collapse_shift", {"threshold": 0.3})],
    "partial_collapse": [],
}

# Known regime for each round (ground truth from historical analysis)
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
}


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


def _build_flat_priors_loo(rounds: list[dict], exclude_rn: int, regime: str) -> np.ndarray:
    """Build regime-matched flat priors excluding one round."""
    include = _REGIME_INCLUDE.get(regime)
    accum = np.zeros((7, NUM_PREDICTION_CLASSES))
    count = np.zeros(7)
    for r in rounds:
        rn = r["round_number"]
        if rn == exclude_rn:
            continue
        if include and rn not in include:
            continue
        rd = Path(r["_dir"])
        for i in range(5):
            gt_p = rd / f"seed_{i}" / "ground_truth.npy"
            gr_p = rd / f"seed_{i}" / "initial_grid.npy"
            if not gt_p.exists() or not gr_p.exists():
                continue
            gt, gr = np.load(gt_p), np.load(gr_p)
            for t in range(7):
                mask = gr == t
                if mask.sum() > 0:
                    accum[t] += gt[mask].sum(axis=0)
                    count[t] += mask.sum()
    priors = np.zeros((7, NUM_PREDICTION_CLASSES))
    for t in range(7):
        priors[t] = accum[t] / count[t] if count[t] > 0 else 1.0 / NUM_PREDICTION_CLASSES
    return priors


def _floor_and_normalize(tensor: np.ndarray) -> np.ndarray:
    safe = np.maximum(tensor, PROBABILITY_FLOOR)
    return safe / safe.sum(axis=-1, keepdims=True)


def _build_prediction(
    grid: np.ndarray,
    model,
    regime: str,
    flat_priors: np.ndarray,
) -> np.ndarray:
    """v3 prediction pipeline (no observation blending for offline backtest)."""
    from web.transforms import apply_transform_chain

    # Step 1: XGBoost
    pred = predict_grid(grid, model)

    # Step 2: Regime ensemble with flat priors
    gi = np.clip(grid.astype(np.int32), 0, flat_priors.shape[0] - 1)
    fp = flat_priors[gi].copy()
    fp[grid == InternalTerrain.OCEAN] = [1, 0, 0, 0, 0, 0]
    fp[grid == InternalTerrain.MOUNTAIN] = [0, 0, 0, 0, 0, 1]
    fp = np.maximum(fp, PROBABILITY_FLOOR)
    fp = fp / fp.sum(axis=2, keepdims=True)
    w_xgb = _REGIME_ENSEMBLE.get(regime, 0.7)
    pred = w_xgb * pred + (1 - w_xgb) * fp

    # Step 3: Power transform
    power = _REGIME_POWER.get(regime, 0.9)
    pred = np.power(np.maximum(pred, 1e-10), power)
    pred = pred / pred.sum(axis=-1, keepdims=True)

    # Step 4: Regime transforms
    transforms = _REGIME_TRANSFORMS.get(regime, [])
    pred = apply_transform_chain(pred, grid, transforms)

    return _floor_and_normalize(pred)


def run_loo_backtest() -> None:
    rounds = _discover_rounds()
    print(f"Rounds: {[r['round_number'] for r in rounds]}")
    print()

    results: dict[int, list[float]] = {}

    for target in rounds:
        rn = target["round_number"]
        rd = Path(target["_dir"])

        seed_scores = []
        regime = _ROUND_REGIME.get(rn, "survive")

        # Build exclude set: all rounds not in regime include + target
        include = _REGIME_INCLUDE.get(regime)
        if include:
            exclude = (set(range(1, 100)) - include) | {rn}
        else:
            exclude = {rn}

        x, y = build_training_data(str(DATA_DIR), exclude_round_numbers=exclude)
        if len(x) == 0:
            # Fallback: train on all except target
            x, y = build_training_data(str(DATA_DIR), exclude_round_numbers={rn})
        model = train_model(x, y, seed=42)

        # Build flat priors (LOO)
        flat_priors = _build_flat_priors_loo(rounds, rn, regime)

        for si in range(5):
            gt_p = rd / f"seed_{si}" / "ground_truth.npy"
            gr_p = rd / f"seed_{si}" / "initial_grid.npy"
            if not gt_p.exists() or not gr_p.exists():
                continue
            gt = np.load(gt_p)
            grid = np.load(gr_p)
            pred = _build_prediction(grid, model, regime, flat_priors)
            score = score_prediction(gt, pred)["score"]
            seed_scores.append(score)

        results[rn] = seed_scores
        avg = np.mean(seed_scores)
        print(f"R{rn:2d}  regime={regime:<16}  avg={avg:.2f}  seeds={seed_scores}")

    # Summary table
    print()
    print("=" * 55)
    print(f"{'Round':>6} {'Regime':<16} {'Avg':>8} {'Min':>8} {'Max':>8}")
    print("-" * 55)
    all_scores = []
    for rn in sorted(results):
        ss = results[rn]
        regime = _ROUND_REGIME.get(rn, "?")
        avg = np.mean(ss)
        all_scores.extend(ss)
        print(f"R{rn:>4}  {regime:<16} {avg:>8.2f} {min(ss):>8.2f} {max(ss):>8.2f}")
    print("-" * 55)
    print(f"{'AVG':>6}  {'':16} {np.mean(all_scores):>8.2f}")
    print("=" * 55)


if __name__ == "__main__":
    run_loo_backtest()
