"""LOO backtest for Phase 2 improvements.

For each round, trains priors on other rounds, predicts, applies
error correction and online calibration, then scores against GT.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.constants import NUM_PREDICTION_CLASSES, PROBABILITY_FLOOR
from src.scoring import score_prediction
from src.terrain import SERVER_TO_INTERNAL, InternalTerrain
from src.transforms import compute_error_corrections, error_correction

logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "rounds"

NUM_TERRAIN_TYPES = 7


def discover_rounds() -> list[dict]:
    """Find rounds sorted by round_number."""
    rounds = []
    for rd in sorted(DATA_DIR.iterdir()):
        rj = rd / "round.json"
        if not rj.exists():
            continue
        with open(rj) as f:
            meta = json.load(f)
        meta["_dir"] = str(rd)
        rounds.append(meta)
    rounds.sort(key=lambda r: r["round_number"])
    return rounds


def build_priors_loo(
    rounds: list[dict],
    exclude_rn: int,
) -> dict[int, np.ndarray]:
    """Build terrain priors excluding one round."""
    sums = {t: np.zeros(NUM_PREDICTION_CLASSES) for t in range(NUM_TERRAIN_TYPES)}
    counts = {t: 0 for t in range(NUM_TERRAIN_TYPES)}
    for r in rounds:
        if r["round_number"] == exclude_rn:
            continue
        _accumulate(r, sums, counts)
    priors = {}
    uniform = np.full(NUM_PREDICTION_CLASSES, 1.0 / NUM_PREDICTION_CLASSES)
    for t in range(NUM_TERRAIN_TYPES):
        priors[t] = sums[t] / counts[t] if counts[t] > 0 else uniform.copy()
    return priors


def _accumulate(
    r: dict,
    sums: dict,
    counts: dict,
) -> None:
    """Accumulate GT data from one round."""
    rd = Path(r["_dir"])
    states = r.get("initial_states", [])
    for si in range(len(states)):
        gt_path = rd / f"seed_{si}" / "ground_truth.npy"
        if not gt_path.exists():
            continue
        gt = np.load(gt_path)
        grid_raw = np.array(states[si]["grid"])
        internal = _to_internal(grid_raw)
        for t in range(NUM_TERRAIN_TYPES):
            mask = internal == t
            n = int(mask.sum())
            if n > 0:
                sums[t] += gt[mask].sum(axis=0)
                counts[t] += n


def predict_baseline(
    grid: np.ndarray,
    priors: dict[int, np.ndarray],
) -> np.ndarray:
    """Flat prior prediction (baseline)."""
    h, w = grid.shape
    pred = np.zeros((h, w, NUM_PREDICTION_CLASSES), dtype=np.float64)
    for t, vec in priors.items():
        mask = grid == t
        if mask.any():
            pred[mask] = vec
    # Static overrides
    _apply_static(pred, grid)
    return _floor_norm(pred)


def predict_with_corrections(
    grid: np.ndarray,
    priors: dict[int, np.ndarray],
    corrections: dict,
) -> np.ndarray:
    """Prior prediction + error correction transform."""
    pred = predict_baseline(grid, priors)
    return error_correction(pred, grid, corrections)


def _apply_static(pred: np.ndarray, grid: np.ndarray) -> None:
    """Override static terrain cells."""
    conf = 0.99
    residual = (1.0 - conf) / (NUM_PREDICTION_CLASSES - 1)
    for terrain, cls_idx in [
        (InternalTerrain.OCEAN, 0),
        (InternalTerrain.MOUNTAIN, 5),
    ]:
        mask = grid == terrain
        if mask.any():
            pred[mask] = residual
            pred[mask, cls_idx] = conf


def _floor_norm(tensor: np.ndarray) -> np.ndarray:
    """Floor and normalize."""
    safe = np.maximum(tensor, PROBABILITY_FLOOR)
    return safe / safe.sum(axis=-1, keepdims=True)


def _to_internal(grid_raw: np.ndarray) -> np.ndarray:
    """Convert server codes to internal terrain."""
    return np.vectorize(lambda v: SERVER_TO_INTERNAL.get(v, 1))(grid_raw).astype(np.int8)


def run_backtest() -> None:
    """Run LOO backtest on all available rounds."""
    rounds = discover_rounds()
    print(f"Found {len(rounds)} rounds: {[r['round_number'] for r in rounds]}")
    print()

    baseline_scores = {}
    corrected_scores = {}

    for target in rounds:
        rn = target["round_number"]
        rd = Path(target["_dir"])
        exclude = {rn}

        # Build priors excluding target
        priors = build_priors_loo(rounds, rn)

        # Build error corrections excluding target
        corrections = compute_error_corrections(str(DATA_DIR), exclude)

        states = target.get("initial_states", [])
        b_scores = []
        c_scores = []

        for si in range(len(states)):
            gt_path = rd / f"seed_{si}" / "ground_truth.npy"
            if not gt_path.exists():
                continue
            gt = np.load(gt_path)
            grid_raw = np.array(states[si]["grid"])
            grid = _to_internal(grid_raw)

            # Baseline
            pred_base = predict_baseline(grid, priors)
            s_base = score_prediction(gt, pred_base)
            b_scores.append(s_base["score"])

            # With corrections
            pred_corr = predict_with_corrections(grid, priors, corrections)
            s_corr = score_prediction(gt, pred_corr)
            c_scores.append(s_corr["score"])

        baseline_scores[rn] = b_scores
        corrected_scores[rn] = c_scores

    # Print results
    print("=" * 65)
    print(f"{'Round':>6} {'Baseline':>10} {'Corrected':>10} {'Delta':>8} {'Seeds':>6}")
    print("-" * 65)

    all_base = []
    all_corr = []
    for rn in sorted(baseline_scores.keys()):
        b_avg = np.mean(baseline_scores[rn])
        c_avg = np.mean(corrected_scores[rn])
        delta = c_avg - b_avg
        n_seeds = len(baseline_scores[rn])
        all_base.extend(baseline_scores[rn])
        all_corr.extend(corrected_scores[rn])
        print(f"R{rn:>4}  {b_avg:>10.2f} {c_avg:>10.2f} {delta:>+8.2f} {n_seeds:>6}")

    print("-" * 65)
    avg_base = np.mean(all_base)
    avg_corr = np.mean(all_corr)
    delta = avg_corr - avg_base
    print(f"{'AVG':>6}  {avg_base:>10.2f} {avg_corr:>10.2f} {delta:>+8.2f}")
    print("=" * 65)


if __name__ == "__main__":
    run_backtest()
