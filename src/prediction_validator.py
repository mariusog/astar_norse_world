"""Pre-submission validation for prediction arrays.

Catches common bugs that caused bad scores in past rounds:
- R2: Laplace smoothing made predictions near-uniform
- R5: Regime detection misclassified survive as collapse
- R6: unified_priors returned uniform 1/6 priors due to wrong data path
"""

from __future__ import annotations

import logging

import numpy as np

from src.constants import (
    NUM_PREDICTION_CLASSES,
    PROBABILITY_FLOOR,
)
from src.terrain import InternalTerrain, Terrain

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Validation thresholds (referenced from constants where possible)
# ---------------------------------------------------------------------------

NORMALIZATION_TOLERANCE = 0.01
MIN_MAX_PROB_THRESHOLD = 0.3
MIN_CONFIDENT_CELL_FRACTION = 0.90
STATIC_TERRAIN_MIN_PROB = 0.9


def validate_predictions(
    predictions: list[np.ndarray],
    grids: list[np.ndarray],
) -> list[str]:
    """Run sanity checks on predictions before submission.

    Args:
        predictions: list of 5 H x W x 6 tensors (one per seed).
        grids: list of 5 H x W initial grids (InternalTerrain values).

    Returns:
        List of error messages. Empty list means all checks passed.
    """
    errors: list[str] = []
    errors.extend(_check_count_and_shapes(predictions, grids))
    if errors:
        return errors

    for i, (pred, grid) in enumerate(zip(predictions, grids, strict=True)):
        errors.extend(_check_single_prediction(pred, grid, seed_idx=i))

    errors.extend(_check_non_trivial(predictions))
    errors.extend(_check_prior_consistency(predictions, grids))
    return errors


def _check_count_and_shapes(
    predictions: list[np.ndarray],
    grids: list[np.ndarray],
) -> list[str]:
    """Validate list lengths and array shapes."""
    errors: list[str] = []
    if len(predictions) != len(grids):
        errors.append(f"Prediction count ({len(predictions)}) != grid count ({len(grids)})")
        return errors

    for i, pred in enumerate(predictions):
        if pred.ndim != 3 or pred.shape[2] != NUM_PREDICTION_CLASSES:
            errors.append(
                f"Seed {i}: shape {pred.shape}, expected (H, W, {NUM_PREDICTION_CLASSES})"
            )
        elif i < len(grids):
            h, w = grids[i].shape[:2]
            if pred.shape[0] != h or pred.shape[1] != w:
                errors.append(f"Seed {i}: pred shape {pred.shape[:2]} != grid shape ({h}, {w})")
    return errors


def _check_single_prediction(
    pred: np.ndarray,
    grid: np.ndarray,
    seed_idx: int,
) -> list[str]:
    """Run per-seed checks on one prediction tensor."""
    errors: list[str] = []
    errors.extend(_check_normalization(pred, seed_idx))
    errors.extend(_check_no_uniform(pred, seed_idx))
    errors.extend(_check_probability_floor(pred, seed_idx))
    errors.extend(_check_static_terrain(pred, grid, seed_idx))
    return errors


def _check_normalization(pred: np.ndarray, seed_idx: int) -> list[str]:
    """Each cell's probabilities must sum to 1.0 within tolerance."""
    sums = pred.sum(axis=2)
    bad_mask = np.abs(sums - 1.0) > NORMALIZATION_TOLERANCE
    bad_count = int(bad_mask.sum())
    if bad_count > 0:
        worst = float(np.max(np.abs(sums - 1.0)))
        return [f"Seed {seed_idx}: {bad_count} cells not normalized (worst deviation: {worst:.4f})"]
    return []


def _check_no_uniform(pred: np.ndarray, seed_idx: int) -> list[str]:
    """At least 90% of cells should have max prob > 0.3."""
    max_probs = pred.max(axis=2)
    confident = float((max_probs > MIN_MAX_PROB_THRESHOLD).mean())
    if confident < MIN_CONFIDENT_CELL_FRACTION:
        return [
            f"Seed {seed_idx}: only {confident:.1%} cells have max prob > "
            f"{MIN_MAX_PROB_THRESHOLD} (need {MIN_CONFIDENT_CELL_FRACTION:.0%}). "
            f"Predictions may be near-uniform."
        ]
    return []


def _check_probability_floor(pred: np.ndarray, seed_idx: int) -> list[str]:
    """No probability value should be below the floor minus tolerance."""
    floor_threshold = PROBABILITY_FLOOR - 0.001
    below = float((pred < floor_threshold).sum())
    if below > 0:
        min_val = float(pred.min())
        return [
            f"Seed {seed_idx}: {int(below)} values below floor "
            f"{PROBABILITY_FLOOR} (min: {min_val:.6f})"
        ]
    return []


def _check_static_terrain(
    pred: np.ndarray,
    grid: np.ndarray,
    seed_idx: int,
) -> list[str]:
    """Ocean cells should predict class 0, mountain cells class 5."""
    errors: list[str] = []
    ocean_mask = grid == InternalTerrain.OCEAN
    mountain_mask = grid == InternalTerrain.MOUNTAIN

    if ocean_mask.any():
        ocean_probs = pred[ocean_mask, Terrain.EMPTY]
        bad_ocean = int((ocean_probs < STATIC_TERRAIN_MIN_PROB).sum())
        if bad_ocean > 0:
            return [
                f"Seed {seed_idx}: {bad_ocean} ocean cells have "
                f"class 0 prob < {STATIC_TERRAIN_MIN_PROB}"
            ]

    if mountain_mask.any():
        mtn_probs = pred[mountain_mask, Terrain.MOUNTAIN]
        bad_mtn = int((mtn_probs < STATIC_TERRAIN_MIN_PROB).sum())
        if bad_mtn > 0:
            errors.append(
                f"Seed {seed_idx}: {bad_mtn} mountain cells have "
                f"class 5 prob < {STATIC_TERRAIN_MIN_PROB}"
            )
    return errors


def _check_prior_consistency(
    predictions: list[np.ndarray],
    grids: list[np.ndarray],
    data_dir: str = "data/rounds",
) -> list[str]:
    """Verify predictions are consistent with freshly-built priors.

    Catches bugs where the submission pipeline applies wrong priors
    (e.g. collapse instead of survive, or stale cached priors).
    """
    try:
        from src.unified_priors import (
            build_distance_priors,
            build_unified_priors,
            predict_from_priors,
        )

        priors = build_unified_priors(data_dir)
        dist_priors = build_distance_priors(data_dir)
    except Exception:
        return []  # can't build priors, skip check

    errors: list[str] = []
    for i, (pred, grid) in enumerate(zip(predictions, grids, strict=True)):
        fresh = predict_from_priors(grid, priors, dist_priors)
        fresh = np.maximum(fresh, PROBABILITY_FLOOR)
        fresh = fresh / fresh.sum(axis=2, keepdims=True)

        # KL(fresh || pred) — how much does our prediction diverge from fresh priors?
        p = np.maximum(fresh, PROBABILITY_FLOOR)
        q = np.maximum(pred, PROBABILITY_FLOOR)
        kl = np.sum(p * np.log(p / q), axis=2)
        avg_kl = float(kl.mean())

        if avg_kl > PRIOR_CONSISTENCY_MAX_DIVERGENCE:
            errors.append(
                f"Seed {i}: prediction diverges from fresh priors "
                f"(avg KL={avg_kl:.3f}, max={PRIOR_CONSISTENCY_MAX_DIVERGENCE}). "
                f"Wrong priors or stale cache?"
            )
    return errors


def _check_non_trivial(predictions: list[np.ndarray]) -> list[str]:
    """Predictions should differ across seeds (not identical copies)."""
    if len(predictions) < 2:
        return []
    ref = predictions[0]
    all_identical = all(np.allclose(ref, p, atol=1e-8) for p in predictions[1:])
    if all_identical:
        return ["All seed predictions are identical -- likely a bug"]
    return []


# ---------------------------------------------------------------------------
# Backtest sanity check (catches wrong-regime bugs like R5)
# ---------------------------------------------------------------------------

BACKTEST_MIN_SCORE = 65.0
PRIOR_CONSISTENCY_MAX_DIVERGENCE = 0.05  # max avg KL between prediction and fresh priors


def backtest_check(
    predictions: list[np.ndarray],
    grids: list[np.ndarray],
    data_dir: str = "data/rounds",
) -> list[str]:
    """Score predictions against the latest historical round as sanity check.

    If the same prior pipeline scores < BACKTEST_MIN_SCORE on a known
    round, something is likely wrong (e.g. wrong regime priors).

    Args:
        predictions: The actual predictions to submit (unused directly,
            but we rebuild from the same pipeline for comparison).
        grids: Initial grids for the current round.
        data_dir: Path to historical round data.

    Returns:
        List of warning messages. Empty = backtest passed.
    """
    from pathlib import Path

    from src.scoring import score_prediction

    rounds_dir = Path(data_dir)
    if not rounds_dir.exists():
        return []

    # Find the most recent round with GT
    import json

    best_round = None
    best_rnum = 0
    for rd in rounds_dir.iterdir():
        if not rd.is_dir():
            continue
        rj = rd / "round.json"
        gt0 = rd / "seed_0" / "ground_truth.npy"
        if rj.exists() and gt0.exists():
            rdata = json.loads(rj.read_text())
            rnum = rdata.get("round_number", 0)
            if rnum > best_rnum:
                best_rnum = rnum
                best_round = rd

    if best_round is None:
        return []

    # Build predictions for the reference round using the SAME pipeline
    from src.unified_priors import (
        build_distance_priors,
        build_unified_priors,
        predict_from_priors,
    )

    priors = build_unified_priors(data_dir)
    dist_priors = build_distance_priors(data_dir)

    scores = []
    for i in range(5):
        gt_path = best_round / f"seed_{i}" / "ground_truth.npy"
        grid_path = best_round / f"seed_{i}" / "initial_grid.npy"
        if not gt_path.exists() or not grid_path.exists():
            continue
        gt = np.load(gt_path)
        grid = np.load(grid_path)
        pred = predict_from_priors(grid, priors, dist_priors)
        pred = np.maximum(pred, PROBABILITY_FLOOR)
        pred = pred / pred.sum(axis=2, keepdims=True)
        scores.append(score_prediction(gt, pred)["score"])

    if not scores:
        return []

    avg = float(np.mean(scores))
    if avg < BACKTEST_MIN_SCORE:
        return [
            f"Backtest FAILED: priors score {avg:.1f} on R{best_rnum} "
            f"(need >= {BACKTEST_MIN_SCORE}). Pipeline may be broken."
        ]

    logger.info("Backtest passed: priors score %.1f on R%d", avg, best_rnum)
    return []
