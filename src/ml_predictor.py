"""XGBoost per-cell probability predictor.

Trains multi-output regressors on spatial features extracted from
terrain grids and predicts 6-class probability vectors per cell.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from scipy.ndimage import uniform_filter
from sklearn.multioutput import MultiOutputRegressor
from xgboost import XGBRegressor

from src.constants import (
    NUM_PREDICTION_CLASSES,
    PROBABILITY_FLOOR,
    STATIC_TERRAIN_CONFIDENCE,
)
from src.features import (
    compute_coastal_mask,
    compute_ocean_distance,
    compute_settlement_distance,
)
from src.terrain import InternalTerrain

logger = logging.getLogger(__name__)

# ── ML hyperparameters (in constants spirit, co-located for now) ──────────
ML_N_ESTIMATORS = 100
ML_MAX_DEPTH = 5
ML_LEARNING_RATE = 0.1
ML_DENSITY_KERNEL = 7  # 7x7 uniform filter for density features
NUM_TERRAIN_TYPES = 7  # InternalTerrain 0-6
NUM_FEATURES = NUM_TERRAIN_TYPES + 5  # one-hot + 5 scalar features


def extract_cell_features(grid: np.ndarray) -> np.ndarray:
    """Extract per-cell feature matrix from a terrain grid.

    Returns (H*W, n_features) float32 matrix with 7 one-hot terrain
    columns plus settlement_dist, settlement_density, forest_density,
    is_coastal, ocean_dist.
    """
    terrain_onehot = _compute_terrain_onehot(grid)
    scalar_feats = _compute_scalar_features(grid)

    features = np.concatenate([terrain_onehot, scalar_feats], axis=1)
    return features.astype(np.float32)


def _compute_terrain_onehot(grid: np.ndarray) -> np.ndarray:
    """One-hot encode terrain types, flattened to (H*W, 7)."""
    flat = grid.ravel().astype(np.int32)
    onehot = np.zeros((flat.shape[0], NUM_TERRAIN_TYPES), dtype=np.float32)
    valid = (flat >= 0) & (flat < NUM_TERRAIN_TYPES)
    onehot[valid, flat[valid]] = 1.0
    return onehot


def _compute_scalar_features(grid: np.ndarray) -> np.ndarray:
    """Compute 5 scalar spatial features, flattened to (H*W, 5)."""
    settle_dist = compute_settlement_distance(grid).ravel()
    settle_mask = _settlement_mask(grid).astype(np.float64)
    settle_dens = uniform_filter(settle_mask, ML_DENSITY_KERNEL).ravel()
    forest_mask = (grid == InternalTerrain.FOREST).astype(np.float64)
    forest_dens = uniform_filter(forest_mask, ML_DENSITY_KERNEL).ravel()
    coastal = compute_coastal_mask(grid).ravel().astype(np.float32)
    ocean_dist = compute_ocean_distance(grid).ravel()

    scalars = np.column_stack(
        [
            settle_dist,
            settle_dens,
            forest_dens,
            coastal,
            ocean_dist,
        ]
    )
    return scalars.astype(np.float32)


def _settlement_mask(grid: np.ndarray) -> np.ndarray:
    """Boolean mask of settlement and port cells."""
    return np.isin(grid, [InternalTerrain.SETTLEMENT, InternalTerrain.PORT])


def build_training_data(
    data_dir: str | Path,
    exclude_round: str | None = None,
    exclude_round_numbers: set[int] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Build (X, y) arrays from saved round data, optionally excluding rounds."""
    import json

    data_path = Path(data_dir)
    round_dirs = sorted(d for d in data_path.iterdir() if d.is_dir())

    all_x: list[np.ndarray] = []
    all_y: list[np.ndarray] = []

    for rdir in round_dirs:
        if exclude_round and rdir.name == exclude_round:
            continue
        if exclude_round_numbers:
            rj = rdir / "round.json"
            if rj.exists():
                rnum = json.loads(rj.read_text()).get("round_number", 0)
                if rnum in exclude_round_numbers:
                    continue
        _load_round_samples(rdir, all_x, all_y)

    x_combined = np.concatenate(all_x, axis=0)
    y_combined = np.concatenate(all_y, axis=0)
    logger.info("Training data: %d samples, %d features", *x_combined.shape)
    return x_combined, y_combined


def _load_round_samples(
    rdir: Path,
    all_x: list[np.ndarray],
    all_y: list[np.ndarray],
) -> None:
    """Load all seed samples from one round directory."""
    seed_dirs = sorted(d for d in rdir.iterdir() if d.is_dir())
    for sdir in seed_dirs:
        grid_path = sdir / "initial_grid.npy"
        gt_path = sdir / "ground_truth.npy"
        if not grid_path.exists() or not gt_path.exists():
            continue
        grid = np.load(grid_path)
        gt = np.load(gt_path)
        x = extract_cell_features(grid)
        y = gt.reshape(-1, NUM_PREDICTION_CLASSES)
        all_x.append(x)
        all_y.append(y)


def train_model(
    x: np.ndarray,
    y: np.ndarray,
    seed: int = 42,
) -> MultiOutputRegressor:
    """Fit a multi-output XGBoost regressor on (N, F) features and (N, 6) targets."""
    base = XGBRegressor(
        n_estimators=ML_N_ESTIMATORS,
        max_depth=ML_MAX_DEPTH,
        learning_rate=ML_LEARNING_RATE,
        random_state=seed,
        verbosity=0,
    )
    model = MultiOutputRegressor(base)
    model.fit(x, y)
    logger.info("Model trained on %d samples", x.shape[0])
    return model


def predict_grid(
    grid: np.ndarray,
    model: MultiOutputRegressor,
) -> np.ndarray:
    """Predict H x W x 6 probabilities with floor and static overrides."""
    height, width = grid.shape
    x = extract_cell_features(grid)
    raw = model.predict(x)
    probs = _floor_and_normalize(raw)
    probs_grid = probs.reshape(height, width, NUM_PREDICTION_CLASSES)
    return _apply_static_overrides(grid, probs_grid)


def _floor_and_normalize(probs: np.ndarray) -> np.ndarray:
    """Floor predictions at PROBABILITY_FLOOR and renormalize.

    After normalization, re-floor and renormalize to guarantee
    no value falls below the floor.
    """
    result = np.maximum(probs, PROBABILITY_FLOOR)
    result = result / result.sum(axis=1, keepdims=True)
    # Iterative floor+renorm until stable (converges in 2-3 passes)
    for _ in range(5):
        below = result < PROBABILITY_FLOOR
        if not below.any():
            break
        result = np.maximum(result, PROBABILITY_FLOOR)
        result = result / result.sum(axis=1, keepdims=True)
    return result


def _apply_static_overrides(
    grid: np.ndarray,
    probs: np.ndarray,
) -> np.ndarray:
    """Override predictions for ocean and mountain cells."""
    result = probs.copy()
    rest = max(PROBABILITY_FLOOR, (1.0 - STATIC_TERRAIN_CONFIDENCE) / (NUM_PREDICTION_CLASSES - 1))
    conf = 1.0 - rest * (NUM_PREDICTION_CLASSES - 1)

    ocean_mask = grid == InternalTerrain.OCEAN
    result[ocean_mask] = rest
    result[ocean_mask, 0] = conf  # class 0 = EMPTY (ocean/plains)

    mountain_mask = grid == InternalTerrain.MOUNTAIN
    result[mountain_mask] = rest
    result[mountain_mask, 5] = conf  # class 5 = MOUNTAIN

    return result


def loo_evaluate(
    data_dir: str | Path,
    seed: int = 42,
) -> dict[str, float]:
    """Leave-one-round-out cross-validation. Returns per-round and avg scores."""
    data_path = Path(data_dir)
    round_dirs = sorted(d for d in data_path.iterdir() if d.is_dir())

    scores: list[float] = []
    results: dict[str, float] = {}

    for rdir in round_dirs:
        round_score = _evaluate_round(data_path, rdir, seed)
        if round_score is not None:
            results[rdir.name] = round_score
            scores.append(round_score)

    if scores:
        results["avg"] = float(np.mean(scores))
        logger.info("LOO avg score: %.2f", results["avg"])

    return results


def _evaluate_round(
    data_path: Path,
    rdir: Path,
    seed: int,
) -> float | None:
    """Evaluate one held-out round, returns score or None."""
    x_train, y_train = build_training_data(data_path, exclude_round=rdir.name)
    if x_train.shape[0] == 0:
        return None

    model = train_model(x_train, y_train, seed=seed)
    seed_scores = _score_round_seeds(rdir, model)

    if not seed_scores:
        return None

    avg = float(np.mean(seed_scores))
    logger.info("Round %s: score=%.2f (%d seeds)", rdir.name, avg, len(seed_scores))
    return avg


def _score_round_seeds(
    rdir: Path,
    model: MultiOutputRegressor,
) -> list[float]:
    """Score all seeds in a round directory."""
    from src.scoring import score_prediction

    seed_dirs = sorted(d for d in rdir.iterdir() if d.is_dir())
    scores: list[float] = []

    for sdir in seed_dirs:
        grid_path = sdir / "initial_grid.npy"
        gt_path = sdir / "ground_truth.npy"
        if not grid_path.exists() or not gt_path.exists():
            continue
        grid = np.load(grid_path)
        gt = np.load(gt_path)
        pred = predict_grid(grid, model)
        result = score_prediction(gt, pred)
        scores.append(result["score"])

    return scores
