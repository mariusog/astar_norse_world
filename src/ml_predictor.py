"""XGBoost per-cell probability predictor."""

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
    compute_settlement_density,
    compute_settlement_distance,
    compute_terrain_neighborhood,
    compute_terrain_onehot,
)
from src.terrain import InternalTerrain

logger = logging.getLogger(__name__)

# ── ML hyperparameters (in constants spirit, co-located for now) ──────────
ML_N_ESTIMATORS = 100
ML_MAX_DEPTH = 5
ML_LEARNING_RATE = 0.1
ML_DENSITY_KERNEL = 7  # 7x7 uniform filter for density features
NUM_TERRAIN_TYPES = 7  # InternalTerrain 0-6
NUM_FEATURES = NUM_TERRAIN_TYPES + 10 + NUM_TERRAIN_TYPES * 2 + 3  # 34 total


def extract_cell_features(grid: np.ndarray) -> np.ndarray:
    """Extract (H*W, 34) float32 feature matrix from terrain grid."""
    terrain_onehot = compute_terrain_onehot(grid, NUM_TERRAIN_TYPES)
    sd = compute_settlement_distance(grid).ravel().astype(np.float32)
    s_dens = compute_settlement_density(grid, window=ML_DENSITY_KERNEL).ravel()
    f_mask = (grid == InternalTerrain.FOREST).astype(np.float64)
    f_dens = uniform_filter(f_mask, ML_DENSITY_KERNEL).ravel().astype(np.float32)
    coastal = compute_coastal_mask(grid).ravel().astype(np.float32)
    od = compute_ocean_distance(grid).ravel().astype(np.float32)
    inv_sd, inv_od = 1.0 / (1.0 + sd), 1.0 / (1.0 + od)
    scalars = _build_scalars(sd, s_dens, f_dens, coastal, od, inv_sd, inv_od)
    nbr_r2 = compute_terrain_neighborhood(grid, radius=2).reshape(-1, NUM_TERRAIN_TYPES)
    nbr_r4 = compute_terrain_neighborhood(grid, radius=4).reshape(-1, NUM_TERRAIN_TYPES)
    is_forest = terrain_onehot[:, InternalTerrain.FOREST].astype(np.float32)
    interactions = np.column_stack(
        [is_forest * sd, is_forest * inv_sd, f_dens * inv_sd],
    ).astype(np.float32)
    return np.concatenate(
        [terrain_onehot, scalars, nbr_r2, nbr_r4, interactions],
        axis=1,
    ).astype(np.float32)


def _build_scalars(
    sd: np.ndarray,
    s_dens: np.ndarray,
    f_dens: np.ndarray,
    coastal: np.ndarray,
    od: np.ndarray,
    inv_sd: np.ndarray,
    inv_od: np.ndarray,
) -> np.ndarray:
    """Stack 10 scalar features into (N, 10) float32 array."""
    return np.column_stack(
        [sd, s_dens, f_dens, coastal, od, inv_sd, sd**2, np.log1p(sd), inv_od, np.log1p(od)]
    ).astype(np.float32)


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


def compute_entropy_weights(y: np.ndarray) -> np.ndarray:
    """Compute Shannon entropy of each target distribution as sample weight.

    Args:
        y: (N, C) array of probability vectors.

    Returns:
        (N,) float32 array of weights, normalized so mean weight = 1.0.
    """
    safe = np.maximum(y, 1e-10)
    entropy = -np.sum(safe * np.log(safe), axis=1)
    # Normalize so mean weight = 1.0
    entropy = entropy / (entropy.mean() + 1e-10)
    return entropy.astype(np.float32)


def train_model(
    x: np.ndarray,
    y: np.ndarray,
    seed: int = 42,
    sample_weight: np.ndarray | None = None,
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
    model.fit(x, y, sample_weight=sample_weight)
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


class MLGridPredictor:
    """Adapter wrapping trained XGBoost model as GridPredictor."""

    def __init__(self, model: MultiOutputRegressor) -> None:
        self._model = model

    def predict_grid(self, grid: np.ndarray) -> np.ndarray:
        """Predict H x W x 6 probability tensor."""
        return predict_grid(grid, self._model)
